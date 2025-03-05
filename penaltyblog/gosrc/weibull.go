package main

import "C"
import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"
)

func gamma(x float64) float64 {
	if x <= 0 {
		return math.NaN() // Gamma function is not defined for x <= 0
	}
	lgamm, sign := math.Lgamma(x)
	return float64(sign) * math.Exp(lgamm) // Correct way to compute gamma
}

func precomputeAlphaTable(c float64, maxGoals int) [][]float64 {
	if c <= 0 {
		return nil
	}

	A := make([][]float64, maxGoals+1)
	for i := range A {
		A[i] = make([]float64, maxGoals+1)
	}

	// Compute raw alpha values
	alphaRaw := make([][]float64, maxGoals+1)
	for i := range alphaRaw {
		alphaRaw[i] = make([]float64, maxGoals+1)
	}

	for j := 0; j <= maxGoals; j++ {
		alphaRaw[0][j] = gamma(c*float64(j)+1.0) / gamma(float64(j)+1.0)
	}

	for x := 0; x < maxGoals; x++ {
		for j := x + 1; j <= maxGoals; j++ {
			tmpSum := 0.0
			for m := x; m < j; m++ {
				tmpSum += (alphaRaw[x][m] * math.Gamma(c*float64(j)-c*float64(m)+1.0)) / math.Gamma(float64(j-m)+1.0)
			}
			alphaRaw[x+1][j] = tmpSum
		}
	}

	for x := 0; x <= maxGoals; x++ {
		for j := 0; j <= maxGoals; j++ {
			sign := math.Pow(-1, float64(x+j))
			denom := math.Gamma(c*float64(j) + 1.0)
			A[x][j] = sign * (alphaRaw[x][j] / denom)
		}
	}

	return A
}

func weibullCountPMF(lam float64, A [][]float64, maxGoals int) []float64 {
	pmf := make([]float64, maxGoals+1)
	if lam <= 0 {
		pmf[0] = 1.0
		return pmf
	}

	lamPowers := make([]float64, maxGoals+1)
	lamPowers[0] = 1.0
	for j := 1; j <= maxGoals; j++ {
		lamPowers[j] = lamPowers[j-1] * lam
	}

	for x := 0; x <= maxGoals; x++ {
		val := 0.0
		for j := x; j <= maxGoals; j++ {
			val += lamPowers[j] * A[x][j]
		}
		if val > 0 {
			pmf[x] = val
		}
	}

	// Normalize PMF
	sum := 0.0
	for _, v := range pmf {
		sum += v
	}
	if sum < 1e-14 {
		// Degenerate case: set all mass to zero except first value
		pmf = make([]float64, maxGoals+1)
		pmf[0] = 1.0
	} else {
		for i := range pmf {
			pmf[i] /= sum
		}
	}

	return pmf
}

func cdfFromPMF(pmf []float64) []float64 {
	cdf := make([]float64, len(pmf))
	cdf[0] = pmf[0]

	for i := 1; i < len(pmf); i++ {
		cdf[i] = cdf[i-1] + pmf[i]
	}

	return cdf
}

func computePxy(x, y int, cdfX, cdfY []float64, maxGoals int, kappa float64) float64 {
	// Define helper functions for boundary checking
	FX := func(k int) float64 {
		if k < 0 {
			return 0.0
		} else if k > maxGoals {
			return 1.0
		}
		return cdfX[k]
	}

	FY := func(k int) float64 {
		if k < 0 {
			return 0.0
		} else if k > maxGoals {
			return 1.0
		}
		return cdfY[k]
	}

	// Compute Frank Copula-based probability for (x, y)
	pXY := frankCopulaCDF(FX(x), FY(y), kappa) -
		frankCopulaCDF(FX(x-1), FY(y), kappa) -
		frankCopulaCDF(FX(x), FY(y-1), kappa) +
		frankCopulaCDF(FX(x-1), FY(y-1), kappa)

	// Ensure non-negative probability
	if pXY < 0.0 {
		pXY = 0.0
	}

	return pXY
}

func frankCopulaCDF(u, v, kappa float64) float64 {
	if math.Abs(kappa) < 1e-8 {
		return u * v // Independence case
	}

	num := (math.Exp(-kappa*u) - 1.0) * (math.Exp(-kappa*v) - 1.0)
	denom := math.Exp(-kappa) - 1.0
	inside := 1.0 + num/denom

	if inside <= 1e-14 {
		return math.Max(0.0, u*v)
	}

	return -(1.0 / kappa) * math.Log(inside)
}

// Compute the Weibull-Copula Model Log-Likelihood
//
//export ComputeWeibullCopulaLoss
func ComputeWeibullCopulaLoss(
	params *C.double, nTeams C.int,
	homeIdx *C.int, awayIdx *C.int,
	goalsHome *C.int, goalsAway *C.int,
	weights *C.double, nMatches C.int, maxGoals C.int) C.double {

	// Convert pointers to slices
	paramSlice := unsafe.Slice(params, int(nTeams)*2+3)
	homeIdxSlice := unsafe.Slice(homeIdx, int(nMatches))
	awayIdxSlice := unsafe.Slice(awayIdx, int(nMatches))
	goalsHomeSlice := unsafe.Slice(goalsHome, int(nMatches))
	goalsAwaySlice := unsafe.Slice(goalsAway, int(nMatches))
	weightsSlice := unsafe.Slice(weights, int(nMatches))

	// Extract model parameters
	attackParams := paramSlice[:nTeams]
	defenseParams := paramSlice[nTeams : 2*nTeams]
	homeAdvantage := paramSlice[2*nTeams]
	shape := paramSlice[2*nTeams+1]
	kappa := paramSlice[2*nTeams+2]

	// Check for invalid shape value
	if shape <= 0 {
		return C.double(1e15) // Penalize invalid shape
	}

	// Precompute alpha table for given shape
	alphaTable := precomputeAlphaTable(float64(shape), int(maxGoals))
	if alphaTable == nil {
		return C.double(1e15) // Shape was invalid
	}

	// Parallel processing setup
	numThreads := runtime.NumCPU()
	batchSize := int(math.Max(10, float64(nMatches)/float64(numThreads))) // Ensure at least 10 matches per thread
	var wg sync.WaitGroup
	results := make([]float64, numThreads)

	// Process matches in parallel batches
	for threadID := 0; threadID < numThreads; threadID++ {
		wg.Add(1)
		go func(threadID int) {
			defer wg.Done()
			start := threadID * batchSize
			end := start + batchSize
			if threadID == numThreads-1 {
				end = int(nMatches)
			}

			localSum := 0.0

			for i := start; i < end; i++ {
				// Compute `lambda_home` and `lambda_away`
				lambdaHome := math.Exp(float64(homeAdvantage) + float64(attackParams[homeIdxSlice[i]]) + float64(defenseParams[awayIdxSlice[i]]))
				lambdaAway := math.Exp(float64(attackParams[awayIdxSlice[i]]) + float64(defenseParams[homeIdxSlice[i]]))

				// Compute Weibull PMF for `lambdaHome` and `lambdaAway`
				pmfH := weibullCountPMF(lambdaHome, alphaTable, int(maxGoals))
				pmfA := weibullCountPMF(lambdaAway, alphaTable, int(maxGoals))

				// Compute cumulative distribution functions (CDFs)
				cdfH := cdfFromPMF(pmfH)
				cdfA := cdfFromPMF(pmfA)

				// Compute the probability of the actual score using the Frank Copula function
				x_i, y_i := int(goalsHomeSlice[i]), int(goalsAwaySlice[i])
				pXY := computePxy(x_i, y_i, cdfH, cdfA, int(maxGoals), float64(kappa))

				// Handle log(0) edge cases
				if pXY < 1e-10 {
					pXY = 1e-10
				}
				localSum += float64(weightsSlice[i]) * math.Log(pXY)
			}

			// Store results
			results[threadID] = localSum
		}(threadID)
	}

	// Wait for all goroutines to complete
	wg.Wait()

	// Aggregate results from all threads
	logLikelihood := 0.0
	for _, v := range results {
		logLikelihood += v
	}

	return C.double(-logLikelihood)
}

// Compute the Weibull-Copula Probability Matrix
//
//export ComputeWeibullCopulaProbabilities
func ComputeWeibullCopulaProbabilities(
	home_attack, away_attack, home_defense, away_defense, home_advantage C.double,
	shape C.double, kappa C.double, max_goals C.int,
	score_matrix *C.double, lambdaH *C.double, lambdaA *C.double) {

	// Compute expected goals
	lamH := math.Exp(float64(home_advantage) + float64(home_attack) + float64(away_defense))
	lamA := math.Exp(float64(away_attack) + float64(home_defense))

	// Store lambda values
	*lambdaH = C.double(lamH)
	*lambdaA = C.double(lamA)

	// Precompute Weibull alpha table for shape
	alphaTable := precomputeAlphaTable(float64(shape), int(max_goals))
	if alphaTable == nil {
		fmt.Println("Invalid shape value for Weibull distribution")
		return
	}

	// Compute Weibull PMFs
	pmfH := weibullCountPMF(lamH, alphaTable, int(max_goals))
	pmfA := weibullCountPMF(lamA, alphaTable, int(max_goals))

	// Compute Weibull CDFs
	cdfH := cdfFromPMF(pmfH)
	cdfA := cdfFromPMF(pmfA)

	// Convert score_matrix pointer to a slice
	scoreMatrixSlice := unsafe.Slice(score_matrix, int(max_goals)*int(max_goals))

	// Compute the score probability matrix using Frank Copula
	for i := 0; i < int(max_goals); i++ {
		for j := 0; j < int(max_goals); j++ {
			p_ij := computePxy(i, j, cdfH, cdfA, int(max_goals), float64(kappa))
			scoreMatrixSlice[i*int(max_goals)+j] = C.double(p_ij)
		}
	}

}
