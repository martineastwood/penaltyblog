package main

import "C"
import (
	"math"
	"unsafe"
)

// Compute the Bivariate Poisson Log-Likelihood
//
//export ComputeBivariatePoissonLoss
func ComputeBivariatePoissonLoss(
	params *C.double, nTeams C.int,
	homeIdx *C.int, awayIdx *C.int,
	goalsHome *C.int, goalsAway *C.int,
	weights *C.double, nMatches C.int) C.double {

	// Convert pointers to slices
	paramSlice := unsafe.Slice(params, int(nTeams)*2+2) // Attack, Defense, HomeAdv, Correlation_log
	homeIdxSlice := unsafe.Slice(homeIdx, int(nMatches))
	awayIdxSlice := unsafe.Slice(awayIdx, int(nMatches))
	goalsHomeSlice := unsafe.Slice(goalsHome, int(nMatches))
	goalsAwaySlice := unsafe.Slice(goalsAway, int(nMatches))
	weightsSlice := unsafe.Slice(weights, int(nMatches))

	attackParams := paramSlice[:nTeams]
	defenseParams := paramSlice[nTeams : 2*nTeams]
	homeAdvantage := paramSlice[2*nTeams]
	correlationLog := float64(paramSlice[2*nTeams+1]) // Last parameter

	lambda3 := math.Exp(correlationLog) // Convert log(correlation) to lambda3
	maxGoals := int(math.Max(float64(max(toInt32Slice(goalsHomeSlice))), float64(max(toInt32Slice(goalsAwaySlice))))) + 1

	// Precompute Poisson PMF for lambda3
	lambda3PMF := precomputePoissonPMF(lambda3, maxGoals)

	pmfCache := make(map[float64][]float64)

	logLikelihood := 0.0

	for i := 0; i < int(nMatches); i++ {
		// Compute `lambda1` and `lambda2` inside Go
		lambda1 := math.Exp(float64(homeAdvantage) + float64(attackParams[homeIdxSlice[i]]) + float64(defenseParams[awayIdxSlice[i]]))
		lambda2 := math.Exp(float64(attackParams[awayIdxSlice[i]]) + float64(defenseParams[homeIdxSlice[i]]))

		pmf1, exists := pmfCache[lambda1]
		if !exists {
			pmf1 = precomputePoissonPMF(lambda1, maxGoals)
			pmfCache[lambda1] = pmf1
		}

		pmf2, exists := pmfCache[lambda2]
		if !exists {
			pmf2 = precomputePoissonPMF(lambda2, maxGoals)
			pmfCache[lambda2] = pmf2
		}

		likeIJ := 0.0
		kMax := int(math.Min(float64(goalsHomeSlice[i]), float64(goalsAwaySlice[i])))

		for k := 0; k <= kMax; k++ {
			likeIJ += pmf1[int(goalsHomeSlice[i])-k] * pmf2[int(goalsAwaySlice[i])-k] * lambda3PMF[k]
		}

		// Prevent log(0)
		if likeIJ < 1e-10 {
			likeIJ = 1e-10
		}

		logLikelihood += float64(weightsSlice[i]) * math.Log(likeIJ)
	}

	return C.double(-logLikelihood) // Return negative log-likelihood
}

// Compute the Bivariate Poisson Probability Matrix
//
//export ComputeBivariatePoissonProbabilities
func ComputeBivariatePoissonProbabilities(
	home_attack, away_attack, home_defense, away_defense, home_advantage C.double,
	correlation_log C.double, max_goals C.int,
	score_matrix *C.double, lambda1 *C.double, lambda2 *C.double) {

	// Compute expected goals
	lam1 := math.Exp(float64(home_advantage) + float64(home_attack) + float64(away_defense))
	lam2 := math.Exp(float64(away_attack) + float64(home_defense))
	lam3 := math.Exp(float64(correlation_log)) // Convert log correlation to lambda3

	// Return lambda values
	*lambda1 = C.double(lam1)
	*lambda2 = C.double(lam2)

	// Compute Poisson PMFs for lambda1, lambda2, and lambda3
	pmf1 := precomputePoissonPMF(lam1, int(max_goals))
	pmf2 := precomputePoissonPMF(lam2, int(max_goals))
	pmf3 := precomputePoissonPMF(lam3, int(max_goals))

	// Convert score_matrix pointer to a slice
	scoreMatrixSlice := unsafe.Slice(score_matrix, int(max_goals)*int(max_goals))

	// Compute the bivariate Poisson probability matrix
	for x := 0; x < int(max_goals); x++ {
		for y := 0; y < int(max_goals); y++ {
			p_xy := 0.0
			for k := 0; k <= int(math.Min(float64(x), float64(y))); k++ {
				p_xy += pmf1[x-k] * pmf2[y-k] * pmf3[k]
			}
			scoreMatrixSlice[x*int(max_goals)+y] = C.double(p_xy)
		}
	}
}
