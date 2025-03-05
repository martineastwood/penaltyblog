package main

import "C"
import (
	"math"
	"unsafe"
)

// Compute the Negative Binomial log-likelihood
//
//export ComputeNegativeBinomialLoss
func ComputeNegativeBinomialLoss(
	params *C.double, nTeams C.int,
	homeIdx *C.int, awayIdx *C.int,
	goalsHome *C.int, goalsAway *C.int,
	weights *C.double, nMatches C.int) C.double {

	// Convert pointers to slices
	paramSlice := unsafe.Slice(params, int(nTeams)*2+2) // Attack, Defense, HomeAdvantage, Dispersion
	homeIdxSlice := unsafe.Slice(homeIdx, int(nMatches))
	awayIdxSlice := unsafe.Slice(awayIdx, int(nMatches))
	goalsHomeSlice := unsafe.Slice(goalsHome, int(nMatches))
	goalsAwaySlice := unsafe.Slice(goalsAway, int(nMatches))
	weightsSlice := unsafe.Slice(weights, int(nMatches))

	attackParams := paramSlice[:nTeams]
	defenseParams := paramSlice[nTeams : 2*nTeams]
	homeAdvantage := paramSlice[2*nTeams]
	dispersion := math.Max(float64(paramSlice[2*nTeams+1]), 1e-5)

	logLikelihood := 0.0

	for i := 0; i < int(nMatches); i++ {
		lambdaHome := math.Exp(float64(homeAdvantage) + float64(attackParams[homeIdxSlice[i]]) + float64(defenseParams[awayIdxSlice[i]]))
		lambdaAway := math.Exp(float64(attackParams[awayIdxSlice[i]]) + float64(defenseParams[homeIdxSlice[i]]))

		pHome := dispersion / (dispersion + lambdaHome)
		pAway := dispersion / (dispersion + lambdaAway)

		// Compute Negative Binomial log-likelihood
		logP_home := negBinomLogPMF(int(goalsHomeSlice[i]), dispersion, pHome)
		logP_away := negBinomLogPMF(int(goalsAwaySlice[i]), dispersion, pAway)

		// Handle NaN or Inf values
		if math.IsNaN(logP_home) || math.IsInf(logP_home, 0) || math.IsNaN(logP_away) || math.IsInf(logP_away, 0) {
			return C.double(math.Inf(1))
		}

		logLikelihood += (logP_home + logP_away) * float64(weightsSlice[i])
	}

	return C.double(-logLikelihood)
}

// Compute Negative Binomial probabilities for scores
//
//export ComputeNegativeBinomialProbabilities
func ComputeNegativeBinomialProbabilities(
	home_attack, away_attack, home_defense, away_defense, home_advantage C.double,
	dispersion C.double, max_goals C.int,
	score_matrix *C.double, lambda_home *C.double, lambda_away *C.double) {

	// Compute expected goals
	lh := math.Exp(float64(home_advantage) + float64(home_attack) + float64(away_defense))
	la := math.Exp(float64(away_attack) + float64(home_defense))

	*lambda_home = C.double(lh)
	*lambda_away = C.double(la)

	// Compute probability vectors
	homeGoalsVector := make([]float64, int(max_goals))
	awayGoalsVector := make([]float64, int(max_goals))

	for g := 0; g < int(max_goals); g++ {
		homeGoalsVector[g] = negBinomPMF(g, float64(dispersion), lh)
		awayGoalsVector[g] = negBinomPMF(g, float64(dispersion), la)
	}

	// Convert score_matrix pointer to a slice
	scoreMatrixSlice := unsafe.Slice(score_matrix, int(max_goals)*int(max_goals))

	// Compute scoreline probability matrix
	for i := 0; i < int(max_goals); i++ {
		for j := 0; j < int(max_goals); j++ {
			index := i*int(max_goals) + j
			scoreMatrixSlice[index] = C.double(homeGoalsVector[i] * awayGoalsVector[j])
		}
	}

}
