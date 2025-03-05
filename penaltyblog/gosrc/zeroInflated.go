package main

import "C"
import (
	"math"
	"unsafe"
)

// Compute the negative log-likelihood for Zero-Inflated Poisson (ZIP) model
//
//export ComputeZeroInflatedPoissonLoss
func ComputeZeroInflatedPoissonLoss(
	params *C.double, nTeams C.int,
	homeIdx *C.int, awayIdx *C.int,
	goalsHome *C.int, goalsAway *C.int,
	weights *C.double, nMatches C.int) C.double {

	// Convert pointers to slices
	paramSlice := unsafe.Slice(params, int(nTeams)*2+2) // Attack, Defense, HomeAdvantage, ZeroInflation
	homeIdxSlice := unsafe.Slice(homeIdx, int(nMatches))
	awayIdxSlice := unsafe.Slice(awayIdx, int(nMatches))
	goalsHomeSlice := unsafe.Slice(goalsHome, int(nMatches))
	goalsAwaySlice := unsafe.Slice(goalsAway, int(nMatches))
	weightsSlice := unsafe.Slice(weights, int(nMatches))

	attackParams := paramSlice[:nTeams]
	defenseParams := paramSlice[nTeams : 2*nTeams]
	homeAdvantage := float64(paramSlice[2*nTeams])
	zeroInflation := float64(paramSlice[2*nTeams+1]) // Last parameter

	logLikelihood := 0.0

	for i := 0; i < int(nMatches); i++ {
		lambdaHome := math.Exp(float64(homeAdvantage) + float64(attackParams[homeIdxSlice[i]]) + float64(defenseParams[awayIdxSlice[i]]))
		lambdaAway := math.Exp(float64(attackParams[awayIdxSlice[i]]) + float64(defenseParams[homeIdxSlice[i]]))

		// Compute ZIP log-likelihood for home team
		if goalsHomeSlice[i] == 0 {
			probZero := zeroInflation + (1-zeroInflation)*math.Exp(-lambdaHome)
			logLikelihood += math.Log(probZero) * float64(weightsSlice[i])
		} else {
			logLikelihood += (math.Log(1-zeroInflation) + poissonLogPMF(int(goalsHomeSlice[i]), lambdaHome)) * float64(weightsSlice[i])
		}

		// Compute ZIP log-likelihood for away team
		if goalsAwaySlice[i] == 0 {
			probZero := zeroInflation + (1-zeroInflation)*math.Exp(-lambdaAway)
			logLikelihood += math.Log(probZero) * float64(weightsSlice[i])
		} else {
			logLikelihood += (math.Log(1-zeroInflation) + poissonLogPMF(int(goalsAwaySlice[i]), lambdaAway)) * float64(weightsSlice[i])
		}
	}

	return C.double(-logLikelihood) // Return negative log-likelihood for optimization
}

// Compute Zero-Inflated Poisson probabilities for scores
//
//export ComputeZeroInflatedPoissonProbabilities
func ComputeZeroInflatedPoissonProbabilities(
	home_attack, away_attack, home_defense, away_defense, home_advantage C.double,
	zero_inflation C.double, max_goals C.int,
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
		homeGoalsVector[g] = zipPoissonPMF(g, lh, float64(zero_inflation))
		awayGoalsVector[g] = zipPoissonPMF(g, la, float64(zero_inflation))
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
