package main

import (
	"C"
	"math"
	"unsafe"
)

// Dixon-Coles rho correction function
func rhoCorrection(goalsHome, goalsAway int, rho float64) float64 {
	gHome := float64(goalsHome)
	gAway := float64(goalsAway)

	// Apply the same formula as the Python version
	return 1.0 +
		(rho * boolToFloat(gHome == 0 && gAway == 0)) -
		(rho * boolToFloat(gHome == 0 && gAway == 1)) -
		(rho * boolToFloat(gHome == 1 && gAway == 0)) +
		(rho * boolToFloat(gHome == 1 && gAway == 1))
}

// Compute the negative log-likelihood for Dixon-Coles model
//
//export DixonColesLogLikelihood
func DixonColesLogLikelihood(
	params *C.double, nTeams C.int,
	homeIdx *C.int, awayIdx *C.int,
	goalsHome *C.int, goalsAway *C.int,
	weights *C.double, nMatches C.int) C.double {

	// Convert pointers to slices
	paramSlice := unsafe.Slice(params, int(nTeams)*2+2) // Attack, Defense, HomeAdvantage, Rho
	homeIdxSlice := unsafe.Slice(homeIdx, int(nMatches))
	awayIdxSlice := unsafe.Slice(awayIdx, int(nMatches))
	goalsHomeSlice := unsafe.Slice(goalsHome, int(nMatches))
	goalsAwaySlice := unsafe.Slice(goalsAway, int(nMatches))
	weightsSlice := unsafe.Slice(weights, int(nMatches))

	attackParams := paramSlice[:nTeams]
	defenseParams := paramSlice[nTeams : 2*nTeams]
	homeAdvantage := paramSlice[2*nTeams]
	rhoValue := float64(paramSlice[2*nTeams+1])

	logLikelihood := 0.0

	for i := 0; i < int(nMatches); i++ {
		lambdaHome := math.Exp(float64(homeAdvantage) + float64(attackParams[homeIdxSlice[i]]) + float64(defenseParams[awayIdxSlice[i]]))
		lambdaAway := math.Exp(float64(attackParams[awayIdxSlice[i]]) + float64(defenseParams[homeIdxSlice[i]]))

		// Compute Poisson Log-PMF
		logP_home := poissonLogPMF(int(goalsHomeSlice[i]), lambdaHome)
		logP_away := poissonLogPMF(int(goalsAwaySlice[i]), lambdaAway)

		// Apply Dixon-Coles rho correction
		phi := rhoCorrection(int(goalsHomeSlice[i]), int(goalsAwaySlice[i]), rhoValue)

		// Apply weighting
		logLikelihood += (logP_home + logP_away + math.Log(phi)) * float64(weightsSlice[i])
	}

	return C.double(-logLikelihood)
}

// Compute Poisson probabilities for scores
//
//export ComputeDixonColesProbabilities
func ComputeDixonColesProbabilities(home_attack, away_attack, home_defense, away_defense, home_advantage, rho C.double, max_goals C.int, score_matrix *C.double, lambda_home *C.double, lambda_away *C.double) {

	lh := math.Exp(float64(home_advantage) + float64(home_attack) + float64(away_defense))
	la := math.Exp(float64(away_attack) + float64(home_defense))

	*lambda_home = C.double(lh)
	*lambda_away = C.double(la)

	// Compute probability vectors
	homeGoalsVector := make([]float64, int(max_goals))
	awayGoalsVector := make([]float64, int(max_goals))

	for g := 0; g < int(max_goals); g++ {
		homeGoalsVector[g] = poissonPMF(g, lh)
		awayGoalsVector[g] = poissonPMF(g, la)
	}

	// Convert score_matrix pointer to a slice to safely access memory
	scoreMatrixSlice := unsafe.Slice(score_matrix, int(max_goals)*int(max_goals))

	// Compute scoreline probability matrix and store it in the pre-allocated buffer
	for i := 0; i < int(max_goals); i++ {
		for j := 0; j < int(max_goals); j++ {
			index := i*int(max_goals) + j
			scoreMatrixSlice[index] = C.double(homeGoalsVector[i] * awayGoalsVector[j])
		}
	}

	// Dixon-Coles adjustments for low-scoring games
	scoreMatrixSlice[0*int(max_goals)+0] *= 1 - *lambda_home**lambda_away*rho // (0,0) scaled by λ_home * λ_away
	scoreMatrixSlice[0*int(max_goals)+1] *= 1 + *lambda_home*rho              // (0,1) scaled by λ_home
	scoreMatrixSlice[1*int(max_goals)+0] *= 1 + *lambda_away*rho              // (1,0) scaled by λ_away
	scoreMatrixSlice[1*int(max_goals)+1] *= 1 - rho                           // (1,1) same as before
}
