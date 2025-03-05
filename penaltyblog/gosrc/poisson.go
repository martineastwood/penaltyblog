package main

import (
	"C"
	"math"
	"unsafe"
)

// Sequential log-likelihood calculation (No Goroutines)
//
//export PoissonLogLikelihood
func PoissonLogLikelihood(params *C.double, n_teams C.int, home_idx *C.int, away_idx *C.int, goals_home *C.int, goals_away *C.int, weights *C.double, n_matches C.int) C.double {
	// Convert C pointers to Go slices
	paramSlice := (*[1 << 30]C.double)(unsafe.Pointer(params))[: 2*n_teams+1 : 2*n_teams+1]
	homeIdxSlice := (*[1 << 30]C.int)(unsafe.Pointer(home_idx))[:n_matches:n_matches]
	awayIdxSlice := (*[1 << 30]C.int)(unsafe.Pointer(away_idx))[:n_matches:n_matches]
	goalsHomeSlice := (*[1 << 30]C.int)(unsafe.Pointer(goals_home))[:n_matches:n_matches]
	goalsAwaySlice := (*[1 << 30]C.int)(unsafe.Pointer(goals_away))[:n_matches:n_matches]
	weightsSlice := (*[1 << 30]C.double)(unsafe.Pointer(weights))[:n_matches:n_matches]

	// Extract parameters
	attackParams := paramSlice[:n_teams:n_teams]
	defenseParams := paramSlice[n_teams : 2*n_teams]
	homeAdvantage := paramSlice[2*n_teams]

	// Sequential computation
	logLikelihood := 0.0
	for i := 0; i < int(n_matches); i++ {
		lambdaHome := math.Exp(float64(homeAdvantage) + float64(attackParams[homeIdxSlice[i]]) + float64(defenseParams[awayIdxSlice[i]]))
		lambdaAway := math.Exp(float64(attackParams[awayIdxSlice[i]]) + float64(defenseParams[homeIdxSlice[i]]))

		ll := (poissonLogPMF(int(goalsHomeSlice[i]), lambdaHome) + poissonLogPMF(int(goalsAwaySlice[i]), lambdaAway)) * float64(weightsSlice[i])
		logLikelihood += ll
	}

	return C.double(-logLikelihood)
}

// Compute Poisson probabilities for scores
//
//export ComputePoissonProbabilities
func ComputePoissonProbabilities(home_attack, away_attack, home_defense, away_defense, home_advantage C.double,
	max_goals C.int, score_matrix *C.double, lambda_home *C.double, lambda_away *C.double) {

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
}
