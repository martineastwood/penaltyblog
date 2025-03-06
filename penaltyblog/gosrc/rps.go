package main

import "C"
import (
	"fmt"
	"unsafe"
)

// ComputeRPSArray computes individual RPS scores for each fixture and writes them into a pre‐allocated output array.
// probs is a flattened double array of shape (nSets * nOutcomes) in row‐major order.
// outcomes is a pointer to an int array (length nSets), where each element is the observed outcome index.

//export ComputeRPSArray
func ComputeRPSArray(probs *C.double, outcomes *C.int, nSets C.int, nOutcomes C.int, out *C.double) {
	total := int(nSets) * int(nOutcomes)
	probsSlice := unsafe.Slice(probs, total)
	outcomesSlice := unsafe.Slice(outcomes, int(nSets))
	outSlice := unsafe.Slice(out, int(nSets))

	// Process each fixture (row)
	for i := 0; i < int(nSets); i++ {
		rowStart := i * int(nOutcomes)
		row := probsSlice[rowStart : rowStart+int(nOutcomes)]
		outcome := int(outcomesSlice[i])
		if outcome < 0 || outcome >= int(nOutcomes) {
			outSlice[i] = 1e6 // Penalty for invalid outcome
			continue
		}

		// Compute cumulative probabilities for this row
		cumProbs := make([]float64, int(nOutcomes))
		cumProbs[0] = float64(row[0])
		for j := 1; j < int(nOutcomes); j++ {
			cumProbs[j] = cumProbs[j-1] + float64(row[j])
		}

		// Build indicator vector: 1 only at index == outcome, 0 otherwise.
		indicator := make([]float64, int(nOutcomes))
		for j := 0; j < int(nOutcomes); j++ {
			if j == outcome {
				indicator[j] = 1.0
			} else {
				indicator[j] = 0.0
			}
		}

		// Compute cumulative sum of the indicator to get cumOutcomes.
		cumOutcomes := make([]float64, int(nOutcomes))
		cumOutcomes[0] = indicator[0]
		for j := 1; j < int(nOutcomes); j++ {
			cumOutcomes[j] = cumOutcomes[j-1] + indicator[j]
		}

		// Compute sum of squared differences.
		diffSum := 0.0
		for j := 0; j < int(nOutcomes); j++ {
			d := cumProbs[j] - cumOutcomes[j]
			diffSum += d * d
		}

		// Compute RPS for this fixture.
		rpsVal := diffSum / float64(int(nOutcomes)-1)
		outSlice[i] = C.double(rpsVal)
	}
}

// ComputeAverageRPS computes the average RPS over all fixtures by averaging the individual RPS scores.
// It uses the same logic as ComputeRPSArray internally.
//
//export ComputeAverageRPS
func ComputeAverageRPS(probs *C.double, outcomes *C.int, nSets C.int, nOutcomes C.int) C.double {
	total := int(nSets) * int(nOutcomes)
	probsSlice := unsafe.Slice(probs, total)
	outcomesSlice := unsafe.Slice(outcomes, int(nSets))

	sum := 0.0
	// For each fixture (row)
	for i := 0; i < int(nSets); i++ {
		rowStart := i * int(nOutcomes)
		row := probsSlice[rowStart : rowStart+int(nOutcomes)]
		outcome := int(outcomesSlice[i])
		if outcome < 0 || outcome >= int(nOutcomes) {
			sum += 1e6
			continue
		}

		// Compute cumulative probabilities for this fixture.
		cumProbs := make([]float64, int(nOutcomes))
		cumProbs[0] = float64(row[0])
		for j := 1; j < int(nOutcomes); j++ {
			cumProbs[j] = cumProbs[j-1] + float64(row[j])
		}

		// Build indicator vector: 1 only at the observed outcome, 0 elsewhere.
		indicator := make([]float64, int(nOutcomes))
		for j := 0; j < int(nOutcomes); j++ {
			if j == outcome {
				indicator[j] = 1.0
			} else {
				indicator[j] = 0.0
			}
		}

		// Compute cumulative outcomes.
		cumOutcomes := make([]float64, int(nOutcomes))
		cumOutcomes[0] = indicator[0]
		for j := 1; j < int(nOutcomes); j++ {
			cumOutcomes[j] = cumOutcomes[j-1] + indicator[j]
		}

		// Sum squared differences.
		diffSum := 0.0
		for j := 0; j < int(nOutcomes); j++ {
			d := cumProbs[j] - cumOutcomes[j]
			diffSum += d * d
		}

		rpsVal := diffSum / float64(int(nOutcomes)-1)
		sum += rpsVal
	}

	avg := sum / float64(nSets)
	fmt.Println("Average RPS in Go:", avg, C.double(avg))
	return C.double(avg)
}
