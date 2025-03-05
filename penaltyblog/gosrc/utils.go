package main

import "math"
import "C"

// Poisson log PMF function
func poissonLogPMF(k int, lambda float64) float64 {
	if k < 0 {
		return -lambda
	}
	// lgamma, _ := math.Lgamma(float64(k + 1))
	lgamma, _ := math.Lgamma(float64(k + 1))
	return float64(k)*math.Log(lambda) - lambda - lgamma
}

// Poisson probability mass function (PMF) using logarithm trick
func poissonPMF(k int, lambda float64) float64 {
	if k < 0 {
		return 0 // PMF should be zero for negative k
	}
	return math.Exp(poissonLogPMF(k, lambda))
}

// Helper function to convert boolean conditions to float64
func boolToFloat(condition bool) float64 {
	if condition {
		return 1.0
	}
	return 0.0
}

// Compute the ZIP Poisson PMF
func zipPoissonPMF(k int, lambda, zeroInflation float64) float64 {
	if k == 0 {
		return zeroInflation + (1-zeroInflation)*math.Exp(-lambda)
	}
	return (1 - zeroInflation) * math.Exp(poissonLogPMF(k, lambda))
}

// Negative Binomial log-PMF
func negBinomLogPMF(k int, r, p float64) float64 {
	if k < 0 {
		return math.Inf(-1) // Log PMF should be negative infinity for invalid k
	}
	// use the math.Lgamma function for better precision
	lgam1, _ := math.Lgamma(float64(k) + r)
	lgam2, _ := math.Lgamma(float64(k) + 1)
	lgam3, _ := math.Lgamma(r)
	return lgam1 - lgam2 - lgam3 + r*math.Log(p) + float64(k)*math.Log(1-p)
}

// Negative Binomial PMF function
func negBinomPMF(k int, r, lambda float64) float64 {
	if k < 0 {
		return 0.0 // PMF should be zero for invalid k
	}
	p := r / (r + lambda)
	lgam1, _ := math.Lgamma(float64(k) + r)
	lgam2, _ := math.Lgamma(float64(k) + 1)
	lgam3, _ := math.Lgamma(r)
	return math.Exp(lgam1 - lgam2 - lgam3 +
		r*math.Log(p) + float64(k)*math.Log(1-p))
}

// Get max from slice
func max(arr []int32) int32 {
	maxVal := arr[0]
	for _, v := range arr {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

// Convert C array to Go slice
func toInt32Slice(cArray []C.int) []int32 {
	result := make([]int32, len(cArray))
	for i, v := range cArray {
		result[i] = int32(v)
	}
	return result
}

// Precompute Poisson PMFs for a given lambda
func precomputePoissonPMF(lambda float64, maxGoals int) []float64 {
	pmf := make([]float64, maxGoals)
	for k := 0; k < maxGoals; k++ {
		lgam, _ := math.Lgamma(float64(k + 1))
		pmf[k] = math.Exp(float64(k)*math.Log(lambda) - lambda - lgam)
	}
	return pmf
}
