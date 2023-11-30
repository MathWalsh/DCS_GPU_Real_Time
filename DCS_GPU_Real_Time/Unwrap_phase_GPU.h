#pragma once
// Unwrap_phase_GPU.h
// 
// Prototype for function that computes the unwrapping of a phase signal on the GPU. We base the unwrapping on the cupy kernel but we unrolled all the loops
// for an architecture with warp_size = 32 and for a block size of 128
// The associated .cu file contains the c-wrapper as well as the actual Cuda kernel. 
// 
// Mathieu Walsh 
// Jerome Genest
// November 2023
//

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <cufft.h> // Must add cufft.lib to linker
#include <stdio.h>

//for __syncthreads() __syncwarp() to remove the intellisens warning
#ifdef __INTELLISENSE__
void __syncthreads();
void __syncwarp();
#endif


#ifdef __cplusplus
extern "C" cudaError_t UnwrapPhase_GPU(double* unwrapped_phase, float* refdfr_angle, int* two_pi_cumsum, int* blocks_edges_cumsum, int* increment_blocks_edges, int sizeIn, bool UnwrapDfr, bool estimateSlope,
	double* start_slope, double* end_slope, const int warp_size, int blocks, cudaStream_t streamId, cudaError_t cudaStatus);
#else
extern cudaError_t	UnwrapPhase_GPU(double* unwrapped_phase, float* refdfr_angle, int* two_pi_cumsum, int* blocks_edges_cumsum, int* increment_blocks_edges, int sizeIn, bool UnwrapDfr, bool estimateSlope,
	double* start_slope, double* end_slope, const int warp_size, int blocks, cudaStream_t streamId, cudaError_t cudaStatus);
#endif