// Linear_interpolation_GPU.h
// 
// Prototype for function that computes the linear interpolation on the GPU. We first find in which index interval the new_grid values fall in. So for each value
// in new_grid, we look +- n_interval points around this value in the old_grid, and we save the proper index. Then we use these index to perform the linear interpolation
// 
// The associated .cu file contains the c-wrapper as well as the actual Cuda kernel. 
// 
// Mathieu Walsh 
// Jerome Genest
// November 2023
//

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cufft.h"


#ifdef __cplusplus
extern "C" cudaError_t Linear_interpolation_GPU(cufftComplex * new_IGMs, double* new_dfr_grid, cufftComplex * old_IGMs, double* old_dfr_grid, int* new_dfr_idx,
	int old_n, int new_n, int n_interval, int threads, int blocks, cudaStream_t streamId, cudaError_t cudaStatus);
#else
extern cudaError_t Linear_interpolation_GPU(cufftComplex* new_IGMs, double* new_dfr_grid, cufftComplex* old_IGMs, double* old_dfr_grid, int* new_dfr_idx,
	int old_n, int new_n, int n_interval, int threads, int blocks, cudaStream_t streamId, cudaError_t cudaStatus);
#endif

