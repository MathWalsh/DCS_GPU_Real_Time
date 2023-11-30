// Linspace_GPU.h
// 
// Prototype for function that computes the linspace on the GPU
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



#ifdef __cplusplus
extern "C" cudaError_t  Linspace_GPU(double* output, double* start, double* end, int sizeIn, int threads, int blocks, int idxLinSpace, cudaStream_t streamId, cudaError_t cudaStatus);
#else
extern cudaError_t  Linspace_GPU(double* output, double* start, double* end, int sizeIn, int threads, int blocks, int idxLinSpace, cudaStream_t streamId, cudaError_t cudaStatus);
#endif

