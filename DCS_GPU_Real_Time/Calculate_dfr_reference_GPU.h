// Calculate_dfr_reference_GPU.h
// 
// Prototype for function that computes a complex multiplication on the GPU. We calculate the angle of two references used for the dfr resampling with the correct sign.
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
#include <cufft.h> // Must add cufft.lib to linker
#include <stdio.h>
#include <Windows.h>


#ifdef __cplusplus
extern "C" cudaError_t Calculate_dfr_reference_GPU(float* refdfr_angle, cufftComplex * inref1, cufftComplex * inFopt3, cufftComplex * inFopt4,
	bool conjFopt3, bool conjFopt4, bool conjdfr1, bool conjdfr2, int sizeIn, int threads, int blocks, cudaStream_t streamId, cudaError_t cudaStatus);
#else
extern cudaError_t Calculate_dfr_reference_GPU(float* refdfr_angle, cufftComplex* inref1, cufftComplex* inFopt3, cufftComplex* inFopt4,
	bool conjFopt3, bool conjFopt4, bool conjdfr1, bool conjdfr2, int sizeIn, int threads, int blocks, cudaStream_t streamId, cudaError_t cudaStatus);
#endif