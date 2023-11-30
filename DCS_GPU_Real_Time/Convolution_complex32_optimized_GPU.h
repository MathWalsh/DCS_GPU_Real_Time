// Convolution_complex32_optimized_GPU.h
// 
// Prototype for function that computes a complex convolution on the GPU. Each channel has a different kernel, so that we can use the constant memory more effectively. We use 
// a 32 tap fir filter computed with fir1 in matlab. It is a complex band-pass filter that keeps only the positive frequencies.
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
#include <iostream>
#ifdef __INTELLISENSE__
void __syncthreads();
void __syncwarp();
#endif

#define MASK_LENGTH_TOT 32
//
__constant__ cufftComplex MASK[5 * MASK_LENGTH_TOT]; // hardcoded, we can make it bigger than it needs to be

#ifdef __cplusplus
extern "C" cudaError_t Convolution_complex32_optimized_GPU(cufftComplex * out, short* in, float* bufferI, float* bufferO, int sizeCh, int threads, int blocks,
	int LoopCount, cufftComplex * h_mask, int nch, int nfilt, int* idxchfilt, cudaStream_t streamId, cudaError_t cudaStatus);
#else
extern cudaError_t  Convolution_complex32_optimized_GPU(cufftComplex* out, short* in, float* bufferI, float* bufferO, int sizeCh, int threads, int blocks,
	int LoopCount, cufftComplex* h_mask, int nch, int nfilt, int* idxchfilt, cudaStream_t streamId, cudaError_t cudaStatus);
#endif#pragma once
