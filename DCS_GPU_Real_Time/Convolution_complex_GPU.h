// Convolution_complex_GPU.h
// 
// Prototype for function that computes a complex convolution on the GPU
// 
// The associated .cu file contains the c-wrapper as well as the actual Cuda kernel. 
// 
// Mathieu Walsh 
// Jerome Genest
// October 2023
//

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>

typedef float2 Complex;

#define MASK_LENGTH 32

__constant__ Complex MASK[MASK_LENGTH];

#ifdef __cplusplus
extern "C" cudaError_t Convolution_complex_GPU(Complex * out, void* in, short* bufferI, short* bufferO, int sizeTot, int blocks, int threads, int LoopCount, Complex * h_mask, int nch, cudaStream_t streamId);
#else
extern cudaError_t Convolution_complex_GPU(Complex* out, void* in, int complex, int sizeTot, int blocks, int threads, int LoopCount, short* bufferI, short* bufferO, Complex* h_mask, int nch, cudaStream_t streamId);
#endif