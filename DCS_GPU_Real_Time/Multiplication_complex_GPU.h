// Multiplication_complex_GPU.h
// 
// Prototype for function that computes a complex multiplication on the GPU
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

#ifdef __cplusplus
extern "C" cudaError_t Multiplication_complex_GPU(Complex * out, Complex * in1, Complex * in2, int conj1, int conj2, int sizeTot, int blocks, int threads, cudaStream_t streamId);
#else
extern cudaError_t Multiplication_complex_GPU(Complex* out, Complex* in1, Complex* in2, int conj1, int conj2, int sizeTot, int blocks, int threads, cudaStream_t streamId);
#endif