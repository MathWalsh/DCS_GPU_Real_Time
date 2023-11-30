// Fast_Phase_Correction_GPU.h
// 
// Prototype for function that computes a the fast phase correction (IGMs.*exp(1j*ref_phase)) on the GPU
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
#include "math.h"

typedef float2 Complex;

#ifdef __cplusplus
extern "C" cudaError_t Fast_phase_correction_GPU(Complex * IGMsout, Complex * IGMsin, Complex * ref, int conj, int sizeTot, int blocks, int threads, cudaStream_t streamId);
#else
extern cudaError_t Fast_phase_correction_GPU(Complex* IGMsout, Complex* IGMsin, Complex* ref, int conj, int sizeTot, int blocks, int threads, cudaStream_t streamId);
#endif