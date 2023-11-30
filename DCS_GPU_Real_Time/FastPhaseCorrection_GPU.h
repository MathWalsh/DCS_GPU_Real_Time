// FastPhaseCorrection_GPU.h
// 
// Prototype for function that computes the fast phase correction with an optical reference on the GPU. We remove the CW contribution by doing a multiplication in the time domain (convolution in frequency).
// We then apply IGMs*exp(1j*angle_ref) with the correct sign
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
extern "C" cudaError_t FastPhaseCorrection_GPU(cufftComplex * outIGMs, cufftComplex * outref1, cufftComplex * inIGMs, cufftComplex * inFopt1, cufftComplex * inFopt2, int sizeIn,
    bool conjFopt1, bool conjFopt2, bool conjFPC, int threads, int blocks, cudaStream_t streamId, cudaError_t cudaStatus);
#else
extern cudaError_t FastPhaseCorrection_GPU(cufftComplex* outIGMs, cufftComplex* outref1, cufftComplex* inIGMs, cufftComplex* inFopt1, cufftComplex* inFopt2, int sizeIn,
    bool conjFopt1, bool conjFopt2, bool conjFPC, int threads, int blocks, cudaStream_t streamId, cudaError_t cudaStatus);
#endif

