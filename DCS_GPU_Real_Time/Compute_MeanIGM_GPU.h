// Compute_MeanIGM_GPU.h
// 
// Prototype for function that calculate the average of the complex IGM of the segment
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
extern "C" cudaError_t Compute_MeanIGM_GPU(cufftComplex * IGMOut, cufftComplex * IGMsIn, int NIGMs, int sizeIn, int ptsPerIGM, cudaStream_t streamId, cudaError_t cudaStatus);
#else
extern cudaError_t Compute_MeanIGM_GPU(cufftComplex* IGMOut, cufftComplex* IGMsIn, int NIGMs, int sizeIn, int ptsPerIGM, cudaStream_t streamId, cudaError_t cudaStatus);
#endif