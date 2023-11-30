// Find_First_ZPD_GPU.h
// 
// Prototype for function that computes Find_First_ZPD_GPU on the GPU. We find the maximum between ptsPerIGM/2 and 3*ptsPerIGM/2. This allows us to find the first or second IGM.
// Then we do a xcorr to find the true position of the ZPD with a the template. This function runs on the first iteration of the loop and outputs the position of the first or second ZPD 
// in idxMidSegments[0]
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
#include <cmath>  // Include the cmath library for the round function

#ifdef __cplusplus
extern "C" cudaError_t  Find_First_ZPD_GPU(int* idxMaxBLocks, float* MaxValBlocks, cufftComplex * IGMs, cufftComplex * IGMTemplate, cufftComplex * xCorrBlocks, int* idxMidSegments, int*& idxStartFirstZPD,
	int NptsSegment, int templateSize, int ptsPerIGM, int sizeIn, int NdelaysPerIGM, int blocksPerDelay, int totalDelays, int totalBlocks, cudaStream_t streamId, cudaError_t cudaStatus);
#else
extern cudaError_t Find_First_ZPD_GPU(int* idxMaxBLocks, float* MaxValBlocks, cufftComplex* IGMs, cufftComplex* IGMTemplate, cufftComplex* xCorrBlocks, int* idxMidSegments, int*& idxStartFirstZPD,
	int NptsSegment, int templateSize, int ptsPerIGM, int sizeIn, int NdelaysPerIGM, int blocksPerDelay, int totalDelays, int totalBlocks, cudaStream_t streamId, cudaError_t cudaStatus);
#endif

