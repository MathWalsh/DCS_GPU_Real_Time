// Find_IGMs_ZPD_GPU.h
// 
// Prototype for function that computes Find_IGMs_ZPD on the GPU. We do a xcorr based on a given number of IGMs and NdelaysPerIGM.
// For the xcorr to work, we need to know the approximate position of the ZPDs. For the first buffer of the application, there will be
// a special function to find the first ZPD postion, but all the subsequent buffers, we will be able to track the approximate positions of the ZPOs
// When we have the position and phase of the maximum, we find it subpoint with a cubic fit for the position and linear fit for the phase
// We finish by cutting the last IGM properly 
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
extern "C" cudaError_t Find_IGMs_ZPD_GPU(double* max_idx_sub, double* phase_sub, cufftComplex * IGMs, cufftComplex * IGMTemplate, cufftComplex * xCorrBlocks,
	int* idxMidSegments, int idxStartFirstSegment, int NptsSegment, int NIGMs, double ptsPerIGM, int sizeIn, int sizeInCropped, int NdelaysPerIGM, int blocksPerDelay, int totalDelays, int totalBlocks,
	double* d_ptsPerIGMSegment, cudaStream_t streamId, cudaError_t cudaStatus);
#else
extern cudaError_t Find_IGMs_ZPD_GPU(double* max_idx_sub, double* phase_sub, cufftComplex* IGMs, cufftComplex* IGMTemplate, cufftComplex* xCorrBlocks,
	int* idxMidSegments, int idxStartFirstSegment, int NptsSegment, int NIGMs, double ptsPerIGM, int sizeIn, int sizeInCropped, int NdelaysPerIGM, int blocksPerDelay, int totalDelays, int totalBlocks,
	double* d_ptsPerIGMSegment, cudaStream_t streamId, cudaError_t cudaStatus);
#endif

