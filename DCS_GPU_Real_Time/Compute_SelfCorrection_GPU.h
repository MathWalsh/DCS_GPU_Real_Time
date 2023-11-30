// Compute_SelfCorrection_GPU.h
// 
// Prototype for function that computes the SelfCorrection dfr and f0 grids on the GPU
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
#include "Linspace_GPU.h"
#include "Linear_interpolation_GPU.h"
#include <stdio.h>
#include "cufft.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <cusolverDn.h> // Add cusolver.lib cublas.lib cublasLt.lib cusparse.lib to linker input
#include <cmath>  // Include the cmath library for the floor function
#ifdef __INTELLISENSE__
void __syncthreads();
#endif


#ifdef __cplusplus
extern "C" cudaError_t Compute_SelfCorrection_GPU(cufftComplex * IGMsOut, cufftComplex * IGMsIn, float* Spline_grid_f0, double* Spline_grid_dfr, double* selfCorr_xaxis_uniform_grid, int* idx_nonuniform_to_uniform,
	double* spline_coefficients_f0, double* spline_coefficients_dfr, double* max_idx_sub, double* phase_sub, double* start_dfr_grid, double* end_dfr_grid, int ptsPerIGM, int NIGMs, int sizeIn,
	int n_interval, int threads, int blocks, double* d_h, double* d_D, double* d_work, int* devInfo, int lwork, cusolverDnHandle_t cuSolver_handle, cudaStream_t streamId, cudaError_t cudaStatus);
#else
extern cudaError_t Compute_SelfCorrection_GPU(cufftComplex* IGMsOut, cufftComplex* IGMsIn, float* Spline_grid_f0, double* Spline_grid_dfr, double* selfCorr_xaxis_uniform_grid, int* idx_nonuniform_to_uniform,
	double* spline_coefficients_f0, double* spline_coefficients_dfr, double* max_idx_sub, double* phase_sub, double* start_dfr_grid, double* end_dfr_grid, int ptsPerIGM, int NIGMs, int sizeIn,
	int n_interval, int threads, int blocks, double* d_h, double* d_D, double* d_work, int* devInfo, int lwork, cusolverDnHandle_t cuSolver_handle, cudaStream_t streamId, cudaError_t cudaStatus);
#endif

