#include "Linspace_GPU.h"


// Does the linspace for the interpolation steps
__global__ void Linspace_GPUkernel(double* output, double* start, double* end, int sizeIn, int idxLinSpace) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

    double step = (end[idxLinSpace] - start[idxLinSpace]) / static_cast<double>(sizeIn - 1);
 
	if (tid < sizeIn) {

		output[tid] = start[idxLinSpace] + static_cast<double>(tid) * step;
	}
}

// This is the C wrapper function that calls the CUDA kernel
extern "C" cudaError_t Linspace_GPU(double* output, double* start, double* end, int sizeIn, int threads, int blocks, int idxLinSpace, cudaStream_t streamId, cudaError_t cudaStatus) {

    Linspace_GPUkernel << <blocks, threads, 0, streamId >> > (output, start, end, sizeIn, idxLinSpace);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();

    return cudaStatus;
}
