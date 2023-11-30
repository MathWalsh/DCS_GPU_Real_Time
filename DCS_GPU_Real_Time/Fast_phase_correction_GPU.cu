#include "Fast_phase_correction_GPU.h"

__global__ void Fast_phase_correction_GPUkernel(Complex* IGMsout, Complex* IGMsin, Complex* ref, int conj, int sizeTot)
{

	int tid = blockIdx.x * (blockDim.x) + threadIdx.x;
	if (tid < sizeTot) {

		float angle = atan2f(ref[tid].y, ref[tid].x);
		if (conj == 1)
		{
			angle = -angle;
		}

		float p1 = IGMsin[tid].x * cosf(angle);
		float p2 = (IGMsin[tid].x + IGMsin[tid].y) * (cosf(angle) + sinf(angle));
		float p3 = IGMsin[tid].y * sinf(angle);

		IGMsout[tid].x = p1 - p3;
		IGMsout[tid].y = p2 - p3 - p1;

	}

}

// This is the C wrapper fucntion that calls the CUDA kernel

extern "C" cudaError_t Fast_phase_correction_GPU(Complex * IGMsout, Complex * IGMsin, Complex * ref, int conj, int sizeTot, int blocks, int threads, cudaStream_t streamId)
{
	cudaError_t cudaStatus = cudaSuccess;

	// Launch a kernel on the GPU with one thread for each element.

	if (0 == blocks)
	{
		blocks = (int)(sizeTot + threads - 1) / threads;
	}

	Fast_phase_correction_GPUkernel << < blocks, threads, 0, streamId >> > (IGMsout, IGMsin, ref, conj, sizeTot);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Fast_phase_correction_GPUkernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	return cudaStatus;
}