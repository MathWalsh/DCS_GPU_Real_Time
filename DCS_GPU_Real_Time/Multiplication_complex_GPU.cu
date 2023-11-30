#include "Multiplication_complex_GPU.h"

// This is the kernel that compute the complex multiplication on the GPU

__global__ void Multiplication_complex_GPUkernel(Complex* out, Complex* in1, Complex* in2, int conj1, int conj2, int sizeTot)
{

	int tid = blockIdx.x * (blockDim.x) + threadIdx.x;
	if (tid < sizeTot) {

		float p1 = in1[tid].x * in2[tid].x;
		if (conj1 == 1)
		{

			float p2 = (in1[tid].x - in1[tid].y) * (in2[tid].x + in2[tid].y);
			float p3 = -1 * in1[tid].y * in2[tid].y;
			out[tid].x = p1 - p3;
			out[tid].y = p2 - p3 - p1;

		}
		else if (conj2 == 1)
		{
			float p2 = (in1[tid].x + in1[tid].y) * (in2[tid].x - in2[tid].y);
			float p3 = -1 * in1[tid].y * in2[tid].y;
			out[tid].x = p1 - p3;
			out[tid].y = p2 - p3 - p1;
		}
		else {

			float p2 = (in1[tid].x + in1[tid].y) * (in2[tid].x + in2[tid].y);
			float p3 = in1[tid].y * in2[tid].y;
			out[tid].x = p1 - p3;
			out[tid].y = p2 - p3 - p1;
		}



	}

}

extern "C" cudaError_t Multiplication_complex_GPU(Complex * out, Complex * in1, Complex * in2, int conj1, int conj2, int sizeTot, int blocks, int threads, cudaStream_t streamId)
{
	cudaError_t cudaStatus = cudaSuccess;

	// Launch a kernel on the GPU with one thread for each element.

	if (0 == blocks)
	{
		blocks = (int)(sizeTot + threads - 1) / threads;
	}


	Multiplication_complex_GPUkernel <<< blocks, threads, 0, streamId >> > (out, in1, in2, conj1, conj2, sizeTot);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Multiplication_complex_GPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	return cudaStatus;
}