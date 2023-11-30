#include "Calculate_dfr_reference_GPU.h"



// Complex multiplication with 3 multiplications instead of 4
template <typename T>
__device__ T ComplexMult(T in1, T in2, bool conj1, bool conj2)
{
	T result;

	float p1 = in1.x * in2.x;
	if (conj1)
	{
		float p2 = (in1.x - in1.y) * (in2.x + in2.y);
		float p3 = -1 * in1.y * in2.y;
		result.x = p1 - p3;
		result.y = p2 - p3 - p1;
	}
	else if (conj2)
	{
		float p2 = (in1.x + in1.y) * (in2.x - in2.y);
		float p3 = -1 * in1.y * in2.y;
		result.x = p1 - p3;
		result.y = p2 - p3 - p1;
	}
	else
	{

		float p2 = (in1.x + in1.y) * (in2.x + in2.y);
		float p3 = in1.y * in2.y;
		result.x = p1 - p3;
		result.y = p2 - p3 - p1;
	}

	return result;
}

// Calulates the wrapped angle of dfr
__global__ void Calculate_dfr_reference_GPUkernel(float* refdfr_angle, cufftComplex* inref1, cufftComplex* inFopt3, cufftComplex* inFopt4, bool conjFopt3, bool conjFopt4, bool conjdfr1, bool conjdfr2, int sizeIn)
{

	int tid = blockIdx.x * (blockDim.x) + threadIdx.x;
	if (tid < sizeIn) {

		cufftComplex ref1 = inref1[tid]; // We put ref1 in the input as refdfr. The output will be at the same pointer as ref1
		cufftComplex ref2 = ComplexMult<cufftComplex>(inFopt3[tid], inFopt4[tid], conjFopt3, conjFopt4);
		cufftComplex refdfr = ComplexMult<cufftComplex>(ref1, ref2, conjdfr1, conjdfr2);
		refdfr_angle[tid] =  atan2f(refdfr.y, refdfr.x);

	}

}


extern "C" cudaError_t Calculate_dfr_reference_GPU(float * refdfr_angle, cufftComplex* inref1, cufftComplex * inFopt3, cufftComplex * inFopt4,
	bool conjFopt3, bool conjFopt4, bool conjdfr1, bool conjdfr2, int sizeIn, int threads, int blocks,  cudaStream_t streamId, cudaError_t cudaStatus)
{


	Calculate_dfr_reference_GPUkernel << < blocks, threads, 0, streamId >> > (refdfr_angle, inref1, inFopt3, inFopt4, conjFopt3, conjFopt4, conjdfr1, conjdfr2, sizeIn);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	return cudaStatus;
}