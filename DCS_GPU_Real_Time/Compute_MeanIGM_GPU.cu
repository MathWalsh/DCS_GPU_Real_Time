#include "Compute_MeanIGM_GPU.h"


template <int blockSize>
__device__ void warpReduceMean(volatile cufftComplex* sdataMean, unsigned int tid) {
	if (blockSize >= 64) {
		sdataMean[tid].x += sdataMean[tid + 32].x;
		sdataMean[tid].y += sdataMean[tid + 32].y;
	}
	if (blockSize >= 32) {
		sdataMean[tid].x += sdataMean[tid + 16].x;
		sdataMean[tid].y += sdataMean[tid + 16].y;
	}
	if (blockSize >= 16) {
		sdataMean[tid].x += sdataMean[tid + 8].x;
		sdataMean[tid].y += sdataMean[tid + 8].y;
	}
	if (blockSize >= 8) {
		sdataMean[tid].x += sdataMean[tid + 4].x;
		sdataMean[tid].y += sdataMean[tid + 4].y;
	}
	if (blockSize >= 4) {
		sdataMean[tid].x += sdataMean[tid + 2].x;
		sdataMean[tid].y += sdataMean[tid + 2].y;
	}
	if (blockSize >= 2) {
		sdataMean[tid].x += sdataMean[tid + 1].x;
		sdataMean[tid].y += sdataMean[tid + 1].y;
	}
}

// Calculate the mean points between IGMs for cropping and update self-correction parameters. ptsPerIGMSegment will be used to keep track of dfr in real time
template<unsigned int blockSize>
__global__ void MeanIGM_GPUkernel(cufftComplex* IGMOut, cufftComplex* IGMsIn, int NIGMs, int ptsPerIGM, int sizeIn)
{

	extern __shared__ cufftComplex sdataMean[];
	int tid = threadIdx.x;

	sdataMean[threadIdx.x] = { 0.0f, 0.0f };

	if (blockIdx.x + tid * ptsPerIGM < sizeIn) {
		sdataMean[tid] = IGMsIn[blockIdx.x + tid * ptsPerIGM];
	}
	__syncthreads();
	// Reduction based on Mark Harriss webinar, modified to find maximum in each block (each block is 1 IGM xcorr)
	// We look at the magnitude to determine the maximum
	if (blockSize >= 512) {
		if (tid < 256) {
			sdataMean[tid].x += sdataMean[tid + 256].x;
			sdataMean[tid].y += sdataMean[tid + 256].y;
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdataMean[tid].x += sdataMean[tid + 128].x;
			sdataMean[tid].y += sdataMean[tid + 128].y;
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdataMean[tid].x += sdataMean[tid + 64].x;
			sdataMean[tid].y += sdataMean[tid + 64].y;
		}
		__syncthreads();
	}

	if (tid < 32) warpReduceMean<blockSize>(sdataMean, tid);

	if (threadIdx.x == 0) {


		IGMOut[blockIdx.x].x = sdataMean[0].x / NIGMs;
		IGMOut[blockIdx.x].y = sdataMean[0].y / NIGMs;
	}

}

extern "C" cudaError_t Compute_MeanIGM_GPU(cufftComplex * IGMOut, cufftComplex * IGMsIn, int NIGMs, int sizeIn, int ptsPerIGM, cudaStream_t streamId, cudaError_t cudaStatus)
{

	int caseSelector = (NIGMs - 1) / 32; // Adjusting the range for each case. For now NIGMs per batch < 352
	switch (caseSelector) {
	case 0:
		MeanIGM_GPUkernel<32> << <ptsPerIGM, 32, 2 * 32 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	case 1:
		MeanIGM_GPUkernel<64> << <ptsPerIGM, 64, 2 * 64 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	case 2:
		MeanIGM_GPUkernel<96> << <ptsPerIGM, 96, 2 * 96 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	case 3:
		MeanIGM_GPUkernel<128> << <ptsPerIGM, 128, 2 * 128 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	case 4:
		MeanIGM_GPUkernel<160> << <ptsPerIGM, 160, 2 * 160 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	case 5:
		MeanIGM_GPUkernel<192> << <ptsPerIGM, 192, 2 * 192 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	case 6:
		MeanIGM_GPUkernel<224> << <ptsPerIGM, 224, 2 * 224 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	case 7:
		MeanIGM_GPUkernel<256> << <ptsPerIGM, 256, 2 * 256 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	case 8:
		MeanIGM_GPUkernel<288> << <ptsPerIGM, 288, 2 * 288 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	case 9:
		MeanIGM_GPUkernel<320> << <ptsPerIGM, 320, 2 * 320 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	case 10:
		MeanIGM_GPUkernel<352> << <ptsPerIGM, 352, 2 * 352 * sizeof(cufftComplex), streamId >> > (IGMOut, IGMsIn, NIGMs, ptsPerIGM, sizeIn);
		break;
	default:
		// Handle the error condition
		fprintf(stderr, "Error: Unsupported value of NIGMs: %d\n", NIGMs);
		break;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	return cudaStatus;

}