#include "Convolution_complex_GPU.h"

// This is the kernel that compute the complex convolution on the GPU

__global__ void Convolution_complex_GPUkernel(Complex* out, short* in, short* bufferI, short* bufferO, int sizeTot, int nch)
{
	int size = sizeTot / nch;
	unsigned int tid = blockIdx.x * (blockDim.x) + threadIdx.x;
	//unsigned int i = idxch + nch * tid;
	int a = nch * (MASK_LENGTH - 1);

	if (tid < sizeTot)
	{

		if (tid >= sizeTot - nch * (MASK_LENGTH - 1)) {
			bufferO[nch * (MASK_LENGTH - 1) - (sizeTot - tid)] = in[tid];
		}

		if (tid < nch * (MASK_LENGTH - 1)) {

			float tempx = 0.0f;
			float tempy = 0.0f;
			for (int j = 0; j < MASK_LENGTH; j++) {
				// Index in the current segment being considered
				int dataIdx = tid - nch * MASK_LENGTH + 1 + nch * j;
				if (dataIdx + nch < 1) {
					// This means we are in the overlap region and should use bufferI
					tempx += bufferI[nch * MASK_LENGTH + dataIdx - 1] * MASK[j].x;
					tempy += bufferI[nch * MASK_LENGTH + dataIdx - 1] * MASK[j].y;
				}
				else {

					tempx += in[dataIdx + nch - 1] * MASK[j].x;
					tempy += in[dataIdx + nch - 1] * MASK[j].y;
				}
			}

			out[(tid % nch) * size + tid / nch].x = tempx;
			out[(tid % nch) * size + tid / nch].y = tempy;
		}

		else {

			out[(tid % nch) * size + tid / nch].x = in[tid - a] * MASK[0].x + in[tid - a + nch * 1] * MASK[1].x + in[tid - a + nch * 2] * MASK[2].x + in[tid - a + nch * 3] * MASK[3].x + in[tid - a + nch * 4] * MASK[4].x
				+ in[tid - a + nch * 5] * MASK[5].x + in[tid - a + nch * 6] * MASK[6].x + in[tid - a + nch * 7] * MASK[7].x + in[tid - a + nch * 8] * MASK[8].x + in[tid - a + nch * 9] * MASK[9].x
				+ in[tid - a + nch * 10] * MASK[10].x + in[tid - a + nch * 11] * MASK[11].x + in[tid - a + nch * 12] * MASK[12].x + in[tid - a + nch * 13] * MASK[13].x + in[tid - a + nch * 14] * MASK[14].x
				+ in[tid - a + nch * 15] * MASK[15].x + in[tid - a + nch * 16] * MASK[16].x + in[tid - a + nch * 17] * MASK[17].x + in[tid - a + nch * 18] * MASK[18].x + in[tid - a + nch * 19] * MASK[19].x
				+ in[tid - a + nch * 20] * MASK[20].x + in[tid - a + nch * 21] * MASK[21].x + in[tid - a + nch * 22] * MASK[22].x + in[tid - a + nch * 23] * MASK[23].x + in[tid - a + nch * 24] * MASK[24].x
				+ in[tid - a + nch * 25] * MASK[25].x + in[tid - a + nch * 26] * MASK[26].x + in[tid - a + nch * 27] * MASK[27].x + in[tid - a + nch * 28] * MASK[28].x + in[tid - a + nch * 29] * MASK[29].x
				+ in[tid - a + nch * 30] * MASK[30].x + in[tid - a + nch * 31] * MASK[31].x;

			out[(tid % nch) * size + tid / nch].y = in[tid - a] * MASK[0].y + in[tid - a + nch * 1] * MASK[1].y + in[tid - a + nch * 2] * MASK[2].y + in[tid - a + nch * 3] * MASK[3].y + in[tid - a + nch * 4] * MASK[4].y
				+ in[tid - a + nch * 5] * MASK[5].y + in[tid - a + nch * 6] * MASK[6].y + in[tid - a + nch * 7] * MASK[7].y + in[tid - a + nch * 8] * MASK[8].y + in[tid - a + nch * 9] * MASK[9].y
				+ in[tid - a + nch * 10] * MASK[10].y + in[tid - a + nch * 11] * MASK[11].y + in[tid - a + nch * 12] * MASK[12].y + in[tid - a + nch * 13] * MASK[13].y + in[tid - a + nch * 14] * MASK[14].y
				+ in[tid - a + nch * 15] * MASK[15].y + in[tid - a + nch * 16] * MASK[16].y + in[tid - a + nch * 17] * MASK[17].y + in[tid - a + nch * 18] * MASK[18].y + in[tid - a + nch * 19] * MASK[19].y
				+ in[tid - a + nch * 20] * MASK[20].y + in[tid - a + nch * 21] * MASK[21].y + in[tid - a + nch * 22] * MASK[22].y + in[tid - a + nch * 23] * MASK[23].y + in[tid - a + nch * 24] * MASK[24].y
				+ in[tid - a + nch * 25] * MASK[25].y + in[tid - a + nch * 26] * MASK[26].y + in[tid - a + nch * 27] * MASK[27].y + in[tid - a + nch * 28] * MASK[28].y + in[tid - a + nch * 29] * MASK[29].y
				+ in[tid - a + nch * 30] * MASK[30].y + in[tid - a + nch * 31] * MASK[31].y;

		}
	}
	
}

// This is the C wrapper fucntion that calls the CUDA kernel

extern "C" cudaError_t Convolution_complex_GPU(Complex* out, void* in, short* bufferI, short* bufferO, int sizeTot, int blocks, int threads, int LoopCount, Complex * h_mask, int nch, cudaStream_t streamId)
{
	cudaError_t cudaStatus = cudaSuccess;
	if (LoopCount == 0) {
		cudaMemcpyToSymbol(MASK, h_mask, MASK_LENGTH * sizeof(Complex), 0, cudaMemcpyHostToDevice);
	}

	// Launch a kernel on the GPU with one thread for each element.

	if (0 == blocks)
	{
		blocks = (int)(sizeTot + threads - 1) / threads;
	}

	Convolution_complex_GPUkernel <<< blocks, threads, 0, streamId >>> (out, (short*)in, bufferI, bufferO, sizeTot, nch);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Convolution_complex_GPUkernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	return cudaStatus;
}