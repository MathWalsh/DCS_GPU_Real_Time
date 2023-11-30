#include "Convolution_complex32_optimized_GPU.h"


__global__ void Convolution_complex32_GPUkernel0(cufftComplex* out, short* in, float* bufferI, float* bufferO, int sizeCh, int nch, int* idxch) {

	const int MASK_Length = MASK_LENGTH_TOT - 1;
	extern __shared__ float sharedData[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int sizeIn = sizeCh * nch; // Total size of buffer

	int in_id = tid * nch + idxch[0];
	int sharedIndex = threadIdx.x + MASK_Length;

	// Load data into shared memory, handling the edges
	if (in_id - nch * MASK_Length >= 0 && in_id - nch * MASK_Length < sizeIn) { // This is for the transition between blocks
		sharedData[sharedIndex - MASK_Length] = static_cast<float>(in[in_id - nch * MASK_Length]);
	}
	else if (in_id < nch * MASK_Length) {

		sharedData[sharedIndex - MASK_Length] = bufferI[in_id]; // The first data of the segment reads the buffer from the previous segment

	}

	if (in_id < sizeIn) {

		sharedData[sharedIndex] = static_cast<float>(in[in_id]);

	}
	else {
		sharedData[sharedIndex] = 0.0f; // Zero padding for out of range tid
	}

	if (in_id >= sizeIn - nch * MASK_Length && in_id < sizeIn)
	{

		bufferO[nch * MASK_Length - (sizeIn - in_id)] = sharedData[sharedIndex]; // Verify accuracy

	}

	// Synchronize to ensure all data is loaded
	__syncthreads();

	// Manually unrolled convolution sum
	cufftComplex sum = { 0.0f , 0.0f };



	sum.x += sharedData[threadIdx.x + 0] * MASK[0].x + sharedData[threadIdx.x + 1] * MASK[1].x + sharedData[threadIdx.x + 2] * MASK[2].x + sharedData[threadIdx.x + 3] * MASK[3].x + sharedData[threadIdx.x + 4] * MASK[4].x
		+ sharedData[threadIdx.x + 5] * MASK[5].x + sharedData[threadIdx.x + 6] * MASK[6].x + sharedData[threadIdx.x + 7] * MASK[7].x + sharedData[threadIdx.x + 8] * MASK[8].x + sharedData[threadIdx.x + 9] * MASK[9].x
		+ sharedData[threadIdx.x + 10] * MASK[10].x + sharedData[threadIdx.x + 11] * MASK[11].x + sharedData[threadIdx.x + 12] * MASK[12].x + sharedData[threadIdx.x + 13] * MASK[13].x + sharedData[threadIdx.x + 14] * MASK[14].x
		+ sharedData[threadIdx.x + 15] * MASK[15].x + sharedData[threadIdx.x + 16] * MASK[16].x + sharedData[threadIdx.x + 17] * MASK[17].x + sharedData[threadIdx.x + 18] * MASK[18].x + sharedData[threadIdx.x + 19] * MASK[19].x
		+ sharedData[threadIdx.x + 20] * MASK[20].x + sharedData[threadIdx.x + 21] * MASK[21].x + sharedData[threadIdx.x + 22] * MASK[22].x + sharedData[threadIdx.x + 23] * MASK[23].x + sharedData[threadIdx.x + 24] * MASK[24].x
		+ sharedData[threadIdx.x + 25] * MASK[25].x + sharedData[threadIdx.x + 26] * MASK[26].x + sharedData[threadIdx.x + 27] * MASK[27].x + sharedData[threadIdx.x + 28] * MASK[28].x + sharedData[threadIdx.x + 29] * MASK[29].x
		+ sharedData[threadIdx.x + 30] * MASK[30].x + sharedData[threadIdx.x + 31] * MASK[31].x;

	sum.y += sharedData[threadIdx.x + 0] * MASK[0].y + sharedData[threadIdx.x + 1] * MASK[1].y + sharedData[threadIdx.x + 2] * MASK[2].y + sharedData[threadIdx.x + 3] * MASK[3].y + sharedData[threadIdx.x + 4] * MASK[4].y
		+ sharedData[threadIdx.x + 5] * MASK[5].y + sharedData[threadIdx.x + 6] * MASK[6].y + sharedData[threadIdx.x + 7] * MASK[7].y + sharedData[threadIdx.x + 8] * MASK[8].y + sharedData[threadIdx.x + 9] * MASK[9].y
		+ sharedData[threadIdx.x + 10] * MASK[10].y + sharedData[threadIdx.x + 11] * MASK[11].y + sharedData[threadIdx.x + 12] * MASK[12].y + sharedData[threadIdx.x + 13] * MASK[13].y + sharedData[threadIdx.x + 14] * MASK[14].y
		+ sharedData[threadIdx.x + 15] * MASK[15].y + sharedData[threadIdx.x + 16] * MASK[16].y + sharedData[threadIdx.x + 17] * MASK[17].y + sharedData[threadIdx.x + 18] * MASK[18].y + sharedData[threadIdx.x + 19] * MASK[19].y
		+ sharedData[threadIdx.x + 20] * MASK[20].y + sharedData[threadIdx.x + 21] * MASK[21].y + sharedData[threadIdx.x + 22] * MASK[22].y + sharedData[threadIdx.x + 23] * MASK[23].y + sharedData[threadIdx.x + 24] * MASK[24].y
		+ sharedData[threadIdx.x + 25] * MASK[25].y + sharedData[threadIdx.x + 26] * MASK[26].y + sharedData[threadIdx.x + 27] * MASK[27].y + sharedData[threadIdx.x + 28] * MASK[28].y + sharedData[threadIdx.x + 29] * MASK[29].y
		+ sharedData[threadIdx.x + 30] * MASK[30].y + sharedData[threadIdx.x + 31] * MASK[31].y;



	// Write the result
	if (tid < sizeIn) {
		out[tid] = sum;
	}
}

__global__ void Convolution_complex32_GPUkernel1(cufftComplex* out, short* in, float* bufferI, float* bufferO, int sizeCh, int nch, int* idxch) {

	const int MASK_Length = MASK_LENGTH_TOT - 1;
	extern __shared__ float sharedData[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int sizeIn = sizeCh * nch; // Total size of buffer
	const int Skip = 1 * (MASK_Length + 1);

	int out_id = tid + sizeIn / nch;
	int in_id = tid * nch + idxch[1];
	int sharedIndex = threadIdx.x + MASK_Length;
	// Load data into shared memory, handling the edges
	if (in_id - nch * MASK_Length >= 0 && in_id - nch * MASK_Length < sizeIn) { // This is for the transition between blocks
		sharedData[sharedIndex - MASK_Length] = static_cast<float>(in[in_id - nch * MASK_Length]);
	}
	else if (in_id < nch * MASK_Length) {

		sharedData[sharedIndex - MASK_Length] = bufferI[in_id]; // The first data of the segment reads the buffer from the previous segment
		//sharedData[sharedIndex - MASK_Length] = 0.0f; // The first data of the segment reads the buffer from the previous segment
	}

	if (in_id < sizeIn) {

		sharedData[sharedIndex] = static_cast<float>(in[in_id]);
	}
	else {
		sharedData[sharedIndex] = 0.0f; // Zero padding for out of range tid
	}

	if (in_id >= sizeIn - nch * MASK_Length && in_id < sizeIn)
	{
		bufferO[nch * MASK_Length - (sizeIn - in_id)] = sharedData[sharedIndex]; // Verify accuracy
	}

	// Synchronize to ensure all data is loaded
	__syncthreads();

	// Manually unrolled convolution sum
	cufftComplex sum = { 0.0f , 0.0f };



	sum.x += sharedData[threadIdx.x + 0] * MASK[0 + Skip].x + sharedData[threadIdx.x + 1] * MASK[1 + Skip].x + sharedData[threadIdx.x + 2] * MASK[2 + Skip].x + sharedData[threadIdx.x + 3] * MASK[3 + Skip].x + sharedData[threadIdx.x + 4] * MASK[4 + Skip].x
		+ sharedData[threadIdx.x + 5] * MASK[5 + Skip].x + sharedData[threadIdx.x + 6] * MASK[6 + Skip].x + sharedData[threadIdx.x + 7] * MASK[7 + Skip].x + sharedData[threadIdx.x + 8] * MASK[8 + Skip].x + sharedData[threadIdx.x + 9] * MASK[9 + Skip].x
		+ sharedData[threadIdx.x + 10] * MASK[10 + Skip].x + sharedData[threadIdx.x + 11] * MASK[11 + Skip].x + sharedData[threadIdx.x + 12] * MASK[12 + Skip].x + sharedData[threadIdx.x + 13] * MASK[13 + Skip].x + sharedData[threadIdx.x + 14] * MASK[14 + Skip].x
		+ sharedData[threadIdx.x + 15] * MASK[15 + Skip].x + sharedData[threadIdx.x + 16] * MASK[16 + Skip].x + sharedData[threadIdx.x + 17] * MASK[17 + Skip].x + sharedData[threadIdx.x + 18] * MASK[18 + Skip].x + sharedData[threadIdx.x + 19] * MASK[19 + Skip].x
		+ sharedData[threadIdx.x + 20] * MASK[20 + Skip].x + sharedData[threadIdx.x + 21] * MASK[21 + Skip].x + sharedData[threadIdx.x + 22] * MASK[22 + Skip].x + sharedData[threadIdx.x + 23] * MASK[23 + Skip].x + sharedData[threadIdx.x + 24] * MASK[24 + Skip].x
		+ sharedData[threadIdx.x + 25] * MASK[25 + Skip].x + sharedData[threadIdx.x + 26] * MASK[26 + Skip].x + sharedData[threadIdx.x + 27] * MASK[27 + Skip].x + sharedData[threadIdx.x + 28] * MASK[28 + Skip].x + sharedData[threadIdx.x + 29] * MASK[29 + Skip].x
		+ sharedData[threadIdx.x + 30] * MASK[30 + Skip].x + sharedData[threadIdx.x + 31] * MASK[31 + Skip].x;

	sum.y += sharedData[threadIdx.x + 0] * MASK[0 + Skip].y + sharedData[threadIdx.x + 1] * MASK[1 + Skip].y + sharedData[threadIdx.x + 2] * MASK[2 + Skip].y + sharedData[threadIdx.x + 3] * MASK[3 + Skip].y + sharedData[threadIdx.x + 4] * MASK[4 + Skip].y
		+ sharedData[threadIdx.x + 5] * MASK[5 + Skip].y + sharedData[threadIdx.x + 6] * MASK[6 + Skip].y + sharedData[threadIdx.x + 7] * MASK[7 + Skip].y + sharedData[threadIdx.x + 8] * MASK[8 + Skip].y + sharedData[threadIdx.x + 9] * MASK[9 + Skip].y
		+ sharedData[threadIdx.x + 10] * MASK[10 + Skip].y + sharedData[threadIdx.x + 11] * MASK[11 + Skip].y + sharedData[threadIdx.x + 12] * MASK[12 + Skip].y + sharedData[threadIdx.x + 13] * MASK[13 + Skip].y + sharedData[threadIdx.x + 14] * MASK[14 + Skip].y
		+ sharedData[threadIdx.x + 15] * MASK[15 + Skip].y + sharedData[threadIdx.x + 16] * MASK[16 + Skip].y + sharedData[threadIdx.x + 17] * MASK[17 + Skip].y + sharedData[threadIdx.x + 18] * MASK[18 + Skip].y + sharedData[threadIdx.x + 19] * MASK[19 + Skip].y
		+ sharedData[threadIdx.x + 20] * MASK[20 + Skip].y + sharedData[threadIdx.x + 21] * MASK[21 + Skip].y + sharedData[threadIdx.x + 22] * MASK[22 + Skip].y + sharedData[threadIdx.x + 23] * MASK[23 + Skip].y + sharedData[threadIdx.x + 24] * MASK[24 + Skip].y
		+ sharedData[threadIdx.x + 25] * MASK[25 + Skip].y + sharedData[threadIdx.x + 26] * MASK[26 + Skip].y + sharedData[threadIdx.x + 27] * MASK[27 + Skip].y + sharedData[threadIdx.x + 28] * MASK[28 + Skip].y + sharedData[threadIdx.x + 29] * MASK[29 + Skip].y
		+ sharedData[threadIdx.x + 30] * MASK[30 + Skip].y + sharedData[threadIdx.x + 31] * MASK[31 + Skip].y;



	// Write the result
	if (tid < sizeIn) {
		out[out_id] = sum;
	}
}
__global__ void Convolution_complex32_GPUkernel2(cufftComplex* out, short* in, float* bufferI, float* bufferO, int sizeCh, int nch, int* idxch) {

	const int MASK_Length = MASK_LENGTH_TOT - 1;
	extern __shared__ float sharedData[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int sizeIn = sizeCh * nch; // Total size of buffer
	const int Skip = 2 * (MASK_Length + 1);

	int out_id = tid + 2 * sizeIn / nch;
	int in_id = tid * nch + idxch[2];
	int sharedIndex = threadIdx.x + MASK_Length;

	// Load data into shared memory, handling the edges
	if (in_id - nch * MASK_Length >= 0 && in_id - nch * MASK_Length < sizeIn) { // This is for the transition between blocks
		sharedData[sharedIndex - MASK_Length] = static_cast<float>(in[in_id - nch * MASK_Length]);
	}
	else if (in_id < nch * MASK_Length) {

		sharedData[sharedIndex - MASK_Length] = bufferI[in_id]; // The first data of the segment reads the buffer from the previous segment
		//sharedData[sharedIndex - MASK_Length] = 0.0f; // The first data of the segment reads the buffer from the previous segment
	}

	if (in_id < sizeIn) {

		sharedData[sharedIndex] = static_cast<float>(in[in_id]);
	}
	else {
		sharedData[sharedIndex] = 0.0f; // Zero padding for out of range tid
	}

	if (in_id >= sizeIn - nch * MASK_Length && in_id < sizeIn)
	{
		bufferO[nch * MASK_Length - (sizeIn - in_id)] = sharedData[sharedIndex]; // Verify accuracy
	}

	// Synchronize to ensure all data is loaded
	__syncthreads();

	// Manually unrolled convolution sum
	cufftComplex sum = { 0.0f , 0.0f };



	sum.x += sharedData[threadIdx.x + 0] * MASK[0 + Skip].x + sharedData[threadIdx.x + 1] * MASK[1 + Skip].x + sharedData[threadIdx.x + 2] * MASK[2 + Skip].x + sharedData[threadIdx.x + 3] * MASK[3 + Skip].x + sharedData[threadIdx.x + 4] * MASK[4 + Skip].x
		+ sharedData[threadIdx.x + 5] * MASK[5 + Skip].x + sharedData[threadIdx.x + 6] * MASK[6 + Skip].x + sharedData[threadIdx.x + 7] * MASK[7 + Skip].x + sharedData[threadIdx.x + 8] * MASK[8 + Skip].x + sharedData[threadIdx.x + 9] * MASK[9 + Skip].x
		+ sharedData[threadIdx.x + 10] * MASK[10 + Skip].x + sharedData[threadIdx.x + 11] * MASK[11 + Skip].x + sharedData[threadIdx.x + 12] * MASK[12 + Skip].x + sharedData[threadIdx.x + 13] * MASK[13 + Skip].x + sharedData[threadIdx.x + 14] * MASK[14 + Skip].x
		+ sharedData[threadIdx.x + 15] * MASK[15 + Skip].x + sharedData[threadIdx.x + 16] * MASK[16 + Skip].x + sharedData[threadIdx.x + 17] * MASK[17 + Skip].x + sharedData[threadIdx.x + 18] * MASK[18 + Skip].x + sharedData[threadIdx.x + 19] * MASK[19 + Skip].x
		+ sharedData[threadIdx.x + 20] * MASK[20 + Skip].x + sharedData[threadIdx.x + 21] * MASK[21 + Skip].x + sharedData[threadIdx.x + 22] * MASK[22 + Skip].x + sharedData[threadIdx.x + 23] * MASK[23 + Skip].x + sharedData[threadIdx.x + 24] * MASK[24 + Skip].x
		+ sharedData[threadIdx.x + 25] * MASK[25 + Skip].x + sharedData[threadIdx.x + 26] * MASK[26 + Skip].x + sharedData[threadIdx.x + 27] * MASK[27 + Skip].x + sharedData[threadIdx.x + 28] * MASK[28 + Skip].x + sharedData[threadIdx.x + 29] * MASK[29 + Skip].x
		+ sharedData[threadIdx.x + 30] * MASK[30 + Skip].x + sharedData[threadIdx.x + 31] * MASK[31 + Skip].x;

	sum.y += sharedData[threadIdx.x + 0] * MASK[0 + Skip].y + sharedData[threadIdx.x + 1] * MASK[1 + Skip].y + sharedData[threadIdx.x + 2] * MASK[2 + Skip].y + sharedData[threadIdx.x + 3] * MASK[3 + Skip].y + sharedData[threadIdx.x + 4] * MASK[4 + Skip].y
		+ sharedData[threadIdx.x + 5] * MASK[5 + Skip].y + sharedData[threadIdx.x + 6] * MASK[6 + Skip].y + sharedData[threadIdx.x + 7] * MASK[7 + Skip].y + sharedData[threadIdx.x + 8] * MASK[8 + Skip].y + sharedData[threadIdx.x + 9] * MASK[9 + Skip].y
		+ sharedData[threadIdx.x + 10] * MASK[10 + Skip].y + sharedData[threadIdx.x + 11] * MASK[11 + Skip].y + sharedData[threadIdx.x + 12] * MASK[12 + Skip].y + sharedData[threadIdx.x + 13] * MASK[13 + Skip].y + sharedData[threadIdx.x + 14] * MASK[14 + Skip].y
		+ sharedData[threadIdx.x + 15] * MASK[15 + Skip].y + sharedData[threadIdx.x + 16] * MASK[16 + Skip].y + sharedData[threadIdx.x + 17] * MASK[17 + Skip].y + sharedData[threadIdx.x + 18] * MASK[18 + Skip].y + sharedData[threadIdx.x + 19] * MASK[19 + Skip].y
		+ sharedData[threadIdx.x + 20] * MASK[20 + Skip].y + sharedData[threadIdx.x + 21] * MASK[21 + Skip].y + sharedData[threadIdx.x + 22] * MASK[22 + Skip].y + sharedData[threadIdx.x + 23] * MASK[23 + Skip].y + sharedData[threadIdx.x + 24] * MASK[24 + Skip].y
		+ sharedData[threadIdx.x + 25] * MASK[25 + Skip].y + sharedData[threadIdx.x + 26] * MASK[26 + Skip].y + sharedData[threadIdx.x + 27] * MASK[27 + Skip].y + sharedData[threadIdx.x + 28] * MASK[28 + Skip].y + sharedData[threadIdx.x + 29] * MASK[29 + Skip].y
		+ sharedData[threadIdx.x + 30] * MASK[30 + Skip].y + sharedData[threadIdx.x + 31] * MASK[31 + Skip].y;



	// Write the result
	if (tid < sizeIn) {
		out[out_id] = sum;
	}
}

__global__ void Convolution_complex32_GPUkernel3(cufftComplex* out, short* in, float* bufferI, float* bufferO, int sizeCh, int nch, int* idxch) {

	const int MASK_Length = MASK_LENGTH_TOT - 1;
	extern __shared__ float sharedData[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int sizeIn = sizeCh * nch; // Total size of buffer
	const int Skip = 3 * (MASK_Length + 1);

	int out_id = tid + 3 * sizeIn / nch;
	int in_id = tid * nch + idxch[3];
	int sharedIndex = threadIdx.x + MASK_Length;

	// Load data into shared memory, handling the edges
	if (in_id - nch * MASK_Length >= 0 && in_id - nch * MASK_Length < sizeIn) { // This is for the transition between blocks
		sharedData[sharedIndex - MASK_Length] = static_cast<float>(in[in_id - nch * MASK_Length]);
	}
	else if (in_id < nch * MASK_Length) {

		sharedData[sharedIndex - MASK_Length] = bufferI[in_id]; // The first data of the segment reads the buffer from the previous segment
		//sharedData[sharedIndex - MASK_Length] = 0.0f; // The first data of the segment reads the buffer from the previous segment
	}

	if (in_id < sizeIn) {

		sharedData[sharedIndex] = static_cast<float>(in[in_id]);
	}
	else {
		sharedData[sharedIndex] = 0.0f; // Zero padding for out of range tid
	}

	if (in_id >= sizeIn - nch * MASK_Length && in_id < sizeIn)
	{
		bufferO[nch * MASK_Length - (sizeIn - in_id)] = sharedData[sharedIndex]; // Verify accuracy
	}

	// Synchronize to ensure all data is loaded
	__syncthreads();

	// Manually unrolled convolution sum
	cufftComplex sum = { 0.0f , 0.0f };



	sum.x += sharedData[threadIdx.x + 0] * MASK[0 + Skip].x + sharedData[threadIdx.x + 1] * MASK[1 + Skip].x + sharedData[threadIdx.x + 2] * MASK[2 + Skip].x + sharedData[threadIdx.x + 3] * MASK[3 + Skip].x + sharedData[threadIdx.x + 4] * MASK[4 + Skip].x
		+ sharedData[threadIdx.x + 5] * MASK[5 + Skip].x + sharedData[threadIdx.x + 6] * MASK[6 + Skip].x + sharedData[threadIdx.x + 7] * MASK[7 + Skip].x + sharedData[threadIdx.x + 8] * MASK[8 + Skip].x + sharedData[threadIdx.x + 9] * MASK[9 + Skip].x
		+ sharedData[threadIdx.x + 10] * MASK[10 + Skip].x + sharedData[threadIdx.x + 11] * MASK[11 + Skip].x + sharedData[threadIdx.x + 12] * MASK[12 + Skip].x + sharedData[threadIdx.x + 13] * MASK[13 + Skip].x + sharedData[threadIdx.x + 14] * MASK[14 + Skip].x
		+ sharedData[threadIdx.x + 15] * MASK[15 + Skip].x + sharedData[threadIdx.x + 16] * MASK[16 + Skip].x + sharedData[threadIdx.x + 17] * MASK[17 + Skip].x + sharedData[threadIdx.x + 18] * MASK[18 + Skip].x + sharedData[threadIdx.x + 19] * MASK[19 + Skip].x
		+ sharedData[threadIdx.x + 20] * MASK[20 + Skip].x + sharedData[threadIdx.x + 21] * MASK[21 + Skip].x + sharedData[threadIdx.x + 22] * MASK[22 + Skip].x + sharedData[threadIdx.x + 23] * MASK[23 + Skip].x + sharedData[threadIdx.x + 24] * MASK[24 + Skip].x
		+ sharedData[threadIdx.x + 25] * MASK[25 + Skip].x + sharedData[threadIdx.x + 26] * MASK[26 + Skip].x + sharedData[threadIdx.x + 27] * MASK[27 + Skip].x + sharedData[threadIdx.x + 28] * MASK[28 + Skip].x + sharedData[threadIdx.x + 29] * MASK[29 + Skip].x
		+ sharedData[threadIdx.x + 30] * MASK[30 + Skip].x + sharedData[threadIdx.x + 31] * MASK[31 + Skip].x;

	sum.y += sharedData[threadIdx.x + 0] * MASK[0 + Skip].y + sharedData[threadIdx.x + 1] * MASK[1 + Skip].y + sharedData[threadIdx.x + 2] * MASK[2 + Skip].y + sharedData[threadIdx.x + 3] * MASK[3 + Skip].y + sharedData[threadIdx.x + 4] * MASK[4 + Skip].y
		+ sharedData[threadIdx.x + 5] * MASK[5 + Skip].y + sharedData[threadIdx.x + 6] * MASK[6 + Skip].y + sharedData[threadIdx.x + 7] * MASK[7 + Skip].y + sharedData[threadIdx.x + 8] * MASK[8 + Skip].y + sharedData[threadIdx.x + 9] * MASK[9 + Skip].y
		+ sharedData[threadIdx.x + 10] * MASK[10 + Skip].y + sharedData[threadIdx.x + 11] * MASK[11 + Skip].y + sharedData[threadIdx.x + 12] * MASK[12 + Skip].y + sharedData[threadIdx.x + 13] * MASK[13 + Skip].y + sharedData[threadIdx.x + 14] * MASK[14 + Skip].y
		+ sharedData[threadIdx.x + 15] * MASK[15 + Skip].y + sharedData[threadIdx.x + 16] * MASK[16 + Skip].y + sharedData[threadIdx.x + 17] * MASK[17 + Skip].y + sharedData[threadIdx.x + 18] * MASK[18 + Skip].y + sharedData[threadIdx.x + 19] * MASK[19 + Skip].y
		+ sharedData[threadIdx.x + 20] * MASK[20 + Skip].y + sharedData[threadIdx.x + 21] * MASK[21 + Skip].y + sharedData[threadIdx.x + 22] * MASK[22 + Skip].y + sharedData[threadIdx.x + 23] * MASK[23 + Skip].y + sharedData[threadIdx.x + 24] * MASK[24 + Skip].y
		+ sharedData[threadIdx.x + 25] * MASK[25 + Skip].y + sharedData[threadIdx.x + 26] * MASK[26 + Skip].y + sharedData[threadIdx.x + 27] * MASK[27 + Skip].y + sharedData[threadIdx.x + 28] * MASK[28 + Skip].y + sharedData[threadIdx.x + 29] * MASK[29 + Skip].y
		+ sharedData[threadIdx.x + 30] * MASK[30 + Skip].y + sharedData[threadIdx.x + 31] * MASK[31 + Skip].y;



	// Write the result
	if (tid < sizeIn) {
		out[out_id] = sum;
	}
}

__global__ void Convolution_complex32_GPUkernel4(cufftComplex* out, short* in, float* bufferI, float* bufferO, int sizeCh, int nch, int* idxch) {

	const int MASK_Length = MASK_LENGTH_TOT - 1;
	extern __shared__ float sharedData[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int sizeIn = sizeCh * nch; // Total size of buffer
	const int Skip = 4 * (MASK_Length + 1);

	int out_id = tid + 4 * sizeIn / nch;
	int in_id = tid * nch + idxch[4];
	int sharedIndex = threadIdx.x + MASK_Length;

	// Load data into shared memory, handling the edges
	if (in_id - nch * MASK_Length >= 0 && in_id - nch * MASK_Length < sizeIn) { // This is for the transition between blocks
		sharedData[sharedIndex - MASK_Length] = static_cast<float>(in[in_id - nch * MASK_Length]);
	}
	else if (in_id < nch * MASK_Length) {

		sharedData[sharedIndex - MASK_Length] = bufferI[in_id]; // The first data of the segment reads the buffer from the previous segment
		//sharedData[sharedIndex - MASK_Length] = 0.0f; // The first data of the segment reads the buffer from the previous segment
	}

	if (in_id < sizeIn) {

		sharedData[sharedIndex] = static_cast<float>(in[in_id]);
	}
	else {
		sharedData[sharedIndex] = 0.0f; // Zero padding for out of range tid
	}

	if (in_id >= sizeIn - nch * MASK_Length && in_id < sizeIn)
	{
		bufferO[nch * MASK_Length - (sizeIn - in_id)] = sharedData[sharedIndex]; // Verify accuracy
	}

	// Synchronize to ensure all data is loaded
	__syncthreads();

	// Manually unrolled convolution sum
	cufftComplex sum = { 0.0f , 0.0f };



	sum.x += sharedData[threadIdx.x + 0] * MASK[0 + Skip].x + sharedData[threadIdx.x + 1] * MASK[1 + Skip].x + sharedData[threadIdx.x + 2] * MASK[2 + Skip].x + sharedData[threadIdx.x + 3] * MASK[3 + Skip].x + sharedData[threadIdx.x + 4] * MASK[4 + Skip].x
		+ sharedData[threadIdx.x + 5] * MASK[5 + Skip].x + sharedData[threadIdx.x + 6] * MASK[6 + Skip].x + sharedData[threadIdx.x + 7] * MASK[7 + Skip].x + sharedData[threadIdx.x + 8] * MASK[8 + Skip].x + sharedData[threadIdx.x + 9] * MASK[9 + Skip].x
		+ sharedData[threadIdx.x + 10] * MASK[10 + Skip].x + sharedData[threadIdx.x + 11] * MASK[11 + Skip].x + sharedData[threadIdx.x + 12] * MASK[12 + Skip].x + sharedData[threadIdx.x + 13] * MASK[13 + Skip].x + sharedData[threadIdx.x + 14] * MASK[14 + Skip].x
		+ sharedData[threadIdx.x + 15] * MASK[15 + Skip].x + sharedData[threadIdx.x + 16] * MASK[16 + Skip].x + sharedData[threadIdx.x + 17] * MASK[17 + Skip].x + sharedData[threadIdx.x + 18] * MASK[18 + Skip].x + sharedData[threadIdx.x + 19] * MASK[19 + Skip].x
		+ sharedData[threadIdx.x + 20] * MASK[20 + Skip].x + sharedData[threadIdx.x + 21] * MASK[21 + Skip].x + sharedData[threadIdx.x + 22] * MASK[22 + Skip].x + sharedData[threadIdx.x + 23] * MASK[23 + Skip].x + sharedData[threadIdx.x + 24] * MASK[24 + Skip].x
		+ sharedData[threadIdx.x + 25] * MASK[25 + Skip].x + sharedData[threadIdx.x + 26] * MASK[26 + Skip].x + sharedData[threadIdx.x + 27] * MASK[27 + Skip].x + sharedData[threadIdx.x + 28] * MASK[28 + Skip].x + sharedData[threadIdx.x + 29] * MASK[29 + Skip].x
		+ sharedData[threadIdx.x + 30] * MASK[30 + Skip].x + sharedData[threadIdx.x + 31] * MASK[31 + Skip].x;

	sum.y += sharedData[threadIdx.x + 0] * MASK[0 + Skip].y + sharedData[threadIdx.x + 1] * MASK[1 + Skip].y + sharedData[threadIdx.x + 2] * MASK[2 + Skip].y + sharedData[threadIdx.x + 3] * MASK[3 + Skip].y + sharedData[threadIdx.x + 4] * MASK[4 + Skip].y
		+ sharedData[threadIdx.x + 5] * MASK[5 + Skip].y + sharedData[threadIdx.x + 6] * MASK[6 + Skip].y + sharedData[threadIdx.x + 7] * MASK[7 + Skip].y + sharedData[threadIdx.x + 8] * MASK[8 + Skip].y + sharedData[threadIdx.x + 9] * MASK[9 + Skip].y
		+ sharedData[threadIdx.x + 10] * MASK[10 + Skip].y + sharedData[threadIdx.x + 11] * MASK[11 + Skip].y + sharedData[threadIdx.x + 12] * MASK[12 + Skip].y + sharedData[threadIdx.x + 13] * MASK[13 + Skip].y + sharedData[threadIdx.x + 14] * MASK[14 + Skip].y
		+ sharedData[threadIdx.x + 15] * MASK[15 + Skip].y + sharedData[threadIdx.x + 16] * MASK[16 + Skip].y + sharedData[threadIdx.x + 17] * MASK[17 + Skip].y + sharedData[threadIdx.x + 18] * MASK[18 + Skip].y + sharedData[threadIdx.x + 19] * MASK[19 + Skip].y
		+ sharedData[threadIdx.x + 20] * MASK[20 + Skip].y + sharedData[threadIdx.x + 21] * MASK[21 + Skip].y + sharedData[threadIdx.x + 22] * MASK[22 + Skip].y + sharedData[threadIdx.x + 23] * MASK[23 + Skip].y + sharedData[threadIdx.x + 24] * MASK[24 + Skip].y
		+ sharedData[threadIdx.x + 25] * MASK[25 + Skip].y + sharedData[threadIdx.x + 26] * MASK[26 + Skip].y + sharedData[threadIdx.x + 27] * MASK[27 + Skip].y + sharedData[threadIdx.x + 28] * MASK[28 + Skip].y + sharedData[threadIdx.x + 29] * MASK[29 + Skip].y
		+ sharedData[threadIdx.x + 30] * MASK[30 + Skip].y + sharedData[threadIdx.x + 31] * MASK[31 + Skip].y;



	// Write the result
	if (tid < sizeIn) {
		out[out_id] = sum;
	}
}


// This is the C wrapper fucntion that calls the CUDA kernel

extern "C" cudaError_t Convolution_complex32_optimized_GPU(cufftComplex * out, short* in, float* bufferI, float* bufferO, int sizeCh, int threads, int blocks,
	int LoopCount, cufftComplex * h_mask, int nch, int nfilt, int* idxchfilt, cudaStream_t streamId, cudaError_t cudaStatus)
{

	if (LoopCount == 0) { // We pass the mask points for the constant memory on the first loop
		cudaMemcpyToSymbolAsync(MASK, h_mask, nfilt * MASK_LENGTH_TOT * sizeof(cufftComplex), 0, cudaMemcpyHostToDevice, streamId);
	}

	// Launch a kernel for each signal. We should make the logic that depends on the number of references
	Convolution_complex32_GPUkernel0 << < blocks, threads, (threads + MASK_LENGTH_TOT) * sizeof(float), streamId >> > (out, (short*)in, bufferI, bufferO, sizeCh, nch, idxchfilt);
	Convolution_complex32_GPUkernel1 << < blocks, threads, (threads + MASK_LENGTH_TOT) * sizeof(float), streamId >> > (out, (short*)in, bufferI, bufferO, sizeCh, nch, idxchfilt);
	Convolution_complex32_GPUkernel2 << < blocks, threads, (threads + MASK_LENGTH_TOT) * sizeof(float), streamId >> > (out, (short*)in, bufferI, bufferO, sizeCh, nch, idxchfilt);
	Convolution_complex32_GPUkernel3 << < blocks, threads, (threads + MASK_LENGTH_TOT) * sizeof(float), streamId >> > (out, (short*)in, bufferI, bufferO, sizeCh, nch, idxchfilt);
	Convolution_complex32_GPUkernel4 << < blocks, threads, (threads + MASK_LENGTH_TOT) * sizeof(float), streamId >> > (out, (short*)in, bufferI, bufferO, sizeCh, nch, idxchfilt);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	return cudaStatus;
}