#include "Find_First_ZPD_GPU.h"

// Complex multiplication with 3 multiplications instead of 4
template <typename T>
__device__ T ComplexMult1(T in1, T in2, bool conj1, bool conj2)
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


// Complex magnitude^2 
__device__ float complexMagnitudeIGMs(cufftComplex a) {
	return a.x * a.x + a.y * a.y;
}

// Finds the maximum value in a warp (based on Mark Harris webinar)
template <int blockSize>
__device__ void warpReduceMax(volatile float* sdata, volatile unsigned int* idxs, unsigned int tid) {
	if (blockSize >= 64 && sdata[tid] < sdata[tid + 32]) {
		sdata[tid] = sdata[tid + 32];
		idxs[tid] = idxs[tid + 32];
	}
	if (blockSize >= 32 && sdata[tid] < sdata[tid + 16]) {
		sdata[tid] = sdata[tid + 16];
		idxs[tid] = idxs[tid + 16];
	}
	if (blockSize >= 16 && sdata[tid] < sdata[tid + 8]) {
		sdata[tid] = sdata[tid + 8];
		idxs[tid] = idxs[tid + 8];
	}
	if (blockSize >= 8 && sdata[tid] < sdata[tid + 4]) {
		sdata[tid] = sdata[tid + 4];
		idxs[tid] = idxs[tid + 4];
	}
	if (blockSize >= 4 && sdata[tid] < sdata[tid + 2]) {
		sdata[tid] = sdata[tid + 2];
		idxs[tid] = idxs[tid + 2];
	}
	if (blockSize >= 2 && sdata[tid] < sdata[tid + 1]) {
		sdata[tid] = sdata[tid + 1];
		idxs[tid] = idxs[tid + 1];
	}
}


// Finds the subpoint maximum of the xcorr and linearly interpolates the phase to have the phase at the subpoint maximum
// Based on Mark Harris reduction webinar
template<unsigned int blockSize>
__global__ void MaxReduceBlock_GPUkernel(int* idxMaxBlocks, float* MaxValBlocks, cufftComplex* IGMs, int* idxMidSegments, int ptsPerIGM, int NptsSegment, int templateSize, int Nmax, int iteration, bool FindTruePosition)
{
	extern __shared__ float sdataM[];
	unsigned int* idxs = (unsigned int*)&sdataM[blockSize];  // Use the second half of shared memory for indices
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid + ptsPerIGM / 2;
	if (iteration > 0) 
		i = blockIdx.x * blockDim.x + tid; // We shift because we want to look for the second IGM

	sdataM[tid] = 0.0f;
	
	if (iteration == 0 && i >= ptsPerIGM / 2 - 1 && i < 3 * ptsPerIGM / 2)
	{
		sdataM[tid] = complexMagnitudeIGMs(IGMs[i]);
		idxs[tid] = i;  // Initialize with i
	}
		
	else if (i < Nmax && iteration > 0)
	{
		sdataM[tid] = MaxValBlocks[i];
		idxs[tid] = idxMaxBlocks[i];
	}
		

	__syncthreads();
	// Reduction based on Mark Harriss webinar, modified to find maximum in each block (each block is 1 IGM xcorr)
	// We look at the magnitude to determine the maximum
	if (blockSize >= 512) {
		if (tid < 256 && sdataM[tid] < sdataM[tid + 256]) {
			sdataM[tid] = sdataM[tid + 256];
			idxs[tid] = idxs[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128 && sdataM[tid] < sdataM[tid + 128]) {
			sdataM[tid] = sdataM[tid + 128];
			idxs[tid] = idxs[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64 && sdataM[tid] < sdataM[tid + 64]) {
			sdataM[tid] = sdataM[tid + 64];
			idxs[tid] = idxs[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32) warpReduceMax<blockSize>(sdataM, idxs, tid);

	if (tid == 0) {
		if (FindTruePosition) {

			MaxValBlocks[blockIdx.x] = sdataM[0];
			idxMaxBlocks[blockIdx.x] = idxs[0];
			if (gridDim.x == 1) {
				// idxMaxBlocks[0] is the max position after the xcorr
				// (NptsSegment - 1) / 2 is the middle position of the xcorr
				//idxMidSegments[0] was the global position of the maximum found
				// At the end we get the global start position of the ZPD
				idxMidSegments[0] += (NptsSegment - 1) / 2 - idxMaxBlocks[0] - (templateSize-1)/2; 
				//printf("idxMidSegments[0] : %d", idxMidSegments[0]);
			}

		}
		else {

			MaxValBlocks[blockIdx.x] = sdataM[0];
			idxMaxBlocks[blockIdx.x] = idxs[0];
			if (gridDim.x == 1) {
				//printf("\n idxMaxBlocks[0]:%d, idxMaxBlocks[1]", idxMaxBlocks[blockIdx.x], idxMaxBlocks[blockIdx.x + 1]);
			}
		}


		
	}
}


// Add two complex number
__device__ cufftComplex addComplex1(cufftComplex a, cufftComplex b) {
	cufftComplex result;
	result.x = a.x + b.x;
	result.y = a.y + b.y;
	return result;
}


// Reduces a warp with the sum of each thread in the warp for complex numbers
template <unsigned int blockSize>
__device__ void warpSumReductionComplex_GPU(volatile cufftComplex* sdata, unsigned int tid) {

	if (blockSize >= 64) {
		sdata[tid].x = sdata[tid].x + sdata[tid + 32].x;
		sdata[tid].y = sdata[tid].y + sdata[tid + 32].y;
	}
	if (blockSize >= 32) {
		sdata[tid].x = sdata[tid].x + sdata[tid + 16].x;
		sdata[tid].y = sdata[tid].y + sdata[tid + 16].y;
	}
	if (blockSize >= 16) {
		sdata[tid].x = sdata[tid].x + sdata[tid + 8].x;
		sdata[tid].y = sdata[tid].y + sdata[tid + 8].y;
	}
	if (blockSize >= 8) {
		sdata[tid].x = sdata[tid].x + sdata[tid + 4].x;
		sdata[tid].y = sdata[tid].y + sdata[tid + 4].y;
	}
	if (blockSize >= 4) {

		sdata[tid].x = sdata[tid].x + sdata[tid + 2].x;
		sdata[tid].y = sdata[tid].y + sdata[tid + 2].y;
	}
	if (blockSize >= 2) {
		sdata[tid].x = sdata[tid].x + sdata[tid + 1].x;
		sdata[tid].y = sdata[tid].y + sdata[tid + 1].y;
	}
}



// Does the convolution of an arbitrary number of segments with an arbitrary size of each segment on complex values
// The template size should be NptsSegment + delayTot size.  We need to know the approximate start of each segment for this to work (idxStartSegments).
//  Delay tot is related to the uncertainty on the idxStartSegments values. 
// The more uncertain we are because the igms are mooving to much, the more delay we need to calculate
// We need to test if this function works for only 1 segment
// The reduction is based on Mark Harris webinar : Optimizing Parallel Reduction in CUDA
// It does the reduction in a block only, so the function calculates how many blocks each delay has
// Each block is then gonna be associated with a particular delay.
// We process multiple segments (multiple igms) at the same time.
template<unsigned int blockSize>
__global__ void xCorrReduceBlock_SingleIGM_GPUkernel(cufftComplex* IGMs, cufftComplex* IGMTemplate, cufftComplex* xCorrBlocks, int* idxMidSegments, int* idxMaxBLocks,
	int NptsSegment, double ptsPerIGM, int numberOfBlocksPerDelay, int sizeIn, int NdelaysPerIGM)
{
	extern __shared__ cufftComplex sdata[];

	int tid = threadIdx.x;
	int idxStartSegment = idxMaxBLocks[0];

	if (tid == 0 && blockIdx.x % (numberOfBlocksPerDelay * NdelaysPerIGM) == 0) {
		idxMidSegments[0] = idxStartSegment;  // ok for odd values of NptsSegment, saving for globalIdx in maximum reduction kernel
	}

	int delay = (blockIdx.x / numberOfBlocksPerDelay) % NdelaysPerIGM;
	int i = (blockIdx.x % numberOfBlocksPerDelay) * (blockSize * 2) + threadIdx.x + idxStartSegment - delay;
	int j = (blockIdx.x % numberOfBlocksPerDelay) * (blockSize * 2) + threadIdx.x; // To get correct idx in template   

	sdata[tid].x = 0.0f;
	sdata[tid].y = 0.0f;
	cufftComplex val1 = { 0, 0 };
	cufftComplex val2 = { 0, 0 };

	//Boundary checks
	if (j < NptsSegment) {

		val1 = ComplexMult1<cufftComplex>(IGMs[i], IGMTemplate[j], false, false);
	}
	if (j + blockSize < NptsSegment) {

		val2 = ComplexMult1<cufftComplex>(IGMs[i + blockSize], IGMTemplate[j + blockSize], false, false);

	}

	sdata[tid] = addComplex1(sdata[tid], addComplex1(val1, val2));

	__syncthreads();

	// Reduction based on Mark Harriss webinar, modified for complex numbers addition
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = addComplex1(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = addComplex1(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] = addComplex1(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) warpSumReductionComplex_GPU<blockSize>(sdata, tid);

	if (tid == 0) xCorrBlocks[blockIdx.x] = sdata[0];

}


// Mark Harris block sum reduce
// This takes the output from ConvolveReduceBlock_GPUkernel
// We know the number of blockperdelay, so we combine them to make 1 points per delay
// Thus each block contains blockPerDelay that needs to be summed together
// If blockPerDelay > blockSize, this does not work
// The output is a vector of xcorr where the xcorr of each igms are sucessive
// We might be able to remove xCorrOut and write directly to xCorrBlocks (TO DO)
template <int blockSize>
__global__ void SumReducexCorrBlocks_GPUkernel(cufftComplex* xCorrBlocks, int blocksPerDelay) {

	extern __shared__ cufftComplex sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blocksPerDelay + tid;

	sdata[tid] = { 0.0f, 0.0f };

	if (tid < blocksPerDelay) {
		sdata[tid] = xCorrBlocks[i];
	}
	__syncthreads();
	// Reduction based on Mark Harriss webinar, modified for complex numbers addition
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = addComplex1(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = addComplex1(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] = addComplex1(sdata[tid], sdata[tid + 64]); } __syncthreads(); }

	if (tid < 32) warpSumReductionComplex_GPU<blockSize>(sdata, tid);

	if (tid == 0) xCorrBlocks[blockIdx.x] = sdata[0];

}







// This is the C wrapper function that calls the CUDA kernel
// NptsLastIGMBuffer is the average number of points in the last segment of IGMs, this should be calculated by doing the mean of the subpoints positions of the previous buffer ZPDs
extern "C" cudaError_t Find_First_ZPD_GPU(int* idxMaxBLocks, float* MaxValBlocks, cufftComplex * IGMs, cufftComplex * IGMTemplate, cufftComplex * xCorrBlocks, int* idxMidSegments, int*& idxStartFirstZPD,
	 int NptsSegment, int templateSize, int ptsPerIGM,int sizeIn, int NdelaysPerIGM, int blocksPerDelay, int totalDelays, int totalBlocks, cudaStream_t streamId, cudaError_t cudaStatus) {

	const int threads = 256;  // Example value, adjust based on your requirements
	int blocks = (ptsPerIGM + threads - 1) / threads;
	int iteration = 0;
	int Nmax = 0;
	bool FindTruePosition = false;
	MaxReduceBlock_GPUkernel<threads> << <blocks, threads, 2*threads * sizeof(float), streamId >> > (idxMaxBLocks, MaxValBlocks, IGMs, idxMidSegments, ptsPerIGM, NptsSegment, templateSize, Nmax, iteration, FindTruePosition);

	Nmax = blocks;

	iteration = 1;

	int blocksReduce = (blocks + threads - 1) / threads; // Grid reduction 

	// We continue the cumsum until the size of the unwrap block is > sizeIn. Iteration 0 : Nmax = 128, Iteration 1 Nmax = 128^2, ...
	// This while loop is synchronous with the CPU, but since the kernel launches are asynchronous, the CPU will go through all the iterations
	// of the while loop without waiting for the GPU (needs to be verified)
	while (Nmax > threads) // Will this be async with CPU thread?? if not we need to change this...
	{

		MaxReduceBlock_GPUkernel<threads> <<<blocksReduce, threads, 2*threads * sizeof(float), streamId >> > (idxMaxBLocks, MaxValBlocks, IGMs, idxMidSegments, ptsPerIGM, NptsSegment, templateSize, Nmax, iteration, FindTruePosition);

		Nmax = blocksReduce;
		if (Nmax < threads) {
			iteration += 1;
			MaxReduceBlock_GPUkernel<threads> <<<1, threads, 2 * threads * sizeof(float), streamId >> > (idxMaxBLocks, MaxValBlocks, IGMs, idxMidSegments, ptsPerIGM, NptsSegment, templateSize, Nmax, iteration, FindTruePosition);
			Nmax = 0;
		}
		else {
			blocksReduce = (blocksReduce + threads - 1) / threads + 1;
			iteration += 1;
		}
		

	}

	xCorrReduceBlock_SingleIGM_GPUkernel<threads> << <totalBlocks, threads, threads * sizeof(cufftComplex), streamId >> > (IGMs, IGMTemplate, xCorrBlocks, idxMidSegments, idxMaxBLocks,
		NptsSegment, ptsPerIGM, blocksPerDelay, sizeIn, NdelaysPerIGM);

	// The switch statement is because we need to launch the right number of threadsPerBlock so that each block contains 1 delay.
	// If the NptsSegment is big, there is going to be more blocksPerDelay, so we need to launch more threadsPerBlock in SumReducexCorrBlocks_GPUkernel
	// We could add more cases to handle more threadsPerBlock for cases where the IGM is chirped a lot

	int caseSelector = (blocksPerDelay - 1) / 32; // Subtracting 1 to handle exact multiples of 32 correctly
	switch (caseSelector) {
	case 0:
		SumReducexCorrBlocks_GPUkernel<32> << <totalDelays, 32, 32 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	case 1:
		SumReducexCorrBlocks_GPUkernel<64> << <totalDelays, 64, 64 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	case 2:
		SumReducexCorrBlocks_GPUkernel<96> << <totalDelays, 96, 96 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	case 3:
		SumReducexCorrBlocks_GPUkernel<128> << <totalDelays, 128, 128 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	case 4:
		SumReducexCorrBlocks_GPUkernel<160> << <totalDelays, 160, 160 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	case 5:
		SumReducexCorrBlocks_GPUkernel<192> << <totalDelays, 192, 192 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	case 6:
		SumReducexCorrBlocks_GPUkernel<224> << <totalDelays, 224, 224 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	case 7:
		SumReducexCorrBlocks_GPUkernel<256> << <totalDelays, 256, 256 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	case 8:
		SumReducexCorrBlocks_GPUkernel<288> << <totalDelays, 288, 288 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	case 9:
		SumReducexCorrBlocks_GPUkernel<320> << <totalDelays, 320, 320 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	case 10:
		SumReducexCorrBlocks_GPUkernel<352> << <totalDelays, 352, 352 * sizeof(cufftComplex), streamId >> > (xCorrBlocks, blocksPerDelay);
		break;
	default:
		// Handle the error condition
		fprintf(stderr, "Error: Unsupported value of blocksPerDelay: %d\n", blocksPerDelay);
		break;
	}

	blocks = (NdelaysPerIGM + threads - 1) / threads;
	iteration = 0;
	Nmax = 0;
	FindTruePosition = true;
	MaxReduceBlock_GPUkernel<threads> << <blocks, threads, 2 * threads * sizeof(float), streamId >> > (idxMaxBLocks, MaxValBlocks, xCorrBlocks, idxMidSegments, NdelaysPerIGM, NptsSegment, templateSize, Nmax, iteration, FindTruePosition);

	Nmax = blocks;

	iteration = 1;

	blocksReduce = (blocks + threads - 1) / threads; // Grid reduction 

	// We continue the cumsum until the size of the unwrap block is > sizeIn. Iteration 0 : Nmax = 128, Iteration 1 Nmax = 128^2, ...
	// This while loop is synchronous with the CPU, but since the kernel launches are asynchronous, the CPU will go through all the iterations
	// of the while loop without waiting for the GPU (needs to be verified)

	if (Nmax < threads) {
		MaxReduceBlock_GPUkernel<threads> << <1, threads, 2 * threads * sizeof(float), streamId >> > (idxMaxBLocks, MaxValBlocks, xCorrBlocks, idxMidSegments, NdelaysPerIGM, NptsSegment, templateSize, Nmax, iteration, FindTruePosition);
	}
	else {

		while (Nmax > threads) // Will this be async with CPU thread?? if not we need to change this...
		{

			MaxReduceBlock_GPUkernel<threads> << <blocksReduce, threads, 2 * threads * sizeof(float), streamId >> > (idxMaxBLocks, MaxValBlocks, xCorrBlocks, idxMidSegments, NdelaysPerIGM, NptsSegment, templateSize, Nmax, iteration, FindTruePosition);

			Nmax = blocksReduce;
			if (Nmax < threads) {
				iteration += 1;
				MaxReduceBlock_GPUkernel<threads> << <1, threads, 2 * threads * sizeof(float), streamId >> > (idxMaxBLocks, MaxValBlocks, xCorrBlocks, idxMidSegments, NdelaysPerIGM, NptsSegment, templateSize, Nmax, iteration, FindTruePosition);
				Nmax = 0;
			}
			else {
				blocksReduce = (blocksReduce + threads - 1) / threads + 1;
				iteration += 1;
			}

		}
	}
	cudaMemcpyAsync(idxStartFirstZPD, idxMidSegments, sizeof(int), cudaMemcpyDeviceToHost, streamId);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	return cudaStatus;
}