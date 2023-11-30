#include "Find_IGMs_ZPD_GPU.h"


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

// Add two complex number
__device__ cufftComplex addComplex(cufftComplex a, cufftComplex b) {
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
__global__ void xCorrReduceBlock_GPUkernel(cufftComplex* IGMs, cufftComplex* IGMTemplate, cufftComplex* xCorrBlocks, int* idxMidSegments, int idxStartFirstSegment,
	int NptsSegment, double ptsPerIGM, int numberOfBlocksPerDelay, int sizeIn, int NdelaysPerIGM)
{
	extern __shared__ cufftComplex sdata[];

	int tid = threadIdx.x;
	//unsigned int numberOfBlocksPerDelay = (NptsSegment + blockSize * 2 - 1) / (blockSize * 2);
	int segmentIdx = blockIdx.x / (numberOfBlocksPerDelay * NdelaysPerIGM);
	int idxStartSegment = idxStartFirstSegment + round(ptsPerIGM * segmentIdx) + NdelaysPerIGM / 2;

	if (tid == 0 && blockIdx.x % (numberOfBlocksPerDelay * NdelaysPerIGM) == 0) {
		idxMidSegments[segmentIdx] = idxStartSegment + (NptsSegment - 1) / 2;  // ok for odd values of NptsSegment, saving for globalIdx in maximum reduction kernel


	}
	int i = (blockIdx.x % numberOfBlocksPerDelay) * (blockSize * 2) + threadIdx.x + idxStartSegment;
	int gridSize = blockSize * 2 * gridDim.x;
	int delay = (blockIdx.x / numberOfBlocksPerDelay) % NdelaysPerIGM;
	int j = (blockIdx.x % numberOfBlocksPerDelay) * (blockSize * 2) + threadIdx.x + delay; // To get correct idx in template   

	sdata[tid].x = 0.0f;
	sdata[tid].y = 0.0f;
	cufftComplex val1 = { 0, 0 };
	cufftComplex val2 = { 0, 0 };
	while (i < idxStartSegment + NptsSegment) { // Reduction when we have a lot of points, should not be needed for self-correction

		//Boundary checks
		if (i < idxStartSegment + NptsSegment && j < NptsSegment + NdelaysPerIGM) {

			val1 = ComplexMult<cufftComplex>(IGMs[i], IGMTemplate[j], false, false);
		}
		if (i + blockSize < idxStartSegment + NptsSegment && j + blockSize < NptsSegment + NdelaysPerIGM) {
			val2 = ComplexMult<cufftComplex>(IGMs[i + blockSize], IGMTemplate[j + blockSize], false, false);
		}

		sdata[tid] = addComplex(sdata[tid], addComplex(val1, val2));


		i += gridSize;
		j += gridSize;
	}

	__syncthreads();

	// Reduction based on Mark Harriss webinar, modified for complex numbers addition
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = addComplex(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = addComplex(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] = addComplex(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
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
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = addComplex(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = addComplex(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] = addComplex(sdata[tid], sdata[tid + 64]); } __syncthreads(); }

	if (tid < 32) warpSumReductionComplex_GPU<blockSize>(sdata, tid);


	if (tid == 0) xCorrBlocks[blockIdx.x] = sdata[0];

}


// Complex magnitude^2 
__device__ float complexMagnitude(cufftComplex a) {
	return a.x * a.x + a.y * a.y;
}

// Double Complex magnitude^2 
__device__ double complexMagnitudeD(const cufftComplex& val) {
	return sqrt(static_cast<double>(val.x) * static_cast<double>(val.x) +
		static_cast<double>(val.y) * static_cast<double>(val.y));
}


// Does a parabolic fit based on 3 input points and finds the position of the maximum of that parabola
__device__ double Poly3Fit(unsigned int x1, double y1, unsigned int x2, double y2, unsigned int x3, double y3)
{
	double x1_d = static_cast<double>(x1);
	double x2_d = static_cast<double>(x2);
	double x3_d = static_cast<double>(x3);

	double alpha = pow(x2_d, 2) / pow(x1_d, 2);
	double beta = pow(x3_d, 2) / pow(x1_d, 2);

	double numc = y3 - beta * y1 + x1_d * beta / (x2_d - x1_d * alpha) * (y2 - alpha * y1) - x3_d / (x2_d - x1_d * alpha) * (y2 - alpha * y1);
	double denumc = 1 - beta - x3_d / (x2_d - x1_d * alpha) * (1 - alpha) + x1_d * beta / (x2_d - x1_d * alpha) * (1 - alpha);
	double c = numc / denumc;
	double b = 1.0f / (x2_d - x1_d * alpha) * (y2 - alpha * y1 - c * (1 - alpha));
	double a = (y1 - b * x1_d - c) / pow(x1_d, 2);

	return -b / (2 * a);

}
// Does a linear fit based on 2 input points. We then interpolate based on the xinterp value
__device__ double Poly1Fit(unsigned int x1, double y1, unsigned int x2, double y2, double xinterp)
{
	double x1_d = static_cast<double>(x1);
	double x2_d = static_cast<double>(x2);
	double xinterp_d = static_cast<double>(xinterp);

	double b = (y2 - y1 * x2_d / x1_d) / (1 - x2_d / x1_d);
	double a = (y1 - b) / x1_d;

	return a * xinterp_d + b;
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
__global__ void MaxReduceBlock_GPUkernel(cufftComplex* xCorr, double* max_idx_sub, double* phase_sub, int* idxMidSegments, int NIGMs, int NdelaysPerIGM, unsigned int blocksPerDelay,
	int sizeIn, int sizeInCropped)
{
	extern __shared__ float sdataM[];
	unsigned int* idxs = (unsigned int*)&sdataM[blockSize];  // Use the second half of shared memory for indices

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * NdelaysPerIGM + tid;
	sdataM[tid] = 0.0f;
	idxs[tid] = tid;  // Initialize with thread ID

	if (tid < NdelaysPerIGM && i < sizeIn) {
		sdataM[tid] = complexMagnitude(xCorr[i]);
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
		// Find the subpoint maximum with cubic interpolation between 3 points
		unsigned int idxMaxTot = blockIdx.x * NdelaysPerIGM + idxs[0];
		double y1 = complexMagnitudeD(xCorr[idxMaxTot - 1]);
		double y2 = complexMagnitudeD(xCorr[idxMaxTot]);
		double y3 = complexMagnitudeD(xCorr[idxMaxTot + 1]);
		double max_idx_temp = Poly3Fit(idxs[0] - 1, y1, idxs[0], y2, idxs[0] + 1, y3);

		max_idx_sub[blockIdx.x] = (idxMidSegments[blockIdx.x] - max_idx_temp + NdelaysPerIGM / 2) / (sizeInCropped - 1); // global normalized idx of the ZPD (should be n or n-1??), works for NdelaysPerIGM even
		if (max_idx_temp > static_cast<float>(idxs[0])) { // To the right of max
			// Find the phase at the subpoint idx found with linear interpolation
			double angle1 = static_cast<double>(atan2f(xCorr[idxMaxTot].y, xCorr[idxMaxTot].x));
			double angle2 = static_cast<double>(atan2f(xCorr[idxMaxTot + 1].y, xCorr[idxMaxTot + 1].x));
			phase_sub[blockIdx.x] = Poly1Fit(idxs[0], angle1, idxs[0] + 1, angle2, max_idx_temp);
		}
		else { // To the left of max
			double angle1 = static_cast<double>(atan2f(xCorr[idxMaxTot - 1].y, xCorr[idxMaxTot - 1].x));
			double angle2 = static_cast<double>(atan2f(xCorr[idxMaxTot].y, xCorr[idxMaxTot].x));
			phase_sub[blockIdx.x] = Poly1Fit(idxs[0] - 1, angle1, idxs[0], angle2, max_idx_temp);

		}
		//printf("\nidxs[0]: %d, phase_sub[%d]: %f, max_idx_sub : %f, sizeInCropped : %d", idxs[0], blockIdx.x,  phase_sub[blockIdx.x], max_idx_sub[blockIdx.x], sizeInCropped);
	}
}


template <int blockSize>
__device__ void warpReduceMean(volatile double* sdataMean, unsigned int tid) {
	if (blockSize >= 64) {
		sdataMean[tid] += sdataMean[tid + 32];
	}
	if (blockSize >= 32) {
		sdataMean[tid] += sdataMean[tid + 16];
	}
	if (blockSize >= 16) {
		sdataMean[tid] += sdataMean[tid + 8];
	}
	if (blockSize >= 8) {
		sdataMean[tid] += sdataMean[tid + 4];
	}
	if (blockSize >= 4) {
		sdataMean[tid] += sdataMean[tid + 2];
	}
	if (blockSize >= 2) {
		sdataMean[tid] += sdataMean[tid + 1];
	}
}

// Calculate the mean points between IGMs for cropping and update self-correction parameters. ptsPerIGMSegment will be used to keep track of dfr in real time
template<unsigned int blockSize>
__global__ void UpdateSelfCorrectionParams_GPUkernel(double* max_idx_sub, int NIGMs, int sizeInCropped, double* d_ptsPerIGMSegment)
{

	extern __shared__ double sdataMean[];
	int tid = threadIdx.x;

	sdataMean[tid] = 0.0f;

	if (tid < NIGMs && tid > 0) {
		sdataMean[tid] = (max_idx_sub[tid] - max_idx_sub[tid - 1]) * (sizeInCropped - 1);
	}
	__syncthreads();
	// Reduction based on Mark Harriss webinar, modified to find maximum in each block (each block is 1 IGM xcorr)
	// We look at the magnitude to determine the maximum
	if (blockSize >= 512) {
		if (tid < 256) {
			sdataMean[tid] += sdataMean[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdataMean[tid] += sdataMean[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdataMean[tid] += sdataMean[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32) warpReduceMean<blockSize>(sdataMean, tid);

	if (tid == 0) {
		d_ptsPerIGMSegment[0] = sdataMean[0] / (NIGMs - 1); // mean points between IGMs // TO DOO
	}


}

// This is the C wrapper function that calls the CUDA kernel
// NptsLastIGMBuffer is the average number of points in the last segment of IGMs, this should be calculated by doing the mean of the subpoints positions of the previous buffer ZPDs
extern "C" cudaError_t Find_IGMs_ZPD_GPU(double* max_idx_sub, double* phase_sub, cufftComplex * IGMs, cufftComplex * IGMTemplate, cufftComplex * xCorrBlocks,
	int* idxMidSegments, int idxStartFirstSegment, int NptsSegment, int NIGMs, double ptsPerIGM, int sizeIn, int sizeInCropped, int NdelaysPerIGM, int blocksPerDelay, int totalDelays, int totalBlocks,
	double* d_ptsPerIGMSegment, cudaStream_t streamId, cudaError_t cudaStatus) {

	const int threadsPerBlock = 256;  // Example value, adjust based on your requirements

	xCorrReduceBlock_GPUkernel<threadsPerBlock> << <totalBlocks, threadsPerBlock, threadsPerBlock * sizeof(cufftComplex), streamId >> > (IGMs, IGMTemplate, xCorrBlocks, idxMidSegments, idxStartFirstSegment,
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


	// The switch statement is because we need to launch the right number of threadsPerBlock so that each block contains at least NdelaysPerIGM.
	// If the dfr is noisy, there is going to be more movement on the ZPD position, so we need to calulate more delays for each IGM
	// We could add more cases to handle more threadsPerBlock for cases where the ZPD is moving a lot

	caseSelector = (NdelaysPerIGM - 1) / 32; // Adjusting the range for each case
	switch (caseSelector) {
	case 0:
		MaxReduceBlock_GPUkernel<32> << <NIGMs, 32, 2 * 32 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	case 1:
		MaxReduceBlock_GPUkernel<64> << <NIGMs, 64, 2 * 64 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	case 2:
		MaxReduceBlock_GPUkernel<96> << <NIGMs, 96, 2 * 96 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	case 3:
		MaxReduceBlock_GPUkernel<128> << <NIGMs, 128, 2 * 128 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	case 4:
		MaxReduceBlock_GPUkernel<160> << <NIGMs, 160, 2 * 160 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	case 5:
		MaxReduceBlock_GPUkernel<192> << <NIGMs, 192, 2 * 192 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	case 6:
		MaxReduceBlock_GPUkernel<224> << <NIGMs, 224, 2 * 224 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	case 7:
		MaxReduceBlock_GPUkernel<256> << <NIGMs, 256, 2 * 256 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	case 8:
		MaxReduceBlock_GPUkernel<288> << <NIGMs, 288, 2 * 288 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	case 9:
		MaxReduceBlock_GPUkernel<320> << <NIGMs, 320, 2 * 320 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	case 10:
		MaxReduceBlock_GPUkernel<352> << <NIGMs, 352, 2 * 352 * sizeof(float), streamId >> > (xCorrBlocks, max_idx_sub, phase_sub, idxMidSegments, NIGMs, NdelaysPerIGM, blocksPerDelay, sizeIn, sizeInCropped);
		break;
	default:
		// Handle the error condition
		fprintf(stderr, "Error: Unsupported value of NdelaysPerIGM: %d\n", NdelaysPerIGM);
		break;
	}


	caseSelector = (NIGMs - 1) / 32; // Adjusting the range for each case. For now NIGMs per batch < 352
	switch (caseSelector) {
	case 0:
		UpdateSelfCorrectionParams_GPUkernel<32> << <1, 32, 2 * 32 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
		break;
	case 1:
		UpdateSelfCorrectionParams_GPUkernel<64> << <1, 64, 2 * 64 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
		break;
	case 2:
		UpdateSelfCorrectionParams_GPUkernel<96> << <1, 96, 2 * 96 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
		break;
	case 3:
		UpdateSelfCorrectionParams_GPUkernel<128> << <1, 128, 2 * 128 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
		break;
	case 4:
		UpdateSelfCorrectionParams_GPUkernel<160> << <1, 160, 2 * 160 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
		break;
	case 5:
		UpdateSelfCorrectionParams_GPUkernel<192> << <1, 192, 2 * 192 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
		break;
	case 6:
		UpdateSelfCorrectionParams_GPUkernel<224> << <1, 224, 2 * 224 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
		break;
	case 7:
		UpdateSelfCorrectionParams_GPUkernel<256> << <1, 256, 2 * 256 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
		break;
	case 8:
		UpdateSelfCorrectionParams_GPUkernel<288> << <1, 288, 2 * 288 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
		break;
	case 9:
		UpdateSelfCorrectionParams_GPUkernel<320> << <1, 320, 2 * 320 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
		break;
	case 10:
		UpdateSelfCorrectionParams_GPUkernel<352> << <1, 352, 2 * 352 * sizeof(double), streamId >> > (max_idx_sub, NIGMs, sizeInCropped, d_ptsPerIGMSegment);
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