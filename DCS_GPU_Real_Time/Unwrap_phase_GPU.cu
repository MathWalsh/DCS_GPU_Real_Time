#include "Unwrap_phase_GPU.h"



// Does the unwrapping in a single block (does the cumsum in each block)
// So depending on the size of the input, we will need to call this function multiple time (this is a recursive algorithm)
// For an input data of n inputs, you will get n/BlockSize blocks of unwrapped data, you thus have to call the function
// as many time as necessary to have only a n/BlockSize^m < BlockSize
// adapted from cupy library
// Could be changed to a sum reduction from mark harris webinar??
 // We unrolled all the loops. We now assum that we have a warp_size of 32 and a block size of 128 
template<unsigned int blockSize>
__global__ void UnwrapPhase_GPUkernel(double* unwrapped_phase, int* two_pi_cumsum, int* blocks_edges, float* refdfr_angle, int* blocks_edges_cumsum, int* increment_blocks_edges,
	int sizeIn, int iteration, const int warp_size, bool UnwrapDfr) {
	//"""Returns a kernel to compute an inclusive scan.

	//    It first performs an inclusive scan in each thread - block and then add the
	//    scan results for the sum / prod of the chunks.


	int tid = blockIdx.x * blockSize + threadIdx.x;

	const int lane_id = threadIdx.x % warp_size;

	// Offset shared memory pointer for sharedRef
	extern __shared__ char sharedMem[];
	int* smem0 = (int*)sharedMem;  // First (threads + 1) elements
	int* smem1 = &smem0[1];  // Start from the second element of smem0
	int* smem2 = &smem0[blockSize + 1];  // Next 32 elements

	double* sharedRef = (double*)&smem2[blockSize + 1 + warp_size];  // Offset for sharedRef after smem2
	sharedRef[threadIdx.x] = 0.0f;

	if (threadIdx.x == 0) {
		smem0[0] = 0;
	}

	// Load ref into shared memory (sharedRef)
	if (threadIdx.x < blockSize && iteration == 0) {

		sharedRef[threadIdx.x] = refdfr_angle[tid];
	}

	__syncthreads(); // Synchronize threads after loading ref to shared memory

	int x = 0; // cumsum of number of 2 pi phase diff in a block 

	if (tid < sizeIn) {
		if (iteration == 0 && UnwrapDfr) // For iteration 0, we calculate the points that have >PI difference
		{
			if (tid == 0) {
				x = 0; // There is never a 2pi unwrap on the first point
			}
			else {
				if (threadIdx.x == 0) { // This is because we are at the start of a blcok

					if ((refdfr_angle[tid - 1] - sharedRef[threadIdx.x]) >= M_PI) {
						//x = M_PI2;
						x = 1;
					}

				}
				else {
					if ((sharedRef[threadIdx.x - 1] - sharedRef[threadIdx.x]) >= M_PI) {
						//x = M_PI2;
						x = 1;
					}
				}

			}

		}
		else {
			x = blocks_edges[tid]; // We start by putting the 2*pi at each points that there is a phase wrap
		}

	}

	// Cumsum in a block begins

	// We unroll the first loop
	// First, each thread writes its own value to shared memory.
	// We then reduce and sum within a warp (we get a cumsum within each warp)
	smem1[threadIdx.x] = x;
	__syncwarp();
	// Now we perform the scan in a series of steps, doubling the offset at each step.
	if (lane_id % 2 == 1) {
		x += smem1[threadIdx.x - 1];
	}
	smem1[threadIdx.x] = x;
	__syncwarp();

	if (lane_id % 4 == 3) {
		x += smem1[threadIdx.x - 2];
	}
	smem1[threadIdx.x] = x;
	__syncwarp();

	if (lane_id % 8 == 7) {
		x += smem1[threadIdx.x - 4];
	}
	smem1[threadIdx.x] = x;
	__syncwarp();

	if (lane_id % 16 == 15) {
		x += smem1[threadIdx.x - 8];
	}
	smem1[threadIdx.x] = x;
	__syncwarp();

	if (lane_id % 32 == 31) {
		x += smem1[threadIdx.x - 16];
	}

	smem1[threadIdx.x] = x;
	__syncthreads();

	// This part is to bridge the cumsum from different warps
	if (threadIdx.x / warp_size == 0) { // threadIdx.x = 0 to 31 for each block.

		x = 0;
		// This is to get the cumsum values of each warp
		if (lane_id < 4) { // blockSize / warp_size = 4 
			// threadIdx.x < 4 (0,1,2,3)
			x = smem0[warp_size * (lane_id + 1)]; // idx 32, 64, 92, 128
		}

		// We assume that there is always 128 thread per block
		// We unroll the second loop. 
		// First, each thread writes its own value to shared memory.
		smem2[lane_id] = x;
		__syncwarp();
		// Now we perform the scan in a series of steps, doubling the offset at each step.
		// This is to achieve the cumsum accross the warps
		if (lane_id % 2 == 1) { // For odd idx, we add the previous value
			x += smem2[lane_id - 1];
		}
		smem2[lane_id] = x;
		__syncwarp();

		if (lane_id % 4 == 3) { // we add the previous 3 values 
			x += smem2[lane_id - 2];
		}


		smem2[lane_id] = x;
		__syncwarp();
		if (lane_id % 2 == 0 && lane_id >= 2) { // For even idx
			x += smem2[lane_id - 1];
		}

		if (lane_id < 4) {
			// This is the cumsum at the end of each warp (idx 32, 64, 92, 128)
			smem0[warp_size * (lane_id + 1)] = x;
		}
	}
	// First, each thread writes its own value to shared memory.
	__syncthreads();
	x = smem0[threadIdx.x];

	// Now we perform the scan in a series of steps, doubling the offset at each step.
	// We finish the cumsum by bridging the warps together with the values calculated in the previous sections
	if (lane_id % 32 == 16) {
		x += smem0[threadIdx.x - 16];
	}
	__syncwarp();
	smem0[threadIdx.x] = x;
	__syncwarp();

	if (lane_id % 16 == 8) {
		x += smem0[threadIdx.x - 8];
	}
	__syncwarp();
	smem0[threadIdx.x] = x;
	__syncwarp();

	if (lane_id % 8 == 4) {
		x += smem0[threadIdx.x - 4];
	}
	__syncwarp();
	smem0[threadIdx.x] = x;
	__syncwarp();

	if (lane_id % 4 == 2) {
		x += smem0[threadIdx.x - 2];
	}
	__syncwarp();
	smem0[threadIdx.x] = x;
	__syncwarp();

	if (lane_id % 2 == 1) {
		x += smem0[threadIdx.x - 1];
	}
	__syncwarp();
	smem0[threadIdx.x] = x;

	__syncthreads();

	// Cumsum in a block ends

	x = smem1[threadIdx.x]; // smem1 shares the same memory as smem0 but +1 on the first pointer index
	if (tid < sizeIn) {
		if (iteration == 0) { // The first iteration unwraps the phase in each block
			// For iteration 0, we will have n/block_size blocks with wrapping at the edges

			two_pi_cumsum[tid] = x;
			if (threadIdx.x == blockSize - 1) {

				blocks_edges_cumsum[tid / blockSize] = x;

			}
		}
		else {

			// For the other iterations, we calculate the cumsum between blocks.
			// For iteration 1, we calculate the cumsum of the block edges. We will thus have block_size^2 blocks 
			// unwrapped properly after the uniformAddKernel
			// For iteration 2,  we calculate the cumsum of the block edges of size  block_size^2.  We will thus have block_size^3 blocks
			// unwrapped properly after the uniformAddKernel

			increment_blocks_edges[tid] = x;

		}
	}
}

// Take the output from UnwrapPhase_GPUkernel and bridges the blocks together
// Will be called each time UnwrapPhase_GPUkernel is called to bridge the blocks

__global__ void UniformAdd_GPUkernel(double* unwrapped_phase, float* refdfr_angle, int* two_pi_cumsum, int* blocks_edges_cumsum, int* increment_blocks_edges, int sizeIn, int iteration,
	int previousBLockIncrementSize, int currentBLockIncrementSize, int Nmax, double twoPi_value) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int block_size = blockDim.x;
	int block_id = i / block_size;
	extern __shared__ int Blockincr[]; //  extern with blockSize values
	int idxBlock = block_id;

	//  The block size increases at each iteration (0,128, 128^2, 128^3,...)
	if (iteration > 1) { // This is to calculate which blockindex we need to get the value from.

		idxBlock = (block_id - previousBLockIncrementSize - 1) / currentBLockIncrementSize;
	}

	Blockincr[threadIdx.x] = increment_blocks_edges[idxBlock - 1]; // Same value for all the threads within a block. 

	if (block_id > currentBLockIncrementSize + previousBLockIncrementSize && i < sizeIn) { // This is to skip the first block because it does not need any cumsum

		two_pi_cumsum[i] += Blockincr[threadIdx.x];


		int modBlock = idxBlock % block_size; // we look at the end of each block, to get the cumsum value of this block
		if (modBlock == 0) {

			if (threadIdx.x == block_size - 1) {

				int idx = idxBlock / block_size;
				blocks_edges_cumsum[idx - 1] = Blockincr[threadIdx.x]; // This is the cumsum of each blockSize^2 blocks

			}
		}



	}
	if (i < sizeIn) {
		if (Nmax > sizeIn) { // This is to add the correct number of 2*pi when the unwrap block size is bigger than the data size (sizeIn)

			unwrapped_phase[i] = static_cast<double>(refdfr_angle[i]) + two_pi_cumsum[i] * twoPi_value;

		}
	}
}

//Reduces a warp with the sum of each thread in the warp for complex numbers
template <unsigned int blockSize>
__device__ void warpReduceSum_linearSlope(volatile double* s_sumx, volatile double* s_sumy, volatile double* s_sumx_x, volatile double* s_sumx_y, unsigned int tid) {

	if (blockSize >= 64) {
		s_sumx[tid] += s_sumx[tid + 32];
		s_sumy[tid] += s_sumy[tid + 32];
		s_sumx_x[tid] += s_sumx_x[tid + 32];
		s_sumx_y[tid] += s_sumx_y[tid + 32];
	}
	if (blockSize >= 32) {
		s_sumx[tid] += s_sumx[tid + 16];
		s_sumy[tid] += s_sumy[tid + 16];
		s_sumx_x[tid] += s_sumx_x[tid + 16];
		s_sumx_y[tid] += s_sumx_y[tid + 16];
	}
	if (blockSize >= 16) {
		s_sumx[tid] += s_sumx[tid + 8];
		s_sumy[tid] += s_sumy[tid + 8];
		s_sumx_x[tid] += s_sumx_x[tid + 8];
		s_sumx_y[tid] += s_sumx_y[tid + 8];
	}
	if (blockSize >= 8) {
		s_sumx[tid] += s_sumx[tid + 4];
		s_sumy[tid] += s_sumy[tid + 4];
		s_sumx_x[tid] += s_sumx_x[tid + 4];
		s_sumx_y[tid] += s_sumx_y[tid + 4];
	}
	if (blockSize >= 4) {

		s_sumx[tid] += s_sumx[tid + 2];
		s_sumy[tid] += s_sumy[tid + 2];
		s_sumx_x[tid] += s_sumx_x[tid + 2];
		s_sumx_y[tid] += s_sumx_y[tid + 2];
	}
	if (blockSize >= 2) {
		s_sumx[tid] += s_sumx[tid + 1];
		s_sumy[tid] += s_sumy[tid + 1];
		s_sumx_x[tid] += s_sumx_x[tid + 1];
		s_sumx_y[tid] += s_sumx_y[tid + 1];
	}
}

// We estimate the parameters of a line y = m*x + b by looking at 512 points on the line
// This reduction is based on Mark Harris algorithm
template <unsigned int blockSize>
__global__ void estimateLinearSlope(double* ydata, double* start_slope, double* end_slope, int sizeIn, int skipValues) {

	extern __shared__ double sharedData[];
	double* s_sumx = sharedData;
	double* s_sumy = s_sumx + blockDim.x; // Offset by block_size
	double* s_sumx_x = s_sumy + blockDim.x;
	double* s_sumx_y = s_sumx_x + blockDim.x; // Offset by block_size

	unsigned int tid = threadIdx.x;
	unsigned int i = skipValues * tid;

	s_sumx[tid] = 0.0f;
	s_sumy[tid] = 0.0f;
	s_sumx_x[tid] = 0.0f;
	s_sumx_y[tid] = 0.0f;

	if (i < sizeIn) {

		s_sumx[tid] = static_cast<double>(i + 1);
		s_sumy[tid] = ydata[i];
		s_sumx_x[tid] = static_cast<double>(i + 1) * static_cast<double>(i + 1);
		s_sumx_y[tid] = static_cast<double>(i + 1) * ydata[i];

	}
	__syncthreads();

	if (blockSize >= 512) {
		if (tid < 256) {
			s_sumx[tid] += s_sumx[tid + 256];
			s_sumy[tid] += s_sumy[tid + 256];
			s_sumx_x[tid] += s_sumx_x[tid + 256];
			s_sumx_y[tid] += s_sumx_y[tid + 256];
		} __syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			s_sumx[tid] += s_sumx[tid + 128];
			s_sumy[tid] += s_sumy[tid + 128];
			s_sumx_x[tid] += s_sumx_x[tid + 128];
			s_sumx_y[tid] += s_sumx_y[tid + 128];
		} __syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			s_sumx[tid] += s_sumx[tid + 64];
			s_sumy[tid] += s_sumy[tid + 64];
			s_sumx_x[tid] += s_sumx_x[tid + 64];
			s_sumx_y[tid] += s_sumx_y[tid + 64];
		} __syncthreads();
	}

	if (tid < 32) warpReduceSum_linearSlope<blockSize>(s_sumx, s_sumy, s_sumx_x, s_sumx_y, tid);


	if (tid == 0) {
		double slope = (blockDim.x * s_sumx_y[0] - s_sumx[0] * s_sumy[0])
			/ (blockDim.x * s_sumx_x[0] - s_sumx[0] * s_sumx[0]); // See matlab script for equation

		start_slope[0] = (s_sumy[0] - slope * s_sumx[0]) / blockDim.x;

		end_slope[0] = start_slope[0] + (sizeIn - 1) * slope;

	}
}



// This is the C wrapper function that calls the CUDA kernel
extern "C" cudaError_t UnwrapPhase_GPU(double* unwrapped_phase, float* refdfr_angle, int* two_pi_cumsum, int* blocks_edges_cumsum, int* increment_blocks_edges, int sizeIn, bool UnwrapDfr, bool estimateSlope,
	double* start_slope, double* end_slope, const int warp_size, int blocks, cudaStream_t streamId, cudaError_t cudaStatus) {

	double M_twoPI = 6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696506842341359642961730265646132941876892191011644634507188162569622349005682054038770422111192892458979098607639;

	// The kernels are optimized and tested for 128 threads and sizeIn < 128^4 = 268435456.
	const int threads = 128; // blocks is passed for threads = 128
	int Nmax = 0;
	int iteration = 0;
	int previousBLockIncrementSize = 0;
	int currentBLockIncrementSize = 0;

	// We perform the first unwrap, this will give us unwrap values with block size of 128
	UnwrapPhase_GPUkernel<threads> << <blocks, threads, (2 * threads + warp_size + 1) * sizeof(double), streamId >> > (unwrapped_phase, two_pi_cumsum, blocks_edges_cumsum, refdfr_angle, blocks_edges_cumsum, increment_blocks_edges,
		sizeIn, iteration, warp_size, UnwrapDfr);

	Nmax = threads;

	iteration = 1;
	int blockReduce = (blocks + threads - 1) / threads + 1; // Grid reduction 

	// We continue the cumsum until the size of the unwrap block is > sizeIn. Iteration 0 : Nmax = 128, Iteration 1 Nmax = 128^2, ...
	// This while loop is synchronous with the CPU, but since the kernel launches are asynchronous, the CPU will go through all the iterations
	// of the while loop without waiting for the GPU (needs to be verified)
	while (Nmax < sizeIn) // Will this be async with CPU thread?? if not we need to change this...
	{
		if (iteration > 2)
		{
			blockReduce = (blockReduce + threads - 1) / threads + 1; // Grid reduction (not certain if we can do it each iteration)
		}

		UnwrapPhase_GPUkernel<threads> << <blockReduce, threads, (2 * threads + warp_size + 1) * sizeof(double), streamId >> > (unwrapped_phase, two_pi_cumsum, blocks_edges_cumsum, refdfr_angle, blocks_edges_cumsum, increment_blocks_edges,
			sizeIn, iteration, warp_size, UnwrapDfr);

		Nmax *= threads;
		// This is to bridge the different blocks
		UniformAdd_GPUkernel << <blocks, threads, threads * sizeof(double), streamId >> > (unwrapped_phase, refdfr_angle, two_pi_cumsum, blocks_edges_cumsum, increment_blocks_edges, sizeIn,
			iteration, previousBLockIncrementSize, currentBLockIncrementSize, Nmax, M_twoPI);



		if (iteration == 1)
		{
			currentBLockIncrementSize = threads;
		}
		else
		{
			previousBLockIncrementSize = currentBLockIncrementSize;
			currentBLockIncrementSize *= previousBLockIncrementSize; // Necessary for UniformAdd_GPUkernel
		}

		iteration += 1;

	}

	// This is to estimate the slope of the unwrap signal. Necessary for resampling with 2 references
	if (estimateSlope) {
		// We always launch 512 points for this kernel
		int skipValuesSlope = sizeIn / 512 + 1;
		const int threadBlocksSLope = 512;
		estimateLinearSlope<threadBlocksSLope> << <1, threadBlocksSLope, 4 * threadBlocksSLope * sizeof(double), streamId >> > (unwrapped_phase, start_slope, end_slope, sizeIn, skipValuesSlope);
	}


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	return cudaStatus;
}