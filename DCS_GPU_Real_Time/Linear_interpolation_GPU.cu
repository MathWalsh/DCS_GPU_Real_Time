#include "Linear_interpolation_GPU.h"


// This is used for the interpolation step for the dfr correction with 2 references. We have an increasing signal of phase_dfr that is imperfect.
// The goal is to map it to a perfect increasing slope of phase_dfr. This kernel get the indexes that match from 1 vector to the other
// Lets say you have phase_imperfect[1] = 1.5 and phase_imperfect[2] = 2.5 and phase_perfect[1] = 1, 
// phase_perfect[2] = 2, phase_perfect[3] = 3; then the output of searchSortedKernel will tell us that we should use the idx = 2 of 
// phase_perfect for phase_imperfect[1] and the idx = 3 for phase_imperfect[2]
// x is the perfect grid values, bins is the imperfect grid values, n_bins number of points in the imperfect grid, y is the output indexes,
// n_interval is the maximum distance in the perfect grid that the index could be (needs to be quantified in the init file)
// We should check if n_bin < size(x), if its not the case we should handle this
__global__ void find_idx_linear_interpolation_GPUkernel(int* idx_old_to_new_grid, double* new_dfr_grid, double* old_dfr_grid, int old_n, int new_n, int n_interval) {
	int tid = blockIdx.x * (blockDim.x) + threadIdx.x;

	double val = 0.0f;

	if (tid < new_n) {
		val = new_dfr_grid[tid]; // We store the current new_dfr_grid value in a register
	}
	

	int left = 0; // This is the idx that we will use to compare the values in old_dfr_grid

	int right = static_cast<long long>(tid + n_interval); // We look +- n_interval values around the current value

	if (tid < new_n) {
		if (tid < n_interval) { // This is because we don't want negative index

			left = 1;

		}
		else {
			left = tid - n_interval;
		}

		if (val < old_dfr_grid[0]) {  // These are extrapolated values
		
			idx_old_to_new_grid[tid] = -1; 

			
		}
		else {
			while (left < right) { // When left >= right, we know that we found the proper index
				if (left >= old_n) { // These are extrapolated values
					left = old_n - 1; // This makes sure that if new_n > old_n, we don't go out of bounds
					right = old_n - 1;
				}
				if (old_dfr_grid[left] <= val) { // We check if the value in the old_grid is < val
					left += 1;
				}
				else { // This is our exit condition (we save the value at the start of the interval)
					right = left - 1;
				}
			}
		
			idx_old_to_new_grid[tid] = right;
		
			

		}
		
	}

}

// Does ther linear interpolation. y_interp[tid] = y[idx] + (x_interp - x[idx]) * (y[idx+1] - y[idx])/(x[idx+1] - x[idx])
// Basically find_idx_linear_interpolation_GPUkernel finds between which indexes of the phase_perfect vector the current tid sits. Then you
// calculate the slope between the idx point and the next and then just calculate the value at x_interp
__global__ void Linear_interpolation_GPUkernel(cufftComplex* new_IGMs, double* new_dfr_grid, cufftComplex* old_IGMs, double* old_dfr_grid, int* new_dfr_idx, int old_n, int new_n) {


	extern __shared__ cufftComplex shared_old_IGMs[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = 0;


		if (tid < new_n) { // Should be tid < new_n if new_n < old_n
		
			idx = new_dfr_idx[tid];

			if (idx == old_n - 1 || idx == -1) { // The extrapolated values are put to 0
				new_IGMs[tid].x = 0.0f;

				new_IGMs[tid].y = 0.0f;

			}
			else
			{

				// Simple linear interpolation equation
				new_IGMs[tid].x = static_cast<float>(old_IGMs[idx].x +	(new_dfr_grid[tid] - old_dfr_grid[idx]) * (old_IGMs[idx + 1].x - old_IGMs[idx].x) / (old_dfr_grid[idx + 1] - old_dfr_grid[idx]));

				new_IGMs[tid].y = static_cast<float>(old_IGMs[idx].y +	(new_dfr_grid[tid] - old_dfr_grid[idx]) * (old_IGMs[idx + 1].y - old_IGMs[idx].y) / (old_dfr_grid[idx + 1] - old_dfr_grid[idx]));
			}
			


		}

}


// This is the C wrapper function that calls the CUDA kernel
extern "C" cudaError_t Linear_interpolation_GPU(cufftComplex * new_IGMs, double* new_dfr_grid, cufftComplex * old_IGMs, double* old_dfr_grid, int* new_dfr_idx,
	int old_n, int new_n, int n_interval, int threads, int blocks, cudaStream_t streamId, cudaError_t cudaStatus) {
	
	find_idx_linear_interpolation_GPUkernel << <blocks, threads, 0, streamId >> > (new_dfr_idx, new_dfr_grid, old_dfr_grid, old_n, new_n, n_interval);

	blocks = (new_n + 2*threads - 1) / (2*threads);
	Linear_interpolation_GPUkernel << <blocks, 2*threads, 0, streamId >> > (new_IGMs, new_dfr_grid, old_IGMs, old_dfr_grid, new_dfr_idx, old_n, new_n);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	return cudaStatus;
}