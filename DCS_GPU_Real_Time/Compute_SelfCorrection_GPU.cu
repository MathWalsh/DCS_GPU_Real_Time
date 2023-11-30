#include "Compute_SelfCorrection_GPU.h"


// Calculate difference between spline idx distances to find the spline coefficients
__global__ void computeHKernel(const double* x, double* h, const int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) {
		h[idx] = x[idx + 1] - x[idx];

	}
}

// Populate the right hand side of the matrix equation Dm = r to find the spline coefficients (see matlab script or chat gpt4 conversation)
__global__ void computeRKernel(const double* y, const double* h, double* r, const int n, bool compute_dfr) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n - 1) {
		if (compute_dfr) {
			const double yval2 = idx + 2;
			const double yval1 = idx + 1;
			const double yval = idx;
			r[idx] = 6.0f * ((yval2 - yval1) / h[idx + 1] - (yval1 - yval) / h[idx]);
		}
		else {
			r[idx] = 6.0f * ((y[idx + 2] - y[idx + 1]) / h[idx + 1] - (y[idx + 1] - y[idx]) / h[idx]);
		}


	}
}

// Populate the left hand side of the matrix equation Dm = r to find the spline coefficients (see matlab script or chat gpt4 conversation)
__global__ void computeDKernel(double* d_h, double* d_D, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n - 1) {
		if (i < n - 2) {
			d_D[i * (n - 1) + i] = 2.0 * (d_h[i] + d_h[i + 1]);
			d_D[i * (n - 1) + i + 1] = d_h[i + 1];
			d_D[(i + 1) * (n - 1) + i] = d_h[i + 1];
		}
		else {
			d_D[i * (n - 1) + i] = 2.0 * (d_h[n - 2] + d_h[n - 1]);
		}
	}
}
// Used to circshift the m spline coefficients to have the natural spline boundary coefficients correctly (0 for the first and last coeff)
__global__ void circshiftRightKernel(double* d_array, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) {
		if (idx == 0) {
			// Store the last element in a variable declared in __shared__ memory
			// This assumes that the grid size is at least 2 blocks, so this thread
			// will not be the last thread to execute.
			extern __shared__ double lastElement[];
			lastElement[0] = d_array[n - 1];
			__syncthreads();  // Wait for all threads to reach this point (and lastElement to be stored)

			// Set the first element to the last
			d_array[0] = lastElement[0];
		}
		else {
			// Shift elements to the right
			d_array[idx] = d_array[idx - 1];
		}
		if (idx == (n - 1)) { // For multi-block
			d_array[idx] = 0;
		}
		if (idx == 0) {
			d_array[idx] = 0;
		}
	}
}


// we need to link cusolver.lib, cublas.lib cublasLt.lib and cusparse.lib
/**
 * @param d_x Device pointer to the vector of nodes.
 * @param d_y Device pointer to the vector of function values at nodes.
 * @param n Number of intervals (length of d_x minus 1).
 * @param stream Cuda stream).
 */
void computeSplineCoefficientsAsync(cusolverDnHandle_t handle, double* d_x, double* d_y, double* d_r, double* d_h, double* d_D, double* d_work, int* devInfo, int lwork, int n, cudaStream_t stream, bool compute_dfr) {


	// Step 1: Calculate the differences 'h' between nodes.
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	cudaMemsetAsync(d_D, 0.0f, n * n * sizeof(double), stream);
	computeHKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (d_x, d_h, n);

	// Step 2: Calculate r based on the differences of y and h values.
	computeRKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (d_y, d_h, d_r, n, compute_dfr);

	// Step 3: Create and setup the diagonal matrix D and right-hand side vector r as given in the MATLAB code.
	computeDKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (d_h, d_D, n);

	// Step 4.1: Perform Cholesky factorization using cusolverDnSpotrf
	// handle: the cuSOLVER handle
	// CUBLAS_FILL_MODE_UPPER: We choose to fill the upper triangular part of the matrix
	// n-1: Order of the matrix D (assuming D is n-1 x n-1)
	// d_D: Pointer to the matrix in device memory (should be changed to D if you have a matrix D on device)
	// n-1: Leading dimension of D
	// d_work: Workspace for computations, computed using bufferSize function
	// lwork: Size of the workspace
	// devInfo: Info on completion, if 0 then the factorization is successful
	cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_UPPER, n - 1, d_D, n - 1, d_work, lwork, devInfo);


	// Step 4.2: Solve the system D*m = r' using the Cholesky factorization from previous step
	// handle: the cuSOLVER handle
	// CUBLAS_FILL_MODE_UPPER: Using the upper triangular part as factorized before
	// n-1: Order of the matrix D
	// 1: Number of right hand side columns (we are solving for one vector m)
	// d_r: Pointer to the factorized matrix in device memory (result from Cholesky factorization)
	// n-1: Leading dimension of D
	// d_r: Right hand side vector
	// n-1: Leading dimension of r
	// devInfo: Info on completion, if 0 then the solve is successful
	cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_UPPER, n - 1, 1, d_D, n - 1, d_r, n - 1, devInfo);

	circshiftRightKernel << <blocksPerGrid + 1, threadsPerBlock, sizeof(double), stream >> > (d_r, n + 1);

}

// Find in which interval of splines each point in the x vector should go
__global__ void find_idx_spline_interpolation_GPUkernel(double* uniform_grid_idx, int n_uniform_grid, double* idx_subpoint_ZPD, int n_ZPD, int* idx_uniform_to_splines, int n_interval) {
	int tid = blockIdx.x * (blockDim.x) + threadIdx.x;


	int Nblocks = n_uniform_grid / n_ZPD; // ptsPerIGM basically


	double idx_uniform = 0;
	if (tid < n_uniform_grid) {
		idx_uniform = uniform_grid_idx[tid];
	}
	int idx_ZPD = 0;

	if ((idx_uniform - idx_subpoint_ZPD[0]) > 0.0f) { // IF idx_uniform is after first ZPD
		idx_ZPD = tid / Nblocks + 1; // start at 1

	}

	int left = 0;


	if (idx_uniform < idx_subpoint_ZPD[0]) { // Take first interval if idx_uniform is before the first ZPD
		idx_uniform_to_splines[tid] = 0;
	}
	else if (idx_uniform >= idx_subpoint_ZPD[n_ZPD - 1]) { // Take last ZPD if idx_uniform is after the last ZPD
		idx_uniform_to_splines[tid] = n_ZPD - 1;
	}
	else if (idx_ZPD < n_ZPD + 1) {

		int right = static_cast<long long>(idx_ZPD + n_interval);
		if (right > n_ZPD) {
			right = n_ZPD - 1;
		}
		int left = idx_ZPD - n_interval;
		if (left < 0) {
			left = 1;
		}
		while (left < right) {
			if (idx_subpoint_ZPD[left] <= idx_uniform) {
				left += 1;
			}
			else {
				right = left;
			}
		}

		idx_uniform_to_splines[tid] = right;
	}
}


// Evaluate the spline with the spline coefficients and the points (see matlab script for more details)
// Need double for dfr
__device__ double cubicSplineEvaldfr(double x0, double x1, double y0, double y1, double m0, double m1, double xi) {
	double h = x1 - x0;
	double a = (x1 - xi) / h;
	double b = (xi - x0) / h;
	return (a * y0 + b * y1 + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * (h * h) / 6.0);
}


// float ok for f0
__forceinline__ __device__ float cubicSplineEvalf0(float x0, float x1, float y0, float y1, float m0, float m1, float xi) {
	float h = x1 - x0;
	float a = (x1 - xi) / h;
	float b = (xi - x0) / h;
	float a2 = a * a;
	float b2 = b * b;
	//float a3 = a2 * a;
	float b3 = b2 * b;

	// Combine terms to reduce the number of operations
	return (a * y0 + b * y1 + (h * h / 6.0f) * ((a2 * a - a) * m0 + (b3 - b) * m1));
}

// Evaluate the cubic spline with the coefficients calculated in computeSplineCoefficients and the intervals computed in find_idx_spline_interpolation
__global__ void cubicSplineEvaluation_GPUkernel(const double* idx_subpoint_ZPD, const double* nonUniformGrid, const double* spline_coefficients, const double* uniformGrid,
	float* SplineOutputf0, double* SplineOutputdfr, int* intervals, int n_SplineOutput, bool dfr_normalize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n_SplineOutput) {

		int i = intervals[idx] - 1; // intervals are computed in find_idx_spline_interpolation
		// Ensure the interval index is not negative using a ternary operator
		i = (i < 0) ? 0 : i;



		if (dfr_normalize) { // 

			double x0 = idx_subpoint_ZPD[i];
			double x1 = idx_subpoint_ZPD[i + 1];
			double y0 = static_cast<double>(i);
			double y1 = static_cast<double>(i + 1);
			double m0 = spline_coefficients[i];
			double m1 = spline_coefficients[i + 1];
			double xi_val = uniformGrid[idx];
			SplineOutputdfr[idx] = cubicSplineEvaldfr(x0, x1, y0, y1, m0, m1, xi_val);

		}
		else {
			float x0 = static_cast<float>(idx_subpoint_ZPD[i]);
			float x1 = static_cast<float>(idx_subpoint_ZPD[i + 1]);
			float y0 = static_cast<float>(nonUniformGrid[i]);
			float y1 = static_cast<float>(nonUniformGrid[i + 1]);
			float m0 = static_cast<float>(spline_coefficients[i]);
			float m1 = static_cast<float>(spline_coefficients[i + 1]);
			float xi_val = static_cast<float>(uniformGrid[idx]);
			SplineOutputf0[idx] = cubicSplineEvalf0(x0, x1, y0, y1, m0, m1, xi_val);
		}

	}
}

// Slow phase correction IGMsPC*exp(-1j*angle_f0);
__global__ void SelfCorrection_phaseCorrection_GPUkernel(cufftComplex* IGMs, float* Spline_f0_grid, int sizeIn) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < sizeIn) {
		// Pre-fetching angle value to a register to avoid re-reading from global memory.
		float angle = -1 * Spline_f0_grid[tid];

		// Using local variables to minimize repeated global memory access.
		float IGM_Valx = IGMs[tid].x;
		float IGM_Valy = IGMs[tid].y;

		// Performing calculations using local variables.
		float cosine = cosf(angle);
		float sine = sinf(angle);
		float p1 = IGM_Valx * cosine;
		float p2 = (IGM_Valx + IGM_Valy) * (cosine + sine); // Simplified to use cosine - sine based on the angle negation.
		float p3 = IGM_Valy * sine;

		// Writing results back to global memory in a single coalesced write per thread.
		IGMs[tid].x = p1 - p3;
		IGMs[tid].y = p2 - p1 - p3;
	}
}


// This is the C wrapper function that calls the CUDA kernel
// NptsPerIGM is the average number of points in the last segment of IGMs
extern "C" cudaError_t Compute_SelfCorrection_GPU(cufftComplex * IGMsOut, cufftComplex * IGMsIn, float* Spline_grid_f0, double* Spline_grid_dfr, double* selfCorr_xaxis_uniform_grid, int* idx_nonuniform_to_uniform,
	double* spline_coefficients_f0, double* spline_coefficients_dfr, double* max_idx_sub, double* phase_sub, double* start_dfr_grid, double* end_dfr_grid, int ptsPerIGM, int NIGMs, int sizeIn,
	int n_interval, int threads, int blocks, double* d_h, double* d_D, double* d_work, int* devInfo, int lwork, cusolverDnHandle_t cuSolver_handle, cudaStream_t streamId, cudaError_t cudaStatus) {

	// Compute spline coeff for f0, should do it in parallel with dfr
	bool compute_dfr = false;
	//computeSplineCoefficientsAsync(cuSolver_handle1, max_idx_sub, phase_sub, spline_coefficients_f0, d_h1, d_D1, d_work1, devInfo1, lwork1, NIGMs - 1, streamId1, compute_dfr); cant launch both at same time
	computeSplineCoefficientsAsync(cuSolver_handle, max_idx_sub, phase_sub, spline_coefficients_f0, d_h, d_D, d_work, devInfo, lwork, NIGMs - 1, streamId, compute_dfr);
	compute_dfr = true;
	computeSplineCoefficientsAsync(cuSolver_handle, max_idx_sub, phase_sub, spline_coefficients_dfr, d_h, d_D, d_work, devInfo, lwork, NIGMs - 1, streamId, compute_dfr);


	Linspace_GPU(selfCorr_xaxis_uniform_grid, start_dfr_grid, end_dfr_grid, sizeIn, threads, blocks, 1, streamId, cudaSuccess);

	int ninterval = 1; // Is it always 1??
	// Find in which interval of splines each point in the selfCorr_xaxis_uniform_grid vector should go
	find_idx_spline_interpolation_GPUkernel << < blocks, threads, 0, streamId >> > (selfCorr_xaxis_uniform_grid, sizeIn, max_idx_sub, NIGMs, idx_nonuniform_to_uniform, ninterval);

	// Compute dfr and f0 grids in parallel if possible
	threads /= 2;
	blocks = (sizeIn + threads - 1) / threads;

	bool dfr_normalize = false; // this is to normalize the Splin_dfr_grid to -0.5 to NIGMs-0.5
	cubicSplineEvaluation_GPUkernel << < blocks, threads, 0, streamId >> > (max_idx_sub, phase_sub, spline_coefficients_f0, selfCorr_xaxis_uniform_grid,
		Spline_grid_f0, Spline_grid_dfr, idx_nonuniform_to_uniform, sizeIn, dfr_normalize);

	dfr_normalize = true;
	cubicSplineEvaluation_GPUkernel << < blocks, threads, 0, streamId >> > (max_idx_sub, phase_sub, spline_coefficients_dfr, selfCorr_xaxis_uniform_grid,
		Spline_grid_f0, Spline_grid_dfr, idx_nonuniform_to_uniform, sizeIn, dfr_normalize);

	//// Do slow phase correction
	threads *= 2;
	blocks = (sizeIn + threads - 1) / threads;
	SelfCorrection_phaseCorrection_GPUkernel << < blocks, threads, 0, streamId >> > (IGMsIn, Spline_grid_f0, sizeIn);

	int NptsSelfCorrection = NIGMs * ptsPerIGM; // moove this inside GPU function

	Linspace_GPU(selfCorr_xaxis_uniform_grid, start_dfr_grid, end_dfr_grid, NptsSelfCorrection, threads, blocks, 2, streamId, cudaSuccess);
	///*	 Remove + 5 if possible. The +5 is necessary when the igm is not cropped properly at the beginning and at the end. If there are too many points at the beginning
	// The Spline_grid starts too low compared to the selfCorr_xaxis_uniform_grid so we need to look around points further...*/
	Linear_interpolation_GPU(IGMsOut, selfCorr_xaxis_uniform_grid, IGMsIn, Spline_grid_dfr, idx_nonuniform_to_uniform, sizeIn, NptsSelfCorrection, n_interval + 25,
		threads, blocks, streamId, cudaSuccess);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	return cudaStatus;
}
