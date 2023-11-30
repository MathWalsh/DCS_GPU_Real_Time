#include "FastPhaseCorrection_GPU.h"


// Extract angle of 1 complex number
__forceinline__ __device__ float AngleComplex(cufftComplex in) { // forceinline for compilation
    return atan2f(in.y, in.x);
}


// Complex multiplication with 3 multiplications instead of 4
template <typename T>
__forceinline__ __device__ T ComplexMult(T in1, T in2, bool conj1, bool conj2)
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

// Fast phase correction IGMs*exp(1j*angle_ref);
// can conjugate to get the two possibilities exp(1j) or exp(-1j)

__global__ void FastPhaseCorrection_GPUkernel(
    cufftComplex* outIGMs, cufftComplex* outref1, cufftComplex* inIGMs,
    cufftComplex* inFopt1, cufftComplex* inFopt2, int sizeIn,
    bool conjFopt1, bool conjFopt2, bool conjFPC) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < sizeIn) {

        // Prefetching input data into registers
        cufftComplex fopt1_val = inFopt1[tid];
        cufftComplex fopt2_val = inFopt2[tid];
        cufftComplex IGMs_val = inIGMs[tid];

        // Perform complex multiplication using local variables
        cufftComplex ref1 = ComplexMult<cufftComplex>(fopt1_val, fopt2_val, conjFopt1, conjFopt2);

        // Computing angle and trigonometric functions once and storing them in registers

        float angle = AngleComplex(ref1);
        if (conjFPC) {
            angle = -angle;
        }
        float cosine = cosf(angle);
        float sine = sinf(angle);

        // Perform the remaining computations using the cached values
        float p1 = IGMs_val.x * cosine;
        float p2 = (IGMs_val.x + IGMs_val.y) * (cosine + sine);
        float p3 = IGMs_val.y * sine;

        // Write the outputs in a coalesced manner
        outref1[tid] = ref1; 
        outIGMs[tid].x = p1 - p3;
        outIGMs[tid].y = p2 - p3 - p1;
    }
}

// This is the C wrapper function that calls the CUDA kernel
extern "C" cudaError_t FastPhaseCorrection_GPU(cufftComplex * outIGMs, cufftComplex * outref1, cufftComplex * inIGMs, cufftComplex * inFopt1, cufftComplex * inFopt2, int sizeIn,
                                               bool conjFopt1, bool conjFopt2, bool conjFPC, int threads, int blocks, cudaStream_t streamId, cudaError_t cudaStatus) {

    FastPhaseCorrection_GPUkernel << <blocks, threads, 0, streamId >> > (outIGMs, outref1, inIGMs, inFopt1, inFopt2, sizeIn, conjFopt1, conjFopt2, conjFPC); // Include ref1 in the call

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();

    return cudaStatus;
}
