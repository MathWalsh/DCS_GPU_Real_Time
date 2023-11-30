// ThreadHandler.h
// 
// Classs definition for object that orchestrates the processing thread
// 
// 
// Mathieu Walsh 
// Jerome Genest
// November 2023



#pragma once


#include "AcquisitionProcessingThread.h"
#include "GaGeCard_Interface.h"
#include "CUDA_GPU_Interface.h"
#include <windows.h>
#include <ctime>
#include <cufft.h> // Must add cufft.lib to linker
#include <cusolverDn.h> // Add cusolver.lib cublas.lib cublasLt.lib cusparse.lib to linker input

#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )
#define	MEMORY_ALIGNMENT 4096
#define MASK_LENGTH 32											// This is the length of the filters ?
#define _CRT_SECURE_NO_WARNINGS

typedef float2 Complex;



// thread handle class

class ThreadHandler
{
private:
	GaGeCard_interface			*	acquisitionCardPtr;
	CUDA_GPU_interface			*	gpuCardPtr;
	AcquisitionThreadFlowControl*   threadControlPtr;	
	DCSProcessingHandler		*	DcsProcessingPtr;
	// Everything needed for thread sync (flags, message queues and mutex)
	// Using  a pointer to avoid copy constructor issues;


// Local copies of variables, used to avoid constantly reading shared variables
// We could / should update that to only used the <commonly> used variables and abstract away the Gage card


	CSSYSTEMINFO					CsSysInfo = { 0 };		// Information on the selected acq system
	CSSTMCONFIG						StreamConfig = { 0 };		// stream configuration info
	CSACQUISITIONCONFIG				CsAcqCfg = { 0 };		// Config of the acq system
	GPUCONFIG						GpuCfg = { 0 };		// GPU config
	DCSCONFIG						DcsCfg = { 0 };		// DCS config

	// local copies of commonly used shared variables...

	//uint32_t						NActiveChannel					= 0;

	// internal variables

	bool							acquisitionCompleteWithSuccess = false;			// Set if all requested acquisition is done

	// this is currently not implemented

	double							CounterFrequency = { 0 };		// Counter frequency, in counts per msecs
	LARGE_INTEGER					CounterStart = { 0 };
	LARGE_INTEGER					CounterStop = { 0 };
	
	wchar_t							szSaveFileNameI[MAX_PATH] = { 0 }; 	// Name of file that saves (in) raw data received from card
	char							szSaveFileNameO[MAX_PATH] = { 0 }; 	// Name of file that saves (out) data after processing

	HANDLE							fileHandle_rawData_in = { 0 };		// Handle to file where raw data is saved					previously hfile
	HANDLE							fileHandle_processedData_out = { 0 };		// Handle to file where processed data is saved				previously hfile o

	long long int					CardTotalData = 0;
	double							diff_time = 0;
	uint32_t						u32TransferSizeSamples = 0;			// number of samples transferred by the acq card in each queued buffer
	uint32_t						SegmentSizePerChannel;

	// CPU buffers

	// Actual allocated buffers
	void* pBuffer1 = NULL;			// Pointer to stream buffer1
	void* pBuffer2 = NULL;			// Pointer to stream buffer2
	void* h_buffer1 = NULL;			// Pointer to aligned stream buffer1 on cpu
	void* h_buffer2 = NULL;			// Pointer to aligned stream buffer2 on cpu

	// Swapping buffers for double buffering
	void* pCurrentBuffer = NULL;			// Pointer to buffer where we will schedule data transferts;
	void* pWorkBuffer = NULL;			// Pointer to previous buffer that we can work on;

	void* hCurrentBuffer = NULL;			// Pointer to buffer where we will schedule data transferts;
	void* hWorkBuffer = NULL;			// Pointer to previous buffer that we can work on;

	// General GPU variables
	int* NIGMs_ptr = new int[2];					// Number of IGMs in the current and previous segments
	int* currentSegmentSize_ptr = new int[3];		// 0, buffer segment size, 1 Find_IGMs_ZPD segment size, 2 Self-correction segment size
	int LastIdxLastIGM = NULL;						// Used to calculate the number of points for self-correction
	double* previousptsPerIGM = new double[1];				// Used to loop, should not be needed
	cufftComplex* IGMsOut1 = NULL;
	cufftComplex* IGMsOut2 = NULL;
	int NptsSave = NULL;
	// Raw data buffers
	short* rawData_inCPU_ptr = NULL;				// Buffer for input data cudaMallocManager	this is a short int*		previously d_buffer
	short* rawData_inGPU_ptr = NULL;				// Buffer for input data on the device, current (cudaMalloc)
	short* rawData_inGPU1_ptr = NULL;				// Buffer for input data on the device, buffer 1 (cudaMalloc)
	short* rawData_inGPU2_ptr = NULL;				// Buffer for input data on the device, buffer 2 (cudaMalloc)

	// Filtering
	cufftComplex* filterCoeffs_ptr = NULL;			// pointer to filter coefficients	
	cufftComplex* filteredSignals_ptr = NULL;		// Filtered signals,  interleaved by channel
	float* Convolution_Buffer1_CPU_ptr = NULL;		// For the first segment	
	float* Convolution_Buffer1_ptr = NULL;			// Short buffer to handle the convolution transcient	
	float* Convolution_Buffer2_ptr = NULL;			// Short buffer to handle the convolution transcient
	int* idxchfilt_ptr = NULL;						// Chooses which channel to filter (used because we have more signal than channels)
	float* currentConvolutionBufferIn_ptr = NULL;		// This chooses the input buffer for the convolution 
	float* currentConvolutionBufferOut_ptr = NULL;		// This chooses the output buffer for the convolution 


	// Fast phase Correction 
	cufftComplex* ref1_ptr = NULL;					// Phase reference										
	cufftComplex* IGMsPC_ptr = NULL;				// Phase corrected IGMs		

	// Unwrapping		
	double* unwrapped_phase_ptr = NULL;				// For unwrapping a phase signal
	int* two_pi_cumsum_ptr = NULL;					// Cumsum pointer for unwrap
	int* blocks_edges_cumsum_ptr = NULL;			// For the unwrapping kernel
	int* increment_blocks_edges_ptr = NULL;			// For the unwrapping kernel
	const int warp_size = 32;						// For the unwrapping kernel, Should be 32 for all the Nvidia architectures up to at least RTX 4090
	bool Unwrapdfr = NULL;							// This is if we want to unwrap something different thant dfr ref (logic not implemented yet)
	bool EstimateSlope = NULL;

	// 2 ref resampling 
	cufftComplex* IGMsPC_resampled_ptr = NULL;		// Phase corrected and resampled with 2 ref IGMs
	float* ref_dfr_angle_ptr = NULL;				// dfr angle of the 2 ref
	double* start_slope_ptr = NULL;					// Used in linspace kernel for 2 ref resampling and self-correction
	double* h_start_slope_ptr = new double[2];
	double* end_slope_ptr = NULL;					// Used in linspace kernel for 2 ref resampling and self-correction
	double* h_end_slope_ptr = new double[2];
	double* LinearGrid_dfr_ptr = NULL;				// Used in linspace kernel for 2 ref resampling and self-correction		
	int* idx_LinearGrid_dfr_ptr = NULL;				// Used in linear interpolation kernel for 2 ref resampling and self-correction

	// Find_IGMs_ZPD_GPU
	cufftComplex* IGMsSelfCorrectionIn_ptr = NULL;	// Used to find ZPDs and do self correction
	cufftComplex* IGMTemplate_ptr = NULL;			// Template IGM for xcorr
	cufftComplex* xcorrBLocksIGMs_ptr = NULL;		// xcorr results for each block in the segment and for the total result
	cufftComplex* LastIGMBuffer_ptr = NULL;			// To keep the discarded data points at the end of the segment for the next segment	
	double* idxMaxSubpoint_ptr = NULL;				// subpoint ZPD positions of each IGM in the segment
	double* phaseMaxSubpoint_ptr = NULL;			// subpoint phase of ZPD  of each IGM in the segment
	double* ptsPerIGMSegment_ptr;					// Subpoint average number of points per IGM in the segment		
	int* idxMidSegments_ptr = NULL;					// ZPD positions of each IGM in the segment, used for the global index of the maximum
	int* NptsLastIGMBuffer_ptr = new int[2];		// Number of points in the LastIGMBuffer for current and previous segment
	int* idxStartFirstZPD_ptr = new int[2];			// Start of first IGM for current and previous segment (used in xcorr, this allows us to calulate the approximate position of the other IGMs)
	int blocksPerDelay = NULL;						// Number of blocks for each delay in the xcorr (depends on the template size and the number of delay we want to calculate)
	int totalDelays = NULL;							// Number of IGMs * number of delays per IGM
	int totalBlocks = NULL;							// Number of blocks for each delay * total number of delays

	// Find_First_ZPD_GPU
	int* idxMaxBLocks_ptr = NULL;					// Index of the maximum in a block to find the maximum of the second IGM
	float* MaxValBlocks_ptr = NULL;					// Value of the maximum in a block to find the maximum of the second IGM

	// For Compute_SelfCorrection_GPU
	cufftComplex* IGMsSelfCorrection_ptr = NULL;    // IGMs for self-correction padded with LastIGMBuffer_ptr at the start and last IGM removed or cropped at the end
	double* spline_coefficients_dfr_ptr = NULL;		// Spline coefficients calculated with cuSolver for the non uniform dfr spline grid
	double* spline_coefficients_f0_ptr = NULL;		// Spline coefficients calculated with cuSolver for the ZPD f0 spline grid
	double* splineGrid_dfr_ptr = NULL;				// non uniform dfr spline grid for the dfr resampling in the self-correction 
	float* splineGrid_f0_ptr = NULL;				// ZPD f0 spline grid for the phase correction in the self-correction 	

	// Variables for cuSOlver to compute spline coefficients in Compute_SelfCorrection_GPU	
	cusolverDnHandle_t	cuSolver_handle;			// Handle for cuSolve
	double* d_h = NULL;
	double* d_D = NULL;
	double* d_work = NULL;
	int* devInfo = NULL;
	int lwork = NULL;

	// For Compute_MeanIGM_GPU
	cufftComplex* IGM_mean_ptr = NULL;

	// CUDA variables
	cudaStream_t cuda_stream = 0;					// Cuda stream
	cudaStream_t cuda_stream1 = 0;					// Cuda stream
	cudaError_t	cudaStatus;							// To check kernel launch errors


	void		UpdateLocalVariablesFromShared_noLock();	// This one is private so that an unsuspecting user does not update without locking
	void		PushLocalVariablesToShared_nolock();

public:														// Constructor
	ThreadHandler(GaGeCard_interface& acq, CUDA_GPU_interface& gpu, AcquisitionThreadFlowControl& flow, DCSProcessingHandler& dcs);
	~ThreadHandler();							// Destructor

	void		UpdateLocalVariablesFromShared_lock();		// Update local vars under mutex lock to get most recent global vars settings
	void		PushLocalVariablesToShared_lock();

	void		ReadandAllocateFilterCoefficients();		// Take filter coeffs from file and put them in CPU memory
	void		ReadandAllocateTemplateData();				// Take template data from file and put them in CPU memory

	bool		readBinaryFileC(const char* filename, cufftComplex* data, size_t numElements);		// utility function to read bin file to complex data

	void		AllocateAcquisitionCardStreamingBuffers();	// Allocate buffers where the card will stream data, in a double buffer manner
	uint32_t	GetSectorSize();							// Get drive sector size to properly adjust buffer sizes for DMA transfert
	void		AdjustBufferSizeForDMA();

	void		sleepUntilDMAcomplete();					// wait (and apparently sleeps the thread) until current DMA completes

	cudaError_t	RegisterAlignedCPUBuffersWithCuda();		// Computed aligned buffers and register them with cuda
	
	void		CreateCudaStream();

	void		CreatecuSolverHandle();						

	void		AllocateGPUBuffers();						// Allocate all CUDA buffers not the cleanest code as it needs to be changed each time we need a new buffer 
	void		AllocateCudaManagedBuffer(void** buffer, uint32_t size);	// Allocate one managed buffer, and zero it out

	void		copyDataToGPU_async(int32_t u32LoopCount);

	void		ProcessInGPU(int32_t u32LoopCount);

	void		setReadyToProcess(bool value);				// Sets the atomic flag to tell the thread is ready to process or not

	void		setCurrentBuffers(bool choice);				// Decides which are the current buffers for the double buffering approach.
	void		setWorkBuffers();							// Work buffers are the current buffers of previous iteration

	void		ScheduleCardTransferWithCurrentBuffer();	// Tells card to transfert data to current buffer
	bool		isAcqComplete();

	void		CreateOuputFiles();
	void		WriteRawDataToFile();						// Saves raw data to file, if this is requested
	void		WriteProcessedDataToFile(int32_t u32LoopCount);					// Saves processed data to file, if this is requested


	void		SetupCounter();
	void		StartCounter();
	void		StopCounter();

	

};