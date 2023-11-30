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

#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )
#define	MEMORY_ALIGNMENT 4096
#define MASK_LENGTH 32											// This is the length of the filters ?


#define filenameFiltR  "FilterCOSR.bin"						// This should be provided by DCSprocessing object
#define filenameFiltI  "FilterCOSI.bin"						// use a different file for each filter


typedef float2 Complex;



// thread handle class

class ThreadHandler
{
private:
	GaGeCard_interface			*	acquisitionCardPtr;
	CUDA_GPU_interface			*	gpuCardPtr;
	AcquisitionThreadFlowControl*   threadControlPtr;								// Everything needed for thread sync (flags, message queues and mutex)
	// Using  a pointer to avoid copy constructor issues;


// Local copies of variables, used to avoid constantly reading shared variables
// We could / should update that to only used the <commonly> used variables and abstract away the Gage card


	CSSYSTEMINFO					CsSysInfo = { 0 };		// Information on the selected acq system
	CSSTMCONFIG						StreamConfig = { 0 };		// stream configuration info
	CSACQUISITIONCONFIG				CsAcqCfg = { 0 };		// Config of the acq system
	GPUCONFIG						GpuCfg = { 0 };		// GPU config

	// local copies of commonly used shared variables...

	//uint32_t						NActiveChannel					= 0;

	// internal variables

	bool							acquisitionCompleteWithSuccess = false;			// Set if all requested acquisition is done

	// this is currently not implemented

	double							CounterFrequency = { 0 };		// Counter frequency, in counts per msecs
	LARGE_INTEGER					CounterStart = { 0 };
	LARGE_INTEGER					CounterStop = { 0 };
	
	TCHAR							szSaveFileNameI[MAX_PATH] = { 0 };		// Name of file that saves (in) raw data received from card
	TCHAR							szSaveFileNameO[MAX_PATH] = { 0 };		// Name of file that saves (out) data after processing

	HANDLE							fileHandle_rawData_in = { 0 };		// Handle to file where raw data is saved					previously hfile
	HANDLE							fileHandle_processedData_out = { 0 };		// Handle to file where processed data is saved				previously hfile o

	long long int					CardTotalData = 0;
	double							diff_time = 0;
	uint32_t						u32TransferSizeSamples = 0;			// number of samples transferred by the acq card in each queued buffer

	// CPU buffers

	// Actual allocated buffers
	void* pBuffer1 = NULL;			// Pointer to stream buffer1
	void* pBuffer2 = NULL;			// Pointer to stream buffer2
	void* h_buffer1 = NULL;			// Pointer to aligned stream buffer1 on cpu
	void* h_buffer2 = NULL;			// Pointer to aligned stream buffer2 on cpu

	Complex* filterCoeffs_ptr = NULL;			 // pointer to filter coefficients							previously h_mask;


	// Swapping buffers for double buffering
	void* pCurrentBuffer = NULL;			// Pointer to buffer where we will schedule data transferts;
	void* pWorkBuffer = NULL;			// Pointer to previous buffer that we can work on;

	void* hCurrentBuffer = NULL;			// Pointer to buffer where we will schedule data transferts;
	void* hWorkBuffer = NULL;			// Pointer to previous buffer that we can work on;




	// GPU buffers

	void* rawData_inGPU_ptr = NULL;				// Buffer for input data	this is a short int*		previously d_buffer
	Complex* filteredSignals_ptr = NULL;				// Filtered signals,  interleaved by channel			previously d_output
	Complex* ref1_ptr = NULL;				// Phase reference										previously ref1
	Complex* IGMsPC_ptr = NULL;				// Phase corrected IGMs									previously 

	short int* convo_tempBuffer1_ptr = NULL;				// Short buffer to handle the convolution transcient	previously d_Maskbuffer1
	short int* convo_tempBuffer2_ptr = NULL;				// Short buffer to handle the convolution transcient	previously d_Maskbuffer1

	// CUDA stuff

	cudaStream_t					cuda_stream = 0;			// Cuda stream


	void		UpdateLocalVariablesFromShared_noLock();	// This one is private so that an unsuspecting user does not update without locking
	void		PushLocalVariablesToShared_nolock();

public:														// Constructor
	ThreadHandler(GaGeCard_interface& acq, CUDA_GPU_interface& gpu, AcquisitionThreadFlowControl& flow);
	~ThreadHandler();							// Destructor

	void		UpdateLocalVariablesFromShared_lock();		// Update local vars under mutex lock to get most recent global vars settings
	void		PushLocalVariablesToShared_lock();

	void		ReadandAllocaterFilterCoefficients();		// Take filter coeffs from file and put them in CPU memory
	bool		readBinaryFileC(const char* filename1, const char* filename2, Complex* data, size_t numElements);		// utility function to read bin file to complex data

	void		AllocateAcquisitionCardStreamingBuffers();	// Allocate buffers where the card will stream data, in a double buffer manner
	uint32_t	GetSectorSize();							// Get drive sector size to properly adjust buffer sizes for DMA transfert
	void		AdjustBufferSizeForDMA();

	void		sleepUntilDMAcomplete();					// wait (and apparently sleeps the thread) until current DMA completes

	cudaError_t	RegisterAlignedCPUBuffersWithCuda();		// Computed aligned buffers and register them with cuda
	void		CreateCudaStream();

	void		AllocateGPUBuffers();						// Allocate all CUDA buffers not the cleanest code as it needs to be changed each time we need a new buffer 
	void		AllocateCudaManagedBuffer(void** buffer, uint32_t size);	// Allocate one managed buffer, and zero it out

	void		copyDataToGPU_async();

	void		ProcessInGPU(int32_t u32LoopCount);

	void		setReadyToProcess(bool value);				// Sets the atomic flag to tell the thread is ready to process or not

	void		setCurrentBuffers(bool choice);				// Decides which are the current buffers for the double buffering approach.
	void		setWorkBuffers();							// Work buffers are the current buffers of previous iteration

	void		ScheduleCardTransferWithCurrentBuffer();	// Tells card to transfert data to current buffer
	bool		isAcqComplete();

	void		CreateOuputFiles();
	void		WriteRawDataToFile();						// Saves raw data to file, if this is requested
	void		WriteProcessedDataToFile();					// Saves processed data to file, if this is requested


	void		SetupCounter();
	void		StartCounter();
	void		StopCounter();

	

};