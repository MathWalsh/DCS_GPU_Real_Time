// ThreadHandler.cpp
// 
// Class function members
//  for object that orchestrates the processing thread
// 
// Mathieu Walsh 
// Jerome Genest
// November 2023



#include <iostream>

#include "ThreadHandler.h"

#include "Convolution_complex_GPU.h"
#include "Multiplication_complex_GPU.h"
#include "Fast_phase_correction_GPU.h"

bool ThreadHandler::isAcqComplete()
{
	return acquisitionCompleteWithSuccess;
}


void ThreadHandler::ProcessInGPU(int32_t u32LoopCount)
{

	// Use correct mask buffers depending on the u32LoopCount
	short* currentMask1 = (u32LoopCount % 2 == 0) ? convo_tempBuffer1_ptr : convo_tempBuffer2_ptr;
	short* currentMask2 = (u32LoopCount % 2 == 0) ? convo_tempBuffer2_ptr : convo_tempBuffer1_ptr;

	// Filter the channels with h_mask coefficients
	cudaError_t cudaStatus = Convolution_complex_GPU(filteredSignals_ptr, rawData_inGPU_ptr, currentMask1, currentMask2, u32TransferSizeSamples,
		GpuCfg.i32GpuBlocks, GpuCfg.i32GpuThreads, u32LoopCount, filterCoeffs_ptr, StreamConfig.NActiveChannel, cuda_stream);

	if (cudaStatus != cudaSuccess)
		ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Convolution_complex_GPU launch failed", ERROR_);


	// Remove the CW contribution with a time multiplication of the two references channels
	//cudaStatus = Multiplication_complex_GPU(ref1_ptr, filteredSignals_ptr + in1Index * SegmentSizePerChannel,
	//	filteredSignals_ptr + in2Index * SegmentSizePerChannel, conj1, conj2, SegmentSizePerChannel,
	//	g_GpuConfig.i32GpuBlocks, g_GpuConfig.i32GpuThreads, stream1);

	//if (cudaStatus != cudaSuccess)	
	//	ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Multiplication_complex_GPU launch failed", ERROR_);


	//// Remove the phase noise on the IGMs
	//cudaStatus = Fast_phase_correction_GPU(IGMsPC, d_output + inIGMsIndex * SegmentSizePerChannel, ref1, conjIGMs, SegmentSizePerChannel, g_GpuConfig.i32GpuBlocks, g_GpuConfig.i32GpuThreads, stream1);

	//if (cudaStatus != cudaSuccess)	
	//	ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Fast_phase_correction_GPU launch failed", ERROR_);


	cudaDeviceSynchronize();// maybe not here if we do triple buffering

}


void ThreadHandler::copyDataToGPU_async()
{
	// Asynchronously copy data from hWorkBuffer to rawData_inGPU_ptr using stream1

	if (hWorkBuffer)		// we copy only if we have a work buffer
		cudaMemcpyAsync(rawData_inGPU_ptr, hWorkBuffer, StreamConfig.u32BufferSizeBytes, cudaMemcpyHostToDevice, cuda_stream);

}

void ThreadHandler::sleepUntilDMAcomplete()
{
	uInt32 u32ErrorFlag = 0;
	uInt32 u32ActualLength = 0;
	uInt32 u8EndOfData = 0;

	if (!acquisitionCompleteWithSuccess)
		acquisitionCardPtr->waitForCurrentDMA(u32ErrorFlag, u32ActualLength, u8EndOfData);

	acquisitionCompleteWithSuccess = (0 != u8EndOfData);

	
	//gpuCard.setTotalData(gpuCard.getTotalData() + u32ActualLength);
	
	CardTotalData += u32ActualLength;

	//std::cout << CardTotalData << " \n";

	//if (CardTotalData / CsSysInfo.u32SampleSize >= StreamConfig.NptsTot) 
	//	acquisitionCompleteWithSuccess = true;

}

void ThreadHandler::ScheduleCardTransferWithCurrentBuffer()
{

	// Should we do this under a lock guard, as we are accessing the shared acquisition card handle ?
	// Is the CsStmTransferToBuffer doing funny things in shared memory ?

	int32_t i32Status = acquisitionCardPtr->queueToTransferBuffer(pCurrentBuffer, u32TransferSizeSamples);

	//std::cout << "Buffer : " << pCurrentBuffer << "  size : " << u32TransferSizeSamples << "\n";

	if (CS_FAILED(i32Status))
	{
		if (CS_STM_COMPLETED == i32Status)
		{
			std::cout << "Acq complete\n";
			acquisitionCompleteWithSuccess = true;
		}
		else
		{
			ErrorHandler("There was an error queing buffer to card ", i32Status);
		}
	}
}



// Writes raw data to disk if requested
// Note that this operate on the Work (previous) buffer

void ThreadHandler::WriteRawDataToFile()
{
	if (StreamConfig.bSaveToFile && NULL != pWorkBuffer)
	{
		DWORD				dwBytesSave = 0;
		BOOL				bWriteSuccess = TRUE;

		// While data transfer of the current buffer is in progress, save the data from pWorkBuffer to hard disk
		bWriteSuccess = WriteFile(fileHandle_rawData_in, pWorkBuffer, StreamConfig.u32BufferSizeBytes, &dwBytesSave, NULL);
		if (!bWriteSuccess || dwBytesSave != StreamConfig.u32BufferSizeBytes)
			ErrorHandler(GetLastError(), "WriteFile() error on card (raw)", ERROR_);
	}
}

void ThreadHandler::WriteProcessedDataToFile()
{
	if (StreamConfig.bSaveToFile)
	{

		DWORD				dwBytesSave = 0;
		BOOL				bWriteSuccess = TRUE;

		bWriteSuccess = WriteFile(fileHandle_processedData_out, filteredSignals_ptr, u32TransferSizeSamples * sizeof(Complex), &dwBytesSave, NULL);
		//bWriteSuccess = WriteFile(hFileO, ref1_ptr, SegmentSizePerChannel * sizeof(Complex), &dwBytesSave, NULL);
		//bWriteSuccess = WriteFile(fileHandle_processedData_out, IGMsPC_ptr, SegmentSizePerChannel * sizeof(Complex), &dwBytesSave, NULL);

		if (!bWriteSuccess || dwBytesSave != u32TransferSizeSamples * sizeof(Complex))
			//if (!bWriteSuccess || dwBytesSave != SegmentSizePerChannel * sizeof(Complex))
			ErrorHandler(GetLastError(), "WriteFile() error on card (raw)", ERROR_);
	}
}


void ThreadHandler::setCurrentBuffers(bool choice)
{
	if (choice)
	{
		pCurrentBuffer = pBuffer2;
		if (GpuCfg.bUseGpu)
		{
			hCurrentBuffer = h_buffer2;
		}
	}
	else
	{
		pCurrentBuffer = pBuffer1;
		if (GpuCfg.bUseGpu)
		{
			hCurrentBuffer = h_buffer1;
		}
	}
}


void ThreadHandler::setWorkBuffers()
{
	// Current buffers will be work buffer for next pass in loop
	pWorkBuffer = pCurrentBuffer;
	hWorkBuffer = hCurrentBuffer;

}




// Handler class member function definitions


/***************************************************************************************************
****************************************************************************************************/
// Constructor


ThreadHandler::ThreadHandler(GaGeCard_interface& acq, CUDA_GPU_interface& gpu, AcquisitionThreadFlowControl& flow) // Constructor

{

	// Locking mutex while we access shared variables to configure object
	// and create local copies of variables what we will read often
	// this means changes to shared variables will not affect the procesing thread until with re-sync the local variables.

	const std::lock_guard<std::shared_mutex> lock(flow.sharedMutex);	// Lock gard unlonck when destroyed (i.e at the end of the constructor)
	// Could use a shared lock since we only read variables
	// but playing safe here, no acquisition runninng in init phase anyway
	acquisitionCardPtr = &acq;
	gpuCardPtr = &gpu;

	threadControlPtr = &flow;


	//Making local copies of variables

	UpdateLocalVariablesFromShared_noLock();

	SetupCounter();

}

/***************************************************************************************************
****************************************************************************************************/
// Destructor

ThreadHandler::~ThreadHandler() // Destructor
{
	if (fileHandle_rawData_in)					// Close files if they are open, the original code was also deleting, we will not do this
		CloseHandle(fileHandle_rawData_in);
	if (fileHandle_processedData_out)
		CloseHandle(fileHandle_processedData_out);


	// under mutex lock
	{
		const std::lock_guard<std::shared_mutex> lock(threadControlPtr->sharedMutex);  // Lock guard unlocks when destroyed

		gpuCardPtr->setTotalData(CardTotalData);
		gpuCardPtr->setDiffTime(diff_time);

		acquisitionCardPtr->FreeStreamBuffer(pBuffer1);
		acquisitionCardPtr->FreeStreamBuffer(pBuffer2);


	}

	if (h_buffer1)
		cudaHostUnregister(h_buffer1);

	if (h_buffer1)
		cudaHostUnregister(h_buffer2);

	if (filterCoeffs_ptr)
		free(filterCoeffs_ptr);

	//  reset cuda here...
	cudaDeviceReset();  // All cuda mallocs are done in the thread  if the thread is kept alive when not acquiring, maybe we should do the reset un GPU object ?

}

// Updates our local copy of the variables by looking at the shared variables
// must be performed under mutex lock

void ThreadHandler::UpdateLocalVariablesFromShared_lock()
{
	const std::lock_guard<std::shared_mutex> lock(threadControlPtr->sharedMutex);  // Lock guard unlocks when destroyed

	//Updating local copies of variables

	UpdateLocalVariablesFromShared_noLock();

}


/***************************************************************************************************
****************************************************************************************************/
// NO LOCK, just for code re-use !!
// NEVER CALL without locking before

void ThreadHandler::UpdateLocalVariablesFromShared_noLock()
{

	//Updating local copies of variables

	CsSysInfo = acquisitionCardPtr->getSystemInfo();
	StreamConfig = acquisitionCardPtr->getStreamConfig();
	CsAcqCfg = acquisitionCardPtr->getAcquisitionConfig();
	GpuCfg = gpuCardPtr->getConfig();

	//csHandle = acquisitionCard->GetSystemHandle();

	StreamConfig.NActiveChannel = CsAcqCfg.u32Mode & CS_MASKED_MODE;

	sprintf_s(szSaveFileNameI, sizeof(szSaveFileNameI), "%s_I%d.dat", StreamConfig.strResultFile, 1);
	sprintf_s(szSaveFileNameO, sizeof(szSaveFileNameO), "%s_O%d.dat", StreamConfig.strResultFile, 1);

	// Consider keeping abstracted copies of often used vars...
	//NActiveChannel = StreamConfig.NActiveChannel;  // for example

	// ultimately, we could get rid of the card specific structs

}

/***************************************************************************************************
****************************************************************************************************/
// NO LOCK, just for code re-use !!
// NEVER CALL without locking before

// any modification to the config variables made during acquisitio is pushed to the shared object

void ThreadHandler::PushLocalVariablesToShared_nolock()
{
	acquisitionCardPtr->setSystemInfo(CsSysInfo);
	acquisitionCardPtr->setStreamComfig(StreamConfig);
	acquisitionCardPtr->setAcquisitionConfig(CsAcqCfg);

	gpuCardPtr->setConfig(GpuCfg);
}

void ThreadHandler::PushLocalVariablesToShared_lock()
{
const std::lock_guard<std::shared_mutex> lock(threadControlPtr->sharedMutex);  // Lock guard unlocks when destroyed

PushLocalVariablesToShared_nolock();

gpuCardPtr->setTotalData(CardTotalData);
}



void ThreadHandler::ReadandAllocaterFilterCoefficients()
{
	filterCoeffs_ptr = (Complex*)malloc(MASK_LENGTH * sizeof(double));

	readBinaryFileC(filenameFiltR, filenameFiltI, filterCoeffs_ptr, MASK_LENGTH);
}

bool ThreadHandler::readBinaryFileC(const char* filename1, const char* filename2, Complex* data, size_t numElements)
{
	// Open the binary file in binary mode for filename1
	FILE* file1 = fopen(filename1, "rb");
	if (!file1) {
		fprintf(stderr, "Unable to open the file: %s\n", filename1);
		return false;
	}

	// Read the data into the provided data pointer for x values
	for (size_t i = 0; i < numElements; ++i) {
		if (fread(&data[i].x, sizeof(float), 1, file1) != 1) {
			fprintf(stderr, "Error reading data from the file: %s\n", filename1);
			fclose(file1);
			return false;
		}
	}

	// Close the file when done
	fclose(file1);

	// Open the binary file in binary mode for filename2
	FILE* file2 = fopen(filename2, "rb");
	if (!file2) {
		fprintf(stderr, "Unable to open the file: %s\n", filename2);
		return false;
	}

	// Read the data into the provided data pointer for y values
	for (size_t i = 0; i < numElements; ++i) {
		if (fread(&data[i].y, sizeof(float), 1, file2) != 1) {
			fprintf(stderr, "Error reading data from the file: %s\n", filename2);
			fclose(file2);
			return false;
		}
	}

	// Close the file when done
	fclose(file2);

	return true;
}


void  ThreadHandler::AllocateAcquisitionCardStreamingBuffers()
{
	AdjustBufferSizeForDMA();

	const std::lock_guard<std::shared_mutex> lock(threadControlPtr->sharedMutex);		// lock to access shared acq card
	// lock guard unlocks at end of function

	acquisitionCardPtr->AllocateStreamingBuffer(1, StreamConfig.u32BufferSizeBytes, &pBuffer1);
	acquisitionCardPtr->AllocateStreamingBuffer(1, StreamConfig.u32BufferSizeBytes, &pBuffer2);

	// Convert the transfer size to BYTEs or WORDs depending on the card.
	u32TransferSizeSamples = StreamConfig.u32BufferSizeBytes / CsSysInfo.u32SampleSize;	// Number of samples for each of the double buffers
}


cudaError_t ThreadHandler::RegisterAlignedCPUBuffersWithCuda()
{
	if (GpuCfg.bUseGpu)
	{
		cudaError_t  cudaStatus = (cudaError_t)0;

		h_buffer1 = (unsigned char*)ALIGN_UP(pBuffer1, MEMORY_ALIGNMENT);

		cudaStatus = cudaHostRegister(h_buffer1, (size_t)StreamConfig.u32BufferSizeBytes, cudaHostRegisterMapped);
		if (cudaStatus != cudaSuccess)
		{
			ErrorHandler(cudaStatus, "cudaHostRegister failed! \n", ERROR_);
			return cudaStatus;
		}
		h_buffer2 = (unsigned char*)ALIGN_UP(pBuffer2, MEMORY_ALIGNMENT);
		cudaStatus = cudaHostRegister(h_buffer2, (size_t)StreamConfig.u32BufferSizeBytes, cudaHostRegisterMapped);
		if (cudaStatus != cudaSuccess)
		{
			ErrorHandler(cudaStatus, "cudaHostRegister failed! \n", ERROR_);
			return cudaStatus;
		}

	}
	else
	{
		ErrorHandler(-1, "Not using GPU, exiting! \n", ERROR_);
		return (cudaError_t)-1;
	}
}


void ThreadHandler::CreateCudaStream()
{
	cudaStreamCreate(&cuda_stream);
}


void ThreadHandler::AllocateGPUBuffers()
{

	// Determine size of each channel's output segment
	int SegmentSizePerChannel = u32TransferSizeSamples / StreamConfig.NActiveChannel;
	int BytesizePerChannel = SegmentSizePerChannel * CsSysInfo.u32SampleSize;

	cudaMalloc(&rawData_inGPU_ptr, StreamConfig.u32BufferSizeBytes);

	// Allocate and zero out
	AllocateCudaManagedBuffer((void**)&filteredSignals_ptr, u32TransferSizeSamples * sizeof(Complex));
	AllocateCudaManagedBuffer((void**)&ref1_ptr, SegmentSizePerChannel * sizeof(Complex));
	AllocateCudaManagedBuffer((void**)&IGMsPC_ptr, SegmentSizePerChannel * sizeof(Complex));

	AllocateCudaManagedBuffer((void**)&convo_tempBuffer1_ptr, StreamConfig.NActiveChannel * (MASK_LENGTH - 1) * sizeof(short int*));
	AllocateCudaManagedBuffer((void**)&convo_tempBuffer2_ptr, StreamConfig.NActiveChannel * (MASK_LENGTH - 1) * sizeof(short int*));


	// Align 
	filteredSignals_ptr = (Complex*)ALIGN_UP(filteredSignals_ptr, MEMORY_ALIGNMENT);
	ref1_ptr = (Complex*)ALIGN_UP(ref1_ptr, MEMORY_ALIGNMENT);
	IGMsPC_ptr = (Complex*)ALIGN_UP(IGMsPC_ptr, MEMORY_ALIGNMENT);

	// Should we align the convolution Transcient buffers ?


}


void ThreadHandler::AllocateCudaManagedBuffer(void** buffer, uint32_t size)
{

	cudaMallocManaged(buffer, size, cudaMemAttachGlobal);

	// Zero out buffer, regardless of type
	uint8_t* zeroOutPtr = 0;
	zeroOutPtr = (uint8_t*)*buffer;

	for (int i = 0; i < size / sizeof(uint8_t); ++i)
	{
		zeroOutPtr[i] = 0;
	}

}


/***************************************************************************************************
****************************************************************************************************/

void ThreadHandler::AdjustBufferSizeForDMA()
{
	uint32_t u32SectorSize = GetSectorSize();

	uint32_t	u32DmaBoundary = 16;

	if (StreamConfig.bFileFlagNoBuffering)
	{
		// If bFileFlagNoBuffering is set, the buffer size should be multiple of the sector size of the Hard Disk Drive.
		// Most of HDDs have the sector size equal 512 or 1024.
		// Round up the buffer size into the sector size boundary
		u32DmaBoundary = u32SectorSize;
	}

	// Round up the DMA buffer size to DMA boundary (required by the Streaming data transfer)
	if (StreamConfig.u32BufferSizeBytes % u32DmaBoundary)
		StreamConfig.u32BufferSizeBytes += (u32DmaBoundary - StreamConfig.u32BufferSizeBytes % u32DmaBoundary);

	std::cout << "Actual buffer size used for data streaming =  " << StreamConfig.u32BufferSizeBytes << "Bytes\n";
}



/***************************************************************************************************
****************************************************************************************************/

uint32_t ThreadHandler::GetSectorSize()
{
	uInt32 size = 0;
	if (!GetDiskFreeSpace(NULL, NULL, &size, NULL, NULL))
		return 0;
	return size;
}





// Setting the flag to inform than the processing thread is ready

void ThreadHandler::setReadyToProcess(bool value)
{
	threadControlPtr->ThreadReady = value;
}

void ThreadHandler::CreateOuputFiles()
{
	if (StreamConfig.bSaveToFile)
	{
		DWORD dwFileFlag = StreamConfig.bFileFlagNoBuffering ? FILE_FLAG_NO_BUFFERING : 0;


		fileHandle_rawData_in = CreateFile(szSaveFileNameI, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, dwFileFlag, NULL);
		fileHandle_processedData_out = CreateFile(szSaveFileNameO, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, dwFileFlag, NULL);


		if ((fileHandle_rawData_in == INVALID_HANDLE_VALUE) || (fileHandle_processedData_out == INVALID_HANDLE_VALUE))
		{
			ErrorHandler(-1, "Unable to create data files.\n", ERROR_);
		}


	}
}


void ThreadHandler::SetupCounter()
{
	LARGE_INTEGER temp;

	QueryPerformanceFrequency((LARGE_INTEGER*)&temp);
	CounterFrequency = ((double)temp.QuadPart) / 1000.0;
}

void ThreadHandler::StartCounter()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&CounterStart);
}

void ThreadHandler::StopCounter()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&CounterStop);

	double extraTime = ((double)CounterStop.QuadPart - (double)CounterStart.QuadPart) / CounterFrequency;
	
	// needs to be under lock guard 
	//gpuCardPtr->setDiffTime(gpuCard.getDiffTime() + extraTime);
	
	diff_time += ((double)CounterStop.QuadPart - (double)CounterStart.QuadPart) / CounterFrequency;
}

