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
#include "Convolution_complex32_optimized_GPU.h"
#include "Calculate_dfr_reference_GPU.h"
#include "FastPhaseCorrection_GPU.h"
#include "Unwrap_phase_GPU.h"
#include "Linspace_GPU.h"
#include "Linear_interpolation_GPU.h"
#include "Find_IGMs_ZPD_GPU.h"
#include "Compute_SelfCorrection_GPU.h"
#include "Compute_MeanIGM_GPU.h"
#include "Find_First_ZPD_GPU.h"

bool ThreadHandler::isAcqComplete()
{
	return acquisitionCompleteWithSuccess;
}


void ThreadHandler::ProcessInGPU(int32_t u32LoopCount)
{

	if (u32LoopCount == 0)
	{
		blocksPerDelay = (DcsCfg.TemplateSize + 2 * 256 - 1) / (2 * 256); // We put 256 because this is the number of threads per block in Find_IGMs_ZPD_GPU
		totalDelays = DcsCfg.TemplateSize; // We need to test this, we might need more
		totalBlocks = blocksPerDelay * totalDelays;
	}
	else if (u32LoopCount == 1){
		blocksPerDelay = (DcsCfg.TemplateSize - DcsCfg.MaxDelayXcorr + 2 * 256 - 1) / (2 * 256); // We put 256 because this is the number of threads per block in Find_IGMs_ZPD_GPU
		totalDelays = DcsCfg.MaxDelayXcorr * NIGMs_ptr[0];
		totalBlocks = blocksPerDelay * totalDelays;
	}

	rawData_inGPU_ptr = (u32LoopCount % 2 == 0) ? rawData_inGPU1_ptr : rawData_inGPU2_ptr;

	// Use correct convolution buffers depending on the u32LoopCount
	currentConvolutionBufferIn_ptr = (u32LoopCount % 2 == 0) ? Convolution_Buffer1_ptr : Convolution_Buffer2_ptr;
	currentConvolutionBufferOut_ptr = (u32LoopCount % 2 == 0) ? Convolution_Buffer2_ptr : Convolution_Buffer1_ptr;



	currentSegmentSize_ptr[0] = SegmentSizePerChannel; // We reset currentSegmentSize_ptr to  SegmentSizePerChannel
	GpuCfg.i32GpuBlocks128 = (currentSegmentSize_ptr[0] + GpuCfg.i32GpuThreads / 2 - 1) / (GpuCfg.i32GpuThreads / 2);
	GpuCfg.i32GpuBlocks256 = (currentSegmentSize_ptr[0] + GpuCfg.i32GpuThreads - 1) / GpuCfg.i32GpuThreads;
	if (u32LoopCount == 1) {

		NIGMs_ptr[0] = static_cast<int>(std::round((currentSegmentSize_ptr[0] - (idxStartFirstZPD_ptr[0] + (DcsCfg.TemplateSize - 1) / 2 + 1)) / previousptsPerIGM[0])); // This finds the number of IGMs remaining given the first ZPD found

		LastIdxLastIGM = idxStartFirstZPD_ptr[0] + (DcsCfg.TemplateSize - 1) / 2 + round(NIGMs_ptr[0] * previousptsPerIGM[0] + (previousptsPerIGM[0] + 1) / 2); // We calculate the position of the last point of the last ZPD


		if (LastIdxLastIGM < currentSegmentSize_ptr[0]) { // TO DO, do we ever it this condition??
			printf("\n WE HAVE A PROBLEM");

		}
		else { // The last ZPD is incomplete, so we need to crop it.
			LastIdxLastIGM = static_cast<int>(std::round(idxStartFirstZPD_ptr[0] + (DcsCfg.TemplateSize - 1) / 2 + (NIGMs_ptr[0] - 0.5) * previousptsPerIGM[0]) + 1);
			NptsLastIGMBuffer_ptr[0] = currentSegmentSize_ptr[0] - LastIdxLastIGM;
			cudaMemcpyAsync(LastIGMBuffer_ptr, IGMsPC_resampled_ptr + (currentSegmentSize_ptr[0] - NptsLastIGMBuffer_ptr[0]),
				NptsLastIGMBuffer_ptr[0] * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, cuda_stream); // can it be done on stream1 ??  transfer data in buffer for next segment
			idxStartFirstZPD_ptr[0] = std::floor(previousptsPerIGM[0] / 2) - (DcsCfg.TemplateSize - 1) / 2; // Removed -1 here why??

		}

	}
	else if (u32LoopCount > 1) {
		idxStartFirstZPD_ptr[0] = idxStartFirstZPD_ptr[1];
		NptsLastIGMBuffer_ptr[0] = NptsLastIGMBuffer_ptr[1]; // We put the previous batch number of points in the first index

	}

	if (u32LoopCount > 0)
	{

		// For Find_IGMs_ZPD_GPU
		currentSegmentSize_ptr[1] = currentSegmentSize_ptr[0] + NptsLastIGMBuffer_ptr[0];
		NIGMs_ptr[0] = static_cast<int>(std::round(currentSegmentSize_ptr[1] / previousptsPerIGM[0]));
		LastIdxLastIGM = round(idxStartFirstZPD_ptr[0] + (DcsCfg.TemplateSize - 1) / 2 + (NIGMs_ptr[0] - 0.5) * previousptsPerIGM[0]);
		if (LastIdxLastIGM < currentSegmentSize_ptr[1]) { // TO DO
			idxStartFirstZPD_ptr[1] = std::floor(previousptsPerIGM[0] / 2) - (DcsCfg.TemplateSize - 1) / 2; 	// idx of first ZPD in new segment - NptsTemplateSize, to be verified, removed -1

			NptsLastIGMBuffer_ptr[1] = currentSegmentSize_ptr[1] - (LastIdxLastIGM + 1); // Number of points in buffer for next segment
			currentSegmentSize_ptr[2] = currentSegmentSize_ptr[1] - NptsLastIGMBuffer_ptr[1]; // Npts for Compute_SelfCorrection
			NIGMs_ptr[1] = NIGMs_ptr[0];
			h_start_slope_ptr[0] = 0.0f;
			h_start_slope_ptr[1] = -0.5;
			cudaMemcpy(start_slope_ptr + 1, h_start_slope_ptr, 2 * sizeof(double), cudaMemcpyHostToDevice);
			h_end_slope_ptr[0] = 1.0f;
			h_end_slope_ptr[1] = NIGMs_ptr[1] - 0.5;
			cudaMemcpy(end_slope_ptr + 1, h_end_slope_ptr, 2 * sizeof(double), cudaMemcpyHostToDevice);

		}
		else { // We will remove the last IGM because it is incomplete
			LastIdxLastIGM = static_cast<int>(std::round(idxStartFirstZPD_ptr[0] + (DcsCfg.TemplateSize - 1) / 2 + (NIGMs_ptr[0] - 1.5) * previousptsPerIGM[0]) + 1);

			idxStartFirstZPD_ptr[1] = std::floor(previousptsPerIGM[0] / 2) - (DcsCfg.TemplateSize - 1) / 2; 	// idx of first ZPD in new segment - NptsTemplateSize, to be verified

			NptsLastIGMBuffer_ptr[1] = currentSegmentSize_ptr[1] - LastIdxLastIGM; // Number of points in buffer for next segment
			NIGMs_ptr[1] = NIGMs_ptr[0] - 1; // NIGMs_ptr for Compute_SelfCorrection
			currentSegmentSize_ptr[2] = currentSegmentSize_ptr[1] - NptsLastIGMBuffer_ptr[1]; // Npts for Compute_SelfCorrection

			h_start_slope_ptr[0] = 0.0f;
			h_start_slope_ptr[1] = -0.5;
			cudaMemcpy(start_slope_ptr + 1, h_start_slope_ptr, 2 * sizeof(double), cudaMemcpyHostToDevice);
			h_end_slope_ptr[0] = 1.0f;
			h_end_slope_ptr[1] = NIGMs_ptr[1] - 0.5;
			cudaMemcpy(end_slope_ptr + 1, h_end_slope_ptr, 2 * sizeof(double), cudaMemcpyHostToDevice);

		}
	}

	// Filter the channels with a 32 taps fir filters

	cudaStatus = Convolution_complex32_optimized_GPU(filteredSignals_ptr, rawData_inGPU_ptr, currentConvolutionBufferIn_ptr, currentConvolutionBufferOut_ptr, currentSegmentSize_ptr[0],
		GpuCfg.i32GpuThreads, GpuCfg.i32GpuBlocks256, u32LoopCount, filterCoeffs_ptr, DcsCfg.Nchannels, DcsCfg.Nfilters, idxchfilt_ptr, cuda_stream, cudaSuccess);

	if (cudaStatus != cudaSuccess)
		ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Convolution_complex32_optimized_GPU launch failed", ERROR_);

	//Remove the CW contribution with a time multiplication of the two references channels and apply IGMs*exp(1j*angle_ref)
	//Here we assume that the signals are placed in this oder: IGMs, foptCW1_C1, foptCW1_C2, foptCW2_C1, foptCW2_C2;
	//CW1 is the optical reference used for the fast phase correction

	if (DcsCfg.Nreferences == 0) {
		IGMsPC_resampled_ptr = filteredSignals_ptr; // We don't do the fast correction, we should still remove a slope for the self-correction
	}
	else if (DcsCfg.Nreferences == 1) { // IF we have at least 1 reference we do the fast phase correction
		cudaStatus = FastPhaseCorrection_GPU(IGMsPC_ptr, ref1_ptr, filteredSignals_ptr, filteredSignals_ptr + 1 * SegmentSizePerChannel,
			filteredSignals_ptr + 2 * SegmentSizePerChannel, SegmentSizePerChannel, DcsCfg.conjugateCW1_C1, DcsCfg.conjugateCW1_C2, DcsCfg.conjugatePhaseCorrection,
			GpuCfg.i32GpuThreads, GpuCfg.i32GpuBlocks256, cuda_stream, cudaSuccess);

		if (cudaStatus != cudaSuccess)
			ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "FastPhaseCorrection_GPU launch failed", ERROR_);

		IGMsPC_resampled_ptr = IGMsPC_ptr;

	}
	else if (DcsCfg.Nreferences == 2) { // IF we have at least 2 references we do the fast phase correction and resampling

		cudaStatus = FastPhaseCorrection_GPU(IGMsPC_ptr, ref1_ptr, filteredSignals_ptr, filteredSignals_ptr + 1 * SegmentSizePerChannel,
			filteredSignals_ptr + 2 * SegmentSizePerChannel, SegmentSizePerChannel, DcsCfg.conjugateCW1_C1, DcsCfg.conjugateCW1_C2, DcsCfg.conjugatePhaseCorrection,
			GpuCfg.i32GpuThreads, GpuCfg.i32GpuBlocks256, cuda_stream, cudaSuccess);

		if (cudaStatus != cudaSuccess)
			ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "FastPhaseCorrection_GPU launch failed", ERROR_);

		// Create the dfr reference 
		// Here we assume that the signals are placed in this oder: IGMs, foptCW1_C1, foptCW1_C2, foptCW2_C1, foptCW2_C2;
		cudaStatus = Calculate_dfr_reference_GPU(ref_dfr_angle_ptr, ref1_ptr, filteredSignals_ptr + 3 * SegmentSizePerChannel,
			filteredSignals_ptr + 4 * SegmentSizePerChannel, DcsCfg.conjugateCW2_C1, DcsCfg.conjugateCW2_C2, DcsCfg.conjugateDfr1, DcsCfg.conjugateDfr1, SegmentSizePerChannel,
			GpuCfg.i32GpuThreads, GpuCfg.i32GpuBlocks256, cuda_stream, cudaSuccess);

		if (cudaStatus != cudaSuccess)
			ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Calculate_dfr_reference_GPU launch failed", ERROR_);



		// Unwrap the phase of the dfr signal
		cudaStatus = UnwrapPhase_GPU(unwrapped_phase_ptr, ref_dfr_angle_ptr, two_pi_cumsum_ptr, blocks_edges_cumsum_ptr, increment_blocks_edges_ptr, SegmentSizePerChannel, Unwrapdfr, EstimateSlope,
			start_slope_ptr, end_slope_ptr, warp_size, GpuCfg.i32GpuBlocks128, cuda_stream, cudaSuccess); // (fast unwrap with 128 threads on 4090 GPU)
		if (cudaStatus != cudaSuccess)
			ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "UnwrapPhase_GPU launch failed", ERROR_);

		// We should add a filter for the phase of the signal

		// Create linspace for resampling with the slope parameters estimated in the unwrap
		cudaStatus = Linspace_GPU(LinearGrid_dfr_ptr, start_slope_ptr, end_slope_ptr, SegmentSizePerChannel, GpuCfg.i32GpuThreads, GpuCfg.i32GpuBlocks256, 0, cuda_stream, cudaSuccess); // idxLinSpace = 0;
		if (cudaStatus != cudaSuccess)
			ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Linspace_GPU launch failed", ERROR_);

		// Do the resampling with two reference with a linear interpolation
		Linear_interpolation_GPU(IGMsPC_resampled_ptr, LinearGrid_dfr_ptr, IGMsPC_ptr, unwrapped_phase_ptr, idx_LinearGrid_dfr_ptr,
			SegmentSizePerChannel, SegmentSizePerChannel, DcsCfg.nintervalInterpolation - 5, GpuCfg.i32GpuThreads / 2, GpuCfg.i32GpuBlocks128, cuda_stream, cudaSuccess); // We assume  GpuCfg.i32GpuThreads = 256

		if (cudaStatus != cudaSuccess)
			ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Linear_interpolation_GPU launch failed", ERROR_);

	}

	// This is for bridging the different segments, needs to be tested
	// We add the cropped IGM from the last segment to this segment for the self-correction
	if (NptsLastIGMBuffer_ptr[0] > 0) {

		cudaMemcpyAsync(IGMsSelfCorrectionIn_ptr, LastIGMBuffer_ptr, NptsLastIGMBuffer_ptr[0] * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, cuda_stream);
		cudaMemcpyAsync(IGMsSelfCorrectionIn_ptr + NptsLastIGMBuffer_ptr[0], IGMsPC_resampled_ptr, currentSegmentSize_ptr[0] * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, cuda_stream);

	}
	else {
		// This is for the first segment, we have an empty buffer
		cudaMemcpyAsync(IGMsSelfCorrectionIn_ptr, IGMsPC_resampled_ptr, currentSegmentSize_ptr[0] * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, cuda_stream);
	}



	if (u32LoopCount == 0) { // For the first segment, we don't know where the first ZPD is, we do a xcorr over a wider range to find it

		cudaStatus = Find_First_ZPD_GPU(idxMaxBLocks_ptr, MaxValBlocks_ptr, IGMsSelfCorrectionIn_ptr, IGMTemplate_ptr, xcorrBLocksIGMs_ptr, idxMidSegments_ptr,
			idxStartFirstZPD_ptr, DcsCfg.TemplateSize, DcsCfg.TemplateSize, DcsCfg.ptsPerIGM, currentSegmentSize_ptr[0], DcsCfg.TemplateSize, blocksPerDelay, totalDelays,
			totalBlocks, cuda_stream, cudaSuccess);
		if (cudaStatus != cudaSuccess)
			ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Find_IGMs_ZPD_GPU launch failed", ERROR_);

	}
	else { // We know where the first ZPD is, so we can do a xcorr on all the igms over a small delay range

		// Copy cropped points at the end of IGM in the buffer
		cudaMemcpyAsync(LastIGMBuffer_ptr, IGMsPC_resampled_ptr + (currentSegmentSize_ptr[0] - NptsLastIGMBuffer_ptr[1]), NptsLastIGMBuffer_ptr[1] * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, cuda_stream); // can it be done on stream1 ??  transfer data in buffer for next segment

		// We find the subpoint position of the ZPDs and their phase
		cudaStatus = Find_IGMs_ZPD_GPU(idxMaxSubpoint_ptr, phaseMaxSubpoint_ptr, IGMsSelfCorrectionIn_ptr, IGMTemplate_ptr, xcorrBLocksIGMs_ptr, idxMidSegments_ptr, idxStartFirstZPD_ptr[0],
			DcsCfg.TemplateSize - DcsCfg.MaxDelayXcorr, NIGMs_ptr[0], previousptsPerIGM[0], currentSegmentSize_ptr[1], currentSegmentSize_ptr[2], DcsCfg.MaxDelayXcorr, blocksPerDelay, totalDelays,
			totalBlocks, ptsPerIGMSegment_ptr, cuda_stream, cudaSuccess);

		if (cudaStatus != cudaSuccess)
			ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Find_IGMs_ZPD_GPU launch failed", ERROR_);

		// We keep track of the dfr of each segment
		cudaMemcpyAsync(previousptsPerIGM, ptsPerIGMSegment_ptr, sizeof(double), cudaMemcpyDeviceToHost, cuda_stream);

		// We adjust the number of blocks for the self-correction
		GpuCfg.i32GpuBlocks128 = (currentSegmentSize_ptr[2] + GpuCfg.i32GpuThreads / 2 - 1) / (GpuCfg.i32GpuThreads / 2);
		GpuCfg.i32GpuBlocks256 = (currentSegmentSize_ptr[2] + GpuCfg.i32GpuThreads - 1) / GpuCfg.i32GpuThreads;

		// Self correction (slow phase correction and resampling)
		cudaStatus = Compute_SelfCorrection_GPU(IGMsSelfCorrection_ptr, IGMsSelfCorrectionIn_ptr, splineGrid_f0_ptr, splineGrid_dfr_ptr, LinearGrid_dfr_ptr, idx_LinearGrid_dfr_ptr, spline_coefficients_f0_ptr,
			spline_coefficients_dfr_ptr, idxMaxSubpoint_ptr, phaseMaxSubpoint_ptr, start_slope_ptr, end_slope_ptr, DcsCfg.ptsPerIGM, NIGMs_ptr[1], currentSegmentSize_ptr[2], DcsCfg.nintervalInterpolation,
			GpuCfg.i32GpuThreads, GpuCfg.i32GpuBlocks256, d_h, d_D, d_work, devInfo, lwork, cuSolver_handle, cuda_stream, cudaSuccess);

		if (cudaStatus != cudaSuccess)
			ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Compute_SelfCorrection_GPU launch failed", ERROR_);

		// Find mean IGM of the self-corrected train of IGMs
		Compute_MeanIGM_GPU(IGM_mean_ptr, IGMsSelfCorrection_ptr, NIGMs_ptr[1], NIGMs_ptr[1] * DcsCfg.ptsPerIGM, DcsCfg.ptsPerIGM, cuda_stream, cudaSuccess);

		if (cudaStatus != cudaSuccess)
			ErrorHandler((int32_t)cudaGetErrorString(cudaStatus), "Compute_MeanIGM_GPU launch failed", ERROR_);


	}


	if (u32LoopCount % 2 == 0) { // for even count (0,2,4...)
		NptsSave = DcsCfg.ptsPerIGM;
		cudaMemcpyAsync(IGMsOut1, IGM_mean_ptr, NptsSave * sizeof(cufftComplex), cudaMemcpyDeviceToHost, cuda_stream);
	}
	else { // for odd count (1,3,5,...)
		NptsSave = DcsCfg.ptsPerIGM;
		cudaMemcpyAsync(IGMsOut2, IGM_mean_ptr, NptsSave * sizeof(cufftComplex), cudaMemcpyDeviceToHost, cuda_stream);
	}


}


void ThreadHandler::copyDataToGPU_async(int32_t u32LoopCount)
{
	// Asynchronously copy data from hWorkBuffer to rawData_inGPU_ptr using stream1

	if (hWorkBuffer)		// we copy only if we have a work buffer

		if (u32LoopCount == 1) // For the first transfer, we need the GPU to wait for the transfer, so we put the transfer on the same stream as the processing
			cudaMemcpyAsync(rawData_inGPU1_ptr, (short*)hWorkBuffer, StreamConfig.u32BufferSizeBytes, cudaMemcpyHostToDevice, cuda_stream);
		else if (u32LoopCount % 2 == 1) { // for odd count (3,5,...) For other transfers, we put it on cuda_stream1 so we transfer while the data is processing
			cudaMemcpyAsync(rawData_inGPU1_ptr, (short *)hWorkBuffer, StreamConfig.u32BufferSizeBytes, cudaMemcpyHostToDevice, cuda_stream1);
		}
		else { // for even count (2,4,6,...)
			cudaMemcpyAsync(rawData_inGPU2_ptr, (short*)hWorkBuffer, StreamConfig.u32BufferSizeBytes, cudaMemcpyHostToDevice, cuda_stream1);
		}



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

void ThreadHandler::WriteProcessedDataToFile(int32_t u32LoopCount)
{
	if (StreamConfig.bSaveToFile)
	{

		DWORD				dwBytesSave = 0;
		BOOL				bWriteSuccess = TRUE;

		if (u32LoopCount % 2 == 0) { // for even count (0,2,4...)
			bWriteSuccess = WriteFile(fileHandle_processedData_out, IGMsOut1, NptsSave * sizeof(cufftComplex), &dwBytesSave, NULL);
		}
		else { // for odd count (1,3,5,...)
			bWriteSuccess = WriteFile(fileHandle_processedData_out, IGMsOut2, NptsSave * sizeof(cufftComplex), &dwBytesSave, NULL);

		}




		

		if (!bWriteSuccess || dwBytesSave != NptsSave * sizeof(Complex))
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


ThreadHandler::ThreadHandler(GaGeCard_interface& acq, CUDA_GPU_interface& gpu, AcquisitionThreadFlowControl& flow, DCSProcessingHandler& dcs) // Constructor

{

	// Locking mutex while we access shared variables to configure object
	// and create local copies of variables what we will read often
	// this means changes to shared variables will not affect the procesing thread until with re-sync the local variables.

	const std::lock_guard<std::shared_mutex> lock(flow.sharedMutex);	// Lock gard unlonck when destroyed (i.e at the end of the constructor)
	// Could use a shared lock since we only read variables
	// but playing safe here, no acquisition runninng in init phase anyway
	acquisitionCardPtr = &acq;
	gpuCardPtr = &gpu;
	DcsProcessingPtr = &dcs;
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
	DcsCfg = DcsProcessingPtr->getDcsConfig();
	//csHandle = acquisitionCard->GetSystemHandle();

	StreamConfig.NActiveChannel = CsAcqCfg.u32Mode & CS_MASKED_MODE;

	DcsCfg.outputDataFilename = StreamConfig.strResultFile;
	/*sprintf_s(szSaveFileNameI, sizeof(szSaveFileNameI), "%s_I", StreamConfig.strResultFile);
	sprintf_s(szSaveFileNameO, sizeof(szSaveFileNameO), "%s_O", StreamConfig.strResultFile);*/

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



void ThreadHandler::ReadandAllocateFilterCoefficients()
{
	std::cout << "Reading Filters filename: " << DcsCfg.filtersFilename << std::endl;
	readBinaryFileC(DcsCfg.filtersFilename, filterCoeffs_ptr, DcsCfg.Nfilters * MASK_LENGTH);

}

void ThreadHandler::ReadandAllocateTemplateData()
{

	// choose which pointer to use
	std::cout << "Reading Template filename: " << DcsCfg.templateDataFilename << std::endl;

	readBinaryFileC(DcsCfg.templateDataFilename, IGMTemplate_ptr, DcsCfg.TemplateSize);
}

bool ThreadHandler::readBinaryFileC(const char* filename, cufftComplex* data, size_t numElements)
{
	// Open the binary file in binary mode for filename1
	FILE* file1 = fopen(filename, "rb");
	if (!file1) {
		fprintf(stderr, "Unable to open the file: %s\n", filename);
		return false;
	}

	// Read the data into the provided data pointer
	for (size_t i = 0; i < numElements; ++i) {
		if (fread(&data[i].x, sizeof(float), 1, file1) != 1) {
			fprintf(stderr, "Error reading real part from the file: %s\n", filename);
			fclose(file1);
			return false;
		}

		if (fread(&data[i].y, sizeof(float), 1, file1) != 1) {
			fprintf(stderr, "Error reading imaginary part from the file: %s\n", filename);
			fclose(file1);
			return false;
		}

		//printf("\n Mask[%d].x: %f, Mask[%d].y : %f", i, data[i].x, i, data[i].y);


	}
	printf("Finished reading filters coefficients file\n");
	// Close the file when done
	fclose(file1);

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
	DcsCfg.Nchannels = StreamConfig.NActiveChannel;
	SegmentSizePerChannel = u32TransferSizeSamples / StreamConfig.NActiveChannel;
	//DcsCfg.N = u32TransferSizeSamples / DcsCfg.Nchannels;
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

	cudaStreamCreate(&cuda_stream1);
}

void ThreadHandler::CreatecuSolverHandle()
{
	cusolverDnCreate(&cuSolver_handle);
	cusolverDnSetStream(cuSolver_handle, cuda_stream);

}

void ThreadHandler::AllocateGPUBuffers()
{

	// General GPU variables
	IGMsOut1 = (cufftComplex*)malloc(SegmentSizePerChannel * DcsCfg.Nchannels * sizeof(cufftComplex));
	if (IGMsOut1 != nullptr) {
		// Zero out the allocated memory
		memset(IGMsOut1, 0, 2 * SegmentSizePerChannel * sizeof(cufftComplex));
	}
	else {
		// Handle the error, perhaps throw an exception or return an error code
	}
	IGMsOut2 = (cufftComplex*)malloc(SegmentSizePerChannel * DcsCfg.Nchannels * sizeof(cufftComplex));
	if (IGMsOut2 != nullptr) {
		// Zero out the allocated memory
		memset(IGMsOut2, 0, 2 * SegmentSizePerChannel * sizeof(cufftComplex));
	}
	else {
		// Handle the error, perhaps throw an exception or return an error code
	}
	currentSegmentSize_ptr[0] = 0;
	currentSegmentSize_ptr[1] = 0;
	currentSegmentSize_ptr[2] = 0;
	// Raw data buffers
	rawData_inCPU_ptr = (short*)malloc(DcsCfg.Nsegments * u32TransferSizeSamples * sizeof(short)); // should not need this one!
	if (rawData_inCPU_ptr != nullptr) {
		// Zero out the allocated memory
		memset(rawData_inCPU_ptr, 0, u32TransferSizeSamples * sizeof(short));
	}
	else {
		// Handle the error, perhaps throw an exception or return an error code
	}
	cudaMalloc((void**)&rawData_inGPU_ptr, u32TransferSizeSamples * sizeof(short));
	cudaMemset(rawData_inGPU_ptr, 0, u32TransferSizeSamples);
	cudaMalloc((void**)&rawData_inGPU1_ptr, u32TransferSizeSamples * sizeof(short));
	cudaMemset(rawData_inGPU1_ptr, 0, u32TransferSizeSamples);
	cudaMalloc((void**)&rawData_inGPU2_ptr, u32TransferSizeSamples * sizeof(short));
	cudaMemset(rawData_inGPU2_ptr, 0, u32TransferSizeSamples);

	// Filtering
	cudaMalloc((void**)&filteredSignals_ptr, DcsCfg.Nfilters * SegmentSizePerChannel * sizeof(cufftComplex));
	filterCoeffs_ptr = (cufftComplex*)malloc(DcsCfg.Nfilters * MASK_LENGTH * sizeof(double)); // We will copy the coefficients in constant memory in convolution kernel
	Convolution_Buffer1_CPU_ptr = (float*)malloc(DcsCfg.Nchannels * (MASK_LENGTH - 1) * sizeof(float));
	cudaMalloc((void**)&Convolution_Buffer1_ptr, DcsCfg.Nchannels * (MASK_LENGTH - 1) * sizeof(float));
	cudaMemset(Convolution_Buffer1_ptr, 0, DcsCfg.Nchannels * (MASK_LENGTH - 1));
	cudaMalloc((void**)&Convolution_Buffer2_ptr, DcsCfg.Nchannels * (MASK_LENGTH - 1) * sizeof(float));
	cudaMemset(Convolution_Buffer2_ptr, 0, DcsCfg.Nchannels * (MASK_LENGTH - 1));
	cudaMalloc((void**)&idxchfilt_ptr, DcsCfg.Nfilters * sizeof(int));
	cudaMemcpy(idxchfilt_ptr, DcsCfg.idxchFilters, DcsCfg.Nfilters * sizeof(int), cudaMemcpyHostToDevice);

	// Fast phase Correction 
	cudaMalloc((void**)&ref1_ptr, SegmentSizePerChannel * sizeof(cufftComplex));
	cudaMalloc((void**)&IGMsPC_ptr, SegmentSizePerChannel * sizeof(cufftComplex));

	// Unwrapping
	int numBLocks = (SegmentSizePerChannel + GpuCfg.i32GpuThreads - 1) / GpuCfg.i32GpuThreads;
	cudaMalloc((void**)&unwrapped_phase_ptr, SegmentSizePerChannel * sizeof(double));
	cudaMemset(unwrapped_phase_ptr, 0, SegmentSizePerChannel);
	cudaMalloc((void**)&two_pi_cumsum_ptr, SegmentSizePerChannel * sizeof(int));
	cudaMemset(two_pi_cumsum_ptr, 0, SegmentSizePerChannel);
	cudaMalloc((void**)&blocks_edges_cumsum_ptr, SegmentSizePerChannel * sizeof(int)); // Should be way smaller than this
	cudaMemset(blocks_edges_cumsum_ptr, 0, SegmentSizePerChannel);
	cudaMalloc(&increment_blocks_edges_ptr, numBLocks * sizeof(int));
	cudaMemset(increment_blocks_edges_ptr, 0, numBLocks);
	Unwrapdfr = true; // This is if we want to unwrap something different thant dfr ref (logic not implemented yet)
	EstimateSlope = true;


	// 2 ref resampling 
	cudaMalloc((void**)&IGMsPC_resampled_ptr, (SegmentSizePerChannel + 2 * DcsCfg.ptsPerIGM) * sizeof(cufftComplex));
	cudaMalloc((void**)&ref_dfr_angle_ptr, SegmentSizePerChannel * sizeof(float));
	AllocateCudaManagedBuffer((void**)&start_slope_ptr, 3 * sizeof(double));
	AllocateCudaManagedBuffer((void**)&end_slope_ptr, 3 * sizeof(double));
	cudaMalloc((void**)&LinearGrid_dfr_ptr, (SegmentSizePerChannel + 2 * DcsCfg.ptsPerIGM) * sizeof(double)); // Factor of 2 to make sure we always have enough space depending on the variations in dfr 
	cudaMalloc((void**)&idx_LinearGrid_dfr_ptr, (SegmentSizePerChannel + 2 * DcsCfg.ptsPerIGM) * sizeof(int)); // Factor of 2 to make sure we always have enough space depending on the variations in dfr 


	// Find_IGMs_ZPD_GPU
	cudaMalloc((void**)&IGMsSelfCorrectionIn_ptr, (SegmentSizePerChannel + 2 * DcsCfg.ptsPerIGM) * sizeof(cufftComplex));
	AllocateCudaManagedBuffer((void**)&IGMTemplate_ptr, 2 * DcsCfg.ptsPerIGM * sizeof(cufftComplex)); // Should not be longer than ptsPerIGM
	IGMTemplate_ptr = (cufftComplex*)ALIGN_UP(IGMTemplate_ptr, MEMORY_ALIGNMENT);
	cudaMalloc((void**)&xcorrBLocksIGMs_ptr, (SegmentSizePerChannel) * sizeof(cufftComplex));
	cudaMalloc((void**)&LastIGMBuffer_ptr, 3 * DcsCfg.ptsPerIGM * sizeof(cufftComplex));  // Factor of 2 to make sure we always have enough space depending on the variations in dfr
	cudaMalloc((void**)&idxMaxSubpoint_ptr, 3 * (SegmentSizePerChannel / DcsCfg.ptsPerIGM) * sizeof(double));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	cudaMalloc((void**)&phaseMaxSubpoint_ptr, 3 * (SegmentSizePerChannel / DcsCfg.ptsPerIGM) * sizeof(double));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	AllocateCudaManagedBuffer((void**)&ptsPerIGMSegment_ptr, sizeof(double)); // We could put the values for all the batches in this if we want
	previousptsPerIGM[0] = DcsCfg.ptsPerIGMSegment;
	cudaMalloc((void**)&idxMidSegments_ptr, 3 * (SegmentSizePerChannel / DcsCfg.ptsPerIGM) * sizeof(int));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	NptsLastIGMBuffer_ptr[0] = 0;
	NptsLastIGMBuffer_ptr[1] = 0;
	//idxStartFirstZPD_ptr[0] = DcsCfg.idxStartFirstZPD;
	idxStartFirstZPD_ptr[0] = 0;
	idxStartFirstZPD_ptr[1] = 0;

	// For Find_First_ZPD_GPU
	cudaMalloc((void**)&idxMaxBLocks_ptr, DcsCfg.ptsPerIGM * sizeof(int)); // Should be DcsCfg.ptsPerIGM/256
	cudaMalloc((void**)&MaxValBlocks_ptr, DcsCfg.ptsPerIGM * sizeof(float)); // Should be DcsCfg.ptsPerIGM/256

	// For Compute_SelfCorrection_GPU
	//AllocateCudaManagedBuffer((void**)&IGMsSelfCorrection_ptr, (SegmentSizePerChannel + 2 * DcsCfg.ptsPerIGM) * sizeof(cufftComplex)); // Factor of 2 to make sure we always have enough space depending on the variations in dfr 
	cudaMalloc((void**)&IGMsSelfCorrection_ptr, (SegmentSizePerChannel + 2 * DcsCfg.ptsPerIGM) * sizeof(cufftComplex));
	cudaMalloc((void**)&spline_coefficients_dfr_ptr, 3 * (SegmentSizePerChannel / DcsCfg.ptsPerIGM) * sizeof(double)); // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	cudaMalloc((void**)&spline_coefficients_f0_ptr, 3 * (SegmentSizePerChannel / DcsCfg.ptsPerIGM) * sizeof(double)); // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	cudaMalloc((void**)&splineGrid_dfr_ptr, (SegmentSizePerChannel + 2 * DcsCfg.ptsPerIGM) * sizeof(double)); // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	cudaMalloc((void**)&splineGrid_f0_ptr, (SegmentSizePerChannel + 2 * DcsCfg.ptsPerIGM) * sizeof(float)); // Factor of 3 to make sure we always have enough space depending on the variations in dfr 

	// Variables for cuSOlver to compute spline coefficients in Compute_SelfCorrection_GPU	
	// This is to compute the spline coefficients
	// We don't know the max size because it can vary based on the number of igms per batch
	// We will put 10 * NIGMs_ptr to be safe for now
	// We launch two of these to do f0 spline and dfr spline at the same time
	NIGMs_ptr[0] = SegmentSizePerChannel / DcsCfg.ptsPerIGM; // Approximate number of IGMs per segment
	NIGMs_ptr[1] = 0;
	cudaMallocAsync(&d_h, sizeof(double) * (10 * NIGMs_ptr[0] - 1) * (10 * NIGMs_ptr[0] - 1), cuda_stream); 
	cudaMallocAsync(&d_D, sizeof(double) * (10 * NIGMs_ptr[0] - 1) * (10 * NIGMs_ptr[0] - 1), cuda_stream);
	cudaMallocAsync(&devInfo, sizeof(int), cuda_stream);

	// Initialize memory to zero asynchronously
	cudaMemsetAsync(d_h, 0, sizeof(double) * (10 * NIGMs_ptr[0] - 1) * (10 * NIGMs_ptr[0] - 1), cuda_stream);
	cudaMemsetAsync(d_D, 0, sizeof(double) * (10 * NIGMs_ptr[0] - 1) * (10 * NIGMs_ptr[0] - 1), cuda_stream);
	// Allocate workspace for cuSOLVER operations

	cusolverDnDpotrf_bufferSize(cuSolver_handle, CUBLAS_FILL_MODE_UPPER, 10 * NIGMs_ptr[0] - 1, d_D, 10 * -1, &lwork);
	cudaMallocAsync(&d_work, sizeof(double) * lwork, cuda_stream); // pointer for the  Workspace for computations

	// For Compute_MeanIGM_GPU
	cudaMalloc((void**)&IGM_mean_ptr, DcsCfg.ptsPerIGM * sizeof(cufftComplex));



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

#include <windows.h>
#include <ctime>

// Assuming other necessary headers and definitions are included

void ThreadHandler::CreateOuputFiles()
{
	if (StreamConfig.bSaveToFile)
	{
		// Using ANSI string directly from configuration
		strncpy_s(szSaveFileNameO, MAX_PATH, DcsCfg.outputDataFilename, _TRUNCATE);

		// Get current date and time
		time_t now = time(NULL);
		struct tm newtime;
		char dateTimeStr[100];
		localtime_s(&newtime, &now);

		// Format date and time: YYYYMMDD_HHMMSS
		snprintf(dateTimeStr, 100, "_%04d%02d%02d_%02dh%02dm%02ds.bin",
			newtime.tm_year + 1900, newtime.tm_mon + 1, newtime.tm_mday,
			newtime.tm_hour, newtime.tm_min, newtime.tm_sec);

		// Append date and time to the filename
		strncat_s(szSaveFileNameO, MAX_PATH, dateTimeStr, _TRUNCATE);

		// Create file using CreateFileA
		HANDLE fileHandle_processedData_out = CreateFileA(szSaveFileNameO, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);

		if (fileHandle_processedData_out == INVALID_HANDLE_VALUE)
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

