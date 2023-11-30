// AcqusitionProcessingThread.cpp
// 
// Contains thread function for the processing and acquisition thread
// that receives data from the acquisition card,  sends it to the GPU for processing and saves to disk
// 
// 
// Mathieu Walsh 
// Jerome Genest
// November 2023

#include <iostream>

#include "AcquisitionProcessingThread.h"
#include "ThreadHandler.h"


void AcqusitionProcessingThreadFunction(GaGeCard_interface& AcquisitionCard, CUDA_GPU_interface& GpuCard, AcquisitionThreadFlowControl& threadControl, DCSProcessingHandler& DcsProcessing)
{
	/**Configuration**/

	ThreadHandler handler(AcquisitionCard, GpuCard, threadControl, DcsProcessing);		// Critical section: Operations in constructor under mutex lock

	std::cout << "We are in processing thread !!!\n";

	try
	{
		handler.CreateOuputFiles();

		handler.ReadandAllocateFilterCoefficients();

		handler.AllocateAcquisitionCardStreamingBuffers();		// critical section: this is performed under a mutex lock 

		handler.RegisterAlignedCPUBuffersWithCuda();

		handler.AllocateGPUBuffers();

		handler.ReadandAllocateTemplateData();

		handler.CreateCudaStream();

		handler.CreatecuSolverHandle();

		handler.setReadyToProcess(true); // Thread ready to process
		std::cout << "Thread Ready to process..\n";
	}
	catch (std::exception& except)
	{
		std::cout << "Can't configure processing thread: " << except.what() << "\n";
		threadControl.ThreadError = 1;
	}

	while(threadControl.AcquisitionStarted ==0)
		{
		// should implement a time out here !!
		}

	/**Acquisition / Real time Processing**/

	uint32_t	u32LoopCount = 0;

	while ( (threadControl.AbortThread == false) && (threadControl.ThreadError == false) && (handler.isAcqComplete() == false))
		// Looping with the acquisition streaming as long as No user about, no error, and not done.
	{

		try
		{
			handler.setCurrentBuffers(u32LoopCount);

			handler.StartCounter();

			handler.ScheduleCardTransferWithCurrentBuffer(); // Instructing acquisition card to transfert to current buffer

		
			handler.copyDataToGPU_async(u32LoopCount);		// Async copy to GPU, the can't occur in current Buffer since we just started the card DMA transfert
															// All processing need to happen on the work buffer.
															// if we want to hide the async transfer to GPU, we probably need a triple buffer approach  
															// (one for acq card DMA (current buffer) , one for DMA transfert to GPU (previous buffer), oen for processing (work buffer)  
			if (u32LoopCount > 1) {
				handler.ProcessInGPU(u32LoopCount -2);				// This is where GPU processing occurs on work buffer. -1 because it is easier to understand
				// should convert to bollean arguemnt u32LoopCount&1
				//handler.WriteRawDataToFile();					 // Note that this operates on the previous workBuffer, while we transfert to current buffer;
			}
			if (u32LoopCount > 2) {
				handler.WriteProcessedDataToFile(u32LoopCount - 3);
			}
			
						
			handler.StopCounter();

															// here we have to wait, but not sleep until DMA and GPU tasks are complete  
			handler.sleepUntilDMAcomplete();				// Right now this call sleeps the thread until DMA is done...	
															// so if / when we do triple buffer we need to check, without sleeping that both mem copies are done

			handler.setWorkBuffers();						// current buffers become the work buffers for next loop
			u32LoopCount++;
		}
		catch (std::exception& except)
		{
			std::cout << "Error in thread processing loop: " << except.what() << "\n";
			threadControl.ThreadError = 1;
		}
	}


	/**Acq done or aborted**/
	std::cout << "\nOut of the acq loop !!!\n";
	std::cout << "\nProcessing final block\n";


	handler.StartCounter();

	// Processing of the final workBuffer

		handler.copyDataToGPU_async(u32LoopCount); // add +1 to loop count? 
		handler.ProcessInGPU(u32LoopCount);				 

		//handler.WriteRawDataToFile();								// I do not have the padding with the sector size of the original code
		//handler.WriteProcessedDataToFile();						// I do not understand what this is doing...

	handler.StopCounter();

	handler.setReadyToProcess(false);

	std::cout << "\nExiting processing thread !!!\n";

}

