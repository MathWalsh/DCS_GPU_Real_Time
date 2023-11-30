// AcqusitionProcessingThread.h
// 
// Contains thread function prototype 
// for the processing and acquisition thread
// 
// Also contains the struct typedef to handle informtion between the two threads
// 
// 
// Mathieu Walsh 
// Jerome Genest
// November 2023


#pragma once

#include <queue>
#include <mutex>
#include <atomic>
#include <shared_mutex>

#include "GaGeCard_Interface.h"
#include "CUDA_GPU_Interface.h"
#include "DCSprocessingHandler.h"
											
 struct AcquisitionThreadFlowControl
{

	// Atomic bools allow sync & flow control between threads 
	// Atomic bools are not necessrily lock free
	// Might consider changing it to flags (but they do not have a test only operator

	std::atomic<bool>		ThreadError;						// Used by thread to report an error
	std::atomic<bool>		AbortThread;					// signaling User requested abort
	std::atomic<bool>		ThreadReady;					// processing thread informs that its ready to process (done initializing)
	std::atomic<bool>		AcquisitionStarted;				// signals that acquisiton has started


	std::queue<std::string> messages_main2processing;		// Messaging queue from parent to acq/proc thread
	std::queue<std::string> messages_processing2main;		// Messaging queue from acq/proc thread to parent (main) thread

	std::shared_mutex		sharedMutex;					// Shared mutex to enable unique  as well as shared locks 
															// Unique : for write operation, Shared for read (let the readers read !)

	AcquisitionThreadFlowControl()
		: ThreadError(false), AbortThread(false), ThreadReady(false), AcquisitionStarted(false) {}

} ;


// thread function Prototopye

void AcqusitionProcessingThreadFunction(GaGeCard_interface& AcquisitionCard, CUDA_GPU_interface& GpuCard, AcquisitionThreadFlowControl& threadControl, DCSProcessingHandler& DcsProcessing);

