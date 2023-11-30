// SimpleMultiChannelFilterGPU
// 
// main.cpp
// 
// Baseline example performing real-time data acquistion from a Gage Card
// N channels are filtered and data is saved to disk
// 
// 
// 
// Mathieu Walsh 
// Jerome Genest
// October 2023
// Nov 1st, 2023: moving to std::treads and objects


// Test objectification branch
// standard includes

#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <shared_mutex>

#include <stdlib.h>
#include <io.h>

//#include "ErrorHandler.h"
#include "AcquisitionProcessingThread.h"
#include "GaGeCard_Interface.h"
#include "CUDA_GPU_Interface.h"



void UpdateProgress(uInt32 u32Elapsed, LONGLONG llTotaBytes)
{
	uInt32	h = 0;
	uInt32	m = 0;
	uInt32	s = 0;
	double	dRate;
	double	dTotal;

	if (u32Elapsed > 0)
	{
		dRate = (llTotaBytes / 1000000.0) / (u32Elapsed / 1000.0);

		if (u32Elapsed >= 1000)
		{
			if ((s = u32Elapsed / 1000) >= 60)	// Seconds
			{
				if ((m = s / 60) >= 60)			// Minutes
				{
					if ((h = m / 60) > 0)		// Hours
						m %= 60;
				}
				s %= 60;
			}
		}
		dTotal = 1.0 * llTotaBytes / 1000000.0;		// Mega samples
		printf("\rTotal: %0.2f MB, Rate: %6.2f MB/s, Elapsed time: %u:%02u:%02u", dTotal, dRate, h, m, s);
		//printf("Total: %0.2f MB, Rate: %6.2f MB/s, Elapsed time: %u:%02u:%02u \n", dTotal, dRate, h, m, s);
	}
}


/***************************************************************************************************
****************************************************************************************************/

//
//DisplayResults(1,
//	g_GpuConfig.bUseGpu,
//	g_CsAcqCfg.u32Mode& CS_MASKED_MODE, // mask of the constant that loaded the expert firmware
//	g_CsAcqCfg.u32SegmentCount,
//	llSystemTotalData,
//	g_CsAcqCfg.u32SampleSize,
//	g_GpuConfig.u32SkipFactor,
//	dSystemTotalTime,
//	g_GpuConfig.strResultFile);

void DisplayResults(GaGeCard_interface& acqCard, CUDA_GPU_interface& gpuCard, int64_t i64TransferLength, double time)
//void DisplayResults(int stream,int gpu,uint32_t u32Mode,uint32_t u32SegmentCount,int64_t i64TransferLength,uint32_t u32SampleSize,uint32_t u32SkipFactor,double time,char* filename)
{
	char s[26];
	char str[255];
	char szHeader[255];
	bool bFileExists;
	bool bWriteToFile = TRUE;
	FILE* file;
	SYSTEMTIME lt;
	GetLocalTime(&lt);

	int stream = 1;

	GPUCONFIG	gpuConfig = gpuCard.getConfig();
	CSACQUISITIONCONFIG acqConfig = acqCard.getAcquisitionConfig();

	int gpu = gpuConfig.bUseGpu;
	uint32_t u32Mode = acqConfig.u32Mode & CS_MASKED_MODE;
	uint32_t u32SegmentCount = acqConfig.u32SegmentCount;
	uint32_t u32SampleSize = acqConfig.u32SampleSize;
	uint32_t u32SkipFactor = gpuConfig.u32SkipFactor;
	//char filename[255] = gpuConfig.strResultFile;

	bFileExists = (-1 == _access(gpuConfig.strResultFile, 0)) ? FALSE : TRUE;

	if (bFileExists)
	{
		if (-1 == _access(gpuConfig.strResultFile, 2) || -1 == _access(gpuConfig.strResultFile, 6))
		{
			printf("\nCannot write to %s\n", gpuConfig.strResultFile);
			bWriteToFile = FALSE;
		}
	}

	sprintf_s(szHeader, _countof(szHeader), "\n\nDate\t  Time\tStream\t GPU\tChannels  Records   Samples\t\tBytes\tSkip\tTime (ms)\n\n");
	printf("%s", szHeader);


	sprintf_s(s, _countof(s), "%04d-%02d-%02d  %2d:%2d:%02d", lt.wYear, lt.wMonth, lt.wDay, lt.wHour, lt.wMinute, lt.wSecond);
	sprintf_s(str, _countof(str), "%s\t  %d\t  %d\t  %d\t    %d\t    %I64d\t\t %d\t %d\t%.3f\n", s, stream, gpu, u32Mode, u32SegmentCount, i64TransferLength, u32SampleSize, u32SkipFactor, time);

	printf("%s", str);

	if (bWriteToFile)
	{
		file = fopen(gpuConfig.strResultFile, "a");
		if (NULL != file)
		{
			if (!bFileExists) // first time so write the header
			{
				fwrite(szHeader, 1, strlen(szHeader), file);
			}
			fwrite(str, 1, strlen(str), file);
			fclose(file);
		}
	}
}



int main()
{
	uInt32						u32TickStart = 0;
	uInt32						u32TickNow = 0;
	//double						dTotalData = 0.0;

	AcquisitionThreadFlowControl threadControl;									// Everything needed for thread flow control and sync
																				// flags, in and out queues and shared mutex

	std::atomic<bool>		test; 
	
	CUDA_GPU_interface GpuCard;													// Object to configure, intialize and control GPU
	GaGeCard_interface AcqusitionCard((std::string)"GaGeCardInitFile.ini");		// Object to configure, intialize and control acqu card

	GpuCard.setTotalData(1000);

	try
	{
		GpuCard.FindandSetMyCard();
	}
	catch (std::exception& except)
	{
		std::cout << "Could not initialize and configure CPU Card:  " << except.what() << "\n";
		return -1;
	}

	
	try
		{
		AcqusitionCard.InitializeAndConfigure();
		AcqusitionCard.LoadStmConfigurationFromInitFile();
		AcqusitionCard.InitializeStream();				// Was after GPU config in original code
		AcqusitionCard.CleanupFiles();					// Erasing data files having the same name, will have to do better, eventually

		AcqusitionCard.Commit();						// Actually send the params and config to the card
		AcqusitionCard.RetreiveAcquisitionConfig();		// Get acq config from card post commit as some thing might have changed
		AcqusitionCard.RetreiveTotalRequestedSamples();	// Get the resquested number of samples to be acquires -1 == infinity

		}
	catch(std::exception &except)
		{
		std::cout << "Could not initialize and configure GaGe Card:  "  << except.what() << "\n";
		return -1;
		}

	/* The processing thread
	 the thread is created to run the function "AcqusitionProcessingThreadFunction"
	 Passing it references to the AcquisitionCard, (GPU), flow control objects  by reference
	 The thread starts when we create it*/


		std::thread AcquistionAndProcessingThread(AcqusitionProcessingThreadFunction, std::ref(AcqusitionCard), std::ref(GpuCard), std::ref(threadControl));

		std::cout << "Waiting until thread is ready to process\n";

		while(threadControl.ThreadReady == 0 && threadControl.ThreadError == 0)
			{
			// should implement a timeout here !!
			}


		std::cout << "Start streaming. Press ESC to abort\n\n";

		AcqusitionCard.StartStreamingAcquisition();	// Starts the acquistion, why is this not done in the thread ?  I guess only one call for multiboard systems
		
		threadControl.AcquisitionStarted = true;

		u32TickStart = u32TickNow = GetTickCount();

		// Even loop, main thread

		//std::cout << "Main thread event loop hit esc to end program\n";

		while( ! threadControl.AbortThread && threadControl.ThreadReady && !threadControl.ThreadError) 
		{									  // Could also launch / join threads from this event loop, so a differentcase would quit the main program
											  // for example  q: quit,  l: lunch acquistion thread  a: abort and join thread....
		if (_kbhit())						  // we would then also check for ThreadError to join the thread, I guess. 
			{
				switch (toupper(_getch()))
				{
				case 27:			// ESC key -> abort
					threadControl.AbortThread = TRUE;
					break;
				default:
					MessageBeep(MB_ICONHAND);
					break;
				}
			}

		u32TickNow = GetTickCount();

		UpdateProgress(u32TickNow - u32TickStart, 0);  // Acquired data is not yet pushed from processing thread
		}


		// Thread and program terminating, endThreadSignal was set to true
		// waiting for thread to end
		AcquistionAndProcessingThread.join();

		
		//DisplayResults(AcqusitionCard, GpuCard, llSystemTotalData, dSystemTotalTime);
		DisplayResults(AcqusitionCard, GpuCard, GpuCard.getTotalData(), GpuCard.getDiffTime());
	return 0;
}


