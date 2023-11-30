// CUDA_GPU_interface.h
// 
// Contains function prototypes, defines, et al
// for all thnings needed to operate CUDA card
// 
// Mathieu Walsh 
// Jerome Genest
// October 2023
//


// Data structures here are NOT CLEAN, as second pass will be needed to avoid redundancies;


#pragma once

// Includes 

#include <string>
#include <stdint.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ErrorHandler.h"

// Defines

#define MAX_PATH	255		// this was defined in one of the gage include

//#include "general_project_defines.h"  // arrrk, temporary patch
#define DEFAULT_SKIP_FACTOR		1
#define DEFAULT_DO_ANALYSIS		1
#define DEFAULT_USE_GPU			1
#define DEFAULT_RESULT_FILE		"gpuResult"

// Typedefs

typedef struct
{
	int32_t		i32GpuBlocks;
	int32_t		i32GpuThreads;
	uint32_t	u32SkipFactor;
	char		strResultFile[MAX_PATH];
	bool		bDoAnalysis;			/* Turn on or off data analysis */
	bool		bUseGpu;				/* Turn on or off GPU usage */
}GPUCONFIG, * PGPUCONFIG;

class CUDA_GPU_interface
{
private:
	GPUCONFIG		GpuCfg					= { 0 };

	int32_t			i32MaxBlocks			= 0;
	int32_t			i32MaxThreadsPerBlock	= 0;
	bool			hasPinGenericMemory		= 0;

	int				currentDevice			= -1;			// Valid device IDS are positive int, 0 beeing the fastest one.
	cudaDeviceProp	currentDeviceProperties	= { 0 };
	cudaError_t		cudaStatus				= cudaSuccess;   // Error of the last cuda call

	double			diff_time = 0;				// in the original code this was a global, might have to make this a shared variable
	long long int	TotalData = 0;			// total acquied data. If we want to 'update progress' in main thread, then we need to send this from processing to main


public:
				CUDA_GPU_interface();					// Constructor

				~CUDA_GPU_interface();					// Destructor

	void		FindandSetMyCard();						// Quick and dirty function that finds a card doing what we need
														//  and sets it.We could \ should do better, ventually

	int			FindAnyDeviceWithHostMemoryMapping();
	cudaError_t SetDevice(int32_t nDevice);

	cudaError_t RetreivePropertiesFromDevice();			// Retreive info from card and populate object variables

	GPUCONFIG	getConfig();							// Just return the config held by object
	void		setConfig(GPUCONFIG Cfg);

	double getDiffTime();
	void setDiffTime(double time);

	long long int getTotalData();
	void setTotalData(long long int size);
};

