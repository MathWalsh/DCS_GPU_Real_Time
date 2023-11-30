// SimpleMultiChannelFilterGPU
// 
// general_project_defines.h
// 
//Contains general defines and type def

// Mathieu Walsh 
// Jerome Genest
// October 2023
//

// JGe 30/10/2023 : Lots could / should be moved in GaGe or GPU related code

#pragma once


#define	MAX_CARDS_COUNT			10					// Max number of cards supported in a M/S Compuscope system 
#define	SEGMENT_TAIL_ADJUST		64					// number of bytes at end of data which holds the timestamp values
#define OUT_FILE				"Data"				// name of the output file 
#define LOOP_COUNT				1000
#define TRANSFER_TIMEOUT		10000				
#define STREAM_BUFFERSZIZE		0x200000
#define STM_SECTION				_T("StmConfig")		// section name in ini file
#define ACQ_SECTION				_T("Acquisition")	// section name in ini file

#define RESULTS_FILE			_T("Result")
#define	MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

#define MASK_LENGTH 32

#define DEFAULT_SKIP_FACTOR 1
#define DEFAULT_DO_ANALYSIS 1
#define DEFAULT_USE_GPU 1
#define GPU_SECTION _T("GpuConfig")		/* section name in ini file for GPU params*/



