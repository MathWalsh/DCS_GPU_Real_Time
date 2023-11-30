// GaGeCard_interface.h
// 
// Contains function prototypes, defines, et al
// for all thnings needed to operate GaGe card
// 
// Mathieu Walsh 
// Jerome Genest
// October 2023
//



#pragma once

#include <string>

#include "CsAppSupport.h"
#include "CsTchar.h"
#include "CsExpert.h"

//#include "general_project_defines.h"

//#define	MAX_CARDS_COUNT			10					// Max number of cards supported in a M/S Compuscope system 
//#define	SEGMENT_TAIL_ADJUST		64					// number of bytes at end of data which holds the timestamp values
#define		OUT_FILE				"Data"				// name of the output file 
//#define	LOOP_COUNT				1000
#define		TRANSFER_TIMEOUT		10000				
#define		STREAM_BUFFERSZIZE		0x200000
#define		STM_SECTION				_T("StmConfig")		// section name in ini file
//#define ACQ_SECTION				_T("Acquisition")	// section name in ini file

// User configuration variables
typedef struct
{
	uInt32			u32BufferSizeBytes;
	uInt32			u32TransferTimeout;
	uInt32			u32DelayStartTransfer;
	TCHAR			strResultFile[MAX_PATH];
	BOOL			bSaveToFile;			// Save data to file or not
	BOOL			bFileFlagNoBuffering;	// Should be 1 for better disk performance
	BOOL			bErrorHandling;			// How to handle the FIFO full error
	CsDataPackMode	DataPackCfg;
	// Modif MW
	uInt32			NActiveChannel;
	uInt32*			IdxChannels;
	uInt32			NptsTot;
}CSSTMCONFIG, * PCSSTMCONFIG;

class GaGeCard_interface
{
private:
	int32			i32Status;			// Status of the latest call to compuscope functions
	LPCTSTR			InitialisationFile;	// Default value for init file
	CSHANDLE		GaGe_SystemHandle;	// Handle to the GaGe acquisition system we will be using
	CSSYSTEMINFO	CsSysInfo;			// Information on the selected acq system
	CSSTMCONFIG		StreamConfig;		// stream configuration info
	CSACQUISITIONCONFIG	CsAcqCfg;
	uInt32			u32Mode;			// This is modified by configure from file, not idea of use JGe nov23

	LONGLONG		TotalRequestedSamples = 0;

public:
	GaGeCard_interface(); // Constructor
	GaGeCard_interface(std::string initFile);
	GaGeCard_interface(LPCTSTR initFile); // Constructor from initialisation file

	~GaGeCard_interface(); // Destructor

	int32 InitializeAndConfigure();  // Calls the function to init driver and config first avail system

	int32				InitializeDriver(); 
	int32				GetFirstSystem();
	
	int32				ConfigureFromInitFile();
	uInt32				CalculateTriggerCountFromInitFile();

	int32				LoadStmConfigurationFromInitFile();

	int32				InitializeStream();
	
	int32				Commit();							// Actually send params to card hardware 
	int32				StartStreamingAcquisition();		// starts the streaming acqusition 

	int32				RetreiveAcquisitionConfig();		// Retreive from field and populate object variables
	int32				RetreiveTotalRequestedSamples();	// Ask the card how many samples were requested
	int32				RetrieveSystemInfo();				// Queries the board for info
	
	CSHANDLE			GetSystemHandle();
	void				setAcquisitionConfig(CSACQUISITIONCONFIG acqConf);
	CSACQUISITIONCONFIG getAcquisitionConfig();				// just return values object has.
	void				setSystemInfo(CSSYSTEMINFO sysImfo);
	CSSYSTEMINFO		getSystemInfo();					// just returns the values object has
	void				setStreamComfig(CSSTMCONFIG stmConf);
	CSSTMCONFIG			getStreamConfig();					// just return values object has.

	int32				AllocateStreamingBuffer(uInt16 nCardIndex, uInt32 u32BufferSizeBytes, PVOID* bufferPtr);
	int32				FreeStreamBuffer(void* buffer);

	

	int32				queueToTransferBuffer(void* buffer, uInt32 numSample);		// Tell the card to transfert to the current buffer in double buffering approach
	int32				waitForCurrentDMA(uInt32& u32ErrorFlag, uInt32& u32ActualLength, uInt32& u8EndOfData);				// wait for the current DMA transfer to finish

	BOOL				CleanupFiles();						// Erase files, we should do better
	

	BOOL				isChannelValid(uInt32 u32ChannelIndex, uInt32 u32mode, uInt16 u16cardIndex);

};
