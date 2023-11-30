#pragma once


#include "CUDA_GPU_Interface.h"
#include <fstream>
#include <sstream>
#include <string>
#include <functional>
#include <typeinfo>
#include <iostream>

// Typedefs

// Configuration structure to do the processing
typedef struct
{
	int N;
	int Nsegments;
	int ptsPerIGM;
	int Nreferences;
	int Nfilters; // Number of filters depending on the number of references
	int Nchannels; // Should be from acquisition card
	int NbytesPerSample;
	const char* inputDataFilename;
	const char* filtersFilename;
	const char* outputDataFilename;
	const char* buffer32DataFilename;
	const char* templateDataFilename;
	int MaxDelayXcorr; // How many delays we want to calculate for xcorr
	int idxStartFirstZPD; // We need the idx of the first ZPD
	int TemplateSize; // We need the width of the template (odd number), should be estimated at init  
	int conjugateCW1_C1; // 0 or 1
	int conjugateCW1_C2; // 0 or 1
	int conjugateCW2_C1; // 0 or 1
	int conjugateCW2_C2; // 0 or 1
	int conjugatePhaseCorrection; // 0 or 1
	int conjugateDfr1; // 0 or 1
	int conjugateDfr2; // 0 or 1
	int nintervalInterpolation; // For find_idx_linear_interpolation, how far can the idx of the point in the new grid be, should be estimated at init
	int* idxchFilters; // Choose which channel each filters will be assigned to
	int unwrap;
	double ptsPerIGMSegment;

}DCSCONFIG, * PDCSCONFIG;

class DCSProcessingHandler
{
private:

	DCSCONFIG			DcsCfg = { 0 };

public:

	DCSProcessingHandler();

	~DCSProcessingHandler();
	void reacDCSConfig(const std::string& configFile);
	void DisplayDCSConfig();
	DCSCONFIG getDcsConfig();
};
