
#include"DCSProcessingHandler.h"


DCSProcessingHandler::DCSProcessingHandler()
{

}


void DCSProcessingHandler::reacDCSConfig(const std::string& configFile)
{   // Need to add error handling when not reading a certain parameter
    std::ifstream file(configFile);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << configFile << std::endl;
        return; // or handle the error as needed
    }

    std::string line, tempString;
    std::cout << "Reading Config filename: " << configFile << std::endl;

    while (getline(file, line)) {

        //std::cout << line << std::endl; // Debug print

        std::istringstream iss(line);
        std::string key, equals, tempString;
        if (!(iss >> key >> equals)) continue;
        if (equals != "=") continue;

       /* if (key == "N") {
            iss >> DcsCfg.N;
        }*/
        if (key == "Nsegments") {
            iss >> DcsCfg.Nsegments;
        }
       /* else if (key == "NbytesPerSample") {
            iss >> DcsCfg.NbytesPerSample;
        }*/
        else if (key == "ptsPerIGM") {
            iss >> DcsCfg.ptsPerIGM;
        }
        else if (key == "Nreferences") {
            iss >> DcsCfg.Nreferences;
            DcsCfg.Nfilters = (2 * DcsCfg.Nreferences + 1);
            std::cout << "Nfilters " << DcsCfg.Nfilters << std::endl; // Debug print

            DcsCfg.idxchFilters = new int[DcsCfg.Nfilters];
        }
       /* else if (key == "Nchannels") {
            iss >> DcsCfg.Nchannels;
        }*/
        else if (key == "inputDataFilename") {
            if (iss >> tempString) {
                // You must handle memory allocation and copying here
                // Example:
                DcsCfg.inputDataFilename = _strdup(tempString.c_str());
            }
        }
        else if (key == "filtersFilename") {
            if (iss >> tempString) {
                // Handle memory allocation and copying
                DcsCfg.filtersFilename = _strdup(tempString.c_str());
            }
        }
        else if (key == "outputDataFilename") {
            if (iss >> tempString) {
                // Handle memory allocation and copying
                DcsCfg.outputDataFilename = _strdup(tempString.c_str());
            }
        }
        else if (key == "buffer32DataFilename") {
            if (iss >> tempString) {
                // Handle memory allocation and copying
                DcsCfg.buffer32DataFilename = _strdup(tempString.c_str());
            }
        }
        else if (key == "templateDataFilename") {
            if (iss >> tempString) {
                // Handle memory allocation and copying
                DcsCfg.templateDataFilename = _strdup(tempString.c_str());
            }
        }
        else if (key == "MaxDelayXcorr") {
            iss >> DcsCfg.MaxDelayXcorr;
        }
        else if (key == "idxStartFirstZPD") {
            iss >> DcsCfg.idxStartFirstZPD;
        }
        else if (key == "TemplateSize") {
            iss >> DcsCfg.TemplateSize;
        }
        else if (key == "conjugateCW1_C1") {
            iss >> DcsCfg.conjugateCW1_C1;
        }
        else if (key == "conjugateCW1_C2") {
            iss >> DcsCfg.conjugateCW1_C2;
        }
        else if (key == "conjugateCW2_C1") {
            iss >> DcsCfg.conjugateCW2_C1;
        }
        else if (key == "conjugateCW2_C2") {
            iss >> DcsCfg.conjugateCW2_C2;
        }
        else if (key == "conjugatePhaseCorrection") {
            iss >> DcsCfg.conjugatePhaseCorrection;
        }
        else if (key == "conjugateDfr1") {
            iss >> DcsCfg.conjugateDfr1;
        }
        else if (key == "conjugateDfr2") {
            iss >> DcsCfg.conjugateDfr2;
        }
        else if (key == "nintervalInterpolation") {
            iss >> DcsCfg.nintervalInterpolation;
        }
        else if (key == "idxchFilters") {
            // Read the rest of the line
            std::string filterIndices;
            std::getline(iss, filterIndices);

            // Remove any leading spaces from the rest of the line if necessary
            size_t start = filterIndices.find_first_not_of(" ");
            filterIndices = (start == std::string::npos) ? "" : filterIndices.substr(start);

            std::istringstream filterStream(filterIndices);
            int idx;
            int count = 0;
            // Parse the integers and store them in idxchFilters
            while (filterStream >> idx) {
                if (count < DcsCfg.Nfilters) {
                    DcsCfg.idxchFilters[count] = idx;
                    count++;
                }
                else {
                    // Error: More indices than expected
                    fprintf(stderr, "Error: More filter indices provided than expected.\n");
                    // Handle the error as needed
                    delete[] DcsCfg.idxchFilters; // Don't forget to free allocated memory
                    DcsCfg.idxchFilters = nullptr; // Reset pointer to avoid dangling pointer
                }
            }

            if (count != DcsCfg.Nfilters) {
                // Error: Fewer indices than expected
                fprintf(stderr, "Error: Fewer filter indices provided than expected.\n");
                // Handle the error as needed
                delete[] DcsCfg.idxchFilters; // Don't forget to free allocated memory
                DcsCfg.idxchFilters = nullptr; // Reset pointer to avoid dangling pointer
            }
        }
        else if (key == "unwrap") {
            iss >> DcsCfg.unwrap;
        }
        else if (key == "ptsPerIGMSegment") {
            iss >> DcsCfg.ptsPerIGMSegment;
        }

    }
}
DCSCONFIG DCSProcessingHandler::getDcsConfig()
{
    return DcsCfg;
}



void DCSProcessingHandler::DisplayDCSConfig()
{
    std::cout << "Reading Filters filename: " << DcsCfg.filtersFilename << std::endl;

}


DCSProcessingHandler::~DCSProcessingHandler()
{

}
