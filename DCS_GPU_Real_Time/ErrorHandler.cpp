// ErrorHandler.cpp
// 
// General error handling functions for GPU \ DCS project
// 
// 
// Jerome Genest
// October 2023
//
// kept the old compuscope DisplayErrorString
// currently defining a better error handling scheme



#include <stdio.h>
//#include <string.h>
#include <string>
#include <iostream>

#include "CsPrototypes.h"
#include "CsTchar.h"

#include "ErrorHandler.h"

static BOOL g_bSuccess = TRUE;



void DisplayErrorString(const int32_t i32Status)
{
	TCHAR	szErrorString[255];
	if (CS_FAILED(i32Status))
	{
		g_bSuccess = FALSE;
	}

	CsGetErrorString(i32Status, szErrorString, 255);
	_ftprintf(stderr, _T("\n%s\n"), szErrorString);
}


// Overloading, this is for Compuscope errors

int ErrorHandler(const char error_string[255],const int32_t i32Status)
{
	TCHAR	szErrorString[255];
	std::string	totErrorString = error_string;
	error_type error_level = NO_ERROR_;

	if (CS_FAILED(i32Status))
	{
		error_level = ERROR_;
	}

	CsGetErrorString(i32Status, szErrorString, 255);

	totErrorString = totErrorString + ": " + szErrorString;

	return ErrorHandler(i32Status, totErrorString.c_str(), error_level);
}


int ErrorHandler(const int32_t error_number, const char error_string[255], error_type error_level)
{
	int returnValue = 0;	// The return value is used for legacy comp with original code and to allow for lighter reading (1 == thera was an eror)

	switch (error_level)
	{
	case NO_ERROR_:
		// Do nothing
		break;
	case MESSAGE_:
		std::cout << "Message :" << error_string << "\n";
		break;
	case WARNING_:
		std::cout << "Warning :" << error_string << "\n";
		break;
	case ERROR_:
		std::cout << "Error (" << error_number << ") ";
		throw std::exception(error_string);
		returnValue = 1;
		break;
	case FATAL_:
		std::cout << "Fatal Error (" << error_number << ") ";
		throw std::exception(error_string);
		returnValue = 1;
		break;  
	}

	return returnValue;

		// eventally add throw statements here ?
		// the idea would be to perform error checking here also

}