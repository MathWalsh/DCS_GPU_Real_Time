// ErrorHandler.h
// 
// General error handling functions for GPU \ DCS project
// 
// 
// Jerome Genest
// October 2023
//
// kept the old compuscope DisplayErrorString
// currently defining a better error handling scheme



#pragma once


typedef enum
{
	NO_ERROR_,
	MESSAGE_,
	WARNING_,
	ERROR_,
	FATAL_
} error_type;
// 

void DisplayErrorString(const int32_t i32Status);

// overloaded definitions
int ErrorHandler(const char error_string[255], const int32_t i32Status);		// This is for compuscope errors
int ErrorHandler(const int32_t error_number, const char error_string[255], error_type error_level);