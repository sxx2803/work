#include "GJ_common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>

/** Checks for errors from CUDA return statements */
bool checkForError( cudaError_t error)
{
	// Check if the status is an error message
	if(error != cudaSuccess)
	{
		// Print out the error message
		printf("CUDA Error: %s\n", cudaGetErrorString(error));
		return true;
	}
	// Otherwise, no error occured
	else
	{
		return false;
	}
}