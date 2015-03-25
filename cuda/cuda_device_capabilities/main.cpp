/**
 * file: main.cpp
 *
 * Description: This program will print out the device parameters of the CUDA
 * 				capable cards onboard the system.
 *
 * Author: Sicheng Xu
 * Date: September 5, 2014
 */
 
// Include statements
#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "common.h"

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

// Entry point for the program.
int main()
{
	//Keep the error status
	cudaError_t status;

	// Device count, driver version, and runtime version
	int devCount, driverVersion, runtimeVersion;
	// Struct for storing properties of the CUDA device
	cudaDeviceProp devProperties;

	// Get the device count and check the return status
	status = cudaGetDeviceCount(&devCount);
	// If there was an error, print the error message, exit and return an error code
	if(checkForError(status))
	{
		return 1;
	}
	// Get the driver version
	status = cudaDriverGetVersion(&driverVersion);
	// If there was an error, print the error message, exit and return an error code
	if(checkForError(status))
	{
		return 1;
	}
	// Get the runtime version
	status = cudaRuntimeGetVersion(&runtimeVersion);
	// If there was an error, print the error message, exit and return an error code
	if(checkForError(status))
	{
		return 1;
	}

	// Print the relevant data
	printf("CUDA Device Capabilities:\n\n");
	printf("CUDA Devices Found: %i\n", devCount);
	printf("CUDA Driver: %i\n", driverVersion);
	printf("CUDA Runtime: %i\n\n", runtimeVersion);

	// Cycle through the CUDA cards if applicable
	for(int i = 0; i < devCount; i++){
		// Get the device properties and store it in a struct
		status = cudaGetDeviceProperties(&devProperties, i);
		// If there was an error, print the error message, exit and return an error code
		if(checkForError(status))
		{
			return 1;
		}
		// Obtain discrete or integrated value
		char* discreteIntegrated = (devProperties.integrated == 1) ? "Integrated" : "Discrete";
		// Obtain name of device
		printf("Device %i: %s (%s)\n", i, devProperties.name, discreteIntegrated);
		// Obtain CUDA capability
		printf("CUDA Capability %i.%i\n", devProperties.major, devProperties.minor);
		printf("Processing:\n");
		// Obtain # of multiprocessors
		printf("\tMultiprocessors: %i\n", devProperties.multiProcessorCount);
		// Obtain max grid size
		printf("\tMax Grid Size: %i x %i x %i\n", devProperties.maxGridSize[0], devProperties.maxGridSize[1], devProperties.maxGridSize[2]);
		// Obtain max block size
		printf("\tMax Block Size: %i x %i x %i\n", devProperties.maxThreadsDim[0], devProperties.maxThreadsDim[1], devProperties.maxThreadsDim[2]);
		// Obtain max threads per block
		printf("\tThreads per Block: %i\n", devProperties.maxThreadsPerBlock);
		// Obtain max threads per multiprocessor
		printf("\tThreads per Multiprocessor: %i\n", devProperties.maxThreadsPerMultiProcessor);
		// Obtain warp size
		printf("\tWarp Size: %i\n", devProperties.warpSize);
		// Obtain core clock rate
		printf("\tClock Rate %4.3f GHz\n", ((float) devProperties.clockRate) / (1000000.0f));
		printf("Memory\n");
		// Obtain total global memory size
		printf("\tGlobal: %i MB \n", devProperties.totalGlobalMem >> 20);
		// Obtain constant memory size
		printf("\tConstant: %i KB\n", devProperties.totalConstMem >> 10);
		// Obtain shared per block size
		printf("\tShared/blk: %i KB\n", devProperties.sharedMemPerBlock >> 10);
		// Obtain number of registers
		printf("\tRegisters/blk: %i\n", devProperties.regsPerBlock);
		// Obtain maximum pitch size
		printf("\tMaximum Pitch: %i MB\n", devProperties.memPitch >> 20);
		// Obtain texture alignment size
		printf("\tTexture Alignment: %i B\n", devProperties.textureAlignment);
		// Obtain L2 cache size
		printf("\tL2 Cache Size: %i B\n", devProperties.l2CacheSize);
		// Obtain memory clock rate
		printf("\tClock Rate: %i MHz\n", devProperties.memoryClockRate/1000);

		// Obtain concurrent copy and execute characteristics
		char* concurrentYesNo = (devProperties.asyncEngineCount != 0) ? "Yes" : "No";
		printf("Concurrent Copy & Execute: %s\n", concurrentYesNo);

		// Obtain kernel time limit characteristics
		char* timeLimitYesNo = (devProperties.kernelExecTimeoutEnabled == 1) ? "Yes" : "No";
		printf("Kernel Time Limit: %s\n", timeLimitYesNo);

		// Obtain page locked memory mapping characteristics
		char* pageLockedMemoryYesNo = (devProperties.canMapHostMemory == 1) ? "Yes" : "No";
		printf("Supports Page-Locked Memory Mapping: %s\n", pageLockedMemoryYesNo);

		// Obtain the computing mode
		switch(devProperties.computeMode){
			// Default computing mode
			case cudaComputeModeDefault:
				printf("Compute mode: Default\n\n");
				break;
			// Exclusive computing mode
			case cudaComputeModeExclusive:
				printf("Compute mode: Exclusive\n\n");
				break;
			// Prohibited computing mode
			case cudaComputeModeProhibited:
				printf("Compute mode: Prohibited\n\n");
				break;
			// Exclusive process mode
			case cudaComputeModeExclusiveProcess:
				printf("Compute mode: Exclusive Process\n\n");
				break;
			default:
				printf("Compute mode: Cannot be found\n\n");
				break;
		}
	}

	// Wait for user input so the console doesn't disappear
	getchar();
	// Return without error
	return 0;
}
