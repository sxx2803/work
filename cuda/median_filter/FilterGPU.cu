/************************************************************************/
// Author: Sicheng Xu
// Date: August 9, 2014
// Course: 0306-724 - High Performance Architectures
//
// File: FilterGPU.cpp
// The purpose of this program is to do median filtering on the GPU
/************************************************************************/

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <iostream>
#include "MedianFilter.h"
#include <cmath>

#define TILE_SIZE 16

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

// Bubble sorting on the device
__device__ void DeviceBubbleSort(char* toSort, int n){
	bool swapped;
	do{
		swapped = false;
		for(int i = 1; i < n; i++){
			// If out of order, swap
			if((unsigned char) toSort[i-1] > (unsigned char) toSort[i]){
				char temp = toSort[i-1];
				toSort[i-1] = toSort[i];
				toSort[i] = temp;
				swapped = true;
			}
		}
	} while(swapped);
}

// Non shared median filter
__global__ void NonSharedMemoryKernel(char* image, char* outputImage, int height, int width){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int idx = threadId % width;
	int idy = threadId / width;
	// If thread ID falls out of bounds...
	if(threadId > height*width){
		return;
	}
	// If current thread is for a border pixel, ignore
	if((idx < WINDOW_SIZE / 2) || (idy < WINDOW_SIZE / 2) || (idx >= width-(WINDOW_SIZE/2)) || (idy >= height-(WINDOW_SIZE/2))){
		return;
	}
	char window[WINDOW_SIZE*WINDOW_SIZE];
	int windowIdx = 0;
	for(int k = idy - WINDOW_SIZE/2; k <= idy + WINDOW_SIZE/2; k++){
		for(int l = idx - WINDOW_SIZE/2; l <= idx + WINDOW_SIZE/2; l++){
			window[windowIdx++] = image[ k * width + l ];
		}
	}
	DeviceBubbleSort(window, WINDOW_SIZE*WINDOW_SIZE);
	outputImage[threadId] = window[(WINDOW_SIZE*WINDOW_SIZE)/2];
}

// Shared memory median filter
__global__ void SharedMemoryKernel(char* image, char* outputImage, int height, int width){
	__shared__ char blockPixels[(TILE_SIZE)*(TILE_SIZE)];
	int col = threadIdx.x+blockIdx.x*blockDim.x;
	int row = threadIdx.y+blockIdx.y*blockDim.y;
	int imgCol = (col-(2*(WINDOW_SIZE/2))*blockIdx.x-1);
	int imgRow = (row-(2*(WINDOW_SIZE/2))*blockIdx.y-1);

	// Load into shared memory
	// Skirt of block case
	if(threadIdx.x < WINDOW_SIZE/2 || threadIdx.x >= (blockDim.x - WINDOW_SIZE/2) || threadIdx.y < WINDOW_SIZE/2 || threadIdx.y >= (blockDim.x - WINDOW_SIZE/2)){
		// Outside of image, don't do anything
		if((imgCol < 0) || (imgRow < 0) || (imgCol >= width) || (imgRow >= height)){
			blockPixels[threadIdx.y*blockDim.x+threadIdx.x] = 0;
		}
		// Interior skirt
		else{
			blockPixels[threadIdx.y*blockDim.x+threadIdx.x] = image[imgRow*width+imgCol];
		}
	}
	// Actual image pixels
	else{
		blockPixels[threadIdx.y*blockDim.x+threadIdx.x] = image[imgRow*width+imgCol];
	}
	__syncthreads();

	// If skirt thread, do nothing
	if(threadIdx.x < WINDOW_SIZE/2 || threadIdx.x >= (blockDim.x - WINDOW_SIZE/2) || threadIdx.y < WINDOW_SIZE/2 || threadIdx.y >= (blockDim.x - WINDOW_SIZE/2)){
		return;
	}
	// If border pixel, do nothing
	else if(imgRow < (WINDOW_SIZE/2) || imgCol < (WINDOW_SIZE/2) || imgCol >= (width-(WINDOW_SIZE/2)) || imgRow >= (height-(WINDOW_SIZE/2))){
		return;
	}
	// Else is an actual image pixel
	else{
		// Do median filtering
		// Create a window
		char window[WINDOW_SIZE*WINDOW_SIZE];
		int windowIdx = 0;
		for(int k = threadIdx.y - WINDOW_SIZE/2; k <= threadIdx.y + WINDOW_SIZE/2; k++){
			for(int l = threadIdx.x - WINDOW_SIZE/2; l <= threadIdx.x + WINDOW_SIZE/2; l++){
				// Load pixel into window array to sort
				window[windowIdx++] = blockPixels[k*blockDim.x+l];
			}
		}
		// Sort the window
		DeviceBubbleSort(window, WINDOW_SIZE*WINDOW_SIZE);
		// Assign the median value
		outputImage[imgRow*width+imgCol] = window[(WINDOW_SIZE*WINDOW_SIZE)/2];
	}
	return;
}

//GPU Median Filtering
bool MedianFilterGPU( Bitmap* image, Bitmap* outputImage, bool sharedMemoryUse ){

	// CUDA status
	cudaError_t status;

	// Size of array
	int size = image->Height()*image->Width()*sizeof(char);

	// Input and output arrays
	char* inImage;
	status = cudaMalloc((void**)&inImage, size);
	if(checkForError(status)){
		printf("Error allocating on device\n");
		return false;
	}
	char* outImage;
	status = cudaMalloc((void**)&outImage, size);
	if(checkForError(status)){
		printf("Error allocating on device\n");
		return false;
	}

	// Copy from host to device
	status = cudaMemcpy(inImage, image->image, size, cudaMemcpyHostToDevice);
	if(checkForError(status)){
		printf("Error copying data to device\n");
		return false;
	}
	// Copy original image into output
	status = cudaMemcpy(outImage, inImage, size, cudaMemcpyDeviceToDevice);
	if(checkForError(status)){
		printf("Error copying data to device\n");
		return false;
	}

	if(sharedMemoryUse){
		dim3 dimBlock(TILE_SIZE, TILE_SIZE);
		dim3 dimGrid((int)ceil((double)image->Width()/(TILE_SIZE-(2*(WINDOW_SIZE/2)))), (int)ceil((double)image->Height()/(TILE_SIZE-(2*(WINDOW_SIZE/2)))));

		SharedMemoryKernel<<<dimGrid, dimBlock>>>(inImage, outImage, image->Height(), image->Width());
		status = cudaDeviceSynchronize();
		if(checkForError(status)){
			printf("Error synchronizing on device\n");
			return false;
		}
	}
	else{
		dim3 dimBlock(TILE_SIZE, TILE_SIZE);
		dim3 dimGrid((int)ceil((double)image->Width()/(TILE_SIZE)), (int)ceil((double)image->Height()/(TILE_SIZE)));

		NonSharedMemoryKernel<<<dimGrid, dimBlock>>>(inImage, outImage, image->Height(), image->Width());
		status = cudaDeviceSynchronize();
		if(checkForError(status)){
			printf("Error synchronizing on device\n");
			return false;
		}
	}

	status = cudaGetLastError();
	if(checkForError(status)){
		printf("Error performing median filter on device\n");
		return false;
	}

	// Copy from device to host
	status = cudaMemcpy(outputImage->image, outImage, size, cudaMemcpyDeviceToHost);
	if(checkForError(status)){
		printf("Error copying results to host from device\n");
		return false;
	}

	status = cudaFree(inImage);
	if(checkForError(status)){
		printf("Error freeing memory on device\n");
		return false;
	}
	status = cudaFree(outImage);
	if(checkForError(status)){
		printf("Error freeing memory on device\n");
		return false;
	}

	return true;
}