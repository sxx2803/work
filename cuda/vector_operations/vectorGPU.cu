#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

// Define the tile size
#define TILE_SIZE 512

__global__ void VectorAddKernel(float* a, float* b, float* c, int size){
	// Get the thread coordinates
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int vecIdx  = blockId * blockDim.x + threadIdx.x;
	if(vecIdx < size){
		c[vecIdx] = a[vecIdx] + b[vecIdx];
	}
}

__global__ void VectorSubKernel(float* a, float* b, float* c, int size){
	// Get the thread coordinates
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int vecIdx  = blockId * blockDim.x + threadIdx.x;
	if(vecIdx < size){
		c[vecIdx] = a[vecIdx] - b[vecIdx];
	}
}

__global__ void VectorScaleKernel(float* a, float* c, float scaleFactor, int size){
	// Get the thread coordinates
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int vecIdx  = blockId * blockDim.x + threadIdx.x;
	if(vecIdx < size){
		c[vecIdx] = a[vecIdx] *scaleFactor;
	}
}

/**
 * Computes the GPU addition algorithm
 * a - Input vector 1
 * b - Input vector 2
 * c - Vector to store the output to
 * size - length of the input vectors
 */
bool addVectorGPU( float* a, float* b, float* c, int size ){

	// Error return value
	cudaError_t status;

	// Number of bytes in the vector
	int bytes = size * sizeof(float);

	// Pointers to input output arrays
	float *ad, *bd, *cd;

	// Allocate memory
	status = cudaMalloc((void**) &ad, bytes);
	if(checkForError(status)){
		std::cout << "CUDA Memory Alloc Error!" << std::endl;
		return false;
	}
	status = cudaMalloc((void**) &bd, bytes);
	if(checkForError(status)){
		std::cout << "CUDA Memory Alloc Error!" << std::endl;
		return false;
	}
	status = cudaMalloc((void**) &cd, bytes);
	if(checkForError(status)){
		std::cout << "CUDA Memory Alloc Error!" << std::endl;
		return false;
	}

	// Copy host data into device
	status = cudaMemcpy(ad, a, bytes, cudaMemcpyHostToDevice);
	if(checkForError(status)){
		std::cout << "Device Error!" << std::endl;
		return false;
	}
	status = cudaMemcpy(bd, b, bytes, cudaMemcpyHostToDevice);
	if(checkForError(status)){
		std::cout << "Device Error!" << std::endl;
		return false;
	}

	dim3 dimBlock(TILE_SIZE);
	int gridLength = (int)ceil(sqrtf((float)size / (float)TILE_SIZE));
	dim3 dimGrid(gridLength, gridLength);


	//dim3 dimGrid((int)ceil((float)size / (float) TILE_SIZE));

	VectorAddKernel<<<dimGrid, dimBlock>>>(ad, bd, cd, size);

	// Wait for completion
	cudaThreadSynchronize();

	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) << std::endl;
		cudaFree(ad);
		cudaFree(bd);
		cudaFree(cd);
		return false;
	}

	// Get the vector results
	status = cudaMemcpy(c, cd, bytes, cudaMemcpyDeviceToHost);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	// Free the memory on device
	status = cudaFree(ad);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	status = cudaFree(bd);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	status = cudaFree(cd);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}

	// Success?
	return true;

}

/**
 * Computes the GPU addition algorithm
 * a - Input vector 1
 * b - Input vector 2
 * c - Vector to store the output to
 * size - length of the input vectors
 */
bool subtractVectorGPU( float* a, float* b, float* c, int size ){

	// Error return value
	cudaError_t status;

	// Number of bytes in the vector
	int bytes = size * sizeof(float);

	// Pointers to input output arrays
	float *ad, *bd, *cd;

	// Allocate memory
	status = cudaMalloc((void**) &ad, bytes);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	status = cudaMalloc((void**) &bd, bytes);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	status = cudaMalloc((void**) &cd, bytes);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}

	// Copy host data into device
	status = cudaMemcpy(ad, a, bytes, cudaMemcpyHostToDevice);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	status = cudaMemcpy(bd, b, bytes, cudaMemcpyHostToDevice);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}

	dim3 dimBlock(TILE_SIZE);
	int gridLength = (int)ceil(sqrtf((float)size / (float)TILE_SIZE));
	dim3 dimGrid(gridLength, gridLength);


	//dim3 dimGrid((int)ceil((float)size / (float) TILE_SIZE));

	VectorSubKernel<<<dimGrid, dimBlock>>>(ad, bd, cd, size);

	// Wait for completion
	cudaThreadSynchronize();

	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) << std::endl;
		cudaFree(ad);
		cudaFree(bd);
		cudaFree(cd);
		return false;
	}

	// Get the vector results
	status = cudaMemcpy(c, cd, bytes, cudaMemcpyDeviceToHost);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	// Free the memory on device
	status = cudaFree(ad);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	status = cudaFree(bd);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	status = cudaFree(cd);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}

	// Success?
	return true;

}

/**
 * Computes the GPU scaling algorithm
 * a - Input vector
 * c - Vector to store the output to
 * scaleFactor - the value to scale all entries of the vector by
 * size - length of the input vectors
 */
bool scaleVectorGPU( float* a, float* c, float scaleFactor, int size ){

	// Error return value
	cudaError_t status;

	// Number of bytes in the vector
	int bytes = size * sizeof(float);

	// Pointers to input output arrays
	float *ad, *cd;

	// Allocate memory
	status = cudaMalloc((void**) &ad, bytes);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	status = cudaMalloc((void**) &cd, bytes);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}

	// Copy host data into device
	status = cudaMemcpy(ad, a, bytes, cudaMemcpyHostToDevice);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}

	dim3 dimBlock(TILE_SIZE);
	int gridLength = (int)ceil(sqrtf((float)size / (float)TILE_SIZE));
	dim3 dimGrid(gridLength, gridLength);


	//dim3 dimGrid((int)ceil((float)size / (float) TILE_SIZE));

	VectorScaleKernel<<<dimGrid, dimBlock>>>(ad, cd, scaleFactor, size);

	// Wait for completion
	cudaThreadSynchronize();

	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) << std::endl;
		cudaFree(ad);
		cudaFree(cd);
		return false;
	}

	// Get the vector results
	status = cudaMemcpy(c, cd, bytes, cudaMemcpyDeviceToHost);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	// Free the memory on device
	status = cudaFree(ad);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}
	status = cudaFree(cd);
	if(checkForError(status)){
		std::cout << "CUDA Device Error!" << std::endl;
		return false;
	}

	// Success?
	return true;
}
