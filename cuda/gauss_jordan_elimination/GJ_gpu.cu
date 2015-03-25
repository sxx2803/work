/************************************************************************/
// Author: Sicheng Xu
// Date: August 9, 2014
// Course: CMPE-755 - High Performance Architectures
//
// File: GJ_gpu.cu
// Performs Gauss-Jordan on the GPU
/************************************************************************/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "GJ_common.h"
#include <cmath>

typedef unsigned int uint32;

#define TILE_SIZE 256

__global__ void ScaleKernel(float* matVector, uint32 numberOfRows, uint32 numberOfColumns, float* outMatVector, int curRow){
	// Get the index within the vector
	int vecIdx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	// Outside of domain
	if(vecIdx > (numberOfRows * numberOfColumns)){
		return;
	}
	// Get the row and column number
	int row = vecIdx / numberOfColumns;
	int col = vecIdx % numberOfColumns;
	// Pivot element
	float pivotElement = matVector[row*numberOfColumns+row];
	// Scale the current row
	if(row == curRow && pivotElement != 0.0f){
		outMatVector[vecIdx] = matVector[vecIdx] / pivotElement;
	}
}

__global__ void SubtractKernel(float* matVector, uint32 numberOfRows, uint32 numberOfColumns, float* outMatVector, int curRow){
	// Get the index within the vector
	int vecIdx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	// Outside of domain
	if(vecIdx > (numberOfRows * numberOfColumns)){
		return;
	}	
	// Get the row and column number
	int row = vecIdx / numberOfColumns;
	int col = vecIdx % numberOfColumns;
	// If vector row is not current row, subtract
	if((row != curRow) && (matVector[row*numberOfColumns+curRow] != 0)){
		outMatVector[vecIdx] = matVector[vecIdx] - (matVector[row*numberOfColumns+curRow] * matVector[curRow*numberOfColumns+col]);
	}
}

/**
 * Computes the GPU Gaussian algorithm
 * matrix - Input matrix to reduce
 * numberOfRows - number of rows
 * numberOfColumns - number of columns
 * outputMatrix - output matrix where the result is stored
 * partialPivot - flag to perform partial pivoting
 */
bool GaussianEliminationGPU( float** matrix, uint32 numberOfRows, uint32 numberOfColumns, float** outputMatrix, bool partialPivot){

	// Error return status
	cudaError_t status;
	
	// Total size of matrix shortcut variable
	int size = numberOfColumns * numberOfRows;

	// Number of bytes in vector
	int bytes = numberOfRows * numberOfColumns * sizeof(float);

	// Pointer to input output arrays
	float *inMat, *outMat;

	// Allocate memory
	status = cudaMalloc((void**) &inMat, bytes);
	if(checkForError(status)){
		std::cout << "Cannot allocate CUDA memory" << std::endl;
		return false;
	}
	status = cudaMalloc((void**) &outMat, bytes);
	if(checkForError(status)){
		std::cout << "Cannot allocate CUDA memory" << std::endl;
		return false;
	}

	// Temporarily store the start of the input matrix
	float *inMatPtrStart = inMat;

	// Copy host data into device
	for(uint32 i = 0; i < numberOfRows; i++){
		status = cudaMemcpy(inMat, matrix[i], (numberOfColumns * sizeof(float)), cudaMemcpyHostToDevice);
		if(checkForError(status)){
			std::cout << "Cannot copy from host to device" << std::endl;
			return false;
		}
		// Increment pointer to next "row"
		inMat += numberOfColumns;
	}	
	
	// Restore original pointer location
	inMat = inMatPtrStart;

	// Initialize the output matrix vector to original input matrix
	status = cudaMemcpy(outMat, inMat, bytes, cudaMemcpyDeviceToDevice);
	if(checkForError(status)){
		std::cout << "Cannot copy from device to device" << std::endl;
		return false;
	}

	// Using a TILE_SIZE sized block
	dim3 dimBlock(TILE_SIZE);
	int gridLength = (int)ceil(sqrtf((float)size / (float)TILE_SIZE));
	dim3 dimGrid(gridLength, gridLength);
	status = cudaGetLastError();
	if(checkForError(status)){
		std::cout << "Error generating grid or block" << std::endl;
		return false;
	}

	// Perform the Greggory Jesus IX elimination
	for(uint32 row = 0; row < numberOfRows; row++){
		// Scale all rows
		ScaleKernel<<<dimGrid, dimBlock>>>(inMat, numberOfRows, numberOfColumns, outMat, row);
		status = cudaGetLastError();
		if(checkForError(status)){
			std::cout << "Scaling error" << std::endl;
			return false;
		}
		cudaDeviceSynchronize();
		// Update the input matrix
		status = cudaMemcpy(inMat, outMat, bytes, cudaMemcpyDeviceToDevice);
		if(checkForError(status)){
			std::cout << "Cannot update input matrix" << std::endl;
			return false;
		}
		// Perform subtractions
		SubtractKernel<<<dimGrid, dimBlock>>>(inMat, numberOfRows, numberOfColumns, outMat, row);
		status = cudaGetLastError();
		if(checkForError(status)){
			std::cout << "Subtraction error" << std::endl;
			return false;
		}
		cudaDeviceSynchronize();
		// Update the input matrix
		status = cudaMemcpy(inMat, outMat, bytes, cudaMemcpyDeviceToDevice);
		if(checkForError(status)){
			std::cout << "Cannot update input matrix" << std::endl;
			return false;
		}
	}
	
	// Copy results back out of device
	for(unsigned int i = 0; i < numberOfRows; i++){
		status = cudaMemcpy(outputMatrix[i], outMat, (numberOfColumns * sizeof(float)), cudaMemcpyDeviceToHost);
		if(checkForError(status)){
			std::cout << "Cannot copy from device to host" << std::endl;
			return false;
		}
		outMat += numberOfColumns;
	}

	return true;

}