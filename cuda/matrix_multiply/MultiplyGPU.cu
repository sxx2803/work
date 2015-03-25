#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

// GPU Kernel to perform a single inner product
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int size)
{
	// Retrieve our coordinates in the block
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// Temporary result
	float p = 0;
	// Perform inner product
	for (int k = 0; k < size; k++) {
		p += Md[ty * size + k] * Nd[k * size + tx];
	}
	// Write to result
	Pd[ty * size + tx] = p;
}

// C Function to run matrix multiplication kernel
bool MatrixMultiplicationGPU(float* M, float* N, float* P, int size)
{
	// Error return value
	cudaError_t status;

	// Number of bytes in the matrix.
	int bytes = size * size * sizeof(float);

	// Pointers to the device arrays
	float *Md, *Nd, *Pd;

	// Allocate memory on the device to store each matrix
	cudaMalloc((void**) &Md, bytes);
	cudaMalloc((void**) &Nd, bytes);
	cudaMalloc((void**) &Pd, bytes);

	// Copy the host input data to the device
	cudaMemcpy(Md, M, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(Nd, N, bytes, cudaMemcpyHostToDevice);

	// Specify the size of the grid and the size of the block
	dim3 dimBlock(size, size);	// Matrix is contained in a block
	dim3 dimGrid(1, 1);			// Only using a single grid element today

	// Launch the kernel on a size-by-size block of threads
	MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, size);

	// Wait for completion
	cudaThreadSynchronize();

	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) << std::endl;
		cudaFree(Md);
		cudaFree(Nd);
		cudaFree(Pd);
		return false;
	}

	// Retrieve the result matrix
	cudaMemcpy(P, Pd, bytes, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);

	// Success
	return true;
}