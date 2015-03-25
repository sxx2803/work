/*
J Nicolas Schrading jxs8172@rit.edu
Sicheng Xu			sxx2803@rit.edu
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <device_launch_parameters.h>
#include "hog.h"

// how many elements in the kernel
const int KERNEL_SIZE = 3;
// how large of a tile to use in GPU kernel code
const int TILE_SIZE = 16;
// the size of each cell (8x8 pixels)
const int CELL_SIZE = 8;
// the number of bins in the histograms
const int HISTO_SIZE = 9;
// how many cells per block (2x2)
const int BLOCK_SIZE = 2;

#define PI 3.141592653589793238463

// angles to determine bins in histograms
__constant__ float bin1 = PI / 9;
__constant__ float bin2 = (2 * PI) / 9;
__constant__ float bin3 = PI / 3;
__constant__ float bin4 = (4 * PI) / 9;
__constant__ float bin5 = (5 * PI) / 9;
__constant__ float bin6 = (2 * PI) / 3;
__constant__ float bin7 = (7 * PI) / 9;
__constant__ float bin8 = (8 * PI) / 9;
__constant__ float bin9 = PI;

// constant convolution kernel for all GPU kernels
__constant__ int KERNEL_D[KERNEL_SIZE];


/**
 * Performs convolution in the X direction on the input matrix using the kernel
 * Input:	in		: 2D input matrix
 *			out		: 2D output matrix
 *			size	: size of square matrix
 */
__global__ void conv1X(int *in, int *out, int size) {

	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);

	out[row*size+col] = 0;
	for(int k = 0; k < KERNEL_SIZE; k++) {
		if((col-k) < 0){
			continue;
		}
		out[row*size+col] += (in[row*size+col-k] * KERNEL_D[k]);
	}
}

/**
 * Performs convolution in the Y direction on the input matrix using the kernel
 * Input:	in		: 2D input matrix
 *			out		: 2D output matrix
 *			size	: size of square matrix
 */
__global__ void conv1Y(int *in, int *out, int size) {

	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);

	out[row*size+col] = 0;
	for(int k = 0; k < KERNEL_SIZE; k++) {
		if((row-k) < 0){
			continue;
		}
		out[row*size+col] += (in[(row-k)*size+col] * KERNEL_D[k]);
	}
}

/**
 * Calculates the histogram values for a single block of the image, but all threads 
 *	calculate at the same time
 * Input:	outX		: vector of values in the block, from the X convolution
 *			outY		: vector of values in the block, from the Y convolution
 *			histograms	: vector of histogram values (9 bins * num_histos)
 *			size		: size of square matrix
 */
__global__ void calculateHistoBlock(int *outX, int *outY, float* histograms, int size) {
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	
	// what cell is this thread in?
	int cellRow = row / CELL_SIZE;
	int cellCol = col / CELL_SIZE;
	
	// what histogram number should we index into based on the cell we are in 
	int histoIdx = ((size / CELL_SIZE) * cellRow) + cellCol;

	// calculate magnitude and orientation of the gradients
	float magnitude = sqrt((float)(outX[row*size + col] * outX[row*size + col] + outY[row*size + col] * outY[row*size + col]));
	float orientation = atan2((float)outY[row*size + col], (float)outX[row*size + col]);
	
	// convert to positive angle if negative
	if(orientation < 0) {
		orientation += PI;
	}

	// add to the correct bin in the correct histogram based on the orientation
	if(orientation <= bin1) {
		atomicAdd(&histograms[histoIdx * HISTO_SIZE], magnitude);
	}
	else if(orientation <= bin2 && orientation > bin1) {
		atomicAdd(&histograms[histoIdx * HISTO_SIZE + 1], magnitude);
	}
	else if(orientation <= bin3 && orientation > bin2) {
		atomicAdd(&histograms[histoIdx * HISTO_SIZE + 2], magnitude);
	}
	else if(orientation <= bin4 && orientation > bin3) {
		atomicAdd(&histograms[histoIdx * HISTO_SIZE + 3], magnitude);
	}
	else if(orientation <= bin5 && orientation > bin4) {
		atomicAdd(&histograms[histoIdx * HISTO_SIZE + 4], magnitude);
	}
	else if(orientation <= bin6 && orientation > bin5) {
		atomicAdd(&histograms[histoIdx * HISTO_SIZE + 5], magnitude);
	}
	else if(orientation <= bin7 && orientation > bin6) {
		atomicAdd(&histograms[histoIdx * HISTO_SIZE + 6], magnitude);
	}
	else if(orientation <= bin8 && orientation > bin7) {
		atomicAdd(&histograms[histoIdx * HISTO_SIZE + 7], magnitude);
	}
	else if(orientation <= bin9 && orientation > bin8) {
		atomicAdd(&histograms[histoIdx * HISTO_SIZE + 8], magnitude);
	}

}

/**
 * Compute HOG on the GPU
 * Input:	size		: the size of the image's dimensions
 *			imgData		: the pixels of the image
 *			kernel		: the kernel to perform convolution
 */
bool  HOGGPU(int size, int* imgData, int* kernel) {
	// Error return value
	cudaError_t status;

	int* outXD;
	int* outYD;
	int* imgDataD;
	// the number of histograms in this image
	const int NUM_HISTOS = (size / CELL_SIZE) * (size / CELL_SIZE);
	float* histogramsD;
	float* histograms;
	histograms = (float*) calloc (NUM_HISTOS * HISTO_SIZE, sizeof(float));

	int bytesData = size*size*sizeof(int);

	status = cudaMalloc((void**)&imgDataD, bytesData);
	status = cudaMalloc((void**)&outXD, bytesData);
	status = cudaMalloc((void**)&outYD, bytesData);
	status = cudaMalloc((void**)&histogramsD, NUM_HISTOS * HISTO_SIZE * sizeof(float));
	if (status != cudaSuccess) {
		std::cout << "Kernel failed mallocing: " << cudaGetErrorString(status) << std::endl;
	}

	// move image onto device
	status = cudaMemcpy(imgDataD, imgData, bytesData, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		std::cout << "Kernel failed moving image data from host to device " << cudaGetErrorString(status) << std::endl;
		return false;
	}

	// move zeroes into histogram array on device
	status = cudaMemcpy(histogramsD, histograms, NUM_HISTOS * HISTO_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		std::cout << "Kernel failed moving histogram data from host to device " << cudaGetErrorString(status) << std::endl;
		return false;
	}

	// move that array into constant memory usable by device
	status = cudaMemcpyToSymbol(KERNEL_D, kernel, sizeof(int)*KERNEL_SIZE);
	if (status != cudaSuccess) {
		std::cout << "Kernel failed memcpy'ing to symbol: " << cudaGetErrorString(status) << std::endl;
		return false;
	}
	
	// a 2d grid of 2d blocks
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float)size / (float)TILE_SIZE), (int)ceil((float)size / (float)TILE_SIZE));
	
	// convolve
	conv1X<<<dimGrid, dimBlock>>>(imgDataD, outXD, size);
	status = cudaGetLastError();
	conv1Y<<<dimGrid, dimBlock>>>(imgDataD, outYD, size);
	status = cudaGetLastError();
	// calculate histograms
	calculateHistoBlock<<<dimGrid, dimBlock>>>(outXD, outYD, histogramsD, size);
	status = cudaGetLastError();

	if (status != cudaSuccess) {
		std::cout << "running kernels failed " << cudaGetErrorString(status) << std::endl;
		return false;
	}

	/** REMOVE AFTER DEBUGGING **/
	//int* outX = new int[size*size];
	//int* outY = new int[size*size];
	/** REMOVE AFTER DEBUGGING **/

	//status = cudaMemcpy(histograms, histogramsD, NUM_HISTOS * HISTO_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	//if (status != cudaSuccess) {
	//	std::cout << "failed copying device to host " << cudaGetErrorString(status) << std::endl;
	//	return false;
	//}
	//cudaMemcpy(outX, outXD, bytesData, cudaMemcpyDeviceToHost);
	//cudaMemcpy(outY, outYD, bytesData, cudaMemcpyDeviceToHost);
	
	//writeImage("confuOutX.raw", outX, size);
	//writeImage("confuOutY.raw", outY, size);

	//writeHisto("histo.raw", histograms, NUM_HISTOS * HISTO_SIZE);

	status = cudaGetLastError();

	// Free device memory
	cudaFree(imgDataD);
	cudaFree(outXD);
	cudaFree(outYD);
	cudaFree(histogramsD);

	// Success
	return true;
}



