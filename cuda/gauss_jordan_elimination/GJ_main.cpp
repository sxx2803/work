/************************************************************************/
// Author: Sicheng Xu
// Date: August 9, 2014
// Course: CMPE-755 - High Performance Architectures
//
// File: GJ_main.cpp
// The purpose of this program is to compare Gaussian Elimination on the
// GPU and CPU.
/************************************************************************/

#include "GJ_common.h"
#include <iostream>
#include <cstring>
#include <ctime>

#define MAXSIZE 4096

/* Computes the L2 error between two matrices */
float computeL2Error(float **M, float **N, int numRows, int numCols){
	float error = 0.0f;
	for(int i = 0; i < numRows; i++){
		for(int j = 0; j < numCols; j++){
			error += (M[i][j] - N[i][j]) * (M[i][j] - N[i][j]);
		}
	}
	return sqrtf(error);
}

int main(){     
	// Allocate the initial maximum size matrix
	float **testMat = new float*[MAXSIZE];
	float **pivotResultMat = new float*[MAXSIZE];
	float **nopivotResultMat = new float*[MAXSIZE];
	float **gpuResultMat = new float*[MAXSIZE];
	srand(33);
	for(int i = 0; i < MAXSIZE; i++){
		testMat[i] = new float[MAXSIZE];
		pivotResultMat[i] = new float[MAXSIZE];
		nopivotResultMat[i] = new float[MAXSIZE];
		gpuResultMat[i] = new float[MAXSIZE];
		for(int j = 0; j < MAXSIZE; j++){
			testMat[i][j] = ((float)rand() / RAND_MAX);
		}
	}

	int ITERS = 50;

	for(int size = 4; size <= 2048; (size = size << 1)){
		clock_t start;
		clock_t end;
		float timeElapsed;

		// Start the CPU timing (with partial pivoting)
		std::cout << "Operating on a " << size << " x " << size << " matrix" << std::endl << std::endl;
		start = clock();
		for(int i = 0; i < ITERS; i++){
			GaussianEliminationCPU(testMat, size, size, pivotResultMat, true);
		}
		end = clock();
		timeElapsed = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

		std::cout << "CPU GJ With Pivot took " << timeElapsed << " ms:" << std::endl << std::endl;

		// Start the CPU timing (without partial pivoting)
		start = clock();
		for(int i = 0; i < ITERS; i++){
			GaussianEliminationCPU(testMat, size, size, nopivotResultMat, false);
		}
		end = clock();
		timeElapsed= (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
		float cpucpuError = computeL2Error(pivotResultMat, nopivotResultMat, size, size);

		std::cout << "CPU GJ No Pivot took " << timeElapsed << " ms:" << std::endl;
		std::cout << "Error: " << cpucpuError << std::endl << std::endl;

		// Start the GPU timing, warm up first doe and check that no errors occur
		bool gpuSuccess = GaussianEliminationGPU(testMat, size, size, gpuResultMat, false);
		if(!gpuSuccess){
			std::cout << "Error performing Gauss-Jordan on the GPU! Skipping to next matrix size" << std::endl;
			std::cout << "---" << std::endl;
			continue;
		}
		start = clock();
		for(int i = 0; i < ITERS; i++){
			gpuSuccess = GaussianEliminationGPU(testMat, size, size, gpuResultMat, false);
		}
		end = clock();
		timeElapsed= (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
		float cpugpuError = computeL2Error(pivotResultMat, gpuResultMat, size, size);

		std::cout << "GPU GJ No Pivot took " << timeElapsed << " ms:" << std::endl;
		std::cout << "Error: " << cpugpuError << std::endl << std::endl;

		std::cout << "---" << std::endl << std::endl;
	}

	getchar();

}
