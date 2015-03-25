#include <cstdlib> // malloc(), free()
#include <ctime> // time(), clock()
#include <cmath> // sqrt()
#include <iostream> // cout, stream
#include <fstream>
#include "common.h"
const int ITERS = 1000;

/* Computes the L2 error between two vectors */
float computeL2Error(float *M, float *N, int size){
	float error = 0.0f;
	for(int i = 0; i < size; i++){
		float errTemp = (M[i]-N[i]) * (M[i]-N[i]);
		error += errTemp;
	}
	return sqrtf(error);
}

/* Entry point for the program. Allocates space for some vectors
 calls a function to do math to them, and displays the results. */
int main()
{
	srand(300);
	// Number of elements in the vector
	int size = 1;
	float err;
	bool success = false;

	// Open a text file for writing results
	std::ofstream resultFile("resultsNotPageLocked.txt", std::ios::trunc);

	// Go up to 2^25 because can't physically allocate enough bytes in memory
	// for 4 arrays of 2^26 elements.
	for(int binPower = 0; binPower < 26; binPower++){
		printf("Doing non-pagelocked memory\n");
		printf("Operating on a vector of length %d\n", size);
		resultFile << "Operating on a vector of length " << size << std::endl << std::endl;
		
		// Number of bytes in the matrix
		int numBytes = size*sizeof(float);

		// Timing data
		float tcpuAdd, tgpuAdd, tcpuSub, tgpuSub, tcpuScale, tgpuScale;
		clock_t start, end;

		// Allocate the vectors
		float *M = new float[size];
		float *N = new float[size];
		float *Pcpu = new float[size];
		float *Pgpu = new float[size];

		// Init M and N to random floats between 0 and 1
		for(int i = 0; i < size; i++){
			M[i] = ((float)rand()) / RAND_MAX;
			N[i] = ((float)rand()) / RAND_MAX;
		}

		// Get the random scaling factor
		float scaleFactor = ((float)rand()) / RAND_MAX;

		// Do the CPU implementation for add
		start = clock();
		for(int i = 0; i < ITERS; i ++){
			addVectorCPU(M, N, Pcpu, size);
		}
		end = clock();
		tcpuAdd = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

		// Do the GPU implementation for add, run a warmup first
		success = addVectorGPU(M, N, Pgpu, size);
		if (!success) {
			std::cout << "\n * Device error! * \n" << std::endl;
			return 1;
		}
		start = clock();
		for(int i = 0; i < ITERS; i ++){
			success = addVectorGPU(M, N, Pgpu, size);
			if (!success) {
				std::cout << "\n * Device error! * \n" << std::endl;
				return 1;
			}
		}
		end = clock();
		tgpuAdd = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

		err = computeL2Error(Pcpu, Pgpu, size);

		// Display the results
		std::cout << "Host Computation for Add took " << tcpuAdd << " ms:" << std::endl;
		std::cout << "Device Computation for Add took " << tgpuAdd << " ms:" << std::endl;
		printf("Addition speedup = %f\n", tcpuAdd/tgpuAdd);
		printf("The L2 error for add is: %5.4f\n", err);

		resultFile << "Host Computation for Add took " << tcpuAdd << " ms:" << std::endl;
		resultFile << "Device Computation for Add took " << tgpuAdd << " ms:" << std::endl;
		resultFile << "Addition speedup = " << tcpuAdd/tgpuAdd << std::endl;
		resultFile << "The L2 error for add is: " << err << std::endl << std::endl;
		
		// Do the CPU implementation for sub
		start = clock();
		for(int i = 0; i < ITERS; i ++){
			subtractVectorCPU(M, N, Pcpu, size);
		}
		end = clock();
		tcpuSub = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

		// Do the GPU implementation for sub, warm up first
		success = subtractVectorGPU(M, N, Pgpu, size);
		if (!success) {
			std::cout << "\n * Device error! * \n" << std::endl;
			return 1;
		}
		start = clock();
		for(int i = 0; i < ITERS; i ++){
			success = subtractVectorGPU(M, N, Pgpu, size);
			if (!success) {
				std::cout << "\n * Device error! * \n" << std::endl;
				return 1;
			}
		}
		end = clock();
		tgpuSub = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

		err = computeL2Error(Pcpu, Pgpu, size);

		// Display the results
		std::cout << "Host Computation for Sub took " << tcpuSub << " ms:" << std::endl;
		std::cout << "Device Computation for Sub took " << tgpuSub << " ms:" << std::endl;
		printf("Subtraction speedup = %f\n", tcpuSub/tgpuSub);
		printf("The L2 error for sub is: %5.4f\n", err);

		resultFile << "Host Computation for Sub took " << tcpuSub << " ms:" << std::endl;
		resultFile << "Device Computation for Sub took " << tgpuSub << " ms:" << std::endl;
		resultFile << "Subtraction speedup = " << tcpuSub/tgpuSub << std::endl;
		resultFile << "The L2 error for Sub is: " << err << std::endl << std::endl;
		

		// Do the CPU implementation for scale
		start = clock();
		for(int i = 0; i < ITERS; i ++){
			scaleVectorCPU(M, Pcpu, scaleFactor, size);
		}
		end = clock();
		tcpuScale = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

		// Do the GPU implementation for scale, warm up first
		success = scaleVectorGPU(M, Pgpu, scaleFactor, size);
		if (!success) {
			std::cout << "\n * Device error! * \n" << std::endl;
			return 1;
		}
		start = clock();
		for(int i = 0; i < ITERS; i ++){
			success = scaleVectorGPU(M, Pgpu, scaleFactor, size);
			if (!success) {
				std::cout << "\n * Device error! * \n" << std::endl;
				return 1;
			}
		}
		end = clock();
		tgpuScale = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

		err = computeL2Error(Pcpu, Pgpu, size);

		// Display the results
		std::cout << "Host Computation for Scale took " << tcpuScale << " ms:" << std::endl;
		std::cout << "Device Computation for Scale took " << tgpuScale << " ms:" << std::endl;
		printf("Scaling speedup = %f\n", tcpuScale/tgpuScale);
		printf("The L2 error for scale is: %5.4f\n\n", err);

		resultFile << "Host Computation for Add took " << tcpuScale << " ms:" << std::endl;
		resultFile << "Device Computation for Add took " << tgpuScale << " ms:" << std::endl;
		resultFile << "Scaling speedup = " << tcpuScale/tgpuScale << std::endl;
		resultFile << "The L2 error for add is: " << err << std::endl << std::endl;
		
		resultFile << "-----" << std::endl << std::endl;

		size = size << 1;

		// Release the matrices
		delete[] M; delete[] N; delete[] Pcpu; delete[] Pgpu;
	}

	resultFile.close();
	getchar();
	return 0;

}