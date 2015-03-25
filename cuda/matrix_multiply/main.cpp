#include <cstdlib> // malloc(), free()
#include <ctime> // time(), clock()
#include <cmath> // sqrt()
#include <iostream> // cout, stream
#include "common.h"
const int ITERS = 1000;
const int SIZE = 4;

/* Prints out the matrix operation. */
void displayResults(float* M, float* N, float* P)
{
	for (int row = 0; row < SIZE; row++) {
		for (int col = 0; col < SIZE; col++)
			std::cout << M[row * SIZE + col] << " ";
		std :: cout << ((row == 0) ? "* " : " ");
		for (int col = 0; col < SIZE; col++)
			std::cout << N[row * SIZE + col] << " ";
		std :: cout << ((row == 0) ? "= " : " ");
		for (int col = 0; col < SIZE; col++)
			std::cout << P[row * SIZE + col] << "\t";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

/* Entry point for the program. Allocates space for two matrices,
 calls a function to multiply them, and displays the results. */
int main()
{
	// Number of bytes in the matrix.
	int bytes = SIZE * SIZE * sizeof(float);

	// Timing data
	float tcpu, tgpu;
	clock_t start, end;

	// Allocate the three arrays of SIZE x SIZE floats.
	// The element i,j is represented by index (i*SIZE + j)
	float* M = new float[SIZE * SIZE];
	float* N = new float[SIZE * SIZE];
	float* Pcpu = new float[SIZE * SIZE];
	float* Pgpu = new float[SIZE * SIZE];

	// Initialize M and N to random integers
	for (int i = 0; i < SIZE*SIZE; i++) {
		M[i] = (float)(rand() % 10);
		N[i] = (float)(rand() % 10);
	}

	// Multiply the two matrices on the host
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		MatrixMultiplicationCPU(M, N, Pcpu, SIZE);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	// Display the results
	std::cout << "Host Computation took " << tcpu << " ms:" << std::endl;
	displayResults(M, N, Pcpu);

	// Multiply the two matrices on the device
	// Perform one warm-up pass and validate
	bool success = MatrixMultiplicationGPU(M, N, Pgpu, SIZE);
	if (!success) {
		std::cout << "\n * Device error! * \n" << std::endl;
	return 1;
	}

	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		MatrixMultiplicationGPU(M, N, Pgpu, SIZE);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	// Display the results
	std::cout << "Device Computation took " << tgpu << " ms:" << std::endl;
	displayResults(M, N, Pgpu);

	// Compare the results for correctness
	float sum = 0, delta = 0;
	for (int i = 0; i < SIZE*SIZE; i++) {
		delta += (Pcpu[i] - Pgpu[i]) * (Pcpu[i] - Pgpu[i]);
		sum += (Pcpu[i] * Pgpu[i]);
	}
	float L2norm = sqrt(delta / sum);
	std::cout << "Relative error: " << L2norm << "\n" << 
		((L2norm < 1e-6) ? "Passed" : "Failed") << std::endl;

	// Release the matrices
	delete[] M; delete[] N; delete[] Pcpu; delete[] Pgpu;

	// Success
	getchar();
	return 0;
}