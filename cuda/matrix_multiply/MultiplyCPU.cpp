#include "common.h"

// Multiplies two matrices
void MatrixMultiplicationCPU(float* M, float* N, float* P, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			float sum = 0;
			for (int k = 0; k < size; k++) {
				sum += M[i * size + k] * N[k * size + j];
			}
			P[i * size + j] = sum;
		}
	}
}
