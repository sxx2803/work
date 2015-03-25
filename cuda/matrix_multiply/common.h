#ifndef COMMON_H
#define COMMON_H

// CPU Implementation
void MatrixMultiplicationCPU(float* M, float* N, float* P, int size);
// CUDA Implementation (returns false on failure)
bool MatrixMultiplicationGPU(float* M, float* N, float* P, int size);

#endif