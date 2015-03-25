#include "common.h"

/**
 * Computes the CPU addition algorithm
 * a - Input vector 1
 * b - Input vector 2
 * c - Vector to store the output to
 * size - length of the input vectors
 */
void addVectorCPU( float* a, float* b, float* c, int size ){
	for(int i = 0; i < size; i++){
		c[i] = a[i] + b[i];
	}
}

/**
 * Computes the CPU subtraction algorithm
 * a - Input vector 1
 * b - Input vector 2
 * c - Vector to store the output to
 * size - length of the input vectors
 */
void subtractVectorCPU( float* a, float* b, float* c, int size ){
	for(int i = 0; i < size; i++){
		c[i] = a[i] - b[i];
	}
}

/**
 * Computes the CPU scaling algorithm
 * a - Input vector
 * c - Vector to store the output to
 * scaleFactor - the value to scale all entries of the vector by
 * size - length of the input vectors
 */
void scaleVectorCPU( float* a, float* c, float scaleFactor, int size ){
	for(int i = 0; i < size; i++){
		c[i] = a[i] * scaleFactor;
	}
}

