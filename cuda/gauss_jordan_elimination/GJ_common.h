/************************************************************************/
// Author: Jason Lowden
// Date: March 30, 2012
// Course: 0306-724 - High Performance Architectures
//
// File: Gaussian.h
// The purpose of this program is to compare Gaussian Elimination on the
// GPU and CPU.
/************************************************************************/

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef __GAUSSIAN_H__
#define __GAUSSIAN_H__

/**
 * Computes the CPU Gaussian algorithm
 * matrix - Input matrix to reduce
 * numberOfRows - number of rows
 * numberOfColumns - number of columns
 * outputMatrix - output matrix where the result is stored
 * partialPivot - flag to perform partial pivoting
 */
void GaussianEliminationCPU( float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot );

/**
 * Computes the GPU Gaussian algorithm
 * matrix - Input matrix to reduce
 * numberOfRows - number of rows
 * numberOfColumns - number of columns
 * outputMatrix - output matrix where the result is stored
 * partialPivot - flag to perform partial pivoting
 */
bool GaussianEliminationGPU( float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot);

/**
 * Prints out the matrix operation. 
 */
void displayResults(float** M, int numRows, int numCols);

/** 
 * Prints out the matrix operation. (1D)
 */
void displayResults1D(float* M, int numRows, int numCols);

/**
 * Checks for error from a CUDA function 
 */
bool checkForError( cudaError_t error);


#endif