/************************************************************************/
// Author: Sicheng Xu
// Date: August 9, 2014
// Course: 0306-724 - High Performance Architectures
//
// File: GJ_cpu.cpp
// The purpose of this program is to compute Gauss Jordan elim on the CPU
/************************************************************************/

#include "GJ_common.h"
#include <iostream>
#include <cstring>

/* Prints out the matrix operation. (2D matrix) */
void displayResults(float** M, int numRows, int numCols)
{
	for(int row = 0; row < numRows; row++){
		for(int col = 0; col < numCols; col++){
			std::cout << M[row][col] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

/* Prints out the matrix operation. (1D matrix) */
void displayResults1D(float* M, int numRows, int numCols)
{
	for(int row = 0; row < numRows; row++){
		for(int col = 0; col < numCols; col++){
			std::cout << M[row*numCols+col] << " \t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

/* Swaps rows row1 and row2 in the matrix M*/
void swapRows(float**M, int row1, int row2){
	float* tempPtr = M[row1];
	M[row1] = M[row2];
	M[row2] = tempPtr;
}


/**
 * Computes the CPU Gaussian algorithm
 * matrix - Input matrix to reduce
 * numberOfRows - number of rows
 * numberOfColumns - number of columns
 * outputMatrix - output matrix where the result is stored
 * partialPivot - flag to perform partial pivoting
 */
void GaussianEliminationCPU( float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot ){
	// To begin, copy the input vector into the output matrix
	for(int i = 0; i < numberOfRows; i++){
		std::memcpy(outputMatrix[i], matrix[i], numberOfColumns*sizeof(float));
	}
	
	// Do a loop over the rows of the matrix
	for(int row = 0; row < numberOfRows; row++){
		// Get the current row
		float *scaleRow = outputMatrix[row];
		// Check if the current row's scaling element is the largest between cur row and all rows underneath it
		// BUT ONLY IF PARTIAL PIVOT IS TRUE
		if(partialPivot){
			int largestIndex = row;
			for(int i = row; i < numberOfRows; i++){
				float *comparisonRow = outputMatrix[i];
				if(comparisonRow[row] > outputMatrix[largestIndex][row]){
					largestIndex = i;
				}
			}
			scaleRow = outputMatrix[largestIndex];
			swapRows(outputMatrix, row, largestIndex);
		}
		// Normalize leading coefficient to 1
		float scaleCoeff = scaleRow[row];
		// If the leading coefficient is not 1, scale row such that leading coeff is 1
		if(scaleCoeff != 1){
			for(int i = 0; i < numberOfColumns; i++){
				scaleRow[i] = scaleRow[i] / scaleCoeff;
			}
			scaleCoeff = 1;
		}
		// Go through each of the rows and scale and subtract
		for(int i = 0; i < numberOfRows; i++){
			// Don't do anything for current row
			if(i == row){
				continue;
			}
			float *curRow = outputMatrix[i];
			float curItem = curRow[row];
			// Keep track of the scaling factor for the row...
			float scaleFactor = curItem / scaleCoeff;
			for(int passCol = 0; passCol < numberOfColumns; passCol++){
				float newVal = curRow[passCol] - (scaleRow[passCol] * scaleFactor);
				curRow[passCol] = newVal;
			}
		}
	}
}
