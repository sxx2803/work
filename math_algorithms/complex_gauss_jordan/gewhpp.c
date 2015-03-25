/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					           				    */ 
/* Homework 6,                                                      */ 
/*                                                                  */ 
/* Gaussian Elimination - Performs Gaussian elimination on a	    */
/* mxn matrix, physically swaps the memory contents of the rows     */
/********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "Timers.h"

/* Declare the matrix content type */
typedef double complex dMatrix;

/* Prints the matrix */
void printMatrix(dMatrix** matrix, int row, int col){
	int i, j;

	/* Print the matrix for output */
	for(i = 1; i <= row; i++){
		for(j = 1; j <= col; j++){
			printf("%f+%fi \t", creal(matrix[i][j]), cimag(matrix[i][j]));
		}
		printf("\n");
	}
}

/* Finds the pivot row. Returns the pivot row index */
int findPivotRow(dMatrix **matrix, int row, int pivotCol, int startRow){
	/* Loop counter */
	int i;
	/* Init pivot row */
	int pivotRow = 1;
	/* Init max pivot */
	dMatrix maxPivot = 0;

	for(i = startRow; i <= row; i++){
		/* Check if current pivot is greater than max pivot */
		if(cabs(matrix[i][pivotCol]) > cabs(maxPivot)){
			/* Set new max pivot */
			maxPivot = matrix[i][pivotCol];
			/* Set new pivot row index */
			pivotRow = i;
		}
	}
	return pivotRow;
}

/* Swaps two rows in a matrix by physically swapping memory contents */
void swapRows(dMatrix **matrix, int rowToSwap1, int rowToSwap2, int col){
	/* Loop counter */
	int j;
	for(j = 1; j <= col; j++){
		/* Temporary variable to hold element */
		dMatrix temp = matrix[rowToSwap1][j];
		/* Perform swap */
		matrix[rowToSwap1][j] = matrix[rowToSwap2][j];
		matrix[rowToSwap2][j] = temp;
	}
}
 
/* Performs Gaussian elimination with partial pivoting on the input vector. */
/* Returns the pivot vector */
int* gepp(dMatrix **matrix, dMatrix** lowerTriag, dMatrix** upperTriag, dMatrix* augment, int row, int col){
	/* Variables used to store data */
	int k, i, j;
	/* Pivot row */
	int pivotRow;
	/* Pivot vector */
	int* pivotVector;
	/* n = row */
	int n;
	/* Temp var */
	int temp;
	/* The pivot */
	dMatrix pivot;

	/* Initial pivot row is first row */
	pivotRow = 1;
	/* Pivot vector intialized to [1,2...n] */
	pivotVector = (int*) malloc(row*sizeof(int));
	for(i = 0 ; i < row; i++){
		pivotVector[i] = (i+1);
	}
	n = row;
	/* k is "pass" index */
	for(k = 1; k <= n-1; k++){
		/* Get the pivot row */
		pivotRow = findPivotRow(matrix, row, k, k);
		/* Swap only if not the same */
		if(pivotRow != k){
			swapRows(matrix, k, pivotRow, (col));
			temp = pivotVector[(k-1)];
			pivotVector[(k-1)] = pivotVector[(pivotRow-1)];
			pivotVector[(pivotRow-1)] = temp;
		}
		/* Get the pivot */
		pivot = matrix[k][k];
		/* Scale all entires below k-th pivot */
		for(i = (k+1); i <= row; i++){
			matrix[i][k] = matrix[i][k] / pivot;
		}
		/* Zero all entries below k-th pivot */
		for(i = (k+1); i <= row; i++){
			for(j = (k+1); j <=(col); j++){
				matrix[i][j] = matrix[i][j] - (matrix[i][k]*matrix[k][j]);
			}
		}
	}
	/* Get the L and U matrices from the final matrix */
	for(i = 1; i <= row; i++){
		for(j = 1; j <= (col-1); j++){
			if(i>j && (i<col)){
				lowerTriag[i][j] = matrix[i][j];
			}
			else{	
				if(j == (col-1)){
					augment[i] = matrix[i][j+1];
				}
				if(i<col)
					upperTriag[i][j] = matrix[i][j];
			}
		}
	}
	return pivotVector;
}

/* Performs backward substitution */
void backSub(dMatrix** U, dMatrix* augment, int row, int col){
	int r;
	int i, j;
	r = row;
	/* num of rows must be less than num of col to perform backward sub */
	while(r>=col){
		r--;
	}
	/* j is row index */
	for(j = r; j >= 1; j--){
		/* select a pivot */
		augment[j] = augment[j] / U[j][j];
		/* do math */
		for(i = 1; i <= (j-1); i++){
			augment[i] = augment[i]-(augment[j]*U[i][j]);
		}
	}
}

void forwardSub(dMatrix** L, dMatrix* augment, int row, int col){
	int i, j;
	for(j = 1; j <= row-1; j++){
		augment[j] = augment[j]/(L[j][j]);
		for(i = (j+1); j <= row; j++){
			augment[i] = augment[i]-(augment[j]*L[i][j-1]);
		}
	}
}

/* Main function. Gets input from a file and performs gaussian elimination to find */
/* all solutions to a system of equations, if solutions exist */
int main(int argc, char* argv[]){

	/* Declare all necessary variables */

	/* Line ina file */
	char line[500];
	/* String token */
	char *token;
	/* Matrix size */
	int row;
	int col;
	/* Real and imag part of complex */
	double re, im;
	/* Char to store extra chars */
	char ch;
	/* Pivot vector */
	int* pivotVector;
	/* Data input file */
	FILE *file;

	/* Matrix to work with */
	dMatrix **matrix;
	/* Lower triang matrix */
	dMatrix **L;
	/* Upper triag matrix */
	dMatrix **U;
	/* Augment */
	dMatrix *augment;

	/* Permutation matrix */
	int** perm;
	/* Loop variables */
	int i, j;

	/* Declare the timer */
	DECLARE_TIMER(FunctionTimer);

	/* Initialize matrix size */
	row = 0;
	col = 1;
	
	/* Check for right num of arguments */
	if(argc==2){
		/* Start the timer */
		START_TIMER(FunctionTimer);

		/* Open the file */
		file = fopen(argv[1], "r");

		/* Check if file open successful */
		if(file != NULL){
			/* Get the matrix size from file */
			fgets(line, sizeof line, file);
			token = strtok(line, " \t\n");
			while(token != NULL){
				col += (row > 0) ? atoi(token): 0;
				row += (row == 0) ? atoi(token): 0;
				token = strtok(NULL, " \t\n");
			}
			printf("%i x %i matrix\n", row, col);
			/* Allocate memory for the matrices */
			matrix = calloc(row, sizeof(dMatrix*));
			L = calloc(row, sizeof(dMatrix*));
			U = calloc(row, sizeof(dMatrix*));
			augment = calloc(row, sizeof(dMatrix));
			perm = calloc(row, sizeof(int*));

			/* Calculate offset */
			--matrix;
			--L;
			--U;
			--augment;
			--perm;

			/* Initialize matrices to 0s */
			for(j = 1; j<=col; j++){
				matrix[j] = calloc(col, sizeof(dMatrix));
				perm[j] = calloc(row, sizeof(int));
				--perm[j];
				--matrix[j];
			}
			for(j = 1; j<col; j++){
				L[j] = calloc((col-1), sizeof(dMatrix));
				U[j] = calloc((col-1), sizeof(dMatrix));
				--L[j];
				--U[j];
			}
			/* Initialize the matrix from data */
			for(i = 1; i <= row; i++){
				for(j = 1; j <= col; j++){
					fscanf(file, "%lf %lf %c", &re, &im, &ch);
					matrix[i][j] = re + im*I;
					if(i==j && (i<=(col-1))){
						L[i][j] = 1;
					}
				}
			}
			/* Print the original matrix */
			printMatrix(matrix, row, col);
			/* Print the new matrix after gaussian elimination */
			printf("New matrix: \n");
			pivotVector = gepp(matrix, L, U, augment, row, col);
			printMatrix(matrix, row, col);
			/* Print the pivot vector */
			printf("Pivot vector: \n");
			/* Check if the matrix is solveable */
			if(col > (row+1)){
				/* Matrix has infinite solutions */
				fprintf(stderr, "Error: Infinite solutions. Outputting L, U, and P matrix as-is\n");
			}
			else if(row+1==col || (matrix[col][col-1]==0 && matrix[col][col]==0)){
				/* Perform back substitution */
				backSub(U, augment, row, col);
				/* Print the answer */
				printf("Answer is: \n");
				for(i = 1; i <= (col-1); i++){
					printf("%f+%fi\n", creal(augment[i]), cimag(augment[i]));
				}	
			}
			else{
				/* Matrix has no solutions */
				fprintf(stderr, "Error: Matrix has no solutions. Outputting L, U, and P matrix as-is\n");
			}
			/* Print the lower and upper triangular matrices */
			printf("Lower triangular: \n");
			printMatrix(L, (col-1), (col-1));
			printf("Upper triangular: \n");
			printMatrix(U, (col-1), (col-1));
			/* Create and print the permutation matrix */
			printf("Permutation matrix: \n");
			for(i = 1; i <= row; i++){
				perm[i][pivotVector[(i-1)]] = 1;
			}
			for(i = 1; i <= row; i++){
				for(j = 1; j <= row; j++){
					printf("%i\t", perm[i][j]);
				}
				printf("\n");;
			}
			/* Stop and print the timer */
			STOP_TIMER(FunctionTimer);
			PRINT_TIMER(FunctionTimer)
			return 0;
		}
		else{
			/* Error opening file */
			fprintf(stderr, "Error opening file: %s\n", argv[1]);
			return 2;
		}
	}
	else{
		/* Wrong num of arguments */
		fprintf(stderr, "Usage: hw6 filename\n");
		return 1;
	}
	return 3;
}