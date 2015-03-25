/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					            */ 
/* Homework 1,                                                      */ 
/*                                                                  */ 
/* Linear Curve Fitting - This program fits a line to the data      */
/* points in the file provided on the command line (one point per   */
/* line of text in the file).					    */
/********************************************************************/

/* Arbitrary default list capacity */
#define DEF_LIST_SIZE 1000

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "data.h"

/* Generic type for list */
typedef double Element;

/* A dynamic list structure implementation */
typedef struct{
	/* The data contained in the list */
	Element* data;

	/* The list's current size */
	int currentSize;
	
	/* The list's max size */
	int maxSize;
}list;

/* Initializes a list structure */
void InitList(list* theList){
	/* Allocate memory to the list's data array */
	theList->data = (Element*) malloc(sizeof(Element) * DEF_LIST_SIZE);

	/* Initialize starting size to 0 */
	theList->currentSize = 0;

	/* Initialize max size to specified value */
	theList->maxSize = DEF_LIST_SIZE;
}

/* Destroys a list structure and frees any allocated memory */
void DestroyList(list* theList){
	/* Frees memory allocated to list's data array */
	free(theList->data);

	/* Reset sizes to 0 */
	theList->currentSize = 0;
	theList->maxSize = 0;
}

/* Adds an element to the specified list */
void AddToList(list* theList, Element element){
	/* If list is near capacity, increase list's max capacity */
	if(theList->currentSize >= (theList->maxSize - 1)){
		/* Reallocate the list's data */
		theList->data = (Element*) realloc(theList->data, (sizeof(Element)*(theList->maxSize*2)));

		/* Double the capacity */
		theList->maxSize = 2*theList->maxSize;
	}
	/* Add the element to the list */
	theList->data[theList->currentSize] = element;
	
	/* Increase the list's current size */
	theList->currentSize++;
}

/* Data structure for a linear fit function */
typedef struct{
	/* X data list */
	list Data_X;

	/* Y data list */
	list Data_Y;

	/* The constant 'B' */
	double B;

	/* The coefficient to the linear term 'A' */
	double A;

	/* Flag indiciating that the coefficients have been computed */
	int CoefficientsComputed;
}LinearFit;

/* Initializes a linear fit data structure */
void LFInit(LinearFit* theFit){
	InitList(&(theFit->Data_X));
	InitList(&(theFit->Data_Y));
	theFit->B = 0;
	theFit->A = 0;
	theFit->CoefficientsComputed = 0;
}

/* Adds a point to the linear fit function */
void LFAddPoint(LinearFit* theFit, Element x, Element y){
	AddToList(&(theFit->Data_X), x);
	AddToList(&(theFit->Data_Y), y);
}

/* Returns the number of points in the linear fit function */
int LFGetNumberOfPoints(LinearFit* theFit){
	return (theFit->Data_X).currentSize;
}

/* Computes the linear fit coefficient */
void LFComputeCoefficients(LinearFit* theFit){
	/* Declare and initialize sum variables */
	double S_XX = 0.0;
	double S_XY = 0.0;
	double S_X = 0.0;
	double S_Y = 0.0;

	/* For loop counter */
	int i;

	/* The X and Y points to compute with */
	Element* X_Points = theFit->Data_X.data;
	Element* Y_Points = theFit->Data_Y.data;

	for(i = 0; i < theFit->Data_X.currentSize; i++){
		/* Iterate and calculate corresponding values */
		S_XX += X_Points[i] * X_Points[i];
		S_XY += X_Points[i] * Y_Points[i];
		S_X += X_Points[i];
		S_Y += Y_Points[i];
	}

	/* Compute the constant */
	theFit->B = (((S_XX * S_Y) - (S_XY * S_X)) / ((theFit->Data_X.currentSize * S_XX) - (S_X * S_X)));

	/* Compute the linear coefficient */
	theFit->A = (((theFit->Data_X.currentSize * S_XY) - (S_X * S_Y)) / ((theFit->Data_X.currentSize * S_XX) - (S_X * S_X)));

	/* Indicate that the Coefficients have been computed */
	theFit->CoefficientsComputed = 1;

}

/* Returns the linear fit constant */
double LFGetConstant(LinearFit* theFit){
	if(theFit->CoefficientsComputed == 0){
		LFComputeCoefficients(theFit);
	}

	return theFit->B;
}

/* Deletes a linear fit data structure and frees any allocated memory */
void LFDelete(LinearFit* theFit){
	DestroyList(&(theFit->Data_X));
	DestroyList(&(theFit->Data_Y));
}

/* Returns the linear fit coefficient */
double LFGetLinearCoefficient(LinearFit* theFit){
	if(theFit->CoefficientsComputed == 0){
		LFComputeCoefficients(theFit);
	}

	return theFit->A;
}

/* Main function to calculate a line of best fit*/
int main(int argc, char *argv[]){

	/* Declare a pointer to the LinearFit struct */
	LinearFit * DataSet = (LinearFit*) malloc(sizeof(LinearFit));

	/* Variables to hold the constant and linear coefficients of the line */
	double A, B;
	
	/* Temporary variables to hold data read from file */
	double X, Y;

	/* Check that a command line argument was provided */
	if(argc < 2){

		/* Initialize the LinearFit struct declared earlier*/
		LFInit(DataSet);

		/* While a data point is returned, add it to the list */
		while(DataPoints(&X, &Y) == 1){
			/* Add the data point */
			LFAddPoint(DataSet, X, Y);
		}

		/* Save the constant value */
		A = LFGetLinearCoefficient(DataSet);

		/* Save the linear coefficient */
		B = LFGetConstant(DataSet);

		/* Print out the line that fits the data set */
		printf("The line is: Y = %7.6f * X + %7.5f \n", A, B);
		printf("There were %d points in the data set. \n", LFGetNumberOfPoints(DataSet));

		/* Destroy the data set struct */
		LFDelete(DataSet);
	}
	else{
		/* Display program usage information */
		fprintf(stderr, "Usage: %s \n", argv[0]);
	}

	/* Frees memory allocated to the data set */
	free(DataSet);

	return 0;
}
