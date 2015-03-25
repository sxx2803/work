/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					           				    */ 
/* Homework 4,                                                      */ 
/*                                                                  */ 
/* Bisection method: Calculates roots using Bisection			    */
/********************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

/* Evaluate the function and the evaluation count */
double evalFunction(double var, int* evalCount){
	*evalCount += 1;
	return (0.76*var*sin((3*var)/5.2)*tan(var/4.7))+(2.9*cos(var+2.5)*sin(0.39*(1.5+var)));
}

/* Checks if a bracket contains a root */
bool containsRoot(double a, double b){
	bool ret = ((a < 0 && b > 0) || (a > 0 && b < 0)) ? true : false;
	return ret;
}

/* Main function */
int main(int argc, char* argv[]){

	/* Variables to hold the bounds, midpoint, and tolerance */
	double a, b, tol, midpt, fatmid, upperBound, lowerBound, fatUpper, fatLower;

	/* Max iterations to prevent infite loop */
	int maxIterations, curIteration;

	/* Numbe of times the function was evaluated */
	int evalCount;

	/* Initialize the iterations and the evaluation count */
	maxIterations = 1000;
	curIteration = 0;
	evalCount = 0;
	
	/* Check for right number of arguments */
	if(argc == 4){
		/* First argument is "lower" bound */
		a = atof(argv[1]);
		/* Second argument is "upper" bound */
		b = atof(argv[2]);
		/* Third argument is tolerance */
		tol = atof(argv[3]);

		/* Real upper bound */
		upperBound = a > b ? a : b;
		/* Real lower bound */
		lowerBound = a > b ? b : a;

		/* Check if starting bracket contains a root */
		if(!containsRoot(evalFunction(a, &evalCount), evalFunction(b, &evalCount))) {
			fprintf(stderr, "No real root in starting bracket \n");
			return 1;
		}

		fatUpper = evalFunction(upperBound, &evalCount);
		fatLower = evalFunction(lowerBound, &evalCount);

		/* Iterate until root found or program times out */
		while(curIteration++ < maxIterations){
			/* Calculate midpoint */
			midpt = (upperBound+lowerBound) / 2.0;
			/* Calculate function at midpoint */
			fatmid = evalFunction(midpt, &evalCount);
			/* Root is found if difference is less than tol */
			if(fabs(upperBound-lowerBound) < tol){
				printf("Root found at x = %4.11f \n", midpt);
				printf("Function evaluated %i times \n", evalCount);
				printf("Iterated %i times\n", curIteration);
				return 0;
			}
			/* Set a new lower or upper bound */
			if(containsRoot(fatmid, fatUpper)){
				lowerBound = midpt;
			}
			else if(containsRoot(fatLower, fatmid)){
				upperBound = midpt;
			}
		}
		/* Calculation diverged */
		fprintf(stderr, "Calculation diverged. current midpt is %4.11f\n", midpt);
	}
	else{
		/* Wrong number of arguments */
		fprintf(stderr, "Usage: Bisection a b tol \n");
		return 2;
	}
	/* Path of execution not supposed to reach here */
	fprintf(stderr, "Stahp wat r u doin");
	return 3;
}