/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					           				    */ 
/* Homework 4,                                                      */ 
/*                                                                  */ 
/* Secant method: Calculates roots using Secant method			    */
/********************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

/* Evaluate the function and increase the evaluation count */
double evalFunction(double var, int* evalCount){
	*evalCount += 1;
	return (0.76*var*sin((3*var)/5.2)*tan(var/4.7))+(2.9*cos(var+2.5)*sin(0.39*(1.5+var)));
}

int main(int argc, char* argv[]){
	/* Variables to hold data */
	double xn, xnm1, fxn, fxnm1, tol, xnp1;

	/* Iteration and evaluation counts */
	int curIteration, maxIterations, fevalCount;

	/* Initialize iterations and evaluation counts */
	maxIterations = 1000;
	curIteration = 0;
	fevalCount = 0;

	/* Check for correct amount of arguments */
	if(argc == 4){
		/* Initial guess */
		xn = atof(argv[1]);
		/* Initial previous guess */
		xnm1 = atof(argv[2]);
		/* Tolerance */
		tol = atof(argv[3]);

		/* Loop until program times out or discovers root */
		while(curIteration++ < maxIterations){

			/* Get f(xn_-1) from previous iteration */
			fxnm1 = fxn;
			/* Calculate f(xn) */
			fxn = evalFunction(xn, &fevalCount);
			/* Calculate the next xn */
			xnp1 = xn - fxn*((xn-xnm1)/(fxn-fxnm1));

			/* If current point and next point have less than a tol difference */
			/* a root has been found */
			if(fabs(xn-xnp1) < tol || fabs(fxn) < tol){
				printf("Root found at x = %4.11f \n", xn);
				printf("Function evaluated %i times\n", fevalCount);
				printf("Iterated %i times\n", curIteration);
				return 0;
			}
			/* Set previous Xn to current Xn for next iteration */
			xnm1 = xn;

			/* Iterate */
			xn = xnp1;

		}
		/* Calculation diverged */
		fprintf(stderr, "Error: Calculation diverged\n");
		return 1;
	}
	else{
		fprintf(stderr, "Usage: Secant x0 x1 tol\n");
		return 2;
	}

	fprintf(stderr, "Stahp wat r u doin");
	return 3;
}