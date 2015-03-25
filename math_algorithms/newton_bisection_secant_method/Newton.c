/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					           				    */ 
/* Homework 4,                                                      */ 
/*                                                                  */ 
/* Newton's method: Calculates roots using Newton's method		    */
/********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

/* Evaluate the function and add to the evaluation count */
double evalFunction(double var, int* evalCount){
	*evalCount += 1;
	return (0.76*var*sin((3*var)/5.2)*tan(var/4.7))+(2.9*cos(var+2.5)*sin(0.39*(1.5+var)));
}

/* Evaluate the derivative and add to the derivative count */
double evalDerivative(double var, int* evalCount){
	*evalCount += 1;
	return (-2.9)*sin(.39*var+.585)*sin(var+2.5)+1.131*cos(0.39*var+0.585)*cos(var+2.5)+0.76*sin(0.576923*var)*tan(0.212766*var)+0.438462*var*cos(0.576923*var)*tan(0.212766*var)+0.16702*var*sin(0.576923*var)*(1/(cos(0.212766*var)*cos(0.212766*var)));
}

int main(int argc, char* argv[]){
	double guess, fatguess, tol, dfatguess, newGuess;

	int curIteration, maxIterations, fevalCount, devalCount;

	/* Max iterations to prevent an infinite loop */
	maxIterations = 1000;

	/* Current iteration starts at 0 */
	curIteration = 0;

	/* Number of times the function was evaluated */
	fevalCount = 0;

	/* Number of times the derivative was evaluated */
	devalCount = 0;

	/* Check for correct number of arguments */
	if(argc == 3){
		/* First argument is the initial guess */
		guess = atof(argv[1]);
		/* Second argument is the tolerance */
		tol = atof(argv[2]);

		/* Loop until program times out or root found */
		while(curIteration++ < maxIterations){

			dfatguess = evalDerivative(guess, &devalCount);

			/* Evaluate the function at Xn */
			fatguess = evalFunction(guess, &fevalCount);

			/* Calculate the next point */
			newGuess = guess - (fatguess / dfatguess);
			/* If difference of current point and next point is less than tol, Xn is a root */
			if(fabs(guess-newGuess) < tol){
				printf("Root found at x = %4.11f \n", guess);
				printf("Function evaluated %i times\n", fevalCount);
				printf("Derivative evaluated %i times\n", devalCount);
				printf("Iterated %i times\n", curIteration);
				return 0;
			}
			/* Evaluate derivative at Xn */

			/* Set the next point */
			guess = newGuess;
		}

		/* Calculation diverged */
		fprintf(stderr, "Error: Calculation diverged\n");
		return 1;
	}
	else{
		/* Wrong arguments */
		fprintf(stderr, "Usage: Newton initial_guess tol\n");
		return 2;
	}

	/* Not supposed to be reached */
	fprintf(stderr, "Stahp wat r u doin");
	return 3;
}