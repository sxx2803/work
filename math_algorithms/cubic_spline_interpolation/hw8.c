/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					           				    */ 
/* Homework 8                                                       */ 
/*                                                                  */ 
/* Spline Interpolation - Performs testing of cubic spline          */
/********************************************************************/

#include "interp.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* Main function */
int main(int argc, char* argv[]){
	/* Declare all necessary variables */
	char line[500];
	char *token;
	double clampa, clampb, slope;
	FILE* file;
	int count;
	int i;
	double d1, d2, d3, d4, d5, d6;

	/* Create splines and data set */
	CSplines theSplines;
	Points thePoints;

	/* Initialize coutnt, splines, and data set */
	count = 0;
	thePoints.X = (double*) malloc(5*sizeof(double));
	thePoints.Y = (double*) malloc(5*sizeof(double));
	theSplines.a = (double*) malloc(5*sizeof(double));
	theSplines.b = (double*) malloc(5*sizeof(double));
	theSplines.c = (double*) malloc(5*sizeof(double));
	theSplines.d = (double*) malloc(5*sizeof(double));

	/* Check for correct # of arguments */
	if(argc == 2){
		/* Open file  */
		file = fopen(argv[1], "r");

		/* Check for file open success */
		if(file != NULL){
			/* Process the file */
			fgets(line, sizeof line, file);
			token = strtok(line, " \t\n");
			while(token != NULL){
				clampa = atof(token);
				token = strtok(NULL, " \t\n");
				clampb = atof(token);
				token = strtok(NULL, " \t\n");
			}
			while(fgets(line, sizeof line, file) != NULL){
				token = strtok(line, " \t\n");
				if(token != NULL){
					thePoints.X[count] = atof(token);
					token = strtok(NULL, " \t\n");
					thePoints.Y[count] = atof(token);
					token = strtok(NULL, " \t\n");
					count++;
				}
			}
			/* Not enough points to interpolate */
			if(count<2){
				fprintf(stderr,"err: not enough points to interpolate\n");
				return 3;
			}
			/* Only 2 poinst to interpolate */
			else if(count==2){
				printf("2 points only, performing linear interpolation\n");
				slope = (thePoints.Y[1] - thePoints.Y[0]) / (thePoints.X[1] - thePoints.X[0]);

				printf("[%lf, %lf] %lf %lf %lf %lf\n", thePoints.X[0], thePoints.X[1], slope, (double)0, (double)0, (double)0);
				return 3;
			}
			
			/* Normal interpolation */

			/* Set the values for data structures */
			thePoints.N = count;
			theSplines.N = count;
			thePoints.Size = count;
			thePoints.Next = 0;

			/* Natural boundary testing testing */
			printf("Natural Boundary Results: \n");
			cspline_natural(&thePoints, &theSplines);
			for(i = 0; i < count-1; i++){
				printf("[%lf, %lf] %lf %lf %lf %lf\n", thePoints.X[i], thePoints.X[i+1],theSplines.a[i],theSplines.b[i],theSplines.c[i],theSplines.d[i]);
			}
			d1 = cspline_eval(&thePoints, &theSplines, -1.0);
			d2 = cspline_eval(&thePoints, &theSplines, 0.0);
			d3 = cspline_eval(&thePoints, &theSplines, 2.0);
			d4 = cspline_eval(&thePoints, &theSplines, 1.0);
			d5 = cspline_eval(&thePoints, &theSplines, 1.8);
			d6 = cspline_eval(&thePoints, &theSplines, 3);
			printf("Natural Spline at x = -1.0 is %lf\n", d1);
			printf("Natural Spline at x = 0.0 is %lf\n", d2);
			printf("Natural Spline at x = 2.0 is %lf\n", d3);
			printf("Natural Spline at x = 1.0 is %lf\n", d4);
			printf("Natural Spline at x = 1.8 is %lf\n", d5);
			printf("Natural Spline at x = 3.0 is %lf\n", d6);
			printf("Natural Boundary Results: \n");

			/* Clamped testing */
			cspline_clamped(&thePoints, &theSplines, clampa, clampb);
			for(i = 0; i < count-1; i++){
				printf("[%lf, %lf] %lf %lf %lf %lf\n", thePoints.X[i], thePoints.X[i+1],theSplines.a[i],theSplines.b[i],theSplines.c[i],theSplines.d[i]);
			}

			d1 = cspline_eval(&thePoints, &theSplines, -1.0);
			d2 = cspline_eval(&thePoints, &theSplines, 0.0);
			d3 = cspline_eval(&thePoints, &theSplines, 2.0);
			d4 = cspline_eval(&thePoints, &theSplines, 1.0);
			d5 = cspline_eval(&thePoints, &theSplines, 1.8);
			d6 = cspline_eval(&thePoints, &theSplines, 3);
			printf("Clamped Spline at x = -1.0 is %lf\n", d1);
			printf("Clamped Spline at x = 0.0 is %lf\n", d2);
			printf("Clamped Spline at x = 2.0 is %lf\n", d3);
			printf("Clamped Spline at x = 1.0 is %lf\n", d4);
			printf("Clamped Spline at x = 1.8 is %lf\n", d5);
			printf("Clamped Spline at x = 3.0 is %lf\n", d6);
			printf("Not-a-Knot Results: \n");
			
			/* Not a knot testing */
			cspline_nak(&thePoints, &theSplines);
			for(i = 0; i < count-1; i++){
				printf("[%lf, %lf] %lf %lf %lf %lf\n", thePoints.X[i], thePoints.X[i+1],theSplines.a[i],theSplines.b[i],theSplines.c[i],theSplines.d[i]);
			}
			d1 = cspline_eval(&thePoints, &theSplines, -1.0);
			d2 = cspline_eval(&thePoints, &theSplines, 0.0);
			d3 = cspline_eval(&thePoints, &theSplines, 2.0);
			d4 = cspline_eval(&thePoints, &theSplines, 1.0);
			d5 = cspline_eval(&thePoints, &theSplines, 1.8);
			d6 = cspline_eval(&thePoints, &theSplines, 3);
			printf("Not-a-Knot Spline at x = -1.0 is %lf\n", d1);
			printf("Not-a-Knot Spline at x = 0.0 is %lf\n", d2);
			printf("Not-a-Knot Spline at x = 2.0 is %lf\n", d3);
			printf("Not-a-Knot Spline at x = 1.0 is %lf\n", d4);
			printf("Not-a-Knot Spline at x = 1.8 is %lf\n", d5);
			printf("Not-a-Knot Spline at x = 3.0 is %lf\n", d6);
		}
		else{
			fprintf(stderr, "error opening file: '%s'\n", argv[1]);
		}
	}
	else{
		fprintf(stderr, "usage: hw8 filename\n");
	}
	return 0;
}