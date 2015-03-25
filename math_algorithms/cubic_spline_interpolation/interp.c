/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					           				    */ 
/* Homework 8                                                       */ 
/*                                                                  */ 
/* Interpolation - Functions to use for cubic spline interpolation  */
/********************************************************************/


#include "interp.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

/* Solves tridiagonal and symmetric system */
void tridiagonal(double* c, double* diag, double* superdiag, int n){
	int k;
	double t;
	double* e = superdiag;
	double* d = diag;
	for(k = 1; k <= n-1; k++){
		t = e[k-1];
		e[k-1] = t / d[k-1];
		d[k] = d[k] - (t * e[k-1]);
	}
	for(k = 1; k <= n-1; k++){
		c[k] = c[k] - (e[k-1]*c[k-1]);
	}
	c[k-1] = c[k-1] / d[k-1];
	for(k = n-2; k >= 0; k--){
		c[k] = c[k]/d[k] - e[k]*c[k+1];
	}
}

/* Evaluates spline at point x_val */
extern double cspline_eval(Points* points, CSplines* csplines, double x_val){
	double xval;
	double retVal;
	int n = csplines->N;
	double* a = csplines->a;
	double* b = csplines->b;
	double* c = csplines->c;
	double* d = csplines->d;
	double* x = points->X;

	int i;

	int interval = 0;

	for(i = 0; i < n-1; i++){
		if(x_val >= x[i] && x_val < x[i+1]){
			interval = i;
		}
	}
	if(x_val >= x[n-2]){
		interval = n-2;
	}
	xval = (x_val-x[interval]);
	retVal = a[interval] + (b[interval] * xval) + (c[interval] * pow(xval, 2)) + (d[interval] * pow(xval, 3));
	return retVal;;
}

/* Solves general tridiagonal matrix */
void tridiagonal_gen(double* subdiag, double* diag, double* superdiag, double* c, int n){
	int i;
	double t;
	double* e = superdiag;
	double* d = diag;
	double* l = subdiag;
	for(i = 1; i <= n-1; i++){
		t = l[i-1]/d[i-1];
		d[i] = d[i] - (t*e[i-1]);
		c[i] = c[i] - (t*c[i-1]);
	}
	c[n-1] = c[n-1]/d[n-1];
	for(i = n-2; i >= 0; i--){
		c[i] = ( c[i]-e[i]*c[i+1] ) / d[i];
	}
}

/* Performs not a knot spline */
extern void cspline_nak(Points* points, CSplines* csplines){
	int n = csplines->N;
	double* a = csplines->a;
	double* b = csplines->b;
	double* c = csplines->c;
	double* d = csplines->d;
	double* x = points->X;
	double* y = points->Y;

	int i;
	double* h = (double*) malloc(n*sizeof(double));
	double* diag = (double*) malloc(n*sizeof(double));
	double* superdiag  = (double*) malloc(n*sizeof(double));
	double* subdiag = (double*) malloc(n*sizeof(double));	

	/* Calculate the h_j values */
	for(i = 0; i < n-1; i++){
		h[i] = x[i+1] - x[i];
	}
	/* Calculate the super and sub diagonals of the matrix */
	for(i = 0; i < n-3; i++){
		subdiag[i] = h[i+1];
		superdiag[i] = h[i+1];
	}
	/* Calculate the main diagonals of the matrix and the right hand side */
	for ( i = 0; i < n-2; i++ ) {
         diag[i] = 2.0 * ( h[i] + h[i+1] );
         c[i]  = ( 3.0 / h[i+1] ) * ( y[i+2] - y[i+1] ) -
                 ( 3.0 / h[i] ) * ( y[i+1] - y[i] );
    }
    /* Complete calculation of the main diagonal */
    diag[0] += ( h[0] + h[0]*h[0] / h[1] );
    diag[n-3] += ( h[n-2] + h[n-2]*h[n-2] / h[n-3] );
    /* Complete calculation of the super and sub diagonal */
    superdiag[0] -= ( h[0]*h[0] / h[1] );
    subdiag[n-4] -= ( h[n-2]*h[n-2] / h[n-3] );
    
    /* Solve the tridiagonal matrix */
 	tridiagonal_gen(subdiag, diag, superdiag, c, n-2);
    
    /* Complete calculation of the C variables */
    for ( i = n-3; i >= 0; i-- ){
        c[i+1] = c[i];
    }
    /* values for c0 and cn-1 */
    c[0] = ( 1.0 + h[0] / h[1] ) * c[1] - h[0] / h[1] * c[2];
    c[n-1] = ( 1.0 + h[n-2] / h[n-3] ) * c[n-2] - h[n-2] / h[n-3] * c[n-3];
    /* solve for a, b, d*/
    for ( i = 0; i < n-1; i++ ) {
        d[i] = ( c[i+1] - c[i] ) / ( 3.0 * h[i] );
        b[i] = ( y[i+1] - y[i] ) / h[i] - h[i] * ( c[i+1] + 2.0*c[i] ) / 3.0;
        a[i] = y[i];
    }
}

/* Performs natural spline */
extern void cspline_natural(Points* points, CSplines* csplines){
	int n = csplines->N;
	double* a = csplines->a;
	double* b = csplines->b;
	double* c = csplines->c;
	double* d = csplines->d;
	double* x = points->X;
	double* y = points->Y;

	int i;
	double* h = (double*) malloc(n*sizeof(double));
	double* diag = (double*) malloc(n*sizeof(double));
	double* superdiag  = (double*) malloc(n*sizeof(double));
	double* subdiag = (double*) malloc(n*sizeof(double));

	/* Calculate the h_j values */
	for(i = 0; i < n-1; i++){
		h[i] = x[i+1] - x[i];
	}
	/* Calculate the sub/super diagonals */
	for(i = 0; i < n-3; i++){
		superdiag[i] = h[i+1];
		subdiag[i] = h[i+1];
	} 
	/* Calculate the main diagonal and the right hand side */
	for(i = 0; i < n-2; i++){
		diag[i] = 2*(h[i]+h[i+1]);
		c[i] = 3*( (y[i+2]-y[i+1])/h[i+1] - (y[i+1]-y[i])/h[i] );
	}

	//tridiagonal_gen(subdiag, diag, superdiag, c, (n-3));
	tridiagonal(c, diag, superdiag, (n-2));


    /* Complete calculation of the C variables */
    for ( i = n-3; i >= 0; i-- ){
        c[i+1] = c[i];
    }
    /* definition of natural boundaries */
    c[0] = 0;
    c[n-1] = 0;

    /* solve for a, b, d */
	for(i = 0; i < n-1; i++){
		d[i] = (c[i+1]-c[i])/(3.0*h[i]);
		b[i] = (y[i+1]-y[i])/h[i] - h[i] * (c[i+1]+2.0*c[i])/3.0;
		a[i] = y[i];
	}
}

/* Performs clamped spline */
extern void cspline_clamped(Points* points, CSplines* csplines, double clampa, double clampb){
	int n = csplines->N;
	double* a = csplines->a;
	double* b = csplines->b;
	double* c = csplines->c;
	double* d = csplines->d;
	double* x = points->X;
	double* y = points->Y;

	int i;
	double* h = (double*) malloc(n*sizeof(double));
	double* diag = (double*) malloc(n*sizeof(double));
	double* superdiag  = (double*) malloc(n*sizeof(double));
	double* subdiag = (double*) malloc(n*sizeof(double));
	/* Calculate the h_j values */
	for(i = 0; i < n-1; i++){
		h[i] = x[i+1] - x[i];
		subdiag[i] = h[i];
		superdiag[i] = h[i];
	}

	/* More diagonal calculation */
	diag[0] = 2 * h[0];
	diag[n-1] = 2 * h[n-2];
	/* right hand side calculation */
	c[0] = 3.0 * ((y[1]-y[0])/h[0] - clampa);
	c[n-1] = 3*(clampb - (y[n-1]-y[n-2])/h[n-2]);
	/* more calculations */
	for(i = 0; i < n-2; i++){
		diag[i+1] = 2.0 * (h[i] + h[i+1]);
		c[i+1] = (3.0 / h[i+1]) * (y[i+2] - y[i+1]) - (3.0 / h[i]) * (y[i+1]-y[i]);
	}
	tridiagonal_gen(subdiag, diag, superdiag, c, n);
	/* solve for a, b, d */
	for(i = 0; i < n-1; i++){
		d[i] = (c[i+1]-c[i])/(3.0*h[i]);
		b[i] = (y[i+1]-y[i])/h[i] - h[i] * (c[i+1]+2.0*c[i])/3.0;
		a[i] = y[i];
	}

}