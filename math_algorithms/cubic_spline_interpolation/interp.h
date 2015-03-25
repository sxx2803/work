/*
 ==================================== 
 Header File for Interpolation Module 
  Revised: JCCK. Oct 2011
 ==================================== 
*/

#ifndef _INTERP_H_
#define _INTERP_H_

/* Data Types */
typedef struct
  {
   int     N;    /* Number of Elements in array          	*/
   double *a;    /* Pointer to Constant coefficients   	*/
   double *b;    /* Pointer to Linear coefficients       	*/
   double *c;    /* Pointer to Quadratic coefficients   	*/
   double *d;    /* Pointer to Cubic coefficients                 */
   double *X;    /* Pointer interpolation interval  partition */
   } CSplines;

typedef struct
  {
   int     N;    /* Number of Elements in array */
   double *X;    /* Pointer to X data                     */
   double *Y;    /* Pointer to Y data                      */
   int  Size;    /* Size of dynamic arrays          */
   int  Next;    /* Index to next point in array  */
  } Points;
  
/* Function Prototypes, add more if needed */
void   cspline_clamped( Points*, CSplines*, double, double);
void   cspline_natural(Points*, CSplines*);
void   cspline_nak(Points*, CSplines*);
void   tridiagonal(double*, double*, double*, int);
void   tridiagonal_gen(double*, double*, double*, double*, int);
double cspline_eval( Points*, CSplines*, double );

#endif /*  _INTERP_H_ */