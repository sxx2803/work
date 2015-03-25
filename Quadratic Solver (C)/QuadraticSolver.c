/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					            */ 
/* Homework 0,                                                      */ 
/*                                                                  */ 
/* Program to compute the roots of a quadratic equation             */ 
/* a*xˆ2+b*x+c=0                                                    */ 
/* The program reads the coefficients from the command line as      */ 
/* a b c, in that order, then it echoes the polynomial equation,    */ 
/* and prints the real or complex conjugate roots.                  */ 
/* Last Update: Juan C. Cockburn 11/28/2011 (juan.cockburn@rit.edu) */ 
/********************************************************************/

/* Standard Libraries */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Macros */
#define COEFF_A (2)
#define COEFF_B (1)
#define COEFF_C (0)

/* Functions */
int main (int argc, char *argv[]) {
  char Version[] = "v1.0e";         /* String with Version of the program */
  int Index;                        /* Loop index variable                */
  int RC=0;                         /* Return code for OS                 */
  double Coefficients[COEFF_A+1];   /* Vector of Polynomial Coefficients  */   
  double Discriminant;              /* Indicates real or complex roots    */
  double Root_1, Root_2;            /* Real roots of equation             */
  double Real, Imaginary;           /* Complex roots of Equation          */

  /* Print title and version of program */
  printf ("Quadratic Solver, version %s\n", Version);
  /* Check for enough command line arguments */
  if (argc >= 4) {
     for (Index = 2; Index >= 0; Index--) {
       Coefficients[Index] = atof (argv[3-Index]);
      } /* for Index */

    /* Show user equation to be solved (for verification) */
    printf ("Equation to be solved:  %gxˆ2 + %gx + %g = 0\n",
            Coefficients[COEFF_A], Coefficients[COEFF_B],
            Coefficients[COEFF_C]);
    
    /* Calculate discriminant */
    Discriminant = Coefficients[COEFF_B] * Coefficients[COEFF_B]
                   - 4.0 * Coefficients[COEFF_A] * Coefficients[COEFF_C];
    
    /* Check for real or complex roots */
    if (Discriminant < 0.0) { /* Complex conjugate roots */
           Real = -Coefficients[COEFF_B] / (2.0 * Coefficients[COEFF_A]);
           Imaginary = sqrt(-Discriminant)/ (2.0 * Coefficients[COEFF_A]);
          /* Print complex roots */
          printf ("The quadratic has complex roots:  %f +/- j%f\n",
                  Real, Imaginary);
        } /* if Discriminant */
        else { /* Real roots */
           Root_1 = (sqrt (Discriminant) - Coefficients[COEFF_B])
                     / (2.0 * Coefficients[COEFF_A]);
          Root_2 = (-sqrt (Discriminant) - Coefficients[COEFF_B])
                   / (2.0 * Coefficients[COEFF_A]);
          /* Print real roots */
          printf ("The quadratic has real roots:  %f and %f\n",
                  Root_1, Root_2);
       } /* else Discriminant */
    } /* if argc */
    else {
       /* Too few command line arguments */
       printf ("Equation: a xˆ2 + b x + c = 0 \n");
       printf ("Usage: %s a b c\n",argv[0]);
       RC = -1; /* Invocation error: return -1 to the OS */
   }  /* else argc */
   return (RC);
} /* main () */
