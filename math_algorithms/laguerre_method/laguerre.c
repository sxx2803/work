/********************************************************************/
/* Sicheng Xu (Fall Quarter)                                        */
/* sxx2803@rit.edu                                                  */
/* Homework 5,                                                      */
/*                                                                  */
/* Laguerre's Method: Uses LaGuerre's method to find roots          */
/* of a polynomial                                                  */
/********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>

/* Evaluates a polynomial at f(x), f'(x), and f''(x) */
void evalPoly(float complex* func, float complex x, float complex* fx, float complex* dx, float complex* ddx, int power){
    int i;
    int n = power;
    float complex p = func[n];
    float complex dp = 0.0 + 0.0I;
    float complex ddp = 0.0 + 0.0I;
    
    /* Expand the polynomial into a series and evaluate */
    for(i = 1; i < n+1; i++){
        ddp = ddp*x + 2.0*dp;
        dp = dp*x + p;
        p = p*x + func[n-i];
    }
    *fx = p;
    *dx = dp;
    *ddx = ddp;
    return;
}

/* Performs one iteration of laguerre's method. Returns the root in the "out" parameter */
void laguerre(float complex* func, float tol, float complex* out, int power){
    float complex x = -3.0 + 0.0I;
    float complex fx;
    float complex dx;
    float complex ddx;
    float complex ax;
    int maxcount = 500;
    int i = 0;
    int n = power;
    
    /* Iterate until a root has been found or max count has been reached */
    for(i = 0; i < maxcount; i++){
        evalPoly(func, x, &fx, &dx, &ddx, n);
        
        /* If the function evaluated at this point is 0, return the x value */
        if(cabsf(fx) < tol){
            *out = x;
            return;
        }
        
        /* Calculate g(x), h(x), and other necessary variables */
        float complex gx = dx/fx;
        float complex hx = (gx*gx) - (ddx/fx);
        float complex denomx = csqrtf((n-1)*(n*hx - gx*gx));
        /* Decide to add or subtract */
        if(cabsf(gx + denomx) > cabsf(gx-denomx)){
            ax = n/(gx+denomx);
        }
        else{
            ax = n/(gx-denomx);
        }
        /* Iterate */
        x = x - ax;
        i++;
        /* Exit if ax is less than tolerance level */
        if(cabsf(ax)<tol){
            *out = x;
            return;
        }
    }
    *out = x;
    return;
}

/* Deflates a polynomial */
void deflPoly(float complex* func, float complex root, float complex* out, int power){
    int i;
    int n = power;
    out[n-1] = func[n];
    /* Do synthetic division */
    for(i = n-2; i>=0; i--){
        out[i] = func[i+1] + root*out[i+1];
    }
    return;
}


/* Main method. Finds roots from a file */
int main(int argc, const char * argv[])
{
    int maxpower;
    int count;
    char line[25];
    char *token;
    float complex *coeff;
    int j;
    int z;
    float complex root;
    float complex *deflated;
    float complex* roots;
    
    count = 0;
    
    /* Check for correct number of arguments */
    if(argc == 2){
        /* Open the file */
        FILE *file = fopen(argv[1], "r");
        /* Check for successful file opening */
        if(file != NULL){
            /* Get the polynomial degree */
            while(fgets(line, sizeof line, file) != NULL){
                count += 1;
            }
            maxpower = count-1;
            fseek(file, 0, SEEK_SET);
            coeff =  (float complex*) malloc(count*sizeof(float complex));
            count = 0;
            /* Tokenize the input and grab it */
            while(fgets(line, sizeof line, file) != NULL){
                token = strtok(line, "\t\n");
                if(token != NULL){
                    coeff[maxpower-count] = atof(token) + 0.0I;
                }
                count++;
            }
            roots = malloc(maxpower*sizeof(float complex));
            /* A nth degree polynomial has n roots */
            for(j = 0; j < maxpower; j++){
                /* Do laguerre's method */
                laguerre(coeff, 1e-12, &root, maxpower);
                /* Store the root */
                roots[j] = root;

                /* Deflate the polynomial */
                deflated = (float complex*) malloc((maxpower-j)*sizeof(float complex));
                deflPoly(coeff, root, deflated, (maxpower-j));
                coeff = deflated;
                /* If root was complex, deflate again */
                if(fabs(cimag(root))>0){
                    j += 1;
                    roots[j] = conjf(root);
                    deflated = (float complex*) malloc((maxpower-j)*sizeof(float complex));
                    deflPoly(coeff, conjf(root), deflated, (maxpower-j));
                    coeff = deflated;
                }
            }
            /* Print the results */
            for(z = 0; z < maxpower; z++){
                printf("Root %i is: %f + %fI\n", (z+1), creal(roots[z]), cimag(roots[z]));
            }
        }
        else{
            fprintf(stderr, "File '%s' cannot be opened\n", argv[1]);
        }
    }
    else{
        fprintf(stderr, "Usage: laguerre filename\n");
    }
    
    return 0;
}

