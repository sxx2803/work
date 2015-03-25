Instructions:

Bisection:

- To use the Bisection program, provide it with 3 arguments: the lower bound X value, the upper bound X value, and the tolerance. The bounds can be provided in either order (i.e. upper bound can be provided as argument 1). The tolerance, however, must be provided last. The program would then run and output the root found, the number of function evaluations, and the number of iterations.

Newton:

- To use the Newton program, provide it with 2 arguments: the initial guess and the tolerance. The initial guess should be sufficiently close to a real root such that the function will converge. However, since the function is a sinusoidal function, the initial guess shouldn't matter as there are infinite real roots. The program would then run and output the root found, the number of function evaluations, the number of derivative evaluations, and the number of iterations.

Secant:

- To use the Secant program, provide it with 3 arguments: the initial guess, the previous inital guess, and the tolerance. The arguments must be provided in this order. The initial guesses should be sufficiently close to a real root such that the function will converge. However, since the function is a sinusoidal function, the initial guesses shouldn't matter as there are infinite real roots. However, the performance of the program will be severly impacted if the initial guesses are too far from a real root. The program would then run and output the root found, the number of function evaluations, and the number of iterations.