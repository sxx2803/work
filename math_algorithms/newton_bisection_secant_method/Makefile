# -------------------------------------------------------
# Sample Makefile for Linux systems (tested on Ubuntu)
# Compiles and tests Bisection, Newton and Secant methods
# Rootfinding Methods (J.C. Cockburn 2012)
# -------------------------------------------------------
## Testing Parameters
TOL=1e-12
# Bisection
A=-2.5
B=+2.5
# Secant
X11=2.0
X12=2.0
# Newton
X01=1.5
X02=-1.5

## Main body of makefile
# Warning, use tabs not spaces

test: Bisection Newton Secant
	./Bisection $(A) $(B) $(TOL)
	./Secant $(X01) $(X11) $(TOL)
	./Secant $(X02) $(X12) $(TOL)
	./Newton $(X01) $(TOL)
	./Newton $(X02) $(TOL)

.c:
	gcc -Wall -pedantic -ansi -o ${*F} ${*F}.c -lm

Bisection: Bisection.c

Newton: Newton.c

Secant: Secant.c

clean:
	rm -f Bisection Newton Secant