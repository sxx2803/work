# -------------------------------------------------------
# Sample Makefile for Linux systems (tested on Ubuntu)
# Compiles and tests Cubic Spline Interpolation
# -------------------------------------------------------
## Testing Parameters
P1 = Data.txt

## Main body of makefile
# Warning, use tabs not spaces

all: .c

.c:
	gcc -Wall -pedantic -ansi -std=c99 -o hw8 hw8.c interp.c -lm

test: hw8
	./hw8 $(P1)

hw8: hw8.c


clean:
	rm -f *.o hw8