# -------------------------------------------------------
# Sample Makefile for Linux systems (tested on Ubuntu)
# Compiles and tests Gaussian Elimination
# -------------------------------------------------------
## Testing Parameters
P1 = gec_test1.txt
P2 = gec_test2.txt
P3 = gec_test3.txt

## Main body of makefile
# Warning, use tabs not spaces

all: .c

.c:
	gcc -Wall -pedantic -ansi -std=c99 -o gewhpp gewhpp.c -lm
	gcc -Wall -pedantic -ansi -std=c99 -o gewspp gewspp.c -lm
	gcc -Wall -pedantic -ansi -std=c99 -o gewpp gewpp.c -lm

test: gewhpp gewspp gewpp
	./gewhpp $(P1) > results.txt
	./gewhpp $(P2) >> results.txt
	./gewhpp $(P3) >> results.txt
	./gewspp $(P1) >> results.txt
	./gewspp $(P2) >> results.txt
	./gewspp $(P3) >> results.txt
	./gewpp $(P1) >> results.txt
	./gewpp $(P2) >> results.txt
	./gewpp $(P3) >> results.txt

timing:
	gcc -Wall -pedantic -ansi -std=c99 -DEN_TIME -o gewhpp gewhpp.c -lm
	gcc -Wall -pedantic -ansi -std=c99 -DEN_TIME -o gewspp gewspp.c -lm
	gcc -Wall -pedantic -ansi -std=c99 -DEN_TIME -o gewpp gewpp.c -lm
	./gewhpp $(P1) > results.txt
	./gewhpp $(P2) >> results.txt
	./gewhpp $(P3) >> results.txt
	./gewspp $(P1) >> results.txt
	./gewspp $(P2) >> results.txt
	./gewspp $(P3) >> results.txt
	./gewpp $(P1) >> results.txt
	./gewpp $(P2) >> results.txt
	./gewpp $(P3) >> results.txt

gewhpp: gewhpp.c

gewspp: gewspp.c

gewpp: gewpp.c

clean:
	rm -f *.o gewhpp gewspp gewpp