all: test clear

test:
	gcc -o test -std=c99 test.c
	./test

clear:
	rm -rf test