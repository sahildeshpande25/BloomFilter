all: test
test:
	gcc -o bloomfilter -std=c99 bloomfilter.c
	./bloomfilter inserts.txt lookups.txt