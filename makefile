all: test2 test1
test2:
	gcc -o bloomfilter -std=c99 bloomfilter.c
	time ./bloomfilter inserts_big.txt lookups_big.txt
	rm -rf bloomfilter

test1:
	nvcc -o bloomfilter bloomfilter.cu
	time ./bloomfilter inserts_big.txt lookups_big.txt
	rm -rf bloomfilter