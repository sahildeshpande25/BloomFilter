all: test2 test1
test2:
	gcc -o seq_bloomfilter -std=c99 seq_bloomfilter.c
	time ./bloomfilter inserts.txt lookups.txt
	rm -rf seq_bloomfilter

test1:
	nvcc -o bloomfilter bloomfilter.cu
	time ./bloomfilter inserts.txt lookups.txt
	rm -rf bloomfilter