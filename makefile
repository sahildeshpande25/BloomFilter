all: test2
test2:
	gcc -o bloomfilter -std=c99 bloomfilter.c
	./bloomfilter inserts.txt lookups.txt
	rm -rf bloomfilter

#test1:
#	nvcc -o bloomfilter bloomfilter.cu
#	./bloomfilter inserts.txt lookups.txt
#	rm -rf bloomfilter