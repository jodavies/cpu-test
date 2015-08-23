all:
	mkdir -p bin
	gcc src/*.c -o bin/cpu-test -std=gnu99 -g -Wall -pedantic -lrt -lm -lpapi -fopenmp -O3 -march=core-avx-i
	icc src/*.c -o bin/cpu-test-icc -std=gnu99 -g -lrt -lm -lpapi -fopenmp -O3 -xAVX


clean:
	rm -r bin
