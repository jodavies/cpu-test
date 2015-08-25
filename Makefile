all:
	mkdir -p bin
	gcc src/*.c -o bin/cpu-test -std=gnu99 -g -Wall -pedantic -lrt -lm -lpapi -fopenmp -O3 -march=core-avx-i
	icc src/*.c -o bin/cpu-test-icc -std=gnu99 -g -lrt -lm -lpapi -fopenmp -O3 -xAVX


# for the following machines, need to use my build of gcc
piledriver:
	mkdir -p bin
	gcc4.8.4 src/*.c -o bin/cpu-test -std=gnu99 -g -Wall -pedantic -lrt -lm -lpapi -fopenmp -O3 -march=bdver2

ivybridge:	
	mkdir -p bin
	gcc4.8.4 src/*.c -o bin/cpu-test -std=gnu99 -g -Wall -pedantic -lrt -lm -lpapi -fopenmp -O3 -march=core-avx-i

magnycours:
	mkdir -p bin
	gcc4.8.4 src/*.c -o bin/cpu-test -std=gnu99 -g -Wall -pedantic -lrt -lm -lpapi -fopenmp -O3 -march=barcelona


clean:
	rm -r bin
