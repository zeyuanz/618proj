
CXXFLAGS += -std=c++11 -Wall -Wextra -pthread -fopenmp -g -O3 -DNDEBUG

.phony: all 

all: 
	mkdir -p objs/
	nvcc cuda_opt_SA.cu -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c -lcurand -lcurand_kernel -o objs/cuda_opt_SA.o
	g++ -m64 opt_SA.cpp -O3 -Wall -std=c++11 -pthread -fopenmp -g -c -o objs/opt_SA.o
	g++ -m64 -O3 -Wall -pthread -fopenmp -g -o opt_SA objs/opt_SA.o objs/cuda_opt_SA.o -L/usr/local/depot/cuda-10.2/lib64/ -lcudart -lGL -lglut -lcudart

clean:
	rm -rf ./wsp_SA_serial ./opt_SA ./objs
