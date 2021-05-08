
CXXFLAGS += -std=c++11 -Wall -Wextra -pthread -fopenmp -g -O3 -DNDEBUG

.phony: all 

all: 
	g++ wsp_SA_space.cpp -o wsp_SA_space $(CXXFLAGS)
	g++ wsp_SA_double.cpp -o wsp_SA_double $(CXXFLAGS)
	mkdir -p objs/
	nvcc wsp_SA_cuda.cu -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c -lcurand -lcurand_kernel -o objs/wsp_SA_cuda.o
	g++ -m64 wsp_SA.cpp -O3 -Wall -std=c++11 -pthread -fopenmp -g -c -o objs/wsp_SA.o
	g++ -m64 -O3 -Wall -pthread -fopenmp -g -o wsp_SA objs/wsp_SA.o objs/wsp_SA_cuda.o -L/usr/local/depot/cuda-10.2/lib64/ -lcudart -lGL -lglut -lcudart

clean:
	rm -f ./wsp_SA_double
	rm -f ./wsp_SA_space
