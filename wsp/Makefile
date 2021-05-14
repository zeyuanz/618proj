
CXXFLAGS += -std=c++11 -Wall -Wextra -pthread -fopenmp -g -O3 -DNDEBUG

.phony: all 

all: 
	g++ wsp_SA_parallel.cpp -o wsp_SA_parallel $(CXXFLAGS)
	mkdir -p objs/
	nvcc wsp_SA_cuda.cu -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c -lcurand -lcurand_kernel -o objs/wsp_SA_cuda.o
	g++ -m64 wsp_SA_gpu_cpu.cpp -O3 -Wall -std=c++11 -pthread -fopenmp -g -c -o objs/wsp_SA_gpu_cpu.o
	g++ -m64 -O3 -Wall -pthread -fopenmp -g -o wsp_SA_gpu_cpu objs/wsp_SA_gpu_cpu.o objs/wsp_SA_cuda.o -L/usr/local/depot/cuda-10.2/lib64/ -lcudart -lGL -lglut -lcudart
	nvcc wsp_SA_cuda_coord.cu -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c -lcurand -lcurand_kernel -o objs/wsp_SA_cuda_coord.o
	g++ -m64 wsp_SA_gpu_coord.cpp -O3 -Wall -std=c++11 -pthread -fopenmp -g -c -o objs/wsp_SA_gpu_coord.o
	g++ -m64 -O3 -Wall -pthread -fopenmp -g -o wsp_SA_gpu_coord objs/wsp_SA_gpu_coord.o objs/wsp_SA_cuda_coord.o -L/usr/local/depot/cuda-10.2/lib64/ -lcudart -lGL -lglut -lcudart

clean:
	rm -rf ./objs ./wsp_SA_parallel ./wsp_SA_gpu_coord rm ./wsp_SA_gpu_cpu


