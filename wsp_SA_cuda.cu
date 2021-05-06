#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <functional>
#include <curand_kernel.h>

#define threadsPerBlock 256
typedef double(*test_func_t)(double*, int);

__device__ double rastrigin_cuda(double *input, int size) {
	if (size == 0) {
		return input[0] * input[0] - 10.0 * cos(2.0 * M_PI * input[0]);
	}
	double first_term = 10 * static_cast<double>(size);
	double second_term = 0.0;
	for (int i = 0; i < size; ++i) {
		second_term += (input[i] * input[i]);
		second_term -= 10.0 * cos(2.0 * M_PI * input[i]);
	}
	return first_term + second_term;
}
__device__ double ackley_cuda(double *input, int size) {
	if (size == 0) {
		return -20.0 * exp(-0.2 * sqrt(0.5 * input[0] * input[0]))
			-exp(0.5 * cos(2.0 * M_PI * input[0]));
	}
	double square_term = 0.0;
	double cosine_term = 0.0;
	for (int i = 0; i < size; ++i) {
		square_term += input[i] * input[i];
		cosine_term += cos(2.0 * M_PI * input[i]);
	}
	double first_term = -20.0 * exp(-0.2 * sqrt(0.5 * square_term));
	double second_term = -exp(cosine_term / double(size)) + exp(1.0) + 20.0;
	return first_term + second_term;
}





__device__ double rand_normal_cuda(double *solution_idx, double lo, double hi, 
		double sigma, curandState *state, test_func_t func) {
	double rand_num = curand_normal_double(state) * sigma;
	double diff = 0.0;
	diff -= func(solution_idx, 0);
	*solution_idx += rand_num;
	if (*solution_idx < lo) {
		*solution_idx = lo;
	} else if (*solution_idx > hi) {
		*solution_idx = hi;
	}
	diff += func(solution_idx, 0);
	return diff;
}

__global__ void sa_kernel(double *solution, int size, double lo, double hi, 
		double sigma, int choice) {
	curandState state;
	test_func_t func;
	switch (choice) {
		case(1):
			func = rastrigin_cuda;
			break;
		case(2):
			func = ackley_cuda;
			break;
	}
	double temperature = 1.0;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(idx, 0, 0, &state);
	if (idx > size) return;
	while(temperature >= 1e-6) {
		double original_sol = solution[idx];
		double diff = rand_normal_cuda(solution+idx, lo, hi, sigma, &state, func);
		if (diff > 0) {
			double alpha = curand_uniform_double(&state);
			double prob = exp(-diff / temperature);
			if (alpha > prob) {
				solution[idx] = original_sol;
			}
		}
		temperature *= 0.99999;
	}
}

void simulate_annealing_cuda(double *solution, int size, double lo,
		double hi, double sigma, float *msec, int choice) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    int blocks = size / threadsPerBlock + 1;
	double *dev_solution;
	cudaMalloc(&dev_solution, sizeof(double)*size);
	cudaMemcpy(dev_solution, solution, sizeof(double)*size, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
    sa_kernel<<<blocks, threadsPerBlock>>>(dev_solution, size, lo, hi, sigma, choice);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(msec, start, stop);
	cudaMemcpy(solution, dev_solution, sizeof(double)*size, cudaMemcpyDeviceToHost);
	cudaFree(&dev_solution);
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}