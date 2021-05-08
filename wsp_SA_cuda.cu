#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <functional>
#include <curand_kernel.h>

#define threadsPerBlock 256

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s at %s:%d\n",
				cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#else
#define cudaCheckError(ans) ans
#endif



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



__device__ void set_dist_cu(int* dist, int n_cities, int i, int j, int value) 
{
	int offset = i * n_cities + j;
	dist[offset] = value;
	return;
}

__device__ int get_dist_cu(int* dist, int n_cities, int i, int j) 
{
	int offset = i * n_cities + j;
	return dist[offset];
}

__device__ void init_path_cu(int *cost_path, curandState_t *state, int n_cities, int *dist, int first)
{
	cost_path[0] = 0;
	// initialize path in 0->1->2->3 ... ->n
	for (int i = 0; i < n_cities; ++i)
	{
		cost_path[i + 1] = i;
	}

	int temp = cost_path[first + 1];
	cost_path[first + 1] = cost_path[1];
	cost_path[1] = temp;

	// create a random permutation of the path
	for (int i = n_cities; i >= 2; --i)
	{
		int j = curand(state) % (i - 1) + 2;
		int temp = cost_path[i];
		cost_path[i] = cost_path[j];
		cost_path[j] = temp;
	}
	// compute the cost after permutation
	for (int i = 0; i < n_cities; ++i)
	{
		if (i > 0)
		{
			cost_path[0] += get_dist_cu(dist, n_cities, cost_path[i], cost_path[i + 1]);
		}
	}
}

__device__ int edge_dist_cu(int *dist, int n_cities, int *cost_path, int *rand_position)
{
	int cost = 0;
	// if the position is not the start
	if (*rand_position != 0)
	{
		cost += get_dist_cu(dist, n_cities, cost_path[*rand_position], cost_path[*rand_position + 1]);
	}
	// if the position is not the end
	if (*rand_position != n_cities - 1)
	{
		cost += get_dist_cu(dist, n_cities, cost_path[*rand_position + 1], cost_path[*rand_position + 2]);
	}
	return cost;
}

__device__ void swap_city_cu(int *cost_path, int *rand_position_1, int *rand_position_2)
{
	int tmp = cost_path[*rand_position_1 + 1];
	cost_path[*rand_position_1 + 1] = cost_path[*rand_position_2 + 1];
	cost_path[*rand_position_2 + 1] = tmp;
}

__device__ int random_swap_city_cost_cu(int *cost_path, int n_cities, int *dist, int *rand_position_1, int *rand_position_2, curandState *state)
{
	int cost = cost_path[0];
	// randomly select to cities. Make sure two cities are different.
	// also, because of search space decomposition, the first city cannot be choosen.
	*rand_position_1 = (curand(state) % (n_cities - 1)) + 1;
	*rand_position_2 = (curand(state) % (n_cities - 1)) + 1;
	while (*rand_position_1 == *rand_position_2)
	{
		*rand_position_1 = curand(state) % n_cities;
	}
	// minus the cost when taking out two cities from path
	cost -= edge_dist_cu(dist, n_cities, cost_path, rand_position_1);
	cost -= edge_dist_cu(dist, n_cities, cost_path, rand_position_2);
	// swap the city
	swap_city_cu(cost_path, rand_position_1, rand_position_2);
	// add the cost when adding two cities to the path
	cost += edge_dist_cu(dist, n_cities, cost_path, rand_position_1);
	cost += edge_dist_cu(dist, n_cities, cost_path, rand_position_2);
	return cost;
}

__global__ void wsp_sa_kernel(int *dev_all_path, int* dist, int n_cities)
{
	curandState state;
	double temperature = 20.0;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(idx, 0, 0, &state);
	if (idx > n_cities) return;
	int *cost_path = new int[n_cities + 1];
	init_path_cu(cost_path, &state,n_cities, dist, idx);
	int *rand_position_1 = new int(1);
	int *rand_position_2 = new int(2);
	while(temperature >= 1e-6) 
	{
		int original_cost = cost_path[0];
		// obtain new cost after swapping
		int new_cost = random_swap_city_cost_cu(cost_path, n_cities, dist, rand_position_1, rand_position_2, &state);
		// if new cost is smaller, accept
		if (new_cost < original_cost)
		{
			cost_path[0] = new_cost;
		}
		else
		{
			// if new cost is bigger, accept with probability
			double diff = static_cast<double>(original_cost - new_cost);
			double prob;
			if (temperature < 1e-12)
			{
				prob = 0.0;
			}
			else
			{
				prob = exp(diff / temperature);
			}
			// obtain a random number in (0,1) to decision
			double rand_number = curand_uniform_double(&state);
			if (rand_number < prob)
			{
				cost_path[0] = new_cost;
			}
			else
			{
				// if not accepted, recover the state
				swap_city_cu(cost_path, rand_position_1, rand_position_2);
			}
		}
		temperature *= 0.99999;
	}
	memcpy(dev_all_path + idx * (n_cities + 1), cost_path, sizeof(int)*(n_cities + 1));
}

void wsp_simulate_annealing_cuda(int *solution, int n_cities, int *dist, float *msec) 
{
	printCudaInfo();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int blocks = n_cities / threadsPerBlock + 1;
	int *dev_all_path;
	int *host_all_path = new int[n_cities * (n_cities + 1)];
	int *dist_cu;
	cudaMalloc(&dev_all_path, sizeof(int)*n_cities*(n_cities+1));
	cudaMalloc(&dist_cu, sizeof(int)*n_cities*n_cities);
	cudaMemcpy(dist_cu, dist, sizeof(int)*n_cities*n_cities, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	wsp_sa_kernel<<<blocks, threadsPerBlock>>>(dev_all_path, dist_cu, n_cities);
	cudaDeviceSynchronize();
	printf("Synchronized\n");
	int index = 0;
	cudaMemcpy(host_all_path, dev_all_path, sizeof(int)*n_cities*(n_cities+1), cudaMemcpyDeviceToHost);
	int min = host_all_path[0];
	for(int i = 0; i < n_cities; i++) 
	{
		int local = host_all_path[i * (n_cities + 1)];
		if(local < min)
		{
			index = i;
			min = local;
		}
	}
	memcpy(solution, host_all_path + index * (n_cities + 1), sizeof(int)*(n_cities + 1));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(msec, start, stop);
	delete[] host_all_path;
	cudaFree(&dev_all_path);
}
