#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <functional>
#include <curand_kernel.h>

#define threadsPerBlock 256

typedef struct path_struct_t
{
	float cost; // path cost.
	int *path;   // best order of city visits
} path_t;

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

__device__ float get_dist_cu(float *coord, int i, int j)
{
	float dx = abs(coord[i * 2] - coord[j * 2]);
	float dy = abs(coord[i * 2 + 1] - coord[j * 2 + 1]);
	return sqrt(dx * dx + dy * dy);
}

__device__ void init_path_cu(path_t *cost_path, int n_cities, float *coord, curandState_t *state, int first)
{
	cost_path->cost = 0.0;
	// initialize path in 0->1->2->3 ... ->n
	for (int i = 0; i < n_cities; i++)
	{
		int city = i;
		cost_path->path[i] = city;
	}

	int temp = cost_path->path[first];
	cost_path->path[first] = cost_path->path[0];
	cost_path->path[0] = temp;
	// create a random permutation of the path
	for (int i = n_cities - 1; i >= 1; --i)
	{
		int j = curand(state) % (i) + 1;
		int temp = cost_path->path[i];
		cost_path->path[i] = cost_path->path[j];
		cost_path->path[j] = temp;
	}

	// compute the cost after permutation
	for (int i = 0; i < n_cities - 1; ++i)
	{
		cost_path->cost += get_dist_cu(coord, cost_path->path[i], cost_path->path[i + 1]);
	}
}

__device__ float edge_coord_cu(float *coord, int n_cities, int *cost_path, int *rand_position)
{
	float cost = 0;
	// if the position is not the start
	if (*rand_position != 0)
	{
		cost += get_dist_cu(coord, cost_path[*rand_position - 1], cost_path[*rand_position]);
	}
	// if the position is not the end
	if (*rand_position != n_cities - 1)
	{
		cost += get_dist_cu(coord, cost_path[*rand_position], cost_path[*rand_position + 1]);
	}
	return cost;
}

__device__ void swap_city_cu(int *cost_path_path, int *rand_position_1, int *rand_position_2)
{
	int tmp = cost_path_path[*rand_position_1];
	cost_path_path[*rand_position_1] = cost_path_path[*rand_position_2];
	cost_path_path[*rand_position_2] = tmp;
}

__device__ float random_swap_city_cost_cu(path_t *cost_path, int n_cities, float *coord, int *rand_position_1, int *rand_position_2, curandState_t *state)
{
	float cost = cost_path->cost;
	// randomly select to cities. Make sure two cities are different.
	// also, because of search space decomposition, the first city cannot be choosen.
	*rand_position_1 = (curand(state) % (n_cities - 1)) + 1;
	*rand_position_2 = (curand(state) % (n_cities - 1)) + 1;
	while (*rand_position_1 == *rand_position_2)
	{
		*rand_position_1 = curand(state) % (n_cities - 1) + 1;
	}
	// minus the cost when taking out two cities from path
	cost -= edge_coord_cu(coord, n_cities,cost_path->path, rand_position_1);
	cost -= edge_coord_cu(coord, n_cities,cost_path->path, rand_position_2);
	// swap the city
	swap_city_cu(cost_path->path, rand_position_1, rand_position_2);
	// add the cost when adding two cities to the path
	cost += edge_coord_cu(coord, n_cities,cost_path->path, rand_position_1);
	cost += edge_coord_cu(coord, n_cities,cost_path->path, rand_position_2);
	return cost;
}

__global__ void wsp_sa_kernel(float *dev_all_cost, int *dev_all_path, float* coord_global, int n_cities)
{
	__shared__ float sdata[threadsPerBlock];
	extern __shared__ float coord[];
	__shared__ int idata[threadsPerBlock];
	int tid = threadIdx.x;
	idata[tid] = tid;

	// save coord into shared mem
	int stride = (n_cities + threadsPerBlock) / threadsPerBlock;
	int start = tid * stride;
	int end = min((tid + 1) * stride, n_cities);
	for (int i = start; i < end; i++)
	{	
		coord[i * 2] = coord_global[i * 2];
		coord[i * 2 + 1] = coord_global[i * 2 + 1];
	}
	__syncthreads();

	curandState state;
	double temperature = 20.0;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(n_cities, idx, 0, &state);
	if (idx >= n_cities) return;
	path_t *cost_path = (path_t *)malloc(sizeof(path_t));
	cost_path->cost = 0.0;
	cost_path->path = new int[n_cities];
	init_path_cu(cost_path, n_cities, coord, &state, idx);
	int *rand_position_1 = new int(1);
	int *rand_position_2 = new int(2);
	int cnt = 0;
	while(cnt < 2000)
	{
		float original_cost = cost_path->cost;
		float new_cost = random_swap_city_cost_cu(cost_path, n_cities, coord, rand_position_1, rand_position_2, &state);
		// if new cost is smaller, accept
		if (new_cost < original_cost)
		{
			cost_path->cost = new_cost;
			cnt = 0;
		}
		else
		{
			// if new cost is bigger, accept with probability
			float diff = original_cost - new_cost;
			float prob;
			if (temperature < 1e-12)
			{
				prob = 0.0;
			}
			else
			{
				prob = exp(diff / temperature);
			}
			// obtain a random number in (0,1) to decision
			float rand_number = curand_uniform(&state);
			if (rand_number < prob)
			{
				cost_path->cost = new_cost;
				cnt = 0;
			}
			else
			{
				// if not accepted, recover the state
				swap_city_cu(cost_path->path, rand_position_1, rand_position_2);
				cnt++;
			}
		}
		// annealing step (i.e. reduce temperature)
		temperature *= 0.999999;
	}
	sdata[tid] = cost_path->cost;
	__syncthreads();

	// fast reduction
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		int idx1 = idx;
		int idx2 = idx1 + s;
		if (tid < s && idx1 < n_cities && idx2 < n_cities) {
			float s1 = sdata[tid];
			float s2 = sdata[tid + s];
			if(s1 > s2)
			{
				sdata[tid] = sdata[tid + s];
				idata[tid] = idata[tid + s];
			}
		}
		__syncthreads();
	}
	// now sdata[0] is the min cost, this cost is from thread minId
	if(tid == idata[0])
	{
		dev_all_cost[blockIdx.x] = (double)sdata[0];
		memcpy(dev_all_path + blockIdx.x * n_cities, cost_path->path, sizeof(int)*n_cities);
	}
}

void wsp_simulate_annealing_cuda(path_t *solution, int n_cities, float *coord, float *msec) 
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int blocks = n_cities / threadsPerBlock + 1;
	int *dev_all_path;
	float *dev_all_cost;
	int *host_all_path = new int[n_cities * blocks];
	float *host_all_cost = new float[blocks];
	float *coord_cu;
	cudaMalloc(&dev_all_path, sizeof(int)*n_cities*blocks);
	cudaMalloc(&dev_all_cost, sizeof(float)*blocks);
	cudaMalloc(&coord_cu, sizeof(float)*n_cities*2);
	cudaMemcpy(coord_cu, coord, sizeof(float)*n_cities*2, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	wsp_sa_kernel<<<blocks, threadsPerBlock, sizeof(float)*(n_cities * 3)>>>(dev_all_cost, dev_all_path, coord_cu, n_cities);
	cudaDeviceSynchronize();
	cudaMemcpy(host_all_path, dev_all_path, sizeof(int)*n_cities*blocks, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_all_cost, dev_all_cost, sizeof(float)*blocks, cudaMemcpyDeviceToHost);
	int index = 0;
	double min = host_all_cost[0];
	for(int i = 0; i < blocks; i++) 
	{
		double local = host_all_cost[i];
		if(local < min)
		{
			index = i;
			min = local;
		}
	}
	solution->cost = host_all_cost[index];
	memcpy(solution->path, host_all_path + index * n_cities, sizeof(int)*n_cities);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(msec, start, stop);
	delete[] host_all_path;
	cudaFree(&dev_all_path);
	cudaFree(&coord_cu);
}
