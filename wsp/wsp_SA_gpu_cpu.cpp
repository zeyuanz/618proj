/*
 * =====================================================================================
 *
 *       Filename:  wsp_SA_gpu.cpp
 *
 *    Description:  parallel implementation of simulate annealing in WSP on CPU and GPU
 *
 *        Created:  04/28/21
 *       Compiler:  gcc
 *
 *         Author:  zeyuan zuo & zeruizhi cheng
 *   Organization:  CMU
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <random>
#include <limits.h>
#include <math.h>
#include <time.h>
#include "wsp.h"
#include <omp.h>

/*
 * global variables
 **/
int *globals;
double *global_costs;

/* @brief: print usage of the program
 **/
void print_usage()
{
	printf("Usage ./wsp_SA_serial[options]\n");
	printf("Program Options:\n");
	printf("	-h			--print this message\n");
	printf("	-f	<STRING>	--file contains distance matrix\n");
	printf("	-p	<INT>		--number of processors, should be positive\n");
	printf("	-t			--output running time\n");
}

void wsp_simulate_annealing_cuda(path_t *solution, int n_cities, double *dist, float *msec);

/* @brief: initialize the path as 0->1->2...->n
 * @para[in]: cost_path. Pointer of integer, length = n_cities+1.
 * cost_path[0] is the cost of the path. cost_path[1] to cost_path[n_cities] is
 * the cityID of the path.
 * @para[in]: n_cities. Integer, number of total cities.
 * @para[in]: dist. Pointer of integer, distance matrix.
 * @para[in]: seed. Pointer of rand seed, required by rand_r().
 * @para[in]: First. First element in the path
 **/
void init_path(path_t *cost_path, int n_cities, double *dist, unsigned int *seed, int first)
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
		int j = rand_r(seed) % (i) + 1;
		int temp = cost_path->path[i];
		cost_path->path[i] = cost_path->path[j];
		cost_path->path[j] = temp;
	}

	// compute the cost after permutation
	for (int i = 0; i < n_cities - 1; ++i)
	{
		cost_path->cost += get_dist(dist, n_cities, cost_path->path[i], cost_path->path[i + 1]);
	}
	//printf("the initial cost is %lf\n", cost_path->cost);
}

/* @brief: acquire the neigboring distance given position. E.g.
 * ...5->3->6...
 * if 3 is the position selected, the function would return the
 * distance of 5->3 + 3->6. Start and end should be regarded as special case.
 * @para[in]: dist. Pointer of integer, distance matrix.
 * @para[in]: n_cities. Integer, number of total cities.
 * @para[in]: cost_path. Pointer of integer, length = n_cities+1.
 * @para[in]: rand_position. Pointer of integer, the position we look for.
 **/
double edge_dist(double *dist, int n_cities, int *cost_path, int *rand_position)
{
	double cost = 0;
	// if the position is not the start
	if (*rand_position != 0)
	{
		cost += get_dist(dist, n_cities, cost_path[*rand_position - 1], cost_path[*rand_position]);
	}
	// if the position is not the end
	if (*rand_position != n_cities - 1)
	{
		cost += get_dist(dist, n_cities, cost_path[*rand_position], cost_path[*rand_position + 1]);
	}
	return cost;
}

/* @brief: swap the cityID given two positions
 * @para[in]: cost_path. Pointer of integer, length = n_cities+1.
 * @para[in]: rand_position_1. Pointer of integer, the first position for
 * swapping.
 * @para[in]: rand_position_2. Pointer of integer, the second position for
 * swapping.
 **/
void swap_city(int *cost_path_path, int *rand_position_1, int *rand_position_2)
{
	int tmp = cost_path_path[*rand_position_1];
	cost_path_path[*rand_position_1] = cost_path_path[*rand_position_2];
	cost_path_path[*rand_position_2] = tmp;
}

/* @brief: compute the cost after swapping two cities.
 * @NOTE: the cities are **SWAPPED**. It means the change is not accepted by
 * SA. The order should be recovered (i.e. swap back to original status).
 * @para[in]: cost_path. Pointer of integer, length = n_cities+1.
 * @para[in]: n_cities. Integer, number of total cities.
 * @para[in]: dist. Pointer of integer, distance matrix.
 * @para[in]: rand_position_1. Pointer of integer, the first position for
 * swapping.
 * @para[in]: rand_position_2. Pointer of integer, the second position for
 * swapping.
 * @para[in]: seed. Pointer of integer, rand_seed, required by rand_r
 **/
double random_swap_city_cost(path_t *cost_path, int n_cities, double *dist, int *rand_position_1, int *rand_position_2, unsigned int *seed)
{
	//printf("in random swap, random position 1 is %d, position 2 is %d\n", *rand_position_1, *rand_position_2);
	double cost = cost_path->cost;
	// randomly select to cities. Make sure two cities are different.
	// also, because of search space decomposition, the first city cannot be choosen.
	*rand_position_1 = (rand_r(seed) % (n_cities - 1)) + 1;
	*rand_position_2 = (rand_r(seed) % (n_cities - 1)) + 1;
	while (*rand_position_1 == *rand_position_2)
	{
		*rand_position_1 = rand_r(seed) % (n_cities - 1) + 1;
	}
	// minus the cost when taking out two cities from path
	cost -= edge_dist(dist, n_cities, cost_path->path, rand_position_1);
	cost -= edge_dist(dist, n_cities, cost_path->path, rand_position_2);
	// swap the city
	//printf("minus the cost when taking two cities from path, the cost is %lf\n", cost);
	swap_city(cost_path->path, rand_position_1, rand_position_2);
	// add the cost when adding two cities to the path
	cost += edge_dist(dist, n_cities, cost_path->path, rand_position_1);
	cost += edge_dist(dist, n_cities, cost_path->path, rand_position_2);
	//printf("add the cost when adding two cities into path, the cost is %lf\n", cost);
	return cost;
}

/* @brief: SA algorithm for WSP problem
 * @NOTE: temperature and decay weight are **hard code**.
 * Should improve to passed in arguments in the future.
 * @para[in]: cost_path. Pointer of integer, length = n_cities+1.
 * @para[in]: n_cities. Integer, number of total cities.
 * @para[in]: dist. Pointer of integer, distance matrix.
 **/
void simulate_annealing(int n_cities, double *dist, int p)
{
	// generate random seeds
	unsigned int *rand_seeds = new unsigned int[n_cities];
	for (int i = 0; i < n_cities; ++i)
	{
		rand_seeds[i] = static_cast<unsigned int>(rand());
	}

#pragma omp parallel for num_threads(p)
	for (int i = 0; i < n_cities; i++)
	{
		// randomly init the path for different procs
		// initialize local cost and path pointer
		path_t *cost_path = (path_t *)malloc(sizeof(path_t));
		cost_path->cost = 0.0;
		cost_path->path = (int *)calloc(n_cities, sizeof(int));
		unsigned int *seed = &rand_seeds[i];
		init_path(cost_path, n_cities, dist, seed, i);
		// simulating annealing (main solver step)
		// set two positions for swapping
		int *rand_position_1 = new int(0);
		int *rand_position_2 = new int(1);
		// hard code starting temperature
		double temperature = 20;
		int cnt = 0;
		while (cnt < 2000)
		{
			double original_cost = cost_path->cost;
			double new_cost = random_swap_city_cost(cost_path, n_cities, dist, rand_position_1, rand_position_2, seed);
			if (new_cost < original_cost)
			{
				cost_path->cost = new_cost;
				cnt = 0;
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
				double rand_number = rand_r(seed) / (RAND_MAX + 1.);
				if (rand_number < prob)
				{
					cost_path->cost = new_cost;
					cnt = 0;
				}
				else
				{
					// if not accepted, recover the state
					swap_city(cost_path->path, rand_position_1, rand_position_2);
					cnt++;
				}
			}
			// annealing step (i.e. reduce temperature)
			temperature *= 0.999999;
		}
		// write the result to globals
		memcpy(globals + i * n_cities, cost_path->path, sizeof(int) * n_cities);
		global_costs[i * 8] = cost_path->cost;
		free(cost_path->path);
		free(cost_path);
	}
}

void gather(path_t *global_cost_path, int n_cities)
{
	int index = 0;
	double min = global_costs[0];
#pragma omp parallel
	{
		int index_local = index;
		double min_local = min;
#pragma omp for nowait
		for (int i = 1; i < n_cities; i++)
		{
			if (global_costs[i * 8] < min_local)
			{
				min_local = global_costs[i * 8];
				index_local = i;
			}
		}
#pragma omp critical
		{
			if (min_local < min)
			{
				min = min_local;
				index = index_local;
			}
		}
	}
	global_cost_path->cost = min;
	memcpy(global_cost_path->path, globals + index * n_cities, sizeof(int) * n_cities);
}

int main(int argc, char **argv)
{
	int p = 1;
	struct timespec before, after;
	bool record_time = false;
	// set rando seed to current time
	srand(time(NULL));
	char *filename = NULL;
	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0)
		{
			print_usage();
			return 0;
		}
		if (strcmp(argv[i], "-f") == 0)
		{
			filename = argv[i + 1];
		}
		if (strcmp(argv[i], "-p") == 0)
		{
			p = atoi(argv[i + 1]);
		}
		if (strcmp(argv[i], "-t") == 0)
		{
			record_time = true;
		}
	}

	FILE *fp = fopen(filename, "r");
	int n_cities;
	fscanf(fp, "%d", &n_cities);
	double *dist = new double[n_cities * n_cities];
	// read dist matrix
	read_dist(fp, dist, n_cities);
	fclose(fp);
	// init global optimal path among all procs
	path_t *global_cost_path = (path_t *)malloc(sizeof(path_t));
	path_t *global_cost_path_cu = (path_t *)malloc(sizeof(path_t));

	// set the intial cost to be max integer
	global_cost_path->cost = 0;
	global_cost_path->path = new int[n_cities];
	global_cost_path_cu->cost = 0;
	global_cost_path_cu->path = new int[n_cities];

	globals = new int[n_cities * n_cities];
	global_costs = new double[n_cities * 8];
	// initializa different random seeds for different procs

	clock_gettime(CLOCK_REALTIME, &before);
	simulate_annealing(n_cities, dist, p);
	gather(global_cost_path, n_cities);
	clock_gettime(CLOCK_REALTIME, &after);

	float msec;
	wsp_simulate_annealing_cuda(global_cost_path_cu, n_cities, dist, &msec);
	// print results
	print_result(global_cost_path, global_cost_path_cu, (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0, msec, n_cities);
	// free heap alloc memory
	delete[] dist;
	delete[] global_cost_path;
	delete[] global_costs;
	delete[] globals;
	return 0;
}
