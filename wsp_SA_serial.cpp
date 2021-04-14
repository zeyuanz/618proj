/*
 * =====================================================================================
 *
 *       Filename:  wsp_SA_serial.cpp
 *
 *    Description:  serial implementation of simulate annealing in WSP
 *    				including baseline parallel version
 *
 *        Created:  04/11/21 10:27:34
 *       Compiler:  gcc
 *
 *         Author:  zeyuan zuo
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

/* @brief: print usage of the program
 **/
void print_usage() {
	printf("Usage ./wsp_SA_serial[options]\n");
	printf("Program Options:\n");
	printf("	-h			--print this message\n");
	printf("	-f	<STRING>	--file contains distance matrix\n");
	printf("	-p	<INT>		--number of processors, should be positive\n");
	printf("	-t			--output running time\n");
}

/* @brief: initialize the path as 0->1->2...->n
 * @para[in]: cost_path. Pointer of integer, length = n_cities+1.
 * cost_path[0] is the cost of the path. cost_path[1] to cost_path[n_cities] is
 * the cityID of the path.
 * @para[in]: n_cities. Integer, number of total cities.
 * @para[in]: dist. Pointer of integer, distance matrix.
 * @para[in]: seed. Pointer of rand seed, required by rand_r().
 **/
void init_path(int *cost_path, int n_cities, int *dist, unsigned int *seed) {
	cost_path[0] = 0;
	// initialize path in 0->1->2->3 ... ->n
	for (int i = 0; i < n_cities; ++i) {
		cost_path[i+1] = i;
	}
	// create a random permutation of the path
	for (int i = n_cities; i >= 1; --i){
		int j = rand_r(seed) % (i)+1;
		int temp = cost_path[i];
		cost_path[i] = cost_path[j];
		cost_path[j] = temp;
	}
	// compute the cost after permutation
	for (int i = 0; i < n_cities; ++i) {
		if (i > 0) {
			cost_path[0] += get_dist(dist, n_cities, cost_path[i], cost_path[i+1]);
		}
	}
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
int edge_dist(int *dist, int n_cities, int *cost_path, int *rand_position) {
	int cost = 0;
	// if the position is not the start
	if (*rand_position != 0) {
		cost += get_dist(dist, n_cities, cost_path[*rand_position], cost_path[*rand_position+1]);
	}
	// if the position is not the end
	if (*rand_position != n_cities-1) {
		cost += get_dist(dist, n_cities, cost_path[*rand_position+1], cost_path[*rand_position+2]);
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
void swap_city(int *cost_path, int* rand_position_1, int* rand_position_2) {
	int tmp = cost_path[*rand_position_1+1];
	cost_path[*rand_position_1+1] = cost_path[*rand_position_2+1];
	cost_path[*rand_position_2+1] = tmp;
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
int random_swap_city_cost(int *cost_path, int n_cities, int *dist, int *rand_position_1, int* rand_position_2, unsigned int *seed) {
	int cost = cost_path[0];
	// randomly select to cities. Make sure two cities are different.
	*rand_position_1 = rand_r(seed) % n_cities;
	*rand_position_2 = rand_r(seed) % n_cities;
	while (*rand_position_1 == *rand_position_2) {
		*rand_position_1 = rand_r(seed) % n_cities;
	}
	// minus the cost when taking out two cities from path
	cost -= edge_dist(dist, n_cities, cost_path, rand_position_1);
	cost -= edge_dist(dist, n_cities, cost_path, rand_position_2);
	// swap the city
	swap_city(cost_path, rand_position_1, rand_position_2);
	// add the cost when adding two cities to the path
	cost += edge_dist(dist, n_cities, cost_path, rand_position_1);
	cost += edge_dist(dist, n_cities, cost_path, rand_position_2);
	return cost;
}

/* @brief: SA algorithm for WSP problem
 * @NOTE: temperature and decay weight are **hard code**.
 * Should improve to passed in arguments in the future.
 * @para[in]: cost_path. Pointer of integer, length = n_cities+1.
 * @para[in]: n_cities. Integer, number of total cities.
 * @para[in]: dist. Pointer of integer, distance matrix.
 * @para[in]: n_iter. Integer. Number of iterations.
 * @para[in]: seed. Pointer of integer, random seed.
 **/
void simulate_annealing(int *cost_path, int n_cities, int *dist, unsigned int *seed) {
	// set two positions for swapping
	int *rand_position_1 = new int(0);
	int *rand_position_2 = new int(1);
	// hard code starting temperature
	double temperature = 0.5;
	// initialize a counter which records when a solution has been accepted
	// if a new solution has been accepted, the counter is reset
	// when the counter equals 2000, break the loop
	// It means it the past 2000 iterations, SA does not acquire a new solution
	// It might have reach the optimal (global or local)
	int cnt = 0;
	while (cnt < 2000) {
		int original_cost = cost_path[0];
		// obtain new cost after swapping
		int new_cost = random_swap_city_cost(cost_path, n_cities, dist, rand_position_1, rand_position_2, seed);
		// if new cost is smaller, accept
		if (new_cost < original_cost) {
			cost_path[0] = new_cost;
			cnt = 0;
		} else {
			// if new cost is bigger, accept with probability
			double diff = static_cast<double>(original_cost - new_cost);
			// prob = exp(diff / temperature)
			double prob;
			// if the temperature is to low, manually set prob equals 0.0
			// to avoid overflow issue
			if (temperature < 1e-12) {
				prob = 0.0;
			} else {
				prob = exp(diff / temperature);
			}
			// obtain a random number in (0,1) to decision
			double rand_number = rand_r(seed) / (RAND_MAX + 1.);
			if (rand_number < prob) {
				cost_path[0] = new_cost;
				cnt = 0;
			} else {
				// if not accepted, recover the state
				swap_city(cost_path, rand_position_1, rand_position_2);
				cnt++;
			}
		}
		// annealing step (i.e. reduce temperature)
		temperature *= 0.9999999;
	}
	delete rand_position_1;
	delete rand_position_2;
}

int main(int argc, char **argv) {
	int p = 1;
	struct timespec before, after;
	bool record_time = false;
	// set rando seed to current time
	srand(time(NULL));
	char *filename = NULL;
	for (int i=0; i < argc; i++) {
		if (strcmp(argv[i], "-h")==0) {
			print_usage();
			return 0;
		}
		if (strcmp(argv[i], "-f")==0) {
			filename=argv[i+1];
		}
		if (strcmp(argv[i], "-p")==0) {
			p = atoi(argv[i+1]);
		}
		if (strcmp(argv[i], "-t")==0) {
			record_time = true;
		}
	}

	FILE *fp = fopen(filename, "r");
	int n_cities;
	fscanf(fp, "%d", &n_cities);
	int *dist = new int[n_cities * n_cities];
	// read dist matrix
	read_dist(fp, dist, n_cities);
	fclose(fp);

	// init global optimal path among all procs
	int *global_cost_path = new int[n_cities+1];
	// set the intial cost to be max integer
	global_cost_path[0] = INT_MAX;
	// initializa different random seeds for different procs
	unsigned int *rand_seeds = new unsigned int[p];
	for (int i = 0; i < p; ++i) {
		rand_seeds[i] = static_cast<unsigned int>(rand());
	}
	
  	clock_gettime(CLOCK_REALTIME, &before);
  	#pragma omp parallel num_threads(p)
	{
		// randomly init the path for different procs
    	int thread_id = omp_get_thread_num();
		// initialize local cost and path pointer
		int *cost_path = new int[n_cities+1];
		init_path(cost_path, n_cities, dist, &rand_seeds[thread_id]);
		// simulating annealing (main solver step)
		simulate_annealing(cost_path, n_cities, dist, &rand_seeds[thread_id]);
		// add lock
		#pragma omp critical
		{
			if (cost_path[0] < global_cost_path[0]) {
				memcpy(global_cost_path, cost_path, sizeof(int) * (n_cities+1));
			}
		}
		delete []cost_path;
	}
  	clock_gettime(CLOCK_REALTIME, &after);
	if (record_time) {
  		double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
		printf("============ Time ============\n");
		printf("Time: %.3f ms (%.3f s)\n", delta_ms, delta_ms / 1000.0);
	}
	// print results
	wsp_print_result(global_cost_path, n_cities);
	
	// free heap alloc memory
	delete []dist;
	delete []rand_seeds;
	delete []global_cost_path;
	return 0;
}










