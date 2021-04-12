/*
 * =====================================================================================
 *
 *       Filename:  wsp_SA_serial.cpp
 *
 *    Description:  serial implementation of simulate annealing in WSP
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
#include <math.h>
#include <time.h>
#include "wsp.h"

/* @brief: print usage of the program
 **/
void print_usage() {
	printf("Usage ./wsp_SA_serial[options]\n");
	printf("Program Options:\n");
	printf("	-h			--print this message\n");
	printf("	-f	<STRING>	--file contains distance matrix\n");
}

/* @brief: initialize the path as 0->1->2...->n
 * @para[in]: cost_path. Pointer of integer, length = n_cities+1.
 * cost_path[0] is the cost of the path. cost_path[1] to cost_path[n_cities] is
 * the cityID of the path.
 * @para[in]: n_cities. Integer, number of total cities.
 * @para[in]: dist. Pointer of integer, distance matrix.
 **/
void init_path(int *cost_path, int n_cities, int *dist) {
	cost_path[0] = 0;
	for (int i = 0; i < n_cities; ++i) {
		cost_path[i+1] = i;
		if (i > 0) {
			cost_path[0] += get_dist(dist, n_cities, i-1, i);
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
 **/
int random_swap_city_cost(int *cost_path, int n_cities, int *dist, int *rand_position_1, int* rand_position_2) {
	int cost = cost_path[0];
	// randomly select to cities. Make sure two cities are different.
	*rand_position_1 = rand() % n_cities;
	*rand_position_2 = rand() % n_cities;
	while (*rand_position_1 == *rand_position_2) {
		*rand_position_1 = rand() % n_cities;
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
 * @para[in]: n_iter. Number of iterations.
 **/
void simulate_annealing(int *cost_path, int n_cities, int *dist, int n_iter) {
	// set two positions for swapping
	int *rand_position_1 = new int(0);
	int *rand_position_2 = new int(1);
	// hard code starting temperature
	double temperature = 0.01;
	for (int i = 0; i < n_iter; i++) {
		int original_cost = cost_path[0];
		// obtain new cost after swapping
		int new_cost = random_swap_city_cost(cost_path, n_cities, dist, rand_position_1, rand_position_2);
		// if new cost is smaller, accept
		if (new_cost < original_cost) {
			cost_path[0] = new_cost;
		} else {
			// if new cost is bigger, accept with probability
			double diff = static_cast<double>(original_cost - new_cost);
			// prob = exp(diff / temperature)
			double prob = exp(diff / temperature);
			// obtain a random number in (0,1) to decision
			double rand_number = rand() / (RAND_MAX + 1.);
			if (rand_number < prob) {
				cost_path[0] = new_cost;
			} else {
				// if not accepted, recover the state
				swap_city(cost_path, rand_position_1, rand_position_2);
			}
		}
		// annealing step (i.e. reduce temperature)
		temperature *= 0.99;
	}
	delete rand_position_1;
	delete rand_position_2;
}

int main(int argc, char **argv) {
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
	}

	FILE *fp = fopen(filename, "r");
	int n_cities;
	fscanf(fp, "%d", &n_cities);
	int *dist = new int[n_cities * n_cities];
	// read dist matrix
	read_dist(fp, dist, n_cities);
	fclose(fp);

	int *cost_path = new int[n_cities+1];
	// the n_iter can be tuned, hard code for now
	int n_iter = n_cities * 2000;
	// initializa path
	init_path(cost_path, n_cities, dist);
	// simulating annealing (main solver step)
	simulate_annealing(cost_path, n_cities, dist, n_iter);
	// print results
	wsp_print_result(cost_path, n_cities);
	delete []dist;
	delete []cost_path;
	return 0;
}










