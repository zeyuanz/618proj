/*
 * =====================================================================================
 *
 *       Filename:  opt_SA_serial.cpp
 *
 *    Description:  serial implementation of non-convex opt with SA
 *
 *        Version:  1.0
 *        Created:  04/14/21 21:31:04
 *       Compiler:  gcc
 *
 *         Author:  zeyuan zuo
 *   Organization:  CMU
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <functional>
#include <omp.h>
#include "opt.h"

void simulate_annealing_cuda(double *solution, int size, double lo, 
		double hi, double sigma, float *msec, int choice);

/* @brief: returns a random double in [lo, hi]
 * @para[in]: lo. lower bound of the interval
 * @para[in]: hi. higher bound of the interval
 * @para[in]: seed. random seed to RNG.
 * @return: a random double in that interval
 **/
double rand_double(double lo, double hi, unsigned int* seed) {
	double rand_val= ((double) rand_r(seed)) / (double) RAND_MAX;
	double diff = hi - lo;
	rand_val *= diff;
	return lo + rand_val;
}

/* @brief: randomly initialize solution in range [lo, hi]
 * @para[in]: solution. Pointer to the array of solution
 * @para[in]: size. Size of the solution.
 * @para[in]: lo. lower bound of the interval
 * @para[in]: hi. higher bound of the interval
 * @para[in]: seed. random seed to RNG.
 **/
void init_solution(double *solution, int size, double lo, double hi, unsigned int* seed) {
	for (int i = 0; i < size; ++i) {
		solution[i] = rand_double(lo, hi, seed);
	}
}

/* @brief. Function returns a random variable follows ~ N(0,1)
 * @para[in]: seed. random seed to RNG.
 * @return. Randomly sampled value.
 **/
double unit_normal(unsigned int* seed) {
	double r,v1,v2,fac;
	r = 2;
	while (r >= 1) {
		v1 = 2 * ((double)rand_r(seed)/(double)RAND_MAX)-1;
		v2 = 2 * ((double)rand_r(seed)/(double)RAND_MAX)-1;
		r = v1 * v1 + v2 * v2;
	}
	fac = sqrt(-2 * log(r) / r);
	return v2 * fac;
}

/* @brief. Add a random sampled value to a given idx of solution
 * @para[in]: solution_idx. Pointer to the solution array of specific idx.
 * @para[in]: lo. lower bound. Perform value clip.
 * @para[in]: hi. higher bound. Perform value clip.
 * @para[in]: sigma. Std of normal distribution.
 * @para[in]: seed. random seed to RNG.
 * @para[in]: test_func. Pointer to the function for evaluation.
 **/
double rand_normal(double *solution_idx, double lo, double hi, double sigma,
	unsigned int *seed, std::function<double(double*, int)> test_func) {
	double rand_num = unit_normal(seed) * sigma;
	double diff = -test_func(solution_idx, 0);
	*solution_idx += rand_num;
	//clamp the value
	if (*solution_idx < lo) {
		*solution_idx = lo;
	} else if (*solution_idx > hi) {
		*solution_idx = hi;
	}

	diff += test_func(solution_idx, 0);
	return diff;
}

/* @brief. It implements SA method. At each iteration, it first performs MH
 * sampling method to acquire a new solution. We use a counter to track if a
 * better solution is received. If so, we reset the counter to zero, otherwise
 * we increment it. If the counter equals, say 200, we break the loop. It
 * indicates that is the past 200 iterations, we have not found a better
 * solution and it might reach a minima (global or local).
 * @para[in]: solution. Pointer to the solution array.
 * @para[in]: size. Size of the solution.
 * @para[in]: lo. lower bound. Perform value clip.
 * @para[in]: hi. higher bound. Perform value clip.
 * @para[in]: sigma. Standard deviation for normal distribution.
 * @para[in]: rand_seeds. random seeds for different procs.
 * @para[in]: test_func. Pointer to the function for evaluation.
 **/
void simulate_annealing(double *solution, int size, double lo, double hi, 
		double sigma, int p, unsigned int *rand_seeds,
		std::function<double(double*, int)> test_func) {
	#pragma omp parallel num_threads(p)
	{	
		int t = omp_get_thread_num();
		int tcount = omp_get_num_threads();
		int start = t * size / tcount;
		int end = (t+1) * size / tcount;
		if (end > size) end = size;
		double *local_solution = new double[end - start];
		memcpy(local_solution, solution+start, sizeof(double)*(end-start));
		double temperature = 1.0;
		while (temperature >= 1e-6) {
			// steal idea from gibbs sampling
			// basically it samples dimension by dimesion
			for (int i = start; i < end; ++i) {
				// store original value
				double original_sol = local_solution[i-start];
				// sample by normal distribution
				double diff = rand_normal(local_solution+i-start, lo, 
				hi, sigma, rand_seeds+t, test_func);
				if (diff > 0) {
					// if it is not good
					double alpha = rand_double(0.0, 1.0, rand_seeds+t);
					// accpet with prob
					double prob = exp(-diff / temperature);
					if (alpha > prob) {
						//restore original solution and increment counter
						local_solution[i-start] = original_sol;
						continue;
					}
				}
			}
			memcpy(solution+start, local_solution, sizeof(double) * (end-start));
			temperature *= 0.99999;
		}
		delete []local_solution;
	}
}

int main(int argc, char **argv) {
	int p = 1;
	int size = 1;
	int choice = 1;
	char *func = NULL;
	struct timespec before, after;
	bool show_solution = false;
	float msec;
	std::function<double(double*, int)> test_func = rastrigin;
	// create the randomly init interval of the solution
	// i.e. each element of solution is randomly chosen in [lo,hi]
	double lo = -100.0;
	double hi = 100.0;
	// set rando seed to current time
	srand(time(NULL));
	// parse the arguments
	for (int i = 0; i < argc; ++i) {
		if (!strcmp(argv[i], "-h")) {
			print_usage();
			return 0;
		}
		if (!strcmp(argv[i], "-n")) {
			size = atoi(argv[i+1]);
		}
		if (!strcmp(argv[i], "-s")) {
			show_solution = true;
		}
		if (!strcmp(argv[i], "-p")) {
			p = atoi(argv[i+1]);
		}
		if (!strcmp(argv[i], "-f")) {
			func = argv[i+1];
			if (!strcmp("ackley", argv[i+1])) {
				test_func = ackley;
				choice = 2;
			}
		}
	}
	// print configurature of the program
	print_info(p, size, func, lo, hi);
	// init different random seeds for different procs
	unsigned int *rand_seeds = new unsigned int[p];
	for (int i = 0; i < p; ++i) {
		rand_seeds[i] = static_cast<unsigned int>(rand());
	}
	// create solution array 
	double *solution = new double[size];
	double *solution_cu = new double[size];
	// sigma is the std of random normal distribution
	double sigma = 2.0;
	// randomly init the solution array
	init_solution(solution, size, lo, hi, rand_seeds);
	memcpy(solution_cu, solution, sizeof(double) * size);
	
	// record time
  	clock_gettime(CLOCK_REALTIME, &before);
	// perform SA
	simulate_annealing(solution, size, lo, hi, sigma, p, rand_seeds, test_func);
  	clock_gettime(CLOCK_REALTIME, &after);

	// perform SA cuda
	simulate_annealing_cuda(solution_cu, size, lo, hi, sigma, &msec, choice);

	//time for cpu version
	double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
	print_result(solution, solution_cu, delta_ms, msec, size, show_solution, test_func);
	// free heap space
	delete []solution;
	delete []solution_cu;
	return 0;
}
