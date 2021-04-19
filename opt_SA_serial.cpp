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

/* @brief: print usage of the program
 **/
void print_usage() {
	printf("Usage ./opt_SA_serial[options]\n");
	printf("Program Options:\n");
	printf("	-h			--print this message\n");
	printf("	-p	<INT>		--number of processors, should be positive\n");
	printf("	-n	<INT>		--dimension of test function, should be positive\n");
	printf("	-t			--output running time\n");
	printf("	-s			--output solutions for each dimension\n");
	printf("	-f	<string>	--functions for testing the performance\n");
	printf("	Valid options:	ackley\n");
	printf("			rastrigin\n");
	printf("			schwefel\n");
}

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
	   	double val, unsigned int *seed, std::function<double(double*, int)> test_func) {
	double rand_num = unit_normal(seed) * sigma;

	val -= test_func(solution_idx, 0);
	*solution_idx += rand_num;
	//clamp the value
	if (*solution_idx < lo) {
		*solution_idx = lo;
	} else if (*solution_idx > hi) {
		*solution_idx = hi;
	}

	val += test_func(solution_idx, 0);
	return val;
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
		for (int iter = 100000; iter > 0; iter--) {
			double temperature = 1.0 * iter / 1000000.0;
			// steal idea from gibbs sampling
			// basically it samples dimension by dimesion
			double sol_val = test_func(solution, size);
			for (int i = start; i < end; ++i) {
				// store original value
				double original_sol = local_solution[i-start];
				// sample by normal distribution
				double new_sol_val = rand_normal(local_solution+i-start, lo, 
				hi, sigma, sol_val, rand_seeds+t, test_func);
				if (new_sol_val > sol_val) {
					// if it is not good
					double diff = sol_val - new_sol_val;
					double alpha = rand_double(0.0, 1.0, rand_seeds+t);
					// accpet with prob
					double prob = exp(diff / temperature);
					if (alpha > prob) {
						//restore original solution and increment counter
						local_solution[i-start] = original_sol;
						continue;
					}
				}
				sol_val = new_sol_val;
			}
			memcpy(solution+start, local_solution, sizeof(double) * (end-start));
		}
		delete []local_solution;
	}
}

/* @brief. Print configuration.
 * @para[in]: p. Numprocs.
 * @para[in]: size. Size of solution (function).
 * @para[in]: func. Function for evaluation.
 * @para[in]: lo. lower bound. Perform value clip.
 * @para[in]: hi. higher bound. Perform value clip.
 **/
void print_info(int p, int size, char* func, double lo, double hi) {
	printf("============ INFO ============\n");
	if (func) {
		printf("Function: %s\n", func);
	} else {
		printf("Function: rastrigin (default)\n");
	}
	printf("Function dimesnion: %d\n", size);
	printf("Number of procs: %d\n", p);
	printf("Solution domain: [%.2f, %.2f]\n", lo, hi);
}

/* @brief. Print the result of solution.
 * @para[in]: solution. Pointer to the solution array.
 * @para[in]: size. Size of solution.
 * @para[in]: print_x. Whether to print all solutions.
 * @para[in]: test_func. Pointer to the test function for evaluation
 **/
void print_result(double *solution, int size, bool print_x,
	std::function<double(double*, int)> test_func) {
	printf("=========== Result ===========\n");
	printf("Final result: %.8f\n", test_func(solution, size));
	if (print_x) {
		printf("Solution is obtained at: \n");
		printf("(");
		for (int i = 0; i < size; ++i) {
			if (i != size-1) {
				printf("%.4f,", solution[i]);
			} else {
				printf("%.4f)\n", solution[i]);
			}
		}
	}
}

int main(int argc, char **argv) {
	int p = 1;
	int size = 1;
	char *func = NULL;
	struct timespec before, after;
	bool record_time = false;
	bool show_solution = false;
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
		if (!strcmp(argv[i], "-t")) {
			record_time = true;
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
	// sigma is the std of random normal distribution
	double sigma = 2.0;
	// randomly init the solution array
	init_solution(solution, size, lo, hi, rand_seeds);

	// record time
  	clock_gettime(CLOCK_REALTIME, &before);
	// perform SA
	simulate_annealing(solution, size, lo, hi, sigma, p, rand_seeds, test_func);
  	clock_gettime(CLOCK_REALTIME, &after);

	if (record_time) {
  		double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
		printf("============ Time ============\n");
		printf("Time: %.3f ms (%.3f s)\n", delta_ms, delta_ms / 1000.0);
	}
	print_result(solution, size, show_solution, test_func);
	// free heap space
	delete []solution;
	return 0;
}
