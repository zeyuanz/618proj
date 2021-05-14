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
#include <random>
#include "opt.h"
//#include <boost/algorithm/clamp.hpp>

void simulate_annealing_cuda(double *solution, int size, double lo, 
		double hi, double sigma, float *msec, int choice);

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
void simulate_annealing(double *solution, int size, double lo, double hi, double sigma,
	   int p,std::function<double(double*, int, double)> test_func) {
	#pragma omp parallel num_threads(p)
	{	
		static thread_local std::mt19937 generator;
		std::uniform_real_distribution<double> uniform_dist(lo,hi);
		std::uniform_real_distribution<double> uniform_dist2(0.0,1.0);
		std::normal_distribution<double> normal_dist(0.0,sigma);
		int t = omp_get_thread_num();
		int tcount = omp_get_num_threads();
		int start = t * size / tcount;
		int end = (t+1) * size / tcount;
		if (end > size) end = size;
		for (int i = start; i < end; ++i) {
			double iter = 0.0;
			double temperature = 1.0;
			while (temperature >= 1e-6) {
					// steal idea from gibbs sampling
					// basically it samples dimension by dimesion
					double original_sol = solution[i];
					double diff = -test_func(solution, size, i);
					solution[i] += normal_dist(generator); 
					diff += test_func(solution, size, i);
					if (diff > 0) {
						// if it is not good
						double alpha = uniform_dist2(generator);
						// accpet with prob
						double prob = exp(-diff / temperature);
						if (alpha > prob) {
							//restore original solution and increment counter
							solution[i] = original_sol;
						}
					}
				temperature = 1.0 / (1.0 + 2.5 * iter);
				iter += 1.0;
			}
		}
		/*
		double *local_solution = new double[size];
		while (temperature >= 1e-6) {
			memcpy(local_solution, solution, sizeof(double)*size);
			for (int i = start; i < end; ++i) {
					// steal idea from gibbs sampling
					// basically it samples dimension by dimesion
					double original_sol = local_solution[i];
					double diff = -test_func(local_solution, size, i);
					local_solution[i] += normal_dist(generator); 
					diff += test_func(local_solution, size, i);
					if (diff > 0) {
						// if it is not good
						double alpha = uniform_dist2(generator);
						// accpet with prob
						double prob = exp(-diff / temperature);
						if (alpha > prob) {
							//restore original solution and increment counter
							local_solution[i] = original_sol;
						}
					}
				}
			temperature = 1.0 / (1.0 + 2.5 * iter);
			iter += 1.0;
			memcpy(solution+start, local_solution+start, sizeof(double)*(end-start));
		}
		delete []local_solution;
		*/ // this is more robust version dealing with much complex functions
	}
}

int main(int argc, char **argv) {
	int p = 1;
	int size = 1;
	int choice = 1;
	char *func = NULL;
	char *dev = NULL;
	struct timespec before, after;
	bool show_solution = false;
	float msec;
	std::function<double(double*, int, double)> test_func = rastrigin;
	// create the randomly init interval of the solution
	// i.e. each element of solution is randomly chosen in [lo,hi]
	double lo = -10.0;
	double hi = 10.0;
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
			if (!strcmp("rosenbrock", argv[i+1])) {
				test_func = rosenbrock;
			}
			if (!strcmp("levi", argv[i+1])) {
				test_func = levi;
			}
		}
		if (!strcmp(argv[i], "-r")) {
			dev = argv[i+1];
		}
	}
	// print configurature of the program
	print_info(p, size, func, lo, hi, dev);
	// create solution array 
	double *solution = new double[size];
	double *solution_cu = new double[size];
	// sigma is the std of random normal distribution
	for (int i = 0; i < size; i++) {
		solution[i] = lo + double(rand())/RAND_MAX * (hi - lo);
	}
	memcpy(solution_cu, solution, sizeof(double) * size);
	printf("Init sol: %.6f\n", test_func(solution, size, 0.0));
	double sigma = 1.5;

	if (dev == NULL || !strcmp(dev, "cpu")) {
		clock_gettime(CLOCK_REALTIME, &before);
		simulate_annealing(solution, size, lo, hi, sigma, p, test_func);
		clock_gettime(CLOCK_REALTIME, &after);
	}

	// perform SA cuda
	if (dev == NULL || !strcmp(dev, "cuda")) {
		simulate_annealing_cuda(solution_cu, size, lo, hi, sigma, &msec, choice);
	}

	//time for cpu version
	double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
	print_result(solution, solution_cu, delta_ms, msec, size, show_solution, test_func);
	// free heap space
	delete []solution;
	delete []solution_cu;
	return 0;
}
