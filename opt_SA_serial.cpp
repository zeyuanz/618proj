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

/* @brief: print usage of the program
 **/
void print_usage() {
	printf("Usage ./opt_SA_serial[options]\n");
	printf("Program Options:\n");
	printf("	-h			--print this message\n");
	printf("	-p	<INT>		--number of processors, should be positive\n");
	printf("	-n	<INT>		--dimension of test function, should be positive\n");
	printf("	-t			--output running time\n");
}

/* @brief: a test function suggested by the paper 
 * Lou, Z., & Reinitz, J. (2016). Parallel simulated annealing using an adaptive
 * resampling interval. Parallel computing, 53, 23-31.
 * It has a global minimum at f(0,0,0,...0) = 0 and the number of 
 * local minima grows exponentially with n.
 * @para[in]: input. Input of x.
 * @para[in]: size. Size of the input
 * @return: function value given the input and size of input
 **/
double test_func(double *input, int size) {
	double first_term = 10 * static_cast<double>(size);
	double second_term = 0.0;
	for (int i = 0; i < size; ++i) {
		second_term += input[i] * input[i] - 10.0 * cos(2 * M_PI * input[i]);
	}
	return first_term + second_term;
}

/* @brief: returns a random double in [lo, hi]
 * @para[in]: lo. lower bound of the interval
 * @para[in]: hi. higher bound of the interval
 * @return: a random double in that interval
 **/
double rand_double(double lo, double hi) {
	double rand_val= ((double) rand()) / (double) RAND_MAX;
	double diff = hi - lo;
	rand_val *= diff;
	return lo + rand_val;
}

/* @brief: randomly initialize solution in range [lo, hi]
 * @para[in]: solution. Pointer to the array of solution
 * @para[in]: size. Size of the solution.
 * @para[in]: lo. lower bound of the interval
 * @para[in]: hi. higher bound of the interval
 **/
void init_solution(double *solution, int size, double lo, double hi) {
	for (int i = 0; i < size; ++i) {
		solution[i] = rand_double(lo, hi);
	}
}

void rand_normal(double *new_solution, double *solution, int size, double sigma) {

}

/* @brief. It implements MH sampling method. For each input solution and a
 * a given iteration:
 * 1. It generates guess in normal distirbution. Mean = solution, std = sigma
 * 2. If new solution is better, accept it.
 * 3. If new solution is worse, accept it with exp difference divided by
 * temperature.
 * @para[in]: solution. Pointer to the solution array.
 * @para[in]: size. Size of the solution.
 * @para[in]: temperature. Temperature used for evaluation exp prob.
 * @para[in]: sigma. Standard deviation for normal distribution.
 * @return: function value of the final solution.
 **/
double metropolis_hastings(double *solution, int size, int n_iter, double temperature, double sigma) {
	double *new_solution = new double[size];
	for (int i = 0; i < n_iter; ++i) {
		// generate new solution in normal distribution
		// mean = solution (old)
		// std = sigma
		rand_normal(new_solution, solution, size, sigma);
		double new_sol_val = test_func(new_solution, size);
		double sol_val = test_func(solution, size);
		if (new_sol_val < sol_val) {
			memcpy(solution, new_solution, sizeof(double) * size);
		} else {
			double diff = sol_val - new_sol_val;
			double alpha = rand_double(0.0, 1.0);
			double prob;
			if (temperature < 1e-12) {
				prob = 0.0;
			} else {
				prob =  exp(diff / temperature);
			}
			if (alpha < prob) {
				memcpy(solution, new_solution, sizeof(double) * size);
			}
		}
	}
	return test_func(solution, size);
}

/* @brief. It implements SA method. At each iteration, it first performs MH
 * sampling method to acquire a new solution. We use a counter to track if a
 * better solution is received. If so, we reset the counter to zero, otherwise
 * we increment it. If the counter equals, say 200, we break the loop. It
 * indicates that is the past 200 iterations, we have not found a better
 * solution and it might reach a minima (global or local).
 * @para[in]: solution. Pointer to the solution array.
 * @para[in]: size. Size of the solution.
 * @para[in]: sigma. Standard deviation for normal distribution.
 * @return: function value of the final solution.
 **/
double simulate_annealing(double *solution, int size, double sigma) {
	int cnt = 0;
	int n_iter = 100;
	double temperature = 0.1;
	double sol_val = test_func(solution, size);
	while (cnt < 200) {
		double new_sol_val = metropolis_hastings(solution, size, n_iter, temperature, sigma);
		if (new_sol_val < sol_val) {
			cnt = 0;
		} else {
			cnt++;
		}
		sol_val = new_sol_val;
		temperature *= 0.999999;
	}
	return sol_val;
}

void print_result(double *solution, int size, double sol_val, bool print_x) {
	printf("Final result: %.4f\n", sol_val);
	if (print_x) {
		printf("Solution is obtained at: \n");
		printf("(");
		for (int i = 0; i < size; ++i) {
			if (i != size-1) {
				printf("%.3f,", solution[i]);
			} else {
				printf("%.3f)\n", solution[i]);
			}
		}
	}
}

int main(int argc, char **argv) {
	int p = 1;
	int size = 1;
	struct timespec before, after;
	bool record_time = false;
	// set rando seed to current time
	srand(time(NULL));
	// parse the arguments
	for (int i=0; i < argc; i++) {
		if (strcmp(argv[i], "-h")==0) {
			print_usage();
			return 0;
		}
		if (strcmp(argv[i], "-p")==0) {
			p = atoi(argv[i+1]);
		}
		if (strcmp(argv[i], "-n")==0) {
			size = atoi(argv[i+1]);
		}
		if (strcmp(argv[i], "-t")==0) {
			record_time = true;
		}
	}
	// create solution array 
	double *solution = new double[size];
	// create the randomly init interval of the solution
	// i.e. each element of solution is randomly chosen in [lo,hi]
	double lo = -100.0;
	double hi = 100.0;
	double sigma = 3.0;


	// randomly init the solution array
	init_solution(solution, size, lo, hi);

  	clock_gettime(CLOCK_REALTIME, &before);
	double sol_val = simulate_annealing(solution, size, sigma);
  	clock_gettime(CLOCK_REALTIME, &after);

	if (record_time) {
  		double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
		printf("============ Time ============\n");
		printf("Time: %.3f ms (%.3f s)\n", delta_ms, delta_ms / 1000.0);
	}

	print_result(solution, size, sol_val, true);
	// free heap space
	delete []solution;
	return 0;
}
