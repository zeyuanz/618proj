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
		double[i] = rand_double(lo, hi);
	}
}

double metropolis_hastings() {

}

double simulate_annealing() {

}

int main(int argc, char **argv) {
	int p = 1;
	int n = 1;
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
			n = atoi(argv[i+1]);
		}
		if (strcmp(argv[i], "-t")==0) {
			record_time = true;
		}
	}
	// create solution array 
	double *solution = new double[n];
	// create the randomly init interval of the solution
	// i.e. each element of solution is randomly chosen in [lo,hi]
	double lo = -100.0;
	double hi = 100.0;

	// randomly init the solution array
	init_solution(solution, n, lo, hi);

	// free heap space
	delete []solution;
	return 0;
}
