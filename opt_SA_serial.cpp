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
#include <random>

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
		second_term += (input[i] * input[i]);
		second_term -= 10.0 * cos(2.0 * M_PI * input[i]);
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

/* @brief. Function returns a random variable follows ~ N(0,1)
 * @return. Randomly sampled value.
 **/
double unit_normal() {
	double r,v1,v2,fac;
	r = 2;
	while (r >= 1) {
		v1 = 2 * ((double)rand()/(double)RAND_MAX)-1;
		v2 = 2 * ((double)rand()/(double)RAND_MAX)-1;
		r = v1 * v1 + v2 * v2;
	}
	fac = sqrt(-2 * log(r) / r);
	return v2 * fac;
}

/* @brief. Add a random sampled value to a given idx of solution
 * @para[in]: solution. Pointer to the solution array.
 * @para[in]: idx. The index of solution that needs to be changed.
 * @para[in]: lo. lower bound. Perform value clip.
 * @para[in]: hi. higher bound. Perform value clip.
 * @para[in]: sigma. Std of normal distribution.
 **/
void rand_normal(double *solution, int idx, double lo, double hi, double sigma) {
	double rand_num = unit_normal() * sigma;
	solution[idx] += rand_num;
	if (solution[idx] > hi) {
		solution[idx] = hi;
	}
	if (solution[idx] < lo) {
		solution[idx] = lo;
	}
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
double simulate_annealing(double *solution, int size, double lo, double hi, double sigma) {
	double sol_val = test_func(solution, size);
	double temperature = 1.0;
	int cnt = 0;
	while (cnt < 3000 && temperature > 1e-12) {
		// steal idea from gibbs sampling
		// basically it samples dimension by dimesion
		// 1|2,3,...,n
		// 2|1,3,...,n
		// i|1,2,...i-1,i+1,...,n
		for (int i = 0; i < size; ++i) {
			// store original value
			double original_val = solution[i];
			// sample by normal distribution
			rand_normal(solution, i, lo, hi, sigma);
			// retrieve new function value given new solution
			double new_sol_val = test_func(solution, size);
			if (new_sol_val < sol_val) {
				// if it is better, keep it and reset counter
				sol_val = new_sol_val;
				cnt = 0;
			} else {
				// if it is not good
				double diff = sol_val - new_sol_val;
				double alpha = rand_double(0.0, 1.0);
				// accpet with prob
				double prob = exp(diff / temperature);
				if (alpha < prob) {
					// if accept, keep new value and reset counter
					sol_val = new_sol_val;
					cnt = 0;
				} else {
					// otherwise, restore original solution and increment
					// counter
					solution[i] = original_val;
					cnt++;
				}
			}
		}
		// perform annealing
		temperature *= 0.999999;
	}
	return sol_val;
}

/* @brief. Print the result of solution.
 * @para[in]: solution. Pointer to the solution array.
 * @para[in]: size. Size of solution.
 * @para[in]: sol_val. Final optimized value.
 * @para[in]: print_x. Whether to print all solutions.
 **/
void print_result(double *solution, int size, double sol_val, bool print_x) {
	printf("=========== Result ===========\n");
	printf("Final result: %.4f\n", sol_val);
	if (print_x) {
		printf("Solution is obtained at: \n");
		printf("(");
		for (int i = 0; i < size; ++i) {
			if (i != size-1) {
				printf("%.2f,", solution[i]);
			} else {
				printf("%.2f)\n", solution[i]);
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
	double lo = -5.12;
	double hi = 5.12;
	// sigma is the std of random normal distribution
	double sigma = 2.0;
	// randomly init the solution array
	init_solution(solution, size, lo, hi);
	// record time
  	clock_gettime(CLOCK_REALTIME, &before);
	// perform SA
	double sol_val = simulate_annealing(solution, size, lo, hi, sigma);
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
