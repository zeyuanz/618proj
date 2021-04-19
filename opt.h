/*
 * =====================================================================================
 *
 *       Filename:  opt.h
 *
 *    Description:  header files include all test functions
 *
 *        Created:  04/18/21 22:29:18
 *
 *         Author:  zeyuan zuo
 *   Organization:  CMU
 *
 * =====================================================================================
 */

#include <math.h>

/* @brief: a test function suggested by the paper called rastrigin
 * Lou, Z., & Reinitz, J. (2016). Parallel simulated annealing using an adaptive
 * resampling interval. Parallel computing, 53, 23-31.
 * It has a global minimum at f(0,0,0,...0) = 0 and the number of 
 * local minima grows exponentially with n.
 * @para[in]: input. Input of x.
 * @para[in]: size. Size of the input. If size is 0, it only returns the value
 * that related to the input[0] in the function.
 * @return: function value given the input and size of input
 **/
double rastrigin(double *input, int size) {
	if (size == 0) {
		return input[0] * input[0] - 10.0 * cos(2.0 * M_PI * input[0]);
	}
	double first_term = 10 * static_cast<double>(size);
	double second_term = 0.0;
	for (int i = 0; i < size; ++i) {
		second_term += (input[i] * input[i]);
		second_term -= 10.0 * cos(2.0 * M_PI * input[i]);
	}
	return first_term + second_term;
}

/* @brief: a test function for non-convex opt performance called ackley 
 * It has a global minimum at f(0,0,0,...0) = 0
 * @para[in]: input. Input of x.
 * @para[in]: size. Size of the input. If size is 0, it only returns the value
 * that related to the input[0] in the function.
 * @return: function value given the input and size of input
 **/
double ackley(double *input, int size) {
	if (size == 0) {
		return -20.0 * exp(-0.2 * sqrt(0.5 * input[0] * input[0]))
			-exp(0.5 * cos(2.0 * M_PI * input[0]));
	}
	double square_term = 0.0;
	double cosine_term = 0.0;
	for (int i = 0; i < size; ++i) {
		square_term += input[i] * input[i];
		cosine_term += cos(2.0 * M_PI * input[i]);
	}
	double first_term = -20.0 * exp(-0.2 * sqrt(0.5 * square_term));
	double second_term = -exp(cosine_term / double(size)) + exp(1.0) + 20.0;
	return first_term + second_term;
}
