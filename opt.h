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
#include <iomanip>
#include <iostream>

/* @brief: print usage of the program
 **/
void print_usage() {
	printf("Usage ./opt_SA_serial[options]\n");
	printf("Program Options:\n");
	printf("	-h			--print this message\n");
	printf("	-p	<INT>		--number of processors, should be positive\n");
	printf("	-n	<INT>		--dimension of test function, should be positive\n");
	printf("	-s			--output solutions for each dimension\n");
	printf("	-f	<string>	--functions for testing the performance\n");
	printf("	Valid options:	ackley\n");
	printf("			rosenbrock (CPU only) \n");
	printf("			levi (CPU only) \n");
	printf("			rastrigin (default)\n");
	printf("	-r	<string>	--devices for testing the performance, if not specified, use both cpu and cuda\n");
	printf("	Valid options:	cuda\n");
	printf("			cpu\n");
}

/* @brief. Print configuration.
 * @para[in]: p. Numprocs.
 * @para[in]: size. Size of solution (function).
 * @para[in]: func. Function for einput[idx]uation.
 * @para[in]: lo. lower bound. Perform input[idx]ue clip.
 * @para[in]: hi. higher bound. Perform input[idx]ue clip.
 **/
void print_info(int p, int size, char* func, double lo, double hi, char *dev) {
	printf("============ INFO ============\n");
	if (func) {
		printf("Function: %s\n", func);
	} else {
		printf("Function: rastrigin (default)\n");
	}
	if (dev) {
		printf("Device: %s\n", dev);
	} else {
		printf("Device: cpu, CUDA\n");
	}
	printf("Function dimesnion: %d\n", size);
	printf("Number of procs (CPU): %d\n", p);
	printf("Solution domain: [%.2f, %.2f]\n", lo, hi);
}

/* @brief. Print the result of solution.
 * @para[in]: solution. Pointer to the solution array.
 * @para[in]: size. Size of solution.
 * @para[in]: print_x. Whether to print all solutions.
 * @para[in]: test_func. Pointer to the test function for einput[idx]uation
 **/
void print_result(double *solution, double *solution_cu, double cpu_ms,
		float gpu_ms, int size, bool print_x,
	std::function<double(double*, int, double)> test_func) {
	printf("\n\n");
	printf("=========== Result ===========\n");
	printf("------------------------------------------------------\n");
	printf("           |CPU                |GPU                  |\n");
	printf("------------------------------------------------------\n");
	std::cout<<"TIME (sec) |"
		<<std::setprecision(8)<<cpu_ms/1000.0<<std::setw(10)<<"|"
		<<std::setprecision(8)<<gpu_ms/1000.0<<std::setw(13)<<"|"
		<<std::endl;
	printf("------------------------------------------------------\n");
	std::cout<<"SOL        |"
		<<std::setprecision(8)<<test_func(solution,size,-1)<<std::setw(7)<<"|"
		<<std::setprecision(8)<<test_func(solution_cu,size,-1)<<std::setw(9)<<"|"
		<<std::endl;
	printf("------------------------------------------------------\n");

	if (print_x) {
		printf("CPU Solution is obtained at: \n");
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
/* @brief: a test function suggested by the paper called rastrigin
 * Lou, Z., & Reinitz, J. (2016). Parallel simulated annealing using an adaptive
 * resampling interinput[idx]. Parallel computing, 53, 23-31.
 * It has a global minimum at f(0,0,0,...0) = 0 and the number of 
 * local minima grows exponentially with n.
 * @para[in]: input. Input of x.
 * @para[in]: size. Size of the input. If size is 0, it only returns the input[idx]ue
 * that related to the input[0] in the function.
 * @return: function input[idx]ue given the input and size of input
 **/
double rastrigin(double *input, int size, int idx) {
	if (idx != -1) {
		return input[idx] * input[idx] - 10.0 * cos(2.0 * M_PI * input[idx]);
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
 * @para[in]: size. Size of the input. If size is 0, it only returns the input[idx]ue
 * that related to the input[0] in the function.
 * @return: function input[idx]ue given the input and size of input
 **/
double ackley(double *input, int size, int idx) {
	if (idx != -1) {
		return -20.0 * exp(-0.2 * sqrt(0.5 * input[idx] * input[idx]))
			-exp(0.5 * cos(2.0 * M_PI * input[idx]));
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

// rosenbrock testing function
double rosenbrock(double *input, int size, int idx) {
	if (idx != -1) {
		if (idx == size-1) {
			return 100.0*(input[idx]*input[idx]-2.0*input[idx]*input[idx-1]*input[idx-1]);
		}
		return 100.0*(-2.0*input[idx+1]*input[idx]*input[idx]+pow(input[idx],4.0))-2.0*input[idx]+input[idx]*input[idx];
	}
	double first_term = 0.0;
	double second_term = 0.0;
	for (int i = 0; i < size-1; ++i) {
		first_term += 100.0*(input[i+1] - input[i] * input[i])*(input[i+1] - input[i] * input[i]);
		second_term += (1.0 - input[i])*(1.0 - input[i]);
	}
	return first_term + second_term;
}

// levi testing function
double levi(double *input, int size, int idx) {
	if (idx != -1){
		if (idx == 0) return pow(sin(M_PI*(input[0]+3.0)/4.0), 2.0);
		if (idx == size-1) return pow((input[size-1]-1)/4,2.0)*(1+pow(sin(2*M_PI*(input[size-1]+3)/4),2.0));
		return pow((input[idx]-1)/4, 2.0)*(1+10*pow(sin(M_PI*(1.0+(input[idx]-1)/4)+1),2.0));
	}
	double first_term = pow(sin(M_PI*(1.0+(input[0]-1)/4)), 2.0);
	double second_term = 0.0;
	double third_term = pow((input[size-1]-1)/4,2.0)*(1+pow(sin(2*M_PI*(input[size-1]+3)/4),2.0));
	for (int i = 1; i < size-1; i++) {
		second_term += pow((input[i]-1)/4, 2.0)*(1+10*pow(sin(M_PI*(1.0+(input[i]-1)/4)+1),2.0));
	}
	return first_term + second_term + third_term;
}
