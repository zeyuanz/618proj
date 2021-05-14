/*
 * =====================================================================================
 *
 *       Filename:  wsp_SA_gpu_coord.cpp
 *
 *    Description:  Test the performance of GPU with coordinates in shared memory.
 *
 *        Created:  04/30/21
 *       Compiler:  gcc
 *
 *         Author:  zeyuan zuo & zeruizhi cheng
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
#include <iomanip>
#include <iostream>

typedef struct path_struct_t
{
	float cost; // best path cost.
	int *path;	// best order of city visits
} path_t;

/* @brief: print usage of the program
 **/
void print_usage()
{
	printf("Usage ./wsp_SA_serial[options]\n");
	printf("Program Options:\n");
	printf("	-h			--print this message\n");
	printf("	-f	<STRING>	--file contains distance matrix\n");
	printf("	-p	<INT>		--number of processors, should be positive\n");
	printf("	-t			--output running time\n");
}
void wsp_print_result(path_t *costBestPath, int n_cities)
{
	printf("========== Solution ==========\n");
	printf("Cost: %d\n", (int)costBestPath->cost);
	printf("Path: ");
	for (int i = 0; i < n_cities; i++)
	{
		if (i == n_cities - 1)
			printf("%d", costBestPath->path[i]);
		else
			printf("%d -> ", costBestPath->path[i]);
	}
	putchar('\n');
	putchar('\n');
	return;
}
void print_result(path_t *solution_cu,
				  float gpu_ms, int n_cities)
{
	printf("\n\n");
	printf("=========== Result ===========\n");
	printf("------------------------------------------------------\n");
	std::cout << "TIME (sec) |"
			  << std::setprecision(8) << gpu_ms / 1000.0 << std::setw(13) << "|"
			  << std::endl;
	printf("-----------------------GPU----------------------------\n");
	wsp_print_result(solution_cu, n_cities);
}

void wsp_simulate_annealing_cuda(path_t *solution, int n_cities, float *dist, float *msec);

typedef struct
{
	char name[20];
	char comment[50];
	char type[10];
	int dimension;
	char wtype[10];
} InstanceData;

InstanceData ins;

float *readEuc2D(char *name)
{
	FILE *file;
	float *coord;
	int i;
	file = fopen(name, "r");
	fscanf(file, "NAME: %[^\n]s", ins.name);
	fscanf(file, "\nTYPE: TSP%[^\n]s", ins.type);
	fscanf(file, "\nCOMMENT: %[^\n]s", ins.comment);
	fscanf(file, "\nDIMENSION: %d", &ins.dimension);
	fscanf(file, "\nEDGE_WEIGHT_TYPE: %[^\n]s", ins.wtype);
	fscanf(file, "\nNODE_COORD_SECTION");
	coord = (float *)malloc(ins.dimension * sizeof(float *) * 2);
	for (i = 0; i < ins.dimension; i++)
		fscanf(file, "\n %*[^ ] %f %f", &coord[i * 2], &coord[i * 2 + 1]);
	fclose(file);
	return coord;
}

int main(int argc, char **argv)
{
	char *filename = NULL;
	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0)
		{
			print_usage();
			return 0;
		}
		if (strcmp(argv[i], "-f") == 0)
		{
			filename = argv[i + 1];
		}
	}

	float *coord = readEuc2D(filename);
	int n_cities = ins.dimension;

	path_t *global_cost_path_cu = (path_t *)malloc(sizeof(path_t));
	global_cost_path_cu->cost = 0;
	global_cost_path_cu->path = new int[n_cities];

	float msec;
	wsp_simulate_annealing_cuda(global_cost_path_cu, n_cities, coord, &msec);
	// print results
	print_result(global_cost_path_cu, msec, n_cities);
	// free heap alloc memory
	delete[] coord;
	free(global_cost_path_cu->path);
	free(global_cost_path_cu);
	return 0;
}