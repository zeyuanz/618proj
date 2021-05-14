/*
 * =====================================================================================
 *
 *       Filename:  wsp.h
 *
 *    Description:  library for some common functions of wsp
 *
 *        Created:  04/11/21 11:38:39
 *
 *         Author:  zeruizhi cheng
 *   Organization:  CMU
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>

typedef struct path_struct_t
{
  double cost; // path cost.
  int *path;   // best order of city visits
} path_t;

void set_dist(double *dist, int n_cities, int i, int j, double value)
{
  int offset = i * n_cities + j;
  dist[offset] = value;
  return;
}

// returns value at dist[i,j]
double get_dist(double *dist, int n_cities, int i, int j)
{
  int offset = i * n_cities + j;
 // printf("i is %d , j is %d, offset is %d\n", i, j, offset);
  return dist[offset];
}

// prints results
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

void read_dist(FILE *fp, double *dist, int n_cities)
{
  for (int i = 1; i < n_cities; i++)
  {
    for (int j = 0; j < i; j++)
    {
      double t;
      fscanf(fp, "%lf", &t);
      set_dist(dist, n_cities, i, j, t);
      set_dist(dist, n_cities, j, i, t);
    }
  }
}

void print_result(path_t *solution, path_t *solution_cu, double cpu_ms,
                  float gpu_ms, int n_cities)
{
  printf("\n\n");
  printf("=========== Result ===========\n");
  printf("------------------------------------------------------\n");
  printf("           |CPU                |GPU                  |\n");
  printf("------------------------------------------------------\n");
  std::cout << "TIME (sec) |"
            << std::setprecision(8) << cpu_ms / 1000.0 << std::setw(10) << "|"
            << std::setprecision(8) << gpu_ms / 1000.0 << std::setw(13) << "|"
            << std::endl;
  printf("-----------------------CPU----------------------------\n");
  wsp_print_result(solution, n_cities);
  printf("-----------------------GPU----------------------------\n");
  wsp_print_result(solution_cu, n_cities);
}

