/*
 * =====================================================================================
 *
 *       Filename:  wsp.h
 *
 *    Description:  library for some common functions of wsp
 *
 *        Created:  04/11/21 11:38:39
 *
 *         Author:  zeyuan zuo
 *   Organization:  CMU
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>

typedef struct path_struct_t
{
  double cost; // best path cost.
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

