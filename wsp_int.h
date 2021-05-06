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

void set_dist(int* dist, int n_cities, int i, int j, int value) {
  int offset = i * n_cities + j;
  dist[offset] = value;
  return;
}

// returns value at dist[i,j]
int get_dist(int* dist, int n_cities, int i, int j) {
  int offset = i * n_cities + j;
  return dist[offset];
}

// prints results
void wsp_print_result(int* costBestPath, int n_cities) {
  printf("========== Solution ==========\n");
  printf("Cost: %d\n", costBestPath[0]);
  printf("Path: ");
  for(int i = 0; i < n_cities; i++) {
    if(i == n_cities-1) printf("%d", costBestPath[i+1]);
    else printf("%d -> ", costBestPath[i+1]);
  }
  putchar('\n');
  putchar('\n');
  return;
}

void read_dist(FILE *fp, int *dist, int n_cities) {
 for(int i = 1; i < n_cities;i++) {
    for(int j = 0;j < i;j++) {
      int t;
      fscanf(fp, "%d", &t);
      set_dist(dist, n_cities, i, j, t);
      set_dist(dist, n_cities, j, i, t);
    }
  }
}