#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <iostream>
#include <cstring>

typedef struct
{
    char name[20];
    char comment[50];
    char type[10];
    int dimension;
    char wtype[10];
} InstanceData;

InstanceData ins;

double **readEuc2D(char *name)
{
    FILE *file;
    double **matrix;
    double **coord;
    int i, j;

    file = fopen(name, "r");
    fscanf(file, "NAME: %[^\n]s", ins.name);
    fscanf(file, "\nTYPE: TSP%[^\n]s", ins.type);
    fscanf(file, "\nCOMMENT: %[^\n]s", ins.comment);
    fscanf(file, "\nDIMENSION: %d", &ins.dimension);
    fscanf(file, "\nEDGE_WEIGHT_TYPE: %[^\n]s", ins.wtype);
    fscanf(file, "\nNODE_COORD_SECTION");

    coord = (double **)malloc(ins.dimension * sizeof(double *));
    for (i = 0; i < ins.dimension; i++)
        coord[i] = (double *)malloc(2 * sizeof(double));
    for (i = 0; i < ins.dimension; i++)
        fscanf(file, "\n %*[^ ] %lf %lf", &coord[i][0], &coord[i][1]);

    //Build Distance Matrix
    matrix = (double **)malloc(sizeof(double *) * (ins.dimension));
    for (i = 0; i < ins.dimension; i++)
        matrix[i] = (double *)malloc(sizeof(double) * (ins.dimension));

    for (i = 0; i < ins.dimension; i++)
    {
        for (j = i + 1; j < ins.dimension; j++)
        {
            matrix[i][j] = sqrt(pow(coord[i][0] - coord[j][0], 2) + pow(coord[i][1] - coord[j][1], 2));
            matrix[j][i] = matrix[i][j];
        }
    }
    free(coord);
    return matrix;
}

int main(int argc, char **argv)
{
    std::string name = "ch130.tsp";
    int n = name.length();
    char char_array[n + 1];
    strcpy(char_array, name.c_str());

    double **matrix = readEuc2D(char_array);
    FILE *file = fopen("dist130", "w");
    fprintf(file, "%d", ins.dimension);
    for (int j = 0; j < ins.dimension; j++)
    {
        for (int i = 0; i < j; i++)
            fprintf(file, "%f ", matrix[j][i]);
        fprintf(file, "\n");
    }
}