#include <stdio.h>
#include <math.h>

#define MAXCITIES 20
#define RANDOMRANGE 100

struct {
  int x,y;
} cities[MAXCITIES];

double distances[MAXCITIES][MAXCITIES];

int numcities;
char filename[100];
FILE *fp, *fopen();
long random();
int skip_random;

main() 
{
  int i,j;

  scanf("%d", &numcities);

  scanf("%d", &skip_random);

  for (i = 0; i < skip_random; i++) {
    j = (int)random();
  }

  scanf("%s",filename);
  fp = fopen(filename, "w");

  for (i = 0; i < numcities; i++) {
    cities[i].x = (int)(random() % RANDOMRANGE) + 1;
    cities[i].y = (int)(random() % RANDOMRANGE) + 1;
    printf("City %2d: (%3d,%3d)\n",i,cities[i].x,cities[i].y);
  }

  for (i = 0; i < numcities; i++) {
    for (j = 0; j < numcities; j++) {
      double dx = cities[i].x - cities[j].x;
      double dy = cities[i].y - cities[j].y;
      distances[i][j] = sqrt(dx*dx + dy*dy);
    }
  }

  fprintf(fp,"%d",numcities);
  for (i = 0; i < numcities; i++) {
    for (j = 0; j < i; j++) {
      int idist = (int)distances[i][j];
      fprintf(fp,"%3d ",idist);
    }
    fprintf(fp,"\n");
  }
}
