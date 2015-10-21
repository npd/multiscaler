//
// Created by Nicola Pierazzo on 21/10/15.
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "multiscaler.h"
#include "iio.h"

int main(int argc, char *argv[]) {
  float sigma = atof(pick_option(&argc, argv, "g", "-1"));
  int n = atoi(pick_option(&argc, argv, "n", "5"));
  int usage = pick_option(&argc, argv, "h", NULL) != NULL;
  if (usage) {
    fprintf(stderr, "Usage: %s [output] [-g sigma] [-n size]\n", argv[0]);
    exit(EXIT_SUCCESS);
  }
  char *output = argc > 1 ? argv[1] : "-";

  float *cov_mat = malloc(n * n * n * n * sizeof(float));
  int ind = 0;
  for (int i1 = 0; i1 < n; ++i1) {
    for (int j1 = 0; j1 < n; ++j1) {
      for (int i2 = 0; i2 < n; ++i2) {
        for (int j2 = 0; j2 < n; ++j2) {
          // Covariance between the pixels (i1, j1) and (i2, j2)
          int dsq = (i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2);
          if (sigma > 0.) {
            cov_mat[ind] = expf(-dsq / (4.f * sigma * sigma)) / (M_PI * 4.f * sigma * sigma);
          } else {
            cov_mat[ind] = dsq ? 0 : 1;
          }
          ++ind;
        }
      }
    }
  }
  iio_save_image_float_vec(output, cov_mat, n * n, n * n, 1);

  return EXIT_SUCCESS;
}