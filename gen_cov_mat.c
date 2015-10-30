//
// Created by Nicola Pierazzo on 21/10/15.
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <fftw3.h>
#include "multiscaler.h"
#include "iio.h"

SMART_PARAMETER_INT(FILTER_SUPPORT, 100);

int main(int argc, char *argv[]) {
  float sigma = atof(pick_option(&argc, argv, "g", "-1"));
  int n = atoi(pick_option(&argc, argv, "n", "5"));
  int usage = pick_option(&argc, argv, "h", NULL) != NULL;
  if (usage) {
    fprintf(stderr, "Usage: %s [output] [-g sigma] [-n size]\n", argv[0]);
    exit(EXIT_SUCCESS);
  }
  char *output = argc > 1 ? argv[1] : "-";

  float *filter;
  int N = FILTER_SUPPORT();
  if (sigma > 0.f) {
    filter = fftwf_malloc(N * N * sizeof(float));
    const float pi2sigma2 = (float) (M_PI * M_PI) * sigma * sigma;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        filter[N * i + j] = expf(-pi2sigma2 * (i * i + j * j) / ((N - 1) * (N - 1))) / ((2 * N - 2) * (2 * N - 2));
      }
    }
    dct1_inplace(filter, N, N, 1);
  }

  float *cov_mat = malloc(n * n * n * n * sizeof(float));
  int ind = 0;
  for (int i1 = 0; i1 < n; ++i1) {
    for (int j1 = 0; j1 < n; ++j1) {
      for (int i2 = 0; i2 < n; ++i2) {
        for (int j2 = 0; j2 < n; ++j2) {
          // Covariance between the pixels (i1, j1) and (i2, j2)
          int di = i1 - i2;
          int dj = j1 - j2;
          di = (di < 0) ? -di : di;
          dj = (dj < 0) ? -dj : dj;
          if (sigma > 0.f) {
            cov_mat[ind] = filter[N * di + dj];
          } else {
            cov_mat[ind] = (di + dj) ? 0 : 1;
          }
          ++ind;
        }
      }
    }
  }

  iio_save_image_float_vec(output, cov_mat, n * n, n * n, 1);

  return EXIT_SUCCESS;
}