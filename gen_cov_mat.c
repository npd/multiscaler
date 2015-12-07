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
  float gauss_s = (float) atof(pick_option(&argc, argv, "g", "-1"));
  bool gauss = gauss_s > 0.f;
  float tukey_a = (float) atof(pick_option(&argc, argv, "t", "-1"));
  bool tukey = tukey_a > 0.f;
  int n = atoi(pick_option(&argc, argv, "n", "5"));
  int usage = pick_option(&argc, argv, "h", NULL) != NULL;
  if (usage || (gauss & tukey)) {
    fprintf(stderr, "Usage: %s [output] [-h] [-g sigma | -t alpha] [-n size]\n", argv[0]);
    exit(EXIT_SUCCESS);
  }
  char *output = argc > 1 ? argv[1] : "-";

  float *filter;
  int N = FILTER_SUPPORT();
  int w = N - 1, h = N - 1;
  filter = fftwf_malloc(N * N * sizeof(float));

  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k) {
      float factor = 1.f;
      if (gauss) {
        const float pi2sigma2 = (float) (M_PI * M_PI) * gauss_s * gauss_s;
        factor = expf(-pi2sigma2 * (j * j / (2.f * w * w) + k * k / (2.f * h * h)));
      } else if (tukey) {
        if (j > h * tukey_a) factor *= .5f * (1 + cosf(M_PI * (j / h - tukey_a) / (1 - tukey_a)));
        if (k > w * tukey_a) factor *= .5f * (1 + cosf(M_PI * (k / w - tukey_a) / (1 - tukey_a)));
      }
      filter[N * j + k] = factor * factor / (4 * w * h);
    }
  }

  dct1_inplace(filter, N, N, 1);

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
          if (gauss) {
            cov_mat[ind] = filter[N * di + dj];
          } else {
            cov_mat[ind] = (di + dj) ? 0 : 1;
          }
          ++ind;
        }
      }
    }
  }

  fftwf_free(filter);
  iio_save_image_float_vec(output, cov_mat, n * n, n * n, 1);
  free(cov_mat);

  return EXIT_SUCCESS;
}