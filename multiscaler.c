//
// Created by Nicola Pierazzo on 20/10/15.
//

#include <string.h>
#include <fftw3.h>
#include "multiscaler.h"


void dct_inplace(float *data, int w, int h, int c) {
  int n[] = {h, w};
  fftwf_r2r_kind dct2[] = {FFTW_REDFT10, FFTW_REDFT10};
  fftwf_plan plan = fftwf_plan_many_r2r(2, n, c, data, NULL, c, 1, data, NULL,
                                        c, 1, dct2, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);

  // Normalization
  for (int i = 0; i < w * h * c; ++i) {
    data[i] /= 4 * w * h;
  }
}

void idct_inplace(float *data, int w, int h, int c) {
  int n[] = {h, w};
  fftwf_r2r_kind idct2[] = {FFTW_REDFT01, FFTW_REDFT01};
  fftwf_plan plan = fftwf_plan_many_r2r(2, n, c, data, NULL, c, 1, data, NULL,
                                        c, 1, idct2, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
}

void dct1_inplace(float *data, int w, int h, int c) {
  int n[] = {h, w};
  fftwf_r2r_kind dct2[] = {FFTW_REDFT00, FFTW_REDFT00};
  fftwf_plan plan = fftwf_plan_many_r2r(2, n, c, data, NULL, c, 1, data, NULL,
                                        c, 1, dct2, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
}

const char *pick_option(int *c, char **v, const char *o, const char *d) {
  int id = d ? 1 : 0;
  for (int i = 0; i < *c - id; i++) {
    if (v[i][0] == '-' && 0 == strcmp(v[i] + 1, o)) {
      char *r = v[i + id] + 1 - id;
      for (int j = i; j < *c - id; j++)
        v[j] = v[j + id + 1];
      *c -= id + 1;
      return r;
    }
  }
  return d;
}