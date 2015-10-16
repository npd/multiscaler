#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include "iio.h"

int main(int argc, char *argv[]) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s prefix levels suffix output\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  char *input_prefix = argv[1];
  int levels = atoi(argv[2]);
  char *input_suffix = argv[3];
  char *output_name = argv[4];

  // Use the bigger image to determine width, height and channels
  int width, height, channels;
  char *filename;
  asprintf(&filename, "%s%d%s", input_prefix, 0, input_suffix);
  float *image = iio_read_image_float_vec(filename, &width, &height, &channels);
  free(image);
  free(filename);

  float *freq = (float *) fftwf_malloc(sizeof(float) * width * height * channels);
  int shape[] = {height, width};
  fftwf_plan plan;

  for (int i = 0; i < levels; ++i) {
    int w, h;
    asprintf(&filename, "%s%d%s", input_prefix, i, input_suffix);
    image = iio_read_image_float_vec(filename, &w, &h, &channels);
    free(filename);

    // Normalization
    for (int j = 0; j < w * h * channels; ++j) {
      image[j] /= 4 * w * h;
    }
    // DCT on the upper-left w x h rectangle
    int n[] = {h, w};
    fftwf_r2r_kind dct2[] = {FFTW_REDFT10, FFTW_REDFT10};
    plan = fftwf_plan_many_r2r(2, n, channels, image, NULL, channels, 1, freq, shape, channels, 1, dct2, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    free(image);
  }

  // IDCT of the output image
  float *output = (float *) fftwf_malloc(sizeof(float) * width * height * channels);
  fftwf_r2r_kind idct2[] = {FFTW_REDFT01, FFTW_REDFT01};
  plan = fftwf_plan_many_r2r(2, shape, channels, freq, shape, channels, 1, output, NULL, channels, 1, idct2, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);

  iio_save_image_float_vec(output_name, output, width, height, channels);
  fftwf_free(freq);
  fftwf_free(output);
  return 0;
}
