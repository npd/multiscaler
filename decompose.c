#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include "iio.h"

int main(int argc, char *argv[]) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s input prefix levels suffix\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  char *input = argv[1];
  char *output_prefix = argv[2];
  int levels = atoi(argv[3]);
  char *output_suffix = argv[4];

  // Read the input image
  int width, height, channels;
  float *image = iio_read_image_float_vec(input, &width, &height, &channels);

  // DCT of the input image
  float *freq = (float *) fftwf_malloc(sizeof(float) * width * height * channels);
  int shape[] = {height, width};
  fftwf_r2r_kind dct2[] = {FFTW_REDFT10, FFTW_REDFT10};
  fftwf_plan plan = fftwf_plan_many_r2r(2, shape, channels, image, shape, channels, 1, freq, NULL, channels, 1, dct2, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);

  // We don't need the input image anymore
  free(image);

  // Normalization
  for (int i = 0; i < width * height * channels; ++i) {
    freq[i] /= 4 * width * height;
  }

  float *output = (float *) fftwf_malloc(sizeof(float) * width * height * channels);
  for (int i = 0; i < levels; ++i) {
    int w = width >> i;
    int h = height >> i;

    // Inverse DCT on the upper-left w x h rectangle
    int n[] = {h, w};
    fftwf_r2r_kind idct2[] = {FFTW_REDFT01, FFTW_REDFT01};
    plan = fftwf_plan_many_r2r(2, n, channels, freq, shape, channels, 1, output, NULL, channels, 1, idct2, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    char *filename;
    asprintf(&filename, "%s%d%s", output_prefix, i, output_suffix);
    iio_save_image_float_vec(filename, output, w, h, channels);
    free(filename);
  }

  fftwf_free(freq);
  fftwf_free(output);
  return 0;
}
