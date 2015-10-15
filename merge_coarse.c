#include <fftw3.h>
#include <stdlib.h>
#include <math.h>
#include "iio.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s image coarse result\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  char *input_name = argv[1];
  char *coarse_name = argv[2];
  char *output_name = argv[3];

  // Read the input image
  int width, height, channels;
  float *image = iio_read_image_float_vec(input_name, &width, &height, &channels);

  // Normalization
  for (int j = 0; j < width * height * channels; ++j) {
    image[j] /= 4 * width * height;
  }

  // DCT of the input image
  float *freq = (float *) fftwf_malloc(sizeof(float) * width * height * channels);
  int shape[] = {height, width};
  fftwf_r2r_kind dct2[] = {FFTW_REDFT10, FFTW_REDFT10};
  fftwf_plan plan = fftwf_plan_many_r2r(2, shape, channels, image, shape, channels, 1, freq, NULL, channels, 1, dct2, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);

  // We don't need the input image anymore
  free(image);

  // Read the low frequencies
  int coarse_width, coarse_height, coarse_channels;
  image = iio_read_image_float_vec(coarse_name, &coarse_width, &coarse_height, &coarse_channels);

  // Normalization
  for (int j = 0; j < coarse_width * coarse_height * channels; ++j) {
    image[j] /= 4 * coarse_width * coarse_height;
  }

  // DCT on the upper-left w x h rectangle
  int n[] = {coarse_height, coarse_width};
  plan = fftwf_plan_many_r2r(2, n, channels, image, NULL, channels, 1, freq, shape, channels, 1, dct2, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  free(image);

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
