#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include "iio.h"
#include "multiscaler.h"

int main(int argc, char *argv[]) {
  float sigma = (float) atof(pick_option(&argc, argv, "g", "-1"));
  float ratio = (float) atof(pick_option(&argc, argv, "r", "2."));
  if (argc != 5) {
    fprintf(stderr, "Usage: %s input prefix levels suffix [-r ratio] [-g sigma]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  char *input = argv[1];
  char *output_prefix = argv[2];
  int levels = atoi(argv[3]);
  char *output_suffix = argv[4];

  // Read the input image
  int width, height, c;
  float *image = iio_read_image_float_vec(input, &width, &height, &c);

  // DCT of the input image
  dct_inplace(image, width, height, c);
  int w = width;
  int h = height;
  float *output = (float *) fftwf_malloc(sizeof(float) * width * height * c);
  for (int i = 0; i < levels; ++i) {
    // Copy data
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        for (int l = 0; l < c; ++l) {
          output[w * c * j + c * k + l] = image[width * c * j + c * k + l];
          // Gaussian blur if not in level zero
          if ((sigma > 0.f) && i) {
            const float pi2sigma2 = (float) (M_PI * M_PI) * sigma * sigma;
            output[w * c * j + c * k + l] *= expf( -pi2sigma2 * (j * j / (2.f * h * h) + k * k / (2.f * w * w)));
          }
        }
      }
    }

    // Inverse DCT
    idct_inplace(output, w, h, c);

    char *filename;
    asprintf(&filename, "%s%d%s", output_prefix, i, output_suffix);
    iio_save_image_float_vec(filename, output, w, h, c);
    free(filename);

    w = (int) (w / ratio);
    h = (int) (h / ratio);
  }

  free(image);
  fftwf_free(output);
  return EXIT_SUCCESS;
}
