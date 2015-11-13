#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include "iio.h"
#include "multiscaler.h"

int main(int argc, char *argv[]) {
  float gauss_s = (float) atof(pick_option(&argc, argv, "g", "-1"));
  bool gauss = gauss_s > 0.f;
  float tukey_a = (float) atof(pick_option(&argc, argv, "t", "-1"));
  bool tukey = tukey_a > 0.f;
  bool conservative = pick_option(&argc, argv, "c", NULL) != NULL;

  if ((argc != 5) || (gauss && tukey)) {
    fprintf(stderr, "Usage: %s prefix levels suffix output [-g sigma | -t alpha [-c]]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  char *input_prefix = argv[1];
  int levels = atoi(argv[2]);
  char *input_suffix = argv[3];
  char *output_name = argv[4];

  // Use the bigger image to determine width, height and number of channels
  int width, height, c;
  char *filename;
  asprintf(&filename, "%s%d%s", input_prefix, 0, input_suffix);
  float *output = iio_read_image_float_vec(filename, &width, &height, &c);
  free(filename);

  // Perform the DCT
  dct_inplace(output, width, height, c);

  for (int i = 1; i < levels; ++i) {
    // Read level i of the pyramid
    int w, h;
    asprintf(&filename, "%s%d%s", input_prefix, i, input_suffix);
    float *image = iio_read_image_float_vec(filename, &w, &h, &c);
    free(filename);

    // Perform the DCT
    dct_inplace(image, w, h, c);

    // Copy data
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        float factor = 1.f;
        if (gauss && i) {
          const float pi2sigma2 = (float) (M_PI * M_PI) * gauss_s * gauss_s;
          factor = expf(-pi2sigma2 * (j * j / (2.f * h * h) + k * k / (2.f * w * w)));
        } else if (tukey && i) {
          if (conservative) {
            if (j > h * tukey_a || k > w * tukey_a) continue;
          } else {
            if (j > h * tukey_a) factor *= .5f * (1 + cosf(M_2_PI * (j / h - .5f)));
            if (k > w * tukey_a) factor *= .5f * (1 + cosf(M_2_PI * (k / w - .5f)));
          }
        }
        for (int l = 0; l < c; ++l) {
          output[width * c * j + c * k + l] *= 1.f - factor;
          output[width * c * j + c * k + l] += image[w * c * j + c * k + l];
        }
      }
    }
    free(image);
  }

  // IDCT of the output image
  idct_inplace(output, width, height, c);

  iio_save_image_float_vec(output_name, output, width, height, c);
  fftwf_free(output);
  return EXIT_SUCCESS;
}
