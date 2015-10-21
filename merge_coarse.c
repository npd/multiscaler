#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "iio.h"
#include "multiscaler.h"

int main(int argc, char *argv[]) {
  float sigma = atof(pick_option(&argc, argv, "g", "-1"));
  if (argc != 4) {
    fprintf(stderr, "Usage: %s image coarse result [-g]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  char *input_name = argv[1];
  char *coarse_name = argv[2];
  char *output_name = argv[3];

  // Read the fine image
  int fw, fh, fc;
  float *fine = iio_read_image_float_vec(input_name, &fw, &fh, &fc);

  // DCT of the fine image
  dct_inplace(fine, fw, fh, fc);

  // Read the low frequencies
  int cw, ch, cc;
  float *coarse = iio_read_image_float_vec(coarse_name, &cw, &ch, &cc);
  assert(fc == cc);

  // DCT of the low frequencies
  dct_inplace(coarse, cw, ch, cc);

  // Copy data
  for (int j = 0; j < ch; ++j) {
    for (int k = 0; k < cw; ++k) {
      float factor = 0.f;
      if (sigma > 0.f) {
        const float pi2sigma2 = (float) (M_PI * M_PI) * sigma * sigma;
        factor = 1.f - expf(-pi2sigma2 * (j * j / (2.f * ch * ch) + k * k / (2.f * cw * cw)));
      }
      for (int l = 0; l < fc; ++l) {
        fine[fw * fc * j + fc * k + l] *= factor;
        fine[fw * fc * j + fc * k + l] += coarse[cw * fc * j + fc * k + l];
      }
    }
  }
  free(coarse);

  // IDCT of the output image
  idct_inplace(fine, fw, fh, fc);

  iio_save_image_float_vec(output_name, fine, fw, fh, fc);
  free(fine);
  return EXIT_SUCCESS;
}
