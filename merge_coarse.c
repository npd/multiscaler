#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "iio.h"
#include "multiscaler.h"

int main(int argc, char *argv[]) {
  float gauss_s = (float) atof(pick_option(&argc, argv, "g", "-1"));
  bool gauss = gauss_s > 0.f;
  float tukey_a = (float) atof(pick_option(&argc, argv, "t", "-1"));
  bool tukey = tukey_a > 0.f;
  bool conservative = pick_option(&argc, argv, "c", NULL) != NULL;

  if ((argc != 4) || (gauss && tukey)) {
    fprintf(stderr, "Usage: %s image coarse result [-g sigma | -t alpha [-c]]\n", argv[0]);
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
      float factor = 1.f;
      if (gauss) {
        const float pi2sigma2 = (float) (M_PI * M_PI) * gauss_s * gauss_s;
        factor = expf(-pi2sigma2 * (j * j / (2.f * ch * ch) + k * k / (2.f * cw * cw)));
      } else if (tukey) {
        if (conservative) {
          if (j > ch * tukey_a || k > cw * tukey_a) continue;
        } else {
          if (j > ch * tukey_a) factor *= .5f * (1 + cosf(M_2_PI * (j / ch - .5f)));
          if (k > cw * tukey_a) factor *= .5f * (1 + cosf(M_2_PI * (k / cw - .5f)));
        }
      }
      for (int l = 0; l < fc; ++l) {
        fine[fw * fc * j + fc * k + l] *= 1.f - factor;
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
