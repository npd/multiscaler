//
// Created by Nicola Pierazzo on 20/10/15.
//

#ifndef MULTISCALER_MULTISCALER_H
#define MULTISCALER_MULTISCALER_H

#include <stdlib.h>

#define SMART_PARAMETER_INT(n, v) static int n(void)\
{\
  static int smapa_known_ ## n = 0;\
  static int smapa_value_ ## n = v;\
  if (!smapa_known_ ## n) {\
    int r;\
    char *sv = getenv(#n);\
    int y;\
    if (sv)\
      r = sscanf(sv, "%d", &y);\
    if (sv && r == 1)\
      smapa_value_ ## n = y;\
    smapa_known_ ## n = 1;\
  }\
  return smapa_value_ ## n;\
}

void dct_inplace(float *data, int w, int h, int c);
void idct_inplace(float *data, int w, int h, int c);
void dct1_inplace(float *data, int w, int h, int c);
const char *pick_option(int *c, char **v, const char *o, const char *d);

#endif //MULTISCALER_MULTISCALER_H
