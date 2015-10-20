//
// Created by Nicola Pierazzo on 20/10/15.
//

#ifndef MULTISCALER_MULTISCALER_H
#define MULTISCALER_MULTISCALER_H

#define STDDEV 0.7f

void dct_inplace(float *data, int w, int h, int c);
void idct_inplace(float *data, int w, int h, int c);
const char *pick_option(int *c, char **v, const char *o, const char *d);

#endif //MULTISCALER_MULTISCALER_H
