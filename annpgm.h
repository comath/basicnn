#ifndef _anngpm_h
#define _annpgm_h
#include <iostream>
#include <random>
#include <math.h>
#include <iomanip>
#include <pthread.h>
#include <sys/stat.h>
#include "pgmreader.h"


using namespace arma;
using namespace std;

vec_data *get_vec_data_ppm(pm_img *img, int numdata);
void write_nn_to_img(nn *thisnn, const char filename[], int height, int width, int func);

#endif