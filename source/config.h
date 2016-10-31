#ifndef _config_h
#define _config_h
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

using namespace std;
class param {
public:
    bool CREATE_VIDEO;
    char *IMAGE_FILE;
    int NUM_GENERATIONS;
    double SLOPE_THRESHOLD;
    int FORCED_DELAY;
    int RESOLUTION;
    int RUNAVGWID;
    int NUM_NN;
    int MAX_NODES;
    int NUM_DATA;
    int MAX_DATA_RESAMPLES;
    int START_NODES;
    int MAX_THREADS;
    double SCALED_TUBE_THREADSHOLD;

    param(const char *filename);
};
#endif