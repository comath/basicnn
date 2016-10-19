#ifndef _selectiontrainer_h
#define _selectiontrainer_h
#include <iostream>
#include <armadillo>
#include <set>
#include <map>
#include <vector>
#include <queue>
#include <cmath>


#include "ann.h"
using namespace arma;
using namespace std;

typedef struct selector {
	vec v;
	double b;
} selector;

std::vector<vec> createSelectionTrainingData(vec_data *data, nn *thisnn,vec newhpNormal, double newhpOffset);
selector remakeSelector(selector oldselector, std::vector<vec> regionData, vec rx);

#endif