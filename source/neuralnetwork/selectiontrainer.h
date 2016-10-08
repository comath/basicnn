#ifndef _selectiontrainer_h
#define _selectiontrainer_h
#include <armadillo>

using namespace arma;
using namespace std;

typedef struct selector {
	vec v;
	double b;
} selector;

#endif