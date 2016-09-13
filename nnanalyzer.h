#ifndef _nnanalyzer_h
#define _nnanalyzer_h
#include <armadillo>

using namespace arma;
using namespace std;

typedef struct hp {
	vec v;
	double b;
} hp;

std::vector<int> getInterSig(vec v, mat A, vec b);
std::vector<int> getRegionSig(vec v, mat A, vec b);
double ** adaptivebackprop(nn *nurnet, vec_data *D, double rate, double objerr, int max_gen, int max_nodes, bool ratedecay);

#endif