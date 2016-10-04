#ifndef _nnanalyzer_h
#define _nnanalyzer_h
#include <armadillo>

using namespace arma;
using namespace std;



double ** adaptivebackprop(nn *nurnet, vec_data *D, double rate, double objerr, int max_gen, int max_nodes, bool ratedecay, bool image);

#endif