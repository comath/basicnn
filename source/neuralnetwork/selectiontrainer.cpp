#include <iostream>
#include <armadillo>
#include <map>
#include <vector>
#include <queue>
#include <cmath>

#include "selectiontrainer.h"

using namespace std;
using namespace arma;

#define NUMDATAPOINTS 5000


bool compareVecs(vec x1, vec x2)
{
	
	if(x1.n_rows != x2.n_rows)
		return false;
	uvec comp = (x1 == x2);
	for(unsigned i = 0; i<x1.n_rows; ++i ){
		if(comp(i) == 0)
			return false;
	}
	return true;
}

class optimalFunction {
private:
	selector s;
	vec rx0;
	vec rx1;
public:
	optimalFunction()
	{
		s.v = zeros<vec>(1);
		s.b = 0;
		rx0 = zeros<vec>(1);
		rx1 = zeros<vec>(1);
		rx1(0) =1;
	}

	optimalFunction(selector constructorS, std::vector<int> constructorRx)
	{
		s = constructorS;
		rx0 = conv_to< vec >::from( constructorRx );
		rx1 = conv_to< vec >::from( constructorRx ); 
		vec temp = zeros<vec>(1);
		rx0.insert_rows(0,temp);
		s.v.insert_rows(0,temp);
		temp(0) = 1;
		rx1.insert_rows(0,temp);
		
	}
	optimalFunction(selector constructorS, vec constructorRx)
	{
		s = constructorS;
		rx0 = constructorRx;
		rx1 = constructorRx;
		vec temp = zeros<vec>(1);
		rx0.insert_rows(0,temp);
		s.v.insert_rows(0,temp);
		temp(0) = 1;
		rx1.insert_rows(0,temp);
	}

	int compute(vec ry)
	{
		double result = dot(s.v,ry) + s.b;
		if(compareVecs(ry,rx1)){
			if(result > 0){
				return 0;
			} else {
				return 1;
			}
		} else {
			if(result > 0){
				return 1;
			} else {
				return 0;
			}
		}
	}
};

selector singleGradientDecent(selector s, optimalFunction of, vec ry,double rate)
{	
	double nout = 1/(1+exp(-dot(s.v,ry) - s.b));
	double err = nout - of.compute(ry);
	s.v += -rate*err*nout*(1-nout)*ry;
	s.b += -rate*err*nout*(1-nout);
	return s;
}

#define RUNAVGWID 5000
#define NUMGENERATIONS 100000


selector remakeSelector(selector oldselector, vec rx)
{	
	selector newselector;
	newselector.v = oldselector.v/(2*oldselector.v.max());
	vec insert = randu<vec>(1); 
	newselector.v.insert_rows(0,insert/2);
	newselector.b = oldselector.b/(2*oldselector.v.max());
	optimalFunction of = optimalFunction(oldselector,rx);

	vec temp = zeros<vec>(1);
	vec rx0 = rx;
	vec rx1 = rx;
	rx0.insert_rows(0,temp);
	temp(0) = 1;
	rx1.insert_rows(0,temp);


	for(int i = 0; i< NUMGENERATIONS; i++){
		if(i%10 == 0){
			newselector = singleGradientDecent(newselector,of,rx1,0.05);
		}
		if(i%11 == 0){
			newselector = singleGradientDecent(newselector,of,rx0,0.05);
		}
		vec ry = randi<vec>( oldselector.v.n_rows + 1, distr_param(0,1) );
		
		newselector = singleGradientDecent(newselector,of,ry,0.05);

	}
	return newselector;
}

/*
selector randomSelector(int initNumHps,double var)
{
	selector ret;
	ret.v = randn<vec>(initNumHps);
	vec b = randn<vec>(1);
	ret.v = ret.v*var;
	ret.b = b(0)*var;
	return ret;
}

int main(int argc, char *argv[])
{
	arma_rng::set_seed_random();
	selector s = randomSelector(15,15);
	vec rx = randi<vec>( 15, distr_param(0,1) );
	cout << "Region Signature: " << endl << rx << endl;
	cout << "Selector Vector:" << endl << s.v << endl;
	cout << "Selector Offset: " << s.b << endl;
	s = remakeSelector(s,rx);
	cout << "Selector Vector:" << endl << s.v << endl;
	cout << "Selector Offset: " << s.b << endl;
}
*/