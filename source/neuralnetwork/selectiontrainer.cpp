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
	vec rx;
public:
	optimalFunction()
	{
		s.v = zeros<vec>(1);
		s.b = 0;
		rx = zeros<vec>(1);
	}

	optimalFunction(selector constructorS, std::vector<int> constructorRx)
	{
		s = constructorS;
		rx = conv_to< vec >::from( constructorRx );
	}
	optimalFunction(selector constructorS, vec constructorRx)
	{
		s = constructorS;
		rx = conv_to< vec >::from( constructorRx );
	}

	int compute(vec ry)
	{
		vec truncRy = ry.rows(1,rx.n_rows);
		double result;
		if(ry(0) == 1){
			result = dot(s.v,truncRy) + s.b;
		} else if(compareVecs(truncRy,rx)){
			result = 2*s.b - dot(s.v,rx)  ;
		} else {
			result = dot(s.v,truncRy) + s.b;
		}
		if(result > 0){
			return 1;
		} else {
			return 0;
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

double erravgslope(double curerr)
{
	static int calltimes = 0;
	if(calltimes<RUNAVGWID){calltimes++;}

	static double lasterror = 0;
	static double slopes[RUNAVGWID];
	int i=0;
	for(i=0;i<calltimes-1;i++){
		slopes[i]=slopes[i+1];
	}
	slopes[calltimes-1]= curerr-lasterror;
	lasterror = curerr;
	double avg =0;
	for(i=0;i<calltimes;i++){avg += slopes[i];}
	return avg/calltimes;
}

#define NUMGENERATIONS 100000


selector remakeSelector(selector oldselector, vec rx)
{	
	selector newselector;
	newselector.v = oldselector.v/(2*oldselector.v.max());
	vec insert = randu<vec>(1); 
	newselector.v.insert_rows(0,insert/2);
	newselector.b = oldselector.b/(2*oldselector.v.max());
	unsigned errcount =0;
	double nout, error;
	optimalFunction of = optimalFunction(oldselector,rx);
	for(int i = 0; i< NUMGENERATIONS; i++){
		vec ry = randi<vec>( oldselector.v.n_rows + 1, distr_param(0,1) );
		#ifdef DEBUG
			//cout << "New Region signature: " << endl << ry;
			//cout << "OptimalFunction: " << of.compute(ry) << endl;
			//cout << "Old selector value: " << dot(oldselector.v,ry.rows(1,rx.n_rows)) + oldselector.b << endl;
		#endif
		newselector = singleGradientDecent(newselector,of,ry,0.05);
		nout = 1/(1+exp(-dot(newselector.v,ry) - newselector.b));
		error = nout - of.compute(ry);		
	}
	cout << endl << "Errors: " << errcount << '/' << NUMGENERATIONS << endl;
	return newselector;
}

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