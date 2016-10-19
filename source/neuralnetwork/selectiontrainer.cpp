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
#define NUMGENERATIONS 10000


std::vector<vec> createSelectionTrainingData(vec_data *data, nn *thisnn,vec newhpNormal, double newhpOffset)
{	
	mat A0 = thisnn->getmat(0);
	vec b0 = thisnn->getoff(0);
	A0.insert_rows(0,newhpNormal.t());
	mat offsetv = {newhpOffset};
	b0.insert_rows(0,offsetv);

	int n = b0.n_rows;

	std::map<std::vector<int>,bool> arificialDataMap;
	for (int i = 0; i < data->numdata; ++i){
		vec w = A0*data->data[i].coords + b0;
		std::vector<int> sig (n,0);
		for (int i = 0; i < n; ++i)
		{
			if(w(i)>0){
				sig[i] = 1;
			} else {
				sig[i] = 0;
			}
		}
		arificialDataMap[sig] =true;
	}
	std::vector<vec> arificialData;
	for (auto& it: arificialDataMap){
		vec regionSig = conv_to<vec>::from(it.first);
		arificialData.push_back(regionSig);
	}
	return arificialData;
}

selector remakeSelector(selector oldselector, std::vector<vec> regionData, vec rx)
{	
	//Renormalize so that it is easier to train. We make it so that the initial training step is small and retrainable.
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

	for(int j = 0; j<NUMGENERATIONS;++j ){
		for(unsigned i = 0; i< regionData.size(); i++){
			newselector = singleGradientDecent(newselector,of,regionData[i],0.05);
		}
		if(j%5 == 0){
			newselector = singleGradientDecent(newselector,of,rx1,0.05);
		}
		if(j%6 == 0){
			newselector = singleGradientDecent(newselector,of,rx0,0.05);
		}
	}
	// Return a selector of the same length as we recieved so that the blur is more or less correct
	newselector.v = norm(oldselector.v)*newselector.v/norm(newselector.v);
	newselector.b = norm(oldselector.v)*newselector.b/norm(newselector.v);

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