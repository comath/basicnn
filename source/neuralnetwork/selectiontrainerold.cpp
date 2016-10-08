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
#define NUMMATRICES 200
#define MUTATIONRATE 1


bool compareUvecs(uvec x1, uvec x2)
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

double optimalFunction(selector s, uvec rx, uvec ry)
{
	uvec truncRy = ry.rows(0,rx.n_rows-1);

	if(ry(rx.n_rows) == 1){
		return dot(s.v,truncRy) + s.b;
	} else if(compareUvecs(truncRy,rx)){
		return 2*s.b - dot(s.v,rx)  ;
	} else {
		return dot(s.v,truncRy) + s.b;
	}
}

selector randomSelector(int initNumHps,double var)
{
	selector ret;
	ret.v = randu<vec>(initNumHps);
	vec b = randu<vec>(1);
	ret.v = ret.v*var;
	ret.b = b(0)*var;
	return ret;
}

class matrixType {
private:
	int initNumHps;	
	mat A;
	double error;
	unsigned survival;
	vec getVec(selector s, uvec rx)
	{
		vec vrx = conv_to< vec >::from( rx );
		vec temp = s.v;
		mat basav  = {s.b};
		temp.insert_rows(0,basav);
		temp = join_cols(temp,vrx);
		temp = join_cols(temp,s.v%vrx);
		return temp;
	}
public:
	matrixType(){}
	matrixType(int numHPs,double var)
	{	
		initNumHps = numHPs;
		A = randu<mat>(initNumHps+2,3*initNumHps+1);
		A = A*var;
		survival = 0;
		error = 1000000000;
	}


	double getError(){return error;}



	selector eval(selector s, uvec rx)
	{
		
		vec temp = A*getVec(s,rx);
		
		selector ret;
		ret.v = temp.rows(1,(s.v).n_rows+1);
		ret.b = temp(0);
		
		return ret;
	}

	void singleGradientDecent(selector s, uvec rx, uvec ry,double rate)
	{
		vec temp = getVec(s,rx);
		vec vry = conv_to< vec >::from( ry );
		mat one  = {1};
		vry.insert_rows(0,one);
		selector newselector = this->eval(s,rx);
		double err = dot(newselector.v,ry) + newselector.b - optimalFunction(s,rx,ry);
		A -= rate*err*(vry * temp.t());
	}
	void gradientDecent(double rate){
		for (int i = 0; i < NUMDATAPOINTS; ++i)
		{
			uvec rx = randi<uvec>( initNumHps, distr_param(0,1) );
			uvec ry = randi<uvec>( initNumHps + 1, distr_param(0,1) );
			selector s = randomSelector(initNumHps, 10);
			singleGradientDecent(s,rx,ry,rate);
		}
		cout << "Matrix After decent: "<< endl;
		cout << A << endl;
	}
	void evalError()
	{
		error = 0;
		for (int i = 0; i < NUMDATAPOINTS; ++i)
		{
			uvec rx = randi<uvec>( initNumHps, distr_param(0,1) );
			uvec ry = randi<uvec>( initNumHps + 1, distr_param(0,1) );
			selector s = randomSelector(initNumHps, 10);
			selector newselector = this->eval(s,rx);
			double err = dot(newselector.v,ry) + newselector.b - optimalFunction(s,rx,ry);
			error += err*err;
		}
		error = error/NUMDATAPOINTS;
	}
	bool operator<(const matrixType &rhs) 
	{
		return this->error < rhs.error;
	}

	friend matrixType operator*(const double &c, const matrixType &rhs);


	matrixType operator+(const matrixType &rhs)
	{
		matrixType ret;
		ret.initNumHps = initNumHps;
		ret.A = this->A + rhs.A;
		ret.survival = 0;
		return ret;
	}

	void survived(){survival++;}

	friend std::ostream &operator<<(std::ostream& os, const matrixType& M);
};

std::ostream &operator<<(std::ostream& os, const matrixType& M)
{
	os << "Matrix : " << endl << M.A << " Error : " << M.error << endl << "Survived: " << M.survival << endl;
	return os;
}

matrixType operator*(const double &c, const matrixType &rhs)
{
	matrixType ret;
	ret.initNumHps = rhs.initNumHps;
	ret.A = c*rhs.A;
	ret.survival = 0;
	return ret;
}



class mycomparison
{
	bool reverse;
	public:
	mycomparison(const bool& revparam=false)
	{
		reverse=revparam;
	}
	bool operator() (const double& lhs, const double&rhs) const
	{
		if (reverse) return (lhs>rhs);
		else return (lhs<rhs);
	}
};

double getMiddleVal(std::vector<matrixType> arr)
{
	
	#ifdef DEBUG
		printf("Determining the 50 percentile\n");
	#endif

	typedef std::priority_queue<double,std::vector<double>,mycomparison> heap;

	heap bigHeap (mycomparison(true));
	heap smallHeap;
	double supSmallHeap = arr[0].getError();
	unsigned maxSize = arr.size()/2 ;
	for(int i=0; i<arr.size(); ++i){
		#ifdef DEBUG
			cout << "Inserting: " << arr[i].getError() << endl;
			cout << "Cur sup: " << supSmallHeap << endl;
		#endif
		if(arr[i].getError() <= supSmallHeap){
			smallHeap.push(arr[i].getError());
		} else {
			bigHeap.push(arr[i].getError()); 
		}
		if(smallHeap.size() > maxSize){
			bigHeap.push(smallHeap.top());
			smallHeap.pop();
		}
		if(bigHeap.size() > maxSize){
			smallHeap.push(bigHeap.top());
			bigHeap.pop();
		}
		supSmallHeap = smallHeap.top();
		#ifdef DEBUG
			cout << "Num Elements in smallHeap: " << smallHeap.size() << endl;
			cout << "Num Elements in bigHeap: " << bigHeap.size() << endl;
		#endif
	}
	return smallHeap.top();
}

std::vector<matrixType> evolve(double var, int initNumHps, std::vector<matrixType> matrices)
{
	#ifdef DEBUG
		printf("Evolving\n");
	#endif
	//printErrors(errors);
	double mode = getMiddleVal(matrices);
	#ifdef DEBUG
		printf("The mode is: %f\n",mode);
	#endif
	std::vector<matrixType> goodMats = {};
	for(unsigned i =0; i<matrices.size();++i){
		if(matrices[i].getError() < mode){
			goodMats.push_back(matrices[i]);
			goodMats[i].survived();
		}
	}
	#ifdef DEBUG
		printf("Trimmed matrices\n");
	#endif
	std::default_random_engine generator;
	unsigned numGoodMats = goodMats.size();
  	std::uniform_int_distribution<int> uniformDist(0,numGoodMats);
  	
	for(unsigned i =numGoodMats; i<matrices.size();++i){
		
		int n1 = uniformDist(generator);
		int n2 = uniformDist(generator);
		#ifdef DEBUG
			cout << "Genenerating " << i << " mat out of " << n1 << " and " << n2 << ". Max should be " << numGoodMats << "." << endl;
		#endif
		double weight1 = (goodMats[n2].getError() - goodMats[n1].getError()) / (goodMats[n1].getError() + goodMats[n2].getError());
		double weight2 = (goodMats[n1].getError() - goodMats[n2].getError()) / (goodMats[n1].getError() + goodMats[n2].getError());
		matrixType newMat = matrixType(initNumHps, var);
		goodMats.push_back(weight1*goodMats[n1] + weight2*goodMats[n2] + newMat);
		goodMats[i].evalError();
	}
	for(unsigned i =0; i<matrices.size();++i){
		
	}
	
	#ifdef DEBUG
		cout << "Have "<< goodMats.size()<<" matrices. First one is:\n"<<endl;
		cout << matrices[0];
	#endif
	std::sort(matrices.begin(),matrices.end());
	return goodMats;
}

std::vector<matrixType> decend(double var, std::vector<matrixType> matrices)
{
	for(unsigned i =0; i<matrices.size();++i){
		matrices[i].gradientDecent(var/10000);
	}
}

int main(int argc, char *argv[])
{
	int numHPs = 3; 
	double initVar = 1;
	double decayRate = 0.99;
	std::vector<matrixType> matrices;
	for(int i =0; i<NUMMATRICES;++i)
	{
		matrices.push_back(matrixType(numHPs,initVar));
		matrices[i].evalError();
	}
	#ifdef DEBUG
		cout << "Created "<< matrices.size()<<" matrices. First one is:\n"<<endl;
	#endif
	for(int j = 0; j< 100;++j){
		for(int i=0; i< 10; ++i){
			cout << "On generation: " << i << endl;
			cout << "Current Winner: " << endl;
			matrixType minMat = matrices[0];
			cout << minMat;
			matrices = evolve(pow (decayRate, i)+0.08, numHPs,matrices);
		}
		for(int i=0; i< 10; ++i){
			cout << "On generation: " << i << endl;
			cout << "Current Winner: " << endl;
			matrices = decend(pow (decayRate, i)+0.08,matrices);
		}
	}
	return 0;
}