#include <iostream>
#include <armadillo>
#include <map>
#include <vector>
#include "ann.h"
#include "nnanalyzer.h"


using namespace std;
using namespace arma;

#ifndef ERRORTHRESHOLD
#define ERRORTHRESHOLD 0.01
#endif

#ifndef TUBETHRESHOLD
#define TUBETHRESHOLD 0.001
#endif

#ifndef SCALEDTUBETHRESHOLD
#define SCALEDTUBETHRESHOLD 2
#endif

hp *gethp(nn *nurnet)
{
	mat A = nurnet->getmat(0);
	vec b = nurnet->getoff(0);
	int n = b.n_rows;
	hp *hps = new hp[n];
	double scaling = 1;
	for (int i = 0; i < n; ++i)
	{
		hps[i].v = (A.row(i)).t();
		hps[i].b = b(i);
		scaling = norm(hps[i].v);
		hps[i].v = hps[i].v/scaling;
		hps[i].b = hps[i].b/scaling;
	}
	return hps;
}



double disttohp(hp p, vec v){ // HP should be normalized.
	return dot(v - p.b*p.v,p.v);
}

vec computeDistToHPS(mat A,vec b, vec v){
	int i =0;
	int n = b.n_rows;
	vec retvec = zeros<vec>(n);
	rowvec curvec;
	rowvec normcurvec;
	double scaling = 1;
	for(i=0;i<n;i++){
		scaling = norm(A.row(i));
		curvec = A.row(i)/scaling;
		normcurvec = b(i)*curvec/scaling;
		retvec(i) = dot((v.t()-normcurvec),curvec);
	}
	return retvec;
}

std::vector<int> getInterSig(vec v, mat A, vec b)
{
	vec dist = computeDistToHPS(A,b,v);
	uvec indsort = sort_index(dist,"decend");
	unsigned j = 0;
	double minval = dist(indsort(j));
	unsigned n = dist.n_rows;
	if(minval < TUBETHRESHOLD){
		while(dist(indsort(j))<TUBETHRESHOLD && j < n){
			j++;
		}
	} else {
		dist = dist/minval;
		while(dist(indsort(j))<2 && j < n){
			j++;
		}
	}
	std::vector<int> sig (2*n,0);
	if(j+1 == b.n_rows) {

		return sig;
	}
	for (unsigned i = 0; i < j+1; ++i)
	{
		sig[indsort(i)] = 1;
	}
	return sig;
}

std::vector<int> getRegionSig(vec v, mat A, vec b)
{
	vec w = A*v + b;
	int n = b.n_rows;
	std::vector<int> sig (n,0);
	for (int i = 0; i < n; ++i)
	{
		if(w(i)>0){
			sig[i] = 1;
		} else {
			sig[i] = -1;
		}
	}
	return sig;
}

typedef struct locInfo {
	locInfo() 
	{
		numvec =0;
		totvec = vec(1);
	}
	locInfo(bool err,vec v)
	{
		if(err) {
			numerrvec =1;
			toterrvec = v;
		}
		numvec =1;
		totvec = v;
	}
	void addvector(bool err, vec v)
	{
		if(err) {
			numerrvec++;
			toterrvec += v;
		}
		totvec+= v;
		numvec++;
	}
	vec getTotAvg()
	{
		return totvec/numvec;
	}
	vec getErrAvg()
	{
		return toterrvec/numerrvec;
	}
	vec totvec;
	unsigned numvec;
	vec toterrvec;
	unsigned numerrvec;
} locInfo;

class nnmap {
private:
	std::map <std::vector<int>, locInfo> reg;
	std::map <std::vector<int>, locInfo> inter;
	std::map <std::vector<int>, locInfo> regInter;
	int numHPs;
public:
	nnmap(int n){ numHPs = n;}
	~nnmap(){}
	void addvector(bool err, vec v, mat A, vec b){
		const vector<int> regsig = getRegionSig(v,A,b);
		if(reg.count(regsig) == 0){
			reg[regsig] = locInfo(err,v);
		} else {
			reg[regsig].addvector(err, v);
		}
		const vector<int> intersig = getRegionSig(v,A,b);
		if(inter.count(intersig) == 0){
			inter[intersig] = locInfo(err, v);
		} else {
			inter[intersig].addvector(err, v);
		}
		unsigned n = b.n_rows;
		vector<int> bothsig (2*n,0);
		for(unsigned i =0;i<n;i++){
			bothsig[i] = intersig[i];
			bothsig[i+n] = regsig[i];
		}
		if(inter.count(intersig) == 0){
			inter[bothsig] = locInfo(err, v);
		} else {
			inter[bothsig].addvector(err, v);
		}
	}
	// -------------------------------All section---------------------------
	locInfo getRegionInfo(const vector<int> sig) 
	{
		if(reg.count(sig) == 0){
			return locInfo();
		} else {
			return reg.at(sig); 
		}
	}
	vec getRegionAvgVec(const vector<int> sig)
	{
		if(reg.count(sig) == 0){
			return 0;
		} else {
			return reg.at(sig).getTotAvg(); 
		}
	}
	int getRegionPop(const vector<int> sig)
	{
		if(reg.count(sig) == 0){
			return 0;
		} else {
			return reg.at(sig).numvec; 
		}
	}
	locInfo getIntersectionInfo(const vector<int> sig) 
	{
		if(inter.count(sig) == 0){
			return locInfo();
		} else {
			return inter.at(sig); 
		}
	}
	vec getIntersectionAvgVec(const vector<int> sig)
	{
		if(inter.count(sig) == 0){
			return 0;
		} else {
			return inter.at(sig).getTotAvg(); 
		}
	}
	int getIntersectionPop(const vector<int> sig)
	{
		if(inter.count(sig) == 0){
			return 0;
		} else {
			return inter.at(sig).numvec; 
		}
	}
	locInfo getRegInterInfo(const vector<int> sig) 
	{
		if(regInter.count(sig) == 0){
			return locInfo();
		} else {
			return regInter.at(sig); 
		}
	}
	vec getRegInterAvgVec(const vector<int> sig)
	{
		if(regInter.count(sig) == 0){
			return 0;
		} else {
			return regInter.at(sig).getTotAvg(); 
		}
	}
	int getRegInterPop(const vector<int> sig)
	{
		if(regInter.count(sig) == 0){
			return 0;
		} else {
			return regInter.at(sig).numvec; 
		}
	}
	//-----------------------------------Error section-----------------------
	vec getRegionAvgErrVec(const vector<int> sig)
	{
		if(reg.count(sig) == 0){
			return 0;
		} else {
			return reg.at(sig).totvec/reg.at(sig).numvec; 
		}
	}
	int getRegionErrPop(const vector<int> sig)
	{
		if(reg.count(sig) == 0){
			return 0;
		} else {
			return reg.at(sig).numvec; 
		}
	}
	vec getIntersectionAvgErrVec(const vector<int> sig)
	{
		if(inter.count(sig) == 0){
			return 0;
		} else {
			return inter.at(sig).totvec/inter.at(sig).numvec; 
		}
	}
	int getIntersectionErrPop(const vector<int> sig)
	{
		if(inter.count(sig) == 0){
			return 0;
		} else {
			return inter.at(sig).numvec; 
		}
	}
	vec getRegInterAvgErrVec(const vector<int> sig)
	{
		if(regInter.count(sig) == 0){
			return 0;
		} else {
			return regInter.at(sig).totvec/regInter.at(sig).numvec; 
		}
	}
	int getRegInterErrPop(const vector<int> sig)
	{
		if(regInter.count(sig) == 0){
			return 0;
		} else {
			return regInter.at(sig).numvec; 
		}
	}
	//--------------------------Returning useful info section-------------
	std::vector<int> getMaxErrRegInter(){
		std::vector<int> sigMaxErr (2*numHPs,-2);
		unsigned maxNumErr = -1;
  		for (std::map<std::vector<int>,locInfo>::iterator it=regInter.begin(); it!=regInter.end(); ++it){
  			if(it->second.numerrvec > maxNumErr){
    			sigMaxErr = it->first;
  			}
  		}
  		return sigMaxErr;
	}

};

nnmap * locateData(nn *nurnet,vec_data *D)
{
	int numdata = D->numdata;
	mat A = nurnet->getmat(0);
	vec b = nurnet->getoff(0);
	bool err = false;
	nnmap *locData = new nnmap(b.n_rows);
	for (int i = 0; i < numdata; ++i)
	{
		err = (nurnet->calcerror(D->data[i],1));
		locData->addvector(err,D->data[i].coords,A,b);
	}
	return locData;
}





#ifndef SLOPETHRESHOLD
#define SLOPETHRESHOLD 0.01
#endif

double ** adaptivebackprop(nn *nurnet, vec_data *D, double rate, double objerr, int max_gen, int max_nodes, bool ratedecay)
{	
	double **returnerror = new double*[2];
	returnerror[0] = new double[max_gen];
	returnerror[1] = new double[max_gen];
	int i=0;
	int lastHPChange = 0;
	double inputrate = rate;
	double curerr = nurnet->calcerror(D,0);
	double curerrorslope = 0;
	int curnodes = nurnet->outdim(0);
	while(i<max_gen && curerr > objerr){
		if(ratedecay){inputrate = rate*((max_gen-(double)i)/max_gen);} 
		nurnet->epochbackprop(D,inputrate);
		curerr = nurnet->calcerror(D,0);
		returnerror[0][i] = curerr;
		returnerror[1][i] = nurnet->calcerror(D,1);
		curerrorslope = nurnet->erravgslope(D,0);
		if(curerrorslope > -SLOPETHRESHOLD*inputrate && curerrorslope < SLOPETHRESHOLD*inputrate 
			&& curnodes < max_nodes && i-lastHPChange>50){
			nurnet->smartaddnode1(D,1);
			lastHPChange = i;
			curnodes = nurnet->outdim(0);	
		}
		i++;
		if(ratedecay){inputrate = rate*((double)max_gen - i)/max_gen;}
	}
	return returnerror;
}