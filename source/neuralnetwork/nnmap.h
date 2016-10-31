#ifndef _anngpm_h
#define _annpgm_h
#include <iostream>
#include <armadillo>
#include <map>
#include <vector>
#include <set>
#include <cmath>  
#include "ann.h"

using namespace arma;

typedef struct locSig {
	std::vector<int> regionSig;
	std::vector<int> interSig;
} locSig;

class locInfo {
public:
	vec totvec;
	unsigned numvec;
	vec toterrvec;
	unsigned numerrvec;
	
	//for the refined map
	std::vector<std::vector<int>> includedRegions;
	std::vector<int> interSig;
	std::set<int> hyperplanesCrossed;
	int cornerDimension;
	locInfo();
	locInfo(bool err,vec v);
	void print();
	void printlocation();
	void addvector(bool err, vec v);
	int getNumErr();
	vec getTotAvg();
	vec getErrAvg();

	void combine(locInfo other);

};
typedef struct IntersectionBasis {
	bool success;
	mat basis;
	vec aSolution;
} IntersectionBasis;
class nnmap {
private:
	int numHPs;
	int dimension;

	mat A0;
	vec b0;
	mat A1;
	vec b1;

	std::vector<vec> hpNormals;
  	std::vector<vec> hpoffsets;
  	std::map <std::set<int>, IntersectionBasis> interCashe;

  	void setUpHps();
  	void setUpCashe(nn *thisnn);
  	IntersectionBasis getIntersectionBasis(std::set<int> indexes);

	std::map <std::vector<int>, std::map<std::vector<int>, locInfo>> regInter;
	

	std::map <std::vector<int>,std::vector<locInfo>> refinedMap;



  	vec computeDistToHPS(vec v);
  	
public:
	double computeDist(vec p, std::set<int> indexes);
	double computeDist(vec p, std::vector<int> indexes);
	std::vector<int> getInterSig(vec v);
	std::vector<int> getRegionSig(vec v);
	//Creates a map 
	~nnmap(){}
	nnmap(nn *nurnet, vec_data *D);
	void refineMap(vec selectV, double offset);
	locInfo getRefinedMaxErrRegInter();


	locSig getMaxErrRegInter(mat A, vec b);


	void addvector(bool err, vec v);
	locInfo getRegInterInfo(const locSig sig);
	vec getRegInterAvgVec(const locSig sig);
	int getRegInterPop(const locSig sig);
	vec getRegInterAvgErrVec(const locSig sig);
	int getRegInterErrPop(const locSig sig);

	void print();
	void printRefined();

};

#endif