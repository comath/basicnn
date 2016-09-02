#include <iostream>
#include <armadillo>
#include <map>
#include <vector>
#include <cmath>  
#include "ann.h"
#include "nnanalyzer.h"

#include "annpgm.h"

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
		retvec(i) = abs(dot((v.t()-normcurvec),curvec));
	}
	return retvec;
}

std::vector<int> getInterSig(vec v, mat A, vec b)
{
	vec dist = computeDistToHPS(A,b,v);
	cout << dist << endl;
	uvec indsort = sort_index(dist,"accend");
	cout << indsort << endl;
	unsigned j = 0;
	double minval = dist(indsort(j));
	cout << j << endl;
	unsigned n = dist.n_rows;
	if(minval < TUBETHRESHOLD){
		while(j < n && dist(indsort(j))<TUBETHRESHOLD){
			j++;
		}
	} else {
		dist = dist/minval;
		while(j < n-1 && dist(indsort(j+1))<2){
			cout << j << endl;
			j++;
		}
	}
	cout << j << endl;
	std::vector<int> sig (n,0);
	if(j+1 == b.n_rows) {
		return sig;
	}
	for (unsigned i = 0; i < j+1; ++i)
	{
		printf("getting to the making sig\n");
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
		printf("Getting here (empty locInfo contstructor)\n");
		numvec =0;
	}
	locInfo(bool err,vec v)
	{
		printf("Getting here (Correct locInfo contstructor)\n");
		if(err) {
			numerrvec =1;
			toterrvec = v;
		}
		numvec =1;
		totvec = v;
	}
	void addvector(bool err, vec v)
	{
		printf("Getting to the add vector\n");
		if(err) {
			numerrvec++;
			toterrvec += v;
		}
		totvec+= v;
		numvec++;
	}
	vec getTotAvg()	{ return totvec/numvec;	}
	vec getErrAvg()	{ return toterrvec/numerrvec; }
	vec totvec;
	unsigned numvec;
	vec toterrvec;
	unsigned numerrvec;
} locInfo;

void printsig(std::vector<int> sig,int n)
{
	for (int i = 0; i < n; ++i)
	{
		cout << sig[i];
	}
	cout << endl;
}

typedef struct locSig {
	std::vector<int> regionSig;
	std::vector<int> interSig;
} locSig;

class nnmap {
private:
	std::map <std::vector<int>, locInfo> reg;
	std::map <std::vector<int>, locInfo> inter;
	std::map <std::vector<int>, std::map<std::vector<int>, locInfo>> regInter;
	int numHPs;
public:
	nnmap(nn *nurnet, vec_data *D){ 
		printf("Creating the NN map\n");
		int numdata = D->numdata;
		mat A = nurnet->getmat(0);
		vec b = nurnet->getoff(0);
		numHPs = b.n_rows;
		bool err = false;
		for (int i = 0; i < numdata; ++i)
		{
			err = (nurnet->calcerror(D->data[i],1));
			this->addvector(err,D->data[i].coords,A,b);
		}
	}
	~nnmap(){}
	void addvector(bool err, vec v, mat A, vec b){
		const vector<int> regionSig = getRegionSig(v,A,b);
		cout << "region sig:";
		printsig(regionSig,numHPs);
		if(reg.count(regionSig) == 0){
			printf("Its new.\n");
			reg[regionSig] = locInfo(err,v);
		} else {
			printf("it's old\n");
			reg[regionSig].addvector(err, v);
		}
		const vector<int> interSig = getInterSig(v,A,b);
		cout << "inter sig:";
		printsig(interSig,numHPs);
		if(inter.count(interSig) == 0){
			inter[interSig] = locInfo(err, v);
		} else {
			inter[interSig].addvector(err, v);
		}
		
		if(regInter[interSig].count(regionSig)){
			regInter[interSig][regionSig] = locInfo(err, v);
		} else {
			if(regInter[interSig].count(regionSig) == 0){
				regInter[interSig][regionSig] = locInfo(err, v);
			} else {
				regInter[interSig][regionSig].addvector(err,v);
			}

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
	locInfo getRegInterInfo(const locSig sig) 
	{
		if(regInter[sig.interSig].count(sig.regionSig)){
			return locInfo();
		} else {
			return regInter[sig.interSig].at(sig.regionSig); 
		}
	}
	vec getRegInterAvgVec(const locSig sig)
	{
		if(regInter[sig.interSig].count(sig.regionSig)){
			return 0;
		} else {
			return regInter[sig.interSig].at(sig.regionSig).getTotAvg(); 
		}
	}
	int getRegInterPop(const locSig sig)
	{
		if(regInter[sig.interSig].count(sig.regionSig)){
			return 0;
		} else {
			return regInter[sig.interSig].at(sig.regionSig).numvec; 
		}
	}
	//-----------------------------------Error section-----------------------
	vec getRegionAvgErrVec(const vector<int> sig)
	{
		if(reg.count(sig) == 0){
			return 0;
		} else {
			return reg.at(sig).getErrAvg(); 
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
			return inter.at(sig).getErrAvg(); 
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
	vec getRegInterAvgErrVec(const locSig sig)
	{
		if(regInter[sig.interSig].count(sig.regionSig)){
			return 0;
		} else {
			return regInter[sig.interSig].at(sig.regionSig).getErrAvg(); 
		} 
	}
	int getRegInterErrPop(const locSig sig)
	{
		if(regInter[sig.interSig].count(sig.regionSig) == 0){
			return 0;
		} else {
			return regInter[sig.interSig].at(sig.regionSig).numvec; 
		}
	}
	//--------------------------Returning useful info section-------------
	locSig getMaxErrRegInter()
	{
		printf("Deterimining the location with maximum error.\n");
		std::vector<int> interSigMaxErr (numHPs,-2);
		std::vector<int> regionSigMaxErr (numHPs,-2);
		int maxNumErr = -1;
  		for (std::map<std::vector<int>,std::map<std::vector<int>,locInfo>>::iterator firstit=regInter.begin(); firstit!=regInter.end(); ++firstit){
  			for (std::map<std::vector<int>,locInfo>::iterator secit=firstit->second.begin(); secit!=firstit->second.end(); ++secit){
  				if(secit->second.numerrvec > maxNumErr){
    				interSigMaxErr = firstit->first;
    				regionSigMaxErr = secit->first;
  				}
  			}
  		}
  		locSig sigMaxErr = {.regionSig = regionSigMaxErr, .interSig = interSigMaxErr};
  		return sigMaxErr;
	}
	std::vector<vec> getLocalRegions(const locSig sig)
	{
		std::vector<vec> localRegions;
  		for (std::map<std::vector<int>,locInfo>::iterator firstit=regInter[sig.interSig].begin(); firstit!=regInter[sig.interSig].end(); ++firstit){
  			vec cursig =  zeros<vec>(numHPs);
  			std::vector<int> regionSig = firstit->first;
  			for (int i = 0; i < numHPs; ++i){
  				cursig(i) = regionSig[i];
  			}
  			localRegions.push_back(cursig);
  		}
  		return localRegions;
	}
};

vec getNormVec(nn *nurnet, nnmap *locInfo, locSig sig)
{
	hp *hps = gethp(nurnet);
	int m = nurnet->getmat(0).n_cols;
	vec normvec = zeros<vec>(m);
	int n = nurnet->getmat(0).n_rows;
	vec regionRep = locInfo->getRegInterAvgVec(sig);
	for(int i =0; i<n; ++i){
		if(sig.interSig[i] == 1){
			if(dot(hps[i].v,regionRep)>0)
				normvec += hps[i].v;
			if(dot(hps[i].v,regionRep)<0)
				normvec -= hps[i].v;
		}
	}
	return normvec;
}

vec correctRegionSig(vec regionSig, vec signs)
{
	int n = regionSig.n_rows;
	vec retSig = zeros<vec>(n);
	for (int i = 0; i < n; ++i)
	{
		if(signs(i) < 0){
			if(regionSig(i) == 0)
				retSig(i) = 1;
			if(regionSig(i) == 1)
				retSig(i) = 0;
		} else {
			retSig(i) = regionSig(i);
		}
	}
	return retSig;
}

nnlayer getSelectionVec(nn *nurnet, nnmap *nurnetMap, locSig sig)
{
	int i = 0;
	int j = 0;
	int numHPs = nurnet->outdim(0);
	//Convert the std vector over to an arma vector to detect which level 2 selection does what to this region.
	//We don't need the actual region vector that is stored in the nnmap as this will be the result.
	
	std::vector<int> indexOfLocalHPs;
	for (i = 0; i < numHPs; ++i) {
		if(sig.interSig[i] == 1)
			indexOfLocalHPs.push_back(i);
	}

	vec regionSig = conv_to<vec>::from(sig.regionSig);

	mat A2 = nurnet->getmat(1);
	vec b2 = nurnet->getoff(1);
	int numSelection = b2.n_rows;
	vec selection = A2*regionSig + b2;
	selection = selection.for_each( [](mat::elem_type& val) { if(val>0){val=1;} else{val=0;} } );
	std::vector<vec> localRegions = nurnetMap->getLocalRegions(sig);
	int numLocalRegions = localRegions.size();
	vec averageDiff = zeros<vec>(numSelection);

	//The shape of the error here is relevant, is it pointy? is it flat? There will have to be an improvement here.
	for (i = 0; i < numLocalRegions; ++i)
	{
		vec curRegionSelection = A2*localRegions[i] + b2;
		curRegionSelection = curRegionSelection.for_each( [](mat::elem_type& val) { if(val>0){val=1;} else{val=0;} } );
		curRegionSelection = curRegionSelection - selection;
		curRegionSelection = curRegionSelection.for_each( [](mat::elem_type& val) { if(val==0){val=0;} else{val=1;} } );
		averageDiff += curRegionSelection;
	}
	averageDiff = averageDiff/numLocalRegions;
	vec newSelectionWeight = zeros<vec>(numSelection);
	for (i = 0; i < numSelection; ++i) {
		if(averageDiff(i) > 0.55) {
			vec curSelectorVec = A2.row(i).t();
			double curSelectorOff = b2(i);
			double regionValue = dot(curSelectorVec,regionSig) + curSelectorOff;
			// Make the selection vector positive (all values >0), and adjust the offset to compensate.
			// For each HP for which we have to swap the sign, we have to negate that hyperplane. 
			// We can do this by manpulating the region signatures.
			// This has to be done per selection vector as they will have different sinages. 
			vec signs = curSelectorVec.for_each( [](mat::elem_type& val) { 
				if(val>0){ val=1; } 
				else if(val<0){
					val=-1;
				} } );
			curSelectorOff += dot(((signs % signs) - signs),curSelectorVec);
			curSelectorVec = curSelectorVec % signs;
			if(regionValue != dot(curSelectorVec,correctRegionSig(regionSig, signs)) + curSelectorOff)
				printf("ERROR, Your region correction code is incorrect");
			vec correctedRegionSig = correctRegionSig(regionSig,signs);
			if(regionValue > 0){
				for(j =0;j<indexOfLocalHPs.size();++j){
					curSelectorVec(indexOfLocalHPs[j]) += (curSelectorOff-regionValue)/(indexOfLocalHPs.size()-1);
				}
				newSelectionWeight(i) = (curSelectorOff-regionValue);
			} else  {
				for(j =0;j<indexOfLocalHPs.size();++j){
					curSelectorVec(indexOfLocalHPs[j]) += (curSelectorOff-regionValue)/(indexOfLocalHPs.size()-1);
				}
				newSelectionWeight(i) = (curSelectorOff-regionValue);
			}
			curSelectorVec = curSelectorVec % signs;
			A2.row(i) = curSelectorVec.t();
		} else {
			newSelectionWeight(i) = 0.01;
		}
	}
	A2.insert_cols(0,newSelectionWeight);
	nnlayer retLayer = {.A = A2, .b = b2};
	return retLayer;
}

void smartaddnode(nn *nurnet, vec_data *D)
{
	printf("Starting smartaddnode\n");
	nnmap *nurnetMap = new nnmap(nurnet,D);
	locSig maxsig = nurnetMap->getMaxErrRegInter();
	vec errlocation = nurnetMap->getRegInterAvgErrVec(maxsig);
	vec normvec = getNormVec(nurnet,nurnetMap,maxsig);
	double offset = dot(normvec,errlocation);
	//This should be combined with the above to make sure the normal vector matches the shape of the error area.
	// If it's a corner of dimension n and there is a HP boundary over which it does not change then its's a corner of dim n-1
	nnlayer newSecondLayer = getSelectionVec(nurnet,nurnetMap, maxsig);
	nurnet->addnode(normvec,offset,newSecondLayer);
}

#ifndef SLOPETHRESHOLD
#define SLOPETHRESHOLD 0.1
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

		char header[100];
		sprintf(header, "imgfiles/sig/train%05d.ppm",i);
		write_nn_to_img(nurnet,header,500,500,0);
		write_data_to_img(D,header);
		sprintf(header, "imgfiles/hea/train%05d.ppm",i);
		write_nn_to_img(nurnet,header,500,500,1);
		write_data_to_img(D,header);
		printf("Error slope: %f Num Nodes: %d Threshold: %f Current gen:%d\n", curerrorslope, curnodes, -SLOPETHRESHOLD*inputrate,i);
		
		if(ratedecay){inputrate = rate*((max_gen-(double)i)/max_gen);} 
		nurnet->epochbackprop(D,inputrate);
		curerr = nurnet->calcerror(D,0);
		returnerror[0][i] = curerr;
		returnerror[1][i] = nurnet->calcerror(D,1);
		curerrorslope = nurnet->erravgslope(D,0);
		
		if(curerrorslope > -SLOPETHRESHOLD*inputrate && curerrorslope < SLOPETHRESHOLD*inputrate 
			&& curnodes < max_nodes && i-lastHPChange>50){

			printf("Inserting hyperplane. Error slope is %f \n",curerrorslope);

			smartaddnode(nurnet,D);
			lastHPChange = i;
			curnodes = nurnet->outdim(0);	
		}
		i++;
		if(ratedecay){inputrate = rate*((double)max_gen - i)/max_gen;}
	}
	return returnerror;
}