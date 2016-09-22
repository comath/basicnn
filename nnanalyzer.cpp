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
#define TUBETHRESHOLD 0.3
#endif

#ifndef SCALEDTUBETHRESHOLD
#define SCALEDTUBETHRESHOLD 2
#endif


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
		retvec(i) = abs(dot((normcurvec+v.t()),curvec));
	}
	return retvec;
}

std::vector<int> getInterSig(vec v, mat A, vec b)
{
	vec dist = computeDistToHPS(A,b,v);

	uvec indsort = sort_index(dist,"accend");
	unsigned j = 1;
	unsigned n = dist.n_rows;
	std::vector<int> sig (n,0);

	//cout << "-----------------------------------------------------" << endl;
	//cout << "Distances to Local HPs: " << endl << dist << endl;
	
	double minval = dist(indsort(j));
	dist = dist/minval;
	if(var(dist) < 0.2){
		return sig;
	}
	for(unsigned k = 0; k<n-2; ++k){
		//cout << "Looking at distance " << indsort(k) << " : " << dist(indsort(k)) <<endl;
		if(dist(indsort(k)) < SCALEDTUBETHRESHOLD && abs(dist(indsort(k+1))-dist(indsort(k+2)))>SCALEDTUBETHRESHOLD){
			j++;
		}
	}
	
	
	if(j > v.n_rows)
		j = v.n_rows;
	
	for (unsigned i = 0; i < j; ++i)
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
			sig[i] = 0;
		}
	}
	return sig;
}

typedef struct locInfo {
	locInfo() 
	{
		numvec =0;
		numerrvec =0;
	}
	locInfo(bool err,vec v)
	{
		if(err) {
			numerrvec =1;
			toterrvec = v;
			numvec =1;
			totvec = v;
		} else {
			numerrvec =0;
			toterrvec = zeros<vec>(v.n_rows);
			numvec =1;
			totvec = v;
		}
	}
	void print()
	{
		cout << "NumTotVec:" << numvec << "      NumErrVec:" << numerrvec << endl;
		if(numerrvec != 0)
			cout << "Average Error Vec:" << toterrvec/numerrvec << endl;
		if(numvec != 0)
			cout << "Average Vec:" << toterrvec/numvec << endl;
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
	int getNumErr() {return numerrvec;}
	vec getTotAvg()	{ return totvec/numvec;	}
	vec getErrAvg()	{
		cout << toterrvec << endl;
		cout << numerrvec << endl;
		return toterrvec/numerrvec; 
	}
	vec totvec;
	unsigned numvec;
	vec toterrvec;
	unsigned numerrvec;
} locInfo;

void printsig(std::vector<int> sig)
{
	for (unsigned i = 0; i < sig.size(); ++i)
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
	int dimension;
public:
	nnmap(nn *nurnet, vec_data *D){ 
		#ifdef DEBUG	
			printf("Creating the NN map\n");
		#endif
		int numdata = D->numdata;
		mat A = nurnet->getmat(0);
		vec b = nurnet->getoff(0);
		numHPs = b.n_rows;
		dimension = A.n_cols;
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
		if(reg.count(regionSig) == 0){
			reg.emplace(regionSig,locInfo(err,v));
		} else {
			reg[regionSig].addvector(err, v);
		}
		const vector<int> interSig = getInterSig(v,A,b);
		if(inter.count(interSig) == 0){
			inter.emplace(interSig, locInfo(err, v));
		} else {
			inter[interSig].addvector(err, v);
		}
		
		if(regInter[interSig].count(regionSig) == 0){
			regInter[interSig].emplace(regionSig,locInfo(err, v));
		} else {
			if(regInter[interSig].count(regionSig) == 0){
				regInter[interSig].emplace(regionSig,locInfo(err, v));
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
		if(regInter[sig.interSig].count(sig.regionSig) == 0){
			return locInfo();
		} else {
			return regInter[sig.interSig].at(sig.regionSig); 
		}
	}
	vec getRegInterAvgVec(const locSig sig)
	{
		if(regInter[sig.interSig].count(sig.regionSig) == 0){
			return 0;
		} else {
			return regInter[sig.interSig].at(sig.regionSig).getTotAvg(); 
		}
	}
	int getRegInterPop(const locSig sig)
	{
		if(regInter[sig.interSig].count(sig.regionSig) == 0){
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
			return reg.at(sig).numerrvec; 
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
			return inter.at(sig).numerrvec; 
		}
	}
	vec getRegInterAvgErrVec(const locSig sig)
	{
		printf("Getting the average error.\n");
		if(regInter[sig.interSig].count(sig.regionSig) == 0){
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
			return regInter[sig.interSig].at(sig.regionSig).numerrvec; 
		}
	}
	//--------------------------Returning useful info section-------------
	locSig getMaxErrRegInter(mat A, vec b)
	{
		#ifdef DEBUG
			printf("Deterimining the location with maximum error.\n");
		#endif
		std::vector<int> interSigMaxErr ;
		std::vector<int> regionSigMaxErr;
		unsigned maxNumErr = 0;
  		for (auto& firstit: regInter){
  			int weight =0;
  			for(int i = 0; i < numHPs; ++i){
  				if(firstit.first[i] == 1)
  					weight++;
  			}
  			if(weight > dimension)
  				weight=dimension;

  			locSig oddLocSig = getOddRegion(firstit.first,A,b);

  			for (auto& secit: firstit.second){
  				if(secit.first == oddLocSig.regionSig){
	  				if(2*sqrt(weight)*secit.second.numerrvec > maxNumErr){
	    				interSigMaxErr = firstit.first;
	    				regionSigMaxErr = secit.first;
	    				maxNumErr = 2*sqrt(weight)*secit.second.numerrvec;
	  				}
	  			} else {
	  				if(sqrt(weight)*secit.second.numerrvec > maxNumErr){
	    				interSigMaxErr = firstit.first;
	    				regionSigMaxErr = secit.first;
	    				maxNumErr = sqrt(weight)*secit.second.numerrvec;
	  				}
	  			}
  			}
  		}
  		#ifdef DEBUG
	  		printf("The region with maximum error is : \n");
	  		printsig(regionSigMaxErr);
	  		printf("near intersection:\n");
	  		printsig(interSigMaxErr);
  		#endif
  		locSig sigMaxErr = {.regionSig = regionSigMaxErr, .interSig = interSigMaxErr};
  		return sigMaxErr;
	}
	std::vector<int> getMaxErrInter()
	{
		#ifdef DEBUG
			printf("Deterimining the location with maximum error.\n");
		#endif
		std::vector<int> interSigMaxErr;
		int maxNumErr = -1;
  		for (auto& interit: inter){
  			int weight =0;
  			for(int i = 0; i < numHPs; ++i){
  				if(interit.first[i] == 1)
  					weight++;
  			}
  			if(weight > dimension)
  				weight=dimension;
  			if(weight > 1){
	  			if(sqrt(weight)*interit.second.numerrvec > maxNumErr){
	    			interSigMaxErr = interit.first;
	    			maxNumErr = sqrt(weight)*interit.second.numerrvec;
	  			}
	  		}
  		}
  		#ifdef DEBUG
  			printf("The intersection with the most error is:\n");
  			printsig(interSigMaxErr);
  		#endif
  		return interSigMaxErr;
	}
	std::vector<std::vector<int>> getLocalRegions(const std::vector<int> sig)
	{
		std::vector<std::vector<int>> localRegions;
  		for (auto& regIt: regInter[sig]){
  			localRegions.push_back(regIt.first);
  		}
  		return localRegions;
	}

	vec getAvgRegionVal(std::vector<int> interSig, mat A, vec b)
	{
		#ifdef DEBUG
			cout << "Matrix: " << endl << A << endl;
			cout << "Offset: " << endl << b << endl;
		#endif
		vec avgRegionVal = zeros<vec>(A.n_rows);
		int numRegions =0;
  		for (auto& regIt: regInter[interSig]){
  			vec curSig =  zeros<vec>(numHPs);
  			std::vector<int> regionSig = regIt.first;
  			for (int i = 0; i < numHPs; ++i){
  				curSig(i) = regionSig[i];
  			}
  			#ifdef DEBUG
  				cout << "Current Region: " << endl << curSig << endl;	
  			#endif 
  			vec result = A*curSig + b;
  			result.for_each( [](mat::elem_type& val) { if(val>0){val=1;} else{val=0;} } );
  			#ifdef DEBUG
  				cout << "Value: " << endl << result << endl;
  			#endif
  			avgRegionVal += result;
  			numRegions++;
  		}
  		avgRegionVal = avgRegionVal/numRegions;
  		cout << "Average Region Val: " << endl << avgRegionVal << endl;
  		return avgRegionVal;
	}

	locSig getOddRegion(std::vector<int> interSig, mat A, vec b)
	{	
		#ifdef DEBUG
			cout << "Matrix: " << endl << A << endl;
			cout << "Offset: " << endl << b << endl;
		#endif
		vec avgRegionVal = getAvgRegionVal(interSig,A,b);

  		double maxDist =-1;
  		std::vector<int> regSig;
  		for (auto& regIt: regInter[interSig]){
  			vec curSig =  zeros<vec>(numHPs);
  			std::vector<int> regionSig = regIt.first;
  			for (int i = 0; i < numHPs; ++i){
  				curSig(i) = regionSig[i];
  			}
  			vec result = A*curSig + b;
  			result.for_each( [](mat::elem_type& val) { if(val>0){val=1;} else{val=0;} } );
  			double curDist = norm(result-avgRegionVal);
  			if(maxDist < curDist){
  				regSig = regionSig;
  				maxDist = curDist;
  			}
  		}
  		#ifdef DEBUG
  			cout << "Chosen Region: " << endl;
  			printsig(regSig);
  		#endif
  		locSig sigMaxErr = {.regionSig = regSig, .interSig = interSig};
  		return sigMaxErr;
	}

	void print()
	{
		for (auto& firstit: regInter){
  			for (auto& secit: firstit.second){
				cout << "Intersection Signature:";
  				printsig(firstit.first);
  				cout << "Region Signature:";
				printsig(secit.first);
    			secit.second.print();  				
  			}
  		}
	}
};

vec getNormVec(nn *nurnet, nnmap *locInfo, locSig sig)
{
	#ifdef DEBUG
		printf("Getting Norm Vec\n");
	#endif
	mat A = nurnet->getmat(0);
	int m = A.n_cols;
	vec normvec = zeros<vec>(m);
	int n = A.n_rows;
	vec regionRep = locInfo->getRegInterAvgVec(sig);
	#ifdef DEBUG	
		cout << "RegionRep: "<< endl << regionRep << endl;
	#endif
	int numLocalRegions =0;
	for (int i = 0; i < n; ++i) {
		if(sig.interSig[i] == 1)
			numLocalRegions++;
	}
	if(numLocalRegions == 1){
		for(int i =0; i<n; ++i){
			#ifdef DEBUG
				cout << "InterSig["<< i <<"]: "<< sig.interSig[i] << endl;
			#endif
			if(sig.interSig[i] == 1){
				vec v = A.row(i).t();
				normvec = normvec.randn();
				vec r = randu<vec>(1);
				double randconst = r(0)*0.66 + 0.2;
				normvec = normvec - randconst*dot(normvec,v)*v/(norm(normvec)*norm(v));
				normvec = normvec/norm(normvec);
				if(dot(normvec,regionRep)>0)
					return normvec;
				if(dot(normvec,regionRep)<0)
					return -normvec;
			}
		}
	}

	for(int i =0; i<n; ++i){
		#ifdef DEBUG
			cout << "InterSig["<< i <<"]: "<< sig.interSig[i] << endl;
		#endif
		vec v = A.row(i).t();
		v = v/norm(v);
		if(sig.interSig[i] == 1){
			if(dot(v,regionRep)>0)
				normvec += v;
			if(dot(v,regionRep)<0)
				normvec -= v;
		}
	}
	#ifdef DEBUG
		for(int i =0; i<n; ++i){
			cout << "RegionSig[" << i << "]: "<< sig.regionSig[i] << endl;
		}
	#endif
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
	//Convert the std vector over to an arma vector to detect which level 2 selection does what to this region.
	//We don't need the actual region vector that is stored in the nnmap as this will be the result.

	vec regionSig = conv_to<vec>::from(sig.regionSig);

	mat A2 = nurnet->getmat(1);
	vec b2 = nurnet->getoff(1);
	int numSelection = b2.n_rows;
	vec selection = A2*regionSig + b2;

	selection = selection.for_each( [](mat::elem_type& val) { if(val>0){val=1;} else{val=0;} } );

	
	std::vector<std::vector<int>> localRegions = nurnetMap->getLocalRegions(sig.interSig);
	int numLocalRegions = localRegions.size();
	vec averageDiff = zeros<vec>(numSelection);

	//The shape of the error here is relevant, is it pointy? is it flat? There will have to be an improvement here.
	for (i = 0; i < numLocalRegions; ++i)
	{	
		vec curRegionSig = conv_to<vec>::from(localRegions[i]);
		vec curRegionSelection = A2*curRegionSig + b2;
		curRegionSelection = curRegionSelection.for_each( [](mat::elem_type& val) { if(val>0){val=1;} else{val=0;} } );
		curRegionSelection = curRegionSelection - selection;
		curRegionSelection = curRegionSelection.for_each( [](mat::elem_type& val) { if(val==0){val=0;} else{val=1;} } );
		averageDiff += curRegionSelection;
		
	}
	averageDiff = averageDiff/numLocalRegions;
	

	vec newSelectionWeight = zeros<vec>(numSelection);
	for (i = 0; i < numSelection; ++i) {
		if(averageDiff(i)>0.2)
		{
			vec curSelectorVec = A2.row(i).t();
			int n = curSelectorVec.n_rows;
			double curSelectorOff = b2(i);
			#ifdef DEBUG
				cout << "Current selection vector: " << curSelectorVec << endl;
				cout << "Current Selection Offset: " << curSelectorOff << endl; 
			#endif
			// Make the selection vector positive (all values >0), and adjust the offset to compensate.
			// For each HP for which we have to swap the sign, we have to negate that hyperplane. 
			// We can do this by manpulating the region signatures.
			// This has to be done per selection vector as they will have different sinages. 
			vec signs = zeros<vec>(n);
			for(int k=0;k<n;++k) {
				if(curSelectorVec(k)>0){ 
					signs(k)=1; 
				} 
				else if(curSelectorVec(k)<0){ 
					signs(k)=-1;
				}
				}
			curSelectorOff += dot(((signs % signs) - signs)/2,curSelectorVec);
			curSelectorVec %= signs;
			#ifdef DEBUG
				cout << "Signs: " << signs << endl;
				cout << ((signs % signs) - signs)/2 << endl;
				cout << "Corrected Selection offset: " << curSelectorOff << endl;
				cout << "Corrected Selection vec: " << curSelectorVec << endl;
			#endif
			vec correctedRegionSig = correctRegionSig(regionSig,signs);
			int numPositiveSides =0;
			for(int k = 0; k<n; ++k){
				if(correctedRegionSig(k) == 1){
					numPositiveSides++;
				} 
			}
			double regionValue = dot(curSelectorVec,correctedRegionSig);
			#ifdef DEBUG
				cout << "Corrected Region Signature: " << correctedRegionSig << endl;
				cout << "Region Value: " << regionValue << endl;
			#endif
	
			
			for (int k = 0; k < n; ++k)	{
				if(correctedRegionSig(k) == 1){
					curSelectorVec(k) -= 1.2*(regionValue + curSelectorOff)/(numPositiveSides);
				} 
			}
			newSelectionWeight(i) = 0.8*(regionValue + curSelectorOff);
			
			if(regionValue > -curSelectorOff){
				curSelectorOff += newSelectionWeight(i);
				newSelectionWeight(i) = -newSelectionWeight(i);
			}
			curSelectorOff += dot(((signs % signs) - signs)/2,curSelectorVec);
			curSelectorVec = curSelectorVec % signs;
	
			A2.row(i) = curSelectorVec.t();
			b2(i) = curSelectorOff; 
		} else {
			newSelectionWeight(i) = 0;
		}
	}

	A2.insert_cols(0,newSelectionWeight);
	nnlayer retLayer = {.A = 1.2*A2, .b = 1.2*b2};
	return retLayer;
}

void smartaddnode(nn *nurnet, vec_data *D)
{
	
	printf("Starting smartaddnode\n");
	nurnet->print();
	nnmap *nurnetMap = new nnmap(nurnet,D);

	mat A = nurnet->getmat(1);
	vec b = nurnet->getoff(1);

	/*
	//The "correct" way
	locSig maxsig = nurnetMap->getMaxErrRegInter(A,b);
	*/
	// THe "incorrect" way
	
	std::vector<int> interSig = nurnetMap->getMaxErrInter();
	locSig maxsig = nurnetMap->getOddRegion(interSig, A, b);
	if(nurnetMap->getRegInterErrPop(maxsig) > 5){
		vec errlocation = nurnetMap->getRegInterAvgErrVec(maxsig);
		vec normvec = getNormVec(nurnet,nurnetMap,maxsig);
		#ifdef DEBUG
			cout << "NormVec: "<< endl << normvec << endl;
			cout << "ErrLoc: "<< endl << errlocation << endl;
		#endif
		double offset = -dot(normvec,errlocation);
		//This should be combined with the above to make sure the normal vector matches the shape of the error area.
		// If it's a corner of dimension n and there is a HP boundary over which it does not change then its's a corner of dim n-1
		nnlayer newSecondLayer = getSelectionVec(nurnet,nurnetMap, maxsig);
		#ifdef DEBUG
			cout << "Adding HP:" << normvec << "With offset: " << offset << endl;
			cout << "---------------------------------------------------" << endl;
			cout << "Selection Layer: " << newSecondLayer.A << "Selection offset: " << newSecondLayer.b << endl;
		#endif
		nurnet->addnode(normvec.t(),offset,newSecondLayer);
		nurnet->print();
	} else {
		printf("Not enough error points\n");
	}
	delete nurnetMap;
}

#ifndef SLOPETHRESHOLD
#define SLOPETHRESHOLD 0.04
#endif

#ifndef FORCEDDELAY
#define FORCEDDELAY 20
#endif

double ** adaptivebackprop(nn *nurnet, vec_data *D, double rate, double objerr, int max_gen, int max_nodes, bool ratedecay, bool images)
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
	fstream fp;
	if(images)
		fp = startHistory("imgfiles/hea/latest.nnh", nurnet, D, max_gen);

	while(i<max_gen && curerr > objerr){
		if(images){
			char header[100];
			sprintf(header, "imgfiles/hea/%05dsig.ppm",i);
			write_nn_to_img(nurnet,header,500,500,0);
			write_data_to_img(D,header);
			sprintf(header, "imgfiles/hea/%05dheav.ppm",i);
			write_nn_to_img(nurnet,header,500,500,1);
			write_data_to_img(D,header);
			sprintf(header, "imgfiles/hea/%05dregions.ppm",i);
			write_nn_regions_to_img(nurnet,header,500,500,1);
			write_data_to_img(D,header);
			sprintf(header, "imgfiles/hea/%05dintersections.ppm",i);
			write_nn_inter_to_img(nurnet,header,500,500,1);
			write_data_to_img(D,header);
			printf("Error slope: %f Num Nodes: %d Threshold: %f Current gen:%d\n", curerrorslope, curnodes, -SLOPETHRESHOLD*inputrate,i);
		}
		if(ratedecay){inputrate = rate*((max_gen-(double)i)/max_gen);} 
		nurnet->epochbackprop(D,inputrate);
		curerr = nurnet->calcerror(D,0);
		returnerror[0][i] = curerr;
		returnerror[1][i] = nurnet->calcerror(D,1);
		curerrorslope = nurnet->erravgslope(D,0);
		
		if(curerrorslope > -SLOPETHRESHOLD*inputrate && curerrorslope < SLOPETHRESHOLD*inputrate 
			&& curnodes < max_nodes && i-lastHPChange>FORCEDDELAY){
			if(images){
				printf("Inserting hyperplane. Error slope is %f \n",curerrorslope);
			}

			smartaddnode(nurnet,D);
			lastHPChange = i;
			curnodes = nurnet->outdim(0);	
		}
		i++;
		if(images)
			appendNNToHistory(nurnet,&fp);
		if(ratedecay){inputrate = rate*((double)max_gen - i)/max_gen;}
	}
	return returnerror;
}