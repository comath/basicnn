#include <iostream>
#include <armadillo>
#include <map>
#include <vector>
#include <set>
#include <cmath>  
#include "ann.h"
#include "nnanalyzer.h"
#include "nnmap.h"

#include "annpgm.h"

using namespace std;
using namespace arma;



vec getRefinedNormVec(mat A, vec v, locInfo li)
{
	#ifdef DEBUG
		printf("Getting Norm Vec\n");
	#endif
	int n = A.n_rows;
	vec normvec = zeros<vec>( A.n_cols);
	
	vec regionRep = li.getTotAvg();
	#ifdef DEBUG	
		cout << "RegionRep: "<< endl << regionRep << endl;
	#endif
	
	if(li.cornerDimension == 1){
		for(int i =0; i<n; ++i){
			#ifdef DEBUG
				cout << "InterSig["<< i <<"]: "<< li.interSig[i] << endl;
			#endif
			if(li.interSig[i] == 1){
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
			cout << "InterSig["<< i <<"]: "<< li.interSig[i] << endl;
		#endif
		vec v = A.row(i).t();
		v = v/norm(v);
		if(li.interSig[i] == 1){
			if(dot(v,regionRep)>0)
				normvec += v;
			if(dot(v,regionRep)<0)
				normvec -= v;
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

nnlayer getSelectionVec(nn *nurnet, nnmap *nurnetMap, locInfo targetLocation, int targetSelectionNode)
{
	int i = 0;
	//Convert the std vector over to an arma vector to detect which level 2 selection does what to this region.
	//We don't need the actual region vector that is stored in the nnmap as this will be the result.

	vec regionSig = conv_to<vec>::from(targetLocation.includedRegions[0]);

	mat A2 = nurnet->getmat(1);
	vec b2 = nurnet->getoff(1);
	int numSelection = b2.n_rows;
	vec selection = A2*regionSig + b2;

	selection = selection.for_each( [](mat::elem_type& val) { if(val>0){val=1;} else{val=0;} } );

	vec newSelectionWeight = zeros<vec>(numSelection);
	for (i = 0; i < numSelection; ++i) {
		if(i == targetSelectionNode)
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
	nnlayer retLayer = {.A = A2, .b = b2};
	return retLayer;
}

void refinedsmartaddnode(nn *nurnet, vec_data *D)
{
	printf("Starting smartaddnode\n");
	#ifdef DEBUG
		nurnet->print();
	#endif
	nnmap *nurnetMap = new nnmap(nurnet,D);

	mat A1 = nurnet->getmat(1);
	vec b1 = nurnet->getoff(1);
	mat A0 = nurnet->getmat(0);
	vec b0 = nurnet->getoff(0);

	locInfo targetLocation;
	int maxErr =-1;
	int targetSelectionVec = -1;
	for(unsigned i =0; i<A1.n_rows;++i){
		nurnetMap->refineMap((A1.row(i)).t(),b1(i));
		locInfo curLoc = nurnetMap->getRefinedMaxErrRegInter();
		if((int)curLoc.numerrvec > maxErr){
			targetLocation = curLoc;
			maxErr = curLoc.numerrvec;
			targetSelectionVec = i;
		}
		#ifdef DEBUG
			//nurnetMap->printrefined();
		#endif
	}



	if(maxErr > 5 && targetSelectionVec != -1){
		vec errlocation = targetLocation.getErrAvg();
		vec normvec = getRefinedNormVec(A0, b0, targetLocation);
		#ifdef DEBUG
			cout << "NormVec: "<< endl << normvec << endl;
			cout << "ErrLoc: "<< endl << errlocation << endl;
		#endif
		double offset = -dot(normvec,errlocation);
		//This should be combined with the above to make sure the normal vector matches the shape of the error area.
		// If it's a corner of dimension n and there is a HP boundary over which it does not change then its's a corner of dim n-1

		nnlayer newSecondLayer = getSelectionVec(nurnet,nurnetMap, targetLocation,targetSelectionVec);
		#ifdef DEBUG
			cout << "Adding HP:" << normvec << "With offset: " << offset << endl;
			cout << "---------------------------------------------------" << endl;
			cout << "Selection Layer: " << newSecondLayer.A << "Selection offset: " << newSecondLayer.b << endl;
		#endif
		nurnet->addnode(normvec.t(),offset,newSecondLayer);
		#ifdef DEBUG
			nurnet->print();
		#endif
	} else {
		printf("Not enough error points\n");
	}
	delete nurnetMap;
}

#ifndef DEBUG
#define SLOPETHRESHOLD 0.001
#define FORCEDDELAY 60
#define RESOLUTION 1000
#endif

#ifdef DEBUG
#define SLOPETHRESHOLD 0.05
#define FORCEDDELAY 20
#define RESOLUTION 250
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
	//if(images)
	//	fp = startHistory("imgfiles/hea/latest.nnh", nurnet, D, max_gen);

	


	while(i<max_gen && curerr > objerr){
		if(images){
			char header[100];
			sprintf(header, "imgfiles/hea/%05dall.ppm",i);			
			write_all_nn_to_image_parallel(nurnet,D,header,RESOLUTION,RESOLUTION);
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

			refinedsmartaddnode(nurnet,D);
			lastHPChange = i;
			curnodes = nurnet->outdim(0);	
		}
		i++;
		//if(images)
		//	appendNNToHistory(nurnet,&fp);
		if(ratedecay){inputrate = rate*((double)max_gen - i)/max_gen;}
	}
	return returnerror;
}