#include <iostream>
#include <armadillo>
#include <map>
#include <vector>
#include <set>
#include <cmath>

#include "ann.h"
#include "nnanalyzer.h"
#include "nnmap.h"
#include "selectiontrainer.h"


#include "../ppmreadwriter/annpgm.h"

using namespace std;
using namespace arma;

#define RUNAVGWID 20

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


vec getRefinedNormVec(mat A, vec b, locInfo li)
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
			if(dot(v,regionRep)+b(i)<0)
				normvec += v;
			if(dot(v,regionRep)+b(i)>0)
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

nnlayer getSelectionLayer(nn *nurnet, std::vector<vec> regionData, locInfo targetLocation, int targetSelectionNode)
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

	selector s;

	vec newSelectionWeight = zeros<vec>(numSelection);
	for (i = 0; i < numSelection; ++i) {
		if(i == targetSelectionNode)
		{
			
			s.v = A2.row(i).t();
			s.b = b2(i);
			
			s = remakeSelector(s,regionData,regionSig);
		} else {
			newSelectionWeight(i) = 0;
		}
	}

	A2.insert_cols(0,newSelectionWeight);
	A2.row(targetSelectionNode) = s.v.t();
	b2(targetSelectionNode) = s.b;
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
	}

	#ifdef DEBUG 
		cout << "---------------------Selected--------------------" << endl;
		targetLocation.printlocation();
		cout << "-------------------------------------------------" << endl;
	#endif

	if(maxErr > 5 && targetSelectionVec != -1){
		vec errlocation = targetLocation.getErrAvg();
		vec normvec = getRefinedNormVec(A0, b0, targetLocation);

		// make it so that the normal vector has length such that the average error point takes value 0.8. Constant chosen 
		// 1/x * ln(0.8/(1-0.8)) = C
		double distToIntersection = nurnetMap->computeDist(targetLocation.getTotAvg(), targetLocation.interSig);
		normvec = 1.389*normvec/distToIntersection;

		#ifdef DEBUG
			cout << "NormVec: "<< endl << normvec << endl;
			cout << "ErrLoc: "<< endl << errlocation << endl;
		#endif

		double offset = -dot(normvec,errlocation);
		//This should be combined with the above to make sure the normal vector matches the shape of the error area.
		// If it's a corner of dimension n and there is a HP boundary over which it does not change then its's a corner of dim n-1
		std::vector<vec> newRegions = createSelectionTrainingData(D,nurnet,normvec,offset);
		nnlayer newSecondLayer = getSelectionLayer(nurnet,newRegions, targetLocation,targetSelectionVec);

		#ifdef DEBUG
			cout << "Adding HP:" << normvec << "With offset: " << offset << endl;
			cout << "---------------------------------------------------" << endl;
			cout << "Selection Layer: " << newSecondLayer.A << "Selection offset: " << newSecondLayer.b << endl;
		#endif

		nurnet->addnode(2*normvec.t(),2*offset,newSecondLayer);

		#ifdef DEBUG
			nurnet->print();
		#endif

	} else {
		printf("Not enough error points\n");
	}
	delete nurnetMap;
}

#ifndef DEBUG
#define SLOPETHRESHOLD 0.01
#define FORCEDDELAY 60
#define RESOLUTION 250
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

	double error;

	while(i<max_gen && curerr > objerr){
		if(images){
			char header[100];
			sprintf(header, "imgfiles/hea/%05dall.ppm",i);			
			write_all_nn_to_image_parallel(nurnet,D,header,RESOLUTION,RESOLUTION);
			//sprintf(header, "gen%05d.pgm",i);			
			//write_hperrs_to_imgs(nurnet, header, "imgfiles/hperrfields/");
			//printf("Error slope: %f Num Nodes: %d Threshold: %f Current gen:%d\n", curerrorslope, curnodes, -SLOPETHRESHOLD*inputrate,i);
		}
		nnmap *thismap = new nnmap(nurnet,D);
		delete thismap;
		if(ratedecay){inputrate = rate*((max_gen-(double)i)/max_gen);} 
		error = nurnet->epochbackprop(D,inputrate);
		curerr = nurnet->calcerror(D,0);
		returnerror[0][i] = curerr;
		returnerror[1][i] = nurnet->calcerror(D,1);
		curerrorslope = erravgslope(error);
		
		if(curerrorslope > -SLOPETHRESHOLD*inputrate && curerrorslope < SLOPETHRESHOLD*inputrate 
			&& curnodes < max_nodes && i-lastHPChange>FORCEDDELAY){
			if(images){
				printf("Inserting hyperplane at generation %d. Error slope is %f \n",i,curerrorslope);
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