#include "nnmap.h"

using namespace std;
using namespace arma;

std::set<int> getSetFromSig(std::vector<int> sig){
	std::set<int> set;
	for (int i = 0; i < (int)sig.size(); ++i){
		if(sig[i]==1){
			set.insert(i);
		}	
	}
	return set;
}

void printset(std::set<int> set){
	std::set<int>::iterator it;
	printf("{");
	for (it = set.begin(); it != set.end(); ++it){
	  	cout << *it;
	  	if(it != set.end())
	  		cout << ", ";
	}
	printf("}\n");
}

void printsig(std::vector<int> sig)
{
	for (unsigned i = 0; i < sig.size(); ++i)
	{
		cout << sig[i];
	}
	cout << endl;
}

int getSelectionOfRegion(std::vector<int> regSig,vec selectV,double offset)
{	
	vec curRegSigV = conv_to<vec>::from(regSig);
	double result = dot(selectV,curRegSigV) + offset;
	int r;
	if(result>0){
		r = 1;
	} else {
		r = 0;
	}
	return r;
}


#ifndef ERRORTHRESHOLD
#define ERRORTHRESHOLD 0.01
#endif

#ifndef TUBETHRESHOLD
#define TUBETHRESHOLD 0.3
#endif

#ifndef SCALEDTUBETHRESHOLD
#define SCALEDTUBETHRESHOLD 2
#endif


void nnmap::setUpHps()
{
	int i =0;
	int n = b0.n_rows;
	rowvec curvec;
	rowvec normcurvec;
	double scaling = 1;
	for(i=0;i<n;i++){
		scaling = norm(A0.row(i));
		hpNormals.push_back(A0.row(i)/scaling);
		hpoffsets.push_back(b0(i)*curvec/scaling);
	}
}

vec nnmap::computeDistToHPS(vec v)
{
	unsigned n = hpNormals.size();
	vec retvec = zeros<vec>(n);
	for(unsigned i =0;i<n;++i){
		retvec(i) = abs(dot((hpoffsets[i]+v.t()),hpNormals[i]));
	}
	return retvec;
}

void nnmap::setUpCashe(nn *thisnn)
{
	A0 = thisnn->getmat(0);
	b0 = thisnn->getoff(0);
	A1 = thisnn->getmat(1);
	b1 = thisnn->getoff(1);
	this->setUpHps();
}
IntersectionBasis nnmap::getIntersectionBasis(std::set<int> indexes)
{
	if(interCashe.count(indexes) == 0){
		std::vector<int> stdVecIndexes(indexes.begin(), indexes.end());
		uvec rowsToInclude = conv_to<uvec>::from(stdVecIndexes);
		mat tempA = A0.rows(rowsToInclude);
		mat B;
		vec x;
		bool test1 = null(B,tempA);
		bool test2 = solve(x,tempA,b0.rows(rowsToInclude));
		IntersectionBasis curIB;
		curIB.basis = B;
		curIB.aSolution = x;
		curIB.success = (test1 && test2);
		interCashe.emplace(indexes,curIB);
		return curIB;
	} else {
		return interCashe[indexes];
	}
}

double nnmap::computeDist(vec p, std::set<int> indexes)
{
	IntersectionBasis curIB = this->getIntersectionBasis(indexes);
	
	vec px = p+curIB.aSolution;
	vec pparallel = zeros<vec>(p.n_rows);
	for (unsigned i = 0; i < (curIB.basis).n_cols; ++i){
		vec Bi = (curIB.aSolution).col(i);
		pparallel += dot(Bi,px)*Bi;
	}
	return norm(px-pparallel);

	return -1;
}

std::vector<int> nnmap::getInterSig(vec v)
{
	vec dist = this->computeDistToHPS(v);
	uvec indsort = sort_index(dist,"accend");
	unsigned j = 1;
	unsigned n = dist.n_rows;
	std::vector<int> sig (n,0);
	unsigned dimension = v.n_rows;
	for(unsigned k = 1; k<dimension+1; ++k){
		std::set<int> rowsToInclude;
		for(unsigned l=0; l<k;++l){
			rowsToInclude.insert(indsort(l));
		}
		double curDist = this->computeDist(v, rowsToInclude);
		if(curDist>0 && dist(indsort(k)) > SCALEDTUBETHRESHOLD*curDist){
			j=k;
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



std::vector<int> nnmap::getRegionSig(vec v)
{
	vec w = A0*v + b0;
	int n = b0.n_rows;
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

nnmap::nnmap(nn *nurnet, vec_data *D){ 
		#ifdef DEBUG	
			printf("Creating the NN map\n");
		#endif
		int numdata = D->numdata;
		setUpCashe(nurnet);
		numHPs = b0.n_rows;
		dimension = A0.n_cols;
		bool err = false;
		for (int i = 0; i < numdata; ++i)
		{
			err = (nurnet->calcerror(D->data[i],1));
			this->addvector(err,D->data[i].coords);
		}
	}

void nnmap::addvector(bool err, vec v){
	const vector<int> regionSig = getRegionSig(v);
	
	const vector<int> interSig = getInterSig(v);
	
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

void nnmap::refineMap(vec selectV, double offset)
{
	if(!refinedMap.empty()){
		refinedMap.clear();
	}
	unsigned i=0;
	std::set<int> interSet;

	for (auto& interit: regInter){
		#ifdef DEBUG
			cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
			cout << "working on intersection: ";
			printsig(interit.first);
		#endif
		interSet = getSetFromSig(interit.first);
		#ifdef DEBUG
			cout << "It has set: ";
			printset(interSet);
		#endif
		std::map<std::vector<int>, locInfo> cpyOfRegMap = interit.second;
		std::vector<locInfo> locReg;

		for (auto& regit: cpyOfRegMap){
			locInfo curReg = regit.second;
			regit.second.includedRegions.push_back(regit.first);
			#ifdef DEBUG
				cout << "Adding Region: ";
				printsig(regit.first);
				printsig(regit.second.includedRegions[0]);
			#endif
		}
		#ifdef DEBUG
			cout << "Successfully populated regions: " << endl;
			for (auto& regit: cpyOfRegMap){
				printsig(regit.second.includedRegions[0]);
			}
		#endif
		while(!cpyOfRegMap.empty()){
			locInfo curRegInfo = cpyOfRegMap.begin()->second;
			std::vector<int> seedRegionForCombo = cpyOfRegMap.begin()->first;
			#ifdef DEBUG
				cout << "======================================================================================" << endl;
				cout << "Working on region: " << endl;
					printsig(cpyOfRegMap.begin()->first);
			#endif
			bool lastItCombine = true;
			while(lastItCombine){
				lastItCombine = false;
				std::vector<std::vector<int>> connectedRegions = curRegInfo.includedRegions;
				int result = getSelectionOfRegion(connectedRegions[0],selectV,offset);
				
				std::set<int>::iterator it;
				
  				for (i = 0; i < connectedRegions.size(); ++i){
  					std::vector<int> curRegSig = connectedRegions[i];
  					#ifdef DEBUG
  						cout << "===========================================" << endl;
						cout << "Working on region: " << endl;
						printsig(curRegSig);
						cout << "With selection: " << getSelectionOfRegion(curRegSig,selectV,offset) << endl;
					#endif
  					for (it = interSet.begin(); it != interSet.end(); ++it){
  						int j = *it;
						std::vector<int> testRegSig = curRegSig;
  						if(curRegSig[j]==1){
  							testRegSig[j]=0;
  						} else {
  							testRegSig[j]=1;
  						}
  						int numRepresentativesInMap = cpyOfRegMap.count(testRegSig);
  						int testRegVal = getSelectionOfRegion(testRegSig,selectV,offset);
  						#ifdef DEBUG
  							cout << "===================" << endl;
							cout << "Testing region: " << endl;
							printsig(testRegSig);
							cout << "With selection: " << testRegVal  << " and count " << numRepresentativesInMap << endl;

						#endif
  						if(numRepresentativesInMap > 0 && result == testRegVal && testRegSig != seedRegionForCombo){  							
	  						curRegInfo.combine(cpyOfRegMap[testRegSig]);
	  						curRegInfo.hyperplanesCrossed.insert(j);
	  						cpyOfRegMap.erase(testRegSig);
	  						#ifdef DEBUG
	  							cout << "After combine " << cpyOfRegMap.count(testRegSig) << endl;
	  						#endif
	  						lastItCombine = true;
	  					}
  					}
  				}
			}
			#ifdef DEBUG
  				printf("Out of combination phase of the reg, adding to region. Pre add is of length %d \n", locReg.size());
	  		#endif
			//Calculates the type of corner, and gives a -1 for a degenerate corner
			double log2NumCurRegions = log(curRegInfo.includedRegions.size())/log(2);
			if(log2NumCurRegions == curRegInfo.hyperplanesCrossed.size()){
				curRegInfo.cornerDimension = interSet.size() - curRegInfo.hyperplanesCrossed.size();
			} else {
				curRegInfo.cornerDimension = -1;
			}
			locReg.push_back(curRegInfo);
			#ifdef DEBUG
  				printf("Adding the following location to intersection ");
  				printsig(interit.first);
  				curRegInfo.printlocation();
				printf("Post add length of region list is %d \n", locReg.size());

				cout << "Count of final region to destroy is: " << cpyOfRegMap.count(cpyOfRegMap.begin()->first) << endl;
	  		#endif
			cpyOfRegMap.erase(cpyOfRegMap.begin());		
		}
		
		refinedMap.emplace(interit.first,locReg);
	}
}

locInfo nnmap::getRefinedMaxErrRegInter()
{
	#ifdef DEBUG
		printf("Deterimining the location with maximum error.\n");
	#endif
	locInfo maxLocInfo;
	std::vector<int> maxInterSig;
	unsigned maxNumErr = 0;
	for (auto& interit: refinedMap){
		std::vector<locInfo> locReg = interit.second;
		for(unsigned i =0;i<locReg.size();++i){
			if(locReg[i].cornerDimension > 0){
				if((locReg[i].cornerDimension)*locReg[i].numerrvec > maxNumErr){
					maxNumErr = locReg[i].numerrvec;
					maxInterSig = interit.first;
					maxLocInfo = locReg[i];
				}
			}
		}
	}

	std::set<int>::iterator it;

	for (it=maxLocInfo.hyperplanesCrossed.begin(); it!=maxLocInfo.hyperplanesCrossed.end(); ++it){
		#ifdef DEBUG
			cout << *it << ", ";
		#endif
		maxInterSig[*it] =0;
	}
	#ifdef DEBUG
		cout << endl ;
	#endif

	maxLocInfo.interSig = maxInterSig;
	return maxLocInfo;
}



void nnmap::print()
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

void nnmap::printRefined()
{
	for (auto& firstit: refinedMap){
		cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
		cout << "Intersection Signature:";
		printsig(firstit.first);
		for (unsigned i = 0; i < firstit.second.size(); ++i)
		{	
			cout << "<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>" << endl;
			cout << "Location:" << i << endl;
			firstit.second[i].printlocation();
		}
	}
}


// -------------------------------All section---------------------------

locInfo nnmap::getRegInterInfo(const locSig sig) 
{
	if(regInter[sig.interSig].count(sig.regionSig) == 0){
		return locInfo();
	} else {
		return regInter[sig.interSig].at(sig.regionSig); 
	}
}
vec nnmap::getRegInterAvgVec(const locSig sig)
{
	if(regInter[sig.interSig].count(sig.regionSig) == 0){
		return 0;
	} else {
		return regInter[sig.interSig].at(sig.regionSig).getTotAvg(); 
	}
}
int nnmap::getRegInterPop(const locSig sig)
{
	if(regInter[sig.interSig].count(sig.regionSig) == 0){
		return 0;
	} else {
		return regInter[sig.interSig].at(sig.regionSig).numvec; 
	}
}
//-----------------------------------Error section-----------------------

vec nnmap::getRegInterAvgErrVec(const locSig sig)
{
	if(regInter[sig.interSig].count(sig.regionSig) == 0){
		return 0;
	} else {
		return regInter[sig.interSig].at(sig.regionSig).getErrAvg(); 
	} 
}
int nnmap::getRegInterErrPop(const locSig sig)
{
	if(regInter[sig.interSig].count(sig.regionSig) == 0){
		return 0;
	} else {
		return regInter[sig.interSig].at(sig.regionSig).numerrvec; 
	}
}




locInfo::locInfo() 
{
	numvec =0;
	numerrvec =0;
}
locInfo::locInfo(bool err,vec v)
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
void locInfo::print()
{
	cout << "NumTotVec:" << numvec << "      NumErrVec:" << numerrvec << endl;
	if(numerrvec != 0)
		cout << "Average Error Vec:" << toterrvec/numerrvec << endl;
	if(numvec != 0)
		cout << "Average Vec:" << toterrvec/numvec << endl;
}
void locInfo::printlocation()
{
	cout << "toterrvec : " << toterrvec << endl;
	cout << "numerrvec : " << numerrvec << endl;
	cout << "numvec : " << numvec <<  endl;
	cout << "totvec : " << totvec <<  endl;
	cout << "cornerDimension : " << cornerDimension <<  endl;
	cout << "hyperplanesCrossed : ";
	printset(hyperplanesCrossed);
	printf("With Regions: \n");
	for(unsigned i=0; i<includedRegions.size();++i){
		printsig(includedRegions[i]);
	}
}
void locInfo::addvector(bool err, vec v)
{
	if(err) {
		numerrvec++;
		toterrvec += v;
	}
	totvec+= v;
	numvec++;
}
int locInfo::getNumErr() {return numerrvec;}
vec locInfo::getTotAvg()	{ return totvec/numvec;	}
vec locInfo::getErrAvg()	{
	return toterrvec/numerrvec; 
}

void locInfo::combine(locInfo other)
{
	#ifdef DEBUG 
		printf("-------\nCombining two regions\n");
		cout << "toterrvec : " << toterrvec << " other : " << other.toterrvec << endl;
		cout << "numerrvec : " << numerrvec << " other : " << other.numerrvec << endl;
		cout << "numvec : " << numvec << " other : " << other.numvec << endl;
		cout << "totvec : " << totvec << " other : " << other.totvec << endl;
	#endif
	toterrvec += other.toterrvec;
	numerrvec += other.numerrvec;
	numvec += other.numvec;
	totvec += other.totvec;
	for(unsigned i=0; i<other.includedRegions.size();++i){
		includedRegions.push_back(other.includedRegions[i]);
	}
	#ifdef DEBUG 
		printf(" Post combo two regions\n");
		cout << "toterrvec : " << toterrvec << endl;
		cout << "numerrvec : " << numerrvec << endl;
		cout << "numvec : " << numvec <<  endl;
		cout << "totvec : " << totvec <<  endl;
	#endif
}