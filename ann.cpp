#include <iostream>
#include <armadillo>
#include "ann.h"
#include "annpgm.h"

using namespace std;
using namespace arma;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void delvec_data(vec_data *D)
{
	delete[] D->data;
	delete D;
}

void nn::initlayerofnn(int i,int indim, int outdim)
{
	layers[i].A = zeros<mat>(outdim,indim);
	layers[i].b = zeros<mat>(outdim); 
}


nn::nn(int inputdim, int width1, int width2, int outdim)
{
	depth = 3;
	layers = new nnlayer[depth];
	initlayerofnn(0,inputdim,width1);
	initlayerofnn(1,width1,width2);
	initlayerofnn(2,width2,outdim);
}

nn::nn(int inputdim, int width1,int outdim)
{
	depth = 2;
	layers = new nnlayer[depth];
	initlayerofnn(0,inputdim,width1);
	initlayerofnn(1,width1,outdim);
}

nn::nn(const char *filename){
	printf("Loading: %s", filename);
	fstream  fp;
	fp.open(filename, ios::in);
	
	if(fp.is_open()){
		int magicnum;
		fp >> magicnum;
		if(magicnum == 34){
		if(fp.get() =='\n'){
		fp >> depth;
		if(fp.get() =='\n'){
		printf(".");
		int i =0;
		bool At, bt;
		layers = new nnlayer[depth];
		for(i=0;i<depth;i++){ initlayerofnn(i,0,0);}
		for(i=0;i<depth;i++){
			printf(".");
			At = layers[i].A.load(fp);
			bt = layers[i].b.load(fp);
			if(!(At) || !(bt)) { break; }
		}
		printf("Load Successful\n");
	}}}
	}
}

nn::~nn()
{
	delete[] layers;
}

void nn::print()
{
	int i=0;
	for(i=0;i<depth;i++){
		cout << "Layer " << i << " matrix: " << endl;
		cout << layers[i].A << endl;
		cout << "Layer " << i << " offset: " << endl;
		cout << layers[i].b << endl;
	}
}

void nn::randfillnn(double weight)
{
	arma_rng::set_seed_random();
	int i = 0;
	for(i=0;i<(depth);i++){
		(layers[i].A.randn())*weight*(1/(1+i));
		(layers[i].b.randn())*weight*(1/(1+i));
	}
}

vec nn::evalnn(vec input, int func)
{
	return evalnn_layer(input, func, depth);
}

vec nn::evalnn_layer(vec input, int func, int layernum)
{
	if(layernum == 0) {
		return input;
	}
	int i =0;
	vec output;
	for(i=0;i<layernum;i++)
	{	
		output = (layers[i].A)*input + (layers[i].b);
		if(func == 0){
			output = output.for_each( [](mat::elem_type& val) { val = 1/(1+exp(-val)); } );
		} else {
			output = output.for_each( [](mat::elem_type& val) { if(val>0){val=1;} else{val=0;} } );
		}
		input = output;
	}
	return output;
}

bool nn::save(const char *filename)
{
	printf("Saving: %s ", filename);
	fstream  fp;
	fp.open(filename, ios::out);
	
	if(fp.is_open()){
		fp << 34 <<endl;
		fp << depth << endl;
		printf(".");
		int i =0;
		bool At, bt;
		for(i=0;i<depth;i++){
			printf(".");
			At = layers[i].A.save(fp);
			bt = layers[i].b.save(fp);
			if(!(At) || !(bt)) {
				goto end;
			}
		}
		printf("Save Successful\n");
		return true;
	}
	
end:
	printf("Save failed\n");
	return false;
}

void nn::singlebackprop(vec_datum datum, double rate)
{
	vec nexterror;
	vec error = (evalnn(datum.coords,0)- datum.value);
	double outputj,outputk,curweight;
	int outdim,indim =0;
	int j,k=0;
	int i = depth;
	for(i=depth;i>0;i--){
		mat curlayout = evalnn_layer(datum.coords,0,i);
		mat prelayout = (evalnn_layer(datum.coords,0,i-1)).t();
		error= error%curlayout%(mat(size(curlayout),fill::ones) - curlayout); 
		mat movemat = rate*(error*prelayout);
		layers[i-1].A = layers[i-1].A - movemat;
		layers[i-1].b = layers[i-1].b - rate*error;
		error = (layers[i-1].A.t()*error);
	}
}

void nn::epochbackprop(vec_data *D, double rate)
{
	int j =0;
	int numdata = D->numdata;
	for(j=0;j<numdata;j++){
		this->singlebackprop(D->data[j],rate);
	}	
}

void nn::trainingbackprop(vec_data *D, double rate, double objerr, int max_gen, bool ratedecay)
{
	int i=0;
	double inputrate = rate;
	double curerr = this->calcerror(D,0);
	while(i<max_gen && curerr > objerr){
		if(ratedecay){inputrate = rate*((max_gen-(double)i)/max_gen);} 
		this->epochbackprop(D,inputrate);
		curerr = this->calcerror(D,0);
		i++;
	}
}

double nn::calcerror(vec_data *D, int func)
{
	int i =0;
	int numdata = D->numdata;
	vec nnvalue;
	double precomputeerror =0;
	double curerr = 0;
	for(i=0;i<numdata;i++){
		nnvalue = this->evalnn(D->data[i].coords,func);
		precomputeerror = norm(nnvalue-(D->data[i].value));
		curerr += (precomputeerror*precomputeerror);
	}
	return curerr/numdata;
}

int nn::outdim() {	return layers[depth-1].A.n_rows; }
int nn::indim() {	return layers[0].A.n_cols; }
int nn::outdim(int i) {	return layers[i].A.n_rows; }
int nn::indim(int i) {	return layers[i].A.n_cols; }
mat nn::getmat(int layernum){	return layers[layernum].A; }
vec nn::getoff(int layernum){	return layers[layernum].b; }

// To add a polarized plane. 
// v is the plane (the weights of the edges from the imput nodes), 
// w is the weights of the new edges to the second hidden layer

bool nn::addnode(int layernum, int nodenum, arma::rowvec v, double offset,arma::vec w){
	/*
	cout << "Adding Node with vec: " << v;
	cout << "offset: " << offset << endl;
	cout << "And selection: " << w << endl;
	*/
	int dim1start = this->indim(layernum);
	int dim2start = this->outdim(layernum);
	int dim3start = this->outdim(layernum+1);
	//mat Abackup = new mat(layers[layernum].A);
	//vec bbackup = new vec(layers[layernum].b);
	if(v.n_cols != this->indim(layernum)){ return false; }
	layers[layernum].A.insert_rows(nodenum,v);
	if(dim1start != this->indim(layernum) || dim2start+1 != this->outdim(layernum))
	{
		return false;
	}
	mat offsetv = {offset};
	layers[layernum].b.insert_rows(nodenum,offsetv);
	layers[layernum+1].A.insert_cols(nodenum,w);
	if(dim2start+1 != this->indim(layernum+1) || dim3start != this->outdim(layernum+1))
	{
		return false;
	}
	return true;
}

#ifndef RUNAVGWID
#define RUNAVGWID 20  //variable to easily change the width of the running average.
#endif
double nn::erravgslope(vec_data *data, int func)
{
	static int calltimes = 0;
	if(calltimes<RUNAVGWID){calltimes++;}

	static double lasterror = 0;
	static double slopes[RUNAVGWID];
	double curerr = this->calcerror(data,func);
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

int * minvals(vec v, int numvecs){
	double x=1000000;  // should be larger than any element of any possible vector

	//THIS NEEDS TO BE IMPROVED TO work with numvecs >2
	numvecs = 2;
	int *ret = new int[2];
	ret[0] = -1;
	ret[1] = -1;
	int l = v.n_rows;
	if(l==1){
		return ret;
	} else {
		ret[1] = 0;
		ret[0] = 1;
	}
	int i = 0;
	for(i=0;i<l;i++){
		if(v(i)<0){ v(i)=-v(i);}
		if(v(i)<x){
			x = v(i);
			ret[1] = ret[0];
			ret[0] = i;
		}
	}
	if(ret[1]=-1){
		x=1000000;
		for(i=0;i<l;i++){
			if(v(i)<x && i != ret[0]){
				x = v(i);
				ret[1] = i;
			}
		}
	}
	if(ret[0]>ret[1]){
		return ret;
	} else {
		l = ret[0];
		ret[0] = ret[1];
		ret[1] = l;
		return ret;
	}
	
}


vec computeDistToHyperplanes(mat A,vec b, vec v){
	int i =0;
	int n = b.n_rows;
	vec retvec = zeros<vec>(n);
	rowvec curvec;
	rowvec normcurvec;
	for(i=0;i<n;i++){
		curvec = A.row(i)/norm(A.row(i));
		normcurvec = b(i)*curvec/norm(A.row(i));
		retvec(i) = dot((v.t()-normcurvec),curvec);
	}
	return retvec;
}

vec nn::hyperplaneIntersection(int i, int j)
{
	mat B = zeros<mat>(2,2);
	vec c = zeros<vec>(2);
	B.row(0) = layers[0].A.row(i); c(0) = layers[0].b(i); 
	B.row(1) = layers[0].A.row(j); c(1) = layers[0].b(j); 
	return solve(B,c);
}

double nn::hyperplaneIntersectionDistance(int i, int j, vec v)
{
	return norm(v-this->hyperplaneIntersection(i,j));
}


indexDistance nn::computeDistToHyperplanesIntersections(vec v)
{
	int i,j =0;
	int n = this->outdim(0);
	double dist;
	double smallestDist = 100000;
	int *index = new int[2];
	index[0] = -1;
	index[1] = -1;
	for(i=0;i<n;i++){
		for(j=0;j<i;j++){
			dist = this->hyperplaneIntersectionDistance(i, j, v);
			if(dist < smallestDist){
				smallestDist = dist;
				index[0] = i; index[1] = j;
			}
		}
	}
	indexDistance ret;
	ret.index = index;
	ret.dist = smallestDist;
	return ret;
}


newHPInfo nn::locateNewHP(vec_data *data, int func, double errorThreshold)
{
	int i,j=0;
	int numnodes = this->outdim(0);
	int inputdim = this->indim();
	int numdata = data->numdata;
	struct errtracker **errArray = new errtracker*[numnodes];
	for(i=0;i<numnodes;i++){ 
		errArray[i] = new errtracker[i];
		for(j=0;j<i;j++){
			errArray[i][j].numerr = 0;
			errArray[i][j].totvecerr = zeros<vec>(inputdim);
			errArray[i][j].intersection = this->hyperplaneIntersection(i,j);
			int *indexes = new int[2];
			indexes[0]=i;
			indexes[1]=j;
			errArray[i][j].arrayindex = indexes;
		}
	}
	int numerrors =0;
	vec intersection;
	for(i=0;i<numdata;i++){
		double err = norm((data->data[i].value-this->evalnn(data->data[i].coords,func)));
		if(err > errorThreshold){
			indexDistance ID = computeDistToHyperplanesIntersections(data->data[i].coords);
			intersection = errArray[ID.index[0]][ID.index[1]].intersection;
			errArray[ID.index[0]][ID.index[1]].totvecerr += (data->data[i].coords);
			errArray[ID.index[0]][ID.index[1]].numerr++;
			numerrors++;
		}
	}
	//printf("Number of total errors: %d\n", numerrors);
	int errtrac1,errtrac2=-1;
	int comparison = -1;
	for(i=0;i<numnodes;i++){
		for(j=0;j<i;j++){
			//printf("Number of local errors %d near %d,%d\n", errArray[errtrac1][errtrac2].numerr,i,j);
			if(errArray[i][j].numerr > comparison){
				comparison = errArray[i][j].numerr;
				errtrac1=i; errtrac2=j;
			}
		}
	}
	/*
	printf("------\n");
	printf("Selected %d near %d,%d\n", errArray[errtrac1][errtrac2].numerr,i,j);
	printf("------\n");
	*/
	vec errVec = errArray[errtrac1][errtrac2].totvecerr/(errArray[errtrac1][errtrac2].numerr);
	errVec = (errVec+(this->hyperplaneIntersection(errtrac1,errtrac2)))/2;
	
	rowvec normVec = zeros<rowvec>(inputdim);
	rowvec normal = layers[0].A.row(errtrac1);
	if(layers[1].A(0,errtrac1)<0){normal = -normal;}
	normVec += (normal)/(norm(normal));
	normal = layers[0].A.row(errtrac2);

	if(layers[1].A(0,errtrac2)<0){normal = -normal;}
	normVec += (normal)/(norm(normal));

	normVec = normVec/(norm(normVec));

	newHPInfo ret;
	ret.offset = dot(normVec.t(),errVec);
	ret.normVec = normVec;
	ret.numerr = errArray[errtrac1][errtrac2].numerr;

	for(i=0;i<numnodes;i++){
		delete[] errArray[i];
	}
	delete[] errArray;

	return ret;
}
//Currently this is only for 2 dimensional inputs.

vec nn::calculateSelectionVector()
{
	int i=0;
	int n= this->outdim(0);
	vec offset = zeros<vec>(n);
	for(i=0;i<n;i++){
		if(layers[1].A(0,i)< 0){
			offset(i) = -layers[1].A(0,i);
		} else {
			offset(i) = layers[1].A(0,i);
		}
	}
	vec ret = {1};
	ret(0) = mean(mean(offset));
	return ret;
}

#ifndef ERRORTHRESHOLD
#define ERRORTHRESHOLD 0.001
#endif

void nn::smartaddnode1(vec_data *data,int func)
{
	/*
	printf("------------------------------------------------------------------------\n");
	printf("NN before insert\n");
	this->print();
	printf("------------------------------------------------------------------------\n");
	*/
	int i,j=0;
	int numnodes = this->outdim(0);
	int inputdim = this->indim();
	int numdata = data->numdata;
	if(numnodes==1){
		rowvec v = randu<rowvec>(inputdim);
		rowvec k = layers[0].A.row(0);
		double offset = layers[0].b(0);
		offset = offset*norm(k);
		if(offset < 0){ offset = -offset;}
		k = k/norm(k);
		v = v - dot(v,k)*k;
		v = v/norm(v);
		vec w = {1};
		this->addnode(0,0,v,offset,w);
	}

	if(numnodes>1){
		newHPInfo info = this->locateNewHP(data,func,ERRORTHRESHOLD);
		vec w = this->calculateSelectionVector();
		if(info.numerr > 5) {
			this->addnode(0,0,4*info.normVec,4*info.offset,w);
			layers[1].b(0) += w(0)/2;
		}
	}
	/*
	printf("------------------------------------------------------------------------\n");
	printf("NN after insert\n");
	this->print();
	printf("------------------------------------------------------------------------\n");
	*/
}

#ifndef SLOPETHRESHOLD
#define SLOPETHRESHOLD 0.0001
#endif

void nn::adaptivebackprop1(vec_data *D, double rate, double objerr, int max_gen, int max_nodes, bool ratedecay)
{
	int i=0;
	int lastHPChange = 0;
	double inputrate = rate;
	double curerr = this->calcerror(D,0);
	double curerrorslope = 0;
	int curnodes = this->outdim(0);
	while(i<max_gen && curerr > objerr){
		if(ratedecay){inputrate = rate*((max_gen-(double)i)/max_gen);} 
		this->epochbackprop(D,inputrate);
		curerr = this->calcerror(D,0);
		curerrorslope = this->erravgslope(D,0);
		if(-curerrorslope < SLOPETHRESHOLD && curerrorslope < 0 && curnodes < max_nodes && i-lastHPChange>50){
			this->smartaddnode1(D,1);
			lastHPChange = i;
			curnodes = this->outdim(0);	
		}
		i++;
	}
}

double ** nn::erroradaptivebackprop1(vec_data *D, double rate, double objerr, int max_gen, int max_nodes, bool ratedecay)
{	
	double **returnerror = new double*[2];
	returnerror[0] = new double[max_gen];
	returnerror[1] = new double[max_gen];
	int i=0;
	int lastHPChange = 0;
	double inputrate = rate;
	double curerr = this->calcerror(D,0);
	double curerrorslope = 0;
	int curnodes = this->outdim(0);
	while(i<max_gen && curerr > objerr){
		if(ratedecay){inputrate = rate*((max_gen-(double)i)/max_gen);} 
		this->epochbackprop(D,inputrate);
		curerr = this->calcerror(D,0);
		returnerror[0][i] = curerr;
		returnerror[1][i] = this->calcerror(D,1);
		curerrorslope = this->erravgslope(D,0);
		if(-curerrorslope < SLOPETHRESHOLD && curerrorslope < 0 && curnodes < max_nodes && i-lastHPChange>50){
			this->smartaddnode1(D,1);
			lastHPChange = i;
			curnodes = this->outdim(0);	
		}
		i++;
	}
	return returnerror;
}

void nn::animatedadaptivebackprop1(vec_data *D, double rate, double objerr, int max_gen, int max_nodes, bool ratedecay)
{
	int i=0;
	double inputrate = rate;
	double curerr = this->calcerror(D,0);
	double curerrorslope = 0;
	int curnodes = this->outdim(0);
	char header[100];
	int lastHPChange = 0;
	while(i<max_gen && curerr > objerr){
		sprintf(header, "imgfiles/sig/train%05d.ppm",i);
		write_nn_to_img(this,header,500,500,0);
		write_data_to_img(D,header);
		sprintf(header, "imgfiles/hea/train%05d.ppm",i);
		write_nn_to_img(this,header,500,500,1);
		write_data_to_img(D,header);
		if(ratedecay){inputrate = rate*((max_gen-(double)i)/max_gen);} 
		this->epochbackprop(D,inputrate);
		curerr = this->calcerror(D,0);
		curerrorslope = this->erravgslope(D,0);
		printf("Error slope: %f Num Nodes: %d Theshold: %f Current gen:%d\n", curerrorslope, curnodes, -SLOPETHRESHOLD,i);
		if(-curerrorslope < SLOPETHRESHOLD && curerrorslope < 0 && curnodes < max_nodes && i-lastHPChange>50){
			printf("Inserting hyperplane. Error slope is %f \n",curerrorslope);
			this->smartaddnode1(D,1);
			lastHPChange = i;
			curnodes = this->outdim(0);	
		}
		i++;
	}
}