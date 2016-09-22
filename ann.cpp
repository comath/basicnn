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
				}
			}
		}
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

bool nn::appendToHistory(fstream *fp)
{
	if(fp->is_open()){
		*fp << depth << endl;
		printf(".");
		int i =0;
		bool At, bt;
		for(i=0;i<depth;i++){
			printf(".");
			At = layers[i].A.save(*fp);
			bt = layers[i].b.save(*fp);
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

double ** nn::trainingbackprop(vec_data *D, double rate, double objerr, int max_gen, bool ratedecay)
{
	double **returnerror = new double*[2];
	returnerror[0] = new double[max_gen];
	returnerror[1] = new double[max_gen];
	int i=0;
	double inputrate = rate;
	double curerr = this->calcerror(D,0);
	while(i<max_gen && curerr > objerr){
		if(ratedecay){inputrate = rate*((max_gen-(double)i)/max_gen);} 
		this->epochbackprop(D,inputrate);
		curerr = this->calcerror(D,0);
		returnerror[0][i] = curerr;
		returnerror[1][i] = this->calcerror(D,1);
		i++;
	}
	return returnerror;
}

double nn::calcerror(vec_datum datum, int func){
	double precomputeerror = norm((this->evalnn(datum.coords,func))-(datum.value));
	return (precomputeerror*precomputeerror);
}

double nn::calcerror(vec_data *D, int func)
{
	int i =0;
	int numdata = D->numdata;
	double curerr = 0;
	for(i=0;i<numdata;i++){
		curerr += this->calcerror(D->data[i],func);
	}
	return curerr/numdata;
}


int nn::outdim() {	return layers[depth-1].A.n_rows; }
int nn::indim() {	return layers[0].A.n_cols; }
int nn::outdim(int i) {	return layers[i].A.n_rows; }
int nn::indim(int i) {	return layers[i].A.n_cols; }
mat nn::getmat(int layernum){	return layers[layernum].A; }
vec nn::getoff(int layernum){	return layers[layernum].b; }
int nn::getDepth(){return depth;}

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
	if(v.n_cols != (unsigned int)this->indim(layernum)){ return false; }
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

void nn::addnode(arma::rowvec v, double offset ,nnlayer selectionLayer){
	layers[0].A.insert_rows(0,v);
	mat offsetv = {offset};
	layers[0].b.insert_rows(0,offsetv);
	layers[1] = selectionLayer;
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

