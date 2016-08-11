#include <iostream>
#include <armadillo>
#include "ann.h"

using namespace std;
using namespace arma;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))



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
	if(fp.is_open()) fp.close();
	printf("Save failed\n");
	return false;
}

void nn::backprop(vec input, vec expected, double rate)
{
	vec nexterror;
	vec error = (evalnn(input,0)- expected);
	double outputj,outputk,curweight;
	int outdim,indim =0;
	int j,k=0;
	int i = depth;
	for(i=depth;i>0;i--){
		mat curlayout = evalnn_layer(input,0,i);
		mat prelayout = (evalnn_layer(input,0,i-1)).t();
		error= error%curlayout%(mat(size(curlayout),fill::ones) - curlayout); 
		mat movemat = rate*(error*prelayout);
		layers[i-1].A = layers[i-1].A - movemat;
		layers[i-1].b = layers[i-1].b - rate*error;
		error = (layers[i-1].A.t()*error);
	}
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