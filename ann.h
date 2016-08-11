#ifndef _nn_h
#define _nn_h
#include <armadillo>

typedef struct vec_datum {
	arma::vec coords;
	arma::vec value;
} vec_datum;

typedef struct vec_data {
	int numdata;
	vec_datum *data;
} vec_data;

typedef struct nnlayer {
	arma::mat A;
	arma::vec b;
} nnlayer;

class nn {
	private:
	int depth;
	nnlayer *layers;
	void initlayerofnn(int i,int indim, int outdim);
public:
	nn(int inputdim, int width1, int width2, int outdim);
	nn(int inputdim, int width1, int outdim);
	nn(const char *filename);
	bool save(const char *filename);
	~nn();
	void randfillnn(double weight);
	arma::mat getmat(int layer);
	arma::vec getoff(int layer);
	arma::vec evalnn( arma::vec input, int func);
	arma::vec evalnn_layer( arma::vec input, int func, int layernum);
	void backprop(arma::vec input, arma::vec expected, double rate);
	int outdim();
	int indim();
	int outdim(int i); // For the in and out of a layer
	int indim(int i);
	/*
	To add a polarized plane. 
	v is the plane (the weights of the edges from the imput nodes), 
	w is the weights of the new edges to the second hidden layer
	*/
	bool addnode(int layernum, int nodenum, arma::rowvec v, double off,arma::vec w); 
};



#endif