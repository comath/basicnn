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

void delvec_data(vec_data *D);

typedef struct errtracker {
	int numerr;
	arma::vec totvecerr;
	arma::vec intersection;
	int * arrayindex;
} errtracker;

typedef struct nnlayer {
	arma::mat A;
	arma::vec b;
} nnlayer;

typedef struct newHPInfo {
	double offset;
	arma::rowvec normVec;
	int numerr;
} newHPInfo;

typedef struct indexDistance {
	int *index;
	double dist;
} indexDistance;

class nn {
private:
	int depth;
	nnlayer *layers;
	void initlayerofnn(int i,int indim, int outdim);
	void smartaddnode1(vec_data *data, int func);
	newHPInfo locateNewHP(vec_data *data, int func, double errorThreshold);
	arma::vec calculateSelectionVector();
	indexDistance * computeDistToHyperplanesIntersections(arma::vec v);
	arma::vec hyperplaneIntersection(int i,int j);
	double hyperplaneIntersectionDistance(int i, int j, arma::vec v);
public:
	nn(int inputdim, int width1, int width2, int outdim);
	nn(int inputdim, int width1, int outdim);
	nn(const char *filename);
	bool save(const char *filename);
	void print();
	~nn();
	void randfillnn(double weight);
	arma::mat getmat(int layer);
	arma::vec getoff(int layer);
	arma::vec evalnn( arma::vec input, int func);
	arma::vec evalnn_layer( arma::vec input, int func, int layernum);
	void singlebackprop(vec_datum datum, double rate);
	void epochbackprop(vec_data *data, double rate);
	void trainingbackprop(vec_data *data, double rate, double objerr, int max_gen, bool ratedecay);
	double calcerror(vec_data *data, int func);
	int outdim();
	int indim();
	int outdim(int i); // For the in and out of a layer
	int indim(int i);
	/*
	To add a node. 
	v is the plane (the weights of the edges from the input nodes), 
	w is the weights of the new edges to the second hidden layer
	*/
	double erravgslope(vec_data *data, int func);
	bool addnode(int layernum, int nodenum, arma::rowvec v, double off,arma::vec w); 
	void adaptivebackprop1(vec_data *D, double rate, double objerr, int max_gen, int max_nodes, bool ratedecay);
	void animatedadaptivebackprop1(vec_data *D, double rate, double objerr, int max_gen, int max_nodes, bool ratedecay);
	double ** erroradaptivebackprop1(vec_data *D, double rate, double objerr, int max_gen, int max_nodes, bool ratedecay);
};



#endif