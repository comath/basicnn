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
	bool silence;
	int depth;
	nnlayer *layers;
	void initlayerofnn(int i,int indim, int outdim);
	
	newHPInfo locateNewHP(vec_data *data, int func, double errorThreshold);
	arma::vec calculateSelectionVector();
	indexDistance * computeDistToHyperplanesIntersections(arma::vec v);
	arma::vec hyperplaneIntersection(int i,int j);
	double hyperplaneIntersectionDistance(int i, int j, arma::vec v);
public:
	nn(int inputdim, int width1, int width2, int outdim);
	nn(int inputdim, int width1, int outdim);
	nn(const char *filename);
	nn(int d, nnlayer *l);
	bool save(const char *filename);
	bool appendToHistory(std::fstream *fp);
	void print();
	~nn();
	void randfillnn(double weight);
	arma::mat getmat(int layer);
	arma::vec getoff(int layer);
	arma::vec evalnn( arma::vec input, int func);
	arma::vec evalnn_layer( arma::vec input, int func, int layernum);
	double singlebackprop(vec_datum datum, double rate);
	double epochbackprop(vec_data *data, double rate);
	double ** trainingbackprop(vec_data *data, double rate, double objerr, int max_gen, bool ratedecay);
	double calcerror(vec_data *data, int func);
	double calcerror(vec_datum datum, int func);
	int outdim();
	int indim();
	int outdim(int i); // For the in and out of a layer
	int indim(int i);
	int getDepth();
	/*
	To add a node. 
	v is the plane (the weights of the edges from the input nodes), 
	w is the weights of the new edges to the second hidden layer
	*/
	bool addnode(int layernum, int nodenum, arma::rowvec v, double off,arma::vec w);
	void addnode(arma::rowvec v, double offset, nnlayer selectionLayer);

	void smartaddnode1(vec_data *data, int func);
	
	double erravgslope(vec_data *data, int func);

	// Preliminary
	arma::vec getHPLayerErrorData(vec_datum datum);
};



#endif