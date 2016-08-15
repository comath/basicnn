#include <iostream>
#include <armadillo>
#include <random>
#include <math.h>
#include <iomanip>
#include <pthread.h>
#include <sys/stat.h>
#include "ann.h"
#include "pgmreader.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace arma;
using namespace std;

struct vec_data *get_vec_data_ppm(pm_img *img, int numdata)
{
	vec_data *thisdata = new vec_data;
	thisdata->data = new vec_datum[numdata];
	thisdata->numdata = numdata;
	int height = img->getheight();
	int width = img->getwidth();
	int i,j = 0;
	int x;
	int y;

	std::default_random_engine generator;
	std::uniform_int_distribution<int> dist_height(0,height-1);
	std::uniform_int_distribution<int> dist_width(0,width-1);

	while(i<numdata){
		x = dist_width(generator);
		y = dist_height(generator);
		//printf("x:%d,y:%d     i:%d,j:%d \n", x,y,i,j);
		if(j > numdata)
		double r,g,b;
		thisdata->data[i].coords = vec(2,fill::zeros);
		thisdata->data[i].coords(1) = ((y-(double)height/2)/height)*10;
		thisdata->data[i].coords(0) = ((x-(double)width/2)/width)*10;
		thisdata->data[i].value = vec(1,fill::zeros);
		if(img->gettype() == 6) {
			double r = ((double)((unsigned char)img->r(x,y))/255);
			double g = ((double)((unsigned char)img->g(x,y))/255);
			double b = ((double)((unsigned char)img->b(x,y))/255);
			
			if(true){
				thisdata->data[i].value = vec(3,fill::zeros);
				thisdata->data[i].value(0) = r;
				thisdata->data[i].value(1) = g;
				thisdata->data[i].value(2) = b;
				i++;
			}
		} else {			
			double r = ((double)((unsigned char)img->r(x,y))/255);
			if(true){
				thisdata->data[i].value = vec(1,fill::zeros);
				thisdata->data[i].value(0) = r;
				i++;
			}
		}
		//printf("value in data: %f, actual value %d\n",r_ thisdata[i]alue, (unsigned char)pixarr[x][y]);
		j++;
	}
	return thisdata;
}



void write_nn_to_img(nn *thisnn, const char filename[], int height, int width, int func)
{
	int i,j =0;
	vec input = vec(2,fill::zeros);
	
	vec value;
	if(thisnn->outdim() == 1){
		pm_img *img = new pm_img(height,width,255,5);
		for(i=0;i<height;i++){
			for(j=0;j<width;j++){
				input(0) = ((i-(double)height/2)/height)*10;
				input(1) = ((j-(double)width/2)/width)*10;
				value = thisnn->evalnn(input, func);
				unsigned char val = (unsigned char)(floor((value(0)*255)));
				img->wr(i,j,val);
			}
		}
		img->pm_write(filename);
		delete img;
	} else if(thisnn->outdim() ==3) {
		pm_img *img = new pm_img(height,width,255,6);
		for(i=0;i<height;i++){
			for(j=0;j<width;j++){
				input(0) = ((i-(double)height/2)/height)*10;
				input(1) = ((j-(double)width/2)/width)*10;
				value = thisnn->evalnn(input, func);
				img->wr(i,j, (unsigned char)(floor((value(0))*255)));
				img->wg(i,j, (unsigned char)(floor((value(1))*255)));
				img->wb(i,j, (unsigned char)(floor((value(2))*255)));
			}
		}
		img->pm_write(filename);
		delete img;
	} else if(thisnn->outdim() ==2) {
		pm_img *img = new pm_img(height,width,255,6);
		for(i=0;i<height;i++){
			for(j=0;j<width;j++){
				input(0) = ((i-(double)height/2)/height)*10;
				input(1) = ((j-(double)width/2)/width)*10;
				value = thisnn->evalnn(input, 0);
				img->wr(i,j, (unsigned char)(floor((value(0))*255)));
				img->wg(i,j, (unsigned char)(floor((value(1))*255)));
				img->wb(i,j, (unsigned char)(0));
			}
		}
		img->pm_write(filename);
		delete img;
	} else {
		printf("Not of a correct dimension\n");
	}
}

#ifndef NUMGEN
#define NUMGEN 500
#endif
#ifndef NUMNEUNRT
#define NUMNEUNRT 500
#endif
#ifndef MAXNODES
#define MAXNODES 20
#endif
#define MAXTHREADS 2 // should be a divisor of NUMNEUNET

struct GED_args {
	int finaldim;
	int tid;
	int begin;
	int end;
	vec_data *D;
} GED_args;

void *geterrordata_thread(void *thread_args)
{	
	struct GED_args *myargs;
	myargs = (struct GED_args *) thread_args;
	vec_data *D;
	D = myargs->D;
	int begin = myargs->begin;
	int end = myargs->end;
	int tid = myargs->tid;
	int finaldim = myargs->finaldim;
	printf("Opened thread %d covering networks %d to %d \n",tid,begin,end-1);

	int i,j =0;
	char header[100];

	double saverages[1000];
	double haverages[1000];
	printf("Creating the heavyside epoch error file in thread tid: %d.",tid);
	sprintf(header, "imgfiles/heaerrordataepoch%02d.txt",tid);
	ofstream heaepocherrdat;
	heaepocherrdat.open(header);
	if(heaepocherrdat.is_open()){printf("   Done\n");} else {printf("   Failed\n");}

	printf("Creating the sigmoderrdat epoch error file in thread tid: %d.",tid);
	sprintf(header, "imgfiles/sigerrordataepoch%02d.txt",tid);
	ofstream sigepocherrdat;
	sigepocherrdat.open(header);
	if(sigepocherrdat.is_open()){printf("   Done\n");} else {printf("   Failed\n");}

	int numnodes =1;
	printf("Running on nodes %d\n", MAXNODES);
	for(numnodes=1;numnodes<MAXNODES+1;numnodes++){
		printf("Creating Neural Network with dim %d,%d,%d\n", 2,numnodes,finaldim);
		nn *nurnet = new nn(2,numnodes,finaldim);

		for(j=0;j<NUMGEN;j++){haverages[j]=0;saverages[j]=0;}

		for(i=begin;i<end;i++){
			nurnet->randfillnn(0.5);
			sprintf(header, "imgfiles/numnodes%03d/sigmodpre/testsigmodpre%05d.pgm",numnodes,i);
			write_nn_to_img(nurnet,header,500,500,0);
			sprintf(header, "imgfiles/numnodes%03d/heavypre/testheavypre%05d.pgm",numnodes,i);
			write_nn_to_img(nurnet,header,500,500,1);
			sprintf(header, "imgfiles/numnodes%03d/netpre/net%03dnodespre.nn",numnodes,i);
			nurnet->save(header);


			for(j=0;j<NUMGEN;j++){
				nurnet->epochbackprop(D, 0.05);
				//printf("%f\n", nurnet->calcerror(D,0));
				saverages[j] = saverages[j] + nurnet->calcerror(D,0);
				haverages[j] = haverages[j] + nurnet->calcerror(D,1);
			}

			sprintf(header, "imgfiles/numnodes%03d/sigmodpost/testsigmodpost%05d.pgm",numnodes,i);
			write_nn_to_img(nurnet,header,500,500,0);
			sprintf(header, "imgfiles/numnodes%03d/heavypost/testheavypost%05d.pgm",numnodes,i);
			write_nn_to_img(nurnet,header,500,500,1);
			sprintf(header, "imgfiles/numnodes%03d/netpost/net%03dnodespost.nn",numnodes,i);
			nurnet->save(header);
		}

		for(j=0;j<NUMGEN;j++){
			saverages[j] = saverages[j]/NUMNEUNRT;
			haverages[j] = haverages[j]/NUMNEUNRT;
		}
		
		heaepocherrdat << numnodes << ',' << 'h' << ',';
		for(j=0;j<NUMGEN;j++){
			heaepocherrdat << std::fixed << std::setprecision(8) << haverages[j] << ',';
		}
		heaepocherrdat << endl;
		sigepocherrdat << numnodes << ',' << 's' << ',';
		for(j=0;j<NUMGEN;j++){
			sigepocherrdat << std::fixed << std::setprecision(8) << saverages[j] << ',';
		}
		sigepocherrdat << endl;
		delete nurnet;
	}
	printf("Exiting thread %d\n", tid);
	pthread_exit(NULL);
}

void geterrordata(int argc, char *argv[])
{
	printf("Opening %s\n",argv[2]);
	pm_img *img = new pm_img(argv[2]);
	int finaldim;
	if(img->gettype() == 6){ finaldim = 3; } else { finaldim = 1; }
	
	printf("Collecting data");
	vec_data *D = get_vec_data_ppm(img, 3000);
	printf("  done\n");
	char header[100];
	mkdir("imgfiles",0777);
	int numnodes = 1;
	for(numnodes=1;numnodes<MAXNODES+1;numnodes++){
		sprintf(header, "imgfiles/numnodes%03d",numnodes);
		mkdir(header,0777);
		sprintf(header, "imgfiles/numnodes%03d/sigmodpre",numnodes);
		mkdir(header,0777);
		sprintf(header, "imgfiles/numnodes%03d/sigmodpost",numnodes);
		mkdir(header,0777);
		sprintf(header, "imgfiles/numnodes%03d/heavypre",numnodes);
		mkdir(header,0777);
		sprintf(header, "imgfiles/numnodes%03d/heavypost",numnodes);
		mkdir(header,0777);
		sprintf(header, "imgfiles/numnodes%03d/netpre",numnodes);
		mkdir(header,0777);
		sprintf(header, "imgfiles/numnodes%03d/netpost",numnodes);
		mkdir(header,0777);
	}

	pthread_t threads[MAXTHREADS];
	struct GED_args *thread_args = new struct GED_args[MAXTHREADS];
	int rc;
	int i;


	// Initialize and set thread joinable
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(i=0;i<MAXTHREADS;i++){
		thread_args[i].finaldim = finaldim;
		thread_args[i].tid = i;
		thread_args[i].D = D;
		thread_args[i].begin = i*(NUMNEUNRT/MAXTHREADS);
		thread_args[i].end = (i+1)*(NUMNEUNRT/MAXTHREADS);
		rc = pthread_create(&threads[i], NULL, geterrordata_thread, (void *)&thread_args[i]);
		if (rc){
			cout << "Error:unable to create thread," << rc << endl;
			exit(-1);
		}
	}

	for( i=0; i < MAXTHREADS; i++ ){
		rc = pthread_join(threads[i], &status);
		if (rc){
			cout << "Error:unable to join," << rc << endl;
			exit(-1);
     	}
		cout << "Main: completed thread id :" << i ;
		cout << "  exiting with status :" << status << endl;
	}
	delvec_data(D);
	delete img;
}

void animatetraining(int argc, char *argv[])
{
	int generations = 1000;
	int numdata = 1000;
	int numnodes = 10;
	double rate_start = 0.05;

	int i =0;
	printf("Opening %s\n",argv[2]);
	pm_img *img = new pm_img(argv[2]);
	int finaldim;
	if(img->gettype() == 6){ finaldim = 3; } else { finaldim = 1; }
	printf("Creating Neural Network with dim %d,%d,%d\n", 2,10,finaldim);
	nn *nurnet = new nn(2,numnodes,finaldim);
	nurnet->randfillnn(0.5);
	vec_data *D = get_vec_data_ppm(img, numdata);
	char header[100];
	sprintf(header, "imgfiles/train%05d.ppm",0);
	write_nn_to_img(nurnet,header,500,500,0);
	double rate;

	for(i=0;i< generations;i++){
		printf("On generation %d of %d \n",i+1 ,generations);
		sprintf(header, "imgfiles/train%05d.ppm",i+1);
		rate = rate_start*((generations-(double)i)/generations);
		nurnet->epochbackprop(D,rate);
		write_nn_to_img(nurnet,header,500,500,0);
	}
	bool test = nurnet->save("test1.nn");
	delvec_data(D);
	delete nurnet;
	delete img;
}


int main(int argc, char *argv[])
{
	int i,j,k =0;
	if(argc == 3 && argv[1][0] == '-' && argv[1][1] == 'i'){
		animatetraining(argc, argv);
		return 0;
	}
	if(argc == 3 && argv[1][0] == '-' && argv[1][1] == 't'){
		geterrordata(argc, argv);
		return 0;
	}
	if(argc == 3 && argv[1][0] == '-' && argv[1][1] == 'n'){
		nn *nurnet = new nn(argv[2]);
		write_nn_to_img(nurnet,"mtest.ppm",500,500,1);
		rowvec v = {1,0};
		double offset = -3;
		vec w;
		if(nurnet->outdim()==3){w = {-100,-100,-100};}
		if(nurnet->outdim()==1){w = {-100};}
		nurnet->addnode(0,0,v,offset,w);
		write_nn_to_img(nurnet,"mtest2.ppm",500,500,1);
	}
	if( argc == 0){
		nn *nurnet = new nn(2,5,3);
		nurnet->randfillnn(0.10);
		const char *filename = "rando.pgm";
		write_nn_to_img(nurnet,filename,1000,1000,1);
		return 0;
	}
}