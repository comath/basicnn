#include <iostream>
#include <armadillo>
#include <random>
#include <math.h>
#include <iomanip>
#include <pthread.h>
#include <sys/stat.h>
#include <ctime>
#include "ann.h"
#include "pgmreader.h"
#include "annpgm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace arma;
using namespace std;

#ifndef NUMGEN
#define NUMGEN 120
#endif
#ifndef NUMNEUNRT
#define NUMNEUNRT 10
#endif
#ifndef MAXNODES
#define MAXNODES 25
#endif
#ifndef MAXDATA
#define MAXDATA 2
#endif
#ifndef STARTNODES
#define STARTNODES 3
#endif
#define MAXTHREADS 5 // should be a divisor of NUMNEUNET

struct GED_args {
	int finaldim;
	int tid;
	int begin;
	int end;
	pm_img *img;
	bool adaptive;
} GED_args;

void *geterrordata_thread(void *thread_args)
{	
	struct GED_args *myargs;
	myargs = (struct GED_args *) thread_args;
	pm_img *img = myargs->img;
	int begin = myargs->begin;
	int end = myargs->end;
	int tid = myargs->tid;
	int finaldim = myargs->finaldim;
	bool adaptive = myargs->adaptive;
	printf("Opened thread %d covering networks %d to %d \n",tid,begin,end-1);

	int i,j =0;
	char header[100];

	double saverages[NUMGEN];
	double haverages[NUMGEN];
	printf("Creating the heavyside epoch error file in thread tid: %d.",tid);
	if(adaptive){
		sprintf(header, "imgfiles/heaerrordataepoch%02dAdaptive.txt",tid);
	} else {
		sprintf(header, "imgfiles/heaerrordataepoch%02dBackprop.txt",tid);
	}
	ofstream heaepocherrdat;
	heaepocherrdat.open(header);
	if(heaepocherrdat.is_open()){printf("   Done\n");} else {printf("   Failed\n");}

	printf("Creating the sigmoderrdat epoch error file in thread tid: %d.",tid);
	if(adaptive){
		sprintf(header, "imgfiles/sigerrordataepoch%02dAdaptive.txt",tid);
	} else {
		sprintf(header, "imgfiles/sigerrordataepoch%02dBackprop.txt",tid);
	}
	ofstream sigepocherrdat;
	sigepocherrdat.open(header);
	if(sigepocherrdat.is_open()){printf("   Done\n");} else {printf("   Failed\n");}

	int numnodes =STARTNODES;
	printf("Running on nodes %d\n", MAXNODES);
	int numDataSets=0;
	int numTotalNets=0;
	for(numnodes=STARTNODES;numnodes<MAXNODES+1;numnodes++){
		for(j=0;j<NUMGEN;j++){haverages[j]=0;saverages[j]=0;}
		for(numDataSets=0;numDataSets<MAXDATA;numDataSets++){
			printf("Collecting data \n");
			vec_data *D = get_vec_data_ppm(img, 3000);
				for(i=begin;i<end;i++){
					nn *nurnet;
					if(adaptive){
						nurnet = new nn(2,STARTNODES,finaldim);
					} else {
						nurnet = new nn(2,numnodes,finaldim);
					}
					nurnet->randfillnn(0.5);
					if(adaptive){
						sprintf(header, "imgfiles/numnodes%03d/netpre/net%03ddata%03dnodespreAdaptive.nn",numnodes,i,numDataSets);
					} else {
						sprintf(header, "imgfiles/numnodes%03d/netpre/net%03ddata%03dnodespreBackprop.nn",numnodes,i,numDataSets);
					}
					nurnet->save(header);
				
					if(adaptive){
						double ** errors = nurnet->erroradaptivebackprop1(D, 0.05, -1, NUMGEN, numnodes, false);
						for(j=0;j<NUMGEN;j++){
							saverages[j] = saverages[j] + errors[0][j];
							haverages[j] = haverages[j] + errors[1][j];
						}
					} else {
						for(j=0;j<NUMGEN;j++){
							nurnet->epochbackprop(D,0.05);
							saverages[j] = saverages[j] + nurnet->calcerror(D,0);
							haverages[j] = haverages[j] + nurnet->calcerror(D,1);
						}
					}
					if(adaptive){
						sprintf(header, "imgfiles/numnodes%03d/netpost/net%03ddata%03dnodespostAdaptive.nn",numnodes,i,numDataSets);
					} else {
						sprintf(header, "imgfiles/numnodes%03d/netpost/net%03ddata%03dnodespostBackprop.nn",numnodes,i,numDataSets);
					}
					nurnet->save(header);
					delete nurnet;
					numTotalNets++;
				}
			delvec_data(D);
		}
		for(j=0;j<NUMGEN;j++){
			saverages[j] = (saverages[j])/(numTotalNets);
			haverages[j] = (haverages[j])/(numTotalNets);
		}
		numTotalNets = 0;
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
		
	char header[100];
	mkdir("imgfiles",0777);
	int numnodes = STARTNODES;
	for(numnodes=STARTNODES;numnodes<MAXNODES+1;numnodes++){
		sprintf(header, "imgfiles/numnodes%03d",numnodes);
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
		thread_args[i].img = img;
		thread_args[i].begin = i*(NUMNEUNRT/MAXTHREADS);
		thread_args[i].end = (i+1)*(NUMNEUNRT/MAXTHREADS);
		thread_args[i].adaptive = true;
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

	for(i=0;i<MAXTHREADS;i++){
		thread_args[i].finaldim = finaldim;
		thread_args[i].tid = i;
		thread_args[i].img = img;
		thread_args[i].begin = i*(NUMNEUNRT/MAXTHREADS);
		thread_args[i].end = (i+1)*(NUMNEUNRT/MAXTHREADS);
		thread_args[i].adaptive = false;
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
	mkdir("imgfiles",0777);
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
	nurnet->save("test1.nn");
	delvec_data(D);
	delete nurnet;
	delete img;
}

void adaptivetraining(int argc, char *argv[])
{
	int generations = 200;
	int numdata = 1000;
	int numnodes = 3;

	printf("Opening %s\n",argv[2]);
	pm_img *img = new pm_img(argv[2]);
	int finaldim;
	if(img->gettype() == 6){ finaldim = 3; } else { finaldim = 1; }
	printf("Creating Neural Network with dim %d,%d,%d\n", 2,numnodes,finaldim);
	nn *nurnet = new nn(2,numnodes,finaldim);
	nurnet->randfillnn(0.5);
	vec_data *D = get_vec_data_ppm(img, numdata);
	mkdir("imgfiles",0777);
	mkdir("imgfiles/sig",0777);
	mkdir("imgfiles/hea",0777);
	nurnet->animatedadaptivebackprop1(D, 0.05, -1, generations, 6, false);
	nurnet->save("test1.nn");
	delvec_data(D);
	delete nurnet;
	delete img;
}


int main(int argc, char *argv[])
{
	if(argc == 3 && argv[1][0] == '-' && argv[1][1] == 'i'){
		animatetraining(argc, argv);
		return 0;
	}
	if((argc == 3)  && argv[1][0] == '-' && argv[1][1] == 'e'){
		int start_s=clock();
		geterrordata(argc, argv);
		int stop_s=clock();
		cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << endl;
		return 0;
	}
	if(argc == 3 && argv[1][0] == '-' && argv[1][1] == 'a'){
		int start_s=clock();
		adaptivetraining(argc, argv);
		int stop_s=clock();
		cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << endl;
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
