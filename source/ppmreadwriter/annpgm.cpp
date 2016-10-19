#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>
#include <pthread.h>
#include <sys/stat.h>

#include <armadillo>
#include "../neuralnetwork/ann.h"
#include "../neuralnetwork/nnanalyzer.h"
#include "../neuralnetwork/nnmap.h"

#include "annpgm.h"
#include "pgmreader.h"

using namespace arma;
using namespace std;

#define MAXTHREADS 7

struct vec_data *get_vec_data_ppm(pm_img *img, int numdata)
{
	vec_data *thisdata = new vec_data;
	thisdata->data = new vec_datum[numdata];
	thisdata->numdata = numdata;
	int height = img->getheight();
	int width = img->getwidth();
	int i = 0;
	int j = 0;
	int x;
	int y;

	std::default_random_engine generator;
	std::uniform_int_distribution<int> dist_height(0,height-1);
	std::uniform_int_distribution<int> dist_width(0,width-1);

	while(i<numdata){
		x = dist_width(generator);
		y = dist_height(generator);
		//printf("x:%d,y:%d     i:%d,j:%d \n", x,y,i,j);
		//if(j > numdata){break;}
		thisdata->data[i].coords = vec(2,fill::zeros);
		thisdata->data[i].coords(1) = ((y-(double)height/2)/height)*10;
		thisdata->data[i].coords(0) = ((x-(double)width/2)/width)*10;
		thisdata->data[i].value = vec(1,fill::zeros);
		if(img->gettype() == 6) {			
			if(true){
				thisdata->data[i].value = vec(3,fill::zeros);
				thisdata->data[i].value(0) = ((double)((unsigned char)img->r(x,y))/255);
				thisdata->data[i].value(1) = ((double)((unsigned char)img->g(x,y))/255);
				thisdata->data[i].value(2) = ((double)((unsigned char)img->b(x,y))/255);
				i++;
			}
		} else {
			if(true){
				thisdata->data[i].value = vec(1,fill::zeros);
				thisdata->data[i].value(0) = ((double)((unsigned char)img->r(x,y))/255);
				i++;
			}
		}
		//printf("value in data: %f, actual value %d\n",r_ thisdata[i]alue, (unsigned char)pixarr[x][y]);
		j++;
	}
	return thisdata;
}

bool saveDataToHistory(struct vec_data *D, fstream *fp)
{
	if(fp->is_open()){
		*fp << D->numdata << endl;
		*fp << D->data[0].coords.n_rows << endl;
		*fp << D->data[0].value.n_rows << endl;
		printf(".");
		int i =0;
		bool At, bt;
		for(i=0;i<D->numdata;++i){
			printf(".");
			At = D->data[i].coords.save(*fp,raw_binary);
			bt = D->data[i].value.save(*fp,raw_binary);
			if(!(At) || !(bt)) {
				goto end;
			}
		}
		printf("Data History Save Successful\n");
		return true;
	}	
end:
	printf("Save failed\n");
	return false;
}

bool appendNNToHistory(nn *thisnn, fstream *fp)
{
	int depth = thisnn->getDepth();
	if(fp->is_open()){
		*fp << depth << endl;
		printf(".");
		int i =0;
		bool At, bt;
		for(i=0;i<depth;i++){
			//printf(".");
			*fp << thisnn->getmat(i).n_rows << endl;
			*fp << thisnn->getmat(i).n_cols << endl;
			At = thisnn->getmat(i).save(*fp,raw_binary);
			bt = thisnn->getoff(i).save(*fp,raw_binary);
			if(!(At) || !(bt)) {
				goto end;
			}
		}
		//printf("Save Successful\n");
		return true;
	}
	
end:
	printf("Save failed\n");
	return false;
}

fstream startHistory(const char *filename, nn *thisnn, vec_data *D, int numGenerations)
{
	printf("Saving: %s ", filename);
	fstream  fp;
	fp.open(filename, ios::out);
	fp << numGenerations << endl;
	if(fp.is_open()){
		bool dt = saveDataToHistory(D,&fp);
		bool nt = appendNNToHistory(thisnn, &fp);
		if(!(dt && nt))
			goto end;
		printf("Start of history Successful\n");
		return fp;
	}
end:
	printf("History start Failed\n");
	return fp;
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
				unsigned char val = (unsigned char)(floor((value(0))*255));
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



void write_nn_regions_to_img(nn *thisnn, nnmap *thismap, const char filename[], int height, int width, int func)
{
	int i=0;
	int j=0;
	int k=0;
	vec input = vec(2,fill::zeros);
	std::vector<int> value;
	pm_img *img = new pm_img(height,width,255,6);
	for(i=0;i<height;i++){
		for(j=0;j<width;j++){
			input(0) = ((i-(double)height/2)/height)*10;
			input(1) = ((j-(double)width/2)/width)*10;
			value = thismap->getRegionSig(input);
			int n = value.size();
			unsigned char red = 0;
			unsigned char blue = 0;
			unsigned char green = 0;
			for(k =0;k<n;++k){
				if(k % 3 == 0){ red+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 2){ blue+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 1){ green+= value[k]*256/(pow(2,k/3+1)); }
			}
			img->wr(i,j,red);
			img->wg(i,j,green);
			img->wb(i,j,blue);
		}
	}
	img->pm_write(filename);
	delete img;
}


void write_nn_inter_to_img(nn *thisnn, nnmap *thismap, const char filename[], int height, int width, int func)
{
	int i=0;
	int j=0;
	int k=0;
	vec input = vec(2,fill::zeros);
	std::vector<int> value;
	pm_img *img = new pm_img(height,width,255,6);
	for(i=0;i<height;i++){
		for(j=0;j<width;j++){
			input(0) = ((i-(double)height/2)/height)*10;
			input(1) = ((j-(double)width/2)/width)*10;
			value = thismap->getInterSig(input);
			int n = value.size();
			unsigned char red = 0;
			unsigned char blue = 0;
			unsigned char green = 0;
			for(k =0;k<n;++k){
				if(k % 3 == 0){ red+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 2){ blue+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 1){ green+= value[k]*256/(pow(2,k/3+1)); }
			}
			img->wr(i,j,red);
			img->wg(i,j,green);
			img->wb(i,j,blue);
		}
	}
	img->pm_write(filename);
	delete img;
}

void write_data_to_img(vec_data *data,const char filename[])
{
	pm_img img = pm_img(filename);
	int height = img.getheight();
	int width = img.getwidth();

	int i =0;
	int numdata = data->numdata;
	int x,y =0;
	for(i=0;i<numdata;i++){
		y = ((data->data[i].coords(1)*height/10+(double)height/2));
		x = ((data->data[i].coords(0)*width/10+(double)width/2));
		if(img.gettype()==6) {
			if(data->data[0].value.n_rows == 3){
				img.wr(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img.wg(x,y,(unsigned char)(floor((data->data[i].value(1))*255)));
				img.wb(x,y,(unsigned char)(floor((data->data[i].value(2))*255)));
			} else if(data->data[0].value.n_rows == 1){
				img.wr(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img.wg(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img.wb(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
			}
		} else if(img.gettype()==5) {
			img.wr(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
		}
	}
	img.pm_write(filename);
}

void write_all_nn_to_image(nn *thisnn,vec_data *data, const char filename[], int height, int width)
{
	int i=0;
	int j=0;
	int k=0;
	vec input = vec(2,fill::zeros);
	std::vector<int> value;
	vec vvalue;
	pm_img *img = new pm_img(height*2,width*2,255,6);

	nnmap *thismap = new nnmap(thisnn,data);

	for(i=0;i<height;i++){
		for(j=0;j<width;j++){
			input(0) = ((i-(double)height/2)/height)*10;
			input(1) = ((j-(double)width/2)/width)*10;
			value = thismap->getInterSig(input);
			int n = value.size();
			unsigned char red = 0;
			unsigned char blue = 0;
			unsigned char green = 0;
			for(k =0;k<n;++k){
				if(k % 3 == 0){ red+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 2){ blue+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 1){ green+= value[k]*256/(pow(2,k/3+1)); }
			}
			img->wr(i+width,j+height,red);
			img->wg(i+width,j+height,green);
			img->wb(i+width,j+height,blue);
			value = thismap->getRegionSig(input);
			n = value.size();
			red = 0;
			blue = 0;
			green = 0;
			for(k =0;k<n;++k){
				if(k % 3 == 0){ red+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 2){ blue+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 1){ green+= value[k]*256/(pow(2,k/3+1)); }
			}
			img->wr(i,j+height,red);
			img->wg(i,j+height,green);
			img->wb(i,j+height,blue);
			if(thisnn->outdim() == 1){
				vvalue = thisnn->evalnn(input, 1);
				unsigned char val = (unsigned char)(floor((vvalue(0))*255));
				img->wr(i+width,j,val);
				img->wg(i+width,j,val);
				img->wb(i+width,j,val);
				vvalue = thisnn->evalnn(input, 0);
				val = (unsigned char)(floor((vvalue(0))*255));
				img->wr(i,j,val);
				img->wg(i,j,val);
				img->wb(i,j,val);
			} else if(thisnn->outdim() ==2) { 
				vvalue = thisnn->evalnn(input, 1);
				img->wr(i+width,j, (unsigned char)(floor((vvalue(0))*255)));
				img->wg(i+width,j, (unsigned char)(floor((vvalue(1))*255)));
				img->wb(i+width,j, (unsigned char)(floor((vvalue(2))*255)));
				vvalue = thisnn->evalnn(input, 0);
				img->wr(i,j, (unsigned char)(floor((vvalue(0))*255)));
				img->wg(i,j, (unsigned char)(floor((vvalue(1))*255)));
				img->wb(i,j, (unsigned char)(floor((vvalue(2))*255)));
			} else if(thisnn->outdim() ==2) {
				vvalue = thisnn->evalnn(input, 1);
				img->wr(i+width,j, (unsigned char)(floor((vvalue(0))*255)));
				img->wg(i+width,j, (unsigned char)(floor((vvalue(1))*255)));
				img->wb(i+width,j, (unsigned char)(0));
				vvalue = thisnn->evalnn(input, 0);
				img->wr(i,j, (unsigned char)(floor((vvalue(0))*255)));
				img->wg(i,j, (unsigned char)(floor((vvalue(1))*255)));
				img->wb(i,j, (unsigned char)(0));
			} 
		}
	}
	
	

	int numdata = data->numdata;
	int x,y =0;
	for(i=0;i<numdata;i++){
		y = ((data->data[i].coords(1)*height/10+(double)height/2));
		x = ((data->data[i].coords(0)*width/10+(double)width/2));
		if(img->gettype()==6) {
			if(data->data[0].value.n_rows == 3){
				img->wr(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x,y,(unsigned char)(floor((data->data[i].value(1))*255)));
				img->wb(x,y,(unsigned char)(floor((data->data[i].value(2))*255)));

				img->wr(x+width,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x+width,y,(unsigned char)(floor((data->data[i].value(1))*255)));
				img->wb(x+width,y,(unsigned char)(floor((data->data[i].value(2))*255)));

				img->wr(x+width,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x+width,y+height,(unsigned char)(floor((data->data[i].value(1))*255)));
				img->wb(x+width,y+height,(unsigned char)(floor((data->data[i].value(2))*255)));

				img->wr(x,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x,y+height,(unsigned char)(floor((data->data[i].value(1))*255)));
				img->wb(x,y+height,(unsigned char)(floor((data->data[i].value(2))*255)));
			} else if(data->data[0].value.n_rows == 1){
				img->wr(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wb(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));

				img->wr(x+width,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x+width,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wb(x+width,y,(unsigned char)(floor((data->data[i].value(0))*255)));

				img->wr(x+width,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x+width,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wb(x+width,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));

				img->wr(x,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wb(x,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
			}
		} else if(img->gettype()==5) {
			img->wr(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
		}
	}
	delete thismap;
	img->pm_write(filename);
}



struct Print_args {
	pthread_mutex_t mutexnnmap;
	nn *thisnn;
	nnmap *thismap;
	pm_img *img;
	vec_data *data;
	int tid;
	int height;
	int width;
} Pring_args;

void *write_all_nn_to_image_thread(void *thread_args)
{

	struct Print_args *myargs;
	myargs = (struct Print_args *) thread_args;
	pthread_mutex_t mutexnnmap = myargs->mutexnnmap;
	int tid = myargs->tid;
	nn *thisnn = myargs->thisnn;
	nnmap *thismap = myargs->thismap;
	pm_img *img = myargs->img;
 	vec_data *data = myargs->data;
	int height = myargs->height;
	int width = myargs->width;
	//printf("In thread %d\n",tid );

	int i=0;
	int j=0;
	int k=0;
	vec input = vec(2,fill::zeros);
	std::vector<int> value;
	vec vvalue;
	
	for(i=tid;i<height;i=i+MAXTHREADS){
		for(j=0;j<width;j++){
			input(0) = ((i-(double)height/2)/height)*10;
			input(1) = ((j-(double)width/2)/width)*10;
			pthread_mutex_lock (&mutexnnmap);
    			value = thismap->getInterSig(input);
    		pthread_mutex_unlock (&mutexnnmap);
				
			int n = value.size();
			unsigned char red = 0;
			unsigned char blue = 0;
			unsigned char green = 0;
			for(k =0;k<n;++k){
				if(k % 3 == 0){ red+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 2){ blue+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 1){ green+= value[k]*256/(pow(2,k/3+1)); }
			}
			img->wr(i+width,j+height,red);
			img->wg(i+width,j+height,green);
			img->wb(i+width,j+height,blue);
			value = thismap->getRegionSig(input);
			n = value.size();
			red = 0;
			blue = 0;
			green = 0;
			for(k =0;k<n;++k){
				if(k % 3 == 0){ red+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 2){ blue+= value[k]*256/(pow(2,k/3+1)); }
				if(k % 3 == 1){ green+= value[k]*256/(pow(2,k/3+1)); }
			}
			img->wr(i,j+height,red);
			img->wg(i,j+height,green);
			img->wb(i,j+height,blue);
			if(thisnn->outdim() == 1){
				vvalue = thisnn->evalnn(input, 1);
				unsigned char val = (unsigned char)(floor((vvalue(0))*255));
				img->wr(i+width,j,val);
				img->wg(i+width,j,val);
				img->wb(i+width,j,val);
				vvalue = thisnn->evalnn(input, 0);
				val = (unsigned char)(floor((vvalue(0))*255));
				img->wr(i,j,val);
				img->wg(i,j,val);
				img->wb(i,j,val);
			} else if(thisnn->outdim() ==3) { 
				vvalue = thisnn->evalnn(input, 1);
				img->wr(i+width,j, (unsigned char)(floor((vvalue(0))*255)));
				img->wg(i+width,j, (unsigned char)(floor((vvalue(1))*255)));
				img->wb(i+width,j, (unsigned char)(floor((vvalue(2))*255)));
				vvalue = thisnn->evalnn(input, 0);
				img->wr(i,j, (unsigned char)(floor((vvalue(0))*255)));
				img->wg(i,j, (unsigned char)(floor((vvalue(1))*255)));
				img->wb(i,j, (unsigned char)(floor((vvalue(2))*255)));
			} else if(thisnn->outdim() ==2) {
				vvalue = thisnn->evalnn(input, 1);
				img->wr(i+width,j, (unsigned char)(floor((vvalue(0))*255)));
				img->wg(i+width,j, (unsigned char)(floor((vvalue(1))*255)));
				img->wb(i+width,j, (unsigned char)(0));
				vvalue = thisnn->evalnn(input, 0);
				img->wr(i,j, (unsigned char)(floor((vvalue(0))*255)));
				img->wg(i,j, (unsigned char)(floor((vvalue(1))*255)));
				img->wb(i,j, (unsigned char)(0));
			} 
		}
	}
	
	

	int numdata = data->numdata;
	int x,y =0;
	for(i=tid;i<numdata;i+=MAXTHREADS){
		y = ((data->data[i].coords(1)*height/10+(double)height/2));
		x = ((data->data[i].coords(0)*width/10+(double)width/2));
		if(img->gettype()==6) {
			if(data->data[0].value.n_rows == 3){
				img->wr(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x,y,(unsigned char)(floor((data->data[i].value(1))*255)));
				img->wb(x,y,(unsigned char)(floor((data->data[i].value(2))*255)));

				img->wr(x+width,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x+width,y,(unsigned char)(floor((data->data[i].value(1))*255)));
				img->wb(x+width,y,(unsigned char)(floor((data->data[i].value(2))*255)));

				img->wr(x+width,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x+width,y+height,(unsigned char)(floor((data->data[i].value(1))*255)));
				img->wb(x+width,y+height,(unsigned char)(floor((data->data[i].value(2))*255)));

				img->wr(x,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x,y+height,(unsigned char)(floor((data->data[i].value(1))*255)));
				img->wb(x,y+height,(unsigned char)(floor((data->data[i].value(2))*255)));
			} else if(data->data[0].value.n_rows == 1){
				img->wr(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wb(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));

				img->wr(x+width,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x+width,y,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wb(x+width,y,(unsigned char)(floor((data->data[i].value(0))*255)));

				img->wr(x+width,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x+width,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wb(x+width,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));

				img->wr(x,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wg(x,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
				img->wb(x,y+height,(unsigned char)(floor((data->data[i].value(0))*255)));
			}
		} else if(img->gettype()==5) {
			img->wr(x,y,(unsigned char)(floor((data->data[i].value(0))*255)));
		}
	}

	//printf("Exiting thread %d\n", tid);
	pthread_exit(NULL);
}


void write_all_nn_to_image_parallel(nn *thisnn,vec_data *data, const char filename[], int height, int width)
{
	pthread_t threads[MAXTHREADS];
	pthread_mutex_t mutexnnmap;

	int rc = 0;
	int i = 0;
	struct Print_args *thread_args = new struct Print_args[MAXTHREADS];
		
	pm_img *img = new pm_img(height*2,width*2,255,6);
	nnmap *nurnetMap = new nnmap(thisnn,data);

	pthread_mutex_init(&mutexnnmap, NULL);
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	

	

	for(i=0;i<MAXTHREADS;i++){
		thread_args[i].mutexnnmap = mutexnnmap;
		thread_args[i].thisnn = thisnn;
		thread_args[i].thismap = nurnetMap;
		thread_args[i].data = data;
		thread_args[i].img = img;
		thread_args[i].height = height;
		thread_args[i].width = width;
		thread_args[i].tid = i;
		rc = pthread_create(&threads[i], NULL, write_all_nn_to_image_thread, (void *)&thread_args[i]);
		if (rc){
			cout << "Error:unable to create thread," << rc << endl;
			exit(-1);
		}
	}
	


	for(i = 0; i<MAXTHREADS;++i){
		rc = pthread_join(threads[i], &status);
		if (rc){
			cout << "Error:unable to join," << rc << endl;
			exit(-1);
		}
		
	}
	img->pm_write(filename);
	delete nurnetMap;
	delete img;
}

/*
void write_hperrs_to_imgs(nn *thisnn, const char filename[], const char dirname[], const char imgfile[])
{
	pm_img *groundTruth = new pm_img(imgfile);
	int height = groundTruth->getheight();
	int width = groundTruth->getwidth();
	int i=0;
	int j=0;
	int k=0;
	int l=0;
	vec input = vec(2,fill::zeros);
	vec output = vec(1,fill::zeros);
	vec value;
	for(l=0;l<thisnn->outdim(0);++l){
		pm_img *img = new pm_img(height,width,255,5);
		for(i=0;i<height;i++){
			for(j=0;j<width;j++){
				input(0) = ((i-(double)height/2)/height)*10;
				input(1) = ((j-(double)width/2)/width)*10;
				output(0) = ((double)groundTruth->r(i,j))*255;
				vec_datum datum;
				datum.coords = input;
				datum.value = output;
				value = thisnn->getHPLayerErrorData(datum);
				char c = value(l)/255;	
				img->wr(i,j,c);
			}
		}
		char header[100];
		sprintf(header, "%s%02d%s",dirname,l,filename);
		img->pm_write(header);
		delete img;
	}
}
*/