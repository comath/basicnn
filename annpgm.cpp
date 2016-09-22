#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>
#include <pthread.h>
#include <sys/stat.h>

#include <armadillo>
#include "ann.h"
#include "pgmreader.h"
#include "annpgm.h"
#include "nnanalyzer.h"


using namespace arma;
using namespace std;



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
			printf(".");
			*fp << thisnn->getmat(i).n_rows << endl;
			*fp << thisnn->getmat(i).n_cols << endl;
			At = thisnn->getmat(i).save(*fp,raw_binary);
			bt = thisnn->getoff(i).save(*fp,raw_binary);
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



void write_nn_regions_to_img(nn *thisnn, const char filename[], int height, int width, int func)
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
			value = getRegionSig(input, thisnn->getmat(0), thisnn->getoff(0));
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

void write_nn_inter_to_img(nn *thisnn, const char filename[], int height, int width, int func)
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
			value = getInterSig(input, thisnn->getmat(0), thisnn->getoff(0));
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