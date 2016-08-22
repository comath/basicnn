#include <iostream>
#include <random>
#include <math.h>
#include <iomanip>
#include <pthread.h>
#include <sys/stat.h>
#include "ann.h"
#include "pgmreader.h"
#include "annpgm.h"


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
			img.wr(x,y,127);
			img.wg(x,y,127);
			img.wb(x,y,127);
		} else if(img.gettype()==5) {
			img.wr(x,y,127);
		}
	}
	img.pm_write(filename);
}