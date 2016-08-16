#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cassert> 
#include "pgmreader.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

char nextch(FILE *fp)
{
	char ch;
	fread(&ch,sizeof(ch),1,fp);
	return ch;
}

pm_img::pm_img(char *filename)
{
	printf("Opening file %s\n",filename);
	FILE *imgfile;
	//log_info("Opening file %s\n",filename);
	imgfile = fopen(filename,"r");
	if(imgfile == NULL){printf("Unable to open %s \n", filename); exit(1);}
	int i=0;
	int j=0;
	if(nextch(imgfile) != 'P'){printf("Invalid PGM file.\n"); exit(1);}
	type = nextch(imgfile)-48;
	if(type !=5 && type != 6){printf("Invalid Greyscale PGM file.\n");exit(1);}


	while(nextch(imgfile) != '\n');	
	while(nextch(imgfile) == '#'){
		while(nextch(imgfile) != '\n');
	}
	fseek(imgfile, -1, SEEK_CUR);
	/*while(!feof(imgfile)){
		printf("%c", nextch(imgfile));
	}*/
	int reader;
	fscanf(imgfile,"%d", &reader);
	width = reader;
	fscanf(imgfile,"%d", &reader);
	height = reader;
	fscanf(imgfile,"%d", &reader);
	maxval = reader;
	while(nextch(imgfile) != '\n');
	/*while(!feof(imgfile)){
		printf("%c", nextch(imgfile));
	}*/

	r_pixarr = new unsigned char*[height];
	for (int i = 0; i < height; ++i){
		r_pixarr[i] = new unsigned char[width];
	}

	if(type == 6) {
		g_pixarr = new unsigned char*[height];
		for (int i = 0; i < height; ++i){
			g_pixarr[i] = new unsigned char[width];
		}
		b_pixarr = new unsigned char*[height];
		for (int i = 0; i < height; ++i){
			b_pixarr[i] = new unsigned char[width];
		}
	}

	//check(maxval == 255, "Not valid maxval for this program, needs to be 255");
	if(type == 6) {
		for(i=0;i<height;i++){
			for(j=0;j<width;j++){
				r_pixarr[i][j]=nextch(imgfile);
				g_pixarr[i][j]=nextch(imgfile);
				b_pixarr[i][j]=nextch(imgfile);
			}
		}
	} else {
		for(i=0;i<height;i++){
			for(j=0;j<width;j++){
				r_pixarr[i][j]=nextch(imgfile);
			}
		}
	}

	fclose(imgfile);

}

pm_img::pm_img(int h, int w, int mv, int t)
{
	height = h;
	width = w;
	maxval = mv;
	type = t;
	//check(pixarr,"Memory Error");
	r_pixarr = new unsigned char*[height];
	for (int i = 0; i < height; ++i){
		r_pixarr[i] = new unsigned char[width];
	}

	if(type == 6) {
		g_pixarr = new unsigned char*[height];
		for (int i = 0; i < height; ++i){
			g_pixarr[i] = new unsigned char[width];
		}
		b_pixarr = new unsigned char*[height];
		for (int i = 0; i < height; ++i){
			b_pixarr[i] = new unsigned char[width];
		}
	}
}



void pm_img::pm_write(const char *filename)
{
	printf("Outputting: %s\n", filename);
	FILE *fp;
	fp = fopen(filename,"wb");
	//check(fp,"Unable to create %s",filename);
	char magicnum[3] = {'P',(char)(type + 48),'\n'};
	fwrite(magicnum,sizeof(char)*3,1,fp);
	int rc = fprintf(fp, "%d",width);
	//check(rc>0,"File write error");
	fputc(' ',fp);
	rc = fprintf(fp, "%d", height);
	//check(rc>0,"File write error");
	fputc('\n',fp);
	rc = fprintf(fp, "%d", maxval);
	//check(rc>0,"File write error");
	fputc('\n',fp);

	int i,j =0;
	if(type == 6) {
		for(i=0;i<(height);i++){
			for(j=0;j<(width);j++){
				fputc(r_pixarr[i][j],fp);
				fputc(g_pixarr[i][j],fp);
				fputc(b_pixarr[i][j],fp);
			}
		}
	} else {
		for(i=0;i<(height);i++){
			for(j=0;j<(width);j++){
				fputc(r_pixarr[i][j],fp);
			}
		}
	}

	//fclose(fp);
error:
	if(fp) fclose(fp);
}

pm_img::~pm_img()
{
	int i =0;
	for(i=0; i < (height); i++){
		
		delete[] r_pixarr[i];
	}
	delete[] r_pixarr;
	if(type == 6) {
		for(i=0; i < (height); i++){
			delete[] g_pixarr[i];
			delete[] b_pixarr[i];
		}
		delete[] g_pixarr;
		delete[] b_pixarr;
	}
}

int pm_img::gettype(){	return type;}
int pm_img::getwidth(){	return width;}
int pm_img::getheight(){return height;}
	
unsigned char pm_img::r(int i, int j){ return r_pixarr[i][j]; }
unsigned char pm_img::g(int i, int j){ return g_pixarr[i][j]; }
unsigned char pm_img::b(int i, int j){ return g_pixarr[i][j]; }

void pm_img::wr(int i, int j,unsigned char val)
{
	r_pixarr[i][j] = val;
}
void pm_img::wg(int i, int j,unsigned char val)
{
	g_pixarr[i][j] = val;
}
void pm_img::wb(int i, int j,unsigned char val)
{
	b_pixarr[i][j] = val;
}