#ifndef _pgmreader_h
#define _pgmreader_h

class pm_img {
private:
	unsigned char **r_pixarr; // In a pgm the greyscale value is stored in r, or the red.
	unsigned char **g_pixarr;
	unsigned char **b_pixarr;
	int height;
	int width;
	int maxval;
	int type; 
public:
	pm_img(int height, int width, int maxval, int type);
	pm_img(const char *filename);
	void pm_write(const char *filename);
	~pm_img();
	int gettype();
	int getwidth();
	int getheight();
	unsigned char r(int i, int j);
	unsigned char g(int i, int j);
	unsigned char b(int i, int j);
	void wr(int i, int j,unsigned char val);
	void wg(int i, int j,unsigned char val);
	void wb(int i, int j,unsigned char val);
};

#endif