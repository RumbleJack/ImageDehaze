#pragma once
#include <string>
#include "opencv2/core.hpp"
using std::string;
using cv::Mat;
using cv::Vec3b;

class DeHazeModel
{
public:
	enum MatType {
		SRC, DST, DARK, TRAN,
		MIN, LUMINANCE,
		CONFIDENCE,
		LIGHT
	};
public:
	DeHazeModel();
	DeHazeModel(string srcFilePath);
	virtual ~DeHazeModel();
	virtual void readImg  (string srcFilePath) ;
	virtual void writeImg(string dstPath,  MatType mt = MatType::DST);
	virtual void showImage(string winName, MatType mt = MatType::DST) ;
	const Mat* getSrcImg() const;
	const Mat* getResImg() const;
	const Mat* getDarkImg() const;

	virtual void dehaze(string srcPath, bool isStore = false, string dstPath = string() ) = 0;
protected:
	void getDarkChannelImg(int darkRadius = 7);
	void getAtmosphericLight();
	void getTranImg();
	void getRestoredImg();
protected:
	Mat srcImg;
	Mat dstImg;
	Mat darkImg;
	Mat tranImg;
	Vec3b atmosphericLight;
};

struct Region
{
	int rmin, cmin;  //origion left-up
	int rmax, cmax;  // right-bottom
	Region(int _rmin, int _cmin, int _rmax, int _cmax) :
		rmin(_rmin), cmin(_cmin), rmax(_rmax), cmax(_cmax) {}
};

class Pixel
{
public:
	Pixel(int inten, int _i, int _j) : intensity(inten), i(_i), j(_j) {}
	bool operator> (const Pixel &b) const { return intensity > b.intensity; }
public:
	uchar intensity;
	int i, j;
};


