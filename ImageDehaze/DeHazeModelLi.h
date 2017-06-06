#pragma once
#include "DeHazeModel.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;


class DeHazeModelLi : virtual public DeHazeModel
{
public:
	DeHazeModelLi(){};
	~DeHazeModelLi(){ destroyAllWindows(); };
public:
	virtual void dehaze(string srcPath, bool isStore = false, string dstPath = string()) override;
	
public:
	void getAFDarkImg(int d = 40, double sigma_color = 150 );
	void getConfidenceImg(float k1 = 8.0, float coeOfk2 = 8.0);
	void getTranImg() ;
	void getPostEnhancedImg();
	void getPostLumAjustImg();

	void getMinImg();
	void getLuminanceImg();
	void getBackgroundImg();
protected:
	Mat getConfidenceImg1(float k1);
	Mat getConfidenceImg2(float coeOfk2);
	uchar getDarkMax();
protected:
	Mat minImg;
	Mat luminanceImg;
	Mat backgroundImg;
	Mat confidenceImg;
};