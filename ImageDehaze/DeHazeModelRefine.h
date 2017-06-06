#pragma once
#include "DeHazeModelLi.h"
#include "DeHazeModelHe.h"
#include "DeHazeModelXu.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

class DeHazeModelRefine : public  DeHazeModelLi, public DeHazeModelHe, public DeHazeModelXu
{
public:
	DeHazeModelRefine(){};
	~DeHazeModelRefine(){ destroyAllWindows(); };
public:
	void dehaze(string srcPath, bool isStore = false, string dstPath = string()) override;
	//void getConfidenceImg(float k1, float coeOfk2);
protected:
	void getConfidenceImgSat(float k);
	void getConfidenceImg(float k1, float coeOfk2);
	void getTranImg();
	void getPostLumAjustImg();
	void getRestoredImg();
	void showImage(string winName, MatType mt);
	Mat maxFilter(Mat &src, int darkRadius);
	Mat minFilter(Mat &src, int darkRadius);
	//void showImage(string winName, MatType mt);
	
protected:
	//Mat lightImg;
};