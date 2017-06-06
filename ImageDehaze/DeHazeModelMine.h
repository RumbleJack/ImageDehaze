#pragma once
#include "DeHazeModelLi.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;

class DeHazeModelMine : public DeHazeModelLi
{
public:
	DeHazeModelMine(){};
	~DeHazeModelMine(){ destroyAllWindows(); };
public:
	void dehaze(string srcPath, bool isStore = false, string dstPath = string()) override;
	void getConfidenceImg(float k1, float coeOfk2);
protected:
	void getAFDarkImg(int d = 40, double sigma_color = 150);
	void getPostLumAjustImg();
};