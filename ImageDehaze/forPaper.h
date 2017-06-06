#pragma once
#include "DeHazeModelLi.h"
#include "DeHazeModelHe.h"
#include "DeHazeModelXu.h"
#include "DeHazeModelRefine.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

class ForPaper : public  DeHazeModelLi, public DeHazeModelHe, public DeHazeModelXu, public DeHazeModelRefine
{
public:
	virtual void dehaze(string srcPath, bool isStore = false, string dstPath = string()) override;
public:
	void getDarkPic(string srcPath);
	void showSaturation(string srcPath);
	Mat maxFilter(Mat &src, int darkRadius);
private:
	void showSaturation(Mat smat);
};