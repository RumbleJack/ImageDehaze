#pragma once
#include<string>
#include "opencv2/core.hpp"
#include "DeHazeModel.h"
using std::string;
using cv::Mat;

class DeHazeModelXu : virtual public DeHazeModel
{
public:
	virtual void dehaze(string srcPath, bool isStore = false, string dstPath = string()) override;
protected:
	void gFilter(int guidedRadius, float eps);
	void getDarkChannelImg(int radius = 7);
	void getLightChannelImg(int darkRadius = 7);
	void getTranImg();
	void getRestoredImg();
	void showImage(string winName, MatType mt);
protected:
	Mat lightImg;
};