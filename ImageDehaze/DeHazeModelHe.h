#pragma once
#include<string>
#include "opencv2/core.hpp"
#include "DeHazeModel.h"
using std::string;
using cv::Mat;

class DeHazeModelHe : virtual public DeHazeModel
{
public:
	virtual void dehaze(string srcPath, bool isStore = false, string dstPath = string()) override;
protected:
	void gFilter(int guidedRadius, float eps);
	void softMatting(int radius, float eps);
private:
	void getLapLacian();
};

