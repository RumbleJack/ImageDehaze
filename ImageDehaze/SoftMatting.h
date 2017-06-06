#pragma once 
#include <opencv2/opencv.hpp>
using cv::Mat;

class SoftMattingImpl;
class SoftMatting
{
public:
	SoftMatting(const Mat &I, int r, double eps,double lamda = 0.0001);
	~SoftMatting();
	Mat matting(const Mat &p) const;
private:
	SoftMattingImpl *impl_;
};
// -guidance image : I(should be a gray - scale / single channel image) or color imgage
// -filtering input image : p(should be a gray - scale / single channel image)
// -local window radius : r  
// -regularization parameter : eps   the smaller eps is ,the more detail preserved 
Mat softMatting(const Mat &I, const Mat &tin, int r, double eps, double lamda = 0.0001);
