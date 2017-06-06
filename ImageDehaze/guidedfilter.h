#pragma once 
#include <opencv2/opencv.hpp>
using cv::Mat;

class GuidedFilterImpl;
class GuidedFilter
{
public:
	GuidedFilter(const Mat &I, int r, double eps);
	~GuidedFilter();
	Mat filter(const Mat &p, int depth = -1) const;
private:
	GuidedFilterImpl *impl_;
};
// -guidance image : I(should be a gray - scale / single channel image) or color imgage
// -filtering input image : p(should be a gray - scale / single channel image)
// -local window radius : r  
// -regularization parameter : eps   the smaller eps is ,the more detail preserved 
Mat guidedFilter(const Mat &I, const Mat &p, int r, double eps, int depth = -1);
