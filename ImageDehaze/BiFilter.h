#include "core.hpp"
using cv::Mat;
using cv::Size;
using cv::Range;



class CrossBilateralFilter
{
public:
	CrossBilateralFilter(){}
	CrossBilateralFilter(Mat& _dest, const Mat& _temp, int _radius, int _maxk,
		int* _space_ofs, float *_space_weight, float *_color_weight) :
		temp(&_temp), dest(&_dest), radius(_radius),
		maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_weight(_color_weight)
	{
	}
	CrossBilateralFilter(Mat& _dest, const Mat& _temp, const Mat& _tempGuide, int _radius, int _maxk,
		int* _space_ofs, float *_color_weight) :
		temp(&_temp), tempGuide(&_tempGuide),dest(&_dest), radius(_radius),
		maxk(_maxk), space_ofs(_space_ofs), color_weight(_color_weight)
	{
	}
	void biFilter();
	void associativeBiFilter();
	void crossBiFilter();
private:
	const Mat *temp;	//����Ե��Դͼ��
	const Mat *tempGuide; // ����Ե������ͼ��
	Mat *dest;			//���ͼ��
	int radius, maxk, *space_ofs;	//ģ��뾶��ģ���С��ģ��Ԫ�������ģ��������temp�е�ƫ��������
	float *space_weight, *color_weight;
};

void biFilter_8u(const Mat& src, Mat& dst, int d, double sigma_color, double sigma_space, int borderType = cv::BORDER_DEFAULT);
void crossBiFilter_8u(const Mat& src, Mat& dst, int d, double sigma_color, double sigma_space, int borderType = cv::BORDER_DEFAULT);
void associativeBiFilter_8u(const Mat& srcCS, const Mat& srcRef, Mat& dst, int d, double sigma_color, int borderType = cv::BORDER_DEFAULT);