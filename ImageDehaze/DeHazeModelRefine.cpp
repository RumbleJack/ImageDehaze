#include "DeHazeModelRefine.h"
#include "guidedfilter.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "BiFilter.h"

using namespace std;
using namespace cv;

void DeHazeModelRefine::dehaze(string srcPath, bool isStore, string dstPath)
{
	//读取图片
	readImg(srcPath);

	//计算暗通道
	getDarkChannelImg(15);
	darkImg.convertTo(darkImg, CV_32FC1);
	darkImg = guidedFilter(srcImg, darkImg, 40, 10);
	darkImg.convertTo(darkImg, CV_8UC1);

	//计算大气光
	getAtmosphericLight();
	getLightChannelImg(7);
	lightImg.convertTo(lightImg, CV_32FC1);
	lightImg = guidedFilter(srcImg, lightImg, 40, 10);

	//计算暗通道置信度
	getConfidenceImg(8, 8.0);
	getConfidenceImgSat(0);

	//计算透射率
	getTranImg();

	//计算恢复图像
	getRestoredImg();

	if (isStore == true)
		writeImg(dstPath + "RefineRes_");
	//进行后置增强
	getPostEnhancedImg();

	//进行HSV空间的亮度映射
	getPostLumAjustImg();


	if (isStore == true)
		writeImg(dstPath + "RefineHLSMapRes_");
}

void DeHazeModelRefine::getPostLumAjustImg()
{
	vector<Mat> channels;
	Mat adjustDstImg = dstImg.clone();
	cvtColor(dstImg, adjustDstImg, CV_BGR2HSV);
	split(adjustDstImg, channels);
	Mat &lumChannel = channels[2];
	//imshow("pre", lumChannel);
	double minv = 0, maxv = 0;
	minMaxLoc(lumChannel, &minv, &maxv);

	Mat temp(lumChannel.size(), CV_32FC1);
	lumChannel.convertTo(temp, temp.type());
	divide(temp, (maxv - 0), temp, 255.0, temp.type());
	temp.convertTo(lumChannel, lumChannel.type());
	//imshow("post", lumChannel);
	merge(channels, adjustDstImg);
	cvtColor(adjustDstImg, dstImg, CV_HSV2BGR);
}
void DeHazeModelRefine::getTranImg()
{
	float atmAve = (atmosphericLight[0] + atmosphericLight[1] + atmosphericLight[2]) / 3;
	float a = 0.6;
	float b = 0.25;

	cv::multiply(darkImg, confidenceImg, tranImg, 1.0, CV_32FC1);
	divide(tranImg, (a*lightImg + b*atmAve), tranImg, 1, CV_32FC1);
	cv::subtract(1.0, tranImg, tranImg, noArray(), CV_32FC1);
	tranImg = abs(tranImg);
}
void DeHazeModelRefine::getConfidenceImgSat(float k)
{
	Mat hlsSrc = srcImg.clone();
	cvtColor(hlsSrc, hlsSrc, CV_BGR2HSV);

	vector<Mat> channels;
	split(hlsSrc, channels);
	Mat &hue = channels[0];
	Mat &lumChannel = channels[2];
	Mat &saturation = channels[1];

	saturation = maxFilter(saturation, 7);
	saturation = guidedFilter(srcImg, saturation, 40, 1);
	lumChannel = minFilter(lumChannel, 7);
	lumChannel = guidedFilter(srcImg, lumChannel, 40, 10);


	lumChannel = darkImg;
	saturation.convertTo(saturation, CV_32FC1);
	saturation /= 256;


	Mat confidence1(srcImg.size(), CV_32FC1);
	divide(0 - saturation, 0.01, confidence1, 1.0, CV_32FC1);
	exp(confidence1, confidence1);
	divide(1.0, 1 + confidence1, confidence1);
	

	Mat confidence3(srcImg.size(), CV_32FC1);
	double maxL;
	minMaxLoc(lumChannel, nullptr, &maxL);
	float k3 = maxL / 8;
	lumChannel.convertTo(lumChannel, CV_32FC1);
	divide(lumChannel - 200, k3, confidence3, 1.0, CV_32FC1);
	exp(confidence3, confidence3);
	divide(1.0, 1 + confidence3, confidence3);


	Mat confidence2(srcImg.size(), CV_32FC1);
	double Dmax;
	minMaxLoc(darkImg, nullptr, &Dmax);
	float k2 = Dmax ;
	Mat floatDarkImg(darkImg.size(), CV_32FC1);
	darkImg.convertTo(floatDarkImg, CV_32FC1);
	divide(floatDarkImg - Dmax, k2, confidence2, 1.0, CV_32FC1);
	exp(confidence2, confidence2);
	divide(1.0, 1 + confidence2, confidence2);


	float* datac = reinterpret_cast<float*>(confidenceImg.data);
	float* datac1 = reinterpret_cast<float*>(confidence1.data);
	float* datac2 = reinterpret_cast<float*>(confidence3.data);
	int size = srcImg.size().area();
	for (int i = 0; i < size; i++){
		datac[i] = sqrt(datac1[i] * datac2[i]);
	}

	//imwrite("paper/c1.bmp", confidence1 * 255);
	//imwrite("paper/c2.bmp", confidence2 * 255);
	//imwrite("paper/c.bmp", confidenceImg * 255);
	//waitKey(0);

}
void DeHazeModelRefine::getConfidenceImg(float k1, float coeOfk2)
{
	Mat c1 = getConfidenceImg1(k1);
	Mat c2 = getConfidenceImg2(coeOfk2);
	confidenceImg.create(srcImg.size(), CV_32FC1);

	Mat temp(c1.size(), c1.type());
	//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	//morphologyEx(c1, temp, MORPH_OPEN, element);
	//c1 = temp.clone();

	temp = guidedFilter(srcImg, c1, 40, 10);
	//bilateralFilter(c1, temp, 40, 120 / 256.0, 90, BORDER_DEFAULT);

	c1.release();
	c1 = temp.clone();
	//imshow("c1", c1);
	//imshow("c2", c2);

	float* datac = reinterpret_cast<float*>(confidenceImg.data);
	float* datac1 = reinterpret_cast<float*>(c1.data);
	float* datac2 = reinterpret_cast<float*>(c2.data);
	int size = srcImg.size().area();
	for (int i = 0; i < size; i++){
		//datac[i] = max(datac1[i], datac2[i]);
		//datac[i] = 1;// max(datac1[i] * datac2[i], static_cast<float>(0.8));
		datac[i] = sqrt(datac1[i] * datac2[i]);
		//datac[i] = datac[i] > 0.8 ? datac[i] :
		//	(datac[i] > 0.1 ? 0.8 : datac[i]);
	}
	//cout << confidenceImg;
}
void DeHazeModelRefine::getRestoredImg()
{
	vector<Mat> srcChannels;
	vector<Mat> dstChannels;
	split(srcImg, srcChannels);
	split(dstImg, dstChannels);

	vector<Mat> tempChannels1(3);
	vector<Mat> tempChannels2(3);
	float a = 0.7;
	float b = 0.25;

	float atmAve = (atmosphericLight[0] + atmosphericLight[1] + atmosphericLight[2]) / 3;

	for (int i = 0; i < 3; i++)
	{
		tempChannels1[i].create(srcChannels[i].size(), CV_32FC1);
		tempChannels2[i].create(dstChannels[i].size(), CV_32FC1);
		srcChannels[i].convertTo(tempChannels1[i], CV_32FC1);

		divide(tempChannels1[i] - (a*lightImg + b*atmAve), tranImg, tempChannels2[i], 1, CV_32FC1);
		add((a*lightImg + b*atmAve), tempChannels2[i], dstChannels[i], noArray(), CV_8UC1);
	}
	merge(dstChannels, dstImg);
}

void DeHazeModelRefine::showImage(string winName, MatType mt)
{
	DeHazeModel::showImage(winName, mt);
	Mat temp(srcImg.size(), CV_8UC1);
	switch (mt) {
	case LIGHT:
		lightImg.convertTo(temp, CV_8UC1);
		imshow(winName, temp);
		break;
	case CONFIDENCE:
		//confidenceImg.convertTo(temp, CV_8UC1);
		imshow(winName, confidenceImg);
		break;
	}
}

Mat DeHazeModelRefine::maxFilter(Mat &srcImg, int darkRadius)
{
	Mat dst(srcImg.size(), srcImg.type());
	uchar* Data = dst.data;
	//边界填充255；
	Mat tempImg;
	copyMakeBorder(srcImg, tempImg, darkRadius, darkRadius, darkRadius, darkRadius, BORDER_CONSTANT, 0);

	//获取参数
	int cn = 1;
	int d = darkRadius * 2 + 1;
	vector<int> offset(d*d);
	int * _offset = &offset[0];
	//计算偏移量
	int sizeOfTP = 0;
	for (int i = -darkRadius; i <= darkRadius; i++)
		for (int j = -darkRadius; j <= darkRadius; j++)
		{
			/* 取消此注释后，mask为圆形
			if (std::sqrt(i*i + j*j) > darkRadius)
			continue;*/
			_offset[sizeOfTP++] = tempImg.step*i + j*cn;
		}
	//遍历行
	for (int i = 0; i < srcImg.rows; ++i) {
		uchar* imgData = tempImg.ptr<uchar>(i + darkRadius) + darkRadius*cn;
		//遍历列
		for (int j = 0; j < srcImg.cols; ++j) {
			uchar pixel = 0;
			for (int index = 0; index < sizeOfTP; index++){
				pixel = max(imgData[j*cn + _offset[index]], pixel);
				/*pixel = min(imgData[j*cn + _offset[index] + 1], pixel);
				pixel = min(imgData[j*cn + _offset[index] + 2], pixel);*/
			}
			Data[i*srcImg.step + j] = pixel;
			//Data[i*srcImg.step + j] *= 2;
		}
	}
	tempImg.release();
	return dst;
}

Mat DeHazeModelRefine::minFilter(Mat &srcImg, int darkRadius)
{
	Mat dst(srcImg.size(), srcImg.type());
	uchar* Data = dst.data;
	//边界填充255；
	Mat tempImg;
	copyMakeBorder(srcImg, tempImg, darkRadius, darkRadius, darkRadius, darkRadius, BORDER_CONSTANT, 255);

	//获取参数
	int cn = 1;
	int d = darkRadius * 2 + 1;
	vector<int> offset(d*d);
	int * _offset = &offset[0];
	//计算偏移量
	int sizeOfTP = 0;
	for (int i = -darkRadius; i <= darkRadius; i++)
		for (int j = -darkRadius; j <= darkRadius; j++)
		{
			/* 取消此注释后，mask为圆形
			if (std::sqrt(i*i + j*j) > darkRadius)
			continue;*/
			_offset[sizeOfTP++] = tempImg.step*i + j*cn;
		}
	//遍历行
	for (int i = 0; i < srcImg.rows; ++i) {
		uchar* imgData = tempImg.ptr<uchar>(i + darkRadius) + darkRadius*cn;
		//遍历列
		for (int j = 0; j < srcImg.cols; ++j) {
			uchar pixel = 255;
			for (int index = 0; index < sizeOfTP; index++){
				pixel = min(imgData[j*cn + _offset[index]], pixel);
				/*pixel = min(imgData[j*cn + _offset[index] + 1], pixel);
				pixel = min(imgData[j*cn + _offset[index] + 2], pixel);*/
			}
			Data[i*srcImg.step + j] = pixel;
			//Data[i*srcImg.step + j] *= 2;
		}
	}
	tempImg.release();
	return dst;
}