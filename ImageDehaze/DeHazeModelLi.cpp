#include "DeHazeModelLi.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "BiFilter.h"
#include <vector>
#include <queue>
#include <functional>
#include <iostream>
using namespace std;
using namespace cv;

void DeHazeModelLi::dehaze(string srcPath, bool isStore, string dstPath )
{
	//读取图片
	readImg(srcPath);
	//计算暗通道
	getDarkChannelImg(7);
	//showImage("dark", ImgDehaze::DARK);
	getMinImg();
	//showImage("min", ImgDehaze::MIN);
	getAFDarkImg(40, 150);
	//showImage("assDark", DARK);
	//计算大气光
	getAtmosphericLight();
	//计算暗通道置信度
	getConfidenceImg(8, 8.0);
	//showImage("ConfidenceMat", CONFIDENCE);
	//计算透射率
	getTranImg();
	//showImage("tran", ImgDehaze::TRAN);
	//计算恢复图像
	getRestoredImg();
	imwrite("paper/cx.bmp",dstImg);
	//showImage("res", DST);
	if (isStore == true)
		writeImg("res_"+dstPath);
	//进行后置增强
	getPostEnhancedImg();
	//getPostLumAjustImg();
	//showImage("resEnhance", DST);
	imwrite("test.bmp", dstImg);
	if (isStore == true)
		writeImg(dstPath + "LiRes_");
}


void DeHazeModelLi::getAFDarkImg(int d, double sigma_color)
{
	Mat assImg(darkImg.size(), darkImg.type());
	//crossBiFilter_8u(darkImg, minImg, assImg, d, sigma_color, cv::BORDER_DEFAULT);
	associativeBiFilter_8u(darkImg, minImg, assImg, d, sigma_color, cv::BORDER_DEFAULT);
	darkImg = assImg;
}
void DeHazeModelLi::getConfidenceImg(float k1, float coeOfk2)
{
	Mat c1 = getConfidenceImg1(k1);
	Mat c2 = getConfidenceImg2(coeOfk2);
	confidenceImg.create(srcImg.size(), CV_32FC1);

	float* datac = reinterpret_cast<float*>(confidenceImg.data);
	float* datac1 = reinterpret_cast<float*>(c1.data);
	float* datac2 = reinterpret_cast<float*>(c2.data);
	int size = srcImg.size().area();
	for (int i = 0; i < size; i++)
		datac[i] = max(datac1[i], datac2[i]);

	//imshow("c1", c1);
	//imshow("c2", c2);
	//imshow("c", confidenceImg);
	//imwrite("paper/c1.bmp", c1*255);
	//imwrite("paper/c2.bmp", c2 * 255);
	//imwrite("paper/c.bmp", confidenceImg * 255);
	//waitKey(0);
}
void DeHazeModelLi::getTranImg()
{
	float atmAve = (atmosphericLight[0] + atmosphericLight[1] + atmosphericLight[2]) / 3;
	cv::multiply(darkImg, confidenceImg, tranImg, 1.0 / atmAve, CV_32FC1);
	//cv::multiply(darkImg, 1, tranImg, 1.0 / atmAve, CV_32FC1);
	cv::subtract(1.0, tranImg, tranImg, noArray(), CV_32FC1);
	tranImg = abs(tranImg);
}

void DeHazeModelLi::getPostEnhancedImg()
{
	getBackgroundImg();
	double maxlumPixVal = 0;
	double maxBGPixVal = 0;
	minMaxLoc(luminanceImg, nullptr, &maxlumPixVal);
	minMaxLoc(backgroundImg, nullptr, &maxBGPixVal);

	Size matSize = srcImg.size();
	//计算亮度映射图
	Mat lumMapImg(matSize, CV_32FC1);
	Mat tempImg(matSize, CV_32FC1);
	luminanceImg.convertTo(tempImg, tempImg.type());
	divide(tempImg, maxlumPixVal, tempImg, 0.5, tempImg.type());
	multiply(Scalar(1.0) - tempImg, luminanceImg, lumMapImg, 1.0, lumMapImg.type());

	//计算背景映射图
	Mat backgroundMapImg(matSize, CV_32FC1);
	backgroundImg.convertTo(tempImg, tempImg.type());
	divide(tempImg, maxBGPixVal, tempImg, 0.5, tempImg.type());
	multiply(Scalar(1.0) - tempImg, backgroundImg, backgroundMapImg, 1.0, backgroundMapImg.type());

	//计算对比度系数
	Mat ratioMat(matSize, CV_32FC1);
	Mat ratioMapMat(matSize, CV_32FC1);
	backgroundImg.convertTo(tempImg, tempImg.type());
	divide(luminanceImg, tempImg, ratioMat, 1.0, ratioMat.type());
	divide(lumMapImg, backgroundMapImg, ratioMapMat, 1.0, ratioMapMat.type());

	//计算增强亮度图像及其与原亮度图像之间Scale
	Mat lumEnhanceImg(matSize, CV_32FC1);
	multiply(lumMapImg, ratioMat / ratioMapMat, lumEnhanceImg);
	Mat scaleMat(matSize, CV_32FC1);
	divide(lumEnhanceImg, luminanceImg, scaleMat, 1.0, scaleMat.type());

	//以scale增强图像的每个颜色通道
	vector<Mat> dstChannels(3);
	split(dstImg, dstChannels);
	for (int i = 0; i < 3; i++){
		multiply(dstChannels[i], scaleMat, dstChannels[i], 1.0, dstChannels[i].type());
	}
	merge(dstChannels, dstImg);
}

Mat DeHazeModelLi::getConfidenceImg1(float k1)
{
	Mat confidence1(srcImg.size(), CV_32FC1);
	//计算亮度图像
	getLuminanceImg();
	//计算背景亮度
	Mat backgroudMat(luminanceImg.size(), luminanceImg.type());
	boxFilter(luminanceImg, backgroudMat, -1, Size(15, 15), Point(-1, -1), true, BORDER_DEFAULT);
	associativeBiFilter_8u(backgroudMat, luminanceImg, backgroudMat, 40, 7, cv::BORDER_DEFAULT);
	//计算亮度变化及其对应JND曲线背景亮度为127时的亮度变化
	Mat vMat(abs(luminanceImg - backgroudMat));
	vMat.convertTo(vMat, CV_32FC1);
	Mat vbestMat(luminanceImg.size(), CV_32FC1);
	divide(vMat, backgroudMat, vbestMat, 127, CV_32FC1);

	/*imshow("back", backgroudMat);
	imshow("lum",  luminanceImg);
	imshow("vMat", vMat);
	imshow("vbestMat", vbestMat);*/

	//计算基于亮度变化的置信度
	float JNDmin = 3;
	divide(JNDmin - vbestMat, k1, confidence1, 1.0, CV_32FC1);
	cv::exp(confidence1, confidence1);
	divide(1.0, 1 + confidence1, confidence1, 1.0, CV_32FC1);
	return confidence1;
}
Mat DeHazeModelLi::getConfidenceImg2(float coeOfk2)
{
	Mat confidence2(srcImg.size(), CV_32FC1);
	float Dmax = getDarkMax();
	float k2 = Dmax / coeOfk2;

	Mat floatDarkImg(darkImg.size(), CV_32FC1);
	darkImg.convertTo(floatDarkImg, CV_32FC1);
	divide(floatDarkImg - Dmax, k2, confidence2, 1.0, CV_32FC1);
	exp(confidence2, confidence2);
	divide(1.0, 1 + confidence2, confidence2);
	return confidence2;
}
uchar DeHazeModelLi::getDarkMax()
{
	uchar maxV = 0;
	uchar* data = darkImg.data;
	int size = darkImg.size().area();
	for (int i = 0; i < size; i++)
		maxV = max(maxV, data[i]);
	return maxV;
}

void DeHazeModelLi::getMinImg()
{
	minImg.create(srcImg.size(), CV_8UC1);
	//遍历行
	uchar* srcData = srcImg.data;
	uchar* minData = minImg.data;
	int srcCn = srcImg.channels();
	int srcStep = srcImg.step;
	int minStep = minImg.step;

	for (int i = 0; i < minImg.rows; ++i) {
		for (int j = 0; j < minImg.cols; ++j) {
			uchar pixel = 255;
			pixel = min(srcData[srcStep*i + j*srcCn], pixel);
			pixel = min(srcData[srcStep*i + j*srcCn + 1], pixel);
			pixel = min(srcData[srcStep*i + j*srcCn + 2], pixel);
			minData[i*minStep + j] = pixel;
		}
	}
}
void DeHazeModelLi:: getLuminanceImg()
{
	luminanceImg.create(srcImg.size(), CV_8UC1);
	//遍历行
	uchar* srcData = srcImg.data;
	uchar* lumData = luminanceImg.data;
	int srcCn = srcImg.channels();
	int srcStep = srcImg.step;
	int lumStep = luminanceImg.step;

	for (int i = 0; i < luminanceImg.rows; ++i) {
		for (int j = 0; j < luminanceImg.cols; ++j) {
			uchar pixel = 0;
			pixel = max(srcData[srcStep*i + j*srcCn], pixel);
			pixel = max(srcData[srcStep*i + j*srcCn + 1], pixel);
			pixel = max(srcData[srcStep*i + j*srcCn + 2], pixel);
			lumData[i*lumStep + j] = pixel;
		}
	}
}

void DeHazeModelLi::getBackgroundImg()
{
	backgroundImg.create(luminanceImg.size(), luminanceImg.type());
	boxFilter(luminanceImg, backgroundImg, -1, Size(15, 15), Point(-1, -1), true, BORDER_DEFAULT);
	associativeBiFilter_8u(backgroundImg, luminanceImg, backgroundImg, 40, 7, cv::BORDER_DEFAULT);
}
void DeHazeModelLi::getPostLumAjustImg()
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
	divide(temp, (maxv - minv), temp, 255.0, temp.type());
	temp.convertTo(lumChannel, lumChannel.type());
	//imshow("post", lumChannel);
	merge(channels, adjustDstImg);
	cvtColor(adjustDstImg, dstImg, CV_HSV2BGR);
}