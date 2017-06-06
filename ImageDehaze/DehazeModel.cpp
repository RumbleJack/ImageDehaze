#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "BiFilter.h"
#include <vector>
#include <queue>
#include <functional>
#include <iostream>
#include "DeHazeModel.h"
using namespace std;
using namespace cv;

DeHazeModel::DeHazeModel()
{
	srcImg.create(Size(1,1),CV_8UC1);
	dstImg = srcImg.clone();
	darkImg.create(srcImg.rows, srcImg.cols, CV_8UC1);
	tranImg.create(srcImg.rows, srcImg.cols, CV_32FC1);
}
DeHazeModel::DeHazeModel(string srcFilePath)
{
	readImg(srcFilePath);
}
DeHazeModel::~DeHazeModel()
{
	destroyAllWindows();
}

void DeHazeModel::readImg(string srcFilePath)
{
	destroyAllWindows();
	//Mat自动释放内存
	srcImg = imread(srcFilePath, IMREAD_COLOR);
	//resize(srcImg, srcImg, cv::Size(0,0), 0.3, 0.3, 1);
	dstImg = srcImg.clone();
	darkImg.create(srcImg.rows, srcImg.cols, CV_8UC1);
	tranImg.create(srcImg.rows, srcImg.cols, CV_32FC1);
}
void DeHazeModel::showImage(string winName, MatType mt )
{
	namedWindow(winName, WINDOW_AUTOSIZE);
	switch (mt) {
	case SRC:
		imshow(winName, srcImg);
		break;
	case DST:
		imshow(winName, dstImg);
		break;
	case DARK:
		imshow(winName, darkImg);
		break;
	case TRAN:
		imshow(winName, tranImg);
		break;
	}
}
void DeHazeModel::writeImg(string dstPath, MatType mt )
{
	dstPath += ".bmp";
	switch (mt) {
	case SRC:
		imwrite(dstPath,srcImg);
		break;
	case DST:
		imwrite(dstPath, dstImg);
		break;
	case DARK:
		imwrite(dstPath, darkImg);
		break;
	case TRAN:
		imwrite(dstPath, tranImg);
		break;
	default:
		imwrite("WrongMatType.jpg", srcImg);
	}
}
const Mat* DeHazeModel::getSrcImg() const
{
	return (&srcImg);
}
const Mat* DeHazeModel::getResImg() const
{
	return (&dstImg);
}
const Mat* DeHazeModel::getDarkImg() const
{
	return (&darkImg);
}

void DeHazeModel::getDarkChannelImg(int darkRadius)
{
	uchar* darkData = darkImg.data;
	//边界填充255；
	Mat tempImg;
	copyMakeBorder(srcImg, tempImg, darkRadius, darkRadius, darkRadius, darkRadius, BORDER_CONSTANT, Scalar(255, 255, 255));

	//获取参数
	int cn = 3;
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
	for (int i = 0; i < darkImg.rows; ++i) {
		uchar* imgData = tempImg.ptr<uchar>(i + darkRadius) + darkRadius*cn;
		//遍历列
		for (int j = 0; j < darkImg.cols; ++j) {
			uchar pixel = 255;
			for (int index = 0; index < sizeOfTP; index++){
				pixel = min(imgData[j*cn + _offset[index]], pixel);
				pixel = min(imgData[j*cn + _offset[index] + 1], pixel);
				pixel = min(imgData[j*cn + _offset[index] + 2], pixel);
			}
			darkData[i*darkImg.step + j] = pixel;
		}
	}
	tempImg.release();
}
void DeHazeModel::getAtmosphericLight()
{
	priority_queue<Pixel, vector<Pixel>, greater<Pixel> > pq;
	int num = darkImg.rows * darkImg.cols * 0.001;
	uchar *data = darkImg.data;
	int step = darkImg.step;

	for (int i = 0; i < darkImg.rows; ++i) {
		for (int j = 0; j < darkImg.cols; ++j) {
			Pixel p(data[i*step + j], i, j);
			pq.push(p);
		}
		while (pq.size() > num)
			pq.pop();
	}
	unsigned long long A[3] = { 0, 0, 0 };

	//取最亮的0.1%暗通道对应像素的平均值作为空气光
	while (!pq.empty()) {
		Pixel tmp = pq.top();
		Vec3b vcb = srcImg.at<Vec3b>(tmp.i, tmp.j);
		A[0] += vcb[0], A[1] += vcb[1], A[2] += vcb[2];
		pq.pop();
	}
	atmosphericLight[0] = A[0] / num;
	atmosphericLight[1] = A[1] / num;
	atmosphericLight[2] = A[2] / num;
	printf("Atmosphere Light:%d %d %d\n", atmosphericLight[0], atmosphericLight[1], atmosphericLight[2]);
}
void DeHazeModel::getTranImg()
{
	
	float atmAve = (atmosphericLight[0] + atmosphericLight[1] + atmosphericLight[2]) / 3;
	cv::multiply(darkImg, 1, tranImg,  0.95 / atmAve, CV_32FC1);
	cv::subtract(1.0, tranImg, tranImg, noArray(), CV_32FC1);
	tranImg = abs(tranImg);
	
}

void DeHazeModel::getRestoredImg()
{
	vector<Mat> srcChannels;
	vector<Mat> dstChannels;
	split(srcImg, srcChannels);
	split(dstImg, dstChannels);

	vector<Mat> tempChannels1(3);
	vector<Mat> tempChannels2(3);
	for (int i = 0; i < 3; i++)
	{
		tempChannels1[i].create(srcChannels[i].size(), CV_32FC1);
		tempChannels2[i].create(dstChannels[i].size(), CV_32FC1);
		srcChannels[i].convertTo(tempChannels1[i], CV_32FC1);
		divide(tempChannels1[i] - atmosphericLight[i], max(tranImg,0.1), tempChannels2[i], 1, CV_32FC1);
		add(atmosphericLight[i], tempChannels2[i], dstChannels[i], noArray(), CV_8UC1);
	}
	merge(dstChannels, dstImg);
}