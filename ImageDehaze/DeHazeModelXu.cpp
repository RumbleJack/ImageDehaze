#include "DeHazeModelXu.h"
#include "guidedfilter.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
using namespace std;
using namespace cv;
void DeHazeModelXu::dehaze(string srcPath, bool isStore, string dstPath)
{
	//��ȡͼƬ
	readImg(srcPath);
	//���㰵ͨ��
	getDarkChannelImg(7);
	darkImg.convertTo(darkImg, CV_32FC1);
	darkImg = guidedFilter(srcImg, darkImg, 40, 10);
	darkImg.convertTo(darkImg, CV_8UC1);

	getAtmosphericLight();
	getLightChannelImg(7);
	lightImg.convertTo(lightImg, CV_32FC1);
	lightImg = guidedFilter(srcImg, lightImg, 40, 10);
	getTranImg();
	//����ָ�ͼ��
	getRestoredImg();
	if (isStore == true)
		writeImg(dstPath + "XuRes_");

	//imshow("XuDark", darkImg);
	//showImage("XuLight", LIGHT);
	//showImage("XuTran", TRAN);
	//showImage("XuRes", DST);
	//waitKey(0);
}

void DeHazeModelXu::gFilter(int guidedRadius, float eps) {
	//maxFilter(tran, cv::Size(15, 15));
	//�����˲������˲��뾶������Ӱ��ϵ�� a �ľֲ������С������������У�ͼ���Ե�仯Խ��a��ֵ���Խ�󣬶Ա�Ե�����Ч��Խǿ
	//eps ��ȡֵ���Ƕ�ͼ���Ե�仯�̶ȵ��ж�ָ�꣬epsֵԽС���Բ����Եı�Ե����Ч��Խ�ã���Ե�Ҳ���ܳ��ֲ���Ҫ����������� 
	tranImg = guidedFilter(srcImg, tranImg, guidedRadius * 2 + 1, eps);
}
void DeHazeModelXu::getDarkChannelImg(int radius)
{
	DeHazeModel::getDarkChannelImg(radius);
}
void DeHazeModelXu::getLightChannelImg(int radius )
{
	lightImg.create(srcImg.size(), CV_8UC1);

	uchar* lightData = (lightImg.data);
	//�߽����255��
	Mat tempImg;
	copyMakeBorder(srcImg, tempImg, radius, radius, radius, radius, BORDER_CONSTANT, Scalar(0, 0, 0));

	//��ȡ����
	int cn = 3;
	int d = radius * 2 + 1;
	vector<int> offset(d*d);
	int * _offset = &offset[0];
	//����ƫ����
	int sizeOfTP = 0;
	for (int i = -radius; i <= radius; i++)
		for (int j = -radius; j <= radius; j++)
		{
			/* ȡ����ע�ͺ�maskΪԲ��
			if (std::sqrt(i*i + j*j) > radius)
			continue;*/
			_offset[sizeOfTP++] = tempImg.step*i + j*cn;
		}
	//������
	for (int i = 0; i < lightImg.rows; ++i) {
		uchar* imgData = tempImg.ptr<uchar>(i + radius) + radius*cn;
		//������
		for (int j = 0; j < lightImg.cols; ++j) {
			uchar pixel = 0;
			for (int index = 0; index < sizeOfTP; index++){
				pixel = max(imgData[j*cn + _offset[index]], pixel);
				pixel = max(imgData[j*cn + _offset[index] + 1], pixel);
				pixel = max(imgData[j*cn + _offset[index] + 2], pixel);
			}
			lightData[i*lightImg.step+ j] = pixel;
		}
	}
	tempImg.release();
	//float atmAve = (atmosphericLight[0] + atmosphericLight[1] + atmosphericLight[2]) / 3;
	//addWeighted(lightImg,0.7,atmAve,0.25, 0, lightImg, CV_8UC1);
}
void DeHazeModelXu::getTranImg()
{
	float atmAve = (atmosphericLight[0] + atmosphericLight[1] + atmosphericLight[2]) / 3;
	float a = 0.7;
	float b = 0.25;

	cv::multiply(darkImg, 1, tranImg, 1.0 , CV_32FC1);
	divide(tranImg, (a*lightImg + b*atmAve), tranImg, 0.95, CV_32FC1);
	cv::subtract(1.0, tranImg, tranImg, noArray(), CV_32FC1);
	tranImg = abs(tranImg);
}

void DeHazeModelXu::getRestoredImg()
{
	vector<Mat> srcChannels;
	vector<Mat> dstChannels;
	split(srcImg, srcChannels);
	split(dstImg, dstChannels);

	vector<Mat> tempChannels1(3);
	vector<Mat> tempChannels2(3);
	float a = 0.7;
	float b = 0.25;
	for (int i = 0; i < 3; i++)
	{
		tempChannels1[i].create(srcChannels[i].size(), CV_32FC1);
		tempChannels2[i].create(dstChannels[i].size(), CV_32FC1);

		srcChannels[i].convertTo(tempChannels1[i], CV_32FC1);

		divide(tempChannels1[i] - (a*lightImg+b*atmosphericLight[i]), max(tranImg,0.1), tempChannels2[i], 1, CV_32FC1);
		add((a*lightImg + b*atmosphericLight[i]), tempChannels2[i], dstChannels[i], noArray(), CV_8UC1);
	}
	merge(dstChannels, dstImg);
}
void DeHazeModelXu::showImage(string winName, MatType mt)
{
	DeHazeModel::showImage(winName, mt);
	Mat temp(srcImg.size(), CV_8UC1);
	switch (mt) {
	case LIGHT:
		lightImg.convertTo(temp, CV_8UC1);
		imshow(winName, temp);
		break;
	}
}