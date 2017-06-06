#include "forPaper.h"
#include "guidedfilter.h"
#include <vector>
using namespace std;



void ForPaper::dehaze(string srcPath, bool isStore, string dstPath)
{
	//读取图片
	DeHazeModelRefine::readImg(srcPath);
	//imshow("test", srcImg);
	waitKey(0);
	//计算暗通道
	DeHazeModelRefine::getDarkChannelImg(7);
	////showImage("dark", ImgDehaze::DARK);
	//getMinImg();
	////showImage("min", ImgDehaze::MIN);
	//getAFDarkImg(40, 150);
	darkImg.convertTo(darkImg, CV_32FC1);
	darkImg = guidedFilter(srcImg, darkImg, 40, 10);
	darkImg.convertTo(darkImg, CV_8UC1);
	//showImage("dark", ImgDehaze::DARK);

	//计算大气光
	DeHazeModelRefine::getAtmosphericLight();
	DeHazeModelRefine::getLightChannelImg(7);
	DeHazeModelRefine::lightImg.convertTo(DeHazeModelRefine::lightImg, CV_32FC1);
	DeHazeModelRefine::lightImg = guidedFilter(srcImg, DeHazeModelRefine::lightImg, 40, 10);

	//计算暗通道置信度
	DeHazeModelRefine::getConfidenceImg(8, 8.0);
	DeHazeModelRefine::getConfidenceImgSat(16);
	//confidenceImg = guidedFilter(srcImg, confidenceImg, 40, 10);
	//showImage("ConfidenceMat", CONFIDENCE);
	//计算透射率
	DeHazeModelRefine::getTranImg();
	//showImage("tran", ImgDehaze::TRAN);
	//计算恢复图像
	DeHazeModelRefine::getRestoredImg();

	imwrite(srcPath+"a.bmp", DeHazeModelRefine::dstImg);
	//进行后置增强
	//DeHazeModelRefine::getPostEnhancedImg();
	//showImage("resEnhance", ImgDehaze::DST);
	//writeImg("resEnhanced_" + dstPath);
	//进行HSV空间的亮度映射
	//DeHazeModelRefine::getPostLumAjustImg();

	waitKey(0);
	imwrite(srcPath+"b.jpg", DeHazeModelRefine::dstImg);
	//tranImg = tranImg * 256;
	//confidenceImg = confidenceImg * 256;
	//imwrite("Hec.jpg", DeHazeModelLi::confidenceImg);
	//imwrite("Het.jpg", DeHazeModelLi::tranImg);
	//imwrite("Hed.bmp", DeHazeModelLi::dstImg);
	
}

void  ForPaper::getDarkPic(string srcPath)
{
	//读取图片
	readImg(srcPath);
	//计算暗通道
	DeHazeModelRefine::getDarkChannelImg(10);
	//showImage("dark", ImgDehaze::DARK);
	//getMinImg();

	imwrite("res_dark.jpg",darkImg);
	//imwrite("res_min.jpg", minImg);
}

void ForPaper::showSaturation(string srcPath)
{
	vector<Mat> channels;
	Mat src = imread(srcPath);
	imshow("src", src);
	Mat hlsSrc = src.clone();
	cvtColor(src, hlsSrc, CV_BGR2HLS);
	split(hlsSrc, channels);
	Mat &hue = channels[0];
	Mat &lumChannel = channels[1];
	Mat &saturation = channels[2];
	imshow("l", lumChannel);
	imshow("s", saturation);
	//saturation *= 3;
	//saturation = maxFilter(saturation, 10);
	saturation = guidedFilter(src, saturation, 40, 10);
	imshow("ss", saturation);
	/*double minv = 0, maxv = 0;
	minMaxLoc(lumChannel, &minv, &maxv);*/

	Mat temp(lumChannel.size(), CV_32FC1);

	//imshow("post", lumChannel);
	merge(channels, hlsSrc);
	cvtColor(hlsSrc, src, CV_HLS2BGR);
	imshow("srca", src);
	waitKey(0);
}

Mat ForPaper::maxFilter(Mat &srcImg, int darkRadius)
{
	Mat dst(srcImg.size(),srcImg.type());
	uchar* Data = dst.data;
	//边界填充255；
	Mat tempImg;
	copyMakeBorder(srcImg, tempImg, darkRadius, darkRadius, darkRadius, darkRadius, BORDER_CONSTANT, Scalar(0, 0, 0));

	//获取参数
	int cn = 1;
	int d = darkRadius * 2 + 1;
	vector<int> offset( d*d );
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

void ForPaper::showSaturation(Mat src)
{
	vector<Mat> channels;

	//imshow("src", src);
	Mat hlsSrc = src.clone();
	cvtColor(src, hlsSrc, CV_BGR2HLS);
	split(hlsSrc, channels);
	Mat &hue = channels[0];
	Mat &lumChannel = channels[1];
	Mat &saturation = channels[2];
	imshow("s", saturation);
	//saturation = maxFilter(saturation, 10);
	//imshow("ss", saturation);
	/*double minv = 0, maxv = 0;
	minMaxLoc(lumChannel, &minv, &maxv);*/

	//imshow("post", lumChannel);
	//merge(channels, hlsSrc);
	//cvtColor(hlsSrc, src, CV_HLS2BGR);
	//imshow("srca", src);
	waitKey(0);
}