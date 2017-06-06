#include "DeHazeModelMine.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "BiFilter.h"
#include "guidedfilter.h"

using namespace std;
using namespace cv;

void DeHazeModelMine::dehaze(string srcPath, bool isStore, string dstPath)
{
	//读取图片
	readImg(srcPath);
	//计算暗通道
	getDarkChannelImg(7);
	//showImage("dark", ImgDehaze::DARK);
	getMinImg();
	//showImage("min", ImgDehaze::MIN);
	getAFDarkImg(40, 150);
	//showImage("assDark", ImgDehaze::DARK);
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
	//showImage("res", DST);
	if (isStore == true)
		writeImg(dstPath + "MineRes_");
	//进行后置增强
	getPostEnhancedImg();
	//showImage("resEnhance", ImgDehaze::DST);
	//writeImg("resEnhanced_" + dstPath);
	//进行HSV空间的亮度映射
	getPostLumAjustImg();
	//showImage("resHLSMap", DST);
	if (isStore == true)
		writeImg(dstPath + "MineHLSMapRes_");
}
void DeHazeModelMine::getConfidenceImg(float k1, float coeOfk2)
{
	Mat c1 = getConfidenceImg1(k1);
	Mat c2 = getConfidenceImg2(coeOfk2);
	confidenceImg.create(srcImg.size(), CV_32FC1);

	//imshow("c1a", c1);
	Mat temp(c1.size(), c1.type());
	//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	//morphologyEx(c1, temp, MORPH_OPEN, element);
	//c1 = temp.clone();
	//imshow("c1b", c1);

	temp = guidedFilter(srcImg, c1, 40, 10);
	//bilateralFilter(c1, temp, 40, 120 / 256.0, 90, BORDER_DEFAULT);
	c1.release();
	c1 = temp.clone();
	//imshow("c1c", c1);
	//imshow("c2", c2);
	//waitKey(0);

	float* datac = reinterpret_cast<float*>(confidenceImg.data);
	float* datac1 = reinterpret_cast<float*>(c1.data);
	float* datac2 = reinterpret_cast<float*>(c2.data);
	int size = srcImg.size().area();
	for (int i = 0; i < size; i++)
		//datac[i] = max(datac1[i], datac2[i]);
		datac[i] = min(datac1[i] * datac2[i], static_cast<float>(0.8));
	//cout << confidenceImg;
}
void DeHazeModelMine::getAFDarkImg(int d, double sigma_color)
{
	Mat assImg(darkImg.size(), darkImg.type());
	associativeBiFilter_8u(darkImg, minImg, assImg, d, sigma_color, cv::BORDER_DEFAULT);
	darkImg = assImg;
}
void DeHazeModelMine::getPostLumAjustImg()
{
	vector<Mat> channels;
	Mat adjustDstImg = dstImg.clone();
	cvtColor(dstImg, adjustDstImg, CV_BGR2HLS);
	split(adjustDstImg, channels);
	Mat &lumChannel = channels[1];
	//imshow("pre", lumChannel);
	double minv = 0, maxv = 0;
	minMaxLoc(lumChannel, &minv, &maxv);

	Mat temp(lumChannel.size(), CV_32FC1);
	lumChannel.convertTo(temp, temp.type());
	divide(temp, (maxv - 0), temp, 255.0, temp.type());
	temp.convertTo(lumChannel, lumChannel.type());
	//imshow("post", lumChannel);
	merge(channels, adjustDstImg);
	cvtColor(adjustDstImg, dstImg, CV_HLS2BGR);
}
