#include "DeHazeModelHe.h"
#include "guidedfilter.h"
#include "SoftMatting.h"

void DeHazeModelHe::dehaze(string srcPath, bool isStore, string dstPath)
{
	//读取图片
	readImg(srcPath);

	//计算暗通道
	getDarkChannelImg(7);
	//showImage("HeDark", DARK);
	//计算大气光
	getAtmosphericLight();
	//计算透射率
	getTranImg();

	gFilter(20, 10);
	//softMatting(1, 0.0000001);
	//showImage("HeTran", TRAN);
	//计算恢复图像
	getRestoredImg();
	//showImage("HeRes", DST);
	if (isStore == true)
		writeImg(dstPath + "HeRes_");
}
void DeHazeModelHe::gFilter(int guidedRadius, float eps) {
	//maxFilter(tran, cv::Size(15, 15));
	//引导滤波器，滤波半径决定了影响系数 a 的局部区块大小，在这个区块中，图像边缘变化越大，a的值相对越大，对边缘保存的效果越强
	//eps 的取值则是对图像边缘变化程度的判定指标，eps值越小，对不明显的边缘保存效果越好，相对的也可能出现不必要的问题和噪声 
	tranImg = guidedFilter(srcImg, tranImg, guidedRadius * 2 + 1, eps);
}
void DeHazeModelHe::softMatting(int radius, float eps)
{
	tranImg = ::softMatting(srcImg, tranImg, radius, eps, 0.0001);
}
