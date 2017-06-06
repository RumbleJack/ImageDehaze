#include "DeHazeModelHe.h"
#include "guidedfilter.h"
#include "SoftMatting.h"

void DeHazeModelHe::dehaze(string srcPath, bool isStore, string dstPath)
{
	//��ȡͼƬ
	readImg(srcPath);

	//���㰵ͨ��
	getDarkChannelImg(7);
	//showImage("HeDark", DARK);
	//���������
	getAtmosphericLight();
	//����͸����
	getTranImg();

	gFilter(20, 10);
	//softMatting(1, 0.0000001);
	//showImage("HeTran", TRAN);
	//����ָ�ͼ��
	getRestoredImg();
	//showImage("HeRes", DST);
	if (isStore == true)
		writeImg(dstPath + "HeRes_");
}
void DeHazeModelHe::gFilter(int guidedRadius, float eps) {
	//maxFilter(tran, cv::Size(15, 15));
	//�����˲������˲��뾶������Ӱ��ϵ�� a �ľֲ������С������������У�ͼ���Ե�仯Խ��a��ֵ���Խ�󣬶Ա�Ե�����Ч��Խǿ
	//eps ��ȡֵ���Ƕ�ͼ���Ե�仯�̶ȵ��ж�ָ�꣬epsֵԽС���Բ����Եı�Ե����Ч��Խ�ã���Ե�Ҳ���ܳ��ֲ���Ҫ����������� 
	tranImg = guidedFilter(srcImg, tranImg, guidedRadius * 2 + 1, eps);
}
void DeHazeModelHe::softMatting(int radius, float eps)
{
	tranImg = ::softMatting(srcImg, tranImg, radius, eps, 0.0001);
}
