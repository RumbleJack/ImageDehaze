#include "BiFilter.h"


void biFilter_8u(const Mat& src, Mat& dst, int d, double sigma_color, double sigma_space, int borderType)
{
	//��֤������Ч��
	CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) && src.data != dst.data);
	if (sigma_color <= 0)
		sigma_color = 1;
	if (sigma_space <= 0)
		sigma_space = 1;

	//����˫���˲�����
	double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
	double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);
	int   radius;
	if (d <= 0)
		radius = cvRound(sigma_space*1.5);
	else
		radius = d / 2;
	radius = MAX(radius, 1);
	d = radius * 2 + 1;

	int cn = src.channels();
	Size size = src.size();
	int maxk = 0;		//ģ����������*ͨ����

	//���߽�
	Mat temp;
	copyMakeBorder(src, temp, radius, radius, radius, radius, borderType);

	std::vector<float> _color_weight(cn * 256);		
	std::vector<float> _space_weight(d*d);		
	std::vector<int> _space_ofs(d*d);			//���������ģ���������ص��ֽ�ƫ����
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];

	// initialize color-related bilateral filter coefficients
	for (int i = 0; i < 256 * cn; i++)
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	
	// initialize space-related bilateral filter coefficients
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if (r > radius)
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*temp.step + j*cn);
		}
	}
	CrossBilateralFilter body(dst, temp, radius, maxk, space_ofs, space_weight, color_weight);
	body.biFilter();
}

void associativeBiFilter_8u(const Mat& srcCS, const Mat& srcRef, Mat& dst, int d, double sigma_color, int borderType )
{
	//��֤������Ч��
	CV_Assert((srcCS.type() == CV_8UC1 && srcRef.type() == CV_8UC1) && srcRef.data != dst.data);
	if (sigma_color <= 0)
		sigma_color = 1;

	//����˫���˲�����
	double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
	int   radius;
	if (d <= 0)
		radius = cvRound(sigma_color*1.5);
	else
		radius = d / 2;
	radius = MAX(radius, 1);
	d = radius * 2 + 1;

	Size size = srcCS.size();
	int maxk = 0;		//ģ����������*ͨ����

	//���߽�
	Mat tempCS,tempRef;
	copyMakeBorder(srcCS, tempCS, radius, radius, radius, radius, borderType);
	copyMakeBorder(srcRef, tempRef, radius, radius, radius, radius, borderType);

	std::vector<float> _color_weight(256);
	std::vector<int>   _space_ofs(d*d);			//���������ģ���������ص��ֽ�ƫ����
	float* color_weight = &_color_weight[0];
	int* space_ofs = &_space_ofs[0];

	// initialize color-related bilateral filter coefficients
	for (int i = 0; i < 256 ; i++)
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);


	// initialize space-related bilateral filter coefficients
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			/*double r = std::sqrt((double)i*i + (double)j*j);
			if (r > radius)
				continue;*/
			space_ofs[maxk++] = (int)(i*tempCS.step + j);
		}
	}
	CrossBilateralFilter body(dst, tempCS, tempRef, radius, maxk, space_ofs, color_weight);
	body.associativeBiFilter();
}
void CrossBilateralFilter::biFilter()
{
	int cn = dest->channels();
	Size size = dest->size();
	
	//����ͼ��ÿһ��
	for (int i = 0; i < dest->rows; i++)
	{
		//����temp��չ��src��Ե����Ҫȡi+radius�У�radius�У��Զ�Ӧdest�ĵ�i����ʼλ��
		const uchar* sptr = temp->ptr(i + radius) + radius*cn;
		uchar* dptr = dest->ptr(i);

		if (cn == 1)
		{
			//����ͼ��ÿһ��
			for (int j = 0; j < size.width; j++)
			{
				float sum = 0, wsum = 0;
				//��ǰģ����������ֵ
				int val0 = sptr[j];
				for (int indexOfTp = 0; indexOfTp < maxk; indexOfTp++)
				{
					int val = sptr[j + space_ofs[indexOfTp]];
					float w = space_weight[indexOfTp] * color_weight[std::abs(val - val0)];
					sum += val*w;
					wsum += w;
				}
				// overflow is not possible here => there is no need to use cv::saturate_cast
				dptr[j] = (uchar)cvRound(sum / wsum);
			}
		}
		else
		{
			assert(cn == 3);
			for (int j = 0; j < size.width * 3; j += 3)
			{
				float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
				int b0 = sptr[j], g0 = sptr[j + 1], r0 = sptr[j + 2];

				for (int indexOfTp = 0; indexOfTp < maxk; indexOfTp++)
				{
					const uchar* sptr_k = sptr + j + space_ofs[indexOfTp];
					int b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
					float w = space_weight[indexOfTp] * color_weight[std::abs(b - b0) +
						std::abs(g - g0) + std::abs(r - r0)];
					sum_b += b*w; sum_g += g*w; sum_r += r*w;
					wsum += w;
				}
				wsum = 1.f / wsum;
				b0 = cvRound(sum_b*wsum);
				g0 = cvRound(sum_g*wsum);
				r0 = cvRound(sum_r*wsum);
				dptr[j] = (uchar)b0; dptr[j + 1] = (uchar)g0; dptr[j + 2] = (uchar)r0;
			}
		}
	}
}
void CrossBilateralFilter::associativeBiFilter()
{
	Size size = dest->size();
	//����ͼ��ÿһ��
	for (int i = 0; i < dest->rows; i++)
	{
		//����temp��չ��src��Ե����Ҫȡi+radius�У�radius�У��Զ�Ӧdest�ĵ�i����ʼλ��
		const uchar* sptr = temp->ptr(i + radius) + radius;
		const uchar* gptr = tempGuide->ptr(i + radius) + radius;
		uchar* dptr = dest->ptr(i);
		//����ͼ��ÿһ��
		for (int j = 0; j < size.width; j++)
		{
			float sum = 0, wsum = 0;
			//��ǰģ����������ֵ
			int val0 = sptr[j];
			for (int indexOfTp = 0; indexOfTp < maxk; indexOfTp++)
			{
				int val = gptr[j + space_ofs[indexOfTp]];
				float w = color_weight[std::abs(val - val0)];
				sum += val*w;
				wsum += w;
			}
			// overflow is not possible here => there is no need to use cv::saturate_cast
			dptr[j] = (uchar)cvRound(sum / wsum);
		}
	}
}