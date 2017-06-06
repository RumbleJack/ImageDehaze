#include "Softmatting.h"
#include "guidedfilter.h"
#include <armadillo> 

using namespace std;
using std::vector;
using arma::mat;

class SoftMattingImpl
{
public:
	virtual ~SoftMattingImpl() {}
	Mat matting(const Mat &p);
protected:
	int Idepth;
private:
	virtual Mat mattingSingleChannel(Mat &p) const = 0;
};

class SoftMattingColor : public SoftMattingImpl
{
public:
	SoftMattingColor(const Mat &I, int r = 1 , double eps = 0.0000001, double lamda = 0.0001);

private:
	virtual Mat mattingSingleChannel(Mat &p) const;

private:
	int r;
	double eps;
	double lamda;
	//������˹����
	arma::sp_fmat laplacian;
};

SoftMattingColor::SoftMattingColor(const Mat &origI, int r, double eps, double lamda) : r(r), eps(eps), lamda(lamda)
{
	//����ͼ������
	uint cols = origI.cols;
	uint rows = origI.rows;
	uint imgSize = cols * rows;
	uint windowSize = std::pow((r * 2 + 1) , 2);
	
	//��opencvMat ת��Ϊarmadillo::fmat
	arma::fmat slices[3];
	arma::fcube imgI(cols, rows, 3);
	Mat imgData = origI.clone();
	imgData.convertTo(imgData, CV_32FC3);
	vector<Mat> Ichannels;
	cv::split(imgData, Ichannels);
	for (int i = 0; i < Ichannels.size(); i++)
	{
		slices[i] = arma::fmat( (float*)(Ichannels[i].data), rows, cols);
		imgI.slice(i) = slices[i].t();
	}
	
	//��������
	//arma::u64 *indicesData = new arma::u64[imgSize];
	uint *indicesData = new uint[imgSize];
	for (int i = 0; i < imgSize; i++)
		indicesData[i] = i;
	arma::umat indices(indicesData, cols, rows);

	//����ϡ��������ݼ�¼
	int vectorSize = windowSize*windowSize*imgSize;
	int curLen = 0;
	arma::umat locations(2,vectorSize,arma::fill::zeros);
	arma::fvec vals(vectorSize, arma::fill::zeros);
	
	//�������п��ܴ���
	for (int k = 0; k < imgSize; k++)
	{
		//������ת��Ϊ�±�
		arma::uvec sub = ind2sub(size(indices), k);

		//��ֹĿ������Խ��
		uint r_min = std::max(0, (int)sub(0) - r);
		uint r_max = std::min(cols - 1, sub(0) + r);
		uint c_min = std::max(0, (int)sub[1] - r);
		uint c_max = std::min(rows - 1, sub[1] + r);

		//��ǰ��������
		arma::umat indsWin = indices(arma::span(r_min, r_max), arma::span(c_min, c_max));
		arma::uvec indsVec = arma::vectorise(indsWin);
		uint  indsVecSize = indsVec.size();

		//ͼ�񴰿�
		arma::fcube imageWin = imgI.tube(arma::span(r_min, r_max), arma::span(c_min, c_max));
		imageWin.reshape(indsVecSize, 3, 1);
		arma::fmat mImgWin(imageWin.slice(0));

		//���㴰�����ؾ�ֵ
		arma::fmat imgWinMean = arma::mean(mImgWin, 0);
		//���㴰������Э����
		arma::fmat imgWinVar;
		try{
			imgWinVar = inv((mImgWin.t() * mImgWin / indsVecSize) - (imgWinMean.t() * imgWinMean) +
				(eps / indsVecSize * arma::eye<arma::fmat>(3, 3)));
		}
		catch (exception e)
		{
			continue;
		}
		//ԭ����ֵ���ֵ�Ĳ�
		arma::fmat imgWinDiff = mImgWin - repmat(imgWinMean, indsVecSize, 1);
		arma::fmat imgWinVals = arma::eye<arma::fmat>(indsVecSize, indsVecSize) 
						- (1 + imgWinDiff * imgWinVar * imgWinDiff.t()) / indsVecSize;

		//����ϡ�����
		int subLen = indsVecSize * indsVecSize;
		arma::umat winInds = repmat(indsVec, 1, indsVecSize);
		locations.row(0).subvec(curLen, curLen + subLen - 1) = arma::vectorise(winInds).t();
		winInds = winInds.t();
		locations.row(1).subvec(curLen, curLen + subLen - 1) = arma::vectorise(winInds).t();
		vals.subvec(curLen, curLen + subLen - 1) = arma::vectorise(imgWinVals); 
		curLen += subLen;
	}

	laplacian = arma::sp_fmat(true, locations, vals, imgSize, imgSize);
}
Mat SoftMattingColor::mattingSingleChannel( Mat &p) const
{
	p.convertTo(p, CV_32FC1);
	arma::fmat trans_est((float*)(p.data), p.rows, p.cols);
	trans_est = trans_est.t();

	arma::sp_fmat A = laplacian + lamda * arma::speye<arma::sp_fmat>(size(laplacian));
	arma::fmat b = lamda * arma::vectorise(trans_est);
	arma::fmat x = spsolve(A, b, "lapack");
	trans_est = reshape(x, p.cols, p.rows);
	trans_est = trans_est.t();

	p.data = (uchar*)(&trans_est[0]);
	return p;  // Eqn. (16) in the paper;
}

Mat SoftMattingImpl::matting(const Mat &p)
{
	Mat p2 = p;

	Mat result;
	if (p.channels() == 1)
	{
		result = mattingSingleChannel(p2);
	}
	else
	{
		std::vector<Mat> pc;
		cv::split(p2, pc);
		for (std::size_t i = 0; i < pc.size(); ++i)
			pc[i] = mattingSingleChannel(pc[i]);
		cv::merge(pc, result);
	}
	return result;
}
Mat SoftMatting::matting(const Mat &p) const
{
	return impl_->matting(p);
}

SoftMatting::SoftMatting(const Mat &I, int r, double eps, double lamda)
{
	CV_Assert(I.channels() == 3);
	impl_ = new SoftMattingColor(I, r, eps);
}
SoftMatting::~SoftMatting()
{
	delete impl_;
}
Mat softMatting(const Mat &I, const Mat &p, int r, double eps, double lamda)
{
 return SoftMatting(I, r, eps).matting(p);
}
