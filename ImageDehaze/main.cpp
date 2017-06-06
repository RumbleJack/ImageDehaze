#pragma warning(disable:4996)
#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "DeHazeModelHe.h"
#include "DeHazeModelLi.h"
#include "DeHazeModelXu.h"
#include "DeHazeModelMine.h"
#include "DeHazeModelRefine.h"
#include "forPaper.h"
using namespace std;
using namespace cv;

const string srcPathName = "paper/cc.bmp";
const string srcPathFileName = "res/resPath.txt";
void dehazeOne(string filename);
void dehazeAll(string filename);
void testHe(string filename);
void getpic();
int main()
{
	//getpic();
	//testHe(srcPathName);
	dehazeOne(srcPathName);
	//dehazeAll(srcPathFileName);
	system("pause");
}

void getpic()
{
	ForPaper fp;
	fp.dehaze(srcPathName);
	//fp.showSaturation(srcPathName);
	//fp.showSaturation("paper/gugong.bmpa.jpg");
}
void dehazeAll(string filename)
{
	DeHazeModel* pDHModelRefine = new DeHazeModelRefine();
	DeHazeModel* pDHModelMine = new DeHazeModelMine();
	DeHazeModel* pDHModelHe = new DeHazeModelHe();
	DeHazeModel* pDHModelLi = new DeHazeModelLi();
	DeHazeModel* pDHModelXu = new DeHazeModelXu();

	//clock_t start = clock();
	//clock_t end = clock(); 
	//printf("所用时间 :%fs\n ", (float)(end - start) / CLOCKS_PER_SEC);
	vector<string> pathnames;
	ifstream inf(filename);
	string  tmpPathin, tmpPathout;
	char* tmpname = new char[120];
	int i = 0;
	while (getline(inf, tmpPathin))
	{
		sprintf(tmpname, "D:/Users/jackren/documents/visual studio 2013/Projects/ImageDehaze/ImageDehaze/res/resultRefine/%d", i++);
		tmpPathout = string(tmpname);
		pDHModelRefine->dehaze(tmpPathin, true, tmpPathout);

		//pDHModelXu->dehaze(tmpPathin, true, tmpPathout);
		//pDHModelHe->dehaze(tmpPathin, true, tmpPathout);
		//pDHModelLi->dehaze(tmpPathin, true, tmpPathout);
		//pDHModelMine->dehaze(tmpPathin, true, tmpPathout);
	}
}

void dehazeOne(string filename)
{
	DeHazeModel* pDHModelRefine = new DeHazeModelRefine();
	DeHazeModel* pDHModelMine = new DeHazeModelMine();
	DeHazeModel* pDHModelHe = new DeHazeModelHe();
	DeHazeModel* pDHModelLi = new DeHazeModelLi();
	DeHazeModel* pDHModelXu = new DeHazeModelXu();

	pDHModelXu->dehaze(filename);
	pDHModelHe->dehaze(filename);
	pDHModelLi->dehaze(filename);
	pDHModelMine->dehaze(filename);
	pDHModelRefine->dehaze(filename);

	const Mat* pSrcImg = pDHModelXu->getSrcImg();
	const Mat* pXuResImg = pDHModelXu->getResImg();
	const Mat* pHeResImg = pDHModelHe->getResImg();
	const Mat* pLiResImg = pDHModelLi->getResImg();
	const Mat* pMineResImg = pDHModelMine->getResImg();
	const Mat* pRefineResImg = pDHModelRefine->getResImg();

	imshow("srcImg", *pSrcImg);
	imshow("pXuRes", *pXuResImg);
	imshow("pHeResImg", *pHeResImg);
	imshow("pLiResImg", *pLiResImg);
	imshow("pMineResImg", *pMineResImg);
	imshow("pRefineResImg", *pRefineResImg);

	imwrite("1.bmp", *pXuResImg);
	imwrite("2.bmp", *pHeResImg);
	imwrite("3.bmp", *pLiResImg);
	imwrite("4.bmp", *pRefineResImg);

	//const Mat* pXuDarkImg = pDHModelXu->getDarkImg();
	//const Mat* pLiDarkImg = pDHModelLi->getDarkImg();
	//imshow("pXuDarkImg", *pXuDarkImg);
	//imshow("pLiDarkImg", *pLiDarkImg);
	waitKey(0);
}

void testHe(string filename)
{
	DeHazeModel* pDHModelRefine = new DeHazeModelRefine();
	char *tmpname = new char[200];
	
	sprintf(tmpname, "D:/Users/jackren/documents/visual studio 2013/Projects/ImageDehaze/ImageDehaze/res/%d", 1);
	pDHModelRefine->dehaze(filename, true, string(tmpname));
	const Mat* pSrcImg = pDHModelRefine->getSrcImg();
	const Mat* pHeResImg = pDHModelRefine->getResImg();
	imshow("srcImg", *pSrcImg);
	imshow("pResImg", *pHeResImg);
	waitKey(0);
}