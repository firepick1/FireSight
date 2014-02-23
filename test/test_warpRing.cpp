#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "FireSight.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "MatUtil.hpp"
#include "test.hpp"

using namespace cv;
using namespace std;
using namespace FireSight;


extern void generate_ringMat();

void test_matRing() {
	Scalar data = Scalar::all(255);
	Mat image;
	Mat result;
	image = Mat(1,1,CV_8U,data);
	matRing(image, result);
	assert(result.rows == 1 && result.cols == 1);
	assert(result.at<uchar>(0,0) == 255);

	cout << "-----------matRing input 8x8@255 with (0,0,4,4)@0" << endl;
	image = Mat(8,8,CV_8U,data);
	image(Rect(0,0,4,4)) = Scalar::all(0);
	cout << matInfo(image) << endl << image << endl;

	cout << "-----------matRing ring 8x8@255 with (0,0,4,4)@0" << endl;
	matRing(image, result);
	cout << matInfo(result) << endl << result << endl;
	assert(result.rows == 10 && result.cols == 10);
	uchar expected8x8[10][10] = { 
			0,   0, 191, 191, 191, 191, 191, 191,   0,   0,
			0, 191, 191, 191, 191, 191, 191, 191, 191,   0,
		191, 191, 191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191, 191, 191,
			0, 191, 191, 191, 191, 191, 191, 191, 191,   0,
			0,   0, 191, 191, 191, 191, 191, 191,   0,   0
	};
	assert(countNonZero(Mat(10,10,CV_8U,expected8x8) != result) == 0);

	cout << "-----------matRing input 9x9@255 with (0,0,4,4)@0" << endl;
	image = Mat(9,9,CV_8U,data);
	image(Rect(0,0,4,4)) = Scalar::all(0);
	cout << matInfo(image) << endl << image << endl;

	cout << "-----------matRing ring 9x9@255 with (0,0,4,4)@0" << endl;
	matRing(image, result);
	cout << matInfo(result) << endl << result << endl;
	assert(result.rows == 11 && result.cols == 11);
	uchar expected9x9[11][11] = { 
			0,   0, 191, 191, 191, 191, 191, 191, 191,   0,   0,
			0, 191, 191, 202, 202, 202, 202, 202, 191, 191,   0,
		191, 191, 202, 204, 204, 204, 204, 204, 202, 191, 191,
		191, 202, 204, 207, 207, 207, 207, 207, 204, 202, 191,
		191, 202, 204, 207, 223, 223, 223, 207, 204, 202, 191,
		191, 202, 204, 207, 223, 255, 223, 207, 204, 202, 191,
		191, 202, 204, 207, 223, 223, 223, 207, 204, 202, 191,
		191, 202, 204, 207, 207, 207, 207, 207, 204, 202, 191,
		191, 191, 202, 204, 204, 204, 204, 204, 202, 191, 191,
			0, 191, 191, 202, 202, 202, 202, 202, 191, 191,   0,
			0,   0, 191, 191, 191, 191, 191, 191, 191,   0,   0
	};
	assert(countNonZero(Mat(11,11,CV_8U,expected9x9) != result) == 0);

	cout << "-----------matRing input 3x3" << endl;
	image = Mat(Matx<uchar,3,3>(9,3,7,1,1,7,3,5,1));
	cout << matInfo(image) << endl << image << endl;

	cout << "-----------matRing ring 3x3" << endl;
	matRing(image, result);
	cout << matInfo(result) << endl << result << endl;
	assert(result.rows == 3 && result.cols == 3);
	uchar expected3x3[3][3] = {
		5,5,5,
		5,1,5,
		5,5,5
	};
	assert(countNonZero(Mat(3,3,CV_8U,expected3x3) != result) == 0);

	cout << "-----------matRing input 7x9" << endl;
	image = Mat(7,9,CV_8U,Scalar::all(50));
	image(Rect(1,1,7,5)) = Scalar::all(100);
	image(Rect(2,2,5,3)) = Scalar::all(150);
	image(Rect(3,3,3,1)) = Scalar::all(200);
	cout << "image:" << matInfo(image) << endl << image << endl;

	cout << "-----------matRing ring 7x9" << endl;
	matRing(image, result);
	cout << matInfo(result) << endl << result << endl;
	assert(result.rows == 9 && result.cols == 9);
	uchar expected7x9[9][9] = {
		 0,  0, 50, 50, 50, 50, 50,  0,  0,
		 0, 50, 75, 75, 75, 75, 75, 50,  0,
		50, 75,119,119,119,119,119, 75, 50,
		50, 75,119,163,163,163,119, 75, 50,
		50, 75,119,163,200,163,119, 75, 50,
		50, 75,119,163,163,163,119, 75, 50,
		50, 75,119,119,119,119,119, 75, 50,
		 0, 50, 75, 75, 75, 75, 75, 50,  0,
		 0,  0, 50, 50, 50, 50, 50,  0,  0
	};
	assert(countNonZero(Mat(9,9,CV_8U,expected7x9) != result) == 0);

	cout << "-----------matRing input 10x12" << endl;
	uchar data10x12[10][12] = {
		0,  0, 50, 50, 50, 50, 50, 50, 50, 50,  0,  0,
		0, 50, 50,100,100,100,100,100,100, 50, 50,  0,
		0, 50,100,150,150,150,150,150,150,100, 50,  0,
		0, 50,100,150,200,200,200,200,150,100, 50,  0,
		0, 50,100,150,200,250,250,200,150,100, 50,  0,
		0, 50,100,150,200,250,250,200,150,100, 50,  0,
		0, 50,100,150,200,200,200,200,150,100, 50,  0,
		0, 50,100,150,150,150,150,150,150,100, 50,  0,
		0, 50, 50,100,100,100,100,100,100, 50, 50,  0,
		0,  0, 50, 50, 50, 50, 50, 50, 50, 50,  0,  0,
	};
	image = Mat(10,12,CV_8U,data10x12);
	cout << "image:" << matInfo(image) << endl << image << endl;

	cout << "-----------matRing ring 10x12" << endl;
	matRing(image, result);
	cout << matInfo(result) << endl << result << endl;
	assert(result.rows == 16 && result.cols == 16);
	uchar expected10x12[16][16] = {
		0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
		0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
		0,  0,  0,  0, 14, 14, 14, 14, 14, 14, 14, 14,  0,  0,  0,  0,
		0,  0,  0, 14, 14, 50, 50, 50, 50, 50, 50, 14, 14,  0,  0,  0,
		0,  0, 14, 14, 50,100,100,100,100,100,100, 50, 14, 14,  0,  0,
		0,  0, 14, 50,100,150,150,150,150,150,150,100, 50, 14,  0,  0,
		0,  0, 14, 50,100,150,200,200,200,200,150,100, 50, 14,  0,  0,
		0,  0, 14, 50,100,150,200,250,250,200,150,100, 50, 14,  0,  0,
		0,  0, 14, 50,100,150,200,250,250,200,150,100, 50, 14,  0,  0,
		0,  0, 14, 50,100,150,200,200,200,200,150,100, 50, 14,  0,  0,
		0,  0, 14, 50,100,150,150,150,150,150,150,100, 50, 14,  0,  0,
		0,  0, 14, 14, 50,100,100,100,100,100,100, 50, 14, 14,  0,  0,
		0,  0,  0, 14, 14, 50, 50, 50, 50, 50, 50, 14, 14,  0,  0,  0,
		0,  0,  0,  0, 14, 14, 14, 14, 14, 14, 14, 14,  0,  0,  0,  0,
		0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
		0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
	};
	assert(countNonZero(Mat(16,16,CV_8U,expected10x12) != result) == 0);
}

