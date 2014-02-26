#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "FireSight.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "MatUtil.hpp"

using namespace cv;
using namespace std;
using namespace firesight;


void test_warpAffine() {
	Mat result;

	cout << "-------------------test_warpAffine 7x9 --------" << endl;
	Point center7x9(4,3);
	uchar data7x9[][9] = {
		11,12,13,14,15,16,17,18,19,
		21,22,23,24,25,26,27,28,29,
		31,32,33,34,35,36,37,38,39,
		41,42,43,44,45,46,47,48,49,
		51,52,53,54,55,56,57,58,59,
		61,62,63,64,65,66,67,68,69,
		71,72,73,74,75,76,77,78,79
	};
	Mat image7x9(7,9,CV_8U,data7x9);
	cout << matInfo(image7x9) << endl << image7x9 << endl;

	cout << "-------------------test_warpAffine 7x9 90 degrees --------" << endl;
	matWarpAffine(image7x9, result, center7x9, 90, 1, Point(0,0), Size(0,0));
	cout << "90 " << matInfo(result) << endl << result << endl;
	assert(result.rows == 9 && result.cols == 7);
	uchar expected7x9_90[9][7] = {
		19, 29, 39, 49, 59, 69, 79,
		18, 28, 38, 48, 58, 68, 78,
		17, 27, 37, 47, 57, 67, 77,
		16, 26, 36, 46, 56, 66, 76,
		15, 25, 35, 45, 55, 65, 75,
		14, 24, 34, 44, 54, 64, 74,
		13, 23, 33, 43, 53, 63, 73,
		12, 22, 32, 42, 52, 62, 72,
		11, 21, 31, 41, 51, 61, 71};
	assert(countNonZero(Mat(9,7,CV_8U, expected7x9_90) != result) == 0);

	cout << "-------------------test_warpAffine 7x9 180 degrees --------" << endl;
	matWarpAffine(image7x9, result, center7x9, 180, 1, Point(0,0), Size(0,0));
	cout << "180 " << matInfo(result) << endl << result << endl;
	assert(result.rows == 7 && result.cols == 9);
	uchar expected7x9_180[7][9] = {
		79,78,77,76,75,74,73,72,71,
		69,68,67,66,65,64,63,62,61,
		59,58,57,56,55,54,53,52,51,
		49,48,47,46,45,44,43,42,41,
		39,38,37,36,35,34,33,32,31,
		29,28,27,26,25,24,23,22,21,
		19,18,17,16,15,14,13,12,11
		};
	assert(countNonZero(Mat(7,9,CV_8U, expected7x9_180) != result) == 0);

	cout << "-------------------test_warpAffine 7x9 270 degrees --------" << endl;
	matWarpAffine(image7x9, result, center7x9, 270, 1, Point(0,0), Size(0,0));
	assert(result.rows == 9 && result.cols == 7);
	cout << "270 " << matInfo(result) << endl << result << endl;
	uchar expected7x9_270[9][7] = {
		71, 61, 51, 41, 31, 21, 11,
		72, 62, 52, 42, 32, 22, 12,
		73, 63, 53, 43, 33, 23, 13,
		74, 64, 54, 44, 34, 24, 14,
		75, 65, 55, 45, 35, 25, 15,
		76, 66, 56, 46, 36, 26, 16,
		77, 67, 57, 47, 37, 27, 17,
		78, 68, 58, 48, 38, 28, 18,
		79, 69, 59, 49, 39, 29, 19
		};
	assert(countNonZero(Mat(9,7,CV_8U, expected7x9_270) != result) == 0);

	cout << "-------------------test_warpAffine 6x8 --------" << endl;
	Point2f center6x8(3.5,2.5);
	uchar data6x8[][8] = {
		11,12,13,14,15,16,17,18,
		21,22,23,24,25,26,27,28,
		31,32,33,34,35,36,37,38,
		41,42,43,44,45,46,47,48,
		51,52,53,54,55,56,57,58,
		61,62,63,64,65,66,67,68
	};
	Mat image6x8(6,8,CV_8U,data6x8);
	cout << matInfo(image6x8) << endl << image6x8 << endl;

	cout << "-------------------test_warpAffine 6x8 90 degrees --------" << endl;
	result = image6x8.clone();
	matWarpAffine(result, result, center6x8, 90, 1, Point(0,0), Size(0,0));
	cout << "90 " << matInfo(result) << endl << result << endl;
	assert(result.rows == 8 && result.cols == 6);
	uchar expected6x8_90[8][6] = {
		18, 28, 38, 48, 58, 68, 
		17, 27, 37, 47, 57, 67,
		16, 26, 36, 46, 56, 66,
		15, 25, 35, 45, 55, 65,
		14, 24, 34, 44, 54, 64,
		13, 23, 33, 43, 53, 63,
		12, 22, 32, 42, 52, 62,
		11, 21, 31, 41, 51, 61};
	assert(countNonZero(Mat(8,6,CV_8U, expected6x8_90) != result) == 0);

	cout << "-------------------test_warpAffine 6x8 180 degrees --------" << endl;
	matWarpAffine(image6x8, result, center6x8, 180, 1, Point(0,0), Size(0,0));
	cout << "180 " << matInfo(result) << endl << result << endl;
	assert(result.rows == 6 && result.cols == 8);
	uchar expected6x8_180[6][8] = {
		68,67,66,65,64,63,62,61,
		58,57,56,55,54,53,52,51,
		48,47,46,45,44,43,42,41,
		38,37,36,35,34,33,32,31,
		28,27,26,25,24,23,22,21,
		18,17,16,15,14,13,12,11
		};
	assert(countNonZero(Mat(6,8,CV_8U, expected6x8_180) != result) == 0);

	cout << "-------------------test_warpAffine 6x8 270 degrees --------" << endl;
	matWarpAffine(image6x8, result, center6x8, 270, 1, Point(0,0), Size(0,0));
	cout << "270 " << matInfo(result) << endl << result << endl;
	assert(result.rows == 8 && result.cols == 6);
	uchar expected6x8_270[8][6] = {
		61, 51, 41, 31, 21, 11,
		62, 52, 42, 32, 22, 12,
		63, 53, 43, 33, 23, 13,
		64, 54, 44, 34, 24, 14,
		65, 55, 45, 35, 25, 15,
		66, 56, 46, 36, 26, 16,
		67, 57, 47, 37, 27, 17,
		68, 58, 48, 38, 28, 18
		};
	assert(countNonZero(Mat(8,6,CV_8U, expected6x8_270) != result) == 0);

	cout << "-------------------test_warpAffine 7x8 45 degrees --------" << endl;
	Point2f center7x8(3.5,3);
	uchar data7x8[7][8] = {
		255,255,255,255,255,255,255,255,
		255,  0,  0,  0,  0,  0,  0,255,
		255,  0,  0,  0,  0,  0,  0,255,
		255,  0,  0,  0,  0,  0,  0,255,
		255,  0,  0,  0,  0,  0,  0,255,
		255,  0,  0,  0,  0,  0,  0,255,
		255,255,255,255,255,255,255,255
		};
	Mat image7x8(7,8,CV_8U,data7x8);
	cout << matInfo(image7x8) << endl << image7x8 << endl;

	cout << "-------------------test_warpAffine 7x8 45 degrees --------" << endl;
	matWarpAffine(image7x8, result, center7x8, 45, 1, Point(0,0), Size(0,0), BORDER_CONSTANT, Scalar(9));
	cout << "45 " << matInfo(result) << endl << result << endl;
	assert(result.rows == 10 && result.cols == 10);
	uchar expected7x8_45[10][10] = {
	  	9,  9,  9,  9,124,247, 71,  9,  9,  9,
	  	9,  9,  9,124,215,109,247, 71,  9,  9,
	  	9,  9,124,215, 32,  0, 88,247, 71,  9,
	  	9,124,207, 32,  0,  0,  0, 88,247, 71,
		124,207, 32,  0,  0,  0,  0,  0,109,247,
		247,102,  0,  0,  0,  0,  0, 32,215,124,
		 71,247, 80,  0,  0,  0, 32,215,124,  9,
	  	9, 71,247, 88,  0, 32,215,124,  9,  9,
	  	9,  9, 71,247,109,215,124,  9,  9,  9,
	  	9,  9,  9, 71,247,124,  9,  9,  9,  9
		};
	assert(countNonZero(Mat(10,10,CV_8U, expected7x8_45) != result) == 0);
}
