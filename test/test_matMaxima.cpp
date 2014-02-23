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

void test_matMaxima() {
	vector<Point> locations;
	vector<Point> locationsExpected;

	uchar dataRows[][6] = {
		11, 0, 0, 0, 0, 11, // 0
		 0, 0, 0, 0, 0,  0, // 1
		11,11, 0, 0,11, 11, // 2
		 0, 0, 0, 0, 0,  0, // 3
		11,11, 0, 0,11, 11, // 4
		 0, 0, 0, 0, 0,  0, // 5
		 0, 11, 0, 0,11, 0, // 6
		 0, 0, 0, 0, 0,  0, // 7
		 0, 11,22,22,11, 0, // 8
		 0, 0, 0, 0, 0,  0, // 9
		 0, 11,22,11,22, 0, // 10
		 0, 0, 0, 0, 0,  0, // 11
		 0, 22,11,22,11, 0, // 12
		 0, 0, 0, 0, 0,  0, // 13
		 0, 11,11,11,22,22, // 14
		 0, 0, 0, 0, 0,  0, // 15
		 0, 22,22,11,11,11 // 16
	};
	Mat matRows(17,6,CV_8U, dataRows);
	cout << "-----test_matMaxima dataRows " << matInfo(matRows) << endl << matRows << endl;
	locations.clear();
	matMaxima(matRows, locations, 1, 255);
	Point expectedDataRows[] = {
		Point(0, 0), Point(5, 0), 
		Point(1, 2), Point(5, 2), 
		Point(1, 4), Point(5, 4), 
		Point(1, 6), Point(4, 6), 
		Point(3, 8), 
		Point(2,10), Point(4,10), 
		Point(1,12), Point(3,12), 
		Point(5,14), 
		Point(2,16)
	};
	for (int i = 0; i < sizeof(expectedDataRows)/sizeof(Point); i++) {
		cout << "expected[" << i << "]:" << expectedDataRows[i] << " actual[" << i << "]:" << locations[i] << endl;
		assert(locations[i] == expectedDataRows[i]);
	}
	assert(locations.size() == sizeof(expectedDataRows)/sizeof(Point));

	uchar data2DEquals[][5] = {
		11, 0,11, 0,11, // 0
		 0,11, 0,11, 0, // 1
		11, 0,11, 0,11, // 2
		 0,11, 0,11, 0, // 3
		11, 0,11, 0,11  // 4
	};
	Mat mat2DEquals(5,5,CV_8U, data2DEquals);
	cout << "-----test_matMaxima data2DEquals " << matInfo(mat2DEquals) << endl << mat2DEquals << endl;
	locations.clear();
	matMaxima(mat2DEquals, locations, 1, 255);
	Point expected2DEquals[] = {
		Point(0, 0), Point(2, 0), Point(4, 0)
	};
	for (int i = 0; i < sizeof(expected2DEquals)/sizeof(Point); i++) {
		cout << "expected[" << i << "]:" << expected2DEquals[i] << " actual[" << i << "]:" << locations[i] << endl;
		assert(locations[i] == expected2DEquals[i]);
	}
	assert(locations.size() == sizeof(expected2DEquals)/sizeof(Point));

	uchar data2DBorder[][8] = {
	// 0  1  2  3  4  5  6  7
		22, 0, 0,11,22, 0, 0,22, // 0
		 0,22, 0,22,11, 0,22, 0, // 1
		 0, 0,11,11,11,11, 0, 0, // 2
		11,22,11,22,22,11,22,11, // 3
		22,11,11,22,22,11,11,22, // 4
		 0, 0,11,11,11,11, 0, 0, // 5
		 0,22, 0,22,11, 0,22, 0, // 6
		22, 0, 0,11,22, 0, 0,22  // 7
	};
	Mat mat2DBorder(8,8,CV_8U, data2DBorder);
	cout << "-----test_matMaxima data2DBorder " << matInfo(mat2DBorder) << endl << mat2DBorder << endl;
	locations.clear();
	matMaxima(mat2DBorder, locations, 1, 255);
	Point expected2DBorder[] = {
		Point(0, 0), Point(4, 0), Point(7, 0),
		Point(1, 3), Point(4, 3), Point(6, 3),
		Point(1, 6), Point(3, 6), Point(6, 6)
	};
	for (int i = 0; i < sizeof(expected2DBorder)/sizeof(Point); i++) {
		cout << "expected[" << i << "]:" << expected2DBorder[i] << " actual[" << i << "]:" << locations[i] << endl;
		assert(locations[i] == expected2DBorder[i]);
	}
	assert(locations.size() == sizeof(expected2DBorder)/sizeof(Point));
}

void test_matMinima() {
}
