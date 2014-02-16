#include <string.h>
#include <fstream>
#include <sstream>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <boost/format.hpp>
#include "FireLog.h"
#include "FireSight.hpp"
#include "version.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "MatUtil.hpp"

using namespace cv;
using namespace std;
using namespace FireSight;

static void assertKeypoint(const KeyPoint &keypoint, double x, double y, double size, double angle, double tolerance) {
	cout << "assertKeyPoint() x:" << keypoint.pt.x << " y:" << keypoint.pt.y ;
	cout << " size:" << keypoint.size << " angle:" << keypoint.angle;
	cout << endl;
	assert(x-tolerance <= keypoint.pt.x && keypoint.pt.x <= x+tolerance);
	assert(y-tolerance <= keypoint.pt.y && keypoint.pt.y <= y+tolerance);
	assert(size-tolerance <= keypoint.size && keypoint.size <= size+tolerance);
	assert(angle-tolerance <= keypoint.angle && keypoint.angle <= angle+tolerance);
}

extern void generate_ringMat();

static void test_matRing() {
	Scalar data = Scalar::all(255);
	Mat image;
	Mat result;
	image = Mat(1,1,CV_8U,data);
	matRing(image, result);
	assert(result.rows == 1 && result.cols == 1);
	assert(result.at<uchar>(0,0) == 255);

	image = Mat(8,8,CV_8U,data);
	image(Rect(0,0,4,4)) = Scalar::all(0);
	matRing(image, result);
	cout << matInfo(result) << endl << result << endl;
	assert(result.rows == 8 && result.cols == 8);
	uchar expected8x8[8][8] = { 
		191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191,
		191, 191, 191, 191, 191, 191, 191, 191
	};
	for (int r=0; r<8; r++) {
		for (int c=0; c<8; c++) {
			assert(expected8x8[r][c] == result.at<uchar>(r,c));
		}
	}

	image = Mat(9,9,CV_8U,data);
	image(Rect(0,0,4,4)) = Scalar::all(0);
	matRing(image, result);
	cout << matInfo(result) << endl << result << endl;
	assert(result.rows == 9 && result.cols == 9);
	uchar expected9x9[9][9] = {
		191, 191, 202, 202, 202, 202, 202, 191, 191,
		191, 202, 204, 204, 204, 204, 204, 202, 191,
		202, 204, 207, 207, 207, 207, 207, 204, 202,
		202, 204, 207, 223, 223, 223, 207, 204, 202,
		202, 204, 207, 223, 255, 223, 207, 204, 202,
		202, 204, 207, 223, 223, 223, 207, 204, 202,
		202, 204, 207, 207, 207, 207, 207, 204, 202,
		191, 202, 204, 204, 204, 204, 204, 202, 191,
		191, 191, 202, 202, 202, 202, 202, 191, 191
	};
	for (int r=0; r<9; r++) {
		for (int c=0; c<9; c++) {
			assert(expected9x9[r][c] == result.at<uchar>(r,c));
		}
	}

	image = Mat(7,9,CV_8U,Scalar::all(0));
	image(Rect(1,1,7,5)) = Scalar::all(255);
	matRing(image, result);
	cout << matInfo(result) << endl << result << endl;
	assert(result.rows == 9 && result.cols == 9);
	uchar expected7x9[9][9] = {
		191, 191, 202, 202, 202, 202, 202, 191, 191,
		191, 202, 204, 204, 204, 204, 204, 202, 191,
		202, 204, 207, 207, 207, 207, 207, 204, 202,
		202, 204, 207, 223, 223, 223, 207, 204, 202,
		202, 204, 207, 223, 255, 223, 207, 204, 202,
		202, 204, 207, 223, 223, 223, 207, 204, 202,
		202, 204, 207, 207, 207, 207, 207, 204, 202,
		191, 202, 204, 204, 204, 204, 204, 202, 191,
		191, 191, 202, 202, 202, 202, 202, 191, 191
	};
	for (int r=0; r<9; r++) {
		for (int c=0; c<9; c++) {
			assert(expected7x9[r][c] == result.at<uchar>(r,c));
		}
	}
}

static void test_warpAffine() {
	cout << "-------------------test_warpAffine--------" << endl;
	Matx<double,3,4> pts(
		0, 1, 1, 0,
		0, 0, 1, 1,
		1, 1, 1, 1);
	Mat transform = getRotationMatrix2D(Point(0.5,0.5), 30, 1);
	cout << "transform:" << matInfo(transform) << endl;
	Mat mpts(pts);
	cout << "pts:" << matInfo(mpts) << endl;
	Mat newPts = transform * mpts;
	cout << "test_warpAffine()" << newPts << endl;
	double minx = newPts.at<double>(0,0);
	double maxx = newPts.at<double>(0,0);
	double miny = newPts.at<double>(1,0);
	double maxy = newPts.at<double>(1,0);
	for (int c=1; c<4; c++) {
		double x = newPts.at<double>(0,c);
		minx = min(minx, x);
		maxx = max(maxx, x);
		double y = newPts.at<double>(1,c);
		miny = min(miny, y);
		maxy = max(maxy, y);
	}
	double width = maxx - minx;
	double height = maxy - miny;
	cout << width << "x" << height << endl;
	transform.at<double>(0,2) -= minx;
	transform.at<double>(1,2) -= miny;
	cout << "transform:" << matInfo(transform) << endl;
	newPts = transform * mpts;
	cout << "test_warpAffine()" << newPts << endl;
	//[0, 0.8660254037844387, 1.366025403784439, 0.4999999999999999;
	//  0.4999999999999999, 0, 0.8660254037844387, 1.366025403784439]
}

static void test_regionKeypoint() {
	Pipeline pipeline("[]");
	KeyPoint keypoint;
	vector<Point> pts;
	double tolerance = .01;

	LOGTRACE("----------------------HORIZONTAL");

	pts.clear();
	pts.push_back(Point(0,10));
	pts.push_back(Point(10,10));
	pts.push_back(Point(20,10));
	cout << "horizontal pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 0, tolerance);

	pts.clear();
	pts.push_back(Point(0,11));
	pts.push_back(Point(10,10));
	pts.push_back(Point(20,9));
	cout << "horizontal slightly down pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 354.29, tolerance);

	pts.clear();
	pts.push_back(Point(0,9));
	pts.push_back(Point(10,10));
	pts.push_back(Point(20,11));
	cout << "horizontal slightly up pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 5.71, tolerance);
	
	LOGTRACE("----------------------VERTICAL");

	pts.clear();
	pts.push_back(Point(10,0));
	pts.push_back(Point(10,10));
	pts.push_back(Point(10,20));
	cout << "vertical pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 90, tolerance);

	pts.clear();
	pts.push_back(Point(11,0));
	pts.push_back(Point(10,10));
	pts.push_back(Point(9,20));
	cout << "vertical slightly left pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 95.71, tolerance);

	pts.clear();
	pts.push_back(Point(9,0));
	pts.push_back(Point(10,10));
	pts.push_back(Point(11,20));
	cout << "vertical slightly right pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 84.29, tolerance);

	LOGTRACE("----------------------DIAGONAL UP");

	pts.clear();
	pts.push_back(Point(0,0));
	pts.push_back(Point(10,10));
	pts.push_back(Point(20,20));
	cout << "diagonal up pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 45, tolerance);

	pts.clear();
	pts.push_back(Point(0,1));
	pts.push_back(Point(10,10));
	pts.push_back(Point(20,19));
	cout << "diagonal up- pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 41.99, tolerance);

	pts.clear();
	pts.push_back(Point(1,0));
	pts.push_back(Point(10,10));
	pts.push_back(Point(19,20));
	cout << "diagonal up+ pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 48.01, tolerance);

	LOGTRACE("----------------------DIAGONAL DOWN");

	pts.clear();
	pts.push_back(Point(20,0));
	pts.push_back(Point(10,10));
	pts.push_back(Point(0,20));
	cout << "diagonal down pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 315, tolerance);

	pts.clear();
	pts.push_back(Point(19,0));
	pts.push_back(Point(10,10));
	pts.push_back(Point(1,20));
	cout << "diagonal down+ pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 131.99, tolerance);

	pts.clear();
	pts.push_back(Point(20,1));
	pts.push_back(Point(10,10));
	pts.push_back(Point(0,19));
	cout << "diagonal down- pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 10, 10, 1.954, 318.01, tolerance);

}

int main(int argc, char *argv[])
{
	char version[30];
	sprintf(version, "FireSight test v%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
	LOGINFO1("%s", version);
	cout << version << endl;
	cout << "https://github.com/firepick1/FireSight" << endl;
	firelog_level(FIRELOG_TRACE);

	test_regionKeypoint();
	test_warpAffine();
	test_matRing();
}
