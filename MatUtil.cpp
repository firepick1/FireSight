#include <stdio.h>
#include <boost/format.hpp>
#include "MatUtil.hpp"
#include "FireLog.h"

using namespace cv;
using namespace std;

static const char *cv_depth_names[] = {
	"CV_8U",
	"CV_8S",
	"CV_16U",
	"CV_16S",
	"CV_32S",
	"CV_32F",
	"CV_64F"
};

string matInfo(const Mat &m) {
	char buf[100];
	sprintf(buf, "%sC%d(%dx%d)", cv_depth_names[m.depth()], m.channels(), m.rows, m.cols);
	return string(buf);
}

void matWarpAffine(const Mat &image, Mat &result, Point2f center, double angle, double scale, 
	Point2f offset, Size size, int borderMode, Scalar borderValue, int flags)
{
	int cols = image.cols;
	int rows = image.rows;
	Mat transform = getRotationMatrix2D( center, angle, scale );
	if (logLevel >= FIRELOG_TRACE) {
		cout << "center.x:" << center.x << " center.y:" << center.y << endl;
		cout << matInfo(transform) << endl << transform << endl;
	}
	transform.at<double>(0,2) += offset.x;
	transform.at<double>(1,2) += offset.y;

	Matx<double,3,4> pts(
		0, cols-1, cols-1, 0,
		0, 0, rows-1, rows-1,
		1, 1, 1, 1);
	Mat mpts(pts);
	Mat newPts = transform * mpts;
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
	char buf[100];
	if (size.width <= 0) {
		size.width = maxx - minx + 1.5;
		transform.at<double>(0,2) += (size.width-1)/2.0 - center.x;
	}
	if (size.height <= 0) {
		size.height = maxy - miny + 1.5;
    transform.at<double>(1,2) += (size.height-1)/2.0 - center.y;
	}
	if (logLevel >= FIRELOG_TRACE) {
		sprintf(buf, "matWarpAffine() minx:%f, maxx:%f, auto-width:%d", minx, maxx, size.width);
		LOGTRACE(buf);
		sprintf(buf, "matWarpAffine() miny:%f, maxy:%f, auto-height:%d", miny, maxy, size.height);
		LOGTRACE(buf);
		cout << matInfo(transform) << endl << transform << endl;
	}

	warpAffine( image, result, transform, size, flags, borderMode, borderValue );
}

