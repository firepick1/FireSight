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

string matInfo(Mat &m) {
	char buf[100];
	sprintf(buf, "%sC%d(%dx%d)", cv_depth_names[m.depth()], m.channels(), m.rows, m.cols);
	return string(buf);
}

void matWarpAffine(Mat &image, Point center, double angle, double scale, 
	Point offset, Size size, Scalar borderValue, int borderMode, int flags)
{
	int cols = image.cols;
	int rows = image.rows;
	Mat transform = getRotationMatrix2D( center, angle, scale );
	transform.at<double>(0,2) += offset.x;
	transform.at<double>(1,2) += offset.y;

	Matx<double,3,4> pts(
		0, cols, cols, 0,
		0, 0, rows, rows,
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
	if (size.width <= 0) {
		if (minx < 0) {
			size.width = ceil(maxx - minx);
			LOGTRACE1("matWarpAffine() auto-width:%d", size.width);
			transform.at<double>(0,2) -= minx;
		} else {
			size.width = ceil(maxx);
			LOGTRACE1("matWarpAffine() auto+width:%d", size.width);
		}
	}
	if (size.height <= 0) {
		if (miny < 0) {
			size.height = ceil(maxy - miny);
			LOGTRACE1("matWarpAffine() auto-height:%d", size.height);
			transform.at<double>(1,2) -= miny;
		} else {
			size.height = ceil(maxy);
			LOGTRACE1("matWarpAffine() auto+height:%d", size.height);
		}
	}

	Mat result;
	warpAffine( image, result, transform, size, flags, borderMode, borderValue );
	image = result;
}

