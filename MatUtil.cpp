#include "MatUtil.hpp"
#include <stdio.h>

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
