
#include <string.h>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <boost/format.hpp>
#include "FireLog.h"
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"

using namespace cv;
using namespace std;
using namespace FireSight;

static float pi = boost::math::constants::pi<float>();

MatchedRegion::MatchedRegion(Range xRange, Range yRange, Point2f average, int pointCount, float covar) {
	this->xRange = xRange;
	this->yRange = yRange;
	this->average = average;
	this->pointCount = pointCount;
	this->ellipse = (xRange.end-xRange.start+1) * (yRange.end-yRange.start+1) * pi/4;
	this->covar = covar;
}

string MatchedRegion::asJson() {
	boost::format fmt(
		"{"
			" \"x\":{\"min\":%d,\"max\":%d,\"avg\":%.2f}"
			",\"y\":{\"min\":%d,\"max\":%d,\"avg\":%.2f}"
			",\"pts\":%d"
			",\"ellipse\":%.2f"
			",\"covar\":%.2f"
		" }"
	);
	fmt % xRange.start;
	fmt % xRange.end;
	fmt % average.x;
	fmt % yRange.start;
	fmt % yRange.end;
	fmt % average.y;
	fmt % pointCount;
	fmt % ellipse;
	fmt % covar;
	return fmt.str();
}

