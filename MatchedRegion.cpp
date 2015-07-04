#include <string.h>
#include <math.h>
#include "FireLog.h"
#include "Pipeline.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"

using namespace cv;
using namespace std;
using namespace firesight;

const float PI = 3.141592653589793f;

MatchedRegion::MatchedRegion(Range xRange, Range yRange, Point2f average, int pointCount, float covar) {
	this->xRange = xRange;
	this->yRange = yRange;
	this->average = average;
	this->pointCount = pointCount;
	this->ellipse = (xRange.end-xRange.start+1) * (yRange.end-yRange.start+1) * PI/4;
	this->covar = covar;
}

json_t *MatchedRegion::as_json_t() {
	json_t *pObj = json_object();

	json_object_set_new(pObj, "xmin", json_integer(xRange.start));
	json_object_set_new(pObj, "xmax", json_integer(xRange.end));
	json_object_set_new(pObj, "xavg", json_real(average.x));
	json_object_set_new(pObj, "ymin", json_integer(yRange.start));
	json_object_set_new(pObj, "ymax", json_integer(yRange.end));
	json_object_set_new(pObj, "yavg", json_real(average.y));
	json_object_set_new(pObj, "pts", json_integer(pointCount));
	json_object_set_new(pObj, "ellipse", json_real(ellipse));
	json_object_set_new(pObj, "covar", json_real(covar));

	return pObj;
}

string MatchedRegion::asJson(JSONSerializer& serializer) {
	json_t *pObj = as_json_t();
	string result = serializer.serialize(pObj);
	json_decref(pObj);
	return result;
}

