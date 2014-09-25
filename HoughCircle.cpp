/*
 * author:  Simon Fojtu (simon.fojtu@gmail.com)
 * date  :  2014-06-09
 */

#include <string.h>
#include <math.h>
#include "FireLog.h"
#include "FireSight.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"

using namespace cv;
using namespace std;
using namespace firesight;

Circle::Circle(float x, float y, float radius) {
    this->x = x;
    this->y = y;
    this->radius = radius;
}

json_t *Circle::as_json_t() {
	json_t *pObj = json_object();

	json_object_set(pObj, "x", json_real(x));
	json_object_set(pObj, "y", json_real(y));
	json_object_set(pObj, "radius", json_real(radius));

	return pObj;
}

string Circle::asJson() {
	json_t *pObj = as_json_t();
	char *pObjStr = json_dumps(pObj, JSON_PRESERVE_ORDER|JSON_COMPACT|JSON_INDENT(2));
	string result(pObjStr);
	return result;
}

HoughCircle::HoughCircle(int minDiameter, int maxDiameter) {
	minDiam = minDiameter;
	maxDiam = maxDiameter;
	_showCircles = CIRCLE_SHOW_NONE;
	LOGTRACE2("HoughCircle() (maxDiam:%d minDiam:%d)", maxDiam, minDiam);
}

void HoughCircle::setFilterParams(int d, double sigmaColor, double sigmaSpace) {
    bf_d = d;
    bf_sigmaColor = sigmaColor;
    bf_sigmaSpace = sigmaSpace;
}

void HoughCircle::setHoughParams(double dp, double minDist, double param1, double param2) {
    hc_dp = dp;
    hc_minDist = minDist;
    hc_param1 = param1;
    hc_param2 = param2;
}

void HoughCircle::setShowCircles(int show) {
  _showCircles = show;
}

void HoughCircle::scan(Mat &image, vector<Circle> &circles) {
	Mat matGray, matFiltered;

	if (image.channels() == 1) {
		matGray = image;
	} else {
		cvtColor(image, matGray, CV_RGB2GRAY);
	}

    cv::bilateralFilter(matGray, matFiltered, bf_d, bf_sigmaColor, bf_sigmaSpace);
	
	vector<Vec3f> vec3f_circles;
    HoughCircles(matGray, vec3f_circles, CV_HOUGH_GRADIENT, hc_dp, hc_minDist, hc_param1, hc_param2, minDiam/2.0, maxDiam/2.0);
    for (size_t i = 0; i < vec3f_circles.size(); i++) {
        circles.push_back(Circle(vec3f_circles[i][0], vec3f_circles[i][1], vec3f_circles[i][2]));
    }

	LOGTRACE1("HoughCircle::scan() -> found %d circles", (int) circles.size());

    if (_showCircles)
        show(image, circles);
}

void HoughCircle::show(Mat & image, vector<Circle> circles) {
  for( size_t i = 0; i < circles.size(); i++ ) {
      Point center(cvRound(circles[i].x), cvRound(circles[i].y));
      int radius = cvRound(circles[i].radius);
      // circle center
      circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
   }

}
