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
#include "jo_util.hpp"

using namespace cv;
using namespace std;
using namespace FireSight;

void Pipeline::covarianceXY(const vector<Point> &pts, Mat &covOut, Mat &meanOut) {
	Mat_<double> data(pts.size(),2);
	for (int i = 0; i < pts.size(); i++) {
		data(i,0) = pts[i].x;
		data(i,1) = pts[i].y;
	}

	calcCovarMatrix(data, covOut, meanOut, CV_COVAR_NORMAL | CV_COVAR_ROWS);

	if (logLevel >= FIRELOG_TRACE) {
		char buf[200];
		sprintf(buf, "covarianceXY() -> covariance:[%f,%f;%f,%f] mean:[%f,%f]",
			covOut.at<double>(0,0), covOut.at<double>(0,1), 
			covOut.at<double>(1,0), covOut.at<double>(1,1),
			meanOut.at<double>(0), meanOut.at<double>(1));
		LOGTRACE1("%s", buf);
	}
}

void Pipeline::eigenXY(const vector<Point> &pts, Mat &eigenvectorsOut, Mat &meanOut, Mat &covOut) {
	covarianceXY(pts, covOut, meanOut);

	Mat eigenvalues;
	eigen(covOut, eigenvalues, eigenvectorsOut);

	if (logLevel >= FIRELOG_TRACE) {
		char buf[200];
		sprintf(buf, "eigenXY() -> [%f,%f;%f,%f]",
			eigenvectorsOut.at<double>(0,0), eigenvectorsOut.at<double>(0,1),
			eigenvectorsOut.at<double>(1,0), eigenvectorsOut.at<double>(1,1));
		LOGTRACE1("%s", buf);
	}
}

KeyPoint Pipeline::regionKeypoint(const vector<Point> &region, double q4Offset) {
	Mat covOut;
	Mat mean;
	Mat eigenvectors;
	eigenXY(region, eigenvectors, mean, covOut);

	double x = mean.at<double>(0);
	double y = mean.at<double>(1);
	double e0x = eigenvectors.at<double>(0,0);
	double e0y = eigenvectors.at<double>(0,1);
	double e1x = eigenvectors.at<double>(1,0);
	double e1y = eigenvectors.at<double>(1,1);
	double radians;
	if (covOut.at<double>(1,0) >= 0) { // Q1
		if (covOut.at<double>(0,0) >= covOut.at<double>(1,1)){ // X >= Y
			if (e0x >= e0y) { // eigenvector X >= Y
				radians = atan2(e0y, e0x);
			} else {
				radians = atan2(e1y, e1x);
			}
		} else { // eigenvector Y < X
			if (e0x >= e0y) {
				radians = atan2(e1y, e1x);
			} else {
				radians = atan2(e0y, e0x);
			}
		}
	} else { // Q2
		if (e0x >= 0 && e0y >= 0) { // eigenvector Q4
			radians = atan2(e1y, e1x);
		} else { // eigenvector Q2
			radians = atan2(e0y, e0x);
		}
		if (radians < 0) { // Q4
			radians = radians + q4Offset;
		}
	}

	double diam = 2*sqrt(region.size()/CV_PI);
	if (logLevel >= FIRELOG_TRACE) {
		char buf[150];
		sprintf(buf, "regionKeypoint() -> x:%f y:%f diam:%f angle:%f",
			x, y, diam, radians * 180./CV_PI);
		LOGTRACE1("%s", buf);
	}

	return KeyPoint(x, y, diam, radians * 180./CV_PI);
}


static void drawRegions(Mat &image, vector<vector<Point> > &regions, Scalar color) {
	int nRegions = (int) regions.size();
	int blue = color[0];
	int green = color[1];
	int red = color[2];
	bool changeColor = red == -1 && green == -1 && blue == -1;

	for( int i = 0; i < nRegions; i++) {
		int nPts = regions[i].size();
		if (changeColor) {
			red = (i & 1) ? 0 : 255;
			green = (i & 2) ? 128 : 192;
			blue = (i & 1) ? 255 : 0;
		}
		for (int j = 0; j < nPts; j++) {
			image.at<Vec3b>(regions[i][j])[0] = blue;
			image.at<Vec3b>(regions[i][j])[1] = green;
			image.at<Vec3b>(regions[i][j])[2] = red;
		}
	}
}

void Pipeline::detectKeypoints(json_t *pStageModel, vector<vector<Point> > &regions, double q4Offset) {
	int nRegions = regions.size();
	json_t *pKeypoints = json_array();
	json_object_set(pStageModel, "keypoints", pKeypoints);

	for (int i=0; i < nRegions; i++) {
		KeyPoint keypoint = regionKeypoint(regions[i], q4Offset);
		json_t *pKeypoint = json_object();
		json_object_set(pKeypoint, "pt.x", json_real(keypoint.pt.x));
		json_object_set(pKeypoint, "pt.y", json_real(keypoint.pt.y));
		json_object_set(pKeypoint, "size", json_real(keypoint.size));
		json_object_set(pKeypoint, "angle", json_real(keypoint.angle));
		json_array_append(pKeypoints, pKeypoint);
	}
}

bool Pipeline::apply_MSER(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
	int delta = jo_int(pStage, "delta", 5);
	int minArea = jo_int(pStage, "minArea", 60);
	int maxArea = jo_int(pStage, "maxArea", 14400);
	float maxVariation = jo_double(pStage, "maxVariation", 0.25);
	float minDiversity = jo_double(pStage, "minDiversity", 0.2);
	int maxEvolution = jo_int(pStage, "maxEvolution", 200);
	double areaThreshold = jo_double(pStage, "areaThreshold", 1.01);
	double minMargin = jo_double(pStage, "minMargin", .003);
	double q4Offset = jo_double(pStage, "q4Offset", 2*CV_PI);
	int edgeBlurSize = jo_int(pStage, "edgeBlurSize", 5);
	json_t *pDetect = json_object_get(pStage, "detect");
	Scalar color = jo_Scalar(pStage, "color", Scalar::all(-1));
	json_t * pMask = json_object_get(pStage, "mask");
	const char *errMsg = NULL;
	char errBuf[100];
	int maskX;
	int maskY;
	int maskW;
	int maskH;

	if (minArea < 0 || maxArea <= minArea) {
	  errMsg = "expected 0<=minArea and minArea<maxArea";
	} else if (maxVariation < 0 || minDiversity < 0) {
	  errMsg = "expected 0<=minDiversity and 0<=maxVariation";
	} else if (maxEvolution<0) {
	  errMsg = "expected 0<=maxEvolution";
	} else if (areaThreshold < 0 || minMargin < 0) {
	  errMsg = "expected 0<=areaThreshold and 0<=minMargin";
	} else if (edgeBlurSize < 0) {
	  errMsg = "expected 0<=edgeBlurSize";
	} if (pMask) {
	  if (!json_is_object(pMask)) {
		  errMsg = "expected mask JSON object with x, y, width, height";
		} else {
			if (pMask) {
				LOGTRACE("mask:{");
			}
			maskX = jo_int(pMask, "x", 0);
			maskY = jo_int(pMask, "y", 0);
			maskW = jo_int(pMask, "width", image.cols);
			maskH = jo_int(pMask, "height", image.rows);
			if (pMask) {
				LOGTRACE("}");
			}
			if (maskX < 0 || image.cols <= maskX) {
				sprintf(errBuf, "expected 0 <= mask.x < %d", image.cols);
				errMsg = errBuf;
			} else if (maskY < 0 || image.rows <= maskY) {
				sprintf(errBuf, "expected 0 <= mask.y < %d", image.cols);
				errMsg = errBuf;
			} else if (maskW <= 0 || image.cols < maskW) {
				sprintf(errBuf, "expected 0 < mask.width <= %d", image.cols);
				errMsg = errBuf;
			} else if (maskH <= 0 || image.rows < maskH) {
				sprintf(errBuf, "expected 0 < mask.height <= %d", image.rows);
				errMsg = errBuf;
			}
		}
	}

  bool isKeypoints = false;
	if (!errMsg && pDetect) {
		if (json_is_string(pDetect)) {
			if (strcmp("keypoints", json_string_value(pDetect)) == 0) {
				isKeypoints = true;
			} else {
				errMsg = "Invalid value for detect";
			}
		} else {
			errMsg = "Expected string value for detect";
		}
	}

	if (!errMsg) {
		MSER mser(delta, minArea, maxArea, maxVariation, minDiversity,
			maxEvolution, areaThreshold, minMargin, edgeBlurSize);
		Mat mask;
		Rect maskRect(maskX, maskY, maskW, maskH);
		if (pMask) {
			mask = Mat::zeros(image.rows, image.cols, CV_8UC1);
			mask(maskRect) = 1;
		}
		vector<vector<Point> > regions;
		mser(image, regions, mask);

		int nRegions = (int) regions.size();
		LOGTRACE1("apply_MSER matched %d regions", nRegions);
		if (isKeypoints) {
			detectKeypoints(pStageModel, regions, q4Offset);
		}
		if (json_object_get(pStage, "color")) {
			if (image.channels() == 1) {
				cvtColor(image, image, CV_GRAY2BGR);
				LOGTRACE("cvtColor(CV_GRAY2BGR)");
			}
			drawRegions(image, regions, color);
			if (pMask) {
				if (color[0]==-1 && color[1]==-1 && color[2]==-1 && color[3]==-1) {
					rectangle(image, maskRect, Scalar(255, 0, 255));
				} else {
					rectangle(image, maskRect, color);
				}
			}
		}
	}

	return stageOK("apply_MSER(%s) %s", errMsg, pStage, pStageModel);
}

