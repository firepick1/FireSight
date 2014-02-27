#include <string.h>
#include <math.h>
#include "FireLog.h"
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "jo_util.hpp"

using namespace cv;
using namespace std;
using namespace firesight;

typedef enum {DETECT_NONE, DETECT_KEYPOINTS, DETECT_RECTS} Detect;

/** (DEPRECATED: should be private) */
void Pipeline::covarianceXY(const vector<Point> &pts, Mat &covOut, Mat &meanOut) {
	LOGERROR("covarianceXY() is deprecated as a public function");
	_covarianceXY(pts, covOut, meanOut);
}

/**
 * Compute covariance and mean of region
 */
void Pipeline::_covarianceXY(const vector<Point> &pts, Mat &covOut, Mat &meanOut) {
	Mat_<double> data(pts.size(),2);
	for (int i = 0; i < pts.size(); i++) {
		data(i,0) = pts[i].x;
		data(i,1) = pts[i].y;
	}

	calcCovarMatrix(data, covOut, meanOut, CV_COVAR_NORMAL | CV_COVAR_ROWS);

	if (logLevel >= FIRELOG_TRACE) {
		char buf[200];
		snprintf(buf, sizeof(buf), "covarianceXY() -> covariance:[%f,%f;%f,%f] mean:[%f,%f]",
			covOut.at<double>(0,0), covOut.at<double>(0,1), 
			covOut.at<double>(1,0), covOut.at<double>(1,1),
			meanOut.at<double>(0), meanOut.at<double>(1));
		LOGTRACE1("%s", buf);
	}
}

/** (DEPRECATED: should be private) */
void Pipeline::eigenXY(const vector<Point> &pts, Mat &eigenvectorsOut, Mat &meanOut, Mat &covOut) {
	LOGERROR("eigenXY() is deprecated as a public function");
	_eigenXY(pts, eigenvectorsOut, meanOut, covOut);
}

/** 
 * Compute eigenvectors, eigenvalues, mean, and covariance of region
 */
void Pipeline::_eigenXY(const vector<Point> &pts, Mat &eigenvectorsOut, Mat &meanOut, Mat &covOut) {
	covarianceXY(pts, covOut, meanOut);

	Mat eigenvalues;
	eigen(covOut, eigenvalues, eigenvectorsOut);

	LOGTRACE4("eigenXY() -> [%f,%f;%f,%f]",
		eigenvectorsOut.at<double>(0,0), eigenvectorsOut.at<double>(0,1),
		eigenvectorsOut.at<double>(1,0), eigenvectorsOut.at<double>(1,1));
}

KeyPoint Pipeline::regionKeypoint(const vector<Point> &region) {
	LOGERROR("regionKeypoint() is deprecated as a public function");
	return _regionKeypoint(region);
}

/**
 * Using eigenvectors and covariance matrix of given region, return a
 * keypoint representing that region. The eigenvectors detemine the angle of the 
 * region on the interval (-PI/2,PI/2]. Caller may provide a fourth quadrant (negative angle)
 * offset of CV_PI for a vertical direction bias ("portrait regions"), 
 * or use the default of 2*CV_PI for a horizontal bias ("landscape regions").
 * @param region of points to analyze
 */
KeyPoint Pipeline::_regionKeypoint(const vector<Point> &region) {
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
	}

	double degrees = radians * 180./CV_PI;
	if (degrees < 0) {
		degrees = degrees + 360;
	} else if (degrees >= 135) {
	  degrees = degrees + 180;
	}

	double diam = 2*sqrt(region.size()/CV_PI);
	LOGTRACE4("regionKeypoint() -> x:%f y:%f diam:%f angle:%f", x, y, diam, degrees);

	return KeyPoint(x, y, diam, degrees);
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

void Pipeline::detectRects(json_t *pStageModel, vector<vector<Point> > &regions) {
	int nRegions = regions.size();
	json_t *pRects = json_array();
	json_object_set(pStageModel, "rects", pRects);

	for (int i=0; i < nRegions; i++) {
		RotatedRect rect = minAreaRect(regions[i]);
		json_t *pRect = json_object();
		json_object_set(pRect, "x", json_real(rect.center.x));
		json_object_set(pRect, "y", json_real(rect.center.y));
		json_object_set(pRect, "width", json_real(rect.size.width));
		json_object_set(pRect, "height", json_real(rect.size.height));
		json_object_set(pRect, "angle", json_real(rect.angle));
		json_array_append(pRects, pRect);
	}
}

void Pipeline::detectKeypoints(json_t *pStageModel, vector<vector<Point> > &regions) {
	int nRegions = regions.size();
	json_t *pKeypoints = json_array();
	json_object_set(pStageModel, "keypoints", pKeypoints);

	for (int i=0; i < nRegions; i++) {
		KeyPoint keypoint = regionKeypoint(regions[i]);
		json_t *pKeypoint = json_object();
		json_object_set(pKeypoint, "pt.x", json_real(keypoint.pt.x));
		json_object_set(pKeypoint, "pt.y", json_real(keypoint.pt.y));
		json_object_set(pKeypoint, "size", json_real(keypoint.size));
		json_object_set(pKeypoint, "angle", json_real(keypoint.angle));
		json_array_append(pKeypoints, pKeypoint);
	}
}

bool Pipeline::apply_MSER(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	int delta = jo_int(pStage, "delta", 5, model.argMap);
	int minArea = jo_int(pStage, "minArea", 60, model.argMap);
	int maxArea = jo_int(pStage, "maxArea", 14400, model.argMap);
	float maxVariation = jo_double(pStage, "maxVariation", 0.25, model.argMap);
	float minDiversity = jo_double(pStage, "minDiversity", 0.2, model.argMap);
	int maxEvolution = jo_int(pStage, "maxEvolution", 200, model.argMap);
	double areaThreshold = jo_double(pStage, "areaThreshold", 1.01, model.argMap);
	double minMargin = jo_double(pStage, "minMargin", .003, model.argMap);
	int edgeBlurSize = jo_int(pStage, "edgeBlurSize", 5, model.argMap);
	json_t *pDetect = jo_object(pStage, "detect", model.argMap);
	Scalar color = jo_Scalar(pStage, "color", Scalar::all(-1), model.argMap);
	json_t * pMask = jo_object(pStage, "mask", model.argMap);
	const char *errMsg = NULL;
	char errBuf[150];
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
			maskX = jo_int(pMask, "x", 0, model.argMap);
			maskY = jo_int(pMask, "y", 0, model.argMap);
			maskW = jo_int(pMask, "width", model.image.cols, model.argMap);
			maskH = jo_int(pMask, "height", model.image.rows, model.argMap);
			if (pMask) {
				LOGTRACE("}");
			}
			if (maskX < 0 || model.image.cols <= maskX) {
				snprintf(errBuf, sizeof(errBuf), "expected 0 <= mask.x < %d", model.image.cols);
				errMsg = errBuf;
			} else if (maskY < 0 || model.image.rows <= maskY) {
				snprintf(errBuf, sizeof(errBuf), "expected 0 <= mask.y < %d", model.image.cols);
				errMsg = errBuf;
			} else if (maskW <= 0 || model.image.cols < maskW) {
				snprintf(errBuf, sizeof(errBuf), "expected 0 < mask.width <= %d", model.image.cols);
				errMsg = errBuf;
			} else if (maskH <= 0 || model.image.rows < maskH) {
				snprintf(errBuf, sizeof(errBuf), "expected 0 < mask.height <= %d", model.image.rows);
				errMsg = errBuf;
			}
		}
	}

	Detect detect = DETECT_NONE;
	if (!errMsg && pDetect) {
		if (json_is_string(pDetect)) {
			if (strcmp("keypoints", json_string_value(pDetect)) == 0) {
				detect = DETECT_KEYPOINTS;
			} else if (strcmp("none", json_string_value(pDetect)) == 0) {
			  detect = DETECT_NONE;
			} else if (strcmp("rects", json_string_value(pDetect)) == 0) {
			  detect = DETECT_RECTS;
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
			mask = Mat::zeros(model.image.rows, model.image.cols, CV_8UC1);
			mask(maskRect) = 1;
		}
		vector<vector<Point> > regions;
		mser(model.image, regions, mask);

		int nRegions = (int) regions.size();
		LOGTRACE1("apply_MSER matched %d regions", nRegions);
		switch (detect) {
			case DETECT_RECTS:
				detectRects(pStageModel, regions);
				break;
			case DETECT_KEYPOINTS:
				detectKeypoints(pStageModel, regions);
				break;
		}
		if (jo_object(pStage, "color", model.argMap)) {
			if (model.image.channels() == 1) {
				cvtColor(model.image, model.image, CV_GRAY2BGR);
				LOGTRACE("cvtColor(CV_GRAY2BGR)");
			}
			drawRegions(model.image, regions, color);
			if (pMask) {
				if (color[0]==-1 && color[1]==-1 && color[2]==-1 && color[3]==-1) {
					rectangle(model.image, maskRect, Scalar(255, 0, 255));
				} else {
					rectangle(model.image, maskRect, color);
				}
			}
		}
	}

	return stageOK("apply_MSER(%s) %s", errMsg, pStage, pStageModel);
}

