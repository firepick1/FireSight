#include <string.h>
#include <math.h>
#include "FireLog.h"
#include "Pipeline.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "jo_util.hpp"

#include "detector.h"

using namespace cv;
using namespace std;
using namespace firesight;

/** (DEPRECATED: should be private) */
//void Pipeline::covarianceXY(const vector<Point> &pts, Mat &covOut, Mat &meanOut) {
//  LOGERROR("covarianceXY() is deprecated as a public function");
//  _covarianceXY(pts, covOut, meanOut);
//}

/**
 * Compute covariance and mean of region
 */
void MSERStage::_covarianceXY(const vector<Point> &pts, Mat &covOut, Mat &meanOut) {
  Mat_<double> data(pts.size(),2);
  for (size_t i = 0; i < pts.size(); i++) {
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
//void Pipeline::eigenXY(const vector<Point> &pts, Mat &eigenvectorsOut, Mat &meanOut, Mat &covOut) {
//  LOGERROR("eigenXY() is deprecated as a public function");
//  _eigenXY(pts, eigenvectorsOut, meanOut, covOut);
//}

/** 
 * Compute eigenvectors, eigenvalues, mean, and covariance of region
 */
void MSERStage::_eigenXY(const vector<Point> &pts, Mat &eigenvectorsOut, Mat &meanOut, Mat &covOut) {
  _covarianceXY(pts, covOut, meanOut);

  Mat eigenvalues;
  eigen(covOut, eigenvalues, eigenvectorsOut);

  LOGTRACE4("eigenXY() -> [%f,%f;%f,%f]",
    eigenvectorsOut.at<double>(0,0), eigenvectorsOut.at<double>(0,1),
    eigenvectorsOut.at<double>(1,0), eigenvectorsOut.at<double>(1,1));
}

/**
 * Using eigenvectors and covariance matrix of given region, return a
 * keypoint representing that region. The eigenvectors detemine the angle of the 
 * region on the interval (-PI/2,PI/2]. Caller may provide a fourth quadrant (negative angle)
 * offset of CV_PI for a vertical direction bias ("portrait regions"), 
 * or use the default of 2*CV_PI for a horizontal bias ("landscape regions").
 * @param region of points to analyze
 */
KeyPoint MSERStage::_regionKeypoint(const vector<Point> &region) {
  Mat covOut;
  Mat mean;
  Mat eigenvectors;
  _eigenXY(region, eigenvectors, mean, covOut);

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

  return KeyPoint((float) x, (float) y, (float) diam, (float) degrees);
}


void MSERStage::drawRegions(Mat &image, vector<vector<Point> > &regions, Scalar color) {
  int nRegions = (int) regions.size();
  int blue = (int) color[0];
  int green = (int) color[1];
  int red = (int) color[2];
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

void MSERStage::detectRects(json_t *pStageModel, vector<vector<Point> > &regions) {
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

void MSERStage::detectKeypoints(json_t *pStageModel, vector<vector<Point> > &regions) {
  int nRegions = regions.size();
  json_t *pKeypoints = json_array();
  json_object_set(pStageModel, "keypoints", pKeypoints);

  for (int i=0; i < nRegions; i++) {
    KeyPoint keypoint = _regionKeypoint(regions[i]);
    json_t *pKeypoint = json_object();
    json_object_set(pKeypoint, "pt.x", json_real(keypoint.pt.x));
    json_object_set(pKeypoint, "pt.y", json_real(keypoint.pt.y));
    json_object_set(pKeypoint, "size", json_real(keypoint.size));
    json_object_set(pKeypoint, "angle", json_real(keypoint.angle));
    json_array_append(pKeypoints, pKeypoint);
  }
}

