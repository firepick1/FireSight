#include <string.h>
#include <iostream>
#include <math.h>
#include "FireLog.h"
#include "Pipeline.h"
#include "version.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "MatUtil.hpp"
#include "jo_util.hpp"

using namespace cv;
using namespace std;
using namespace firesight;

#define MAX_RADIUS 128

namespace firesight {
  extern short ringMap[MAX_RADIUS][MAX_RADIUS];
}

void matWarpRing(const Mat &image, Mat &result, vector<float> angles) {
  if (angles.size() == 0) { // ring
    matRing(image, result);
  } else { // discrete angles
    Size imageSize(image.cols, image.rows);
    float cx = (image.cols-1)/2.0f;
    float cy = (image.rows-1)/2.0f;
    Point2f center(cx,cy);
    float all_minx;
    float all_maxx;
    float all_miny;
    float all_maxy;
    for (size_t i=0; i<angles.size(); i++) {
      float minx;
      float maxx;
      float miny;
      float maxy;
      matRotateSize(imageSize, center, angles[i], minx, maxx, miny, maxy, 1);
      if (i == 0) {
        all_minx = minx;
        all_miny = miny;
        all_maxx = maxx;
        all_maxy = maxy;
      } else {
        all_minx = min(all_minx, minx);
        all_miny = min(all_miny, miny);
        all_maxx = max(all_maxx, maxx);
        all_maxy = max(all_maxy, maxy);
      }
      LOGTRACE4("matWarpRing() all_minx:%f all_maxx:%f all_miny:%f all_maxy:%f", all_minx, all_maxx,  all_miny, all_maxy);
    }
    Size resultSize((int)(all_maxx - all_minx + 1.5), (int)(all_maxy - all_miny + 1.5));
    LOGTRACE2("matWarpRing() resultSize.width:%d resultSize.height:%d", resultSize.width, resultSize.height);
    int sumType = CV_MAKETYPE(CV_32F, image.channels());
    Mat resultSum(resultSize.height, resultSize.width, sumType, Scalar(0));
    Point2f translate((resultSize.width-1.0f)/2 - cx, (resultSize.height-1.0f)/2 - cy);
    for (size_t i=0; i<angles.size(); i++) {
      float angle = angles[i];
      Mat localResult;
      matWarpAffine(image, localResult, center, angle, 1, translate, resultSize);
      if (localResult.type() != sumType ) {
        localResult.convertTo(localResult, sumType);
      }
      resultSum += localResult;
    }
    float scale = 1.0f/angles.size();
    resultSum = resultSum * scale;
    resultSum.convertTo(result, CV_MAKETYPE(image.type(), image.channels()));
  }
  LOGTRACE1("matWarpRing() => %s", matInfo(result).c_str()); 
}

void matRing(const Mat &image, Mat &result) {
  int mx = image.cols - 1;
  int my = image.rows - 1;
  bool xodd = image.cols & 1;
  bool yodd = image.rows & 1;
  int cx = mx/2; // 1x1=>0; 2x2=>0; 3x3=>1; 4x4=>1
  int cx2 = xodd ? cx : cx+1;
  int cy = my/2; // 1x1=>0; 2x2=>0; 3x3=>1; 4x4=>1
  int cy2 = yodd ? cy : cy+1;
  short radius = (short) max(ceil(sqrt((float) mx*mx+my*my)/2.0), 1.0);
  assert(radius<MAX_RADIUS);
  int sum1D[MAX_RADIUS];
  memset(sum1D, 0, sizeof(sum1D));
  short count1D[MAX_RADIUS];
  memset(count1D, 0, sizeof(count1D));

  for (int c=0; c<=cx; c++) {
    for (int r=0; r<=cy; r++) {
      int rcSum = image.at<uchar>(cy-r,cx-c); // top-left image
      short rcCount = 1;
      if (!xodd || c) { // top-right image
        rcSum += image.at<uchar>(cy-r,cx2+c);
        rcCount++;
      }
      if (!yodd || r) { // bottom-left image
        rcSum += image.at<uchar>(cy2+r,cx-c);
        rcCount++;
      }
      if (r && c || !xodd && !yodd) { // bottom-right image
        rcSum += image.at<uchar>(cy2+r,cx2+c);
        rcCount++;
      }
      short d = ringMap[r][c];
      count1D[d] += rcCount;
      sum1D[d] += rcSum;
    }
  }
  short avg1D[MAX_RADIUS];
  memset(avg1D,0,sizeof(avg1D));
  LOGTRACE3("matRing() image %s cx:%d cy:%d", matInfo(image).c_str(), cx, cy);
  for (int i=0; i < radius; i++) {
    avg1D[i] = count1D[i] ? ((short)(sum1D[i] / (float) count1D[i] + 0.5)) : 0;
    LOGTRACE4("matRing()  avg1D[%d] = %d/%d = %d", i, (int)sum1D[i], (int)count1D[i], (int)avg1D[i]);
  }

  int rCols = image.cols;
  int rRows = image.rows;
  rCols = 2 * radius + (xodd ? -1 : 0);
  rRows = 2 * radius + (yodd ? -1 : 0);
  int dy = (rRows - image.rows)/2;
  int dx = (rCols - image.cols)/2;
  cx += dx;
  cy += dy;
  cx2 += dx;
  cy2 += dy;
  LOGTRACE4("matRing() dx:%d dy:%d cx:%d cy:%d", dx, dy, cx, cy);

  result = Mat(rRows, rCols, image.depth(), Scalar(0,0,0));
  LOGTRACE3("matRing() result %s cx:%d cy:%d", matInfo(result).c_str(), cx, cy);
  short rcAvgExtend = avg1D[MAX_RADIUS-1];
  for (int r=0; r<=cy; r++) {
    for (int c=0; c<=cx; c++) {
      int d = ringMap[r][c];
      uchar rcAvg = (uchar)(d >= MAX_RADIUS ? rcAvgExtend : avg1D[d]);
      if (rcAvg) {
        result.at<uchar>(cy-r,cx-c) = rcAvg;
        result.at<uchar>(cy-r,cx2+c) = rcAvg;
        result.at<uchar>(cy2+r,cx-c) = rcAvg;
        result.at<uchar>(cy2+r,cx2+c) = rcAvg;
      }
    }
  }
  LOGTRACE1("matRing() => %s", matInfo(result).c_str()); 
}

