#include <string.h>
#include <fstream>
#include <sstream>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <boost/format.hpp>
#include "FireLog.h"
#include "FireSight.hpp"
#include "version.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "MatUtil.hpp"
#include "jo_util.hpp"

using namespace cv;
using namespace std;
using namespace FireSight;

#define MAX_RADIUS 128

namespace FireSight {
	extern short ringMap[MAX_RADIUS][MAX_RADIUS];
}

bool Pipeline::apply_warpRing(json_t *pStage, json_t *pStageModel, Model &model) {
  const char *angleStr = jo_string(pStage, "angle", "ring");
  bool grow = jo_bool(pStage, "grow", true);
	const char *errMsg = NULL;
	assert(0<model.image.rows && 0<model.image.cols);

	if (!errMsg) {
		matRing(model.image, model.image, grow);
	}

	return stageOK("apply_ring(%s) %s", errMsg, pStage, pStageModel);
}

void matRing(const Mat &image, Mat &result, bool grow) {
	int mx = image.cols - 1;
	int my = image.rows - 1;
	bool xodd = image.cols & 1;
	bool yodd = image.rows & 1;
	int cx = mx/2;
	int cx2 = xodd ? cx : cx+1;
	int cy = my/2;
	int cy2 = yodd ? cy : cy+1;
	short radius = max(sqrt(image.rows*image.rows+image.cols*image.cols)/2.0, 1.0);
	assert(radius<MAX_RADIUS);
	short sum1D[MAX_RADIUS];
	memset(sum1D, 0, sizeof(sum1D));
	short count1D[MAX_RADIUS];
	memset(count1D, 0, sizeof(count1D));

	for (int c=0; c<=cx; c++) {
		for (int r=0; r<=cy; r++) {
			short rcSum = image.at<uchar>(cy-r,cx-c); // top-left image
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
	if (logLevel >= FIRELOG_TRACE) {
		string info = matInfo(image);
	  LOGTRACE3("matRing() image %s cx:%d cy:%d", info.c_str(), cx, cy);
	}
	for (int i=0; i < radius; i++) {
		avg1D[i] = (short)(sum1D[i] / (float) count1D[i] + 0.5);
		LOGTRACE4("matRing()  avg1D[%d] = %d/%d = %d", i, (int)sum1D[i], (int)count1D[i], (int)avg1D[i]);
	}

	int rCols = image.cols;
	int rRows = image.rows;
	if (grow) {
		rCols = 2 * radius + (xodd ? -1 : 0);
		rRows = 2 * radius + (yodd ? -1 : 0);
		int dy = (rRows - image.rows)/2;
		int dx = (rCols - image.cols)/2;
		cx += dx;
		cy += dy;
		cx2 += dx;
		cy2 += dy;
		LOGTRACE4("matRing() grow dx:%d dy:%d cx:%d cy:%d", dx, dy, cx, cy);
	}
	result = Mat(rRows, rCols, image.depth(), Scalar(0));
	if (logLevel >= FIRELOG_TRACE) {
		string info = matInfo(result);
	  LOGTRACE3("matRing() result %s cx:%d cy:%d", info.c_str(), cx, cy);
	}
	for (int c=0; c<=cx; c++) {
		for (int r=0; r<=cy; r++) {
			int d = ringMap[r][c];
			short rcAvg = avg1D[d];
			if (rcAvg) {
				result.at<uchar>(cy-r,cx-c) = rcAvg;
				result.at<uchar>(cy-r,cx2+c) = rcAvg;
				result.at<uchar>(cy2+r,cx-c) = rcAvg;
				result.at<uchar>(cy2+r,cx2+c) = rcAvg;
			}
		}
	}
}

