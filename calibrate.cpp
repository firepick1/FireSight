#include <string.h>
#include <math.h>
#include <iostream>
#include <stdexcept>
#include "FireLog.h"
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "jansson.h"
#include "jo_util.hpp"
#include "MatUtil.hpp"
#include "version.h"

using namespace cv;
using namespace std;
using namespace firesight;

enum CompareOp {
    COMPARE_XY,
    COMPARE_YX
};

typedef class ComparePoint2f {
    private:
        CompareOp op;

    public:
        ComparePoint2f(CompareOp op=COMPARE_XY) {
            this->op = op;
        }
    public:
        bool operator()(const Point2f &lhs, const Point2f &rhs) const {
            assert(!isnan(lhs.x));
            assert(!isnan(rhs.x));
            int cmp;
            switch (op) {
            case COMPARE_XY:
                cmp = lhs.x - rhs.x;
                if (cmp == 0) {
                    cmp = lhs.y - rhs.y;
                }
                break;
            case COMPARE_YX:
                cmp = lhs.y - rhs.y;
                if (cmp == 0) {
                    cmp = lhs.x - rhs.x;
                }
                break;
            }
            return cmp < 0;
        }
} ComparePoint2f;

typedef map<Point2f,Point2f,ComparePoint2f> PointMap;

static void identifyColumns(PointMap &pointMapXY, float &dyMedian, Point2f &dyTot1, Point2f &dyTot2, 
	int &dyCount1, int &dyCount2, double tolerance) 
{
	vector<float> dyList;
	Point2f prevPt;
	for (PointMap::iterator it=pointMapXY.begin(); it!=pointMapXY.end(); it++) {
		if (it != pointMapXY.begin()) {
			float dy = prevPt.y - it->first.y;
			dyList.push_back(dy);
		}
		prevPt = it->first;
	}
	sort(dyList.begin(), dyList.end());
	dyMedian = dyList[dyList.size()/2];
	float maxTol = dyMedian < 0 ? 1-tolerance : 1+tolerance;
	float minTol = dyMedian < 0 ? 1+tolerance : 1-tolerance;
	float maxDy1 = dyMedian * maxTol;
	float minDy1 = dyMedian * minTol;
	float maxDy2 = 2*dyMedian * maxTol;
	float minDy2 = 2*dyMedian * minTol;

	Point2f prevPt1;
	Point2f prevPt2;
	int n = 0;
	for (map<Point2f,Point2f,ComparePoint2f>::iterator it=pointMapXY.begin();
			it!=pointMapXY.end(); it++) {
		const Point2f &curPt = it->first;
		if (n > 0) {
			LOGDEBUG3("matchGrid() pointMapXY[%d] (%g,%g)", n, curPt.x, curPt.y);
			int dy1 = prevPt1.y - curPt.y;
			if (minDy1 <= dy1 && dy1 <= maxDy1) {
				dyTot1 = dyTot1 + (prevPt1 - curPt);
				dyCount1++;
			}
			if (n > 1) {
				int dy2 = prevPt2.y - curPt.y;
				if (minDy2 <= dy2 && dy2 <= maxDy2) {
					dyTot2 = dyTot2 + (prevPt2 - curPt);
					dyCount2++;
				}
			}
		}
		prevPt2 = prevPt1;
		prevPt1 = curPt;
		n++;
	}
} // identifyColumns

static void identifyRows(PointMap &pointMapYX, float &dxMedian, Point2f &dxTot1, Point2f &dxTot2, 
	int &dxCount1, int &dxCount2, double tolerance) 
{
	vector<float> dxList;
	Point2f prevPt;
	for (PointMap::iterator it=pointMapYX.begin(); it!=pointMapYX.end(); it++) {
		if (it != pointMapYX.begin()) {
			float dx = prevPt.x - it->first.x;
			dxList.push_back(dx);
		}
		prevPt = it->first;
	}
	sort(dxList.begin(), dxList.end());
	dxMedian = dxList[dxList.size()/2];
	float maxTol = dxMedian < 0 ? 1-tolerance : 1+tolerance;
	float minTol = dxMedian < 0 ? 1+tolerance : 1-tolerance;
	float maxDx1 = dxMedian * maxTol;
	float minDx1 = dxMedian * minTol;
	float maxDx2 = 2*dxMedian * maxTol;
	float minDx2 = 2*dxMedian * minTol;

	Point2f prevPt1;
	Point2f prevPt2;
	int n = 0;
	for (map<Point2f,Point2f,ComparePoint2f>::iterator it=pointMapYX.begin();
			it!=pointMapYX.end(); it++) {
		const Point2f &curPt = it->first;
		if (n > 0) {
			LOGDEBUG3("matchGrid() pointMapYX[%d] (%g,%g)", n, curPt.x, curPt.y);
			int dx1 = prevPt1.x - curPt.x;
			if (minDx1 <= dx1 && dx1 <= maxDx1) {
				dxTot1 = dxTot1 + (prevPt1 - curPt);
				dxCount1++;
			}
			if (n > 1) {
				int dx2 = prevPt2.x - curPt.x;
				if (minDx2 <= dx2 && dx2 <= maxDx2) {
					dxTot2 = dxTot2 + (prevPt2 - curPt);
					dxCount2++;
				}
			}
		}
		prevPt2 = prevPt1;
		prevPt1 = curPt;
		n++;
	}
} // identifyRows

bool Pipeline::apply_matchGrid(json_t *pStage, json_t *pStageModel, Model &model) {
    string rectsModelName = jo_string(pStage, "model", "", model.argMap);
    double sepX = jo_double(pStage, "sepX", 5.0, model.argMap);
    double sepY = jo_double(pStage, "sepY", 5.0, model.argMap);
    double tolerance = jo_double(pStage, "tolerance", 0.35, model.argMap);
    json_t *pRectsModel = json_object_get(model.getJson(false), rectsModelName.c_str());
    string errMsg;

    if (rectsModelName.empty()) {
        errMsg = "matchGrid model: expected name of stage with rects";
    } else if (!json_is_object(pRectsModel)) {
        errMsg = "Named stage is not in model";
    }

    json_t *pRects = NULL;
    if (errMsg.empty()) {
        pRects = json_object_get(pRectsModel, "rects");
        if (!json_is_array(pRects)) {
            errMsg = "Expected array of rects to match";
        } else if (json_array_size(pRects) < 2) {
            errMsg = "Expected array of at least 2 rects to match";
        }
    }

    Point2f dxTot1;
    Point2f dyTot1;
    Point2f dxTot2;
    Point2f dyTot2;
    int dxCount1 = 0;
    int dxCount2 = 0;
    int dyCount1 = 0;
    int dyCount2 = 0;
    float dyMedian = FLT_MAX;
    float dxMedian = FLT_MAX;
    const ComparePoint2f cmpXY(COMPARE_XY);
    const ComparePoint2f cmpYX(COMPARE_YX);
    if (errMsg.empty()) {
        PointMap pointMapXY(cmpXY);
        PointMap pointMapYX(cmpYX);
        json_t *pValue;
        int index;
        json_array_foreach(pRects, index, pValue) {
            json_t *pX = json_object_get(pValue, "x");
            json_t *pY = json_object_get(pValue, "y");
            if (json_is_number(pX) && json_is_number(pY)) {
                double x = json_real_value(pX);
                double y = json_real_value(pY);
                const Point2f key(x,y);
                pointMapXY[key] = Point2f(x,y);
                pointMapYX[key] = Point2f(x,y);
            }
        }

		identifyColumns(pointMapXY, dyMedian, dyTot1, dyTot2, dyCount1, dyCount2, tolerance);
		identifyRows(pointMapYX, dxMedian, dxTot1, dxTot2, dxCount1, dxCount2, tolerance);

        json_object_set(pStageModel, "dxMedian", json_real(dxMedian));
        json_object_set(pStageModel, "dxCount1", json_integer(dxCount1));
        json_object_set(pStageModel, "dxCount2", json_integer(dxCount2));
        json_object_set(pStageModel, "dyMedian", json_real(dyMedian));
        json_object_set(pStageModel, "dyCount1", json_integer(dyCount1));
        json_object_set(pStageModel, "dyCount2", json_integer(dyCount2));
        if (dxCount1 == 0) {
            errMsg = "No grid points matched within tolerance (level 1) dxCount1:0";
        } else if (dxCount2 == 0) {
            json_object_set(pStageModel, "dxAvg1", json_real(dxTot1.x/dxCount1));
            json_object_set(pStageModel, "dyAvg1", json_real(dxTot1.y/dxCount1));
            errMsg = "No grid points matched within tolerance (level 2) dxCount2:0";
        } else {
            float dxAvg1 = dxTot1.x/dxCount1;
            float dyAvg1 = dxTot1.y/dxCount1;
            float dxAvg2 = dxTot2.x/dxCount2/2;
            float dyAvg2 = dxTot2.y/dxCount2/2;
            json_object_set(pStageModel, "dxdxAvg1", json_real(dxAvg1));
            json_object_set(pStageModel, "dxdyAvg1", json_real(dyAvg1));
            json_object_set(pStageModel, "dxdxAvg2", json_real(dxAvg2));
            json_object_set(pStageModel, "dxdyAvg2", json_real(dyAvg2));
            float normXY = sqrt(dxAvg2*dxAvg2 + dyAvg2*dyAvg2);
            json_object_set(pStageModel, "gridX", json_real(normXY/sepX));
        }
        if (dyCount1 == 0) {
            errMsg = "No grid points matched within tolerance (level 1) dyCount1:0";
        } else if (dyCount2 == 0) {
            json_object_set(pStageModel, "dxAvg1", json_real(dyTot1.x/dyCount1));
            json_object_set(pStageModel, "dyAvg1", json_real(dyTot1.y/dyCount1));
            errMsg = "No grid points matched within tolerance (level 2) dyCount2:0";
        } else {
            float dxAvg1 = dyTot1.x/dyCount1;
            float dyAvg1 = dyTot1.y/dyCount1;
            float dxAvg2 = dyTot2.x/dyCount2/2;
            float dyAvg2 = dyTot2.y/dyCount2/2;
            json_object_set(pStageModel, "dydxAvg1", json_real(dxAvg1));
            json_object_set(pStageModel, "dydyAvg1", json_real(dyAvg1));
            json_object_set(pStageModel, "dydxAvg2", json_real(dxAvg2));
            json_object_set(pStageModel, "dydyAvg2", json_real(dyAvg2));
            float normXY = sqrt(dxAvg2*dxAvg2 + dyAvg2*dyAvg2);
            json_object_set(pStageModel, "gridY", json_real(normXY/sepY));
        }
    }

    return stageOK("apply_matchGrid(%s) %s", errMsg.c_str(), pStage, pStageModel);
}

