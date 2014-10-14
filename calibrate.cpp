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

class ComparePoint2f {
    private:
        bool isXY;

    public:
        ComparePoint2f(bool isXY=true) {
            this->isXY = isXY;
        }
    public:
        bool operator()(const Point2f &lhs, const Point2f &rhs) const {
			assert(!isnan(lhs.x));
			assert(!isnan(rhs.x));
			int cmp;
			if ( isXY ) {
				cmp = lhs.x - rhs.x;
				if (cmp == 0) {
					cmp = lhs.y - rhs.y;
				}
			} else {
				cmp = lhs.y - rhs.y;
				if (cmp == 0) {
					cmp = lhs.x - rhs.x;
				}
			}
            return cmp < 0;
        }
};

typedef map<Point2f,Point2f,ComparePoint2f> PointMap;


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

    Point2f dTot1;
    Point2f dTot2;
    int dCount1 = 0;
    int dCount2 = 0;
    float dyMedian = FLT_MAX;
    const ComparePoint2f cmp;
    if (errMsg.empty()) {
        PointMap pointMap(cmp);
        json_t *pValue;
        int index;
        json_array_foreach(pRects, index, pValue) {
            json_t *pX = json_object_get(pValue, "x");
            json_t *pY = json_object_get(pValue, "y");
            if (json_is_number(pX) && json_is_number(pY)) {
                double x = json_real_value(pX);
                double y = json_real_value(pY);
                const Point2f key(x,y);
				cout << "adding " << key << " to pointMap(" << pointMap.size() << ")" << endl;
                pointMap[key] = Point2f(x,y);
            }
        }

		vector<float> dyList;
        Point2f prevPt;
        for (PointMap::iterator it=pointMap.begin(); it!=pointMap.end(); it++) {
            if (it != pointMap.begin()) {
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
        for (map<Point2f,Point2f,ComparePoint2f>::iterator it=pointMap.begin();
                it!=pointMap.end(); it++) {
            const Point2f &curPt = it->first;
            cout << "(" << curPt.x << "," << curPt.y << ")" << endl;
            if (n > 0) {
                int dy1 = prevPt1.y - curPt.y;
                if (minDy1 <= dy1 && dy1 <= maxDy1) {
                    dTot1 = dTot1 + (prevPt1 - curPt);
                    dCount1++;
                }
                if (n > 1) {
                    int dy2 = prevPt2.y - curPt.y;
                    if (minDy2 <= dy2 && dy2 <= maxDy2) {
                        dTot2 = dTot2 + (prevPt2 - curPt);
                        dCount2++;
                    }
                }
            }
            prevPt2 = prevPt1;
            prevPt1 = curPt;
			n++;
        }
        json_object_set(pStageModel, "dyMedian", json_real(dyMedian));
        json_object_set(pStageModel, "dCount1", json_integer(dCount1));
        json_object_set(pStageModel, "dCount2", json_integer(dCount2));
        if (dCount1 == 0) {
            errMsg = "No grid points matched within tolerance (level 1)";
        } else if (dCount2 == 0) {
            json_object_set(pStageModel, "dxAvg1", json_real(dTot1.x/dCount1));
            json_object_set(pStageModel, "dyAvg1", json_real(dTot1.y/dCount1));
            errMsg = "No grid points matched within tolerance (level 2)";
        } else {
            float dxAvg2 = dTot2.x/dCount2/2;
            float dyAvg2 = dTot2.y/dCount2/2;
            json_object_set(pStageModel, "dxAvg1", json_real(dTot1.x/dCount1));
            json_object_set(pStageModel, "dyAvg1", json_real(dTot1.y/dCount1));
            json_object_set(pStageModel, "dxAvg2", json_real(dxAvg2));
            json_object_set(pStageModel, "dyAvg2", json_real(dyAvg2));
            float gridY = sqrt(dxAvg2*dxAvg2 + dyAvg2*dyAvg2) / sepY;
            float gridX = sqrt(dxAvg2*dxAvg2 + dyAvg2*dyAvg2) / sepX;
            json_object_set(pStageModel, "gridY", json_real(gridY));
            json_object_set(pStageModel, "gridX", json_real(gridX));
        }
    }

    return stageOK("apply_matchGrid(%s) %s", errMsg.c_str(), pStage, pStageModel);
}

