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
	bool operator()(const Point2f &lhs, Point2f &rhs) {
		int cmp = lhs.x - rhs.x;
		if (cmp == 0) {
			cmp = lhs.y - rhs.y;
		}
		return cmp < 0;
	}
};

typedef struct GridPoint {
	float x;
	float y;
	float dx;
	GridPoint(float dx=1) {
		this->x = FLT_MAX;
		this->y = FLT_MAX;
		this->dx = dx;
	}
	GridPoint(float x, float y, float dx=1) {
		this->x = x;
		this->y = y;
		this->dx = dx;
	}
	inline bool isnan() { return x==FLT_MAX || y==FLT_MAX; }
	inline GridPoint operator-(const GridPoint &that) {
		return GridPoint(x-that.x,y-that.y,dx);
	}
	inline GridPoint operator+(const GridPoint &that) {
		return GridPoint(x+that.x,y+that.y,dx);
	}
	inline GridPoint& operator=(const GridPoint &that) {
		this->x = that.x;
		this->y = that.y;
		this->dx = that.dx;
		return *this;
	}
	inline bool friend operator<(const GridPoint &lhs, const GridPoint &rhs) {
		int cmp = (lhs.x - rhs.x)/lhs.dx;
		if (cmp == 0) {
			cmp = lhs.y - rhs.y;
		}
		return cmp < 0;
	}
} GridPoint;


bool Pipeline::apply_matchGrid(json_t *pStage, json_t *pStageModel, Model &model) {
    string rectsModelName = jo_string(pStage, "model", "", model.argMap);
	double sepX = jo_double(pStage, "sepX", 5.0, model.argMap); 
	double sepY = jo_double(pStage, "sepY", 5.0, model.argMap); 
	double dx = jo_double(pStage, "dx", 1.0, model.argMap); 
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

	GridPoint dTot1(0,0,dx);
	GridPoint dTot2(0,0,dx);
	int dCount1 = 0;
	int dCount2 = 0;
	float dyMedian = FLT_MAX;
    if (errMsg.empty()) {
		map<GridPoint,GridPoint> pointMap;
		json_t *pValue;
		int index;
		json_array_foreach(pRects, index, pValue) {
			json_t *pX = json_object_get(pValue, "x");
			json_t *pY = json_object_get(pValue, "y");
			if (json_is_number(pX) && json_is_number(pY)) {
				double x = json_real_value(pX);
				double y = json_real_value(pY);
				GridPoint key(x,y,dx);
				pointMap[key] = GridPoint(x,y,dx);
			}
		}

		map<float,float> dyMap;
		GridPoint prevPt;
		for (map<GridPoint,GridPoint>::iterator it=pointMap.begin(); it!=pointMap.end(); it++) {
			if (it != pointMap.begin()) {
				float dy = prevPt.y - it->first.y;
				dyMap[dy] = dy;
			}
			prevPt = it->first;
		}
		int iMedian = dyMap.size()/2;
		cout << "iMedian" << iMedian << endl;
		map<float,float>::iterator itMedian1=dyMap.begin();
		for (int i=0; i<iMedian; i++) { 
			cout << "itMedian1" <<  itMedian1->first << endl;
			itMedian1++; 
		}
		map<float,float>::iterator itMedian2 = itMedian1;
		itMedian2++;
		dyMedian = itMedian1->first - itMedian2->first;
		float maxTol = dyMedian < 0 ? 1-tolerance : 1+tolerance;
		float minTol = dyMedian < 0 ? 1+tolerance : 1-tolerance;
		float maxDy1 = dyMedian * maxTol;
		float minDy1 = dyMedian * minTol;
		float maxDy2 = 2*dyMedian * maxTol;
		float minDy2 = 2*dyMedian * minTol;

		GridPoint prevPt1;
		GridPoint prevPt2;
		for (map<GridPoint,GridPoint>::iterator it=pointMap.begin(); it!=pointMap.end(); it++) {
			const GridPoint &curPt = it->first;
			cout << "(" << curPt.x << "," << curPt.y << ")" << endl;
			if (!prevPt1.isnan()) {
				int dy1 = prevPt1.y - curPt.y;
				if (minDy1 <= dy1 && dy1 <= maxDy1) {
					dTot1 = dTot1 + (prevPt1 - curPt);
					dCount1++;
				}
				if (!prevPt2.isnan()) {
					int dy2 = prevPt2.y - curPt.y;
					if (minDy2 <= dy2 && dy2 <= maxDy2) {
						dTot2 = dTot2 + (prevPt2 - curPt);
						dCount2++;
					}
				}
			}
			prevPt2 = prevPt1;
			prevPt1 = curPt;
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

