#include <string.h>
#include <math.h>
#include <set>
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

json_t * json_matrix(const Mat &mat) {
    json_t *jmat = json_array();
    for (int r=0; r < mat.rows; r++) {
        for (int c=0; c < mat.cols; c++) {
            json_array_append(jmat, json_real(mat.at<double>(r,c)));
        }
    }
    return jmat;
}

enum CalibrateOp {
	CAL_TILE,
	CAL_CELTIC_CROSS,

	CAL_DEFAULT,
	CAL_NONE,
	CAL_FULL,
	CAL_XYAXES,
	CAL_QUADRANT,
	CAL_CROSS
};

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

static void initCameraVectors(vector<double> &cmDefault, vector<double> &dcDefault,
                              Mat &image)
{
    cmDefault.push_back(1.0);
    cmDefault.push_back(0.0);
    cmDefault.push_back(image.cols/2);
    cmDefault.push_back(0.0);
    cmDefault.push_back(1.0);
    cmDefault.push_back(image.rows/2);
    cmDefault.push_back(0.0);
    cmDefault.push_back(0.0);
    cmDefault.push_back(1.0);

    dcDefault.push_back(0);
    dcDefault.push_back(0);
    dcDefault.push_back(0);
    dcDefault.push_back(0);
}

typedef struct GridMatcher {
    vector<Point2f> imagePts;
    vector<Point3f> objectPts;
    Point3f objTotals;
    Point2f imgTotals;
    Rect imgRect;
    const ComparePoint2f cmpYX;
    set<Point2f,ComparePoint2f> imgSet;
    vector<vector<Point2f> > vImagePts;
    vector<vector<Point3f> > vObjectPts;
    Size imgSize;
    Point2f imgSep;
    Point2f objSep;
    Mat gridIndexes;	// object grid matrix of imagePts/objectPts vector indexes or -1
    GridMatcher(Size imgSize, Point2f imgSep, Point2f objSep)
        : cmpYX(ComparePoint2f(COMPARE_YX)), imgSet(set<Point2f,ComparePoint2f>(cmpYX)) {
        this->imgSize = imgSize;
        this->imgSep = imgSep;
        this->objSep = objSep;
        this->imgRect = Rect(imgSize.width/2, imgSize.height/2, 0, 0);
    }

    bool add(Point2f &ptImg, Point3f &ptObj) {
        set<Point2f,ComparePoint2f>::iterator it = imgSet.find(ptImg);

        if (it != imgSet.end()) {
            return false;
        }
        objectPts.push_back(ptObj);
        imagePts.push_back(ptImg);
        objTotals += ptObj;
        imgTotals += ptImg;
        imgSet.insert(ptImg);
        return true;
    }

    void calcGridIndexes() {
        int ny = imgSize.height/imgSep.y+1.5;
        int nx = imgSize.width/imgSep.x+1.5;
        gridIndexes = Mat(Size(nx,ny),CV_16S);
        gridIndexes = -1;
        LOGTRACE2("calcGridIndexes() [%d,%d]", ny, nx);
        for (short i=0; i < objectPts.size(); i++) {
            int r = objectPts[i].y;
            int c = objectPts[i].x;
            assert( 0 <= r && r < gridIndexes.rows);
            assert( 0 <= c && c < gridIndexes.cols);
            gridIndexes.at<short>(r,c) = i;
        }
    }

    bool addSubImage(int row, int col, int rows, int cols, int minPts) {
        vector<Point2f> subImgPts;
        vector<Point3f> subObjPts;
        float cy = (rows-1)/2.0;
        float cx = (cols-1)/2.0;
        float z = 0;
        for (int r=0; r < rows; r++) {
            for (int c=0; c < cols; c++) {
                short index = gridIndexes.at<short>(r+row, c+col);
                if (0 <= index) {
                    Point2f ptImg = imagePts[index];
                    Point3f ptObj = objectPts[index];
                    Point3f subObjPt(objSep.x*(ptObj.x-cx), objSep.y*(ptObj.y-cy), z);
    //                cout << "index:" << index << " r:" << r << " c:" << c
     //                    << " ptImg:" << ptImg << " subObjPt:" << subObjPt << endl;
                    subObjPts.push_back(subObjPt);
                    subImgPts.push_back(ptImg);
                }
            }
        }
        if (subImgPts.size() < minPts) {
			LOGTRACE3("addSubImage(%d,%d) REJECT:%ld", row, col, (long) subObjPts.size());
            return false;
        }
		LOGTRACE3("addSubImage(%d,%d) ADD:%ld", row, col, (long) subObjPts.size());
        vObjectPts.push_back(subObjPts);
        vImagePts.push_back(subImgPts);
        return true;
    }

    int size() {
        return objectPts.size();
    }

    Point2f getImageCentroid() {
        int n = objectPts.size();
        return Point2f(imgTotals.x/n, imgTotals.y/n);
    }

    Point3f getObjectCentroid() {
        int n = objectPts.size();
        return Point3f(objTotals.x/n, objTotals.y/n, objTotals.z/n);
    }

    void subImageXYAxesFactory() {
        int minPts = 4;
        int xh = max(3, gridIndexes.rows/4);
        int yw = max(3, gridIndexes.cols/4);
        int c2 = gridIndexes.cols/2;
        int r2 = gridIndexes.rows/2;
        int cLast = gridIndexes.cols - c2;
        int rLast = gridIndexes.rows - r2;

        addSubImage(rLast/2, 0, xh, c2, minPts);
        addSubImage(rLast/2, cLast, xh, c2, minPts);
        addSubImage(0, cLast/2, r2, yw, minPts);
        addSubImage(rLast, cLast/2, r2, yw, minPts);
    }

    void subImageQuadrantFactory() {
        int minPts = 4;
        int qw = gridIndexes.cols/2;
        int qh = gridIndexes.rows/2;

        addSubImage(0, 					0, 					qh, qw, minPts);
        addSubImage(0, 					gridIndexes.cols-qw,qh, qw, minPts);
        addSubImage(gridIndexes.rows-qh,0, 					qh, qw, minPts);
        addSubImage(gridIndexes.rows-qh,gridIndexes.cols-qw,qh, qw, minPts);
    }

    void subImageCrossFactory(int dMajor, int dMinor) {
        int minPts = 4;
        int rLast = gridIndexes.rows-dMajor;
        int cLast = gridIndexes.cols-dMajor;

        addSubImage(rLast/2, cLast/2, dMinor, dMajor, minPts);
        addSubImage(rLast/2, cLast/2, dMajor, dMinor, minPts);

        addSubImage(rLast/2, max(0,cLast/2-1), dMinor, dMajor, minPts);
        addSubImage(rLast/2, min(cLast,cLast/2+1), dMinor, dMajor, minPts);
        addSubImage(max(0,rLast/2-1), cLast/2, dMajor, dMinor, minPts);
        addSubImage(min(rLast,rLast/2+1), cLast/2, dMajor, dMinor, minPts);

        addSubImage(rLast/2, max(0,cLast/2-2), dMinor, dMajor, minPts);
        addSubImage(rLast/2, min(cLast,cLast/2+2), dMinor, dMajor, minPts);
        addSubImage(max(0,rLast/2-2), cLast/2, dMajor, dMinor, minPts);
        addSubImage(min(rLast,rLast/2+2), cLast/2, dMajor, dMinor, minPts);

        addSubImage(rLast/2, max(0,cLast/2-3), dMinor, dMajor, minPts);
        addSubImage(rLast/2, min(cLast,cLast/2+3), dMinor, dMajor, minPts);
        addSubImage(max(0,rLast/2-3), cLast/2, dMajor, dMinor, minPts);
        addSubImage(min(rLast,rLast/2+3), cLast/2, dMajor, dMinor, minPts);
    }

    string calibrateImage(json_t *pStageModel, Mat &cameraMatrix, Mat &distCoeffs,
                          Mat &image, CalibrateOp op=CAL_DEFAULT)
    {
        json_t *pCalibrate = json_object();
        json_object_set(pStageModel, "calibrate", pCalibrate);
        string errMsg;
        vObjectPts.clear();
        vImagePts.clear();

        calcGridIndexes();

		if (op == CAL_DEFAULT) {
			op = CAL_NONE; // may change
		}

        switch (op) {
        default:
        case CAL_NONE:
            break;
        case CAL_XYAXES:
            subImageXYAxesFactory();
            break;
        case CAL_CROSS:
            subImageCrossFactory(6, 3);
            break;
        case CAL_QUADRANT:
            subImageQuadrantFactory();
            break;
        case CAL_FULL:
            addSubImage(0, 0, gridIndexes.rows, gridIndexes.cols, 4);
            break;
        }

        double rmserror = 0;
        vector<Mat> rvecs;
        vector<Mat> tvecs;

        if (op == CAL_NONE) {
            vector<double> cmDefault;
            vector<double> dcDefault;
            initCameraVectors(cmDefault, dcDefault, image);
            cameraMatrix = Mat(3, 3, CV_64F);
			for (int i=0; i<cmDefault.size(); i++) {
				cameraMatrix.at<double>(i/3, i%3) = cmDefault[i];
			}
            distCoeffs = Mat(dcDefault);
        } else {
            try {
                rmserror = calibrateCamera(vObjectPts, vImagePts, imgSize,
                                           cameraMatrix, distCoeffs, rvecs, tvecs);
            } catch (cv::Exception ex) {
                errMsg = "calibrateImage(FAILED) ";
                errMsg += ex.msg;
            } catch (...) {
                errMsg = "calibrateImage(FAILED...)";
            }
        }

        json_object_set(pCalibrate, "cameraMatrix", json_matrix(cameraMatrix));
        json_object_set(pCalibrate, "distCoeffs", json_matrix(distCoeffs));
        json_object_set(pCalibrate, "rmserror", json_real(rmserror));
        json_object_set(pCalibrate, "images", json_real(vImagePts.size()));
#ifdef RVECS_TVECS
        json_t *pRvecs = json_array();
        json_object_set(pCalibrate, "rvecs", pRvecs);
        for (int i=0; i < rvecs.size(); i++) {
            json_array_append(pRvecs, json_matrix(rvecs[i]));
        }
        json_t *pTvecs = json_array();
        json_object_set(pCalibrate, "tvecs", pTvecs);
        for (int i=0; i < tvecs.size(); i++) {
            json_array_append(pTvecs, json_matrix(tvecs[i]));
        }
#endif

        return errMsg;
    }
} GridMatcher;

typedef map<Point2f,Point2f,ComparePoint2f> PointMap;

static string identifyRows(json_t *pStageModel, vector<Point2f> &pointsXY, float &dyMedian, Point2f &dyTot1,
                           Point2f &dyTot2, int &dyCount1, int &dyCount2, double tolerance, int sepY, float &gridY)
{
    vector<float> dyList;
    Point2f prevPt;
    for (vector<Point2f>::iterator it=pointsXY.begin(); it!=pointsXY.end(); it++) {
        if (it != pointsXY.begin()) {
            float dy = prevPt.y - it->y;
            dyList.push_back(dy);
        }
        prevPt = *it;
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
    for (vector<Point2f>::iterator it=pointsXY.begin();
            it!=pointsXY.end(); it++) {
        const Point2f &curPt = *it;
        if (n > 0) {
            LOGDEBUG3("matchGrid() pointsXY[%d] (%g,%g)", n, curPt.x, curPt.y);
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

    string errMsg;
    json_object_set(pStageModel, "dyMedian", json_real(dyMedian));
    json_object_set(pStageModel, "dyCount1", json_integer(dyCount1));
    json_object_set(pStageModel, "dyCount2", json_integer(dyCount2));
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
        gridY = normXY / sepY;
        json_object_set(pStageModel, "gridY", json_real(gridY));
    }

    return errMsg;
} // identifyRows

static string identifyColumns(json_t *pStageModel, vector<Point2f> &pointsYX, float &dxMedian, Point2f &dxTot1,
                              Point2f &dxTot2, int &dxCount1, int &dxCount2, double tolerance, int sepX, float &gridX)
{
    vector<float> dxList;
    Point2f prevPt;
    for (vector<Point2f>::iterator it=pointsYX.begin(); it!=pointsYX.end(); it++) {
        if (it != pointsYX.begin()) {
            float dx = prevPt.x - it->x;
            dxList.push_back(dx);
        }
        prevPt = *it;
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
    for (vector<Point2f>::iterator it=pointsYX.begin();
            it!=pointsYX.end(); it++) {
        const Point2f &curPt = *it;
        if (n > 0) {
            LOGDEBUG3("matchGrid() pointsYX[%d] (%g,%g)", n, curPt.x, curPt.y);
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

    string errMsg;
    json_object_set(pStageModel, "dxMedian", json_real(dxMedian));
    json_object_set(pStageModel, "dxCount1", json_integer(dxCount1));
    json_object_set(pStageModel, "dxCount2", json_integer(dxCount2));
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
        gridX = normXY / sepX;
        json_object_set(pStageModel, "gridX", json_real(gridX));
    }

    return errMsg;
} // identifyColumns

void initializePointMaps(json_t *pRects, vector<Point2f> &pointsXY, vector<Point2f> &pointsYX) {
    json_t *pValue;
    int index;
    json_array_foreach(pRects, index, pValue) {
        json_t *pX = json_object_get(pValue, "x");
        json_t *pY = json_object_get(pValue, "y");
        if (json_is_number(pX) && json_is_number(pY)) {
            double x = json_real_value(pX);
            double y = json_real_value(pY);
            const Point2f key(x,y);
            pointsXY.push_back(key);
            pointsYX.push_back(key);
        }
    }
    const ComparePoint2f cmpXY(COMPARE_XY);
    sort(pointsXY, cmpXY);
    const ComparePoint2f cmpYX(COMPARE_YX);
    sort(pointsYX, cmpYX);
}

inline Point3f calcObjPointDiff(const Point2f &curPt, const Point2f &prevPt, const Point2f &imgSep) {
    float dObjX = (curPt.x-prevPt.x)/imgSep.x;
    float dObjY = (curPt.y-prevPt.y)/imgSep.y;
    dObjX += dObjX < 0 ? -0.5 : 0.5;
    dObjY += dObjY < 0 ? -0.5 : 0.5;
    return Point3f((int)(dObjX), (int) (dObjY), 0);
}

bool Pipeline::apply_matchGrid(json_t *pStage, json_t *pStageModel, Model &model) {
    string rectsModelName = jo_string(pStage, "model", "", model.argMap);
    string opStr = jo_string(pStage, "calibrate", "none", model.argMap);
    Point2f objSep(
        jo_double(pStage, "sepX", 5.0, model.argMap),
        jo_double(pStage, "sepY", 5.0, model.argMap));
    double tolerance = jo_double(pStage, "tolerance", 0.35, model.argMap);
    Size imgSize(model.image.cols, model.image.rows);
    Point2f imgCenter(model.image.cols/2.0, model.image.rows/2.0);
    json_t *pRectsModel = json_object_get(model.getJson(false), rectsModelName.c_str());
    string errMsg;
    CalibrateOp op;

    if (opStr.compare("none") == 0) {
        op = CAL_NONE;
    } else if (opStr.compare("full") == 0) {
        op = CAL_FULL;
    } else if (opStr.compare("quadrant") == 0) {
        op = CAL_QUADRANT;
    } else if (opStr.compare("cross") == 0) {
        op = CAL_CROSS;
    } else if (opStr.compare("xyaxes") == 0) {
        op = CAL_XYAXES;
    }

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
    float gridX = FLT_MAX;
    float gridY = FLT_MAX;
    Point2f dmedian(FLT_MAX,FLT_MAX);
    const ComparePoint2f cmpYX(COMPARE_YX);
    vector<Point2f> pointsXY;
    vector<Point2f> pointsYX;
    Mat cameraMatrix;
    Mat distCoeffs;

    if (errMsg.empty()) {
        initializePointMaps(pRects, pointsXY, pointsYX);
        errMsg = identifyColumns(pStageModel, pointsYX, dmedian.x, dxTot1, dxTot2,
                                 dxCount1, dxCount2, tolerance, objSep.x, gridX);
        string errMsg2 = identifyRows(pStageModel, pointsXY, dmedian.y, dyTot1, dyTot2,
                                      dyCount1, dyCount2, tolerance, objSep.y, gridY);

        if (errMsg.empty()) {
            errMsg = errMsg2;
        } else if (!errMsg2.empty()) {
            errMsg.append("; ");
            errMsg.append(errMsg2);
        }
    }

    if (errMsg.empty()) {
        Point3f ptObj;
        Point2f ptImg;

        float maxDx1 = dmedian.x * (dmedian.x < 0 ? 1-tolerance : 1+tolerance);
        float minDx1 = dmedian.x * (dmedian.x < 0 ? 1+tolerance : 1-tolerance);
        Point2f imgSep(gridX*objSep.x, gridY*objSep.y);
        GridMatcher gm(imgSize, imgSep, objSep);

        vector<Point2f>::iterator itYX = pointsYX.begin();
        for (Point2f ptImg0=(*itYX); ++itYX!=pointsYX.end(); ) {
            const Point2f &ptImg1(*itYX);
            int dx1 = ptImg0.x - ptImg1.x;
            if (minDx1 <= dx1 && dx1 <= maxDx1) {
                if (gm.imagePts.size() == 0) {
                    ptImg = ptImg0;
                    ptObj.x = (int)(ptImg.x/imgSep.x + 0.5);
                    ptObj.y = (int)(ptImg.y/imgSep.y + 0.5);
                    gm.add(ptImg, ptObj);
                    ptObj += calcObjPointDiff(ptImg1, ptImg, imgSep);
                    ptImg = ptImg1;
                    gm.add(ptImg, ptObj);
                } else {
                    if (ptImg != ptImg0) {
                        ptObj += calcObjPointDiff(ptImg0, ptImg, imgSep);
                        ptImg = ptImg0;
                        gm.add(ptImg, ptObj);
                    }
                    ptObj += calcObjPointDiff(ptImg1, ptImg, imgSep);
                    ptImg = ptImg1;
                    gm.add(ptImg, ptObj);
                }
            }
            ptImg0 = ptImg1;
        }

        float maxDy1 = dmedian.y * (dmedian.y < 0 ? 1-tolerance : 1+tolerance);
        float minDy1 = dmedian.y * (dmedian.y < 0 ? 1+tolerance : 1-tolerance);
        vector<Point2f>::iterator itXY = pointsXY.begin();
        for (Point2f ptImg0=(*itXY); ++itXY!=pointsXY.end(); ) {
            const Point2f &ptImg1(*itXY);
            int dy1 = ptImg0.y - ptImg1.y;
            if (minDy1 <= dy1 && dy1 <= maxDy1) {
                if (gm.imagePts.size() == 0) {
                    ptImg = ptImg0;
                    ptObj.x = (int)(ptImg.x/imgSep.x + 0.5);
                    ptObj.y = (int)(ptImg.y/imgSep.y + 0.5);
                    gm.add(ptImg, ptObj);
                    ptObj += calcObjPointDiff(ptImg1, ptImg, imgSep);
                    ptImg = ptImg1;
                    gm.add(ptImg, ptObj);
                } else {
                    if (ptImg != ptImg0) {
                        ptObj += calcObjPointDiff(ptImg0, ptImg, imgSep);
                        ptImg = ptImg0;
                        gm.add(ptImg, ptObj);
                    }
                    ptObj += calcObjPointDiff(ptImg1, ptImg, imgSep);
                    ptImg = ptImg1;
                    gm.add(ptImg, ptObj);
                }
            }
            ptImg0 = ptImg1;
        }

        Point3f objCentroid = gm.getObjectCentroid();
        Point2f imgCentroid = gm.getImageCentroid();
        json_t *pRects = json_array();
        json_object_set(pStageModel, "rects", pRects);
        for (int i=0; i < gm.size(); i++) {
            json_t *pRect = json_object();
            json_object_set(pRect, "x", json_real(gm.imagePts[i].x));
            json_object_set(pRect, "y", json_real(gm.imagePts[i].y));
            json_object_set(pRect, "objX", json_real(objSep.x*(gm.objectPts[i].x-objCentroid.x)));
            json_object_set(pRect, "objY", json_real(objSep.y*(gm.objectPts[i].y-objCentroid.y)));
            json_array_append(pRects, pRect);
        }

        if (logLevel == FIRELOG_TRACE) {
            for (int i=0; i < gm.size(); i++) {
                LOGTRACE4("apply_matchGrid() objectPt[%d] (%g,%g,%g)",
                          i, gm.objectPts[i].x, gm.objectPts[i].y, gm.objectPts[i].z);
            }
        }

        errMsg = gm.calibrateImage(pStageModel, cameraMatrix, distCoeffs, model.image);
        json_t *pCalibrate  = json_object_get(pStageModel, "calibrate");
        json_object_set(pCalibrate, "op", json_string(opStr.c_str()));
    }

    return stageOK("apply_matchGrid(%s) %s", errMsg.c_str(), pStage, pStageModel);
}

bool Pipeline::apply_undistort(json_t *pStage, json_t *pStageModel, Model &model) {
    string errMsg;
    string modelName = jo_string(pStage, "model", "", model.argMap);
    vector<double> cm;
    vector<double> dc;
    vector<double> cmDefault;
    vector<double> dcDefault;
    Mat cameraMatrix;
    Mat distCoeffs;

    initCameraVectors(cmDefault, dcDefault, model.image);

    json_t *pCalibrate;
    json_t *pCalibrateModel = json_object_get(model.getJson(false), modelName.c_str());
    if (json_is_object(pCalibrateModel)) {
        pCalibrate = json_object_get(pCalibrateModel, "calibrate");
        if (!json_is_object(pCalibrate)) {
            errMsg = "Expected \"calibrate\" JSON object in stage \"";
            errMsg += modelName;
            errMsg += "\"";
        }
    } else {
        pCalibrate = pStage;
    }

    if (errMsg.empty()) {
        cm = jo_vectord(pCalibrate, "cameraMatrix", cmDefault, model.argMap);
        dc = jo_vectord(pCalibrate, "distCoeffs", dcDefault, model.argMap);

        if (cm.size() != 9) {
            errMsg = "expected cameraMatrix: [v11,v12,v13,v21,v22,v23,v31,v32,v33]";
        } else {
            cameraMatrix = Mat(3, 3, CV_64F);
			for (int i=0; i<cmDefault.size(); i++) {
				cameraMatrix.at<double>(i/3, i%3) = cmDefault[i];
			}
        }

        if (dc.size() != 5 && dc.size() != 4 && dc.size() != 8) {
            errMsg = "expected distCoeffs of 4, 5, or 8 elements";
        } else {
            distCoeffs = Mat(dc);
        }
    }

    if (errMsg.empty()) {
        InputArray newCameraMatrix=noArray();
        Mat dst;
        cout << "cameraMatrix:" << matInfo(cameraMatrix) << cameraMatrix << endl;
        cout << "distCoeffs:" << matInfo(distCoeffs) << distCoeffs << endl;
        undistort(model.image, dst, cameraMatrix, distCoeffs, newCameraMatrix);
        model.image = dst;
    }

    return stageOK("apply_undistort(%s) %s", errMsg.c_str(), pStage, pStageModel);
}
