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

	CAL_BEST,
	CAL_I,
	CAL_NONE,
	CAL_PERSPECTIVE,
	CAL_ELLIPSE,
	CAL_CORNERS,
	CAL_DIAMOND,
	CAL_QUAD0,
	CAL_QUAD1,
	CAL_QUAD2,
	CAL_QUAD3,
	CAL_TILE1,
	CAL_TILE2,
	CAL_TILE3,
	CAL_TILE4,
	CAL_TILE5,
	CAL_CELTIC_CROSS,
	CAL_XYAXES,
	CAL_XYORIGIN,
	CAL_CROSS
};

enum CompareOp {
    COMPARE_XY,
    COMPARE_YX
};

typedef class ComparePoint2f {
    private:
        CompareOp op;
		float tolerance;

    public:
        ComparePoint2f(CompareOp op=COMPARE_XY, float tolerance=10) {
            this->op = op;
			this->tolerance = tolerance;
        }
    public:
        bool operator()(const Point2f &lhs, const Point2f &rhs) const {
            assert(!isnan(lhs.x));
            assert(!isnan(rhs.x));
            int cmp;
            switch (op) {
            case COMPARE_XY:
                cmp = lhs.x - rhs.x;
                if (-tolerance <= cmp && cmp <= tolerance) {
                    cmp = lhs.y - rhs.y;
                }
                break;
            case COMPARE_YX:
                cmp = lhs.y - rhs.y;
                if (-tolerance <= cmp && cmp <= tolerance) {
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
	double focalLength = 1000; // 
    cmDefault.push_back(focalLength);
    cmDefault.push_back(0.0);
    cmDefault.push_back(image.cols/2);
    cmDefault.push_back(0.0);
    cmDefault.push_back(focalLength);
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
    vector<Point2f> imagePts;		// input image 
	vector<Point2f> calibrationPts;	// image calibration points (post-perspective)	
    vector<Point3f> objectPts;		// 
    Rect imgRect;
    const ComparePoint2f cmpYX;
    set<Point2f,ComparePoint2f> imgSet;
    set<Point2f,ComparePoint2f> subImgSet;
    vector<vector<Point2f> > vImagePts;
    vector<vector<Point3f> > vObjectPts;
	Point2f grid;
    Size imgSize;
    Mat perspective;
    Point2f imgSep;
    Point2f objSep;
    Mat gridIndexes;	// object grid matrix of imagePts/calibrationPts/objectPts vector indexes or -1
	double tolerance;
    GridMatcher(Size imgSize, Point2f objSep, double tolerance)
        : cmpYX(ComparePoint2f(COMPARE_YX)),
          imgSet(set<Point2f,ComparePoint2f>(cmpYX)),
          subImgSet(set<Point2f,ComparePoint2f>(cmpYX)),
		  grid(FLT_MAX,FLT_MAX),
		  objSep(objSep),
		  tolerance(tolerance)
    {
        this->imgSize = imgSize;
        this->imgRect = Rect(imgSize.width/2, imgSize.height/2, 0, 0);
	}

	vector<Point2f> create_gridImgPts(const vector<Point2f> &imgPts, string &errMsg) {
		vector<Point2f> gridImgPts(imgPts.size());

        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
		int r2 = rows/2;
		int c2 = cols/2;
		int index00 = gridIndexes.at<short>(r2,c2);
		Point2f center(imgPts[index00]);
		LOGTRACE2("create_gridImgPts() r2:%d c2:%d", r2, c2);
		LOGTRACE3("create_gridImgPts() index00:%d imgPt:(%g,%g)", 
			index00, imgPts[index00].x, imgPts[index00].y);

		float dxTot = 0;
		float dxCount = 0;
		for (int r=0; r < rows; r++) {
			int prevIndex = gridIndexes.at<short>(r,0);
			int n = 0;
			float yTot = 0;
			for (int c=1; c < cols; c++) {
				int index = gridIndexes.at<short>(r,c);
				if (index >= 0 && prevIndex >= 0) {
					dxTot += imgPts[index].x - imgPts[prevIndex].x;
					yTot += imgPts[index].y;
					n++;
				}
				prevIndex = index;
			}
			dxCount += n;
			if (r == r2) {
				center.y = yTot/n;
			}
		}
		float dyTot = 0;
		float dyCount = 0;
		for (int c=0; c < cols; c++) {
			int prevIndex = gridIndexes.at<short>(0,c);
			int n = 0;
			float xTot = 0;
			for (int r=1; r < rows; r++) {
				int index = gridIndexes.at<short>(r,c);
				if (index >= 0 && prevIndex >= 0) {
					dyTot += imgPts[index].y - imgPts[prevIndex].y;
					xTot += imgPts[index].x;
					n++;
				}
				prevIndex = index;
			}
			dyCount += n;
			if (c == c2) {
				center.x = xTot/n;
			}
		}
		Point2f gridSep(dxTot/dxCount, dyTot/dyCount);
		LOGTRACE2("create_gridImgPts() dxTot:%g dxCount:%g", dxTot, dxCount);
		LOGTRACE2("create_gridImgPts() dyTot:%g dyCount:%g", dyTot, dyCount);

		if (index00 < 0) {
			const char * msg = "GridMatcher::create_gridImgPts() could not calculate grid center. ";
			LOGWARN1("%s", msg);
			if (errMsg.find(msg) == string::npos) {
				errMsg += msg;
			}
		} else if (gridSep.x == 0) {
			const char * msg = "GridMatcher::create_gridImgPts() could not calculate grid separation. ";
			LOGWARN1("%s", msg);
			if (errMsg.find(msg) == string::npos) {
				errMsg += msg;
			}
		} else if (gridSep.y == 0) {
			const char * msg = "GridMatcher::create_gridImgPts() could not calculate grid.y separation. ";
			LOGWARN1("%s", msg);
			if (errMsg.find(msg) == string::npos) {
				errMsg += msg;
			}
		} else {  
			int n = 0;
			double totalSquaredError = 0;

			LOGTRACE4("create_gridImgPts() center:(%g,%g) gridSep:(%g,%g)", 
				center.x, center.y, gridSep.x, gridSep.y);
			if (gridSep.x > 0 && gridSep.y > 0) {
				for (int r=0; r < rows; r++) {
					for (int c=0; c < cols; c++) {
						int index = gridIndexes.at<short>(r,c);
						if (index >= 0) {
							Point2f gridPt = Point2f((c-c2)*gridSep.x, (r-r2)*gridSep.y) + center;
							gridImgPts[index] = gridPt;
						}
					}
				}
			}
		}
		return gridImgPts;
	}

	Point2f calcGridness(const vector<Point2f> &imgPts, string &errMsg) {
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
		int n = 0;
		Point2f totalSquaredError;
		vector<Point2f> gridImgPts = create_gridImgPts(imgPts, errMsg);

		if (gridImgPts.size() == 0) {
			return Point2f(nanf(""),nanf(""));
		}
		for (int r=0; r < rows; r++) {
			for (int c=0; c < cols; c++) {
				int index = gridIndexes.at<short>(r,c);
				if (index >= 0) {
					Point2f diff = gridImgPts[index] - imgPts[index];
					LOGTRACE4("calcGridness() r:%d c:%d diff:(%g,%g)", r, c, diff.x, diff.y);
					totalSquaredError.x += diff.x*(double)diff.x;
					totalSquaredError.y += diff.y*(double)diff.y;
					n++;
				}
			}
		}

		return Point2f(sqrt(totalSquaredError.x/n),sqrt(totalSquaredError.y/n));
	}

	string identifyRows(json_t *pStageModel, vector<Point2f> &pointsXY, float &dyMedian, Point2f &dyTot1,
                           Point2f &dyTot2, int &dyCount1, int &dyCount2)
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
				LOGDEBUG3("indentifyRows() pointsXY[%d] (%g,%g)", n, curPt.x, curPt.y);
				float dy1 = prevPt1.y - curPt.y;
				if (dy1 < minDy1) {
					LOGTRACE2("identifyRows() reject dy1:%g < minDy1:%g", dy1, minDy1);
				} else if (maxDy1 < dy1) {
					LOGTRACE2("identifyRows() reject maxDy1:%g < dy1:%g", maxDy1, dy1);
				} else {
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
			grid.y = normXY / objSep.y;
			json_object_set(pStageModel, "grid.y", json_real(grid.y));
			imgSep.y = grid.y*objSep.y;
		}

		return errMsg;
	} // identifyRows

	string identifyColumns(json_t *pStageModel, vector<Point2f> &pointsYX, float &dxMedian, Point2f &dxTot1,
								  Point2f &dxTot2, int &dxCount1, int &dxCount2)
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
				LOGDEBUG3("identify_Columns() pointsYX[%d] (%g,%g)", n, curPt.x, curPt.y);
				float dx1 = prevPt1.x - curPt.x;
				if (dx1 < minDx1) {
					LOGTRACE2("identifyColumns() reject dx1:%g < minDx1:%g", dx1, minDx1);
				} else if (maxDx1 < dx1) {
					LOGTRACE2("identifyColumns() reject maxDx1:%g < dx1:%g", maxDx1, dx1);
				} else {
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
			grid.x = normXY / objSep.x;
			json_object_set(pStageModel, "grid.x", json_real(grid.x));
			imgSep.x = grid.x*objSep.x;
		}

		return errMsg;
	} // identifyColumns

    bool add(Point2f &ptImg, Point3f &ptObj) {
        set<Point2f,ComparePoint2f>::iterator it = imgSet.find(ptImg);

        if (it != imgSet.end()) {
            return false;
        }
        objectPts.push_back(ptObj);
        imagePts.push_back(ptImg);
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
			if (logLevel == FIRELOG_TRACE) {
				char buf[255];
				snprintf(buf, sizeof(buf), "r:%d c:%d objectPts[%d]:(%g,%g) imagePts[%d]:(%g,%g)",
					r, c, i, objectPts[i].x, objectPts[i].y, i, imagePts[i].x, imagePts[i].y);
				LOGTRACE1("calcGridIndexes() %s", buf);
			}
        }
    }

    Point3f calcObjPointDiff(const Point2f &curPt, const Point2f &prevPt, const Point2f &imgSep) {
        float dObjX = (curPt.x-prevPt.x)/imgSep.x;
        float dObjY = (curPt.y-prevPt.y)/imgSep.y;
        dObjX += dObjX < 0 ? -0.5 : 0.5;
        dObjY += dObjY < 0 ? -0.5 : 0.5;
        return Point3f((int)(dObjX), (int) (dObjY), 0);
    }

	bool initPerspective(int r, int c, int dr, int dc, vector<Point2f> srcPts,
		Point2f &perspectiveDst, Point2f &perspectiveSrc, 
		set<Point2f,ComparePoint2f> *pSubImgSet) 
	{
		int found = 0;
		int index;
		Point2f pDst(0,0);
		Point2f pSrc(0,0);

		if (gridIndexes.at<short>(r,c) < 0) {
			return false;
		}

		int n = 1;	
		for (int i=0; i <= n+dr; i++) {
			for (int j=0; j <= n+dc; j++) {
				if (0 <= r+i && r+i < gridIndexes.rows && 0 <= c+j && c+j < gridIndexes.cols) {
					index = gridIndexes.at<short>(r+i,c+j);
					if (0 <= index) {
						if (pSubImgSet) { pSubImgSet->insert(imagePts[index]); }
						pDst += imagePts[index];
						pSrc += Point2f(srcPts[index].x, srcPts[index].y);
						if (logLevel == FIRELOG_TRACE) {
							char buf[255];
							snprintf(buf, sizeof(buf), "r:%d c:%d index:%d dst:(%g,%g) src:(%g,%g)", 
								r+i, c+j, index, pDst.x, pDst.y, pSrc.x, pSrc.y);
								LOGTRACE1("initPerspective() %s", buf);
						}
						found++;
					}
				}
			}
		}
		if (!found) {
			return false;
		}

		perspectiveDst = Point2f(pDst.x/found, pDst.y/found);
		perspectiveSrc = Point2f(pSrc.x/found, pSrc.y/found);
		LOGTRACE4("initPerspective() dst:(%g,%g) src:(%g,%g)", 
			perspectiveDst.x, perspectiveDst.y, perspectiveSrc.x, perspectiveSrc.y);
		return true;
	}

    Mat calcPerspective(vector<Point2f> &srcPts, set<Point2f,ComparePoint2f> *pSubImgSet=NULL) {
		Point2f perspectiveSrc[4];
		Point2f perspectiveDst[4];
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
        int r2 = rows/2;
        int c2 = cols/2;
        for (int r=0; r <= r2; r++ ) {
			if (initPerspective(r,(r*c2)/r2, 1, 1, srcPts, perspectiveDst[0], perspectiveSrc[0], pSubImgSet)) {
				break;
			}
        }
        for (int r=0; r <= r2; r++ ) {
			if (initPerspective(rows-r-1,(r*c2)/r2, -1, 1, srcPts, perspectiveDst[1], perspectiveSrc[1], pSubImgSet)) {
				break;
			}
        }
        for (int r=0; r <= r2; r++ ) {
			if (initPerspective(r,cols-1-(r*c2)/r2, 1, -1, srcPts, perspectiveDst[2], perspectiveSrc[2], pSubImgSet)) {
				break;
			}
        }
        for (int r=0; r <= r2; r++ ) {
			if (initPerspective(rows-r-1,cols-1-(r*c2)/r2, -1, -1, srcPts, perspectiveDst[3], perspectiveSrc[3], pSubImgSet)) {
                break;
            }
        }

        return getPerspectiveTransform(perspectiveDst, perspectiveSrc);
    }

    void matchPoints(Point2f dmedian, vector<Point2f> &pointsYX, vector<Point2f> &pointsXY) {
        Point3f ptObj;
        Point2f ptImg;

        float maxDx1 = dmedian.x * (dmedian.x < 0 ? 1-tolerance : 1+tolerance);
        float minDx1 = dmedian.x * (dmedian.x < 0 ? 1+tolerance : 1-tolerance);

        vector<Point2f>::iterator itYX = pointsYX.begin();
        for (Point2f ptImg0=(*itYX); ++itYX!=pointsYX.end(); ) {
            const Point2f &ptImg1(*itYX);
            float dx1 = ptImg0.x - ptImg1.x;
            if (dx1 < minDx1) {
                LOGTRACE2("matchPoints() reject dx1:%g < minDx1:%g", dx1, minDx1);
            } else if (maxDx1 < dx1) {
                LOGTRACE2("matchPoints() reject minDx1:%g < dx1:%g", maxDx1, dx1);
            } else {
                if (imagePts.size() == 0) {
                    ptImg = ptImg0;
                    ptObj.x = (int)(ptImg.x/imgSep.x + 0.5);
                    ptObj.y = (int)(ptImg.y/imgSep.y + 0.5);
                    add(ptImg, ptObj);
                    ptObj += calcObjPointDiff(ptImg1, ptImg, imgSep);
                    ptImg = ptImg1;
                    add(ptImg, ptObj);
                } else {
                    if (ptImg != ptImg0) {
                        ptObj += calcObjPointDiff(ptImg0, ptImg, imgSep);
                        ptImg = ptImg0;
                        add(ptImg, ptObj);
                    }
                    ptObj += calcObjPointDiff(ptImg1, ptImg, imgSep);
                    ptImg = ptImg1;
                    add(ptImg, ptObj);
                }
            }
            ptImg0 = ptImg1;
        }

        float maxDy1 = dmedian.y * (dmedian.y < 0 ? 1-tolerance : 1+tolerance);
        float minDy1 = dmedian.y * (dmedian.y < 0 ? 1+tolerance : 1-tolerance);
        vector<Point2f>::iterator itXY = pointsXY.begin();
        for (Point2f ptImg0=(*itXY); ++itXY!=pointsXY.end(); ) {
            const Point2f &ptImg1(*itXY);
            float dy1 = ptImg0.y - ptImg1.y;
            if (dy1 < minDy1) {
                LOGTRACE2("matchPoints() reject dy1:%g < minDy1:%g", dy1, minDy1);
            } else if (maxDy1 < dy1) {
                LOGTRACE2("matchPoints() reject minDy1:%g < dy1:%g", maxDy1, dy1);
            } else {
                if (imagePts.size() == 0) {
                    ptImg = ptImg0;
                    ptObj.x = (int)(ptImg.x/imgSep.x + 0.5);
                    ptObj.y = (int)(ptImg.y/imgSep.y + 0.5);
                    add(ptImg, ptObj);
                    ptObj += calcObjPointDiff(ptImg1, ptImg, imgSep);
                    ptImg = ptImg1;
                    add(ptImg, ptObj);
                } else {
                    if (ptImg != ptImg0) {
                        ptObj += calcObjPointDiff(ptImg0, ptImg, imgSep);
                        ptImg = ptImg0;
                        add(ptImg, ptObj);
                    }
                    ptObj += calcObjPointDiff(ptImg1, ptImg, imgSep);
                    ptImg = ptImg1;
                    add(ptImg, ptObj);
                }
            }
            ptImg0 = ptImg1;
        }

        if (logLevel == FIRELOG_TRACE) {
            for (int i=0; i < size(); i++) {
                LOGTRACE4("apply_matchGrid() objectPt[%d] (%g,%g,%g)",
                          i, objectPts[i].x, objectPts[i].y, objectPts[i].z);
            }
        }
    }

    bool addSubImagePoint(int r, int c, Point2f objCenter, vector<Point2f> &subImgPts, 
							vector<Point3f> &subObjPts, 
						  set<Point2f,ComparePoint2f> *pSubImgSet=NULL) {
        short index = gridIndexes.at<short>(r, c);
		bool added = false;
        if (0 <= index) {
            Point2f ptImg = calibrationPts[index];
            Point3f ptObj = objectPts[index];
            //Point3f subObjPt(objSep.x*(ptObj.x-objCenter.x), objSep.y*(ptObj.y-objCenter.y), 0);
            Point3f subObjPt(ptObj);
            added = true;
			LOGTRACE4("addSubImagePoint(r:%d,c:%d,objCenter.x:%g,objCenter.y:%g)", r, c, objCenter.x, objCenter.y);
			if (pSubImgSet) {
				bool added = pSubImgSet->insert(ptImg).second;
				(void) added;
			}
            subObjPts.push_back(subObjPt);
            subImgPts.push_back(ptImg);
        }

		return added;
    }

    bool addSubImage(int row, int col, int rows, int cols, int minPts, bool combine=false) {
        LOGDEBUG4("addSubImage(%d,%d,%d,%d)", row, col, rows, cols);
        assert(row >= 0);
        assert(col >= 0);
        assert(row+rows <= gridIndexes.rows);
        assert(col+cols <= gridIndexes.cols);
        vector<Point2f> subImgPts;
        vector<Point3f> subObjPts;
		Point2f objCenter((rows-1)/2.0, (cols-1)/2.0);
        float z = 0;
        for (int r=0; r < rows; r++) {
            for (int c=0; c < cols; c++) {
                addSubImagePoint(r+row, c+col, objCenter, subImgPts, subObjPts);
            }
        }
        if (!combine && subImgPts.size() < minPts) {
            if (logLevel >= FIRELOG_TRACE) {
                char buf[255];
                snprintf(buf, sizeof(buf), "addSubImage(%d,%d,%d,%d) REJECT:%ld",
                         row, col, rows, cols, (long) subObjPts.size());
                LOGTRACE1("%s", buf);
            }
            return false;
        }
		for (int i=0; i<subImgPts.size(); i++) {
			Point2f ptImg = subImgPts[i];
			if (subImgSet.insert(ptImg).second) {
				LOGTRACE2("addSubImage() point:[%d,%d]", (int)ptImg.x, (int)ptImg.y);
			}
		}
        if (logLevel >= FIRELOG_TRACE) {
            char buf[255];
            snprintf(buf, sizeof(buf), "addSubImage(%d,%d,%d,%d) ADD:%ld",
                     row, col, rows, cols, (long) subObjPts.size());
            LOGTRACE1("%s", buf);
        }
        if (subObjPts.size()) {
            if (!combine || vObjectPts.size()==0) {
                vObjectPts.push_back(subObjPts);
                vImagePts.push_back(subImgPts);
            } else {
                vector<Point3f> vObj0 = vObjectPts[0];
                vObj0.insert(vObj0.end(), subObjPts.begin(), subObjPts.end());
                vector<Point2f> vImg0 = vImagePts[0];
                vImg0.insert(vImg0.end(), subImgPts.begin(), subImgPts.end());
            }
        }
        return true;
    }

    int size() {
        return objectPts.size();
    }

    void subImageCornersFactory(Point2f scale) {
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
		int n2 = min(rows,cols)/2;

		Point2f oc((cols-1)/2.0, (rows-1)/2.0);
		int h = 2;
		int w = 2;
		int rlast = rows-h;
		int clast = cols-w;
		int minPts = 4;
		
		for (int i = 0; i < n2; i++) {
			if (addSubImage(i, i, h, w, minPts)) {
				break;
			}
		}
		for (int i = 0; i < n2; i++) {
			if (addSubImage(i, clast-i, h, w, minPts)) {
				break;
			}
		}
		for (int i = 0; i < n2; i++) {
			if (addSubImage(rlast-i, i, h, w, minPts)) {
				break;
			}
		}
		for (int i = 0; i < n2; i++) {
			if (addSubImage(rlast-i, clast-i, h, w, minPts)) {
				break;
			}
		}
    }

    void subImageIFactory(Point2f scale) {
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
		Point2f oc((cols-1)/2.0, (rows-1)/2.0);
		int xh = (int)(0.5+scale.y*rows/4.0);
		int yw = (int)(0.5+scale.x*cols/4.0);
		int c2 = (cols-yw)/2;
		int minPts = 4;
		
		addSubImage(0, 0, xh, cols, minPts);
		addSubImage(rows-xh, 0, xh, cols, minPts);
		addSubImage(0, c2, rows, yw, minPts);
    }

    bool subImageQuadrilateralFactory(Point2f scale, int extension, bool centerPoint=false) {
        vector<Point2f> subImgPts;
        vector<Point3f> subObjPts;
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
		int rStart = max(0,min(rows-1,(int)(0.5+rows*(1-scale.y))));
		int rLast = rows-rStart-1;
		int cStart = max(0,min(cols-1,(int)(0.5+cols*(1-scale.x))));
		int cLast = cols-cStart-1;
		Point2f oc((cols-1)/2.0, (rows-1)/2.0);

        LOGTRACE3("subImageQuadrilateralFactory(%d,%g,%g)", extension, scale.x, scale.y);

		if (centerPoint) {
			addSubImagePoint(oc.y, oc.x, oc, subImgPts, subObjPts, &subImgSet)
			|| addSubImagePoint(oc.y+1, oc.x, oc, subImgPts, subObjPts, &subImgSet)
			|| addSubImagePoint(oc.y, oc.x+1, oc, subImgPts, subObjPts, &subImgSet)
			|| addSubImagePoint(oc.y+1, oc.x+1, oc, subImgPts, subObjPts, &subImgSet);
		}

		for (int r=rStart; r <= oc.y; r++) {
			if (addSubImagePoint(r, r, oc, subImgPts, subObjPts, &subImgSet)) {
				for (int i=1; i <= extension; i++) {
					addSubImagePoint(r, r+i, oc, subImgPts, subObjPts, &subImgSet);
					addSubImagePoint(r+i, r, oc, subImgPts, subObjPts, &subImgSet);
				}
				break;
            }
        }
		for (int r=rStart; r <= oc.y; r++) {
			if (addSubImagePoint(r, cLast-(r-rStart), oc, subImgPts, subObjPts, &subImgSet)) {
				for (int i=1; i <= extension; i++) {
					addSubImagePoint(r, cLast-(r-rStart)-i, oc, subImgPts, subObjPts, &subImgSet);
					addSubImagePoint(r+i, cLast-(r-rStart), oc, subImgPts, subObjPts, &subImgSet);
				}
				break;
            }
        }
		for (int r=rLast; oc.y < r; r--) {
			if (addSubImagePoint(r, cStart+(rLast-r), oc, subImgPts, subObjPts, &subImgSet)) {
				for (int i=1; i <= extension; i++) {
					addSubImagePoint(r-i, cStart+(rLast-r), oc, subImgPts, subObjPts, &subImgSet);
					addSubImagePoint(r, cStart+(rLast-r)+i, oc, subImgPts, subObjPts, &subImgSet);
				}
				break;
            }
        }
		for (int r=rLast; oc.y < r; r--) {
			if (addSubImagePoint(r, cLast-(rLast-r), oc, subImgPts, subObjPts, &subImgSet)) {
				for (int i=1; i <= extension; i++) {
					addSubImagePoint(r-i, cLast-(rLast-r), oc, subImgPts, subObjPts, &subImgSet);
					addSubImagePoint(r, cLast-(rLast-r)-i, oc, subImgPts, subObjPts, &subImgSet);
				}
				break;
            }
        }
		if (subObjPts.size() < 4) {
			LOGERROR1("subImageQuadrilateralFactor() could not find 4 points to match (%d found)", (int)subObjPts.size());
			return false;
		}
		vObjectPts.push_back(subObjPts);
		vImagePts.push_back(subImgPts);
		return true;
    }

    bool subImageDiamondFactory(Point2f scale) {
        vector<Point2f> subImgPts;
        vector<Point3f> subObjPts;
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
		int rStart = max(0,min(rows-1,(int)(0.5+rows*(1-scale.y))));
		int rLast = rows-rStart-1;
		int cStart = max(0,min(cols-1,(int)(0.5+cols*(1-scale.x))));
		int cLast = cols-cStart-1;
		Point2f oc((cols-1)/2.0, (rows-1)/2.0);

        LOGTRACE2("subImageDiamondFactory(%g,%g)", scale.x, scale.y);
		for (int r=rStart; r <= oc.y; r++) {
			if (addSubImagePoint(r, oc.x, oc, subImgPts, subObjPts, &subImgSet)) {
				break;
            }
        }
		for (int r=rLast; oc.y < r; r--) {
			if (addSubImagePoint(r, oc.x, oc, subImgPts, subObjPts, &subImgSet)) {
				break;
            }
        }
		for (int c=cStart; c <= oc.x; c++) {
			if (addSubImagePoint(oc.y, c, oc, subImgPts, subObjPts, &subImgSet)) {
				break;
            }
        }
		for (int c=cLast; oc.x < c; c--) {
			if (addSubImagePoint(oc.y, c, oc, subImgPts, subObjPts, &subImgSet)) {
				break;
            }
        }
		if (subObjPts.size() < 4) {
			LOGERROR1("subImageDiamondFactory() could not find 4 points to match (%d found)", (int)subObjPts.size());
			return false;
		}
		vObjectPts.push_back(subObjPts);
		vImagePts.push_back(subImgPts);
		return true;
    }

    void subImageEllipseFactory(Point2f scale) {
        vector<Point2f> subImgPts;
        vector<Point3f> subObjPts;
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
		Point2f oc((cols-1)/2.0, (rows-1)/2.0);
		float ocy2 = oc.y * oc.y;
		float ocx2 = oc.x * oc.x;
        float nMax = ocy2 * ocx2 * scale.x * scale.x * scale.y * scale.y;

        LOGTRACE2("subImageEllipseFactory(%g,%g)", scale.x, scale.y);
        for (int r=0; r < rows; r++) {
            float dr = r-oc.y;
            for (int c=0; c <= cols; c++) {
                float dc = c-oc.x;
                float n = scale.x*scale.x*dr*dr*ocx2 + scale.y*scale.y*dc*dc*ocy2;
                if (n <= nMax) {
                    addSubImagePoint(r, c, oc, subImgPts, subObjPts, &subImgSet);
                }
            }
        }
        vObjectPts.push_back(subObjPts);
        vImagePts.push_back(subImgPts);
    }

    void subImageCelticCrossFactory() {
        int minPts = 4;
        int xh = max(3, gridIndexes.rows/4);
        int yw = max(3, gridIndexes.cols/4);
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
        int c2 = cols/2;
        int r2 = rows/2;

        LOGTRACE("subImageCelticCrossFactory()");
        // origin cross
        addSubImage(r2-xh/2, c2-c2/2, xh, c2, minPts);
        addSubImage(r2-r2/2, c2-yw/2, r2, yw, minPts);

        // xy axes
        addSubImage(r2-xh/2, 0, xh, c2, minPts);
        addSubImage(r2-xh/2, cols-c2, xh, c2, minPts);
        addSubImage(0, 		 c2-yw/2, r2, yw, minPts);
        addSubImage(rows-r2, c2-yw/2, r2, yw, minPts);

        // near quadrants
        addSubImage(r2-xh-1, c2-yw-1, xh, yw, minPts);
        addSubImage(r2+1,    c2-yw-1, xh, yw, minPts);
        addSubImage(r2-xh-1, c2+1, xh, yw, minPts);
        addSubImage(r2+1,    c2+1, xh, yw, minPts);
    }

    void subImageXYOriginFactory() {
        int minPts = 4;
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
        int xh = max(2, (rows)/5);
        int yw = max(2, (cols)/5);
        int c2 = cols/2;
        int r2 = rows/2;
        int cLast = cols - c2;
        int rLast = rows - r2;

        LOGTRACE("subImageXYOriginFactory()");

        // Cross
        addSubImage((rows-xh)/2, (cols-c2)/2, xh, c2, minPts);
        addSubImage((rows-r2)/2, (cols-yw)/2, r2, yw, minPts);

        // Axes
        addSubImage((rows-xh)/2, 0, xh, c2, minPts);
        addSubImage((rows-xh)/2, cLast, xh, c2, minPts);
        addSubImage(0, (cols-yw)/2, r2, yw, minPts);
        addSubImage(rLast, (cols-yw)/2, r2, yw, minPts);
    }

    void subImageXYAxesFactory() {
        int minPts = 4;
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
        int xh = max(2, (rows)/5);
        int yw = max(2, (cols)/5);
        int c2 = cols/2;
        int r2 = rows/2;
        int cLast = cols - c2;
        int rLast = rows - r2;

        LOGTRACE("subImageXYAxesFactory()");
        addSubImage((rows-xh)/2, 0, xh, c2, minPts);
        addSubImage((rows-xh)/2, cLast, xh, c2, minPts);
        addSubImage(0, (cols-yw)/2, r2, yw, minPts);
        addSubImage(rLast, (cols-yw)/2, r2, yw, minPts);
    }

    void subImageTileFactory(int n) {
        int minPts = 4;
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
        int w = (cols+n-1)/n;
        int h = (rows+n-1)/n;
        int n2 = n/2;

        LOGTRACE1("subImageTileFactory(%d)", n);
        if (n == 1) {
            addSubImage(0, 0, rows, cols, minPts);
        } else {
            if (n % 2) {
                int c2 = (cols-w)/2;
                int r2 = (rows-h)/2;
                addSubImage(r2, c2, h, w, minPts);	// center
                for (int i=0; i < n2; i++) {
                    addSubImage(r2, i*w, h, w, minPts);
                    addSubImage(r2, cols-(i+1)*w, h, w, minPts);
                    addSubImage(i*h, c2, h, w, minPts);
                    addSubImage(rows-(i+1)*h, c2, h, w, minPts);
                }
            }

            for (int i=0; i < n2; i++) {
                for (int j=0; j < n2; j++) {
                    addSubImage(i*h, j*w, h, w, minPts);
                    addSubImage(rows-i*h-h, j*w, h, w, minPts);
                    addSubImage(i*h, cols-j*w-w, h, w, minPts);
                    addSubImage(rows-i*h-h, cols-j*w-w, h, w, minPts);
                }
            }
        }
    }

    void subImageCrossFactory() {
        int minPts = 4;
        int rows = gridIndexes.rows;
        int cols = gridIndexes.cols;
        int rowMajor = rows/2;
        int rowMinor = rows % 2 ? 3 : 2;
        int colMajor = cols/2;
        int colMinor = cols % 2 ? 3 : 2;
        int rLastMa = rows-rowMajor;
        int rLastMi = rows-rowMinor;
        int cLastMa = cols-colMajor;
        int cLastMi = cols-colMinor;

        LOGTRACE("subImageCrossFactory()");
        addSubImage(rLastMi/2, cLastMa/2, rowMinor, colMajor, minPts);
        addSubImage(rLastMa/2, cLastMi/2, rowMajor, colMinor, minPts);

        addSubImage(rLastMi/2, max(0,cLastMa/2-1), rowMinor, colMajor, minPts);
        addSubImage(rLastMi/2, min(cLastMa,cLastMa/2+1), rowMinor, colMajor, minPts);
        addSubImage(max(0,rLastMa/2-1), cLastMi/2, rowMajor, colMinor, minPts);
        addSubImage(min(rLastMa,rLastMa/2+1), cLastMi/2, rowMajor, colMinor, minPts);

        addSubImage(rLastMi/2, max(0,cLastMa/2-2), rowMinor, colMajor, minPts);
        addSubImage(rLastMi/2, min(cLastMa,cLastMa/2+2), rowMinor, colMajor, minPts);
        addSubImage(max(0,rLastMa/2-2), cLastMi/2, rowMajor, colMinor, minPts);
        addSubImage(min(rLastMa,rLastMa/2+2), cLastMi/2, rowMajor, colMinor, minPts);

        addSubImage(rLastMi/2, max(0,cLastMa/2-3), rowMinor, colMajor, minPts);
        addSubImage(rLastMi/2, min(cLastMa,cLastMa/2+3), rowMinor, colMajor, minPts);
        addSubImage(max(0,rLastMa/2-3), cLastMi/2, rowMajor, colMinor, minPts);
        addSubImage(min(rLastMa,rLastMa/2+3), cLastMi/2, rowMajor, colMinor, minPts);
    }

    CalibrateOp parseCalibrateOp(string opStr, string &errMsg) {
        CalibrateOp op = CAL_NONE;

        if (opStr.compare("none") == 0) {
            op = CAL_NONE;
        } else if (opStr.compare("I") == 0) {
            op = CAL_I;
        } else if (opStr.compare("perspective") == 0) {
            op = CAL_PERSPECTIVE;
        } else if (opStr.compare("corners") == 0) {
            op = CAL_CORNERS;
        } else if (opStr.compare("diamond") == 0) {
            op = CAL_DIAMOND;
        } else if (opStr.compare("quad0") == 0) {
            op = CAL_QUAD0;
        } else if (opStr.compare("quad1") == 0) {
            op = CAL_QUAD1;
        } else if (opStr.compare("quad2") == 0) {
            op = CAL_QUAD2;
        } else if (opStr.compare("quad3") == 0) {
            op = CAL_QUAD3;
        } else if (opStr.compare("ellipse") == 0) {
            op = CAL_ELLIPSE;
        } else if (opStr.compare("tile1") == 0) {
            op = CAL_TILE1;
        } else if (opStr.compare("tile2") == 0) {
            op = CAL_TILE2;
        } else if (opStr.compare("tile3") == 0) {
            op = CAL_TILE3;
        } else if (opStr.compare("tile4") == 0) {
            op = CAL_TILE4;
        } else if (opStr.compare("tile5") == 0) {
            op = CAL_TILE5;
        } else if (opStr.compare("celtic") == 0) {
            op = CAL_CELTIC_CROSS;
        } else if (opStr.compare("cross") == 0) {
            op = CAL_CROSS;
        } else if (opStr.compare("xyorigin") == 0) {
            op = CAL_XYORIGIN;
        } else if (opStr.compare("xyaxes") == 0) {
            op = CAL_XYAXES;
        } else if (opStr.compare("best") == 0) {
            op = CAL_BEST;
        } else {
            errMsg = "Unknown calibrate option:";
            errMsg += opStr;
        }

        return op;
    }

	void initCameraMatrix(Mat &cameraMatrix, Mat &distCoeffs, 
		vector<Mat> &rvecs, vector<Mat> &tvecs, Mat image) 
	{
		vector<double> cmDefault;
		vector<double> dcDefault;
		initCameraVectors(cmDefault, dcDefault, image);
		cameraMatrix = Mat(3, 3, CV_64F);
		for (int i=0; i<cmDefault.size(); i++) {
			cameraMatrix.at<double>(i/3, i%3) = cmDefault[i];
		}
		distCoeffs = Mat(4, 1, CV_64F);
		for (int i=0; i<dcDefault.size(); i++) {
			distCoeffs.at<double>(i,0) = dcDefault[i];
		}
		rvecs.push_back(Mat::zeros(3, 1, CV_64F));
		tvecs.push_back(Mat::zeros(3, 1, CV_64F));
	}

	void createSubimages(CalibrateOp op, Point2f scale) {
        if (op == CAL_BEST) { 
            op = CAL_TILE3; // may change to best available
        }
        switch (op) {
        default:
        case CAL_NONE:
            break;
        case CAL_CELTIC_CROSS:
            subImageCelticCrossFactory();
            break;
        case CAL_XYORIGIN:
            subImageXYOriginFactory();
            break;
        case CAL_XYAXES:
            subImageXYAxesFactory();
            break;
        case CAL_CROSS:
            subImageCrossFactory();
            break;
        case CAL_I:
            subImageIFactory(scale);
            break;
        case CAL_PERSPECTIVE:
            break;
        case CAL_CORNERS:
            subImageCornersFactory(scale);
            break;
        case CAL_DIAMOND:
            subImageDiamondFactory(scale);
            break;
        case CAL_QUAD0:
            subImageQuadrilateralFactory(scale,0);
            break;
        case CAL_QUAD1:
            subImageQuadrilateralFactory(scale,1);
            break;
        case CAL_QUAD2:
            subImageQuadrilateralFactory(scale,2);
            break;
        case CAL_QUAD3:
            subImageQuadrilateralFactory(scale,3);
            break;
        case CAL_ELLIPSE:
            subImageEllipseFactory(scale);
            break;
        case CAL_TILE1:
            subImageTileFactory(1);
            break;
        case CAL_TILE2:
            subImageTileFactory(2);
            break;
        case CAL_TILE3:
            subImageTileFactory(3);
            break;
        case CAL_TILE4:
            subImageTileFactory(4);
            break;
        case CAL_TILE5:
            subImageTileFactory(5);
            break;
        }
	}

    string calibrateImage(json_t *pStageModel, Mat &cameraMatrix, Mat &distCoeffs,
                          Mat &image, string opStr, Scalar color, Point2f scale)
    {
        json_t *pRects = json_array();
        json_object_set(pStageModel, "rects", pRects);
        json_t *pCalibrate = json_object();
        json_object_set(pStageModel, "calibrate", pCalibrate);
        string errMsg;
        vObjectPts.clear();
        vImagePts.clear();

        calcGridIndexes();

        CalibrateOp op = parseCalibrateOp(opStr, errMsg);
        if (!errMsg.empty()) {
            return errMsg;
        }

        double rmserror = nan("");
        vector<Mat> rvecs;
        vector<Mat> tvecs;

		try {
			if (op == CAL_NONE) {
				initCameraMatrix(cameraMatrix, distCoeffs, rvecs, tvecs, image);
				subImgSet = imgSet;
			} else if (op == CAL_PERSPECTIVE) {
				initCameraMatrix(cameraMatrix, distCoeffs, rvecs, tvecs, image);
				vector<Point2f> gridImgPts = create_gridImgPts(imagePts, errMsg);
				perspective = calcPerspective(gridImgPts, &subImgSet);
				perspectiveTransform(imagePts, calibrationPts, perspective);
				subImageTileFactory(1);
				rmserror = calibrateCamera(vObjectPts, vImagePts, imgSize, 
					cameraMatrix, distCoeffs, rvecs, tvecs);
			} else {
				calibrationPts = imagePts;
				createSubimages(op, scale);
				rmserror = calibrateCamera(vObjectPts, vImagePts, imgSize,
										   cameraMatrix, distCoeffs, rvecs, tvecs);
					
			}
		} catch (cv::Exception ex) {
			errMsg = "calibrateImage(FAILED) ";
			errMsg += ex.msg;
		} catch (...) {
			errMsg = "calibrateImage(FAILED...)";
		}

		Point2f gridnessIn = Point2f(nanf(""),nanf(""));
		Point2f gridnessOut = Point2f(nanf(""),nanf(""));
        if (errMsg.empty()) {
			gridnessIn = calcGridness(imagePts, errMsg);
			if (rvecs.size() > 0) {
				gridnessOut = Point2f();
				for (int i=0; i < rvecs.size(); i++) {
					if (op == CAL_PERSPECTIVE) {
						vector<Point2f> perspectivePts;
						perspectiveTransform(imagePts, perspectivePts, perspective);
						gridnessOut += calcGridness(perspectivePts, errMsg);
					} else {
						vector<Point2f> projectedPts;
						projectPoints(objectPts, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPts);
						gridnessOut += calcGridness(projectedPts, errMsg);
					}
				}
				gridnessOut.x = gridnessOut.x/rvecs.size();
				gridnessOut.y = gridnessOut.y/rvecs.size();
			}

            for (int i=0; i < size(); i++) {
                json_t *pRect = json_object();
                json_object_set(pRect, "x", json_real(imagePts[i].x));
                json_object_set(pRect, "y", json_real(imagePts[i].y));
                json_object_set(pRect, "objX", json_real(objSep.x*(objectPts[i].x)));
                json_object_set(pRect, "objY", json_real(objSep.y*(objectPts[i].y)));
                json_array_append(pRects, pRect);
                set<Point2f,ComparePoint2f>::iterator it = subImgSet.find(calibrationPts[i]);
                if (it == subImgSet.end()) { // color marks points not used in calibration
                    json_t *pBGR = json_array();
                    json_object_set(pRect,"color", pBGR);
                    json_array_append(pBGR, json_integer(color[0]));
                    json_array_append(pBGR, json_integer(color[1]));
                    json_array_append(pBGR, json_integer(color[2]));
                }
            }
            for (set<Point2f,ComparePoint2f>::iterator it=subImgSet.begin();
                    it != subImgSet.end(); it++) {
                LOGTRACE2("subImgSet:[%g,%g]", it->x, it->y);
            }
        }

        json_object_set(pCalibrate, "perspective", json_matrix(perspective));
        json_object_set(pCalibrate, "op", json_string(opStr.c_str()));
        json_object_set(pCalibrate, "cameraMatrix", json_matrix(cameraMatrix));
        json_object_set(pCalibrate, "distCoeffs", json_matrix(distCoeffs));
        json_object_set(pCalibrate, "candidates", json_integer(size()));
        json_object_set(pCalibrate, "matched", json_integer(subImgSet.size()));
        json_object_set(pCalibrate, "images", json_real(vImagePts.size()));
        json_object_set(pCalibrate, "rmserror", 
			isnan(rmserror) ? json_string("NaN") : json_real(rmserror));
		json_t * jgridnessIn = json_array();
        json_object_set(pCalibrate, "gridnessIn", jgridnessIn);
		json_array_append(jgridnessIn, 
			isnan(gridnessIn.x) ? json_string("NaN") : json_real(gridnessIn.x));
		json_array_append(jgridnessIn, 
			isnan(gridnessIn.y) ? json_string("NaN") : json_real(gridnessIn.y));
		json_t * jgridnessOut = json_array();
        json_object_set(pCalibrate, "gridnessOut", jgridnessOut);
		json_array_append(jgridnessOut, 
			isnan(gridnessOut.x) ? json_string("NaN") : json_real(gridnessOut.x));
		json_array_append(jgridnessOut, 
			isnan(gridnessOut.y) ? json_string("NaN") : json_real(gridnessOut.y));
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

bool Pipeline::apply_matchGrid(json_t *pStage, json_t *pStageModel, Model &model) {
    string rectsModelName = jo_string(pStage, "model", "", model.argMap);
    string opStr = jo_string(pStage, "calibrate", "best", model.argMap);
    Scalar color = jo_Scalar(pStage, "color", Scalar(255,255,255), model.argMap);
	Point2f scale = jo_Point2f(pStage, "scale", Point2f(1,1), model.argMap);
    Point2f objSep = jo_Point2f(pStage, "sep", Point2f(5,5), model.argMap);
    double tolerance = jo_double(pStage, "tolerance", 0.35, model.argMap);
    Size imgSize(model.image.cols, model.image.rows);
    Point2f imgCenter(model.image.cols/2.0, model.image.rows/2.0);
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
    Point2f dmedian(FLT_MAX,FLT_MAX);
    const ComparePoint2f cmpYX(COMPARE_YX);
    vector<Point2f> pointsXY;
    vector<Point2f> pointsYX;
    Mat cameraMatrix;
    Mat distCoeffs;
	GridMatcher gm(imgSize, objSep, tolerance);

    if (errMsg.empty()) {
        initializePointMaps(pRects, pointsXY, pointsYX);
        errMsg = gm.identifyColumns(pStageModel, pointsYX, dmedian.x, dxTot1, dxTot2,
                                 dxCount1, dxCount2);
        string errMsg2 = gm.identifyRows(pStageModel, pointsXY, dmedian.y, dyTot1, dyTot2,
                                      dyCount1, dyCount2);

        if (errMsg.empty()) {
            errMsg = errMsg2;
        } else if (!errMsg2.empty()) {
            errMsg.append("; ");
            errMsg.append(errMsg2);
        }
    }

    if (errMsg.empty()) {
        gm.matchPoints(dmedian, pointsYX, pointsXY);
        errMsg = gm.calibrateImage(pStageModel, cameraMatrix, distCoeffs, model.image, opStr, color, scale);
    }

    return stageOK("apply_matchGrid(%s) %s", errMsg.c_str(), pStage, pStageModel);
}

bool Pipeline::apply_undistort(const char *pName, json_t *pStage, json_t *pStageModel, Model &model) {
    string errMsg;
    string modelName = jo_string(pStage, "model", pName, model.argMap);
    vector<double> cm;
    vector<double> pm;
    vector<double> dc;
    vector<double> cmDefault;
    vector<double> dcDefault;
    Mat cameraMatrix;
    Mat distCoeffs;

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

        if (cm.size() == 9) {
            cameraMatrix = Mat(3, 3, CV_64F);
            for (int i=0; i<cm.size(); i++) {
                cameraMatrix.at<double>(i/3, i%3) = cm[i];
            }
        } else if (cm.size() == 0) {
            LOGTRACE("apply_undistort() no cameraMatrix => no transformation");
        } else {
            errMsg = "expected cameraMatrix: [v11,v12,v13,v21,v22,v23,v31,v32,v33]";
        }

        switch (dc.size()) {
        case 0:
            LOGTRACE("apply_undistort() no distCoeffs => no transformation");
            break;
        case 4:
        case 5:
        case 8:
            distCoeffs = Mat(dc);
            break;
        default:
            errMsg = "expected distCoeffs of 4, 5, or 8 elements";
            break;
        }
    }

    if (errMsg.empty()) {
        if (distCoeffs.rows >= 4 && cameraMatrix.rows == 3) {
            json_object_set(pStageModel, "model", json_string(modelName.c_str()));
            Mat dst;
            InputArray newCameraMatrix=noArray();
            undistort(model.image, dst, cameraMatrix, distCoeffs, newCameraMatrix);
            model.image = dst;
        } else {
            json_object_set(pStageModel, "model", json_string(""));
        }
    }

    return stageOK("apply_undistort(%s) %s", errMsg.c_str(), pStage, pStageModel);
}
