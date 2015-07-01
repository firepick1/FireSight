#ifndef MINAREARECT_H
#define MINAREARECT_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class MinAreaRect: public Stage {
public:
    MinAreaRect(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        channel = jo_int(pStage, "channel", 0, model.argMap);
        _params["chanel"] = new IntParameter(this, channel);
        minVal = jo_int(pStage, "min", 1, model.argMap);
        _params["min"] = new IntParameter(this, minVal);
        maxVal = jo_int(pStage, "max", 255, model.argMap);
        _params["max"] = new IntParameter(this, maxVal);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        const char *errMsg = NULL;
        vector<Point>  points;

        int rows = model.image.rows;
        int cols = model.image.cols;
        json_t *pRects = json_array();
        json_object_set(pStageModel, "rects", pRects);

        const int channels = model.image.channels();
        LOGTRACE1("apply_minAreaRect() channels:%d", channels);
        switch(channels) {
        case 1: {
            for (int iRow = 0; iRow < rows; iRow++) {
                for (int iCol = 0; iCol < cols; iCol++) {
                    uchar val = model.image.at<uchar>(iRow, iCol);
                    if (minVal <= val && val <= maxVal) {
                        points.push_back(Point(iCol, iRow));
                    }
                }
            }
            break;
        }
        case 3: {
            Mat_<Vec3b> image3b = model.image;
            for (int iRow = 0; iRow < rows; iRow++) {
                for (int iCol = 0; iCol < cols; iCol++) {
                    uchar val = image3b(iRow, iCol)[channel];
                    if (minVal <= val && val <= maxVal) {
                        points.push_back(Point(iCol, iRow));
                    }
                }
            }
            break;
        }
        }

        LOGTRACE1("apply_minAreaRect() points found: %d", (int) points.size());
        json_object_set(pStageModel, "points", json_integer(points.size()));
        if (points.size() > 0) {
            RotatedRect rect = minAreaRect(points);
            json_t *pRect = json_object();
            json_object_set(pRect, "x", json_real(rect.center.x));
            json_object_set(pRect, "y", json_real(rect.center.y));
            json_object_set(pRect, "width", json_real(rect.size.width));
            json_object_set(pRect, "height", json_real(rect.size.height));
            json_object_set(pRect, "angle", json_real(rect.angle));
            json_array_append(pRects, pRect);
        }

        return stageOK("apply_minAreaRect(%s) %s", errMsg, pStage, pStageModel);
    }

    int channel;
    int minVal;
    int maxVal;

};

}

#endif // MINAREARECT_H
