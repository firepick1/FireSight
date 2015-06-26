#ifndef DETECTOR_H
#define DETECTOR_H
/*
 * @Author  : Simon Fojtu
 * @Date    : 26.06.2015
 */

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class PartDetector : public Stage {
public:
    PartDetector(json_t *pStage, Model &model) : Stage(pStage) {
    }

    string getName() const { return "PartDetector"; }

private:
    vector<RotatedRect> detect(cv::Mat& image);

    bool apply_internal(json_t *pStageModel, Model &model) {
        const char *errMsg = NULL;

        if (model.image.channels() != 1) {
            errMsg = "PartDetector: single channel image required";
        }

        vector<RotatedRect> rects = PartDetector::detect(model.image);

        json_t *pRects = json_array();
        json_object_set(pStageModel, "rects", pRects);
        for (const auto& rect : rects) {
            json_t *pRect = json_object();
            json_object_set(pRect, "x", json_real(rect.center.x));
            json_object_set(pRect, "y", json_real(rect.center.y));
            json_object_set(pRect, "width", json_real(rect.size.width));
            json_object_set(pRect, "height", json_real(rect.size.height));
            json_object_set(pRect, "angle", json_real(rect.angle));
            json_array_append(pRects, pRect);
        }

        return stageOK("apply_detectParts(%s) %s", errMsg, pStage, pStageModel);
    }

};

}

#endif // DETECTOR_H
