#ifndef THRESHOLD_H
#define THRESHOLD_H

/*
 * @Author  : Simon Fojtu
 * @Date    : 26.06.2015
 */

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class Threshold : public Stage {
public:
    Threshold(json_t *pStage, Model &model) : Stage(pStage) {
        mapType[THRESH_BINARY]		= "THRESH_BINARY";
        mapType[THRESH_BINARY_INV]  = "THRESH_BINARY_INV";
        mapType[THRESH_TRUNC]		= "THRESH_TRUNC";
        mapType[THRESH_TOZERO]		= "THRESH_TOZERO";
        mapType[THRESH_TOZERO_INV]	= "THRESH_TOZERO_INV";
        string stype = jo_string(pStage, "type", "THRESH_BINARY", model.argMap);
        auto findType = std::find_if(std::begin(mapType), std::end(mapType), [&](const std::pair<int, string> &pair)
        {
            return stype.compare(pair.second) == 0;
        });
        if (findType != std::end(mapType)) {
            type = findType->first;
            _params["type"] = new EnumParameter(this, type, mapType);
        } else
            throw std::invalid_argument("unknown 'type'");

        maxval = jo_float(pStage, "maxval", 255, model.argMap);
        _params["maxval"] = new FloatParameter(this, maxval);

        //otsu = jo_string(pStage, "thresh", "OTSU", model.argMap);
        isOtsu = jo_bool(pStage, "otsu", false, model.argMap);
        _params["otsu"] = new BoolParameter(this, isOtsu);

        thresh = jo_float(pStage, "thresh", 128, model.argMap);
        _params["thresh"] = new FloatParameter(this, thresh);

        gray = jo_bool(pStage, "gray", true, model.argMap);
        _params["gray"] = new BoolParameter(this, gray);
    }

    virtual string getName() const { return "Threshold"; }

private:
    virtual bool apply_internal(json_t *pStageModel, Model &model) {
        Pipeline::validateImage(model.image);
        const char *errMsg = NULL;

        if (!gray && isOtsu) {
            errMsg = "Otsu's method cannot be used with color images. Specify a thresh value for color images.";
        }

        if (!errMsg) {
            if (isOtsu) {
                type |= THRESH_OTSU;
            }
            if ((isOtsu || gray) && model.image.channels() > 1) {
                LOGTRACE("apply_threshold() converting image to grayscale");
                cvtColor(model.image, model.image, CV_BGR2GRAY, 0);
            }
            threshold(model.image, model.image, thresh, maxval, type);
        }

        return stageOK("apply_threshold(%s) %s", errMsg, pStage, pStageModel);
    }

    int type;
    map<int, string> mapType;

    float maxval;
    bool isOtsu;
    float thresh;
    bool gray;
};

}


#endif // THRESHOLD_H
