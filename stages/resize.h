#ifndef RESIZE_H
#define RESIZE_H


#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class Resize: public Stage {
public:
    Resize(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        fx = jo_float(pStage, "fx", 1, model.argMap);
        _params["fx"]  = new FloatParameter(this, fx);
        fy = jo_float(pStage, "fy", 1, model.argMap);
        _params["fy"]  = new FloatParameter(this, fy);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        const char *errMsg = NULL;

        if (fx <= 0 || fy <= 0) {
            errMsg = "Expected 0<fx and 0<fy";
        }
        if (!errMsg) {
            Mat result;
            resize(model.image, result, Size(), fx, fy, INTER_AREA);
            model.image = result;
        }

        return stageOK("apply_resize(%s) %s", errMsg, pStage, pStageModel);
    }

    float fx;
    float fy;
};

}


#endif // RESIZE_H
