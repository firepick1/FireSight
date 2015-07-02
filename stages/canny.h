#ifndef CANNY_H
#define CANNY_H

/*
 * @Author  : Simon Fojtu
 * @Date    : 22.06.2015
 */

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

class Canny : public Stage
{
public:

    Canny(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        threshold1 = jo_float(pStage, "threshold1", 0, model.argMap);
        _params["threshold1"] = new FloatParameter(this, threshold1);

        threshold2 = jo_float(pStage, "threshold2", 50, model.argMap);
        _params["threshold2"] = new FloatParameter(this, threshold2);

        apertureSize = jo_int(pStage, "apertureSize", 3, model.argMap);
        _params["apertureSize"] = new IntParameter(this, apertureSize);

        L2gradient = jo_bool(pStage, "L2gradient", false);
        _params["L2gradient"] = new BoolParameter(this, L2gradient);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model)
    {
        const char *errMsg = NULL;
        validateImage(model.image);

        if (!errMsg) {
            cv::Canny(model.image, model.image, threshold1, threshold2, apertureSize, L2gradient);
        }

        return stageOK("apply_Canny(%s) %s", errMsg, pStage, pStageModel);
    }

protected:

    float threshold1;
    float threshold2;
    int apertureSize;
    bool L2gradient;
};

}

#endif // CANNY_H
