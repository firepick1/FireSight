#ifndef CALIBRATE_H
#define CALIBRATE_H

#include "Pipeline.h"
#include "jo_util.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class MatchGrid: public Stage {
public:
    MatchGrid(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        rectsModelName = jo_string(pStage, "model", "", model.argMap);
        _params["model"] = new StringParameter(this, rectsModelName);
        opStr = jo_string(pStage, "calibrate", "best", model.argMap);
        _params["calibrate"] = new StringParameter(this, opStr);
        color = jo_Scalar(pStage, "color", Scalar(255,255,255), model.argMap);
        _params["color"] = new ScalarParameter(this, color);
        scale = jo_Point2f(pStage, "scale", Point2f(1,1), model.argMap);
        _params["scale"] = new Point2fParameter(this, scale);
        objSep = jo_Point2f(pStage, "sep", Point2f(5,5), model.argMap);
        tolerance = jo_double(pStage, "tolerance", 0.35, model.argMap);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model);

    string rectsModelName;
    string opStr;
    Scalar color;
    Point2f scale;
    Point2f objSep;
    double tolerance;
};

class Undistort: public Stage {
public:
    Undistort(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        modelName = jo_string(pStage, "model", pName.c_str(), model.argMap);
        _params["model"] = new StringParameter(this, modelName);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model);

    string modelName;
};

}

#endif // CALIBRATE_H
