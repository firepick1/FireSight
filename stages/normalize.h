#ifndef NORMALIZE_H
#define NORMALIZE_H
#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class Normalize: public Stage {
public:
    Normalize(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        domain = jo_vectorf(pStage, "domain", vector<float>(), model.argMap);
        range = jo_vectorf(pStage, "range", vector<float>(), model.argMap);
        normTypeStr = jo_string(pStage, "normType", "NORM_L2", model.argMap);
        _params["normType"] = new StringParameter(this, normTypeStr);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model);

    vector<float> domain;
    vector<float> range;
    string normTypeStr;

};

}

#endif // NORMALIZE_H
