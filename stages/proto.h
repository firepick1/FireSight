#ifndef PROTO_H
#define PROTO_H
#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class Proto: public Stage {
public:
    Proto(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        width = jo_int(pStage, "width", 14, model.argMap);
        _params["width"] = new IntParameter(this, width);
        height = jo_int(pStage, "height", 21, model.argMap);
        _params["height"] = new IntParameter(this, height);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model);

    int width;
    int height;
};

}

#endif // PROTO_H
