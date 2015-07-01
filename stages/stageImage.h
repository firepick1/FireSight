#ifndef STAGEIMAGE_H
#define STAGEIMAGE_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class StageImage: public Stage {
public:
    StageImage(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        stageStr = jo_string(pStage, "stage", "input", model.argMap);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        const char *errMsg = NULL;

        if (stageStr.empty()) {
            errMsg = "Expected name of stage for image";
        } else {
            model.image = model.imageMap[stageStr.c_str()];
            if (!model.image.rows || !model.image.cols) {
                model.image = model.imageMap["input"].clone();
                LOGTRACE1("Could not locate stage image '%s', using input image", stageStr.c_str());
            }
        }

        return stageOK("apply_stageImage(%s) %s", errMsg, pStage, pStageModel);
    }

    string stageStr;
};

}

#endif // STAGEIMAGE_H
