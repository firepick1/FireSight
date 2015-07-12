#ifndef NOP_H
#define NOP_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class NOPStage : public Stage {
public:
    NOPStage(json_t *pStage, Model &model, string pName)
        : Stage(pStage, pName)
    { }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        LOGDEBUG("Skipping nop...");
        return true;
    }
};

}

#endif // NOP_H
