#ifndef MODEL_H
#define MODEL_H
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

class ModelStage : public Stage {
public:
    ModelStage(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        json_t *pModel = json_object_get(pStage, "model");
        const char *errMsg = NULL;

        if (!errMsg) {
            if (!json_is_object(pModel)) {
                errMsg = "Expected JSON object for stage model";
            }
        }
        if (!errMsg && pModel) {
            const char * pKey;
            json_t *pValue;
            json_object_foreach(pModel, pKey, pValue) {
                json_object_set(pStageModel, pKey, pValue);
            }
        }

        return stageOK("apply_model(%s) %s", errMsg, pStage, pStageModel);
    }
};

}

#endif // MODEL_H
