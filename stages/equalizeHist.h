#ifndef EQUALIZEHIST_H
#define EQUALIZEHIST_H


#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"

namespace firesight {

using namespace cv;

class EqualizeHist: public Stage {
public:
    EqualizeHist(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) { }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        const char *errMsg = NULL;

        if (!errMsg) {
            equalizeHist(model.image, model.image);
        }

        return stageOK("apply_equalizeHist(%s) %s", errMsg, pStage, pStageModel);
    }
};

}

#endif // EQUALIZEHIST_H
