#ifndef MEANSTDDEV_H
#define MEANSTDDEV_H


#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class MeanStdDev: public Stage {
public:
    MeanStdDev(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) { }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        const char *errMsg = NULL;

        Scalar mean;
        Scalar stdDev;
        meanStdDev(model.image, mean, stdDev);

        json_t * jmean = json_array();
        json_object_set(pStageModel, "mean", jmean);
        json_t * jstddev = json_array();
        json_object_set(pStageModel, "stdDev", jstddev);
        for (int i = 0; i < 4; i++) {
            json_array_append(jmean, json_real(mean[i]));
            json_array_append(jstddev, json_real(stdDev[i]));
        }

        return stageOK("apply_meanStdDev(%s) %s", errMsg, pStage, pStageModel);
    }

};

}
#endif // MEANSTDDEV_H
