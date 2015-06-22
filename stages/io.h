#ifndef IO_H
#define IO_H

/*
 * @Author  : Simon Fojtu
 * @Date    : 17.06.2015
 */

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class ImWrite : public Stage
{
public:
    ImWrite(json_t *pStage, Model &model) : Stage(pStage) {
        path = jo_string(pStage, "path");
    }

    string getName() const { return "ImWrite"; }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        Pipeline::validateImage(model.image);

        const char *errMsg = NULL;

        if (path.empty()) {
            errMsg = "Expected path for imwrite";
        } else {
            bool result = imwrite(path.c_str(), model.image);
            json_object_set(pStageModel, "result", json_boolean(result));
        }

        return stageOK("apply_imwrite(%s) %s", errMsg, pStage, pStageModel);
    }

protected:
    string path;
};

class ImRead : public Stage
{
public:
    ImRead(json_t *pStage, Model &model) : Stage(pStage) {
        path = jo_string(pStage, "path", "", model.argMap);
    }

    string getName() const { return "ImRead"; }

    bool apply_internal(json_t *pStageModel, Model &model) {
        const char *errMsg = NULL;

        if (path.empty()) {
            errMsg = "expected path for imread";
        } else {
            model.image = imread(path.c_str(), CV_LOAD_IMAGE_COLOR);
            if (model.image.data) {
                json_object_set(pStageModel, "rows", json_integer(model.image.rows));
                json_object_set(pStageModel, "cols", json_integer(model.image.cols));
            } else {
                LOGERROR1("imread(%s) failed", path.c_str());
                errMsg = "apply_imread() failed";
            }
        }

        return stageOK("apply_imread(%s) %s", errMsg, pStage, pStageModel);
    }

protected:
    string path;
};

}




#endif // IO_H
