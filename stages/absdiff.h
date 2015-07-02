#ifndef ABSDIFF_H
#define ABSDIFF_H
/*
 * @Author  : Simon Fojtu
 * @Date    : 26.06.2015
 */

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class AbsDiff : public Stage {
public:
    AbsDiff(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        img2_path = jo_string(pStage, "path", "", model.argMap);

        if (img2_path.empty()) {
        throw std::invalid_argument("Expected path to image for absdiff");
        }

        _params["path"] = new StringParameter(this, img2_path);
    }

private:
    virtual bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);

        const char *errMsg = NULL;
        Mat img2;

        if (model.image.channels() == 1) {
            img2 = imread(img2_path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        } else {
            img2 = imread(img2_path.c_str(), CV_LOAD_IMAGE_COLOR);
        }
        if (img2.data) {
            LOGTRACE2("apply_absdiff() path:%s %s", img2_path.c_str(), matInfo(img2).c_str());
        } else {
            errMsg = "Could not read image from given path";
        }

        if (!errMsg) {
            absdiff(model.image, img2, model.image);
        }

        return stageOK("apply_absdiff(%s) %s", errMsg, pStage, pStageModel);
    }

    string img2_path;

};

}

#endif // ABSDIFF_H
