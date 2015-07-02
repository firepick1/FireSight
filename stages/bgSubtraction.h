#ifndef BGSUBTRACTION_H
#define BGSUBTRACTION_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class BackgroundSubtraction: public Stage {

    class SubtractorStageData : public StageData {
    public:
        BackgroundSubtractor *pSubtractor;

        SubtractorStageData(string stageName, BackgroundSubtractor *pSubtractor) : StageData(stageName) {
            assert(pSubtractor);
            this->pSubtractor = pSubtractor;
        }

        ~SubtractorStageData() {
            LOGTRACE("Freeing BackgroundSubtractor");
            delete pSubtractor;
        }
    };

public:

    BackgroundSubtraction(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        history = jo_int(pStage, "history", 0, model.argMap);
        _params["history"] = new IntParameter(this, history);
        varThreshold = jo_float(pStage, "varThreshold", 16, model.argMap);
        _params["varThreshold"] = new FloatParameter(this, varThreshold);
        bShadowDetection = jo_bool(pStage, "bShadowDetection", true, model.argMap);
        _params["bShadowDetection"] = new BoolParameter(this, bShadowDetection);
        background = jo_string(pStage, "background", "", model.argMap);
        _params["background"] = new StringParameter(this, background);

        // method
        method = MOG; //!< default value
        string smethod = jo_string(pStage, "method", BGSubTypeParser::get(method).c_str(), model.argMap);
        method = BGSubTypeParser::get(smethod);
        mapMethod = BGSubTypeParser::get();
        _params["method"] = new EnumParameter(this, method, mapMethod);

        //        stageName = jo_string(pStage, "name", method.c_str(), model.argMap);
        learningRate = jo_double(pStage, "learningRate", -1, model.argMap);
        _params["learningRate"] = new DoubleParameter(this, learningRate);
    }
protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);

        const char *errMsg = NULL;
        StageDataPtr pStageData = model.stageDataMap[this->getName()];
        int maxval = 255;

        BackgroundSubtractor *pSubtractor;
        bool is_absdiff = false;
        if (!errMsg) {
            if (method == MOG) {
                if (pStageData) {
                    pSubtractor = ((SubtractorStageData *) pStageData)->pSubtractor;
                } else {
                    pSubtractor = new BackgroundSubtractorMOG(history, varThreshold, bShadowDetection);
                    model.stageDataMap[this->getName()] = new SubtractorStageData(this->getName(), pSubtractor);
                }
            } else if (method == MOG2) {
                if (pStageData) {
                    pSubtractor = ((SubtractorStageData *) pStageData)->pSubtractor;
                } else {
                    pSubtractor = new BackgroundSubtractorMOG2(history, varThreshold, bShadowDetection);
                    model.stageDataMap[this->getName()] = new SubtractorStageData(this->getName(), pSubtractor);
                }
            } else if (method == ABSDIFF) {
                is_absdiff = true;
            } else {
                errMsg = "Expected method: MOG2 or MOG or ABSDIFF";
            }
        }

        Mat bgImage;
        if (!background.empty()) {
            if (history != 0) {
                errMsg = "Expected history=0 if background image is specified";
            } else {
                if (model.image.channels() == 1) {
                    bgImage = imread(background.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
                } else {
                    bgImage = imread(background.c_str(), CV_LOAD_IMAGE_COLOR);
                }
                if (bgImage.data) {
                    LOGTRACE2("apply_backgroundSubtractor(%s) %s", background.c_str(), matInfo(bgImage).c_str());
                    if (model.image.rows!=bgImage.rows || model.image.cols!=bgImage.cols) {
                        errMsg = "Expected background image of same size as pipeline image";
                    }
                } else {
                    errMsg = "Could not load background image";
                }
            }
        }
        if (!errMsg) {
            if (history < 0) {
                errMsg = "Expected history >= 0";
            }
        }

        if (!errMsg) {
            Mat fgMask;
            if (is_absdiff) {
                absdiff(model.image, bgImage, fgMask);
                if (fgMask.channels() > 1) {
                    cvtColor(fgMask, fgMask, CV_BGR2GRAY);
                }
                threshold(fgMask, model.image, varThreshold, maxval, THRESH_BINARY);
            } else {
                if (bgImage.data) {
                    pSubtractor->operator()(bgImage, fgMask, learningRate);
                }
                pSubtractor->operator()(model.image, fgMask, learningRate);
                model.image = fgMask;
            }
        }

        return stageOK("apply_backgroundSubtractor(%s) %s", errMsg, pStage, pStageModel);
    }

    int history;
    float varThreshold;
    bool bShadowDetection;
    string background;
    int method;
    map<int, string> mapMethod;
    double learningRate;
};

}
#endif // BGSUBTRACTION_H
