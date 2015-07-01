#ifndef TRANSPARENT_H
#define TRANSPARENT_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class Transparent: public Stage {
public:
    Transparent(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        roi = jo_Rect(pStage, "roi", Rect(0, 0, model.image.cols, model.image.rows), model.argMap);
        // TODO parametrize roi
        alphafg = jo_float(pStage, "alphafg", 1, model.argMap);
        _params["alphafg"] = new FloatParameter(this, alphafg);
        alphabg = jo_float(pStage, "alphabg", 0, model.argMap);
        _params["alphabg"] = new FloatParameter(this, alphabg);
        bgcolor = jo_vectori(pStage, "bgcolor", vector<int>(), model.argMap);
        // TODO parametrize bgcolor
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);

        const char *errMsg = NULL;

        int fgIntensity = 255 * alphafg;
        if (fgIntensity < 0 || 255 < fgIntensity) {
            errMsg = "Expected 0 < alphafg < 1";
        }

        int bgIntensity = 255 * alphabg;
        if (bgIntensity < 0 || 255 < bgIntensity) {
            errMsg = "Expected 0 < alphabg < 1";
        }

        bool isBgColor = bgcolor.size() == 3;
        if (bgcolor.size() != 0 && !isBgColor) {
            errMsg = "Expected JSON [B,G,R] array for bgcolor";
        }

        int roiRowStart = max(0, roi.y);
        int roiColStart = max(0, roi.x);
        int roiRowEnd = min(model.image.rows, roi.y+roi.height);
        int roiColEnd = min(model.image.cols, roi.x+roi.width);

        if (roiRowEnd <= 0 || model.image.rows <= roiRowStart ||
                roiColEnd <= 0 || model.image.cols <= roiColStart) {
            errMsg = "Region of interest is not in image";
        }

        if (!errMsg) {
            Mat imageAlpha;
            cvtColor(model.image, imageAlpha, CV_BGR2BGRA, 0);
            LOGTRACE1("apply_alpha() imageAlpha %s", matInfo(imageAlpha).c_str());

            int rows = imageAlpha.rows;
            int cols = imageAlpha.cols;
            int bgBlue = isBgColor ? bgcolor[0] : 255;
            int bgGreen = isBgColor ? bgcolor[1] : 255;
            int bgRed = isBgColor ? bgcolor[2] : 255;
            for (int r=roiRowStart; r < roiRowEnd; r++) {
                for (int c=roiColStart; c < roiColEnd; c++) {
                    if (isBgColor) {
                        if ( bgBlue == imageAlpha.at<Vec4b>(r,c)[0] &&
                                bgGreen == imageAlpha.at<Vec4b>(r,c)[1] &&
                                bgRed == imageAlpha.at<Vec4b>(r,c)[2]) {
                            imageAlpha.at<Vec4b>(r,c)[3] = bgIntensity;
                        } else {
                            imageAlpha.at<Vec4b>(r,c)[3] = fgIntensity;
                        }
                    } else {
                        imageAlpha.at<Vec4b>(r,c)[3] = fgIntensity;
                    }
                }
            }
            model.image = imageAlpha;
        }

        return stageOK("apply_alpha(%s) %s", errMsg, pStage, pStageModel);
    }

    Rect roi;
    float alphafg;
    float alphabg;
    vector<int> bgcolor;

};

}

#endif // TRANSPARENT_H
