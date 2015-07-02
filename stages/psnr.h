#ifndef PSNR_H
#define PSNR_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class PSNR: public Stage {
public:
    PSNR(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        path = jo_string(pStage, "path", "", model.argMap);
        _params["path"] = new StringParameter(this, path);
        psnrSame = jo_string(pStage, "psnrSame", "SAME", model.argMap);
        _params["psnrSame"] = new StringParameter(this, psnrSame);
        threshold = jo_double(pStage, "threshold", -1, model.argMap);
        _params["threshold"] = new DoubleParameter(this, threshold);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);

        const char *errMsg = NULL;
        Mat thatImage;

        if (path.empty()) {
            errMsg = "apply_PSNR() expected path for imread";
        } else {
            thatImage = imread(path.c_str(), CV_LOAD_IMAGE_COLOR);
            LOGTRACE2("apply_PSNR(%s) %s", path.c_str(), matInfo(thatImage).c_str());
            if (thatImage.data) {
                assert(model.image.cols == thatImage.cols);
                assert(model.image.rows == thatImage.rows);
                assert(model.image.channels() == thatImage.channels());
            } else {
                errMsg = "apply_PSNR() imread failed";
            }
        }

        if (!errMsg) {
            Mat s1;
            absdiff(model.image, thatImage, s1);  // |I1 - I2|
            s1.convertTo(s1, CV_32F);              // cannot make a square on 8 bits
            s1 = s1.mul(s1);                       // |I1 - I2|^2
            Scalar s = sum(s1);                   // sum elements per channel
            double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

        #define SSE_THRESHOLD 1e-10
            if( sse > 1e-10) {
                double  mse =sse /(double)(model.image.channels() * model.image.total());
                double psnr = 10.0*log10((255*255)/mse);
                json_object_set(pStageModel, "PSNR", json_real(psnr));
                if (threshold >= 0) {
                    if (psnr >= threshold) {
                        LOGTRACE2("apply_PSNR() threshold passed: %f >= %f", psnr, threshold);
                        json_object_set(pStageModel, "PSNR", json_string(psnrSame.c_str()));
                    } else {
                        LOGTRACE2("apply_PSNR() threshold failed: %f < %f", psnr, threshold);
                    }
                }
            } else if (sse == 0) {
                LOGTRACE("apply_PSNR() identical images: SSE == 0");
                json_object_set(pStageModel, "PSNR", json_string(psnrSame.c_str()));
            } else {
                LOGTRACE2("apply_PSNR() threshold passed: SSE %f < %f", sse, SSE_THRESHOLD);
                json_object_set(pStageModel, "PSNR", json_string(psnrSame.c_str()));
            }
        }

        return stageOK("apply_PSNR(%s) %s", errMsg, pStage, pStageModel);

    }

    string path;
    string psnrSame;
    double threshold;

};

}


#endif // PSNR_H
