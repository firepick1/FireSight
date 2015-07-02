#ifndef CONVERT_H
#define CONVERT_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class CvtColor : public Stage {
public:
    CvtColor(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        code = CV_BGR2GRAY;
        cvtCode[CV_BGR2GRAY]	= "CV_BGR2GRAY";
        codeStr = jo_string(pStage, "code", "CV_BGR2GRAY", model.argMap);
        auto findCode = std::find_if(std::begin(cvtCode), std::end(cvtCode), [&](const std::pair<int, string> &pair)
        {
            return codeStr.compare(pair.second) == 0;
        });
        if (findCode != std::end(cvtCode))
            code = findCode->first;

        _params["code"] = new EnumParameter(this, code, cvtCode);

        dstCn = jo_int(pStage, "dstCn", 0, model.argMap);
        if (dstCn < 0) {
            throw invalid_argument("expected 0<dstCn");
        }
        _params["dstCn"] = new IntParameter(this, dstCn);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);

        const char *errMsg = NULL;
        int code = CV_BGR2GRAY;

        if (codeStr.compare("CV_BGR2BGRA")==0) {
            code = CV_BGR2BGRA;
        } else if (codeStr.compare("CV_RGB2RGBA")==0) {
            code = CV_RGB2RGBA;
        } else if (codeStr.compare("CV_BGRA2BGR")==0) {
            code = CV_BGRA2BGR;
        } else if (codeStr.compare("CV_RGBA2RGB")==0) {
            code = CV_RGBA2RGB;
        } else if (codeStr.compare("CV_BGR2RGBA")==0) {
            code = CV_BGR2RGBA;
        } else if (codeStr.compare("CV_RGB2BGRA")==0) {
            code = CV_RGB2BGRA;
        } else if (codeStr.compare("CV_RGBA2BGR")==0) {
            code = CV_RGBA2BGR;
        } else if (codeStr.compare("CV_BGRA2RGB")==0) {
            code = CV_BGRA2RGB;
        } else if (codeStr.compare("CV_BGR2RGB")==0) {
            code = CV_BGR2RGB;
        } else if (codeStr.compare("CV_RGB2BGR")==0) {
            code = CV_RGB2BGR;
        } else if (codeStr.compare("CV_BGRA2RGBA")==0) {
            code = CV_BGRA2RGBA;
        } else if (codeStr.compare("CV_RGBA2BGRA")==0) {
            code = CV_RGBA2BGRA;
        } else if (codeStr.compare("CV_BGR2GRAY")==0) {
            code = CV_BGR2GRAY;
        } else if (codeStr.compare("CV_RGB2GRAY")==0) {
            code = CV_RGB2GRAY;
        } else if (codeStr.compare("CV_GRAY2BGR")==0) {
            code = CV_GRAY2BGR;
        } else if (codeStr.compare("CV_GRAY2RGB")==0) {
            code = CV_GRAY2RGB;
        } else if (codeStr.compare("CV_GRAY2BGRA")==0) {
            code = CV_GRAY2BGRA;
        } else if (codeStr.compare("CV_GRAY2RGBA")==0) {
            code = CV_GRAY2RGBA;
        } else if (codeStr.compare("CV_BGRA2GRAY")==0) {
            code = CV_BGRA2GRAY;
        } else if (codeStr.compare("CV_RGBA2GRAY")==0) {
            code = CV_RGBA2GRAY;
        } else if (codeStr.compare("CV_BGR2BGR565")==0) {
            code = CV_BGR2BGR565;
        } else if (codeStr.compare("CV_RGB2BGR565")==0) {
            code = CV_RGB2BGR565;
        } else if (codeStr.compare("CV_BGR5652BGR")==0) {
            code = CV_BGR5652BGR;
        } else if (codeStr.compare("CV_BGR5652RGB")==0) {
            code = CV_BGR5652RGB;
        } else if (codeStr.compare("CV_BGRA2BGR565")==0) {
            code = CV_BGRA2BGR565;
        } else if (codeStr.compare("CV_RGBA2BGR565")==0) {
            code = CV_RGBA2BGR565;
        } else if (codeStr.compare("CV_BGR5652BGRA")==0) {
            code = CV_BGR5652BGRA;
        } else if (codeStr.compare("CV_BGR5652RGBA")==0) {
            code = CV_BGR5652RGBA;
        } else if (codeStr.compare("CV_GRAY2BGR565")==0) {
            code = CV_GRAY2BGR565;
        } else if (codeStr.compare("CV_BGR5652GRAY")==0) {
            code = CV_BGR5652GRAY;
        } else if (codeStr.compare("CV_BGR2BGR555")==0) {
            code = CV_BGR2BGR555;
        } else if (codeStr.compare("CV_RGB2BGR555")==0) {
            code = CV_RGB2BGR555;
        } else if (codeStr.compare("CV_BGR5552BGR")==0) {
            code = CV_BGR5552BGR;
        } else if (codeStr.compare("CV_BGR5552RGB")==0) {
            code = CV_BGR5552RGB;
        } else if (codeStr.compare("CV_BGRA2BGR555")==0) {
            code = CV_BGRA2BGR555;
        } else if (codeStr.compare("CV_RGBA2BGR555")==0) {
            code = CV_RGBA2BGR555;
        } else if (codeStr.compare("CV_BGR5552BGRA")==0) {
            code = CV_BGR5552BGRA;
        } else if (codeStr.compare("CV_BGR5552RGBA")==0) {
            code = CV_BGR5552RGBA;
        } else if (codeStr.compare("CV_GRAY2BGR555")==0) {
            code = CV_GRAY2BGR555;
        } else if (codeStr.compare("CV_BGR5552GRAY")==0) {
            code = CV_BGR5552GRAY;
        } else if (codeStr.compare("CV_BGR2XYZ")==0) {
            code = CV_BGR2XYZ;
        } else if (codeStr.compare("CV_RGB2XYZ")==0) {
            code = CV_RGB2XYZ;
        } else if (codeStr.compare("CV_XYZ2BGR")==0) {
            code = CV_XYZ2BGR;
        } else if (codeStr.compare("CV_XYZ2RGB")==0) {
            code = CV_XYZ2RGB;
        } else if (codeStr.compare("CV_BGR2YCrCb")==0) {
            code = CV_BGR2YCrCb;
        } else if (codeStr.compare("CV_RGB2YCrCb")==0) {
            code = CV_RGB2YCrCb;
        } else if (codeStr.compare("CV_YCrCb2BGR")==0) {
            code = CV_YCrCb2BGR;
        } else if (codeStr.compare("CV_YCrCb2RGB")==0) {
            code = CV_YCrCb2RGB;
        } else if (codeStr.compare("CV_BGR2HSV")==0) {
            code = CV_BGR2HSV;
        } else if (codeStr.compare("CV_RGB2HSV")==0) {
            code = CV_RGB2HSV;
        } else if (codeStr.compare("CV_BGR2Lab")==0) {
            code = CV_BGR2Lab;
        } else if (codeStr.compare("CV_RGB2Lab")==0) {
            code = CV_RGB2Lab;
        } else if (codeStr.compare("CV_BayerBG2BGR")==0) {
            code = CV_BayerBG2BGR;
        } else if (codeStr.compare("CV_BayerGB2BGR")==0) {
            code = CV_BayerGB2BGR;
        } else if (codeStr.compare("CV_BayerRG2BGR")==0) {
            code = CV_BayerRG2BGR;
        } else if (codeStr.compare("CV_BayerGR2BGR")==0) {
            code = CV_BayerGR2BGR;
        } else if (codeStr.compare("CV_BayerBG2RGB")==0) {
            code = CV_BayerBG2RGB;
        } else if (codeStr.compare("CV_BayerGB2RGB")==0) {
            code = CV_BayerGB2RGB;
        } else if (codeStr.compare("CV_BayerRG2RGB")==0) {
            code = CV_BayerRG2RGB;
        } else if (codeStr.compare("CV_BayerGR2RGB")==0) {
            code = CV_BayerGR2RGB;
        } else if (codeStr.compare("CV_BGR2Luv")==0) {
            code = CV_BGR2Luv;
        } else if (codeStr.compare("CV_RGB2Luv")==0) {
            code = CV_RGB2Luv;
        } else if (codeStr.compare("CV_BGR2HLS")==0) {
            code = CV_BGR2HLS;
        } else if (codeStr.compare("CV_RGB2HLS")==0) {
            code = CV_RGB2HLS;
        } else if (codeStr.compare("CV_HSV2BGR")==0) {
            code = CV_HSV2BGR;
        } else if (codeStr.compare("CV_HSV2RGB")==0) {
            code = CV_HSV2RGB;
        } else if (codeStr.compare("CV_Lab2BGR")==0) {
            code = CV_Lab2BGR;
        } else if (codeStr.compare("CV_Lab2RGB")==0) {
            code = CV_Lab2RGB;
        } else if (codeStr.compare("CV_Luv2BGR")==0) {
            code = CV_Luv2BGR;
        } else if (codeStr.compare("CV_Luv2RGB")==0) {
            code = CV_Luv2RGB;
        } else if (codeStr.compare("CV_HLS2BGR")==0) {
            code = CV_HLS2BGR;
        } else if (codeStr.compare("CV_HLS2RGB")==0) {
            code = CV_HLS2RGB;
        } else if (codeStr.compare("CV_BayerBG2BGR_VNG")==0) {
            code = CV_BayerBG2BGR_VNG;
        } else if (codeStr.compare("CV_BayerGB2BGR_VNG")==0) {
            code = CV_BayerGB2BGR_VNG;
        } else if (codeStr.compare("CV_BayerRG2BGR_VNG")==0) {
            code = CV_BayerRG2BGR_VNG;
        } else if (codeStr.compare("CV_BayerGR2BGR_VNG")==0) {
            code = CV_BayerGR2BGR_VNG;
        } else if (codeStr.compare("CV_BayerBG2RGB_VNG")==0) {
            code = CV_BayerBG2RGB_VNG;
        } else if (codeStr.compare("CV_BayerGB2RGB_VNG")==0) {
            code = CV_BayerGB2RGB_VNG;
        } else if (codeStr.compare("CV_BayerRG2RGB_VNG")==0) {
            code = CV_BayerRG2RGB_VNG;
        } else if (codeStr.compare("CV_BayerGR2RGB_VNG")==0) {
            code = CV_BayerGR2RGB_VNG;
        } else if (codeStr.compare("CV_BGR2HSV_FULL")==0) {
            code = CV_BGR2HSV_FULL;
        } else if (codeStr.compare("CV_RGB2HSV_FULL")==0) {
            code = CV_RGB2HSV_FULL;
        } else if (codeStr.compare("CV_BGR2HLS_FULL")==0) {
            code = CV_BGR2HLS_FULL;
        } else if (codeStr.compare("CV_RGB2HLS_FULL")==0) {
            code = CV_RGB2HLS_FULL;
        } else if (codeStr.compare("CV_HSV2BGR_FULL")==0) {
            code = CV_HSV2BGR_FULL;
        } else if (codeStr.compare("CV_HSV2RGB_FULL")==0) {
            code = CV_HSV2RGB_FULL;
        } else if (codeStr.compare("CV_HLS2BGR_FULL")==0) {
            code = CV_HLS2BGR_FULL;
        } else if (codeStr.compare("CV_HLS2RGB_FULL")==0) {
            code = CV_HLS2RGB_FULL;
        } else if (codeStr.compare("CV_LBGR2Lab")==0) {
            code = CV_LBGR2Lab;
        } else if (codeStr.compare("CV_LRGB2Lab")==0) {
            code = CV_LRGB2Lab;
        } else if (codeStr.compare("CV_LBGR2Luv")==0) {
            code = CV_LBGR2Luv;
        } else if (codeStr.compare("CV_LRGB2Luv")==0) {
            code = CV_LRGB2Luv;
        } else if (codeStr.compare("CV_Lab2LBGR")==0) {
            code = CV_Lab2LBGR;
        } else if (codeStr.compare("CV_Lab2LRGB")==0) {
            code = CV_Lab2LRGB;
        } else if (codeStr.compare("CV_Luv2LBGR")==0) {
            code = CV_Luv2LBGR;
        } else if (codeStr.compare("CV_Luv2LRGB")==0) {
            code = CV_Luv2LRGB;
        } else if (codeStr.compare("CV_BGR2YUV")==0) {
            code = CV_BGR2YUV;
        } else if (codeStr.compare("CV_RGB2YUV")==0) {
            code = CV_RGB2YUV;
        } else if (codeStr.compare("CV_YUV2BGR")==0) {
            code = CV_YUV2BGR;
        } else if (codeStr.compare("CV_YUV2RGB")==0) {
            code = CV_YUV2RGB;
        } else if (codeStr.compare("CV_BayerBG2GRAY")==0) {
            code = CV_BayerBG2GRAY;
        } else if (codeStr.compare("CV_BayerGB2GRAY")==0) {
            code = CV_BayerGB2GRAY;
        } else if (codeStr.compare("CV_BayerRG2GRAY")==0) {
            code = CV_BayerRG2GRAY;
        } else if (codeStr.compare("CV_BayerGR2GRAY")==0) {
            code = CV_BayerGR2GRAY;
        #ifdef CV_YUV420i2RGB
        } else if (codeStr.compare("CV_YUV420i2RGB")==0) {
            code = CV_YUV420i2RGB;
        } else if (codeStr.compare("CV_YUV420i2BGR")==0) {
            code = CV_YUV420i2BGR;
        #endif
        } else if (codeStr.compare("CV_YUV420sp2RGB")==0) {
            code = CV_YUV420sp2RGB;
        } else if (codeStr.compare("CV_YUV420sp2BGR")==0) {
            code = CV_YUV420sp2BGR;
        } else {
            errMsg = "Unknown cvtColor conversion code";
        }

        if (!errMsg) {
            cvtColor(model.image, model.image, code, dstCn);
        }

        return stageOK("apply_cvtColor(%s) %s", errMsg, pStage, pStageModel);
    }

protected:
    map<int, string> cvtCode;
    int code;
    string codeStr;
    int dstCn;
};

}


#endif // CONVERT_H
