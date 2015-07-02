#ifndef IMAGE_H
#define IMAGE_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class MatStage: public Stage {
public:
    MatStage(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        width = jo_int(pStage, "width", model.image.cols, model.argMap);
        _params["width"] = new IntParameter(this, width);
        height = jo_int(pStage, "height", model.image.rows, model.argMap);
        _params["height"] = new IntParameter(this, height);

        type = CV_8UC3; //!< default value
        string stype = jo_string(pStage, "type", CvTypeParser::get(type).c_str(), model.argMap);
        type = CvTypeParser::get(stype);
        mapType= CvTypeParser::get();
        _params["type"] = new EnumParameter(this, type, mapType);

        color = jo_Scalar(pStage, "color", Scalar::all(0), model.argMap);
        _params["color"] = new ScalarParameter(this, color);
    }
protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        const char *errMsg = NULL;

        if (width <= 0 || height <= 0) {
            errMsg = "Expected 0<width and 0<height";
        } else if (color[0] <0 || color[1]<0 || color[2]<0) {
            errMsg = "Expected color JSON array with non-negative values";
        }

        if (!errMsg) {
            model.image = Mat(height, width, type, color);
        }

        return stageOK("apply_Mat(%s) %s", errMsg, pStage, pStageModel);
    }

    int width;
    int height;
    int type;
    map<int, string> mapType;
    string typeStr;
    Scalar color;
};

class Split: public Stage {
public:
    Split(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) { }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        json_t *pFromTo = jo_object(pStage, "fromTo", model.argMap);
        const char *errMsg = NULL;
    #define MAX_FROMTO 32
        int fromTo[MAX_FROMTO];
        int nFromTo;

        if (!json_is_array(pFromTo)) {
            errMsg = "Expected JSON array for fromTo";
        }

        if (!errMsg) {
            json_t *pInt;
            size_t index;
            json_array_foreach(pFromTo, index, pInt) {
                if (index >= MAX_FROMTO) {
                    errMsg = "Too many channels";
                    break;
                }
                nFromTo = index+1;
                fromTo[index] = (int)json_integer_value(pInt);
            }
        }

        if (!errMsg) {
            int depth = model.image.depth();
            int channels = 1;
            Mat outImage( model.image.rows, model.image.cols, CV_MAKETYPE(depth, channels) );
            LOGTRACE1("Creating output model.image %s", matInfo(outImage).c_str());
            Mat out[] = { outImage };
            mixChannels( &model.image, 1, out, 1, fromTo, nFromTo/2 );
            model.image = outImage;
        }

        return stageOK("apply_split(%s) %s", errMsg, pStage, pStageModel);
    }
};

class ConvertTo: public Stage {
public:
    ConvertTo(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        alpha = jo_float(pStage, "alpha", 1, model.argMap);
        delta = jo_float(pStage, "delta", 0, model.argMap);
        transform = jo_string(pStage, "transform", "", model.argMap);

        rType = CV_8U;
        string stype = jo_string(pStage, "rtype", CvTypeParser::get(rType).c_str(), model.argMap);
        rType = CvTypeParser::get(stype);
        mapType= CvTypeParser::get();
        _params["rtype"] = new EnumParameter(this, rType, mapType);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        const char *errMsg = NULL;

        if (!transform.empty()) {
            if (transform.compare("log") == 0) {
                LOGTRACE("log()");
                log(model.image, model.image);
            }
        }

        if (!errMsg) {
            model.image.convertTo(model.image, rType, alpha, delta);
        }

        return stageOK("apply_convertTo(%s) %s", errMsg, pStage, pStageModel);
    }

    double alpha;
    double delta;
    string transform;
    int rType;
    map<int, string> mapType;
};

}

#endif // IMAGE_H
