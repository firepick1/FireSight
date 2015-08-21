#ifndef FILTER_H
#define FILTER_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class RectFilter: public Stage {
public:
    RectFilter(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        w = jo_int(pStage, "min_width", 0, model.argMap);
        _params["min_width"] = new IntParameter(this, w);
        W = jo_int(pStage, "max_width", 0, model.argMap);
        _params["max_width"] = new IntParameter(this, W);
        h = jo_int(pStage, "min_height", 0, model.argMap);
        _params["min_height"] = new IntParameter(this, h);
        H = jo_int(pStage, "max_height", 0, model.argMap);
        _params["max_height"] = new IntParameter(this, H);
        rectsModelName = jo_string(pStage, "model", "", model.argMap);
        _params["model"] = new StringParameter(this, rectsModelName);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        const char *errMsg = NULL;

        json_t *pRectsModel = json_object_get(model.getJson(false), rectsModelName.c_str());

        if (rectsModelName.empty()) {
            errMsg = "model: expected name of stage with rects";
        } else if (!json_is_object(pRectsModel)) {
            cout << "stage name:" << rectsModelName << endl;
            errMsg = "Named stage is not in model";
        }

        json_t *pRects = NULL;
        if (!errMsg) {
            pRects = json_object_get(pRectsModel, "rects");
            if (!json_is_array(pRects)) {
                errMsg = "Expected array of rects";
            }
        }

        json_t *pRects_out = json_array();
        json_object_set(pStageModel, "rects", pRects_out);

        if (!errMsg) {
            size_t index;
            json_t *pRect;

            json_array_foreach(pRects, index, pRect) {
                int x = jo_int(pRect, "x", SHRT_MAX, model.argMap);
                int y = jo_int(pRect, "y", SHRT_MAX, model.argMap);
                int width = jo_int(pRect, "width", -1, model.argMap);
                int height = jo_int(pRect, "height", -1, model.argMap);
                float angle = jo_float(pRect, "angle", FLT_MAX, model.argMap);

                if (width >= w && width <= W && height >= h && height <= H) {
//                    RotatedRect rect(Point(x,y), Size(width, height), angle);
//                    json_t *pRect_out = json_object();
//                    json_object_set(pRect, "x", json_real(rect.center.x));
//                    json_object_set(pRect, "y", json_real(rect.center.y));
//                    json_object_set(pRect, "width", json_real(rect.size.width));
//                    json_object_set(pRect, "height", json_real(rect.size.height));
//                    json_object_set(pRect, "angle", json_real(rect.angle));
                    json_array_append(pRects_out, pRect);
                }
            }
        }

        return stageOK("apply_rectFilter(%s) %s", errMsg, pStage, pStageModel);
    }

    // min and MAX width and height
    int w, W, h, H;
    // name of stage with rectangles
    string rectsModelName;

};

}

#endif // FILTER_H
