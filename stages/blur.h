#ifndef BLUR_H
#define BLUR_H

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

class Blur : public Stage
{
public:
    enum Type {
        BILATERAL,
        BILATERAL_ADAPTIVE,
        BOX,
        BOX_NORMALIZED,
        GAUSSIAN,
        MEDIAN
    };

    Blur(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        /* Blur type */
        type = GAUSSIAN;
        mapType[BILATERAL]			= "Bilateral";
        mapType[BILATERAL_ADAPTIVE] = "Adaptive Bilateral";
        mapType[BOX]				= "Box";
        mapType[BOX_NORMALIZED]		= "Box Normalized";
        mapType[GAUSSIAN]			= "Gaussian";
        mapType[MEDIAN]				= "Median";
        string stype = jo_string(pStage, "type", "Gaussian", model.argMap);
        auto findType = std::find_if(std::begin(mapType), std::end(mapType), [&](const std::pair<int, string> &pair)
        {
            return stype.compare(pair.second) == 0;
        });
        if (findType != std::end(mapType)) {
            type = findType->first;
            _params["type"] = new EnumParameter(this, type, mapType);
        } else
            throw std::invalid_argument("unknown 'type'");


        /* Kernel size */
        int width = jo_int(pStage, "ksize.width", 3, model.argMap);
        int height = jo_int(pStage, "ksize.height", 3, model.argMap);
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("expected 0<width and 0<height");
        }
        size = Size(width, height);
        _params["Kernel Size"] = new SizeParameter(this, size);


        /* Anchor */
        int anchorx = jo_int(pStage, "anchor.x", -1, model.argMap);
        int anchory = jo_int(pStage, "anchor.y", -1, model.argMap);
        anchor = Point(anchorx, anchory);
        _params["Anchor"] = new PointParameter(this, anchor);


        /* Border type */
        border = BORDER_DEFAULT;
        mapBorder[BORDER_DEFAULT]	= "Default";
        //mapBorder[BORDER_CONSTANT]	= "Constant";
        //mapBorder[BORDER_REPLICATE] = "Replicate";
        mapBorder[BORDER_ISOLATED]	= "Isolated";
        mapBorder[BORDER_REFLECT]	= "Reflect";
        mapBorder[BORDER_REFLECT_101] = "Reflect 101";
        mapBorder[BORDER_WRAP]		= "Wrap";
        string sborder = jo_string(pStage, "border", "Default", model.argMap);
        auto findBorder = std::find_if(std::begin(mapBorder), std::end(mapBorder), [&](const std::pair<int, string> &pair)
        {
            return sborder.compare(pair.second) == 0;
        });
        if (findBorder != std::end(mapBorder))
            border = findBorder->first;
        _params["Border"] = new EnumParameter(this, border, mapBorder);

        /* Bilateral filter */
        diameter = jo_int(pStage, "diameter", 1, model.argMap);
        _params["diameter"] = new IntParameter(this, diameter);

        /* Adaptive Bilateral Filter */
        sigmaSpace = jo_double(pStage, "sigmaSpace", 1, model.argMap);
        _params["sigmaSpace"] = new DoubleParameter(this, sigmaSpace);
        sigmaColor = jo_double(pStage, "sigmaColor", 1, model.argMap);
        _params["sigmaColor"] = new DoubleParameter(this, sigmaColor);

        /* Gaussian Filter */
        sigmaX = jo_double(pStage, "sigmaX", 0, model.argMap);
        _params["sigmaX"] = new DoubleParameter(this, sigmaX);
        sigmaY = jo_double(pStage, "sigmaY", 0, model.argMap);
        _params["sigmaY"] = new DoubleParameter(this, sigmaY);

        /* Median Filter */
        medianKSize = jo_int(pStage, "medianKSize", 3, model.argMap);
        _params["medianKSize"] = new IntParameter(this, medianKSize);

    }

private:
    bool apply_internal(json_t *pStageModel, Model &model)
    {
        const char *errMsg = NULL;
        Pipeline::validateImage(model.image);

        if (!errMsg) {
            switch (type) {
            case BILATERAL:
                bilateralFilter(model.image,
                                model.image,
                                diameter,
                                sigmaColor,
                                sigmaSpace,
                                border);
                break;
            case BILATERAL_ADAPTIVE:
                adaptiveBilateralFilter(model.image,
                                        model.image,
                                        size,
                                        sigmaSpace,
                                        sigmaColor,
                                        anchor,
                                        border);
                break;
            case BOX:
                boxFilter(model.image,
                          model.image,
                          -1,
                          size,
                          anchor,
                          false, // not normalized -> call BOX_NORMALIZED if you want it normalized
                          border);
                break;
            case BOX_NORMALIZED:
                blur(model.image, model.image, size);
                break;
            case GAUSSIAN:
                GaussianBlur(model.image,
                             model.image,
                             size,
                             sigmaX,
                             sigmaY,
                             border);
                break;
            case MEDIAN:
                medianBlur(model.image, model.image, medianKSize);
                break;
            default:
                break;
            }
        }

        return stageOK("apply_blur(%s) %s", errMsg, pStage, pStageModel);
    }

protected:
    int type;
    map<int, string> mapType;

    Size size; //< Kernel size
    Point anchor; //< Anchor
    int border; //< Border type
    map<int, string> mapBorder;

    int diameter; // bilateral filter
    double sigmaSpace; // sigmaSpace for (adaptive)BilateralFilter
    double sigmaColor; // maxSigmaColor for (adaptive)BilateralFilter
    double sigmaX, sigmaY; // Gaussian kternel standard deviation in X/Y direction
    int medianKSize; // ksize for Median Blur
};

}

#endif // BLUR_H


