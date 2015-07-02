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


    Blur(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        /* Blur type */
        type = GAUSSIAN; //!< default value
        string stype = jo_string(pStage, "type", BlurTypeParser::get(type).c_str(), model.argMap);
        type = BlurTypeParser::get(stype);
        mapType = BlurTypeParser::get();
        _params["type"] = new EnumParameter(this, type, mapType);

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
        border = BORDER_DEFAULT; //!< default value
        string sborder = jo_string(pStage, "border", BorderTypeParser::get(border).c_str(), model.argMap);
        border = BorderTypeParser::get(sborder);
        mapBorder = BorderTypeParser::get();
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
        validateImage(model.image);

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


