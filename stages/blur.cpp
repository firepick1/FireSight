#include "blur.h"
#include "jo_util.hpp"

namespace firesight {

Blur::Blur(enum Type t, Size ks, double sp, double clr, double si, int bdr) :
    type(t),
    size(ks),
    border(bdr),
    sigma(si),
    color(clr),
    space(sp),
    anchor(-1, -1)
{
    map<int, string> mapType;
    map<int, string> mapBorder;

    mapType[BILATERAL]			= "Bilateral";
    //mapType[BILATERAL_ADAPTIVE] = "Adaptive Bilateral";
    mapType[BOX]				= "Box";
    mapType[BOX_NORMALIZED]		= "Box Normalized";
    mapType[GAUSSIAN]			= "Gaussian";
    mapType[MEDIAN]				= "Median";


    mapBorder[cv::BORDER_DEFAULT]	= "Default";
    //mapBorder[BORDER_CONSTANT]	= "Constant";
    //mapBorder[BORDER_REPLICATE] = "Replicate";
    mapBorder[BORDER_ISOLATED]	= "Isolated";
    mapBorder[BORDER_REFLECT]	= "Reflect";
    mapBorder[BORDER_REFLECT_101] = "Reflect 101";
    mapBorder[BORDER_WRAP]		= "Wrap";

    _params["Type"]		= new EnumParameter(this, (int&) type, mapType);
    _params["Border"]		= new EnumParameter(this, (int&) border, mapBorder);
    _params["Kernel Size"] = new SizeParameter(this, size);
    _params["Space"]		= new DoubleParameter(this, space);
    _params["Color"]		= new DoubleParameter(this, color);
    _params["Sigma"]		= new DoubleParameter(this, sigma);
}

bool Blur::apply(json_t *pStage, json_t *pStageModel, Model &model)
{
    Pipeline::validateImage(model.image);
    const char *errMsg = NULL;
    int width = jo_int(pStage, "ksize.width", 3, model.argMap);
    int height = jo_int(pStage, "ksize.height", 3, model.argMap);
    int anchorx = jo_int(pStage, "anchor.x", -1, model.argMap);
    int anchory = jo_int(pStage, "anchor.y", -1, model.argMap);

    if (width <= 0 || height <= 0) {
        errMsg = "expected 0<width and 0<height";
    }

    if (!errMsg) {
        blur(model.image, model.image, Size(width,height));
    }

    return stageOK("apply_blur(%s) %s", errMsg, pStage, pStageModel);
}

}
