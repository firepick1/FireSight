#include "blur.h"

Blur::Blur(enum Type t, Size ks, double sp, double clr, double si, int bdr) :
    type(t),
    size(ks),
    border(bdr),
    sigma(si),
    color(clr),
    space(sp)
{
    map<int, string> mapType;
    map<int, string> mapBorder;

    mapType[BILATERAL]			= "Bilateral";
    //mapType[BILATERAL_ADAPTIVE] = "Adaptive Bilateral";
    mapType[BOX]				= "Box";
    mapType[BOX_NORMALIZED]		= "Box Normalized";
    mapType[GAUSSIAN]			= "Gaussian";
    mapType[MEDIAN]				= "Median";

    mapBorder[BORDER_DEFAULT]	= "Default";
    //mapBorder[BORDER_CONSTANT]	= "Constant";
    //mapBorder[BORDER_REPLICATE] = "Replicate";
    mapBorder[BORDER_ISOLATED]	= "Isolated";
    mapBorder[BORDER_REFLECT]	= "Reflect";
    mapBorder[BORDER_REFLECT_101] = "Reflect 101";
    mapBorder[BORDER_WRAP]		= "Wrap";

    settings["Type"]		= new EnumSetting(this, (int&) type, mapType);
    settings["Border"]		= new EnumSetting(this, (int&) border, mapBorder);
    settings["Kernel Size"] = new SizeSetting(this, size);
    settings["Space"]		= new DoubleParameter(this, space);
    settings["Color"]		= new DoubleParameter(this, color);
    settings["Sigma"]		= new DoubleParameter(this, sigma);
}
