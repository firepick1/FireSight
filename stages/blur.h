#ifndef BLUR_H
#define BLUR_H

#include "Pipeline.h"

namespace firesight {

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

    Blur(enum Type t, Size ks, double sp = 10, double clr = 20, double sigma = 2, int bdr = BORDER_DEFAULT);

    bool apply(json_t *pStage, json_t *pStageModel, Model &model);

protected:
    enum Blur::Type type;

    Size size;
    Point anchor;
    int border;

    double sigma;
    double color;
    double space;
};

}

#endif // BLUR_H
