#ifndef BLUR_H
#define BLUR_H

#include "Pipeline.h"

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

protected:
    enum Blur::Type type;

    Size size;
    Point anchor = Point(-1, -1);
    int border;

    double sigma;
    double color;
    double space;
};

#endif // BLUR_H
