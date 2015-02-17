/*
 * @Author  : Simon Fojtu
 * @Date    : 17.02.2015
 */

#ifndef __SHARPNESS_H_
#define __SHARPNESS_H_

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class Sharpness {

    public:

    static double GRAS(cv::Mat & image);

};

#endif

