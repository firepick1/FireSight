/*
 * @Author  : Simon Fojtu
 * @Date    : 21.05.2015
 */

#ifndef __PARTDETECTOR_H_
#define __PARTDETECTOR_H_

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class PartDetector {

    public:


        static cv::RotatedRect detect(cv::Mat& image);

};

#endif

