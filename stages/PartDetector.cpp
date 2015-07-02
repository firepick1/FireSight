/*
 * @Author  : Simon Fojtu
 * @Date    : 21.05.2015
 */

#include "detector.h"

#include <stdio.h>
#include <iostream>

using namespace cv;

namespace firesight {

vector<RotatedRect> PartDetector::detect(Mat& image) {
    vector<RotatedRect> result;

    std::vector<std::vector<Point> > contours;
    findContours(image,contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    for( const auto& contour : contours ) //int i = 0; i< contours.size(); i++ )
    {
        RotatedRect boundingBox = minAreaRect(contour);

        result.push_back(boundingBox);

        // draw
//        Point2f corners[4];
//        boundingBox.points(corners);
//        line(image, corners[0], corners[1], Scalar(255,255,255));
//        line(image, corners[1], corners[2], Scalar(255,255,255));
//        line(image, corners[2], corners[3], Scalar(255,255,255));
//        line(image, corners[3], corners[0], Scalar(255,255,255));
    }

    return result;
}

}
