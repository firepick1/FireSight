/*
 * @Author  : Simon Fojtu
 * @Date    : 17.02.2015
 */

#include "Sharpness.h"
#include <stdio.h>

using namespace cv;

double Sharpness::GRAS(Mat & image) {
    int sum = 0;
    Mat matGray;

    if (image.channels() == 1)
        matGray = image;
    else
        cvtColor(image, matGray, CV_RGB2GRAY);

    for (int r = 0; r < matGray.rows; r++) {
        for (int c = 0; c < matGray.cols - 1; c++) {
            const int df = (int) matGray.at<uint8_t>(r, c) - (int) matGray.at<uint8_t>(r, c+1);
            sum += df * df;
        }
    }

    return ((double) sum / matGray.rows / (matGray.cols - 1));
}


