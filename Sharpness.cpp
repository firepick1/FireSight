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

double Sharpness::LAPE(Mat & image) {

    Mat src, dst;

    const int kernel_size = 3;
    const int scale = 1;
    const int delta = 0;
    const int ddepth = CV_16S;
  
    if( !image.data )
        { return 0; }
  
    if (image.channels() == 1)
        src = image;
    else
        cvtColor(image, src, CV_RGB2GRAY);

    /// Apply Laplace function
    Laplacian(src, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
  
    int32_t sum = 0;
    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols - 1; c++) {
            const int16_t x = dst.at<int16_t>(r, c);
            sum += (int32_t) x*x;
        }
    }
  
    return ((double) sum / (dst.cols * dst.rows));;
}

