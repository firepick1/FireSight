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

    Mat src, dst, kernel;

    const int kernel_size = 1;
    const int scale = 1;
    const int delta = 0;
    const int ddepth = CV_16S;
  
    if( !image.data )
        { return 0; }
  
    if (image.channels() == 1)
        src = image;
    else
        cvtColor(image, src, CV_RGB2GRAY);

    // Laplace operator according to MATLAB: fspecial('laplacian')
    kernel = Mat::zeros(3, 3, CV_32F);
    const double alpha = 0.2;
    kernel.at<float>(0,0) = 4.0 / (alpha+1) * alpha / 4;
    kernel.at<float>(0,1) = 4.0 / (alpha+1) * (1-alpha)/4;;
    kernel.at<float>(0,2) = 4.0 / (alpha+1) * alpha / 4;
    kernel.at<float>(1,0) = 4.0 / (alpha+1) * (1-alpha) / 4;
    kernel.at<float>(1,1) = -4.0 / (alpha+1);
    kernel.at<float>(1,2) = 4.0 / (alpha+1) * (1-alpha) / 4;
    kernel.at<float>(2,0) = 4.0 / (alpha+1) * alpha / 4;
    kernel.at<float>(2,1) = 4.0 / (alpha+1) * (1-alpha)/4;;
    kernel.at<float>(2,2) = 4.0 / (alpha+1) * alpha / 4;

    /// Apply Laplace function
    filter2D(src, dst, ddepth, kernel, Point(-1, -1), delta, BORDER_REPLICATE);

  
    int32_t sum = 0;
    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols - 1; c++) {
            const int16_t x = dst.at<int16_t>(r, c);
            sum += (int32_t) x*x;
        }
    }
  
    return ((double) sum / (dst.cols * dst.rows));;
}

double Sharpness::LAPM(Mat & image) {
    Mat src, dst;
  
    Mat kernel;
    Point anchor = Point(-1, -1);;
    const double delta = 0;
    const int ddepth = CV_16S;

    int32_t sum = 0;

    if( !image.data )
        { return 0; }
  
    if (image.channels() == 1)
        src = image;
    else
        cvtColor(image, src, CV_RGB2GRAY);

    kernel = Mat::zeros(3, 3, CV_32F);
    kernel.at<float>(0,1) = -1;
    kernel.at<float>(1,1) = 2;
    kernel.at<float>(2,1) = -1;
    filter2D(src, dst, ddepth, kernel, anchor, delta, BORDER_REPLICATE);

    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            const int16_t x = abs(dst.at<int16_t>(r, c));
            sum += (int32_t) x;
        }
    }

    kernel = Mat::zeros(3, 3, CV_32F);
    kernel.at<float>(1,0) = -1;
    kernel.at<float>(1,1) = 2;
    kernel.at<float>(1,2) = -1;
    filter2D(src, dst, ddepth, kernel, anchor, delta, BORDER_REPLICATE);

    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            const int16_t x = abs(dst.at<int16_t>(r, c));
            sum += (int32_t) x;
        }
    }

    return ((double) sum / (dst.cols * dst.rows));;
}
