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

    /**
     * GRAS - Absolute squared gradient
     *
     * A.M. Eskicioglu and P. S. Fisher. Image quality measures and their performance. Communications, IEEE Transactions on, 43(12):2959–2965, 1995
     *
     * MATLAB:
     * Ix = diff(Image, 1, 2);
     * FM = Ix.^2;
     * FM = mean2(FM);
     */
    static double GRAS(cv::Mat & image);

    /**
     * LAPE - Energy of laplacian [Subbarao92a]
     *
     * MATLAB:
     * LAP = fspecial('laplacian');
     * FM = imfilter(Image, LAP, 'replicate', 'conv');
     * FM = mean2(FM.^2);
     */
    static double LAPE(cv::Mat & image);

    /**
     * LAPM - Modified Laplacian [Nayar89]
     *
     * MATLAB:
     * M = [-1 2 -1];        
     * Lx = imfilter(Image, M, 'replicate', 'conv');
     * Ly = imfilter(Image, M', 'replicate', 'conv');
     * FM = abs(Lx) + abs(Ly);
     * FM = mean2(FM);
     */
    static double LAPM(cv::Mat & image);
};

#endif

