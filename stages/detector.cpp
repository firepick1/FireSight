/*
 * author:  Simon Fojtu (simon.fojtu@gmail.com)
 * date  :  2014-06-09
 */

#include <string.h>
#include <math.h>
#include "FireLog.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"

#include "stages/detector.h"

using namespace cv;
using namespace std;

namespace firesight {


    void HoughCircles::scan(Mat &image, vector<Circle> &circles) {
        Mat matGray;
        if (image.channels() > 1)
            cvtColor(image, matGray, CV_RGB2GRAY);
        else
            matGray = image.clone();


        vector<Vec3f> vec3f_circles;
        cv::HoughCircles(matGray, vec3f_circles, CV_HOUGH_GRADIENT, hc_dp, hc_minDist, hc_param1, hc_param2, diamMin/2.0, diamMax/2.0);
        for (size_t i = 0; i < vec3f_circles.size(); i++) {
            circles.push_back(Circle(vec3f_circles[i][0], vec3f_circles[i][1], vec3f_circles[i][2]));
        }

        LOGTRACE1("HoughCircle::scan() -> found %d circles", (int) circles.size());

        if (showCircles)
            show(image, circles);
    }

    void HoughCircles::show(Mat & image, vector<Circle> circles) {
      for( size_t i = 0; i < circles.size(); i++ ) {
          Point center(cvRound(circles[i].x), cvRound(circles[i].y));
          int radius = cvRound(circles[i].radius);
          // circle center
          circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );
          // circle outline
          circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
       }

    }

}
