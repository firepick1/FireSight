#ifndef CONSOLE_H
#define CONSOLE_H

/*
 * @Author  : Simon Fojtu
 * @Date    : 26.06.2015
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>

namespace firesight {

using namespace cv;

struct Console {
public:
    Console(Mat& img) : image(img), y(0) {}
    void print(const string &str, int width = 1, Scalar color = Scalar(255,255,255));
private:
    Mat &image;
    int y;
};

void Console::print(const string &str, int width, Scalar color)
{
    putText(image, str, Point(10, 10 + y * 10),
            FONT_HERSHEY_PLAIN,
            0.8,
            Scalar(255,255,255)-color,
            width+1,
            CV_AA);

    putText(image, str, Point(10, 10 + y * 10),
            FONT_HERSHEY_PLAIN,
            0.8,
            color,
            width,
            CV_AA);

    y++;
}

}

#endif // CONSOLE_H
