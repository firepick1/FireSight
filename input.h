#ifndef INPUT_H
#define INPUT_H

/*
 * @Author  : Simon Fojtu
 * @Date    : 26.06.2015
 */

#include <stdexcept>
#include "FireLog.h"

namespace firesight {

using namespace std;
using namespace cv;


struct Input {
    virtual const Mat get() {}
};

struct ImageInput : public Input {
    ImageInput(const Mat& input) {
       image = input.clone();
    }

    ImageInput(string imagePath) {
        LOGTRACE1("Reading image: %s", imagePath.c_str());
        image = imread(imagePath);
        if (!image.data) {
            LOGERROR1("main() imread(%s) failed", imagePath.c_str());
          throw invalid_argument("Failed to read image" + imagePath);
        }
    }

    const Mat get() {
        return image;
    }

    cv::Mat image;
};

struct VideoInput : public Input {
    VideoInput() {
        cap = VideoCapture(0); // open the default camera
        if(!cap.isOpened()) {  // check if we succeeded
          LOGERROR("Could not open camera");
          throw invalid_argument("Could not open camera");
        }
    }

    const Mat get() {
        cap >> lastFrame;
//        bool success = cap.read(lastFrame);
//        if (!success)
//            throw std::runtime_error("Failed to read frame");

        return lastFrame;
    }

    VideoCapture cap;
    Mat lastFrame;
};

}

#endif // INPUT_H
