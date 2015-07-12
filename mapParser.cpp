#include "mapParser.h"

#include <algorithm>
#include <stdexcept>

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>


using namespace std;
using namespace cv;

namespace firesight {

map<int, string> CvTypeParser_::amap = [] {
    map<int, string> ret;
    ret[CV_8UC3] = "CV_8UC3";
    ret[CV_8UC2] = "CV_8UC2";
    ret[CV_8UC1] = "CV_8UC1";
    ret[CV_8UC1] = "CV_8U";
    ret[CV_32F]  = "CV_32F";
    ret[CV_32FC1]  = "CV_32FC1";
    ret[CV_32FC2]  = "CV_32FC2";
    ret[CV_32FC3]  = "CV_32FC3";
    return ret;
}();

map<int, string> BorderTypeParser_::amap = [] {
    map<int, string> ret;
    ret[BORDER_DEFAULT]	= "Default";
    ret[BORDER_CONSTANT]	= "Constant";
    ret[BORDER_REPLICATE] = "Replicate";
    ret[BORDER_ISOLATED]	= "Isolated";
    ret[BORDER_REFLECT]	= "Reflect";
    ret[BORDER_REFLECT_101] = "Reflect 101";
    ret[BORDER_WRAP]		= "Wrap";
    return ret;
}();

map<int, string> BlurTypeParser_::amap = [] {
    map<int, string> ret;
    ret[BILATERAL]			= "Bilateral";
    ret[BILATERAL_ADAPTIVE] = "Adaptive Bilateral";
    ret[BOX]				= "Box";
    ret[BOX_NORMALIZED]		= "Box Normalized";
    ret[GAUSSIAN]			= "Gaussian";
    ret[MEDIAN]         	= "Median";
    return ret;
}();

map<int, string> BGSubTypeParser_::amap = [] {
    map<int, string> ret;
    ret[MOG]			= "MOG";
    ret[MOG2]			= "MOG2";
    ret[ABSDIFF]			= "absdiff";
    return ret;
}();



}
