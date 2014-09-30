#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "FireLog.h"
#include "FireSight.hpp"
#include "version.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "jansson.h"
#include "MatUtil.hpp"

using namespace cv;
using namespace std;
using namespace firesight;

extern void test_matRing();
extern void test_regionKeypoint(); 
extern void test_warpAffine(); 
extern void test_matMaxima(); 
extern void test_matMinima(); 
extern void test_jo_util();
extern void test_calibrate();

int main(int argc, char *argv[])
{
    LOGINFO3("FireSight test v%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    firelog_level(FIRELOG_TRACE);

    cout << "test_calibrate()" << endl;
    test_calibrate();
    cout << "test_regionKeypoint()" << endl;
    test_regionKeypoint();
    cout << "test_matRing()" << endl;
    test_matRing();
    cout << "test_warpAffine()" << endl;
    test_warpAffine();
    cout << "test_matMaxima()" << endl;
    test_matMaxima();
    cout << "test_matMinima()" << endl;
    test_matMinima();
    cout << "test_jo_util()" << endl;
    test_jo_util();

    cout << "END OF TEST main()" << endl;
}
