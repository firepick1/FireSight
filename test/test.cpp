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

void dump_calibration(double result, const Mat &cameraMatrix, const Mat &distCoeffs,
                      const vector<Mat> &rvecs, const vector<Mat> &tvecs) {
    cout << "calibrateCamera() => " << result << endl;
	cout << "cameraMatrix " << matInfo(cameraMatrix) << endl << cameraMatrix << endl;
	cout << "distCoeffs " << matInfo(distCoeffs) << endl << distCoeffs << endl;
	for (int i=0; i < rvecs.size(); i++) {
		cout << "rvecs[" << i << "] " << matInfo(rvecs[i]) << endl << rvecs[i] << endl;
	}
	for (int i=0; i < tvecs.size(); i++) {
		cout << "tvecs[" << i << "] " << matInfo(tvecs[i]) << endl << tvecs[i] << endl;
	}
}

void test_calibrateCamera() {
	cout << "test_calibrateCamera() BEGIN-------------" << endl;

    vector< vector<Point2f> > imagePoints;
    vector< vector<Point3f> > objectPoints;
    Size imageSize(200,200);
    Mat cameraMatrix;
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
	vector<Point3f> object;
	vector<Point2f> image;

	cout << "test_calibrateCamera() scale(10,9)" << endl;
	objectPoints.clear();
	imagePoints.clear();
	object.clear();
	object.push_back(Point3f(0,0,0));
	object.push_back(Point3f(10,0,0));
	object.push_back(Point3f(10,20,0));
	object.push_back(Point3f(0,20,0));
	object.push_back(Point3f(3,5,0));
	object.push_back(Point3f(7,7,0));
	objectPoints.push_back(object);
	image.clear();
	image.push_back(Point2f(0,0));
	image.push_back(Point2f(100,0));
	image.push_back(Point2f(100,180));
	image.push_back(Point2f(0,180));
	image.push_back(Point2f(30,45));
	image.push_back(Point2f(70,63));
	imagePoints.push_back(image);

    double result = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
                                    rvecs, tvecs);
    dump_calibration(result, cameraMatrix, distCoeffs, rvecs, tvecs);
}

int main(int argc, char *argv[])
{
    LOGINFO3("FireSight test v%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    firelog_level(FIRELOG_TRACE);

    test_calibrateCamera();
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
