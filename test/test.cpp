#include <string.h>
#include <fstream>
#include <sstream>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <boost/format.hpp>
#include "FireLog.h"
#include "FireSight.hpp"
#include "version.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"

using namespace cv;
using namespace std;
using namespace FireSight;

static void assertKeypoint(const KeyPoint &keypoint, double x, double y, double tolerance) {
	cout << " x:" << keypoint.pt.x << " y:" << keypoint.pt.y ;
	cout << " size:" << keypoint.size << " angle:" << keypoint.angle * 180./CV_PI;
	cout << endl;
	assert(x-tolerance <= keypoint.pt.x && keypoint.pt.x <= x+tolerance);
	assert(y-tolerance <= keypoint.pt.y && keypoint.pt.y <= y+tolerance);
}

static void test_regionKeypoint() {
	Pipeline pipeline("[]");
	KeyPoint keypoint;
	vector<Point> pts;

	pts.clear();
	pts.push_back(Point(0,1));
	pts.push_back(Point(1,1));
	pts.push_back(Point(2,1));
	cout << "pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 1, 1, 0.001);

	pts.clear();
	pts.push_back(Point(1,0));
	pts.push_back(Point(1,1));
	pts.push_back(Point(1,2));
	cout << "pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 1, 1, 0.001);

	pts.clear();
	pts.push_back(Point(0,0));
	pts.push_back(Point(1,1));
	pts.push_back(Point(2,2));
	cout << "pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 1, 1, 0.001);

	pts.clear();
	pts.push_back(Point(2,0));
	pts.push_back(Point(1,1));
	pts.push_back(Point(0,2));
	cout << "pts:" << pts << endl;
	keypoint = pipeline.regionKeypoint(pts);
	assertKeypoint(keypoint, 1, 1, 0.001);
}

int main(int argc, char *argv[])
{
	char version[30];
	sprintf(version, "FireSight test v%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
	LOGINFO1("%s", version);
	cout << version << endl;
	cout << "https://github.com/firepick1/FireSight" << endl;
	firelog_level(FIRELOG_TRACE);

	test_regionKeypoint();
}
