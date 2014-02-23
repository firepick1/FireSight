#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "FireLog.h"
#include "FireSight.hpp"
#include "version.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "MatUtil.hpp"
#include "test.hpp"

using namespace cv;
using namespace std;
using namespace FireSight;

int main(int argc, char *argv[])
{
	LOGINFO3("FireSight test v%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
	firelog_level(FIRELOG_TRACE);

	test_regionKeypoint();
	test_matRing();
	test_warpAffine();
	test_matMaxima();
	test_matMinima();
}
