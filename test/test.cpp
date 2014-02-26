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

using namespace cv;
using namespace std;
using namespace firesight;

extern void test_matRing();
extern void test_regionKeypoint(); 
extern void test_warpAffine(); 
extern void test_matMaxima(); 
extern void test_matMinima(); 
extern void test_jo_util();

int main(int argc, char *argv[])
{
	LOGINFO3("FireSight test v%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
	firelog_level(FIRELOG_TRACE);

	test_regionKeypoint();
	test_matRing();
	test_warpAffine();
	test_matMaxima();
	test_matMinima();
	test_jo_util();
}
