#include <string.h>
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


int main(int argc, char *argv[])
{
	firelog_level(FIRELOG_TRACE);
	LOGTRACE2("FireSightMain %d.%d", VERSION_MAJOR, VERSION_MINOR);

	Mat matRGB = imread("/dev/firefuse/cam.jpg", CV_LOAD_IMAGE_COLOR);
	LOGTRACE2("matRGB cols:%d rows:%d", matRGB.cols, matRGB.rows);

	json_t *pNode = json_pack("[o,o,o]",
		json_pack("{s:s,s:s}", "op", "imread", "path", "/dev/firefuse/cam.jpg"),
		json_pack("{s:s,s:f,s:f}", "op", "HoleRecognizer", "aMin", 26/1.15, "aMax", 26*1.15),
		json_pack("{s:s,s:s}", "op", "imwrite", "path", "/home/pi/camcv.bmp"),
		"");

	Analyzer analyzer;
	char *pJson = json_dumps(pNode, 0);
	analyzer.process(pJson);

	free(pJson);

	vector<MatchedRegion> matches;
	HoleRecognizer recognizer(26/1.15, 26*1.15);
  recognizer.scan(matRGB, matches);

	imwrite("/home/pi/camcv.bmp", matRGB);

	for (int i = 0; i < matches.size(); i++) {
		cout << "match " << i+1 << " = " << matches[i].asJson() << endl;
	}

  return 0;
}


