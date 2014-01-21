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
	LOGTRACE3("FireSightMain %d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);

	json_t *pNode = json_pack("[o,o,o]",
		json_pack("{s:s,s:s}", "apply", "imread", "path", "/dev/firefuse/cam.jpg"),
		json_pack("{s:s,s:f,s:f,s:i}", "apply", "HoleRecognizer", "diamMin", 26/1.15, "diamMax", 26*1.15, "show", 1),
		json_pack("{s:s,s:s}", "apply", "imwrite", "path", "/home/pi/asdf.bmp"),
		"");

	Analyzer analyzer;
	char *pJson = json_dumps(pNode, 0);
	analyzer.process(pJson);
	free(pJson);

  return 0;
}


