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
	firelog_level(FIRELOG_INFO);
	LOGINFO3("FireSightMain %d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);

	json_t *pNode = json_pack("[o,o,o]",
		json_pack("{s:s,s:s,s:s}", 
			"name", "s1", "op", "imread", "path", "/dev/firefuse/cam.jpg"),
		json_pack("{s:s,s:f,s:f,s:i}", 
	 /* "name", "s2", */ "op", "HoleRecognizer", "diamMin", 26/1.15, "diamMax", 26*1.15, "show", 1),
		json_pack("{s:s,s:s,s:s}", 
			"name", "s3","op", "imwrite", "path", "/home/pi/asdf.bmp"),
		"");

	Analyzer analyzer;
	char *pPipelineJson = json_dumps(pNode, 0);
	string modelString = analyzer.process(pPipelineJson);
	cout << modelString << endl;
	free(pPipelineJson);

  return 0;
}


