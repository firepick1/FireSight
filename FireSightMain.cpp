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

static void help() {
	cout << "FireSight image processing pipeline" << endl;
	cout << "https://github.com/firepick1/FireSight/wiki" << endl;
	cout << endl;
	cout << "Example:" << endl;
	cout << "   firesightmain -p pipeline.json -i image.jpg" << endl;
}

int main(int argc, char *argv[])
{
	firelog_level(FIRELOG_TRACE);
	LOGINFO3("FireSightMain %d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);

	char * pipelinePath = NULL;
	char * imagePath = NULL;
	for (int i = 1; i < argc; i++) {
		if (strcmp("-p",argv[i]) == 0 && i+1<argc) {
			pipelinePath = argv[i+1];
		} else if (strcmp("-i",argv[i]) == 0 && i+1<argc) {
			imagePath = argv[i+1];
		}
	}
	if (!pipelinePath) {
		help();
		exit(-1);
	}
	ifstream ifs(pipelinePath);
	stringstream pipelineStream;
	pipelineStream << ifs.rdbuf();
	Pipeline pipeline(pipelineStream.str().c_str());

	Mat image;
	if (imagePath) {
		image = imread(imagePath);
	}

	json_t *pModel = pipeline.process(image);

	char *pModelJson = json_dumps(pModel, JSON_PRESERVE_ORDER|JSON_COMPACT|JSON_INDENT(2));
	cout << pModelJson << endl;

	free(pModel);
	free(pModelJson);

  return 0;
}


