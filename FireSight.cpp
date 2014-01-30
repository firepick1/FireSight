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
	cout << "   firesight -p json/pipeline0.json -i img/cam.jpg " << endl;
	cout << "   firesight -p json/pipeline1.json " << endl;
	cout << "   firesight -p json/pipeline2.json " << endl;
}

int main(int argc, char *argv[])
{
	char version[30];
	sprintf(version, "FireSight v%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
	LOGINFO1("%s", version);
	cout << version << endl;
	cout << "https://github.com/firepick1/FireSight" << endl;

	char * pipelinePath = NULL;
	char * imagePath = NULL;
	int firelogLevel = FIRELOG_INFO;
	for (int i = 1; i < argc; i++) {
		if (strcmp("-p",argv[i]) == 0) {
			if (i+1>=argc) {
				LOGERROR("expected pipeline path after -p");
				exit(-1);
			}
			pipelinePath = argv[++i];
		} else if (strcmp("-i",argv[i]) == 0) {
			if (i+1>=argc) {
				LOGERROR("expected image path after -i");
				exit(-1);
			}
			imagePath = argv[++i];
		} else if (strcmp("-trace", argv[i]) == 0) {
			firelogLevel = FIRELOG_TRACE;
		} else {
			LOGERROR1("argument error detected at: %s", argv[i]);
			help();
			exit(-1);
		}
	}
	firelog_level(firelogLevel);
	if (!pipelinePath) {
		help();
		exit(-1);
	}
	LOGTRACE1("Reading pipeline: %s", pipelinePath);
	ifstream ifs(pipelinePath);
	stringstream pipelineStream;
	pipelineStream << ifs.rdbuf();
	string pipelineString = pipelineStream.str();
	const char *pJsonPipeline = pipelineString.c_str();
	if (strlen(pJsonPipeline) < 10) {
		cout << "invalid pipeline: " << pipelinePath << endl;
		exit(-1);
	}

	Mat image;
	if (imagePath) {
		LOGTRACE1("Reading image: %s", imagePath);
		image = imread(imagePath);
		if (!image.data) {
		  cout << "imread failed: " << imagePath << endl;
			exit(-1);
		}
	}

	// To use FireSight as a library, simple create a Pipeline with a JSON string
	Pipeline pipeline(pJsonPipeline);

	// FOR-EACH-IMAGE-START: Use pipeline to process image (remember to free the returned model when done!)
	json_t *pModel = pipeline.process(image);

	// Print out returned model 
	char *pModelJson = json_dumps(pModel, JSON_PRESERVE_ORDER|JSON_COMPACT|JSON_INDENT(2));
	cout << pModelJson << endl;
	free(pModelJson);

	// Free model
	free(pModel);

	// FOR-EACH-IMAGE-END: Repeat for each image

  return 0;
}


