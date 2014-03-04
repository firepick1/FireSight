#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include "FireLog.h"
#include "FireSight.hpp"
#include "version.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"

using namespace cv;
using namespace std;
using namespace firesight;

typedef enum{UI_STILL, UI_VIDEO} UIMode;

static void help() {
	cout << "FireSight image processing pipeline v" << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << endl;
	cout << "https://github.com/firepick1/FireSight/wiki" << endl;
	cout << endl;
	cout << "Example:" << endl;
	cout << "   firesight -p json/pipeline0.json -i img/cam.jpg -o target/output.jpg" << endl;
	cout << "   firesight -p json/pipeline1.json " << endl;
	cout << "   firesight -p json/pipeline2.json " << endl;
}

void parseArgs(int argc, char *argv[], string &pipelineString, char *&imagePath, char * &outputPath, UIMode &uimode, ArgMap &argMap) {
	char *pipelinePath = NULL;
	uimode = UI_STILL;
	firelog_level(FIRELOG_INFO);

	for (int i = 1; i < argc; i++) {
		if (strcmp("-p",argv[i]) == 0) {
			if (i+1>=argc) {
				LOGERROR("expected pipeline path after -p");
				exit(-1);
			}
			pipelinePath = argv[++i];
			LOGTRACE1("parseArgs(-p) \"%s\" is JSON pipeline path", pipelinePath);
		} else if (strcmp("-o",argv[i]) == 0) {
			if (i+1>=argc) {
				LOGERROR("expected output path after -o");
				exit(-1);
			}
			outputPath = argv[++i];
			LOGTRACE1("parseArgs(-o) \"%s\" is output image path", outputPath);
		} else if (strncmp("-D",argv[i],2) == 0) {
			char * pEq = strchr(argv[i],'=');
			if (!pEq || (pEq-argv[i])<=2) {
				LOGERROR("expected argName=argValue pair after -D");
				exit(-1);
			}
			*pEq = 0;
			char *pName = argv[i] + 2;
			char *pVal = pEq + 1;
			argMap[pName] = pVal;
			LOGTRACE2("parseArgs(-D) argMap[%s]=\"%s\"", pName, pVal );
			*pEq = '=';
		} else if (strcmp("-i",argv[i]) == 0) {
			if (i+1>=argc) {
				LOGERROR("expected image path after -i");
				exit(-1);
			}
			imagePath = argv[++i];
			LOGTRACE1("parseArgs(-i) \"%s\" is input image path", imagePath);
		} else if (strcmp("-video", argv[i]) == 0) {
			uimode = UI_VIDEO;
			LOGTRACE("parseArgs(-video) UI_VIDEO user interface selected");
		} else if (strcmp("-info", argv[i]) == 0) {
			firelog_level(FIRELOG_INFO);
		} else if (strcmp("-debug", argv[i]) == 0) {
			firelog_level(FIRELOG_DEBUG);
		} else if (strcmp("-trace", argv[i]) == 0) {
			firelog_level(FIRELOG_TRACE);
		} else {
			LOGERROR1("unknown firesight argument: '%s'", argv[i]);
			help();
			exit(-1);
		}
	}
	if (!pipelinePath) {
		help();
		exit(-1);
	}

	LOGTRACE1("Reading pipeline: %s", pipelinePath);
	ifstream ifs(pipelinePath);
	stringstream pipelineStream;
	pipelineStream << ifs.rdbuf();
	pipelineString = pipelineStream.str();
	const char *pJsonPipeline = pipelineString.c_str();
	if (strlen(pJsonPipeline) < 10) {
		cout << "invalid pipeline: " << pipelinePath << endl;
		exit(-1);
	}
}

/**
 * Single image example of FireSight lib_firesight library use
 */
static int uiStill(const char * pJsonPipeline, Mat &image, ArgMap &argMap) {
	Pipeline pipeline(pJsonPipeline);

	json_t *pModel = pipeline.process(image, argMap);

	// Print out returned model 
	char *pModelStr = json_dumps(pModel, JSON_PRESERVE_ORDER|JSON_COMPACT|JSON_INDENT(2));
	cout << pModelStr << endl;
	free(pModelStr);

	// Free model
	json_decref(pModel);

  return 0;
}

/**
 * Video capture example of FireSight lib_firesight library use
 */
static int uiVideo(const char * pJsonPipeline, ArgMap &argMap) {
	VideoCapture cap(0); // open the default camera
	if(!cap.isOpened()) {  // check if we succeeded
		LOGERROR("Could not open camera");
		exit(-1);
	}

	namedWindow("image",1);

	Pipeline pipeline(pJsonPipeline);

	for(;;) {
		Mat frame;
		cap >> frame; // get a new frame from camera

		json_t *pModel = pipeline.process(frame, argMap);

		// Display pipeline output
		imshow("image", frame);
		if(waitKey(30) >= 0) break;

		// Free model
		json_decref(pModel);
	}

  return 0;
}

int main(int argc, char *argv[])
{
	UIMode uimode;
	string pipelineString;
	char * imagePath = NULL;
	char * outputPath = NULL;
	ArgMap argMap;
	parseArgs(argc, argv, pipelineString, imagePath, outputPath, uimode, argMap); 

	Mat image;
	if (imagePath) {
		LOGTRACE1("Reading image: %s", imagePath);
		image = imread(imagePath);
		if (!image.data) {
		  cout << "imread failed: " << imagePath << endl;
			exit(-1);
		}
	} else {
		LOGDEBUG("No image specified.");
	}

  const char *pJsonPipeline = pipelineString.c_str();

	switch (uimode) {
		case UI_STILL: 
			uiStill(pJsonPipeline, image, argMap); 
			break;
		case UI_VIDEO: 
			uiVideo(pJsonPipeline, argMap); 
			break;
		default: 
			LOGERROR("Unknown UI mode");
			exit(-1);
	}

	if (outputPath) {
		if (!imwrite(outputPath, image)) {
		  LOGERROR1("Could not write image to: %s", outputPath);
			exit(-1);
		}
		LOGTRACE1("Image written to: %s", outputPath);
	}

	return 0;
}
