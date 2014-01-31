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

typedef enum{UI_STILL, UI_VIDEO} UIMode;

static void help() {
	cout << "FireSight image processing pipeline" << endl;
	cout << "https://github.com/firepick1/FireSight/wiki" << endl;
	cout << endl;
	cout << "Example:" << endl;
	cout << "   firesight -p json/pipeline0.json -i img/cam.jpg -o target/output.jpg" << endl;
	cout << "   firesight -p json/pipeline1.json " << endl;
	cout << "   firesight -p json/pipeline2.json " << endl;
}

void parseArgs(int argc, char *argv[], string &pipelineString, char *&imagePath, char * &outputPath, UIMode &uimode) {
	char *pipelinePath;
	uimode = UI_STILL;
	firelog_level(FIRELOG_INFO);

	for (int i = 1; i < argc; i++) {
		if (strcmp("-p",argv[i]) == 0) {
			if (i+1>=argc) {
				LOGERROR("expected pipeline path after -p");
				exit(-1);
			}
			pipelinePath = argv[++i];
			LOGTRACE1("-p %s is JSON pipeline path", pipelinePath);
		} else if (strcmp("-o",argv[i]) == 0) {
			if (i+1>=argc) {
				LOGERROR("expected output path after -o");
				exit(-1);
			}
			outputPath = argv[++i];
			LOGTRACE1("-o %s is output image path", outputPath);
		} else if (strcmp("-i",argv[i]) == 0) {
			if (i+1>=argc) {
				LOGERROR("expected image path after -i");
				exit(-1);
			}
			imagePath = argv[++i];
			LOGTRACE1("-i %s is input image path", imagePath);
		} else if (strcmp("-video", argv[i]) == 0) {
			uimode = UI_VIDEO;
			LOGTRACE("-video UI_VIDEO user interface selected");
		} else if (strcmp("-trace", argv[i]) == 0) {
			firelog_level(FIRELOG_TRACE);
		} else {
			LOGERROR1("argument error detected at: %s", argv[i]);
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
static int uiStill(const char * pJsonPipeline, Mat &image) {
	Pipeline pipeline(pJsonPipeline);

	json_t *pModel = pipeline.process(image);

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
static int uiVideo(const char * pJsonPipeline) {
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

		json_t *pModel = pipeline.process(frame);

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
	char version[30];
	sprintf(version, "FireSight v%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
	LOGINFO1("%s", version);
	cout << version << endl;
	cout << "https://github.com/firepick1/FireSight" << endl;

	UIMode uimode;
	string pipelineString;
	char * imagePath = NULL;
	char * outputPath = NULL;
	parseArgs(argc, argv, pipelineString, imagePath, outputPath, uimode); 

	Mat image;
	if (imagePath) {
		LOGTRACE1("Reading image: %s", imagePath);
		image = imread(imagePath);
		if (!image.data) {
		  cout << "imread failed: " << imagePath << endl;
			exit(-1);
		}
	} else {
	  LOGINFO("No image specified in command line. Pipeline must specify image.");
	}

  const char *pJsonPipeline = pipelineString.c_str();

	switch (uimode) {
		case UI_STILL: 
			uiStill(pJsonPipeline, image); 
			break;
		case UI_VIDEO: 
			uiVideo(pJsonPipeline); 
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


