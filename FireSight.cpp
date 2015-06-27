#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include "FireLog.h"
#include "FireSight.hpp"
#include "version.h"
#include "jo_util.hpp"
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
  cout << "Copyright 2014,2015 Karl Lew, Simon Fojtu" << endl;
  cout << "https://github.com/firepick1/FireSight/wiki" << endl;
  cout << "OpenCV " CV_VERSION << " (" << FIRESIGHT_PLATFORM_BITS << "-bit)" << endl;
  cout << endl;
  cout << "Example:" << endl;
  cout << "   firesight -p json/pipeline0.json -i img/cam.jpg -o target/output.jpg" << endl;
  cout << "   firesight -p json/pipeline1.json " << endl;
  cout << "   firesight -p json/pipeline2.json " << endl;
  cout << endl;
  cout << "All FireSight parameters are optional." << endl;
  cout << endl;
  cout << "Input parameters:" << endl;
  cout << " -i input-image-file" << endl;
  cout << "    File path of pipeline input image" << endl;
  cout << " -video " << endl;
  cout << "    Use video for pipeline input" << endl;
  cout << endl;
  cout << "Transformation parameters:" << endl;
  cout << " -Dvar=value" << endl;
  cout << "    Define pipeline parameter value" << endl;
  cout << " -ji JSON-model-indent" << endl;
  cout << "    Specify 0 for compact JSON output of model" << endl;
  cout << " -jp JSON-model-real-precision" << endl;
  cout << "    Default is 6-digit precision for real numbers in JSON model output" << endl;
  cout << " -o output-image-file" << endl;
  cout << "    File for saving pipeline image " << endl;
  cout << " -p JSON-pipeline-file" << endl;
  cout << "    JSON pipeline specification file. If omitted, input and output images must be specified." << endl;
  cout << endl;
  cout << "Diagnostic parameters:" << endl;
  cout << " -opencv " << endl;
  cout << "    OpenCV version for FireSight test compatibility" << endl;
  cout << " -debug " << endl;
  cout << "    Start logging at DEBUG log level" << endl;
  cout << " -error " << endl;
  cout << "    Start logging at ERROR log level" << endl;
  cout << " -time " << endl;
  cout << "    Time multiple executions of pipeline iterations and return average" << endl;
  cout << " -trace " << endl;
  cout << "    Start logging at TRACE log level" << endl;
  cout << " -warn " << endl;
  cout << "    Start logging at WARN log level" << endl;
  cout << " -version " << endl;
  cout << "    Print out FireSight version" << endl;
}

bool parseArgs(int argc, char *argv[], 
  string &pipelinePath, char *&imagePath, char * &outputPath, UIMode &uimode, ArgMap &argMap, bool &isTime, int &jsonIndent, int &jsonPrecision) 
{
  uimode = UI_STILL;
  isTime = false;
  firelog_level(FIRELOG_INFO);
 
  if (argc <= 1) {
    return false;
  }

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == 0) {
      // empty argument
    } else if (strcmp("-opencv",argv[i]) == 0) {
      cout << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;
      exit(0);
    } else if (strcmp("-p",argv[i]) == 0) {
      if (i+1>=argc) {
        LOGERROR("expected pipeline path after -p");
        exit(-1);
      }
      pipelinePath = argv[++i];
      LOGTRACE1("parseArgs(-p) \"%s\" is JSON pipeline path", pipelinePath.c_str());
    } else if (strcmp("-jp",argv[i]) == 0) {
      if (i+1>=argc) {
        LOGERROR("expected JSON precision after -jp");
        exit(-1);
      }
      jsonPrecision = atoi(argv[++i]);
	  if (jsonPrecision < 0) {
	  	LOGERROR1("invalid JSON precision: %d", jsonPrecision);
	  }
      LOGTRACE1("parseArgs(-jp) JSON precision:%d", jsonPrecision);
    } else if (strcmp("-ji",argv[i]) == 0) {
      if (i+1>=argc) {
        LOGERROR("expected JSON indent after -ji");
        exit(-1);
      }
      jsonIndent = atoi(argv[++i]);
      LOGTRACE1("parseArgs(-ji) JSON indent:%d", jsonIndent);
    } else if (strcmp("-o",argv[i]) == 0) {
      if (i+1>=argc) {
        LOGERROR("expected output path after -o");
        exit(-1);
      }
      outputPath = argv[++i];
      LOGTRACE1("parseArgs(-o) \"%s\" is output image path", outputPath);
    } else if (strcmp("-version",argv[i]) == 0) {
	  cout << "{\"version\":\"" << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << "\"}" << endl;
	  exit(0);
    } else if (strcmp("-time",argv[i]) == 0) {
      isTime = true;
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
    } else if (strcmp("-warn", argv[i]) == 0) {
      firelog_level(FIRELOG_WARN);
    } else if (strcmp("-error", argv[i]) == 0) {
      firelog_level(FIRELOG_ERROR);
    } else if (strcmp("-info", argv[i]) == 0) {
      firelog_level(FIRELOG_INFO);
    } else if (strcmp("-debug", argv[i]) == 0) {
      firelog_level(FIRELOG_DEBUG);
    } else if (strcmp("-trace", argv[i]) == 0) {
      firelog_level(FIRELOG_TRACE);
    } else {
      LOGERROR1("unknown firesight argument: '%s'", argv[i]);
      return false;
    }
  }
  return true;
}

/**
 * Single image example of FireSight lib_firesight library use
 */
static int uiStill(const char * pipelinePath, Mat &image, ArgMap &argMap, bool isTime, int jsonIndent, int jsonPrecision) {
  Pipeline pipeline(pipelinePath, Pipeline::PATH);
  
  json_t *pModel = pipeline.process(image, argMap);

  if (isTime) {
    long long tickStart = cvGetTickCount();
    //cout << "cvGetTickCount()" << cvGetTickCount() << endl;
    //cout << "tickStart" << tickStart << endl;
    int iterations = 100;
    for (int i=0; i < iterations; i++) {
      json_decref(pModel);
      pModel = pipeline.process(image, argMap);
    }
    float ticksElapsed = cvGetTickCount() - tickStart;
    //cout << "ticksElapsed:" << ticksElapsed << endl;
    float msElapsed = ticksElapsed/cvGetTickFrequency()*1E-3;
    //cout << "msElapsed:" << msElapsed << endl;
    float msIter = msElapsed/iterations;
    //cout << "msIter:" << msIter << endl;
    LOGINFO2("timed %d iterations with an average of %.1fms per iteration", iterations, msIter);
  }

  // Print out returned model 
  char *pModelStr = json_dumps(pModel, JSON_PRESERVE_ORDER|JSON_COMPACT|JSON_INDENT(jsonIndent)|JSON_REAL_PRECISION(jsonPrecision));
  cout << pModelStr << endl;
  free(pModelStr);

  // Free model
  json_decref(pModel);

  return 0;
}

/**
 * Video capture example of FireSight lib_firesight library use
 */
static int uiVideo(const char * pipelinePath, ArgMap &argMap) {
  VideoCapture cap(0); // open the default camera
  if(!cap.isOpened()) {  // check if we succeeded
    LOGERROR("Could not open camera");
    exit(-1);
  }

  namedWindow("image",1);

  Pipeline pipeline(pipelinePath, Pipeline::PATH);

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
  string pipelinePath;
  char * imagePath = NULL;
  char * outputPath = NULL;
  ArgMap argMap;
  bool isTime;
  int jsonIndent = 2;
  int jsonPrecision = 6;
  bool argsOk = parseArgs(argc, argv, pipelinePath, imagePath, outputPath, uimode, argMap, isTime, jsonIndent, jsonPrecision);
  if (!argsOk) {
    help();
    exit(-1);
  }

  Mat image;
  if (imagePath) {
    LOGTRACE1("Reading image: %s", imagePath);
    image = imread(imagePath);
    if (!image.data) {
      LOGERROR1("main() imread(%s) failed", imagePath);
      exit(-1);
    }
  } else {
    LOGDEBUG("No image specified.");
  }

  switch (uimode) {
    case UI_STILL: 
      uiStill(pipelinePath.c_str(), image, argMap, isTime, jsonIndent, jsonPrecision);
      break;
    case UI_VIDEO: 
      uiVideo(pipelinePath.c_str(), argMap); 
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
