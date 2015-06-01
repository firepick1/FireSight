#include <cassert>
#include <string.h>
#include <math.h>
#include <iostream>
#include <stdexcept>
#include "FireLog.h"
#include "Pipeline.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "jo_util.hpp"
#include "MatUtil.hpp"
#include "version.h"

using namespace cv;
using namespace std;
using namespace firesight;

#define TRUE 1

class SubtractorStageData : public StageData {
  public:
    BackgroundSubtractor *pSubtractor;

    SubtractorStageData(string stageName, BackgroundSubtractor *pSubtractor) : StageData(stageName) {
      assert(pSubtractor);
      this->pSubtractor = pSubtractor;
    }

    ~SubtractorStageData() {
      LOGTRACE("Freeing BackgroundSubtractor");
      delete pSubtractor;
    }
};


bool Pipeline::apply_backgroundSubtractor(json_t *pStage, json_t *pStageModel, Model &model) {
  validateImage(model.image);
  int history = jo_int(pStage, "history", 0, model.argMap);
  float varThreshold = jo_float(pStage, "varThreshold", 16, model.argMap);
  bool bShadowDetection = jo_bool(pStage, "bShadowDetection", TRUE, model.argMap);
  string background = jo_string(pStage, "background", "", model.argMap);
  string method = jo_string(pStage, "method", "MOG2", model.argMap);
  string stageName = jo_string(pStage, "name", method.c_str(), model.argMap);
  int maxval = 255;
  double learningRate = jo_double(pStage, "learningRate", -1, model.argMap);
  const char *errMsg = NULL;
  StageDataPtr pStageData = model.stageDataMap[stageName];

  BackgroundSubtractor *pSubtractor;
  bool is_absdiff = false;
  if (!errMsg) {
    if (method.compare("MOG2") == 0) {
      if (pStageData) {
	pSubtractor = ((SubtractorStageData *) pStageData)->pSubtractor;
      } else {
	pSubtractor = new BackgroundSubtractorMOG2(history, varThreshold, bShadowDetection);
	model.stageDataMap[stageName] = new SubtractorStageData(stageName, pSubtractor);
      }
    } else if (method.compare("absdiff") == 0) {
      is_absdiff = true;
    } else {
        errMsg = "Expected method: MOG2";
    }
  }

  Mat bgImage;
  if (!background.empty()) {
    if (history != 0) {
      errMsg = "Expected history=0 if background image is specified";
    } else {
      if (model.image.channels() == 1) {
        bgImage = imread(background.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
      } else {
        bgImage = imread(background.c_str(), CV_LOAD_IMAGE_COLOR);
      }
      if (bgImage.data) {
        LOGTRACE2("apply_backgroundSubtractor(%s) %s", background.c_str(), matInfo(bgImage).c_str());
        if (model.image.rows!=bgImage.rows || model.image.cols!=bgImage.cols) {
          errMsg = "Expected background image of same size as pipeline image";
        }
      } else {
        errMsg = "Could not load background image";
      }
    }
  }
  if (!errMsg) {
    if (history < 0) {
      errMsg = "Expected history >= 0";
    }
  }

  if (!errMsg) { 
    Mat fgMask;
    if (is_absdiff) {
      absdiff(model.image, bgImage, fgMask);
      if (fgMask.channels() > 1) {
        cvtColor(fgMask, fgMask, CV_BGR2GRAY);
      }
      threshold(fgMask, model.image, varThreshold, maxval, THRESH_BINARY);
    } else {
      if (bgImage.data) {
	pSubtractor->operator()(bgImage, fgMask, learningRate);
      }
      pSubtractor->operator()(model.image, fgMask, learningRate);
      model.image = fgMask;
    }
  }

  return stageOK("apply_backgroundSubtractor(%s) %s", errMsg, pStage, pStageModel);
}
