#include <string.h>
#include <math.h>
#include <iostream>
#include <stdexcept>
#include "FireLog.h"
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "jo_util.hpp"
#include "MatUtil.hpp"
#include "version.h"

using namespace cv;
using namespace std;
using namespace firesight;

bool Pipeline::morph(json_t *pStage, json_t *pStageModel, Model &model, String mop, const char* fmt) {
  validateImage(model.image);
  const char *errMsg = NULL;
  int morphOp = MORPH_OPEN;
  int kwidth = jo_int(pStage, "ksize.width", 3, model.argMap);
  kwidth = jo_int(pStage, "kwidth", kwidth, model.argMap);
  int kheight = jo_int(pStage, "ksize.height", 3, model.argMap);
  kheight = jo_int(pStage, "kheight", kheight, model.argMap);
  String shapeStr = jo_string(pStage, "shape", "MORPH_ELLIPSE", model.argMap);
  int shape = MORPH_ELLIPSE;

  if (!errMsg) {
    if (mop.compare("MORPH_ERODE") == 0) {
      morphOp = MORPH_ERODE;
    } else if (mop.compare("MORPH_DILATE") == 0) {
      morphOp = MORPH_DILATE;
    } else if (mop.compare("MORPH_OPEN") == 0) {
      morphOp = MORPH_OPEN;
    } else if (mop.compare("MORPH_CLOSE") == 0) {
      morphOp = MORPH_CLOSE;
    } else if (mop.compare("MORPH_GRADIENT") == 0) {
      morphOp = MORPH_GRADIENT;
    } else if (mop.compare("MORPH_TOPHAT") == 0) {
      morphOp = MORPH_TOPHAT;
    } else if (mop.compare("MORPH_BLACKHAT") == 0) {
      morphOp = MORPH_BLACKHAT;
    } else {
      errMsg = "Unknown morphology operation: ";
    }
  }

  if (!errMsg) {
    if (shapeStr.compare("MORPH_ELLIPSE") == 0) {
      shape = MORPH_ELLIPSE;
    } else if (shapeStr.compare("MORPH_CROSS") == 0) {
      shape = MORPH_CROSS;
    } else if (shapeStr.compare("MORPH_RECT") == 0) {
      shape = MORPH_RECT;
    } else {
      shape = jo_shape(pStage, "shape", errMsg);
    }
  }

  if (!errMsg) {
    Mat structuringElement = getStructuringElement(shape, Size(kwidth, kheight));
    switch (morphOp) {
      case MORPH_ERODE:
	erode(model.image, model.image, structuringElement);
	break;
      case MORPH_DILATE:
	dilate(model.image, model.image, structuringElement);
	break;
      default:
	morphologyEx(model.image, model.image, morphOp, structuringElement);
	break;
    }
  }

  return stageOK(fmt, errMsg, pStage, pStageModel);
}

bool Pipeline::apply_morph(json_t *pStage, json_t *pStageModel, Model &model) {
  String mop = jo_string(pStage, "mop", "MORPH_OPEN", model.argMap);
  return morph(pStage, pStageModel, model, mop, "apply_morph(%s) %s");
}

bool Pipeline::apply_erode(json_t *pStage, json_t *pStageModel, Model &model) {
  return morph(pStage, pStageModel, model, String("MORPH_ERODE"), "apply_erode(%s) %s");
}

bool Pipeline::apply_dilate(json_t *pStage, json_t *pStageModel, Model &model) {
  return morph(pStage, pStageModel, model, String("MORPH_DILATE"), "apply_dilate(%s) %s");
}

