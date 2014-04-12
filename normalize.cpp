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

bool Pipeline::apply_normalize(json_t *pStage, json_t *pStageModel, Model &model) {
  vector<float> domain = jo_vectorf(pStage, "domain", vector<float>(), model.argMap);
  vector<float> range = jo_vectorf(pStage, "range", vector<float>(), model.argMap);
  double alpha = 1;
  double beta = 0;
  string normTypeStr = jo_string(pStage, "normType", "NORM_L2", model.argMap);
  int normType;
  const char *errMsg = NULL;

  if (!errMsg) {
    if (model.image.depth() == CV_8U) {
      if (domain.size() == 0) {
	domain.push_back(0);
	domain.push_back(255);
      }
    }
    if (domain.size() > 0) {
      if (domain.size() != 2 || domain[0] >= domain[1]) {
	errMsg = "Expected domain interval with 2 inclusive values (mininum, maximum)";
      }
    } 
  }
  if (!errMsg && domain.size() > 0) {
    if (model.image.depth() == CV_8U) {
      if (domain[1] != 255) {
	threshold(model.image, model.image, domain[1], domain[1], THRESH_TRUNC);
      }
    } else {
      threshold(model.image, model.image, domain[1], domain[1], THRESH_TRUNC);
    }
    if (domain[0] != 0) {
      subtract(model.image, Scalar::all(domain[0]), model.image);
    }
  }

  if (!errMsg) {
    if (model.image.depth() == CV_8U) {
      if (range.size() == 0) {
	range.push_back(0);
	range.push_back(255);
      }
    }
    if (range.size() > 0) {
      if (range.size() != 2 || range[0] >= range[1]) {
	errMsg = "Expected range interval with 2 values (mininum, maximum)";
      }
    }
  }

  if (!errMsg) {
    if (normTypeStr.compare("NORM_L2") == 0) {
      normType = NORM_L2;
      if (model.image.depth() == CV_8U) {
	alpha = sqrt((double) model.image.cols * (double) model.image.rows * range[1] * range[1]);
      }
    } else if (normTypeStr.compare("NORM_L1") == 0) {
      normType = NORM_L1;
      if (model.image.depth() == CV_8U) {
	alpha = (double) model.image.cols * (double) model.image.rows * range[1];
      }
    } else if (normTypeStr.compare("NORM_INF") == 0) {
      normType = NORM_INF;
      if (model.image.depth() == CV_8U) {
	alpha = range[1];
      }
    } else if (normTypeStr.compare("NORM_MINMAX") == 0) {
      normType = NORM_MINMAX;
      if (model.image.depth() == CV_8U) {
	alpha = range[0];
	beta = range[1];
      }
    } else {
      errMsg = "Unknown normType";
    }
  }

  if (!errMsg && normType != NORM_MINMAX) {
    if (range[0] != 0) {
      errMsg = "Range minimum can only be non-zero for NORM_MINMAX";
    }
  }

  if (!errMsg) {
    alpha = jo_float(pStage, "alpha", alpha, model.argMap);
    beta = jo_float(pStage, "beta", beta, model.argMap);

    normalize(model.image, model.image, alpha, beta, normType);
  }

  return stageOK("apply_normalize(%s) %s", errMsg, pStage, pStageModel);
}

