#include "calcHist.h"

using namespace cv;
using namespace std;


namespace firesight {

bool CalcHist::apply_internal(json_t *pStageModel, Model &model) {
  validateImage(model.image);

  int histSize[] = {bins,bins,bins,bins};
  bool split = !accumulate && nChannels > 1 && histChannels.size() > 1;
  bool uniform = true;
  const char *errMsg = NULL;
  Mat mask;

  if (rangeMin > rangeMax) {
    errMsg = "Expected rangeMin <= rangeMax";
  } else if (bins < 1 && bins <= (rangeMax-rangeMin)) {
    errMsg = "Expected 0<bins and bins<=(rangeMax-rangeMin)";
  }
  if (dims != 1) {
    errMsg = "The only supported value for dims is 1";
  }
  if (nChannels > 4) {
    errMsg = "Expected at most 4 image channels";
  }
  if (histChannels.size() > 4) {
    errMsg = "Expected at most 4 histogram channels";
  }

  if (!errMsg) {
    float rangeC0[] = { rangeMin, rangeMax }; 
    float rangeC1[] = { rangeMin, rangeMax }; 
    float rangeC2[] = { rangeMin, rangeMax }; 
    float rangeC3[] = { rangeMin, rangeMax }; 
    const float* ranges[] = { rangeC0, rangeC1, rangeC2, rangeC3 };
    Mat hist[] = {Mat(),Mat(),Mat(),Mat()};
    int channels[4];
    for (int i = 0; i < 4; i++) {
      channels[i] =  i < histChannels.size() ? histChannels[i] : i;
    }
    if (split) {
      for (int channel=0; channel<histChannels.size(); channel++) {
	calcHist(&model.image, 1, &channels[channel], mask, hist[channel], 1, histSize, ranges, uniform, false);
      }
    } else {
      for (int channel=0; channel<histChannels.size(); channel++) {
	calcHist(&model.image, 1, &channels[channel], mask, hist[0], 1, histSize, ranges, uniform, true);
      }
    }
    json_t *pHist = json_object();
    if (split) {
      for (int i = 0; i < bins; i++) {
	json_t *pDimHist = json_array();
	bool keep = false;
	for (int channel = 0; channel < histChannels.size(); channel++) {
	  float binValue = hist[channel].at<float>(i);
	  keep = keep || binMin <= binValue && (binMax==0 || binValue < binMax);
	  json_t *pNum = (int)binValue == binValue ? json_integer(binValue) : json_real(binValue);
	  json_array_append(pDimHist, pNum);
	}
	if (keep) {
	  char numBuf[20];
	  snprintf(numBuf, sizeof(numBuf), "%d", i);
	  json_object_set(pHist, numBuf, pDimHist);
	} else {
	  json_decref(pDimHist);
	}
      }
    } else {
      for (int i = 0; i < bins; i++) {
	json_t *pDimHist = json_array();
	float binValue = hist[0].at<float>(i);
	if ((binValue == 0 || binMax && binValue > binMax)) {
	  // ignore value
	} else {
	  char numBuf[20];
	  snprintf(numBuf, sizeof(numBuf), "%d", i);
	  json_t *pNum = (int)binValue == binValue ? json_integer(binValue) : json_real(binValue);
	  json_object_set(pHist, numBuf, pNum);
	}
      }
    }
    json_object_set(pStageModel, "hist", pHist);
    if (locations) {
      json_t *pLocations = json_array();
      json_object_set(pStageModel, "locations", pLocations);
    }
  }

  return stageOK("apply_calcHist(%s) %s", errMsg, pStage, pStageModel);
}

}
