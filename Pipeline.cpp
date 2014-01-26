#include <string.h>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <boost/format.hpp>
#include "FireLog.h"
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"

using namespace cv;
using namespace std;
using namespace FireSight;


static const int jo_int(const json_t *pObj, const char *key, int defaultValue=0) {
	json_t *pValue = json_object_get(pObj, key);
	return json_is_integer(pValue) ? json_integer_value(pValue) : defaultValue;
}

static const double jo_double(const json_t *pObj, const char *key, double defaultValue=0) {
	json_t *pValue = json_object_get(pObj, key);
	return json_is_real(pValue) ? json_real_value(pValue) : defaultValue;
}

static const char * jo_string(const json_t *pObj, const char *key, const char *defaultValue = NULL) {
	json_t *pValue = json_object_get(pObj, key);
	return json_is_string(pValue) ? json_string_value(pValue) : defaultValue;
}

static bool stageOK(const char *errFmt, json_t *pStage, const char *errMsg) {
	if (errFmt) {
		char *pStageJson = json_dumps(pStage, JSON_COMPACT|JSON_PRESERVE_ORDER);
		LOGERROR2(errFmt, pStageJson, errMsg);
		free(pStageJson);
		return false;
	}

	return true;
}

static Mat apply_imread(json_t *pStage, json_t *pStageModel) {
  const char *path = jo_string(pStage, "path", NULL);
	const char *errFmt = NULL;
	const char *errMsg = NULL;
	Mat matRGB;

	if (!path) {
		errFmt = "apply_imread(%s) expected \"path\" for image file to read";
	} else {
		try {
			LOGTRACE1("apply_imread(%s)", path);
			matRGB = imread(path, CV_LOAD_IMAGE_COLOR);
			json_object_set(pStageModel, "rows", json_integer(matRGB.rows));
			json_object_set(pStageModel, "cols", json_integer(matRGB.cols));
		} catch (runtime_error &ex) {
		  errMsg = ex.what();
			errFmt = "apply_imread(%s) exception: %s";
		}
	}

	if (stageOK(errFmt, pStage, errMsg) && logLevel >= FIRELOG_DEBUG) {
		char *pModelJson = json_dumps(pStageModel, 0);
		LOGDEBUG2("apply_imread(%s) => %s", path, pModelJson);
		free(pModelJson);
	}
	return matRGB;
}

static void apply_imwrite(json_t *pStage, json_t *pStageModel, Mat image) {
  const char *path = jo_string(pStage, "path", NULL);
	const char *errFmt = NULL;
	const char *errMsg = NULL;
	Mat matRGB;

	if (!path) {
		errFmt = "apply_imwrite(%s) expected \"path\" for image file to write";
	} else {
		try {
			LOGTRACE1("apply_imwrite(%s)", path);
			bool result = imwrite(path, image);
			json_object_set(pStageModel, "result", json_boolean(result));
		} catch (runtime_error &ex) {
		  errMsg = ex.what();
			errFmt = "apply_imwrite(%s) exception: %s";
		}
	}

	if (stageOK(errFmt, pStage, errMsg) && logLevel >= FIRELOG_DEBUG) {
		char *pModelJson = json_dumps(pStageModel, 0);
		LOGDEBUG2("apply_imwrite(%s) => %s", path, pModelJson);
		free(pModelJson);
	}
}

static void apply_cvtColor(json_t *pStage, json_t *pStageModel, Mat &image) {
  const char *codeStr = jo_string(pStage, "code", "CV_BGR2GRAY");
	int dstCn = jo_int(pStage, "dstCn", 0);
	const char *errFmt = NULL;
	const char *errMsg = NULL;
	int code = CV_BGR2GRAY;
	Mat matOut;

	if (strcmp("CV_RGB2GRAY",codeStr)==0) {
	  code = CV_RGB2GRAY;
	} else if (strcmp("CV_BGR2GRAY",codeStr)==0) {
	  code = CV_BGR2GRAY;
	} else if (strcmp("CV_GRAY2BGR",codeStr)==0) {
	  code = CV_GRAY2BGR;
	} else if (strcmp("CV_GRAY2RGB",codeStr)==0) {
	  code = CV_GRAY2RGB;
	} else {
	  errFmt = "apply_cvtColor(%s) code unsupported";
	}
	if (dstCn < 0) {
		errFmt = "apply_cvtColor(%s) dstCn < 0";
	}

	if (!errFmt) {
		cvtColor(image, image, code, dstCn);
	}

	if (stageOK(errFmt, pStage, errMsg) && logLevel >= FIRELOG_DEBUG) {
		char *pStageJson = json_dumps(pStage, 0);
		LOGDEBUG1("apply_cvtColor(%s)", pStageJson);
		free(pStageJson);
	}
}

static void apply_HoleRecognizer(json_t *pStage, json_t *pStageModel, Mat image) {
	double diamMin = jo_double(pStage, "diamMin");
	double diamMax = jo_double(pStage, "diamMax");
	int showMatches = jo_int(pStage, "show", 0);
	const char *errFmt = NULL;
	const char *errMsg = NULL;

	if (diamMin <= 0 || diamMax <= 0 || diamMin > diamMax) {
		errFmt = "apply_HoleRecognizer(%s) expected: 0 < diamMin < diamMax ";
	} else if (showMatches < 0) {
		errFmt = "apply_HoleRecognizer(%s) expected: 0 < showMatches ";
	} else if (logLevel >= FIRELOG_TRACE) {
		char *pStageJson = json_dumps(pStage, 0);
		LOGTRACE1("apply_HoleRecognizer(%s)", pStageJson);
		free(pStageJson);
	}
	if (!errFmt) {
		vector<MatchedRegion> matches;
		HoleRecognizer recognizer(diamMin, diamMax);
		recognizer.showMatches(showMatches);
		recognizer.scan(image, matches);
		json_t *holes = json_array();
		json_object_set(pStageModel, "holes", holes);
		for (int i = 0; i < matches.size(); i++) {
			json_array_append(holes, matches[i].as_json_t());
		}
	}

	if (stageOK(errFmt, pStage, errMsg) && logLevel >= FIRELOG_DEBUG) {
	  char *pStageJson = json_dumps(pStage, 0);
		char *pModelJson = json_dumps(pStageModel, 0);
		LOGDEBUG2("apply_HoleRecognizer(%s) => %s", pStageJson, pModelJson);
		free(pStageJson);
		free(pModelJson);
	}
}

Pipeline::Pipeline(const char *pJson) {
	json_error_t jerr;
	pPipeline = json_loads(pJson, 0, &jerr);

	if (!pPipeline) {
		LOGERROR3("Pipeline::process cannot parse json: %s src:%s line:%d", jerr.text, jerr.source, jerr.line);
		throw jerr;
	}
}

Pipeline::Pipeline(json_t *pJson) {
  pPipeline = json_incref(pJson);
}

Pipeline::~Pipeline() {
	json_decref(pPipeline);
}

json_t *Pipeline::process(Mat &workingImage) { 
	if (!json_is_array(pPipeline)) {
		const char * errMsg = "Pipeline::process expected json array for pipeline definition";
		LOGERROR1(errMsg, "");
		return json_string(errMsg);
	}

	size_t index;
	json_t *pStage;
	json_t *pModel = json_object();
	char nameBuf[16];
	LOGTRACE1("Pipeline::process(%d functions)", json_array_size(pPipeline));
	json_array_foreach(pPipeline, index, pStage) {
		const char *errFmt = NULL;
		const char *pOp = jo_string(pStage, "op", "");
		sprintf(nameBuf, "s%d", index+1);
		const char *pName = jo_string(pStage, "name", nameBuf);
		json_t *pStageModel = json_object();
		json_object_set(pModel, pName, pStageModel);
		if (pOp) {
			if (strcmp(pOp, "imread")==0) {
				workingImage = apply_imread(pStage, pStageModel);
			} else if (strcmp(pOp, "imwrite")==0) {
				apply_imwrite(pStage, pStageModel, workingImage);
			} else if (strcmp(pOp, "cvtColor")==0) {
				apply_cvtColor(pStage, pStageModel, workingImage);
			} else if (strcmp(pOp, "HoleRecognizer")==0) {
				apply_HoleRecognizer(pStage, pStageModel, workingImage);
			} else {
			  errFmt = "%s. Pipeline::process unknown value provided for \"op\" key in %s";
			}
		} else {
			errFmt = "%s. Pipeline::process missing value for \"op\" in %s";
		} //if (pOp)
		if (errFmt) {
			char *pJson = json_dumps(pStage, 0);
			LOGERROR2(errFmt, pName, pJson);
			free(pJson);
		}
	} // json_array_foreach

	return pModel;
}


