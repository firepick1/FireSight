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


static const bool jo_bool(const json_t *pObj, const char *key, bool defaultValue=0) {
	json_t *pValue = json_object_get(pObj, key);
	return json_is_true(pValue);
}

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

static int jo_shape(json_t *pStage, const char *key, const char *&errMsg) {
	const char * pShape = jo_string(pStage, key, "MORPH_ELLIPSE");
	int shape = MORPH_ELLIPSE;

	if (strcmp("MORPH_ELLIPSE",pShape)==0) {
		shape = MORPH_ELLIPSE;
	} else if (strcmp("MORPH_RECT",pShape)==0) {
		shape = MORPH_RECT;
	} else if (strcmp("MORPH_CROSS",pShape)==0) {
		shape = MORPH_CROSS;
	} else {
		errMsg = "shape not supported";
	}
	return shape;
}

static bool stageOK(const char *fmt, const char *errMsg, json_t *pStage, json_t *pStageModel) {
	if (errMsg) {
		char *pStageJson = json_dumps(pStage, JSON_COMPACT|JSON_PRESERVE_ORDER);
		LOGERROR2(fmt, pStageJson, errMsg);
		free(pStageJson);
		return false;
	}

	if (logLevel >= FIRELOG_DEBUG) {
	  char *pStageJson = json_dumps(pStage, 0);
		char *pModelJson = json_dumps(pStageModel, 0);
		LOGDEBUG2(fmt, pStageJson, pModelJson);
		free(pStageJson);
		free(pModelJson);
	}

	return true;
}

static bool apply_imread(json_t *pStage, json_t *pStageModel, Mat &image) {
  const char *path = jo_string(pStage, "path", NULL);
	const char *errMsg = NULL;

	if (!path) {
		errMsg = "expected path for imread";
	} else {
		try {
			image = imread(path, CV_LOAD_IMAGE_COLOR);
			json_object_set(pStageModel, "rows", json_integer(image.rows));
			json_object_set(pStageModel, "cols", json_integer(image.cols));
		} catch (runtime_error &ex) {
		  errMsg = ex.what();
		}
	}

	return stageOK("apply_imread(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_imwrite(json_t *pStage, json_t *pStageModel, Mat image) {
  const char *path = jo_string(pStage, "path", NULL);
	const char *errMsg = NULL;

	if (!path) {
		errMsg = "expected path for imwrite";
	} else {
		try {
			bool result = imwrite(path, image);
			json_object_set(pStageModel, "result", json_boolean(result));
		} catch (runtime_error &ex) {
		  errMsg = ex.what();
		}
	}

	return stageOK("apply_imwrite(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_cvtColor(json_t *pStage, json_t *pStageModel, Mat &image) {
  const char *codeStr = jo_string(pStage, "code", "CV_BGR2GRAY");
	int dstCn = jo_int(pStage, "dstCn", 0);
	const char *errMsg = NULL;
	int code = CV_BGR2GRAY;

	if (strcmp("CV_RGB2GRAY",codeStr)==0) {
	  code = CV_RGB2GRAY;
	} else if (strcmp("CV_BGR2GRAY",codeStr)==0) {
	  code = CV_BGR2GRAY;
	} else if (strcmp("CV_GRAY2BGR",codeStr)==0) {
	  code = CV_GRAY2BGR;
	} else if (strcmp("CV_GRAY2RGB",codeStr)==0) {
	  code = CV_GRAY2RGB;
	} else {
	  errMsg = "code unsupported";
	}
	if (dstCn < 0) {
		errMsg = "expected 0<dstCn";
	}

	if (!errMsg) {
		cvtColor(image, image, code, dstCn);
	}

	return stageOK("apply_cvtColor(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_dilate(json_t *pStage, json_t *pStageModel, Mat &image) {
	const char *errMsg = NULL;
	int kwidth = jo_int(pStage, "ksize.width", 3);
	int kheight = jo_int(pStage, "ksize.height", 3);
	int shape = jo_shape(pStage, "shape", errMsg);

	if (!errMsg) {
	  Mat structuringElement = getStructuringElement(shape, Size(kwidth, kheight));
		dilate(image, image, structuringElement);
	}

	return stageOK("apply_dilate(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_erode(json_t *pStage, json_t *pStageModel, Mat &image) {
	const char *errMsg = NULL;
	int kwidth = jo_int(pStage, "ksize.width", 3);
	int kheight = jo_int(pStage, "ksize.height", 3);
	int shape = jo_shape(pStage, "shape", errMsg);

	if (!errMsg) {
	  Mat structuringElement = getStructuringElement(shape, Size(kwidth, kheight));
		erode(image, image, structuringElement);
	}

	return stageOK("apply_erode(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_blur(json_t *pStage, json_t *pStageModel, Mat &image) {
	const char *errMsg = NULL;
	int width = jo_int(pStage, "ksize.width", 3);
	int height = jo_int(pStage, "ksize.height", 3);
	int anchorx = jo_int(pStage, "anchor.x", -1);
	int anchory = jo_int(pStage, "anchor.y", -1);

	if (width <= 0 || height <= 0) {
		errMsg = "expected 0<width and 0<height";
	}

	if (!errMsg) {
		blur(image, image, Size(width,height));
	}

	return stageOK("apply_blur(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_Canny(json_t *pStage, json_t *pStageModel, Mat &image) {
	double threshold1 = jo_double(pStage, "threshold1", 0);
	double threshold2 = jo_double(pStage, "threshold2", 50);
	double apertureSize = jo_double(pStage, "apertureSize", 3);
	bool L2gradient = jo_bool(pStage, "L2gradient", false);
	const char *errMsg = NULL;

	if (!errMsg) {
		Canny(image, image, threshold1, threshold2, apertureSize, L2gradient);
	}

	return stageOK("apply_imread(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_HoleRecognizer(json_t *pStage, json_t *pStageModel, Mat image) {
	double diamMin = jo_double(pStage, "diamMin");
	double diamMax = jo_double(pStage, "diamMax");
	int showMatches = jo_int(pStage, "show", 0);
	const char *errMsg = NULL;

	if (diamMin <= 0 || diamMax <= 0 || diamMin > diamMax) {
		errMsg = "expected: 0 < diamMin < diamMax ";
	} else if (showMatches < 0) {
		errMsg = "expected: 0 < showMatches ";
	} else if (logLevel >= FIRELOG_TRACE) {
		char *pStageJson = json_dumps(pStage, 0);
		LOGTRACE1("apply_HoleRecognizer(%s)", pStageJson);
		free(pStageJson);
	}
	if (!errMsg) {
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

	return stageOK("apply_imread(%s) %s", errMsg, pStage, pStageModel);
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
			if (strcmp(pOp, "blur")==0) {
				apply_blur(pStage, pStageModel, workingImage);
			} else if (strcmp(pOp, "Canny")==0) {
				apply_Canny(pStage, pStageModel, workingImage);
			} else if (strcmp(pOp, "cvtColor")==0) {
				apply_cvtColor(pStage, pStageModel, workingImage);
			} else if (strcmp(pOp, "dilate")==0) {
				apply_dilate(pStage, pStageModel, workingImage);
			} else if (strcmp(pOp, "erode")==0) {
				apply_erode(pStage, pStageModel, workingImage);
			} else if (strcmp(pOp, "HoleRecognizer")==0) {
				apply_HoleRecognizer(pStage, pStageModel, workingImage);
			} else if (strcmp(pOp, "imread")==0) {
				apply_imread(pStage, pStageModel, workingImage);
			} else if (strcmp(pOp, "imwrite")==0) {
				apply_imwrite(pStage, pStageModel, workingImage);
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


