
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

static Mat apply_imread(int index, json_t *pObj) {
  const char *path = jo_string(pObj, "path", NULL);
	const char *errFmt = NULL;
	const char *errMsg = NULL;
	Mat matRGB;

	if (!path) {
		errFmt = "%d. apply_imread(%s) expected \"path\" for image file to read";
	} else {
		try {
			matRGB = imread(path, CV_LOAD_IMAGE_COLOR);
			LOGTRACE2("%d. apply_imread(%s)", index, path);
		} catch (runtime_error &ex) {
		  errMsg = ex.what();
			errFmt = "%d. apply_imread(%s) exception: %s";
		}
	}

	if (errFmt) {
		char *pJson = json_dumps(pObj, 0);
		LOGERROR3(errFmt, index, pJson, errMsg);
		free(pJson);
	}
	return matRGB;
}

static void apply_imwrite(int index, json_t *pObj, Mat image) {
  const char *path = jo_string(pObj, "path", NULL);
	const char *errFmt = NULL;
	const char *errMsg = NULL;
	Mat matRGB;

	if (!path) {
		errFmt = "%d. apply_imwrite(%s) expected \"path\" for image file to write";
	} else {
		try {
			matRGB = imread(path, CV_LOAD_IMAGE_COLOR);
			imwrite(path, image);
			LOGTRACE2("%d. apply_imwrite(%s)", index, path);
		} catch (runtime_error &ex) {
		  errMsg = ex.what();
			errFmt = "%d. apply_imwrite(%s) exception: %s";
		}
	}

	if (errFmt) {
		char *pJson = json_dumps(pObj, 0);
		LOGERROR3(errFmt, index, pJson, errMsg);
		free(pJson);
	}
}

static void apply_HoleRecognizer(int index, json_t *pObj, Mat image) {
	double diamMin = jo_double(pObj, "diamMin");
	double diamMax = jo_double(pObj, "diamMax");
	int showMatches = jo_int(pObj, "show", 0);
	const char *errFmt = NULL;

	if (diamMin <= 0 || diamMax <= 0 || diamMin > diamMax) {
		errFmt = "%d. apply_HoleRecognizer(%s) expected: 0 < diamMin < diamMax ";
	} else if (showMatches < 0) {
		errFmt = "%d. apply_HoleRecognizer(%s) expected: 0 < showMatches ";
	} else if (logLevel >= FIRELOG_TRACE) {
		char *pJson = json_dumps(pObj, 0);
		LOGTRACE2("%d. apply_HoleRecognizer(%s)", index, pJson);
		free(pJson);
	}
	if (!errFmt) {
		vector<MatchedRegion> matches;
		HoleRecognizer recognizer(diamMin, diamMax);
		recognizer.showMatches(showMatches);
		recognizer.scan(image, matches);
		for (int i = 0; i < matches.size(); i++) {
			cout << "match " << i+1 << " = " << matches[i].asJson() << endl;
		}
	}

	if (errFmt) {
		char *pJson = json_dumps(pObj, 0);
		LOGERROR2(errFmt, index, pJson);
		free(pJson);
	}
}

Analyzer::Analyzer(int perceptionDepth) {
  this->perceptionDepth = perceptionDepth;
}

void Analyzer::process(const char* json, int time) {
	json_error_t jerr;
	json_t *pNode = json_loads(json, 0, &jerr);

	if (!pNode) {
		LOGERROR3("Analyzer::process cannot parse json: %s src:%s line:%d", jerr.text, jerr.source, jerr.line);
		return;
	} else if (!json_is_array(pNode)) {
		LOGERROR1("Analyzer::process expected json array: %s", json);
		return;
	}
	size_t index;
	json_t *pObj;
	LOGTRACE2("Analyzer::process(%d functions, %d)", json_array_size(pNode), time);
	json_array_foreach(pNode, index, pObj) {
		const char *errFmt = NULL;
		const char *pApply = jo_string(pObj, "apply", "");
		if (pApply) {
			if (strcmp(pApply, "imread")==0) {
				this->workingImage = apply_imread(index+1, pObj);
			} else if (strcmp(pApply, "imwrite")==0) {
				apply_imwrite(index+1, pObj, this->workingImage);
			} else if (strcmp(pApply, "HoleRecognizer")==0) {
				apply_HoleRecognizer(index+1, pObj, this->workingImage);
			} else {
			  errFmt = "%d. Analyzer::process unknown value provided for \"apply\" key in %s";
			}
		} else {
			errFmt = "%d. Analyzer::process no value provided for \"apply\" key in %s";
		} //if (pApply)
		if (errFmt) {
			char *pJson = json_dumps(pObj, 0);
			LOGERROR2(errFmt, index, pJson);
			free(pJson);
		}
	} // json_array_foreach
	free(pNode);
}


