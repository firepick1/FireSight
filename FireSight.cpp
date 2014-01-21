
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


static const double jo_double(const json_t *pObj, const char *key, double defaultValue=0) {
	json_t *pValue = json_object_get(pObj, key);
	return json_is_real(pValue) ? json_real_value(pValue) : defaultValue;
}

static const char * jo_string(const json_t *pObj, const char *key, const char *defaultValue = NULL) {
	json_t *pValue = json_object_get(pObj, key);
	return json_is_string(pValue) ? json_string_value(pValue) : defaultValue;
}

static Mat apply_imread(const char *path) {
	LOGTRACE1("apply_imread(%s)", path);
	Mat matRGB = imread(path, CV_LOAD_IMAGE_COLOR);
	return matRGB;
}

static void apply_imwrite(const char *path, const Mat image) {
	LOGTRACE1("apply_imwrite(%s)", path);
	imwrite(path, image);
}

static void apply_HoleRecognizer(double diamMin, double diamMax) {
	char args[64];
	sprintf(args, "%f,%f", diamMin, diamMax);
	if (diamMin <= 0 || diamMax <= 0 || diamMin > diamMax) {
		LOGERROR1("apply_HoleRecognizer(%s) expected: 0 < diamMin < diamMax ", args);
	}
	LOGTRACE1("apply_HoleRecognizer(%s)", args);
}

Analyzer::Analyzer(int perceptionDepth) {
  this->perceptionDepth = perceptionDepth;
}

void Analyzer::process(const char* json, int time) {
	json_error_t jerr;
	json_t *pNode = json_loads(json, 0, &jerr);
	LOGTRACE2("Analyzer::process(%s,%d)", json, time);
	if (!pNode) {
		LOGERROR3("Analyzer::process %s src:%s line:%d", jerr.text, jerr.source, jerr.line);
		return;
	} else if (!json_is_array(pNode)) {
		LOGERROR1("Analyzer::process expected json array: %s", json);
		return;
	}
	size_t index;
	json_t *pObj;
	LOGTRACE1("Analyzer::process %d functions", json_array_size(pNode));
	Mat image;
	json_array_foreach(pNode, index, pObj) {
		const char *pApply = jo_string(pObj, "apply", "");
		if (pApply) {
			if (strcmp(pApply, "imread")==0) {
				image = apply_imread(jo_string(pObj, "path"));
			} else if (strcmp(pApply, "imwrite")==0) {
				apply_imwrite(jo_string(pObj, "path"), image);
			} else if (strcmp(pApply, "HoleRecognizer")==0) {
				apply_HoleRecognizer(jo_double(pObj, "diamMin"), jo_double(pObj, "diamMax"));
			} else {
				char *s = json_dumps(pObj, 0);
			  LOGERROR1("Analyzer::process unknown value provided for \"apply\" key in %s", s);
				free(s);
			}
		} else {
			char *s = json_dumps(pObj, 0);
			LOGERROR1("Analyzer::process no value provided for \"apply\" key in %s", s);
			free(s);
		} //if (pApply)
	} // json_array_foreach
	free(pNode);
}


