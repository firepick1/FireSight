
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
	json_t *pValue;
	json_array_foreach(pNode, index, pValue) {
		char *s = json_dumps(pValue, 0);
		LOGTRACE1("Analyzer::process %s", s);
		free(s);
	}
	free(pNode);
}

