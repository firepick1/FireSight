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
#include "jo_util.hpp"

using namespace cv;
using namespace std;
using namespace FireSight;

bool Pipeline::apply_dft(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
	const char *errMsg = NULL;
	char errBuf[150];
	json_t *pFlags = json_object_get(pStage, "flags");
	int flags = 0;

	if (json_is_array(pFlags)) {
		int index;
		json_t *pStr;
		json_array_foreach(pFlags, index, pStr) {
			const char *flag = json_string_value(pStr);
			if (!flag) {
				errMsg = "Expected array of flag name strings";
				break;
			}
			if (strcmp(flag, "DFT_COMPLEX_OUTPUT") == 0) {
				flags |= DFT_COMPLEX_OUTPUT;
			} else if (strcmp(flag, "DFT_REAL_OUTPUT") == 0) {
				flags |= DFT_REAL_OUTPUT;
			} else if (strcmp(flag, "DFT_SCALE") == 0) {
				flags |= DFT_SCALE;
			} else if (strcmp(flag, "DFT_INVERSE") == 0) {
				flags |= DFT_INVERSE;
			} else if (strcmp(flag, "DFT_ROWS") == 0) {
				flags |= DFT_ROWS;
			} else {
				sprintf(errBuf, "Unknown flag %s", flag);
				errMsg = errBuf;
			}
		}
	}

	if (!errMsg) {
		if (image.type() != CV_32F) {
			Mat fImage;
			LOGTRACE("apply_dft(): Convert image to CV_32F");
			image.convertTo(fImage, CV_32F);
			image = fImage;
		}
		Mat dftImage;
		LOGTRACE1("apply_dft() flags:%d", flags);
		dft(image, dftImage, flags);
		image = dftImage;
		if (flags & DFT_INVERSE) {
			Mat invImage;
			LOGTRACE("apply_dft(): Convert image to CV_8U");
			image.convertTo(invImage, CV_8U);
			image = invImage;
		}
	}

	return stageOK("apply_dft(%s) %s", errMsg, pStage, pStageModel);
}

