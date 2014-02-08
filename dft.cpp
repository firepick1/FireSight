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

static void dftMirror(Mat &image) {
	int cx = image.cols/2;
	Mat imageL(image,Rect(0,0,cx,image.rows));
	Mat imageR(image,Rect(cx,0,cx,image.rows));
	flip(imageR, imageL, 1);
}

static void dftShift(Mat &image, const char *&errMsg) {
	if ((image.cols & 1) || (image.rows&1)) {
		LOGTRACE("Cropping image to even number of rows and columns");
		image = image(Rect(0, 0, image.cols & -2, image.rows & -2)); 
	}
	int cx = image.cols/2;
	int cy = image.rows/2;
	Mat q1(image, Rect(0,0,cx,cy));
	Mat q2(image, Rect(cx,0,cx,cy));
	Mat q3(image, Rect(0,cy,cx,cy));
	Mat q4(image, Rect(cx,cy,cx,cy));

	Mat tmp;

	q1.copyTo(tmp);
	q4.copyTo(q1);
	tmp.copyTo(q4);

	q2.copyTo(tmp);
	q3.copyTo(q2);
	tmp.copyTo(q3);
}

bool Pipeline::apply_dftSpectrum(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
	int delta = jo_int(pStage, "delta", 1);
  bool isShift = jo_bool(pStage, "shift", true);	
  bool isLog = jo_bool(pStage, "log", true);	
	bool isMagnitude = false;
	bool isPhase = false;
	bool isReal = false;
	bool isImaginary = false;
	bool isMirror = jo_bool(pStage, "mirror", true);
	const char * showStr = jo_string(pStage, "show", "magnitude");
	const char *errMsg = NULL;

	assert(0<image.rows && 0<image.cols);

	if (!errMsg) {
		if (strcmp("magnitude", showStr) == 0) {
			isMagnitude = true;
		} else if (strcmp("phase", showStr) == 0) {
			isPhase = true;
		} else if (strcmp("real", showStr) == 0) {
			isReal = true;
		} else if (strcmp("imaginary", showStr) == 0) {
			isImaginary = true;
		} else {
			errMsg = "Expected 'magnitude' or 'phase' for show";
		}
	}

	if (!errMsg) {
		if (isReal) {
			if (image.channels() != 1) {
				errMsg = "Expected real (1-channel) Mat";
			}
		} else {
			if (image.channels() != 2) {
				errMsg = "Expected complex (2-channel) Mat";
			}
		}
	}

	if (!errMsg) {
		if (image.channels() > 1) {
			Mat planes[] = {
				Mat::zeros(image.size(), CV_32F),
				Mat::zeros(image.size(), CV_32F)
			};
			split(image, planes);
			if (isMagnitude) {
				magnitude(planes[0], planes[1], image);
			} else if (isPhase) {
				phase(planes[0], planes[1], image);
			} else if (isReal) {
				image = planes[0];
			} else if (isImaginary) {
				image = planes[1];
			}
		}
		if (delta) {
			image += Scalar::all(delta);
		}
		if (isLog) {
			log(image, image);
		}
		if (isShift) {
			dftShift(image, errMsg);
		}
		if (isMirror) {
			dftMirror(image);
		}
	}
	return stageOK("apply_dftSpectrum(%s) %s", errMsg, pStage, pStageModel);
}

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

	assert(0<image.rows && 0<image.cols);

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

