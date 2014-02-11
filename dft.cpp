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
#include "MatUtil.hpp"

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

bool Pipeline::apply_matchTemplate(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
  const char * methodStr = jo_string(pStage, "method", "CV_TM_CCORR");
  const char *tmpltPath = jo_string(pStage, "template", NULL);
	float rangeMin = jo_double(pStage, "rangeMin", 253);
	float rangeMax = jo_double(pStage, "rangeMax", 256);
	int bins = rangeMax-rangeMin;
	float angle = jo_double(pStage, "angle", 0);
  const char *errMsg = NULL;
	int method;
	Mat tmplt;

	assert(0<image.rows && 0<image.cols);

	if (!tmpltPath) {
		errMsg = "Expected template path for imread";
	} else {
		if (image.channels() == 1) {
			tmplt = imread(tmpltPath, CV_LOAD_IMAGE_GRAYSCALE);
		} else {
			tmplt = imread(tmpltPath, CV_LOAD_IMAGE_COLOR);
		}
		if (tmplt.data) {
			LOGTRACE2("apply_matchTemplate(%s) %s", tmpltPath, matInfo(tmplt).c_str());
			if (image.rows<tmplt.rows || image.cols<tmplt.cols) {
				errMsg = "Expected template smaller than image to match";
			}
		} else {
			errMsg = "imread failed";
		}
	}

	if (!errMsg) {
		if (strcmp(methodStr, "CV_TM_SQDIFF_NORMED")==0) {
			method = CV_TM_SQDIFF_NORMED;
		} else if (strcmp(methodStr, "CV_TM_CCORR")==0) {
			method = CV_TM_CCORR;
		} else if (strcmp(methodStr, "CV_TM_CCORR_NORMED")==0) {
			method = CV_TM_CCORR_NORMED;
		} else if (strcmp(methodStr, "CV_TM_CCOEFF")==0) {
			method = CV_TM_CCOEFF;
		} else if (strcmp(methodStr, "CV_TM_CCOEFF_NORMED")==0) {
			method = CV_TM_CCOEFF_NORMED;
		} else {
			errMsg = "Expected method name";
		}
	} 

	if (!errMsg) {
		if (rangeMin > rangeMax) {
			errMsg = "Expected rangeMin <= rangeMax";
		} else if (bins < 2 || bins > 256 ) {
			errMsg = "Expected 1<bins and bins<=256";
		}
	}

	if (!errMsg) {
		Mat result;
		matchTemplate(image, tmplt, result, method);
		image = result;
		int histSize = bins;
		bool uniform = true;
		bool accumulate = false;
		const char *errMsg = NULL;
		Mat mask;
		float rangeC0[] = { rangeMin, rangeMax }; 
		const float* ranges[] = { rangeC0 };
		Mat hist;
		normalize(result, result, 0, 255, NORM_MINMAX);
		calcHist(&result, 1, 0, mask, hist, 1, &histSize, ranges, uniform, accumulate);
		json_t *pHist = json_array();
		assert(result.channels() == 1);
		vector<RotatedRect> rects;
		int rDelta = tmplt.rows/2;
		int cDelta = tmplt.cols/2;
		for (int r=0; r < result.rows; r++) {
			for (int c=0; c < result.cols; c++) {
				float val = result.at<float>(r,c);
				bool isOverlap = false;
				if (rangeMin <= val && val <= rangeMax) {
					for (int irect=0; irect<rects.size(); irect++) {
						int cx = rects[irect].center.x;
						int cy = rects[irect].center.y;
						if (cx-cDelta < c && c < cx+cDelta && cy-rDelta < r && r < cy+rDelta) {
							isOverlap = true;
							if (val > result.at<float>(cy, cx)) {
								rects[irect].center.x = c;
								rects[irect].center.y = r;
							}
							break;
						}
					}
					if (!isOverlap) {
						rects.push_back(RotatedRect(Point(c,r), Size(tmplt.cols, tmplt.rows), angle));
					}
				}
			}
		}
		json_t *pRects = json_array();
		json_object_set(pStageModel, "rects", pRects);
		for (int irect=0; irect<rects.size(); irect++) {
			int cx = rects[irect].center.x;
			int cy = rects[irect].center.y;
			float val = result.at<float>(cy,cx);
			cout << val << "[" << cx << "," << cy << "]" << endl;
			json_t *pRect = json_object();
			json_object_set(pRect, "x", json_real(cx));
			json_object_set(pRect, "y", json_real(cy));
			json_object_set(pRect, "width", json_real(tmplt.cols));
			json_object_set(pRect, "height", json_real(tmplt.rows));
			json_object_set(pRect, "angle", json_real(angle));
			json_object_set(pRect, "corr", json_real(val));
			json_array_append(pRects, pRect);
		}
	}

	return stageOK("apply_matchTemplate(%s) %s", errMsg, pStage, pStageModel);
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
	const char *depthStr = jo_string(pStage, "depth", "CV_8U");

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
		if (flags & DFT_INVERSE && strcmp("CV_8U",depthStr)==0) {
			Mat invImage;
			LOGTRACE("apply_dft(): Convert image to CV_8U");
			image.convertTo(invImage, CV_8U);
			image = invImage;
		}
	}

	return stageOK("apply_dft(%s) %s", errMsg, pStage, pStageModel);
}

