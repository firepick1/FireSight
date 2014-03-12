#include <string.h>
#include <math.h>
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
using namespace firesight;

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

static void modelMatches(Point offset, const Mat &tmplt, const Mat &result, const vector<float> &angles, 
	const vector<Point> &matches, json_t *pStageModel, float maxVal, bool isMin) 
{
	LOGTRACE1("modelMatches(%d)", (int)matches.size());
	json_t *pRects = json_array();
	assert(pRects);
	for (size_t iMatch=0; iMatch<matches.size(); iMatch++) {
		int cx = matches[iMatch].x;
		int cy = matches[iMatch].y;
		LOGTRACE2("modelMatches() matches(%d,%d)", cx, cy);
		float val = result.at<float>(cy,cx);
		json_t *pRect = json_object();
		assert(pRect);
		json_object_set(pRect, "x", json_real(cx+offset.x));
		json_object_set(pRect, "y", json_real(cy+offset.y));
		json_object_set(pRect, "width", json_real(tmplt.cols));
		json_object_set(pRect, "height", json_real(tmplt.rows));
		if (angles.size() == 1) {
			json_object_set(pRect, "angle", json_real(-angles[0]));
		}
		json_object_set(pRect, "corr", json_real(val/maxVal));
		json_array_append(pRects, pRect);
	}
	json_object_set(pStageModel, "rects", pRects);
	json_object_set(pStageModel, "maxVal", json_real(maxVal));
	json_object_set(pStageModel, "matches", json_integer(matches.size()));
	LOGTRACE("modelMatches() end");
}

bool Pipeline::apply_matchTemplate(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
  string methodStr = jo_string(pStage, "method", "CV_TM_CCOEFF_NORMED", model.argMap);
  string tmpltPath = jo_string(pStage, "template", "", model.argMap);
	float threshold = jo_float(pStage, "threshold", 0.7f);
	float corr = jo_float(pStage, "corr", 0.85f);
	string outputStr = jo_string(pStage, "output", "current", model.argMap);
	string borderModeStr = jo_string(pStage, "borderMode", "BORDER_REPLICATE", model.argMap);
	const char *angleKey = "angles";
	json_t *pAngles = jo_object(pStage, angleKey, model.argMap);
	if (!pAngles) {
		angleKey = "angle";
		pAngles = jo_object(pStage, angleKey, model.argMap);
	}
  const char *errMsg = NULL;
	int flags = INTER_LINEAR;
	int method;
	Mat tmplt;
	int borderMode;
	bool isOutputCurrent = outputStr.compare("current") == 0;
	bool isOutputInput = outputStr.compare("input") == 0;
	bool isOutputCorr = outputStr.compare("corr") == 0;
	vector<float> angles;

	if (tmpltPath.empty()) {
		errMsg = "Expected template path for imread";
	} else {
		if (model.image.channels() == 1) {
			tmplt = imread(tmpltPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		} else {
			tmplt = imread(tmpltPath.c_str(), CV_LOAD_IMAGE_COLOR);
		}
		if (tmplt.data) {
			LOGTRACE2("apply_matchTemplate(%s) %s", tmpltPath.c_str(), matInfo(tmplt).c_str());
			if (model.image.rows<tmplt.rows || model.image.cols<tmplt.cols) {
				errMsg = "Expected template smaller than image to match";
			}
		} else {
			errMsg = "imread failed";
		}
	}

	if (!errMsg) {
		if (!pAngles) {
			angles.push_back(0);
		} else if (json_is_string(pAngles)) {
			float angle = (float) atof(json_string_value(pAngles));
			angles.push_back(angle);
		} else if (json_is_number(pAngles)) {
			angles.push_back((float) json_number_value(pAngles));
		} else if (json_is_array(pAngles)) {
			size_t index;
			json_t *pAngle;
			json_array_foreach(pAngles, index, pAngle) {
				if (json_is_number(pAngle)) {
					angles.push_back((float) json_number_value(pAngle));
				} else {
					errMsg = "Expected numeric angle";
				}
			}
		} else {
			errMsg = "Expected numeric angle or JSON array of angles";
		}
	}

	int separation = jo_int(pStage, "separation", min(tmplt.cols,tmplt.rows), model.argMap);

	if (!errMsg) {
		if (borderModeStr.compare("BORDER_CONSTANT") == 0) {
			borderMode = BORDER_CONSTANT;
		} else if (borderModeStr.compare("BORDER_REPLICATE") == 0) {
			borderMode = BORDER_REPLICATE;
		} else if (borderModeStr.compare("BORDER_REFLECT") == 0) {
			borderMode = BORDER_REFLECT;
		} else if (borderModeStr.compare("BORDER_REFLECT_101") == 0) {
			borderMode = BORDER_REFLECT_101;
		} else if (borderModeStr.compare("BORDER_REFLECT101") == 0) {
			borderMode = BORDER_REFLECT101;
		} else if (borderModeStr.compare("BORDER_WRAP") == 0) {
			borderMode = BORDER_WRAP;
		} else {
			errMsg = "Expected borderMode: BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101, BORDER_WRAP";
		}
	}

	if (!errMsg && !isOutputInput && !isOutputCorr && !isOutputCurrent) {
		errMsg = "Expected \"output\" value: input, current, or corr";
	}

	if (!errMsg) {
		if (methodStr.compare("CV_TM_SQDIFF")==0) {
			method = CV_TM_SQDIFF;
		} else if (methodStr.compare( "CV_TM_SQDIFF_NORMED")==0) {
			method = CV_TM_SQDIFF_NORMED;
		} else if (methodStr.compare( "CV_TM_CCORR")==0) {
			method = CV_TM_CCORR;
		} else if (methodStr.compare( "CV_TM_CCORR_NORMED")==0) {
			method = CV_TM_CCORR_NORMED;
		} else if (methodStr.compare( "CV_TM_CCOEFF")==0) {
			method = CV_TM_CCOEFF;
		} else if (methodStr.compare( "CV_TM_CCOEFF_NORMED")==0) {
			method = CV_TM_CCOEFF_NORMED;
		} else {
			errMsg = "Expected method name";
		}
	} 

	Mat warpedTmplt;
	if (!errMsg) {
		if (pAngles) {
			matWarpRing(tmplt, warpedTmplt, angles);
		} else {
			warpedTmplt = tmplt;
		}
	}

	if (!errMsg) {
		Mat result;
		Mat imageSource = isOutputCurrent ? model.image.clone() : model.image;

		matchTemplate(imageSource, warpedTmplt, result, method);
		LOGTRACE4("apply_matchTemplate() matchTemplate(%s,%s,%s,%d)", 
			matInfo(imageSource).c_str(), matInfo(warpedTmplt).c_str(), matInfo(result).c_str(), method);

		vector<Point> matches;
		float maxVal = *max_element(result.begin<float>(),result.end<float>());
		bool isMin = method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED;
		if (isMin) {
			float rangeMin = 0;
			float rangeMax = corr * maxVal;
			matMinima(result, matches, rangeMin, rangeMax);
		} else {
			float rangeMin = corr * maxVal;
			float rangeMax = maxVal;
			matMaxima(result, matches, rangeMin, rangeMax);
		}

		int xOffset = isOutputCorr ? 0 : warpedTmplt.cols/2;
		int yOffset = isOutputCorr ? 0 : warpedTmplt.rows/2;
		modelMatches(Point(xOffset, yOffset), tmplt, result, angles, matches, pStageModel, maxVal, isMin);

		if (isOutputCorr) {
			LOGTRACE("apply_matchTemplate() normalize()");
			normalize(result, result, 0, 255, NORM_MINMAX);
			result.convertTo(model.image, CV_8U); 
		} else if (isOutputInput) {
			LOGTRACE("apply_matchTemplate() clone input");
			model.image = model.imageMap["input"].clone();
		}
	}

	return stageOK("apply_matchTemplate(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_dftSpectrum(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	int delta = jo_int(pStage, "delta", 1, model.argMap);
  bool isShift = jo_bool(pStage, "shift", true);	
  bool isLog = jo_bool(pStage, "log", true);	
	bool isMagnitude = false;
	bool isPhase = false;
	bool isReal = false;
	bool isImaginary = false;
	bool isMirror = jo_bool(pStage, "mirror", true);
	string showStr = jo_string(pStage, "show", "magnitude", model.argMap);
	const char *errMsg = NULL;

	if (!errMsg) {
		if (showStr.compare("magnitude") == 0) {
			isMagnitude = true;
		} else if (showStr.compare("phase") == 0) {
			isPhase = true;
		} else if (showStr.compare("real") == 0) {
			isReal = true;
		} else if (showStr.compare("imaginary") == 0) {
			isImaginary = true;
		} else {
			errMsg = "Expected 'magnitude' or 'phase' for show";
		}
	}

	if (!errMsg) {
		if (isReal) {
			if (model.image.channels() != 1) {
				errMsg = "Expected real (1-channel) Mat";
			}
		} else {
			if (model.image.channels() != 2) {
				errMsg = "Expected complex (2-channel) Mat";
			}
		}
	}

	if (!errMsg) {
		if (model.image.channels() > 1) {
			Mat planes[] = {
				Mat::zeros(model.image.size(), CV_32F),
				Mat::zeros(model.image.size(), CV_32F)
			};
			split(model.image, planes);
			if (isMagnitude) {
				magnitude(planes[0], planes[1], model.image);
			} else if (isPhase) {
				phase(planes[0], planes[1], model.image);
			} else if (isReal) {
				model.image = planes[0];
			} else if (isImaginary) {
				model.image = planes[1];
			}
		}
		if (delta) {
			model.image += Scalar::all(delta);
		}
		if (isLog) {
			log(model.image, model.image);
		}
		if (isShift) {
			dftShift(model.image, errMsg);
		}
		if (isMirror) {
			dftMirror(model.image);
		}
	}
	return stageOK("apply_dftSpectrum(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_dft(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	const char *errMsg = NULL;
	string depthStr = jo_string(pStage, "depth", "CV_8U", model.argMap);

	char errBuf[200];
	json_t *pFlags = jo_object(pStage, "flags", model.argMap);
	int flags = 0;

	if (json_is_array(pFlags)) {
		size_t index;
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
				snprintf(errBuf, sizeof(errBuf), "Unknown flag %s", flag);
				errMsg = errBuf;
			}
		}
	}

	if (!errMsg) {
		if (model.image.type() != CV_32F) {
			Mat fImage;
			LOGTRACE("apply_dft(): Convert image to CV_32F");
			model.image.convertTo(fImage, CV_32F); 
			model.image = fImage;
		}
		Mat dftImage;
		LOGTRACE1("apply_dft() flags:%d", flags);
		dft(model.image, dftImage, flags);
		model.image = dftImage;
		if (flags & DFT_INVERSE && depthStr.compare("CV_8U")==0) {
			Mat invImage;
			LOGTRACE("apply_dft(): Convert image to CV_8U");
			model.image.convertTo(invImage, CV_8U);
			model.image = invImage;
		}
	}

	return stageOK("apply_dft(%s) %s", errMsg, pStage, pStageModel);
}

