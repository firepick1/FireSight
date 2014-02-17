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

bool Pipeline::apply_matchTemplate(json_t *pStage, json_t *pStageModel, Model &model) {
  const char * methodStr = jo_string(pStage, "method", "CV_TM_CCOEFF_NORMED");
  const char *tmpltPath = jo_string(pStage, "template", NULL);
	float threshold = jo_double(pStage, "threshold", 0.8);
	float corr = jo_double(pStage, "corr", 0.92);
	const char* outputStr = jo_string(pStage, "output", "current");
	const char* borderModeStr = jo_string(pStage, "borderMode", "BORDER_REPLICATE");
	float angle = jo_double(pStage, "angle", 0);
  const char *errMsg = NULL;
	int flags = INTER_LINEAR;
	int method;
	Mat tmplt;
	int borderMode;
	bool isOutputCurrent = strcmp(outputStr, "current") == 0;
	bool isOutputInput = strcmp(outputStr, "input") == 0;
	bool isOutputCorr = strcmp(outputStr, "corr") == 0;

	assert(0<model.image.rows && 0<model.image.cols);

	if (!tmpltPath) {
		errMsg = "Expected template path for imread";
	} else {
		if (model.image.channels() == 1) {
			tmplt = imread(tmpltPath, CV_LOAD_IMAGE_GRAYSCALE);
		} else {
			tmplt = imread(tmpltPath, CV_LOAD_IMAGE_COLOR);
		}
		if (tmplt.data) {
			LOGTRACE2("apply_matchTemplate(%s) %s", tmpltPath, matInfo(tmplt).c_str());
			if (model.image.rows<tmplt.rows || model.image.cols<tmplt.cols) {
				errMsg = "Expected template smaller than image to match";
			}
		} else {
			errMsg = "imread failed";
		}
	}

	int separation = jo_int(pStage, "separation", min(tmplt.cols,tmplt.rows));

	if (!errMsg) {
		if (strcmp("BORDER_CONSTANT", borderModeStr) == 0) {
			borderMode = BORDER_CONSTANT;
		} else if (strcmp("BORDER_REPLICATE", borderModeStr) == 0) {
			borderMode = BORDER_REPLICATE;
		} else if (strcmp("BORDER_REFLECT", borderModeStr) == 0) {
			borderMode = BORDER_REFLECT;
		} else if (strcmp("BORDER_REFLECT_101", borderModeStr) == 0) {
			borderMode = BORDER_REFLECT_101;
		} else if (strcmp("BORDER_REFLECT101", borderModeStr) == 0) {
			borderMode = BORDER_REFLECT101;
		} else if (strcmp("BORDER_WRAP", borderModeStr) == 0) {
			borderMode = BORDER_WRAP;
		} else {
			errMsg = "Expected borderMode: BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101, BORDER_WRAP";
		}
	}

	if (!errMsg && !isOutputInput && !isOutputCorr && !isOutputCurrent) {
		errMsg = "Expected \"output\" value: input, current, or corr";
	}

	if (!errMsg) {
		if (strcmp(methodStr, "CV_TM_SQDIFF")==0) {
			method = CV_TM_SQDIFF;
		} else if (strcmp(methodStr, "CV_TM_SQDIFF_NORMED")==0) {
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
		Mat result;
		Mat imageSource = isOutputCurrent ? model.image.clone() : model.image;
		int tw = tmplt.cols;
		int th = tmplt.rows;

		if (angle) {
			matWarpAffine(tmplt, Point(tw/2.0,th/2.0), angle, 1, Point(0,0), Size(-1,-1), Scalar(0,0,0), 
				borderMode, flags);
		}

		matchTemplate(imageSource, tmplt, result, method);
		assert(result.isContinuous());
		assert(result.channels() == 1);
		vector<RotatedRect> rects;

		bool isMin = method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED;
		float maxVal = *max_element(result.begin<float>(),result.end<float>());
		float minVal = *min_element(result.begin<float>(),result.end<float>());
		float rejectedMax = -1;
		float rejectedMin = maxVal+1;
		if (isMin && minVal <= threshold || !isMin && threshold <= maxVal) { // Filter matches
			int rDelta = separation/2; // tmplt.rows/2;
			int cDelta = separation/2; // tmplt.cols/2;
			float rangeMin = isMin ? 0 : corr * maxVal;
			float rangeMax = isMin ? corr * maxVal : maxVal;
			for (int r=0; r < result.rows; r++) {
				for (int c=0; c < result.cols; c++) {
					float val = result.at<float>(r,c);
					bool isOverlap = false;
					if (val < rangeMin) {
						rejectedMax = max(rejectedMax, val);
					} else if (val > rangeMax) {
						rejectedMin = min(rejectedMin, val);
					} else {
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
							rects.push_back(RotatedRect(Point(c,r), Size(tw,th), -angle));
						}
					}
				}
			}
		} else {
			rejectedMax = maxVal;
			rejectedMin = minVal;
			LOGTRACE("No match (maxVal is below threshold)");
		}

		// Model matches
		json_t *pRects = json_array();
		json_object_set(pStageModel, "maxVal", json_real(maxVal));
		if (isMin) {
			json_object_set(pStageModel, "rejectedMin", json_real(rejectedMin));
		} else {
			json_object_set(pStageModel, "rejectedMax", json_real(rejectedMax));
		}
		json_object_set(pStageModel, "rects", pRects);
		int xOffset = isOutputCorr ? 0 : tmplt.cols/2;
		int yOffset = isOutputCorr ? 0 : tmplt.rows/2;
		for (int irect=0; irect<rects.size(); irect++) {
			int cx = rects[irect].center.x;
			int cy = rects[irect].center.y;
			float val = result.at<float>(cy,cx);
			json_t *pRect = json_object();
			json_object_set(pRect, "x", json_real(cx+xOffset));
			json_object_set(pRect, "y", json_real(cy+yOffset));
			json_object_set(pRect, "width", json_real(tw));
			json_object_set(pRect, "height", json_real(th));
			json_object_set(pRect, "angle", json_real(-angle));
			json_object_set(pRect, "corr", json_real(val/maxVal));
			json_array_append(pRects, pRect);
		}

		if (isOutputCorr) {
			normalize(result, result, 0, 255, NORM_MINMAX);
			result.convertTo(model.image, CV_8U); 
		} else if (isOutputInput) {
			model.image = model.imageMap["input"].clone();
		}

	}

	return stageOK("apply_matchTemplate(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_dftSpectrum(json_t *pStage, json_t *pStageModel, Model &model) {
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

	assert(0<model.image.rows && 0<model.image.cols);

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

	assert(0<model.image.rows && 0<model.image.cols);

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
		if (flags & DFT_INVERSE && strcmp("CV_8U",depthStr)==0) {
			Mat invImage;
			LOGTRACE("apply_dft(): Convert image to CV_8U");
			model.image.convertTo(invImage, CV_8U);
			model.image = invImage;
		}
	}

	return stageOK("apply_dft(%s) %s", errMsg, pStage, pStageModel);
}

