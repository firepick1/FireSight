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
	bool result = pValue ? json_is_true(pValue) : defaultValue;
	LOGTRACE3("%s:%d default:%d", key, result, defaultValue);
	return result;
}

static const int jo_int(const json_t *pObj, const char *key, int defaultValue=0) {
	json_t *pValue = json_object_get(pObj, key);
	int result = json_is_integer(pValue) ? json_integer_value(pValue) : defaultValue;
	LOGTRACE3("%s:%d default:%d", key, result, defaultValue);
	return result;
}

static const double jo_double(const json_t *pObj, const char *key, double defaultValue=0) {
	json_t *pValue = json_object_get(pObj, key);
	double result = json_is_number(pValue) ? json_number_value(pValue) : defaultValue;
	if (logLevel >= FIRELOG_TRACE) {
		char buf[128];
		sprintf(buf, "%f default:%f", result, defaultValue);
		LOGTRACE2("%s:%s", key, buf);
	}
	return result;
}

static const char * jo_string(const json_t *pObj, const char *key, const char *defaultValue = NULL) {
	json_t *pValue = json_object_get(pObj, key);
	const char * result = json_is_string(pValue) ? json_string_value(pValue) : defaultValue;
	LOGTRACE3("%s:%s default:%s", key, result, defaultValue);
	return result;
}

static Scalar jo_Scalar(const json_t *pObj, const char *key, const Scalar &defaultValue) {
	Scalar result = defaultValue;
	json_t *pValue = json_object_get(pObj, key);
	if (pValue) {
		if (!json_is_array(pValue)) {
			LOGERROR1("expected JSON array for %s", key);
		} else { 
			switch (json_array_size(pValue)) {
				case 1: 
					result = Scalar(json_number_value(json_array_get(pValue, 0)));
					break;
				case 2: 
					result = Scalar(json_number_value(json_array_get(pValue, 0)),
						json_number_value(json_array_get(pValue, 1)));
					break;
				case 3: 
					result = Scalar(json_number_value(json_array_get(pValue, 0)),
						json_number_value(json_array_get(pValue, 1)),
						json_number_value(json_array_get(pValue, 2)));
					break;
				case 4: 
					result = Scalar(json_number_value(json_array_get(pValue, 0)),
						json_number_value(json_array_get(pValue, 1)),
						json_number_value(json_array_get(pValue, 2)),
						json_number_value(json_array_get(pValue, 3)));
					break;
				default:
					LOGERROR1("expected JSON array with 1, 2, 3 or 4 integer values 0-255 for %s", key);
					return defaultValue;
			}
		}
	}
	if (pValue && logLevel >= FIRELOG_TRACE) {
	  char buf[100];
		sprintf(buf, "[%f %f %f %f] default:[%f %f %f %f]", 
			result[0], result[1], result[2], result[3],
			defaultValue[0], defaultValue[1], defaultValue[2], defaultValue[3]);
		LOGTRACE2("%s:%s", key, buf);
	}
	return result;
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

static bool apply_imread(json_t *pStage, json_t *pStageModel, json_t *pMode, Mat &image) {
  const char *path = jo_string(pStage, "path", NULL);
	const char *errMsg = NULL;

	if (!path) {
		errMsg = "expected path for imread";
	} else {
		image = imread(path, CV_LOAD_IMAGE_COLOR);
		if (image.data) {
			json_object_set(pStageModel, "rows", json_integer(image.rows));
			json_object_set(pStageModel, "cols", json_integer(image.cols));
		} else {
			errMsg = "imread failed";
			cout << "ERROR:" << errMsg << endl;
		}
	}

	return stageOK("apply_imread(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_imwrite(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
  const char *path = jo_string(pStage, "path", NULL);
	const char *errMsg = NULL;

	if (!path) {
		errMsg = "expected path for imwrite";
	} else {
		bool result = imwrite(path, image);
		json_object_set(pStageModel, "result", json_boolean(result));
	}

	return stageOK("apply_imwrite(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_cvtColor(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
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

static bool apply_drawKeypoints(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
	const char *errMsg = NULL;
	Scalar color = jo_Scalar(pStage, "color", Scalar::all(-1));
	int flags = jo_int(pStage, "flags", DrawMatchesFlags::DEFAULT);
	const char *stageModel = jo_string(pStage, "keypointStage", NULL);
	json_t *pKeypointStage = json_object_get(pModel, stageModel);

	if (!errMsg && flags < 0 || 7 < flags) {
		errMsg = "expected 0 < flags < 7";
	}

	if (!errMsg && !pKeypointStage) {
		errMsg = "expected keypointStage name";
	}

	vector<KeyPoint> keypoints;
	if (!errMsg) {
		json_t *pKeypoints = json_object_get(pKeypointStage, "keypoints");
		if (!json_is_array(pKeypoints)) {
		  errMsg = "keypointStage has no keypoints JSON array";
		} else {
			int index;
			json_t *pKeypoint;
			json_array_foreach(pKeypoints, index, pKeypoint) {
				double x = jo_double(pKeypoint, "pt.x", -1);
				double y = jo_double(pKeypoint, "pt.y", -1);
				double size = jo_double(pKeypoint, "size", 10);
				double angle = jo_double(pKeypoint, "angle", -1);
				KeyPoint keypoint(x, y, size, angle);
				keypoints.push_back(keypoint);
			}
		}
	}

	if (!errMsg) {
		drawKeypoints(image, keypoints, image, color, flags);
	}

	return stageOK("apply_drawKeypoints(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_dilate(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
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

static bool apply_erode(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
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

static bool apply_blur(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
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

static const char * modelKeyPoints(json_t*pStageModel, const vector<KeyPoint> &keyPoints) {
	json_t *pKeyPoints = json_array();
	json_object_set(pStageModel, "keypoints", pKeyPoints);
	for (int i=0; i<keyPoints.size(); i++){
	  json_t *pKeyPoint = json_object();
		json_object_set(pKeyPoint, "pt.x", json_real(keyPoints[i].pt.x));
		json_object_set(pKeyPoint, "pt.y", json_real(keyPoints[i].pt.y));
		json_object_set(pKeyPoint, "size", json_real(keyPoints[i].size));
		if (keyPoints[i].angle != -1) {
			json_object_set(pKeyPoint, "angle", json_real(keyPoints[i].angle));
		}
		if (keyPoints[i].response != 0) {
			json_object_set(pKeyPoint, "response", json_real(keyPoints[i].response));
		}
		json_array_append(pKeyPoints, pKeyPoint);
	}
}

static void drawRegions(Mat &image, vector<vector<Point> > &regions, Scalar color) {
	int nRegions = (int) regions.size();
	int blue = color[0];
	int green = color[1];
	int red = color[2];
	bool changeColor = red == -1 && green == -1 && blue == -1;

	for( int i = 0; i < nRegions; i++) {
		int nPts = regions[i].size();
		if (changeColor) {
			red = (i & 1) ? 0 : 255;
			green = (i & 2) ? 128 : 192;
			blue = (i & 1) ? 255 : 0;
		}
		for (int j = 0; j < nPts; j++) {
			image.at<Vec3b>(regions[i][j])[0] = blue;
			image.at<Vec3b>(regions[i][j])[1] = green;
			image.at<Vec3b>(regions[i][j])[2] = red;
		}
	}
}

static bool apply_MSER(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
	int delta = jo_int(pStage, "delta", 5);
	int minArea = jo_int(pStage, "minArea", 60);
	int maxArea = jo_int(pStage, "maxArea", 14400);
	float maxVariation = jo_double(pStage, "maxVariation", 0.25);
	float minDiversity = jo_double(pStage, "minDiversity", 0.2);
	int maxEvolution = jo_int(pStage, "maxEvolution", 200);
	double areaThreshold = jo_double(pStage, "areaThreshold", 1.01);
	double minMargin = jo_double(pStage, "minMargin", .003);
	int edgeBlurSize = jo_int(pStage, "edgeBlurSize", 5);
	Scalar color = jo_Scalar(pStage, "color", Scalar::all(-1));
	json_t * pMask = json_object_get(pStage, "mask");
	const char *errMsg = NULL;
	char errBuf[100];
	int maskX;
	int maskY;
	int maskW;
	int maskH;

	if (minArea < 0 || maxArea <= minArea) {
	  errMsg = "expected 0<=minArea and minArea<maxArea";
	} else if (maxVariation < 0 || minDiversity < 0) {
	  errMsg = "expected 0<=minDiversity and 0<=maxVariation";
	} else if (maxEvolution<0) {
	  errMsg = "expected 0<=maxEvolution";
	} else if (areaThreshold < 0 || minMargin < 0) {
	  errMsg = "expected 0<=areaThreshold and 0<=minMargin";
	} else if (edgeBlurSize < 0) {
	  errMsg = "expected 0<=edgeBlurSize";
	} if (pMask) {
	  if (!json_is_object(pMask)) {
		  errMsg = "expected mask JSON object with x, y, width, height";
		} else {
			if (pMask) {
				LOGTRACE("mask:{");
			}
			maskX = jo_int(pMask, "x", 0);
			maskY = jo_int(pMask, "y", 0);
			maskW = jo_int(pMask, "width", image.cols);
			maskH = jo_int(pMask, "height", image.rows);
			if (pMask) {
				LOGTRACE("}");
			}
			if (maskX < 0 || image.cols <= maskX) {
				sprintf(errBuf, "expected 0 <= mask.x < %d", image.cols);
				errMsg = errBuf;
			} else if (maskY < 0 || image.rows <= maskY) {
				sprintf(errBuf, "expected 0 <= mask.y < %d", image.cols);
				errMsg = errBuf;
			} else if (maskW <= 0 || image.cols < maskW) {
				sprintf(errBuf, "expected 0 < mask.width <= %d", image.cols);
				errMsg = errBuf;
			} else if (maskH <= 0 || image.rows < maskH) {
				sprintf(errBuf, "expected 0 < mask.height <= %d", image.rows);
				errMsg = errBuf;
			}
		}
	}

	if (!errMsg) {
		MSER mser(delta, minArea, maxArea, maxVariation, minDiversity,
			maxEvolution, areaThreshold, minMargin, edgeBlurSize);
		Mat mask;
		Rect maskRect(maskX, maskY, maskW, maskH);
		if (pMask) {
			mask = Mat::zeros(image.rows, image.cols, CV_8UC1);
			mask(maskRect) = 1;
		}
		vector<vector<Point> > regions;
		mser(image, regions, mask);

		int nRegions = (int) regions.size();
		LOGTRACE1("apply_MSER matched %d regions", nRegions);
		if (json_object_get(pStage, "color")) {
			if (image.channels() == 1) {
				cvtColor(image, image, CV_GRAY2BGR);
			}
			drawRegions(image, regions, color);
			if (pMask) {
				if (color[0]==-1 && color[1]==-1 && color[2]==-1 && color[3]==-1) {
					rectangle(image, maskRect, Scalar(255, 0, 255));
				} else {
					rectangle(image, maskRect, color);
				}
			}
		}
	}

	return stageOK("apply_MSER(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_SimpleBlobDetector(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
	SimpleBlobDetector::Params params;
	params.thresholdStep = jo_double(pStage, "thresholdStep", params.thresholdStep);
	params.minThreshold = jo_double(pStage, "minThreshold", params.minThreshold);
	params.maxThreshold = jo_double(pStage, "maxThreshold", params.maxThreshold);
	params.minRepeatability = jo_int(pStage, "minRepeatability", params.minRepeatability);
	params.minDistBetweenBlobs = jo_double(pStage, "minDistBetweenBlobs", params.minDistBetweenBlobs);
	params.filterByColor = jo_bool(pStage, "filterByColor", params.filterByColor);
	params.blobColor = jo_int(pStage, "blobColor", params.blobColor);
	params.filterByArea = jo_bool(pStage, "filterByArea", params.filterByArea);
	params.minArea = jo_double(pStage, "minArea", params.minArea);
	params.maxArea = jo_double(pStage, "maxArea", params.maxArea);
	params.filterByCircularity = jo_bool(pStage, "filterByCircularity", params.filterByCircularity);
	params.minCircularity = jo_double(pStage, "minCircularity", params.minCircularity);
	params.maxCircularity = jo_double(pStage, "maxCircularity", params.maxCircularity);
	params.filterByInertia = jo_bool(pStage, "filterByInertia", params.filterByInertia);
	params.minInertiaRatio = jo_double(pStage, "minInertiaRatio", params.minInertiaRatio);
	params.maxInertiaRatio = jo_double(pStage, "maxInertiaRatio", params.maxInertiaRatio);
	params.filterByConvexity = jo_bool(pStage, "filterByConvexity", params.filterByConvexity);
	params.minConvexity = jo_double(pStage, "minConvexity", params.minConvexity);
	params.maxConvexity = jo_double(pStage, "maxConvexity", params.maxConvexity);
	const char *errMsg = NULL;

	if (!errMsg) {
		SimpleBlobDetector detector(params);
		SimpleBlobDetector(params);
		detector.create("SimpleBlob");
		vector<cv::KeyPoint> keyPoints;
	  LOGTRACE("apply_SimpleBlobDetector detect()");
		detector.detect(image, keyPoints);
		modelKeyPoints(pStageModel, keyPoints);
	}

	return stageOK("apply_SimpleBlobDetector(%s) %s", errMsg, pStage, pStageModel);
}

static bool apply_Canny(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
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

static bool apply_HoleRecognizer(json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &image) {
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

static const char * dispatch(const char *pOp, json_t *pStage, json_t *pStageModel, json_t *pModel, Mat &workingImage) {
  const char *errMsg = NULL;
 
	if (strcmp(pOp, "blur")==0) {
		apply_blur(pStage, pStageModel, pModel, workingImage);
	} else if (strcmp(pOp, "Canny")==0) {
		apply_Canny(pStage, pStageModel, pModel, workingImage);
	} else if (strcmp(pOp, "cvtColor")==0) {
		apply_cvtColor(pStage, pStageModel, pModel, workingImage);
	} else if (strcmp(pOp, "dilate")==0) {
		apply_dilate(pStage, pStageModel, pModel, workingImage);
	} else if (strcmp(pOp, "drawKeypoints")==0) {
		apply_drawKeypoints(pStage, pStageModel, pModel, workingImage);
	} else if (strcmp(pOp, "erode")==0) {
		apply_erode(pStage, pStageModel, pModel, workingImage);
	} else if (strcmp(pOp, "HoleRecognizer")==0) {
		apply_HoleRecognizer(pStage, pStageModel, pModel, workingImage);
	} else if (strcmp(pOp, "imread")==0) {
		apply_imread(pStage, pStageModel, pModel, workingImage);
	} else if (strcmp(pOp, "imwrite")==0) {
		apply_imwrite(pStage, pStageModel, pModel, workingImage);
	} else if (strcmp(pOp, "MSER")==0) {
		apply_MSER(pStage, pStageModel, pModel, workingImage);
	} else if (strcmp(pOp, "SimpleBlobDetector")==0) {
		apply_SimpleBlobDetector(pStage, pStageModel, pModel, workingImage);
	} else {
		errMsg = "unknown op";
	}

	return errMsg;
}

static bool logErrorMessage(const char *errMsg, const char *pName, json_t *pStage) {
	if (errMsg) {
		char *pStageJson = json_dumps(pStage, 0);
		LOGERROR3("Pipeline::process stage:%s error:%s pStageJson:%s", pName, errMsg, pStageJson);
		free(pStageJson);
		return false;
	}
	return true;
}

json_t *Pipeline::process(Mat &workingImage) { 
	if (!json_is_array(pPipeline)) {
		const char * errMsg = "Pipeline::process expected json array for pipeline definition";
		LOGERROR1(errMsg, "");
		return json_string(errMsg);
	}

	bool ok = 1;
	size_t index;
	json_t *pStage;
	json_t *pModel = json_object();
	char nameBuf[16];
	LOGTRACE1("Pipeline::process(%d functions)", json_array_size(pPipeline));
	json_array_foreach(pPipeline, index, pStage) {
		const char *pOp = jo_string(pStage, "op", "");
		sprintf(nameBuf, "s%d", index+1);
		const char *pName = jo_string(pStage, "name", nameBuf);
		json_t *pStageModel = json_object();
		json_object_set(pModel, pName, pStageModel);
		LOGTRACE2("Pipeline::process stage:%s op:%s", pName, pOp);
		if (pOp) {
			try {
			  const char *errMsg = dispatch(pOp, pStage, pStageModel, pModel, workingImage);
				ok = logErrorMessage(errMsg, pName, pStage);
			} catch (runtime_error &ex) {
				ok = logErrorMessage(ex.what(), pName, pStage);
			} catch (cv::Exception &ex) {
				ok = logErrorMessage(ex.what(), pName, pStage);
			}
		} else {
			ok = logErrorMessage("expected op", pName, pStage);
		} //if (pOp)
		if (!ok) { 
			LOGERROR("cancelled pipeline execution");
			break; 
		}
	} // json_array_foreach

	return pModel;
}


