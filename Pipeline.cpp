#include <string.h>
#include <math.h>
#include <iostream>
#include <stdexcept>
#include "FireLog.h"
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "jo_util.hpp"
#include "MatUtil.hpp"
#include "version.h"

using namespace cv;
using namespace std;
using namespace FireSight;

bool Pipeline::stageOK(const char *fmt, const char *errMsg, json_t *pStage, json_t *pStageModel) {
	if (errMsg) {
		char *pStageJson = json_dumps(pStage, JSON_COMPACT|JSON_PRESERVE_ORDER);
		LOGERROR2(fmt, pStageJson, errMsg);
		free(pStageJson);
		return false;
	}

	if (logLevel >= FIRELOG_TRACE) {
	  char *pStageJson = json_dumps(pStage, 0);
		char *pModelJson = json_dumps(pStageModel, 0);
		LOGTRACE2(fmt, pStageJson, pModelJson);
		free(pStageJson);
		free(pModelJson);
	}

	return true;
}

bool Pipeline::apply_warpAffine(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	const char *errMsg = NULL;
  double scale = jo_double(pStage, "scale", 1);
  double angle = jo_double(pStage, "angle", 0);
  double dx = jo_double(pStage, "dx", (scale-1)*model.image.cols/2.0);
  double dy = jo_double(pStage, "dy", (scale-1)*model.image.rows/2.0);
	const char* borderModeStr = jo_string(pStage, "borderMode", "BORDER_REPLICATE");
	int borderMode;

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

	if (scale <= 0) {
		errMsg = "Expected 0<scale";
	}
	int width = jo_int(pStage, "width", model.image.cols);
	int height = jo_int(pStage, "height", model.image.rows);
	double cx = jo_double(pStage, "cx", model.image.cols/2.0);
	double cy = jo_double(pStage, "cy", model.image.rows/2.0);
	Scalar borderValue = jo_Scalar(pStage, "borderValue", Scalar::all(0));

	if (!errMsg) {
		Mat result;
		matWarpAffine(model.image, result, Point(cx,cy), angle, scale, Point(dx,dy), Size(width,height), borderMode, borderValue);
		model.image = result;
	}

	return stageOK("apply_warpAffine(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_stageImage(json_t *pStage, json_t *pStageModel, Model &model) {
  const char *stageStr = jo_string(pStage, "stage", NULL);
	const char *errMsg = NULL;

	if (!stageStr) {
		errMsg = "Expected name of stage for image";
	} else {
		model.image = model.imageMap[stageStr];
		if (!model.image.rows || !model.image.cols) {
			model.image = model.imageMap["input"].clone();
			LOGTRACE1("Could not locate stage image '%s', using input image", stageStr);
		}
	}

	return stageOK("apply_stageImage(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_imread(json_t *pStage, json_t *pStageModel, Model &model) {
  const char *path = jo_string(pStage, "path", NULL);
	const char *errMsg = NULL;

	if (!path) {
		errMsg = "expected path for imread";
	} else {
		model.image = imread(path, CV_LOAD_IMAGE_COLOR);
		if (model.image.data) {
			json_object_set(pStageModel, "rows", json_integer(model.image.rows));
			json_object_set(pStageModel, "cols", json_integer(model.image.cols));
		} else {
			errMsg = "imread failed";
			cout << "ERROR:" << errMsg << endl;
		}
	}

	return stageOK("apply_imread(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_imwrite(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
  const char *path = jo_string(pStage, "path", NULL);
	const char *errMsg = NULL;

	if (!path) {
		errMsg = "expected path for imwrite";
	} else {
		bool result = imwrite(path, model.image);
		json_object_set(pStageModel, "result", json_boolean(result));
	}

	return stageOK("apply_imwrite(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_cvtColor(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
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
		cvtColor(model.image, model.image, code, dstCn);
	}

	return stageOK("apply_cvtColor(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_drawRects(json_t *pStage, json_t *pStageModel, Model &model) {
	const char *errMsg = NULL;
	Scalar color = jo_Scalar(pStage, "color", Scalar::all(-1));
	int thickness = jo_int(pStage, "thickness", 2);
	const char* rectsModelName = jo_string(pStage, "model", "");
	json_t *pRectsModel = json_object_get(model.getJson(false), rectsModelName);

	if (!json_is_object(pRectsModel)) {
		errMsg = "Expected name of stage model with rects";
	}

	json_t *pRects = NULL;
	if (!errMsg) {
		pRects = json_object_get(pRectsModel, "rects");
		if (!json_is_array(pRects)) {
			errMsg = "Expected array of rects";
		}
	}

	if (!errMsg) {
		if (model.image.channels() == 1) {
			LOGTRACE("Converting grayscale image to color image");
			cvtColor(model.image, model.image, CV_GRAY2BGR, 0);
		}
		int index;
		json_t *pRect;
		Point2f vertices[4];
		int blue = color[0];
		int green = color[1];
		int red = color[2];
		bool changeColor = red == -1 && green == -1 && blue == -1;

		json_array_foreach(pRects, index, pRect) {
			double x = jo_double(pRect, "x", -1);
			double y = jo_double(pRect, "y", -1);
			double width = jo_double(pRect, "width", -1);
			double height = jo_double(pRect, "height", -1);
			double angle = jo_double(pRect, "angle", -1);
			if (changeColor) {
				red = (index & 1) ? 0 : 255;
				green = (index & 2) ? 128 : 192;
				blue = (index & 1) ? 255 : 0;
				color = Scalar(blue, green, red);
			}
			RotatedRect rect(Point(x,y), Size(width, height), angle);
			rect.points(vertices);
			for (int i = 0; i < 4; i++) {
		    line(model.image, vertices[i], vertices[(i+1)%4], color, thickness);
			}
		}
	}

	return stageOK("apply_drawRects(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_drawKeypoints(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	const char *errMsg = NULL;
	Scalar color = jo_Scalar(pStage, "color", Scalar::all(-1));
	int flags = jo_int(pStage, "flags", DrawMatchesFlags::DRAW_OVER_OUTIMG|DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	const char *modelName = jo_string(pStage, "model", NULL);
	json_t *pKeypointStage = json_object_get(model.getJson(false), modelName);

	if (!pKeypointStage) {
		const char *keypointStageName = jo_string(pStage, "keypointStage", NULL);
		pKeypointStage = json_object_get(model.getJson(false), keypointStageName);
	}

	if (!errMsg && flags < 0 || 7 < flags) {
		errMsg = "expected 0 < flags < 7";
	}

	if (!errMsg && !pKeypointStage) {
		errMsg = "expected name of stage model";
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
		drawKeypoints(model.image, keypoints, model.image, color, flags);
	}

	return stageOK("apply_drawKeypoints(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_dilate(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	const char *errMsg = NULL;
	int kwidth = jo_int(pStage, "ksize.width", 3);
	int kheight = jo_int(pStage, "ksize.height", 3);
	int shape = jo_shape(pStage, "shape", errMsg);

	if (!errMsg) {
	  Mat structuringElement = getStructuringElement(shape, Size(kwidth, kheight));
		dilate(model.image, model.image, structuringElement);
	}

	return stageOK("apply_dilate(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_erode(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	const char *errMsg = NULL;
	int kwidth = jo_int(pStage, "ksize.width", 3);
	int kheight = jo_int(pStage, "ksize.height", 3);
	int shape = jo_shape(pStage, "shape", errMsg);

	if (!errMsg) {
	  Mat structuringElement = getStructuringElement(shape, Size(kwidth, kheight));
		erode(model.image, model.image, structuringElement);
	}

	return stageOK("apply_erode(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_equalizeHist(json_t *pStage, json_t *pStageModel, Model &model) {
	const char *errMsg = NULL;

	if (!errMsg) {
		equalizeHist(model.image, model.image);
	}

	return stageOK("apply_equalizeHist(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_blur(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	const char *errMsg = NULL;
	int width = jo_int(pStage, "ksize.width", 3);
	int height = jo_int(pStage, "ksize.height", 3);
	int anchorx = jo_int(pStage, "anchor.x", -1);
	int anchory = jo_int(pStage, "anchor.y", -1);

	if (width <= 0 || height <= 0) {
		errMsg = "expected 0<width and 0<height";
	}

	if (!errMsg) {
		blur(model.image, model.image, Size(width,height));
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

bool Pipeline::apply_SimpleBlobDetector(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
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
		detector.detect(model.image, keyPoints);
		modelKeyPoints(pStageModel, keyPoints);
	}

	return stageOK("apply_SimpleBlobDetector(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_rectangle(json_t *pStage, json_t *pStageModel, Model &model) {
	double x = jo_double(pStage, "x", 0);
	double y = jo_double(pStage, "y", 0);
	double width = jo_double(pStage, "width", model.image.cols);
	double height = jo_double(pStage, "height", model.image.rows);
	int thickness = jo_int(pStage, "thickness", 1);
	int lineType = jo_int(pStage, "lineType", 8);
	Scalar color = jo_Scalar(pStage, "color", Scalar::all(0));
	Scalar flood = jo_Scalar(pStage, "flood", Scalar::all(-1));
	Scalar fill = jo_Scalar(pStage, "fill", Scalar::all(-1));
	int shift = jo_int(pStage, "shift", 0);
	const char *errMsg = NULL;

  if ( x < 0 || y < 0) {
		errMsg = "Expected 0<=x and 0<=y";
	} else if (shift < 0) {
		errMsg = "Expected shift>=0";
	}

	if (!errMsg) {
		if (model.image.cols == 0 || model.image.rows == 0) {
			model.image = Mat(height, width, CV_8UC3, Scalar(0,0,0));
		}
		if (thickness) {
			rectangle(model.image, Rect(x,y,width,height), color, thickness, lineType, shift);
		}
		if (thickness >= 0) {
			double outThickness = thickness/2;
			double inThickness = thickness - outThickness;
			if (fill[0] >= 0) {
				rectangle(model.image, Rect(x+inThickness,y+inThickness,width-2*inThickness,height-2*inThickness), fill, -1, lineType, shift);
			}
			if (flood[0] >= 0) {
				double left = x - outThickness;
				double top = y - outThickness;
				double right = x+width+outThickness;
				double bot = y+height+outThickness;
				rectangle(model.image, Rect(0,0,model.image.cols,top), flood, -1, lineType, shift);
				rectangle(model.image, Rect(0,bot,model.image.cols,model.image.rows-bot), flood, -1, lineType, shift);
				rectangle(model.image, Rect(0,top,left,height+2*outThickness), flood, -1, lineType, shift);
				rectangle(model.image, Rect(right,top,model.image.cols-right,height+2*outThickness), flood, -1, lineType, shift);
			}
		}
	}

	return stageOK("apply_rectangle(%s) %s", errMsg, pStage, pStageModel);
}

int Pipeline::parseCvType(const char *typeStr, const char *&errMsg) {
	int type = CV_8U;

	if (strcmp("CV_8UC3", typeStr) == 0) {
		type = CV_8UC3;
	} else if (strcmp("CV_8UC2", typeStr) == 0) {
		type = CV_8UC2;
	} else if (strcmp("CV_8UC1", typeStr) == 0) {
		type = CV_8UC1;
	} else if (strcmp("CV_8U", typeStr) == 0) {
		type = CV_8UC1;
	} else if (strcmp("CV_32F", typeStr) == 0) {
		type = CV_32F;
	} else if (strcmp("CV_32FC1", typeStr) == 0) {
		type = CV_32FC1;
	} else if (strcmp("CV_32FC2", typeStr) == 0) {
		type = CV_32FC2;
	} else if (strcmp("CV_32FC3", typeStr) == 0) {
		type = CV_32FC3;
	} else {
		errMsg = "Unsupported type";
	}

	return type;
}

bool Pipeline::apply_Mat(json_t *pStage, json_t *pStageModel, Model &model) {
	double width = jo_double(pStage, "width", model.image.cols);
	double height = jo_double(pStage, "height", model.image.rows);
	const char *typeStr = jo_string(pStage, "type", "CV_8UC3");
	Scalar color = jo_Scalar(pStage, "color", Scalar::all(0));
	const char *errMsg = NULL;
	int type = CV_8UC3;

	if (width <= 0 || height <= 0) {
		errMsg = "Expected 0<width and 0<height";
	} else if (color[0] <0 || color[1]<0 || color[2]<0) {
		errMsg = "Expected color JSON array with non-negative values";
	} 
	
	if (!errMsg) {
		type = parseCvType(typeStr, errMsg);
	}

	if (!errMsg) {
		model.image = Mat(height, width, type, color);
	}

	return stageOK("apply_Mat(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_calcHist(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	float rangeMin = jo_double(pStage, "rangeMin", 0);
	float rangeMax = jo_double(pStage, "rangeMax", 256);
	int locations = jo_int(pStage, "locations", 0);
	int bins = jo_int(pStage, "bins", rangeMax-rangeMin);
	int histSize = bins;
	bool uniform = true;
	bool accumulate = false;
	const char *errMsg = NULL;
	Mat mask;

	if (rangeMin > rangeMax) {
		errMsg = "Expected rangeMin <= rangeMax";
	} else if (bins < 2 || bins > 256 ) {
		errMsg = "Expected 1<bins and bins<=256";
	}

	if (!errMsg) {
		float rangeC0[] = { rangeMin, rangeMax }; 
		const float* ranges[] = { rangeC0 };
		Mat hist;
		calcHist(&model.image, 1, 0, mask, hist, 1, &histSize, ranges, uniform, accumulate);
		json_t *pHist = json_array();
		for (int i = 0; i < histSize; i++) {
			json_array_append(pHist, json_real(hist.at<float>(i)));
		}
		json_object_set(pStageModel, "hist", pHist);
		json_t *pLocations = json_array();
		json_object_set(pStageModel, "locations", pLocations);
	}

	return stageOK("apply_calcHist(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_split(json_t *pStage, json_t *pStageModel, Model &model) {
	json_t *pFromTo = json_object_get(pStage, "fromTo");
	const char *errMsg = NULL;
#define MAX_FROMTO 32
	int fromTo[MAX_FROMTO];
	int nFromTo;

	if (!json_is_array(pFromTo)) {
		errMsg = "Expected JSON array for fromTo";
	}

	if (!errMsg) {
		json_t *pInt;
		int index;
		json_array_foreach(pFromTo, index, pInt) {
			if (index >= MAX_FROMTO) {
				errMsg = "Too many channels";
				break;
			}
			nFromTo = index+1;
			fromTo[index] = json_integer_value(pInt);
		}
	}

	if (!errMsg) {
		int depth = model.image.depth();
		int channels = 1;
		Mat outImage( model.image.rows, model.image.cols, CV_MAKETYPE(depth, channels) );
		LOGTRACE1("Creating output model.image %s", matInfo(outImage).c_str());
		Mat out[] = { outImage };
		mixChannels( &model.image, 1, out, 1, fromTo, nFromTo/2 );
		model.image = outImage;
	}

	return stageOK("apply_split(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_convertTo(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	double alpha = jo_double(pStage, "alpha", 1);
	double delta = jo_double(pStage, "delta", 0);
	const char *transform = jo_string(pStage, "transform", NULL);
	const char *rTypeStr = jo_string(pStage, "rType", "CV_8U");
	const char *errMsg = NULL;
	int rType;

	if (!errMsg) {
		rType = parseCvType(rTypeStr, errMsg);
	}

	if (transform) {
		if (strcmp("log", transform) == 0) {
			LOGTRACE("log()");
			log(model.image, model.image);
		}
	}
	
	if (!errMsg) {
		model.image.convertTo(model.image, rType, alpha, delta);
	}

	return stageOK("apply_convertTo(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_cout(json_t *pStage, json_t *pStageModel, Model &model) {
	int col = jo_int(pStage, "col", 0);
	int row = jo_int(pStage, "row", 0);
	int cols = jo_int(pStage, "cols", model.image.cols);
	int rows = jo_int(pStage, "rows", model.image.rows);
	int precision = jo_int(pStage, "precision", 1);
	int width = jo_int(pStage, "width", 5);
	int channel = jo_int(pStage, "channel", 0);
	const char *comment = jo_string(pStage, "comment", NULL);
	const char *errMsg = NULL;

	if (row<0 || col<0 || rows<=0 || cols<=0) {
		errMsg = "Expected 0<=row and 0<=col and 0<cols and 0<rows";
	}
	if (rows > model.image.rows) {
		rows = model.image.rows;
	}
	if (cols > model.image.cols) {
		cols = model.image.cols;
	}

	if (!errMsg) {
		int depth = model.image.depth();
		cout << matInfo(model.image);
		cout << " show:[" << row << "-" << row+rows-1 << "," << col << "-" << col+cols-1 << "]";
		if (comment) {
			cout << " " << comment;
		}
		cout << endl;
		for (int r = row; r < row+rows; r++) {
			for (int c = col; c < col+cols; c++) {
				cout.precision(precision);
				cout.width(width);
				if (model.image.channels() == 1) {
					switch (depth) {
						case CV_8S:
						case CV_8U:
							cout << (short) model.image.at<unsigned char>(r,c,channel) << " ";
							break;
						case CV_16U:
							cout << model.image.at<unsigned short>(r,c) << " ";
							break;
						case CV_16S:
							cout << model.image.at<short>(r,c) << " ";
							break;
						case CV_32S:
							cout << model.image.at<int>(r,c) << " ";
							break;
						case CV_32F:
							cout << std::fixed;
							cout << model.image.at<float>(r,c) << " ";
							break;
						case CV_64F:
							cout << std::fixed;
							cout << model.image.at<double>(r,c) << " ";
							break;
						default:
							cout << "UNSUPPORTED-CONVERSION" << " ";
							break;
					}
				} else {
					switch (depth) {
						case CV_8S:
						case CV_8U:
							cout << (short) model.image.at<Vec2b>(r,c)[channel] << " ";
							break;
						case CV_16U:
							cout << model.image.at<Vec2w>(r,c)[channel] << " ";
							break;
						case CV_16S:
							cout << model.image.at<Vec2s>(r,c)[channel] << " ";
							break;
						case CV_32S:
							cout << model.image.at<Vec2i>(r,c)[channel] << " ";
							break;
						case CV_32F:
							cout << std::fixed;
							cout << model.image.at<Vec2f>(r,c)[channel] << " ";
							break;
						case CV_64F:
							cout << std::fixed;
							cout << model.image.at<Vec2d>(r,c)[channel] << " ";
							break;
						default:
							cout << "UNSUPPORTED-CONVERSION" << " ";
							break;
					}
				}
			}
			cout << endl;
		}
	}

	return stageOK("apply_cout(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_normalize(json_t *pStage, json_t *pStageModel, Model &model) {
	double alpha = jo_double(pStage, "alpha", 1);
	double beta = jo_double(pStage, "beta", 0);
	const char * normTypeStr = jo_string(pStage, "normType", "NORM_L2");
	int normType  = NORM_L2;
	const char *errMsg = NULL;

	if (strcmp("NORM_L2",normTypeStr) == 0) {
		normType = NORM_L2;
	} else if (strcmp("NORM_L1",normTypeStr) == 0) {
		normType = NORM_L1;
	} else if (strcmp("NORM_MINMAX",normTypeStr) == 0) {
		normType = NORM_MINMAX;
	} else if (strcmp("NORM_INF",normTypeStr) == 0) {
		normType = NORM_INF;
	} else {
		errMsg = "Unknown normType";
	}

	if (!errMsg) {
		normalize(model.image, model.image, alpha, beta, normType);
	}

	return stageOK("apply_normalize(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_Canny(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
	double threshold1 = jo_double(pStage, "threshold1", 0);
	double threshold2 = jo_double(pStage, "threshold2", 50);
	double apertureSize = jo_double(pStage, "apertureSize", 3);
	bool L2gradient = jo_bool(pStage, "L2gradient", false);
	const char *errMsg = NULL;

	if (!errMsg) {
		Canny(model.image, model.image, threshold1, threshold2, apertureSize, L2gradient);
	}

	return stageOK("apply_Canny(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_HoleRecognizer(json_t *pStage, json_t *pStageModel, Model &model) {
	validateImage(model.image);
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
		recognizer.scan(model.image, matches);
		json_t *holes = json_array();
		json_object_set(pStageModel, "holes", holes);
		for (int i = 0; i < matches.size(); i++) {
			json_array_append(holes, matches[i].as_json_t());
		}
	}

	return stageOK("apply_HoleRecognizer(%s) %s", errMsg, pStage, pStageModel);
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
	if (pPipeline->refcount) {
		LOGERROR1("~Pipeline() pPipeline->refcount:%d EXPECTED 0", pPipeline->refcount);
	} else {
		LOGTRACE1("~Pipeline() pPipeline->refcount:%d", pPipeline->refcount);
	}
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

void Pipeline::validateImage(Mat &image) {
	if (image.cols == 0 || image.rows == 0) {
		image = Mat(100,100, CV_8UC3);
		putText(image, "FireSight:", Point(10,20), FONT_HERSHEY_PLAIN, 1, Scalar(128,255,255));
		putText(image, "No input", Point(10,40), FONT_HERSHEY_PLAIN, 1, Scalar(128,255,255));
		putText(image, "image?", Point(10,60), FONT_HERSHEY_PLAIN, 1, Scalar(128,255,255));
	}
}

json_t *Pipeline::process(Mat &workingImage) { 
	Model model;
	json_t *pModelJson = model.getJson(true);

	json_t *pFireSight = json_object();
	char version[100];
	snprintf(version, sizeof(version), "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
	json_object_set(pFireSight, "version", json_string(version));
	json_object_set(pFireSight, "url", json_string("https://github.com/firepick1/FireSight"));
	json_object_set(pModelJson, "FireSight", pFireSight);

	model.image = workingImage;
	model.imageMap["input"] = model.image.clone();
	bool ok = processModel(model);
	workingImage = model.image;

	return pModelJson;
}

bool Pipeline::processModel(Model &model) { 
	if (!json_is_array(pPipeline)) {
		const char * errMsg = "Pipeline::process expected json array for pipeline definition";
		LOGERROR1(errMsg, "");
		throw errMsg;
	}

	bool ok = 1;
	size_t index;
	json_t *pStage;
	char debugBuf[255];
	json_array_foreach(pPipeline, index, pStage) {
		const char *pOp = jo_string(pStage, "op", "");
		const char *pName = jo_string(pStage, "name", NULL);
		bool isSaveImage = pName != NULL;
		if (!pName || strlen(pName)==0) {
			char defaultName[100];
			snprintf(defaultName, sizeof(defaultName), "s%d", index+1);
			pName = defaultName;
		}
		const char *pComment = jo_string(pStage, "comment", "");
		json_t *pStageModel = json_object();
		json_object_set(model.getJson(false), pName, pStageModel);
		snprintf(debugBuf,sizeof(debugBuf), "process() %s op:%s stage:%s %s", matInfo(model.image).c_str(), pOp, pName, pComment);
		if (strncmp(pOp, "nop", 3)==0) {
			LOGTRACE1("%s (NO ACTION TAKEN)", debugBuf);
		} else if (strcmp(pName, "input")==0) {
			ok = logErrorMessage("\"input\" is the reserved stage name for the input image", pName, pStage);
		} else {
			LOGDEBUG1("%s", debugBuf);
			try {
			  const char *errMsg = dispatch(pOp, pStage, pStageModel, model);
				ok = logErrorMessage(errMsg, pName, pStage);
				if (isSaveImage) {
					model.imageMap[pName] = model.image.clone();
				}
			} catch (runtime_error &ex) {
				ok = logErrorMessage(ex.what(), pName, pStage);
			} catch (cv::Exception &ex) {
				ok = logErrorMessage(ex.what(), pName, pStage);
			}
		} //if-else (pOp)
		if (!ok) { 
			LOGERROR("cancelled pipeline execution");
			ok = false;
			break; 
		}
		if (model.image.cols <=0 || model.image.rows<=0) {
			LOGERROR2("Empty working image: %dr x %dc", model.image.rows, model.image.cols);
			ok = false;
			break;
		}
	} // json_array_foreach

	LOGDEBUG2("Pipeline::processModel(stages:%d) -> %s", json_array_size(pPipeline), matInfo(model.image).c_str());

	return ok;
}

const char * Pipeline::dispatch(const char *pOp, json_t *pStage, json_t *pStageModel, Model &model) {
	bool ok = true;
  const char *errMsg = NULL;
 
	if (strcmp(pOp, "blur")==0) {
		ok = apply_blur(pStage, pStageModel, model);
	} else if (strcmp(pOp, "calcHist")==0) {
		ok = apply_calcHist(pStage, pStageModel, model);
	} else if (strcmp(pOp, "convertTo")==0) {
		ok = apply_convertTo(pStage, pStageModel, model);
	} else if (strcmp(pOp, "cout")==0) {
		ok = apply_cout(pStage, pStageModel, model);
	} else if (strcmp(pOp, "Canny")==0) {
		ok = apply_Canny(pStage, pStageModel, model);
	} else if (strcmp(pOp, "cvtColor")==0) {
		ok = apply_cvtColor(pStage, pStageModel, model);
	} else if (strcmp(pOp, "dft")==0) {
		ok = apply_dft(pStage, pStageModel, model);
	} else if (strcmp(pOp, "dftSpectrum")==0) {
		ok = apply_dftSpectrum(pStage, pStageModel, model);
	} else if (strcmp(pOp, "dilate")==0) {
		ok = apply_dilate(pStage, pStageModel, model);
	} else if (strcmp(pOp, "drawKeypoints")==0) {
		ok = apply_drawKeypoints(pStage, pStageModel, model);
	} else if (strcmp(pOp, "drawRects")==0) {
		ok = apply_drawRects(pStage, pStageModel, model);
	} else if (strcmp(pOp, "equalizeHist")==0) {
		ok = apply_equalizeHist(pStage, pStageModel, model);
	} else if (strcmp(pOp, "erode")==0) {
		ok = apply_erode(pStage, pStageModel, model);
	} else if (strcmp(pOp, "HoleRecognizer")==0) {
		ok = apply_HoleRecognizer(pStage, pStageModel, model);
	} else if (strcmp(pOp, "imread")==0) {
		ok = apply_imread(pStage, pStageModel, model);
	} else if (strcmp(pOp, "imwrite")==0) {
		ok = apply_imwrite(pStage, pStageModel, model);
	} else if (strcmp(pOp, "Mat")==0) {
		ok = apply_Mat(pStage, pStageModel, model);
	} else if (strcmp(pOp, "matchTemplate")==0) {
		ok = apply_matchTemplate(pStage, pStageModel, model);
	} else if (strcmp(pOp, "MSER")==0) {
		ok = apply_MSER(pStage, pStageModel, model);
	} else if (strcmp(pOp, "normalize")==0) {
		ok = apply_normalize(pStage, pStageModel, model);
	} else if (strcmp(pOp, "proto")==0) {
		ok = apply_proto(pStage, pStageModel, model);
	} else if (strcmp(pOp, "rectangle")==0) {
		ok = apply_rectangle(pStage, pStageModel, model);
	} else if (strcmp(pOp, "SimpleBlobDetector")==0) {
		ok = apply_SimpleBlobDetector(pStage, pStageModel, model);
	} else if (strcmp(pOp, "split")==0) {
		ok = apply_split(pStage, pStageModel, model);
	} else if (strcmp(pOp, "stageImage")==0) {
		ok = apply_stageImage(pStage, pStageModel, model);
	} else if (strcmp(pOp, "warpAffine")==0) {
		ok = apply_warpAffine(pStage, pStageModel, model);
	} else if (strcmp(pOp, "warpRing")==0) {
		ok = apply_warpRing(pStage, pStageModel, model);

	} else if (strncmp(pOp, "nop", 3)==0) {
		LOGDEBUG("Skipping nop...");
	} else {
		errMsg = "unknown op";
	}

	if (!ok) {
		errMsg = "Pipeline stage failed";
	}

	return errMsg;
}
