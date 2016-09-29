#include <string.h>
#include <math.h>
#include <fstream>
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
#include "Sharpness.h"

using namespace cv;
using namespace std;
using namespace firesight;

StageData::StageData(string stageName) {
    LOGTRACE1("StageData constructor %s", stageName.c_str());
}

StageData::~StageData() {
    LOGTRACE("StageData destructor");
}

bool Pipeline::stageOK(const char *fmt, const char *errMsg, json_t *pStage, json_t *pStageModel) {
    if (errMsg && *errMsg) {
        char *pStageJson = json_dumps(pStage, JSON_COMPACT|JSON_PRESERVE_ORDER);
        LOGERROR2(fmt, pStageJson, errMsg);
        free(pStageJson);
        json_object_set(pStageModel, "error", json_string(errMsg));
        return false;
    }

    if (logLevel >= FIRELOG_TRACE) {
        char *pStageJson = json_dumps(pStage, 0);
        //char *pModelJson = json_dumps(pStageModel, 0);
        //LOGTRACE2(fmt, pStageJson, pModelJson);
        LOGTRACE2(fmt, pStageJson, "");
        //free(pModelJson);
        free(pStageJson);
    }

    return true;
}

bool Pipeline::apply_model(json_t *pStage, json_t *pStageModel, Model &model) {
    json_t *pModel = json_object_get(pStage, "model");
    const char *errMsg = NULL;

    if (!errMsg) {
        if (!json_is_object(pModel)) {
            errMsg = "Expected JSON object for stage model";
        }
    }
    if (!errMsg && pModel) {
        const char * pKey;
        json_t *pValue;
        json_object_foreach(pModel, pKey, pValue) {
            json_object_set(pStageModel, pKey, pValue);
        }
    }

    return stageOK("apply_model(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_FireSight(json_t *pStage, json_t *pStageModel, Model &model) {
    json_t *pFireSight = json_object();
    const char *errMsg = NULL;
    char version[100];
    snprintf(version, sizeof(version), "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    json_object_set(pFireSight, "version", json_string(version));
    json_object_set(pFireSight, "url", json_string("https://github.com/firepick1/FireSight"));
    snprintf(version, sizeof(version), "%d.%d.%d", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
    json_object_set(pFireSight, "opencv", json_string(version));
    json_object_set(pStageModel, "FireSight", pFireSight);

    return stageOK("apply_FireSight(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_meanStdDev(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    const char *errMsg = NULL;

	Scalar mean;
	Scalar stdDev;
	meanStdDev(model.image, mean, stdDev);

	json_t * jmean = json_array();
    json_object_set(pStageModel, "mean", jmean);
	json_t * jstddev = json_array();
    json_object_set(pStageModel, "stdDev", jstddev);
	for (int i = 0; i < 4; i++) {
		json_array_append(jmean, json_real(mean[i]));
		json_array_append(jstddev, json_real(stdDev[i]));
	}

    return stageOK("apply_meanStdDev(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_minAreaRect(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    const char *errMsg = NULL;
    vector<Point>  points;
    int channel = jo_int(pStage, "channel", 0, model.argMap);
    int minVal = jo_int(pStage, "min", 1, model.argMap);
    int maxVal = jo_int(pStage, "max", 255, model.argMap);
    int rows = model.image.rows;
    int cols = model.image.cols;
    json_t *pRects = json_array();
    json_object_set(pStageModel, "rects", pRects);

    const int channels = model.image.channels();
    LOGTRACE1("apply_minAreaRect() channels:%d", channels);
    switch(channels) {
    case 1: {
        for (int iRow = 0; iRow < rows; iRow++) {
            for (int iCol = 0; iCol < cols; iCol++) {
                uchar val = model.image.at<uchar>(iRow, iCol);
                if (minVal <= val && val <= maxVal) {
                    points.push_back(Point(iCol, iRow));
                }
            }
        }
        break;
    }
    case 3: {
        Mat_<Vec3b> image3b = model.image;
        for (int iRow = 0; iRow < rows; iRow++) {
            for (int iCol = 0; iCol < cols; iCol++) {
                uchar val = image3b(iRow, iCol)[channel];
                if (minVal <= val && val <= maxVal) {
                    points.push_back(Point(iCol, iRow));
                }
            }
        }
        break;
    }
    }

    LOGTRACE1("apply_minAreaRect() points found: %d", (int) points.size());
    json_object_set(pStageModel, "points", json_integer(points.size()));
    if (points.size() > 0) {
        RotatedRect rect = minAreaRect(points);
        json_t *pRect = json_object();
        json_object_set(pRect, "x", json_real(rect.center.x));
        json_object_set(pRect, "y", json_real(rect.center.y));
        json_object_set(pRect, "width", json_real(rect.size.width));
        json_object_set(pRect, "height", json_real(rect.size.height));
        json_object_set(pRect, "angle", json_real(rect.angle));
        json_array_append(pRects, pRect);
    }

    return stageOK("apply_minAreaRect(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_warpAffine(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    const char *errMsg = NULL;
    float scale = jo_float(pStage, "scale", 1, model.argMap);
    float angle = jo_float(pStage, "angle", 0, model.argMap);
    int dx = jo_int(pStage, "dx", (int)((scale-1)*model.image.cols/2.0+.5), model.argMap);
    int dy = jo_int(pStage, "dy", (int)((scale-1)*model.image.rows/2.0+.5), model.argMap);
    Point2f reflect = jo_Point2f(pStage, "reflect", Point(0,0), model.argMap);
    string  borderModeStr = jo_string(pStage, "borderMode", "BORDER_REPLICATE", model.argMap);
    int borderMode;

	if (reflect.x && reflect.y && reflect.x != reflect.y) {
		errMsg = "warpAffine only handles reflections around x- or y-axes";
	}

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

    if (scale <= 0) {
        errMsg = "Expected 0<scale";
    }
    int width = jo_int(pStage, "width", model.image.cols, model.argMap);
    int height = jo_int(pStage, "height", model.image.rows, model.argMap);
    int cx = jo_int(pStage, "cx", (int)(0.5+model.image.cols/2.0), model.argMap);
    int cy = jo_int(pStage, "cy", (int)(0.5+model.image.rows/2.0), model.argMap);
    Scalar borderValue = jo_Scalar(pStage, "borderValue", Scalar::all(0), model.argMap);

    if (!errMsg) {
        Mat result;
		int flags=cv::INTER_LINEAR || WARP_INVERSE_MAP;
        matWarpAffine(model.image, result, Point(cx,cy), angle, scale, Point(dx,dy), Size(width,height), 
			borderMode, borderValue, reflect, flags);
        model.image = result;
    }

    return stageOK("apply_warpAffine(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_warpPerspective(const char *pName, json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
	string errMsg;

    string  borderModeStr = jo_string(pStage, "borderMode", "BORDER_REPLICATE", model.argMap);
    int borderMode;

    string modelName = jo_string(pStage, "model", pName, model.argMap);
    vector<double> pm;
    json_t *pCalibrate;
    json_t *pCalibrateModel = json_object_get(model.getJson(false), modelName.c_str());
    if (json_is_object(pCalibrateModel)) {
        pCalibrate = json_object_get(pCalibrateModel, "calibrate");
        if (!json_is_object(pCalibrate)) {
            errMsg = "Expected \"calibrate\" JSON object in stage \"";
            errMsg += modelName;
            errMsg += "\"";
        }
    } else {
        pCalibrate = pStage;
    }
	pm = jo_vectord(pCalibrate, "perspective", vector<double>(), model.argMap);
	if (pm.size() == 0) {
		double default_vec_d[] = {1,0,0,0,1,0,0,0,1};
		vector<double> default_vecd(default_vec_d, default_vec_d + sizeof(default_vec_d) / sizeof(double) );
		pm = jo_vectord(pCalibrate, "matrix", default_vecd, model.argMap);
	}
	Mat matrix = Mat::zeros(3, 3, CV_64F);
	if (pm.size() == 9) {
		for (int i=0; i<pm.size(); i++) {
			matrix.at<double>(i/3, i%3) = pm[i];
		}
	} else {
		char buf[255];
		snprintf(buf, sizeof(buf), "apply_warpPerspective() invalid perspective matrix. Elements expected:9 actual:%d",
			(int) pm.size());
		LOGERROR1("%s", buf);
		errMsg = buf;
	}

    if (errMsg.empty()) {
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

    Scalar borderValue = jo_Scalar(pStage, "borderValue", Scalar::all(0), model.argMap);

    if (errMsg.empty()) {
        Mat result = Mat::zeros(model.image.rows, model.image.cols, model.image.type());
        warpPerspective(model.image, result, matrix, result.size(), cv::INTER_LINEAR, borderMode, borderValue );
        model.image = result;
    }

    return stageOK("apply_warpPerspective(%s) %s", errMsg.c_str(), pStage, pStageModel);
}

bool Pipeline::apply_putText(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    string text = jo_string(pStage, "text", "FireSight", model.argMap);
    Scalar color = jo_Scalar(pStage, "color", Scalar(0,255,0), model.argMap);
    string fontFaceName = jo_string(pStage, "fontFace", "FONT_HERSHEY_PLAIN", model.argMap);
    int thickness = jo_int(pStage, "thickness", 1, model.argMap);
    int fontFace = FONT_HERSHEY_PLAIN;
    bool italic = jo_bool(pStage, "italic", false, model.argMap);
    double fontScale = jo_double(pStage, "fontScale", 1, model.argMap);
    Point org = jo_Point(pStage, "org", Point(5,-6), model.argMap);
    if (org.y < 0) {
        org.y = model.image.rows + org.y;
    }
    const char *errMsg = NULL;

    if (fontFaceName.compare("FONT_HERSHEY_SIMPLEX") == 0) {
        fontFace = FONT_HERSHEY_SIMPLEX;
    } else if (fontFaceName.compare("FONT_HERSHEY_PLAIN") == 0) {
        fontFace = FONT_HERSHEY_PLAIN;
    } else if (fontFaceName.compare("FONT_HERSHEY_COMPLEX") == 0) {
        fontFace = FONT_HERSHEY_COMPLEX;
    } else if (fontFaceName.compare("FONT_HERSHEY_DUPLEX") == 0) {
        fontFace = FONT_HERSHEY_DUPLEX;
    } else if (fontFaceName.compare("FONT_HERSHEY_TRIPLEX") == 0) {
        fontFace = FONT_HERSHEY_TRIPLEX;
    } else if (fontFaceName.compare("FONT_HERSHEY_COMPLEX_SMALL") == 0) {
        fontFace = FONT_HERSHEY_COMPLEX_SMALL;
    } else if (fontFaceName.compare("FONT_HERSHEY_SCRIPT_SIMPLEX") == 0) {
        fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    } else if (fontFaceName.compare("FONT_HERSHEY_SCRIPT_COMPLEX") == 0) {
        fontFace = FONT_HERSHEY_SCRIPT_COMPLEX;
    } else {
        errMsg = "Unknown fontFace (default is FONT_HERSHEY_PLAIN)";
    }

    if (!errMsg && italic) {
        fontFace |= FONT_ITALIC;
    }

    if (!errMsg) {
        putText(model.image, text.c_str(), org, fontFace, fontScale, color, thickness);
    }

    return stageOK("apply_putText(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_resize(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    double fx = jo_float(pStage, "fx", 1, model.argMap);
    double fy = jo_float(pStage, "fy", 1, model.argMap);
    const char *errMsg = NULL;

    if (fx <= 0 || fy <= 0) {
        errMsg = "Expected 0<fx and 0<fy";
    }
    if (!errMsg) {
        Mat result;
        resize(model.image, result, Size(), fx, fy, INTER_AREA);
        model.image = result;
    }

    return stageOK("apply_resize(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_stageImage(json_t *pStage, json_t *pStageModel, Model &model) {
    string stageStr = jo_string(pStage, "stage", "input", model.argMap);
    const char *errMsg = NULL;

    if (stageStr.empty()) {
        errMsg = "Expected name of stage for image";
    } else {
        model.image = model.imageMap[stageStr.c_str()];
        if (!model.image.rows || !model.image.cols) {
            model.image = model.imageMap["input"].clone();
            LOGTRACE1("Could not locate stage image '%s', using input image", stageStr.c_str());
        }
    }

    return stageOK("apply_stageImage(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_imread(json_t *pStage, json_t *pStageModel, Model &model) {
    string path = jo_string(pStage, "path", "", model.argMap);
    const char *errMsg = NULL;

    if (path.empty()) {
        errMsg = "expected path for imread";
    } else {
        model.image = imread(path.c_str(), CV_LOAD_IMAGE_COLOR);
        if (model.image.data) {
            json_object_set(pStageModel, "rows", json_integer(model.image.rows));
            json_object_set(pStageModel, "cols", json_integer(model.image.cols));
        } else {
            LOGERROR1("imread(%s) failed", path.c_str());
            errMsg = "apply_imread() failed";
        }
    }

    return stageOK("apply_imread(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_imwrite(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    string path = jo_string(pStage, "path");
    const char *errMsg = NULL;

    if (path.empty()) {
        errMsg = "Expected path for imwrite";
    } else {
        bool result = imwrite(path.c_str(), model.image);
        json_object_set(pStageModel, "result", json_boolean(result));
    }

    return stageOK("apply_imwrite(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_cvtColor(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    string codeStr = jo_string(pStage, "code", "CV_BGR2GRAY", model.argMap);
    int dstCn = jo_int(pStage, "dstCn", 0, model.argMap);
    const char *errMsg = NULL;
    int code = CV_BGR2GRAY;

    if (codeStr.compare("CV_BGR2BGRA")==0) {
        code = CV_BGR2BGRA;
    } else if (codeStr.compare("CV_RGB2RGBA")==0) {
        code = CV_RGB2RGBA;
    } else if (codeStr.compare("CV_BGRA2BGR")==0) {
        code = CV_BGRA2BGR;
    } else if (codeStr.compare("CV_RGBA2RGB")==0) {
        code = CV_RGBA2RGB;
    } else if (codeStr.compare("CV_BGR2RGBA")==0) {
        code = CV_BGR2RGBA;
    } else if (codeStr.compare("CV_RGB2BGRA")==0) {
        code = CV_RGB2BGRA;
    } else if (codeStr.compare("CV_RGBA2BGR")==0) {
        code = CV_RGBA2BGR;
    } else if (codeStr.compare("CV_BGRA2RGB")==0) {
        code = CV_BGRA2RGB;
    } else if (codeStr.compare("CV_BGR2RGB")==0) {
        code = CV_BGR2RGB;
    } else if (codeStr.compare("CV_RGB2BGR")==0) {
        code = CV_RGB2BGR;
    } else if (codeStr.compare("CV_BGRA2RGBA")==0) {
        code = CV_BGRA2RGBA;
    } else if (codeStr.compare("CV_RGBA2BGRA")==0) {
        code = CV_RGBA2BGRA;
    } else if (codeStr.compare("CV_BGR2GRAY")==0) {
        code = CV_BGR2GRAY;
    } else if (codeStr.compare("CV_RGB2GRAY")==0) {
        code = CV_RGB2GRAY;
    } else if (codeStr.compare("CV_GRAY2BGR")==0) {
        code = CV_GRAY2BGR;
    } else if (codeStr.compare("CV_GRAY2RGB")==0) {
        code = CV_GRAY2RGB;
    } else if (codeStr.compare("CV_GRAY2BGRA")==0) {
        code = CV_GRAY2BGRA;
    } else if (codeStr.compare("CV_GRAY2RGBA")==0) {
        code = CV_GRAY2RGBA;
    } else if (codeStr.compare("CV_BGRA2GRAY")==0) {
        code = CV_BGRA2GRAY;
    } else if (codeStr.compare("CV_RGBA2GRAY")==0) {
        code = CV_RGBA2GRAY;
    } else if (codeStr.compare("CV_BGR2BGR565")==0) {
        code = CV_BGR2BGR565;
    } else if (codeStr.compare("CV_RGB2BGR565")==0) {
        code = CV_RGB2BGR565;
    } else if (codeStr.compare("CV_BGR5652BGR")==0) {
        code = CV_BGR5652BGR;
    } else if (codeStr.compare("CV_BGR5652RGB")==0) {
        code = CV_BGR5652RGB;
    } else if (codeStr.compare("CV_BGRA2BGR565")==0) {
        code = CV_BGRA2BGR565;
    } else if (codeStr.compare("CV_RGBA2BGR565")==0) {
        code = CV_RGBA2BGR565;
    } else if (codeStr.compare("CV_BGR5652BGRA")==0) {
        code = CV_BGR5652BGRA;
    } else if (codeStr.compare("CV_BGR5652RGBA")==0) {
        code = CV_BGR5652RGBA;
    } else if (codeStr.compare("CV_GRAY2BGR565")==0) {
        code = CV_GRAY2BGR565;
    } else if (codeStr.compare("CV_BGR5652GRAY")==0) {
        code = CV_BGR5652GRAY;
    } else if (codeStr.compare("CV_BGR2BGR555")==0) {
        code = CV_BGR2BGR555;
    } else if (codeStr.compare("CV_RGB2BGR555")==0) {
        code = CV_RGB2BGR555;
    } else if (codeStr.compare("CV_BGR5552BGR")==0) {
        code = CV_BGR5552BGR;
    } else if (codeStr.compare("CV_BGR5552RGB")==0) {
        code = CV_BGR5552RGB;
    } else if (codeStr.compare("CV_BGRA2BGR555")==0) {
        code = CV_BGRA2BGR555;
    } else if (codeStr.compare("CV_RGBA2BGR555")==0) {
        code = CV_RGBA2BGR555;
    } else if (codeStr.compare("CV_BGR5552BGRA")==0) {
        code = CV_BGR5552BGRA;
    } else if (codeStr.compare("CV_BGR5552RGBA")==0) {
        code = CV_BGR5552RGBA;
    } else if (codeStr.compare("CV_GRAY2BGR555")==0) {
        code = CV_GRAY2BGR555;
    } else if (codeStr.compare("CV_BGR5552GRAY")==0) {
        code = CV_BGR5552GRAY;
    } else if (codeStr.compare("CV_BGR2XYZ")==0) {
        code = CV_BGR2XYZ;
    } else if (codeStr.compare("CV_RGB2XYZ")==0) {
        code = CV_RGB2XYZ;
    } else if (codeStr.compare("CV_XYZ2BGR")==0) {
        code = CV_XYZ2BGR;
    } else if (codeStr.compare("CV_XYZ2RGB")==0) {
        code = CV_XYZ2RGB;
    } else if (codeStr.compare("CV_BGR2YCrCb")==0) {
        code = CV_BGR2YCrCb;
    } else if (codeStr.compare("CV_RGB2YCrCb")==0) {
        code = CV_RGB2YCrCb;
    } else if (codeStr.compare("CV_YCrCb2BGR")==0) {
        code = CV_YCrCb2BGR;
    } else if (codeStr.compare("CV_YCrCb2RGB")==0) {
        code = CV_YCrCb2RGB;
    } else if (codeStr.compare("CV_BGR2HSV")==0) {
        code = CV_BGR2HSV;
    } else if (codeStr.compare("CV_RGB2HSV")==0) {
        code = CV_RGB2HSV;
    } else if (codeStr.compare("CV_BGR2Lab")==0) {
        code = CV_BGR2Lab;
    } else if (codeStr.compare("CV_RGB2Lab")==0) {
        code = CV_RGB2Lab;
    } else if (codeStr.compare("CV_BayerBG2BGR")==0) {
        code = CV_BayerBG2BGR;
    } else if (codeStr.compare("CV_BayerGB2BGR")==0) {
        code = CV_BayerGB2BGR;
    } else if (codeStr.compare("CV_BayerRG2BGR")==0) {
        code = CV_BayerRG2BGR;
    } else if (codeStr.compare("CV_BayerGR2BGR")==0) {
        code = CV_BayerGR2BGR;
    } else if (codeStr.compare("CV_BayerBG2RGB")==0) {
        code = CV_BayerBG2RGB;
    } else if (codeStr.compare("CV_BayerGB2RGB")==0) {
        code = CV_BayerGB2RGB;
    } else if (codeStr.compare("CV_BayerRG2RGB")==0) {
        code = CV_BayerRG2RGB;
    } else if (codeStr.compare("CV_BayerGR2RGB")==0) {
        code = CV_BayerGR2RGB;
    } else if (codeStr.compare("CV_BGR2Luv")==0) {
        code = CV_BGR2Luv;
    } else if (codeStr.compare("CV_RGB2Luv")==0) {
        code = CV_RGB2Luv;
    } else if (codeStr.compare("CV_BGR2HLS")==0) {
        code = CV_BGR2HLS;
    } else if (codeStr.compare("CV_RGB2HLS")==0) {
        code = CV_RGB2HLS;
    } else if (codeStr.compare("CV_HSV2BGR")==0) {
        code = CV_HSV2BGR;
    } else if (codeStr.compare("CV_HSV2RGB")==0) {
        code = CV_HSV2RGB;
    } else if (codeStr.compare("CV_Lab2BGR")==0) {
        code = CV_Lab2BGR;
    } else if (codeStr.compare("CV_Lab2RGB")==0) {
        code = CV_Lab2RGB;
    } else if (codeStr.compare("CV_Luv2BGR")==0) {
        code = CV_Luv2BGR;
    } else if (codeStr.compare("CV_Luv2RGB")==0) {
        code = CV_Luv2RGB;
    } else if (codeStr.compare("CV_HLS2BGR")==0) {
        code = CV_HLS2BGR;
    } else if (codeStr.compare("CV_HLS2RGB")==0) {
        code = CV_HLS2RGB;
    } else if (codeStr.compare("CV_BayerBG2BGR_VNG")==0) {
        code = CV_BayerBG2BGR_VNG;
    } else if (codeStr.compare("CV_BayerGB2BGR_VNG")==0) {
        code = CV_BayerGB2BGR_VNG;
    } else if (codeStr.compare("CV_BayerRG2BGR_VNG")==0) {
        code = CV_BayerRG2BGR_VNG;
    } else if (codeStr.compare("CV_BayerGR2BGR_VNG")==0) {
        code = CV_BayerGR2BGR_VNG;
    } else if (codeStr.compare("CV_BayerBG2RGB_VNG")==0) {
        code = CV_BayerBG2RGB_VNG;
    } else if (codeStr.compare("CV_BayerGB2RGB_VNG")==0) {
        code = CV_BayerGB2RGB_VNG;
    } else if (codeStr.compare("CV_BayerRG2RGB_VNG")==0) {
        code = CV_BayerRG2RGB_VNG;
    } else if (codeStr.compare("CV_BayerGR2RGB_VNG")==0) {
        code = CV_BayerGR2RGB_VNG;
    } else if (codeStr.compare("CV_BGR2HSV_FULL")==0) {
        code = CV_BGR2HSV_FULL;
    } else if (codeStr.compare("CV_RGB2HSV_FULL")==0) {
        code = CV_RGB2HSV_FULL;
    } else if (codeStr.compare("CV_BGR2HLS_FULL")==0) {
        code = CV_BGR2HLS_FULL;
    } else if (codeStr.compare("CV_RGB2HLS_FULL")==0) {
        code = CV_RGB2HLS_FULL;
    } else if (codeStr.compare("CV_HSV2BGR_FULL")==0) {
        code = CV_HSV2BGR_FULL;
    } else if (codeStr.compare("CV_HSV2RGB_FULL")==0) {
        code = CV_HSV2RGB_FULL;
    } else if (codeStr.compare("CV_HLS2BGR_FULL")==0) {
        code = CV_HLS2BGR_FULL;
    } else if (codeStr.compare("CV_HLS2RGB_FULL")==0) {
        code = CV_HLS2RGB_FULL;
    } else if (codeStr.compare("CV_LBGR2Lab")==0) {
        code = CV_LBGR2Lab;
    } else if (codeStr.compare("CV_LRGB2Lab")==0) {
        code = CV_LRGB2Lab;
    } else if (codeStr.compare("CV_LBGR2Luv")==0) {
        code = CV_LBGR2Luv;
    } else if (codeStr.compare("CV_LRGB2Luv")==0) {
        code = CV_LRGB2Luv;
    } else if (codeStr.compare("CV_Lab2LBGR")==0) {
        code = CV_Lab2LBGR;
    } else if (codeStr.compare("CV_Lab2LRGB")==0) {
        code = CV_Lab2LRGB;
    } else if (codeStr.compare("CV_Luv2LBGR")==0) {
        code = CV_Luv2LBGR;
    } else if (codeStr.compare("CV_Luv2LRGB")==0) {
        code = CV_Luv2LRGB;
    } else if (codeStr.compare("CV_BGR2YUV")==0) {
        code = CV_BGR2YUV;
    } else if (codeStr.compare("CV_RGB2YUV")==0) {
        code = CV_RGB2YUV;
    } else if (codeStr.compare("CV_YUV2BGR")==0) {
        code = CV_YUV2BGR;
    } else if (codeStr.compare("CV_YUV2RGB")==0) {
        code = CV_YUV2RGB;
    } else if (codeStr.compare("CV_BayerBG2GRAY")==0) {
        code = CV_BayerBG2GRAY;
    } else if (codeStr.compare("CV_BayerGB2GRAY")==0) {
        code = CV_BayerGB2GRAY;
    } else if (codeStr.compare("CV_BayerRG2GRAY")==0) {
        code = CV_BayerRG2GRAY;
    } else if (codeStr.compare("CV_BayerGR2GRAY")==0) {
        code = CV_BayerGR2GRAY;
#ifdef CV_YUV420i2RGB
    } else if (codeStr.compare("CV_YUV420i2RGB")==0) {
        code = CV_YUV420i2RGB;
    } else if (codeStr.compare("CV_YUV420i2BGR")==0) {
        code = CV_YUV420i2BGR;
#endif
    } else if (codeStr.compare("CV_YUV420sp2RGB")==0) {
        code = CV_YUV420sp2RGB;
    } else if (codeStr.compare("CV_YUV420sp2BGR")==0) {
        code = CV_YUV420sp2BGR;
    } else {
        errMsg = "Unknown cvtColor conversion code";
    }
    if (dstCn < 0) {
        errMsg = "expected 0<dstCn";
    }

    if (!errMsg) {
        cvtColor(model.image, model.image, code, dstCn);
    }

    return stageOK("apply_cvtColor(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_points2resolution_RANSAC(json_t *pStage, json_t *pStageModel, Model &model) {

    char *errMsg = NULL;
    // input parameters
    double thr1 = jo_double(pStage, "threshold1", 1.2, model.argMap);
    double thr2 = jo_double(pStage, "threshold2", 1.2, model.argMap);
    double confidence = jo_double(pStage, "confidence", (1.0-1e-12), model.argMap);
    double separation = jo_double(pStage, "separation", 4.0, model.argMap); // separation [mm]

    string pointsModelName = jo_string(pStage, "model", "", model.argMap);
    json_t *pPointsModel = json_object_get(model.getJson(false), pointsModelName.c_str());

    try {
        if (pointsModelName.empty()) {
            throw runtime_error("model: expected name of stage with points");
        } else if (!json_is_object(pPointsModel)) {
            throw runtime_error("Named stage is not in model");
        }

        json_t *pPoints = NULL;
        do {
            pPoints = json_object_get(pPointsModel, "circles");
            if (json_is_array(pPoints))
                break;

            pPoints = json_object_get(pPointsModel, "points");
            if (json_is_array(pPoints))
                break;

            throw runtime_error("Expected array of points (circles, ...)");
        } while (0);

        size_t index;
        json_t *pPoint;

        vector<XY> coords;

        json_array_foreach(pPoints, index, pPoint) {
            double x = jo_double(pPoint, "x", DBL_MAX, model.argMap);
            double y = jo_double(pPoint, "y", DBL_MAX, model.argMap);
            //double r = jo_double(pCircle, "radius", DBL_MAX, model.argMap);

            if (x == DBL_MAX || y == DBL_MAX) {
                LOGERROR("apply_points2resolution_RANSAC() x, y are required values (skipping)");
                continue;
            }

            XY xy;
            xy.x = x;
            xy.y = y;
            coords.push_back(xy);
        }

        Pt2Res pt2res;

        double resolution;

        try {
            resolution = pt2res.getResolution(thr1, thr2, confidence, separation, coords);
            json_object_set(pStageModel, "resolution", json_real(resolution));
        } catch (runtime_error &e) {
            errMsg = (char *) malloc(sizeof(char) * (strlen(e.what()) + 1));
            strcpy(errMsg, e.what());
        }

    } catch (exception &e) {
        errMsg = (char *) malloc(sizeof(char) * (strlen(e.what()) + 1));
        strcpy(errMsg, e.what());
    }

    return stageOK("apply_points2resolution_RANSAC(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_crop(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    string errMsg;

    int x = jo_int(pStage, "x", 0, model.argMap);
    int y = jo_int(pStage, "y", 0, model.argMap);
    int defaultWidth = model.image.cols ? model.image.cols-x : 64;
    int defaultHeight = model.image.rows ? model.image.rows-y : 64;
    int width = jo_int(pStage, "width", defaultWidth, model.argMap);
    int height = jo_int(pStage, "height", defaultHeight, model.argMap);

    if (errMsg.empty() && x < 0) {
        errMsg = "crop() x cannot be negative";
    }
    if (errMsg.empty() && y < 0) {
        errMsg = "crop() y cannot be negative";
    }
    if (errMsg.empty() && model.image.cols && x+width > model.image.cols) {
        errMsg = "crop() cropped area extends beyond image right edge";
    }
    if (errMsg.empty() && model.image.rows && y+height > model.image.rows) {
        errMsg = "crop() cropped area extends beyond image bottom edge";
    }

    if (logLevel >= FIRELOG_TRACE) {
        char *pStageJson = json_dumps(pStage, 0);
        LOGTRACE1("apply_crop(%s)", pStageJson);
        free(pStageJson);
    }

    if (errMsg.empty()) {
        model.image = model.image(Rect(x, y, width, height)).clone();
    } else {
        LOGERROR4("crop(%d,%d,%d,%d)", x, y, width, height);
        LOGERROR2("image(%d,%d)", model.image.cols, model.image.rows);
    }

    return stageOK("apply_crop(%s) %s", errMsg.c_str(), pStage, pStageModel);
}

#ifdef LGPL2_1
bool Pipeline::apply_qrdecode(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    char *errMsg = NULL;

    int show = jo_int(pStage, "show", 0, model.argMap);

    if (logLevel >= FIRELOG_TRACE) {
        char *pStageJson = json_dumps(pStage, 0);
        LOGTRACE1("apply_qrdecode(%s)", pStageJson);
        free(pStageJson);
    }

    try {
        ZbarQrDecode qr;
        vector<QRPayload> payload = qr.scan(model.image, show);

        json_t *payload_json = json_array();
        json_object_set(pStageModel, "qrdata", payload_json);
        for (size_t i = 0; i < payload.size(); i++) {
            json_array_append(payload_json, payload[i].as_json_t());
        }


    } catch (runtime_error &e) {
        errMsg = (char *) malloc(sizeof(char) * (strlen(e.what())+1));
        strcpy(errMsg, e.what());
    }

    return stageOK("apply_qrdecode(%s) %s", errMsg, pStage, pStageModel);
}
#endif // LGPL2_1

bool Pipeline::apply_drawRects(json_t *pStage, json_t *pStageModel, Model &model) {
    const char *errMsg = NULL;
    Scalar color = jo_Scalar(pStage, "color", Scalar(-1,-1,-1,255), model.argMap);
    int radius = jo_int(pStage, "radius", 0, model.argMap);
    int thickness = jo_int(pStage, "thickness", 2, model.argMap);
    string rectsModelName = jo_string(pStage, "model", "", model.argMap);
    json_t *pRectsModel = json_object_get(model.getJson(false), rectsModelName.c_str());

    if (rectsModelName.empty()) {
        errMsg = "model: expected name of stage with rects";
    } else if (!json_is_object(pRectsModel)) {
        errMsg = "Named stage is not in model";
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
        size_t index;
        json_t *pRect;
        Point2f vertices[4];
        int blue = (int)color[0];
        int green = (int)color[1];
        int red = (int)color[2];
        bool changeColor = red == -1 && green == -1 && blue == -1;

        json_array_foreach(pRects, index, pRect) {
            int x = jo_int(pRect, "x", SHRT_MAX, model.argMap);
            int y = jo_int(pRect, "y", SHRT_MAX, model.argMap);
            int width = jo_int(pRect, "width", -1, model.argMap);
            int height = jo_int(pRect, "height", -1, model.argMap);
            float angle = jo_float(pRect, "angle", FLT_MAX, model.argMap);
            Scalar rectColor = color;
            if (changeColor) {
                red = (index & 1) ? 0 : 255;
                green = (index & 2) ? 128 : 192;
                blue = (index & 1) ? 255 : 0;
                rectColor = Scalar(blue, green, red, 255);
            }
            rectColor = jo_Scalar(pRect, "color", rectColor, model.argMap);
            if (x == SHRT_MAX || y == SHRT_MAX || width == SHRT_MAX || height == SHRT_MAX) {
                LOGERROR("apply_drawRects() x, y, width, height are required values");
                break;
            }
            if (rectColor[3] != 0) {	// alpha=0 implies non-display
                if (angle == FLT_MAX || radius > 0) {
                    int r;
                    if (radius > 0) {
                        r = radius;
                    } else if (width > 0 && height > 0) {
                        r = (int)(0.5+min(width,height)/2.0);
                    } else {
                        r = 5;
                    }
                    circle(model.image, Point(x,y), r, rectColor, thickness);
                } else {
                    RotatedRect rect(Point(x,y), Size(width, height), angle);
                    rect.points(vertices);
                    for (int i = 0; i < 4; i++) {
                        line(model.image, vertices[i], vertices[(i+1)%4], rectColor, thickness);
                    }
                }
            }
        }
    }

    return stageOK("apply_drawRects(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_drawKeypoints(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    const char *errMsg = NULL;
    Scalar color = jo_Scalar(pStage, "color", Scalar::all(-1), model.argMap);
    int flags = jo_int(pStage, "flags", DrawMatchesFlags::DRAW_OVER_OUTIMG|DrawMatchesFlags::DRAW_RICH_KEYPOINTS, model.argMap);
    string modelName = jo_string(pStage, "model", "", model.argMap);
    json_t *pKeypointStage = jo_object(model.getJson(false), modelName.c_str(), model.argMap);

    if (!pKeypointStage) {
        string keypointStageName = jo_string(pStage, "keypointStage", "", model.argMap);
        pKeypointStage = jo_object(model.getJson(false), keypointStageName.c_str(), model.argMap);
    }

    if (!errMsg && flags < 0 || 7 < flags) {
        errMsg = "expected 0 < flags < 7";
    }

    if (!errMsg && !pKeypointStage) {
        errMsg = "expected name of stage model";
    }

    vector<KeyPoint> keypoints;
    if (!errMsg) {
        json_t *pKeypoints = jo_object(pKeypointStage, "keypoints", model.argMap);
        if (!json_is_array(pKeypoints)) {
            errMsg = "keypointStage has no keypoints JSON array";
        } else {
            size_t index;
            json_t *pKeypoint;
            json_array_foreach(pKeypoints, index, pKeypoint) {
                float x = jo_float(pKeypoint, "pt.x", -1, model.argMap);
                float y = jo_float(pKeypoint, "pt.y", -1, model.argMap);
                float size = jo_float(pKeypoint, "size", 10, model.argMap);
                float angle = jo_float(pKeypoint, "angle", -1, model.argMap);
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
    int width = jo_int(pStage, "ksize.width", 3, model.argMap);
    int height = jo_int(pStage, "ksize.height", 3, model.argMap);
    int anchorx = jo_int(pStage, "anchor.x", -1, model.argMap);
    int anchory = jo_int(pStage, "anchor.y", -1, model.argMap);

    if (width <= 0 || height <= 0) {
        errMsg = "expected 0<width and 0<height";
    }

    if (!errMsg) {
        blur(model.image, model.image, Size(width,height));
    }

    return stageOK("apply_blur(%s) %s", errMsg, pStage, pStageModel);
}

static void modelKeyPoints(json_t*pStageModel, const vector<KeyPoint> &keyPoints) {
    json_t *pKeyPoints = json_array();
    json_object_set(pStageModel, "keypoints", pKeyPoints);
    for (size_t i=0; i<keyPoints.size(); i++) {
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
    params.thresholdStep = jo_float(pStage, "thresholdStep", params.thresholdStep, model.argMap);
    params.minThreshold = jo_float(pStage, "minThreshold", params.minThreshold, model.argMap);
    params.maxThreshold = jo_float(pStage, "maxThreshold", params.maxThreshold, model.argMap);
    params.minRepeatability = jo_int(pStage, "minRepeatability", params.minRepeatability, model.argMap);
    params.minDistBetweenBlobs = jo_float(pStage, "minDistBetweenBlobs", params.minDistBetweenBlobs, model.argMap);
    params.filterByColor = jo_bool(pStage, "filterByColor", params.filterByColor);
    params.blobColor = jo_int(pStage, "blobColor", params.blobColor, model.argMap);
    params.filterByArea = jo_bool(pStage, "filterByArea", params.filterByArea);
    params.minArea = jo_float(pStage, "minArea", params.minArea, model.argMap);
    params.maxArea = jo_float(pStage, "maxArea", params.maxArea, model.argMap);
    params.filterByCircularity = jo_bool(pStage, "filterByCircularity", params.filterByCircularity);
    params.minCircularity = jo_float(pStage, "minCircularity", params.minCircularity, model.argMap);
    params.maxCircularity = jo_float(pStage, "maxCircularity", params.maxCircularity, model.argMap);
    params.filterByInertia = jo_bool(pStage, "filterByInertia", params.filterByInertia, model.argMap);
    params.minInertiaRatio = jo_float(pStage, "minInertiaRatio", params.minInertiaRatio, model.argMap);
    params.maxInertiaRatio = jo_float(pStage, "maxInertiaRatio", params.maxInertiaRatio, model.argMap);
    params.filterByConvexity = jo_bool(pStage, "filterByConvexity", params.filterByConvexity);
    params.minConvexity = jo_float(pStage, "minConvexity", params.minConvexity, model.argMap);
    params.maxConvexity = jo_float(pStage, "maxConvexity", params.maxConvexity, model.argMap);
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

bool Pipeline::apply_circle(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    Point center = jo_Point(pStage, "center", Point(0,0), model.argMap);
    int radius = jo_int(pStage, "radius", 0, model.argMap);
    Scalar color = jo_Scalar(pStage, "color", Scalar::all(0), model.argMap);
    int thickness = jo_int(pStage, "thickness", 1, model.argMap);
    int lineType = jo_int(pStage, "lineType", 8, model.argMap);
    Scalar fill = jo_Scalar(pStage, "fill", Scalar::all(-1), model.argMap);
    int shift = jo_int(pStage, "shift", 0, model.argMap);
    string errMsg;

    if (shift < 0) {
        errMsg = "Expected shift>=0";
    }

    if (errMsg.empty()) {
        if (thickness) {
            circle(model.image, center, radius, color, thickness, lineType, shift);

        }
        if (thickness >= 0) {
            int outThickness = thickness/2;
            int inThickness = (int)(thickness - outThickness);
            if (fill[0] >= 0) {
                circle(model.image, center, radius-inThickness, fill, -1, lineType, shift);
            }
        }
    }

    return stageOK("apply_circle(%s) %s", errMsg.c_str(), pStage, pStageModel);
}

bool Pipeline::apply_sharpness(json_t *pStage, json_t *pStageModel, Model &model) {
    const char *errMsg = NULL;
    string methodStr = jo_string(pStage, "method", "GRAS", model.argMap);
    
    /* Apply selected method */
    double sharpness = 0;
    if (strcmp("GRAS", methodStr.c_str()) == 0) {
        sharpness = Sharpness::GRAS(model.image);
    } else if (strcmp("LAPE", methodStr.c_str()) == 0) {
        sharpness = Sharpness::LAPE(model.image);
    } else if (strcmp("LAPM", methodStr.c_str()) == 0) {
        sharpness = Sharpness::LAPM(model.image);
    }

    json_object_set(pStageModel, "sharpness", json_real(sharpness));

    return stageOK("apply_sharpness(%s) %s", errMsg, pStage, pStageModel);

}

bool Pipeline::apply_rectangle(json_t *pStage, json_t *pStageModel, Model &model) {
    int x = jo_int(pStage, "x", 0, model.argMap);
    int y = jo_int(pStage, "y", 0, model.argMap);
    int defaultWidth = model.image.cols ? model.image.cols : 64;
    int defaultHeight = model.image.rows ? model.image.rows : 64;
    int width = jo_int(pStage, "width", defaultWidth, model.argMap);
    int height = jo_int(pStage, "height", defaultHeight, model.argMap);
    int thickness = jo_int(pStage, "thickness", 1, model.argMap);
    int lineType = jo_int(pStage, "lineType", 8, model.argMap);
    Scalar color = jo_Scalar(pStage, "color", Scalar::all(0), model.argMap);
    Scalar flood = jo_Scalar(pStage, "flood", Scalar::all(-1), model.argMap);
    Scalar fill = jo_Scalar(pStage, "fill", Scalar::all(-1), model.argMap);
    int shift = jo_int(pStage, "shift", 0, model.argMap);
    const char *errMsg = NULL;

    if ( x == -1 ) {
        x = (model.image.cols-width)/2;
    }
    if ( y == -1 ) {
        y = (model.image.rows-height)/2;
    }
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
            int outThickness = thickness/2;
            int inThickness = (int)(thickness - outThickness);
            if (fill[0] >= 0) {
                rectangle(model.image, Rect(x+inThickness,y+inThickness,width-inThickness*2,height-inThickness*2), fill, -1, lineType, shift);
            }
            if (flood[0] >= 0) {
                int left = x - outThickness;
                int top = y - outThickness;
                int right = x+width+outThickness;
                int bot = y+height+outThickness;
                rectangle(model.image, Rect(0,0,model.image.cols,top), flood, -1, lineType, shift);
                rectangle(model.image, Rect(0,bot,model.image.cols,model.image.rows-bot), flood, -1, lineType, shift);
                rectangle(model.image, Rect(0,top,left,height+outThickness*2), flood, -1, lineType, shift);
                rectangle(model.image, Rect(right,top,model.image.cols-right,height+outThickness*2), flood, -1, lineType, shift);
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
    int width = jo_int(pStage, "width", model.image.cols, model.argMap);
    int height = jo_int(pStage, "height", model.image.rows, model.argMap);
    string typeStr = jo_string(pStage, "type", "CV_8UC3", model.argMap);
    Scalar color = jo_Scalar(pStage, "color", Scalar::all(0), model.argMap);
    const char *errMsg = NULL;
    int type = CV_8UC3;

    if (width <= 0 || height <= 0) {
        errMsg = "Expected 0<width and 0<height";
    } else if (color[0] <0 || color[1]<0 || color[2]<0) {
        errMsg = "Expected color JSON array with non-negative values";
    }

    if (!errMsg) {
        type = parseCvType(typeStr.c_str(), errMsg);
    }

    if (!errMsg) {
        model.image = Mat(height, width, type, color);
    }

    return stageOK("apply_Mat(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_split(json_t *pStage, json_t *pStageModel, Model &model) {
    json_t *pFromTo = jo_object(pStage, "fromTo", model.argMap);
    const char *errMsg = NULL;
#define MAX_FROMTO 32
    int fromTo[MAX_FROMTO];
    int nFromTo;

    if (!json_is_array(pFromTo)) {
        errMsg = "Expected JSON array for fromTo";
    }

    if (!errMsg) {
        json_t *pInt;
        size_t index;
        json_array_foreach(pFromTo, index, pInt) {
            if (index >= MAX_FROMTO) {
                errMsg = "Too many channels";
                break;
            }
            nFromTo = index+1;
            fromTo[index] = (int)json_integer_value(pInt);
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
    double alpha = jo_float(pStage, "alpha", 1, model.argMap);
    double delta = jo_float(pStage, "delta", 0, model.argMap);
    string transform = jo_string(pStage, "transform", "", model.argMap);
    string rTypeStr = jo_string(pStage, "rType", "CV_8U", model.argMap);
    const char *errMsg = NULL;
    int rType;

    if (!errMsg) {
        rType = parseCvType(rTypeStr.c_str(), errMsg);
    }

    if (!transform.empty()) {
        if (transform.compare("log") == 0) {
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
    int col = jo_int(pStage, "col", 0, model.argMap);
    int row = jo_int(pStage, "row", 0, model.argMap);
    int cols = jo_int(pStage, "cols", model.image.cols, model.argMap);
    int rows = jo_int(pStage, "rows", model.image.rows, model.argMap);
    int precision = jo_int(pStage, "precision", 1, model.argMap);
    int width = jo_int(pStage, "width", 5, model.argMap);
    int channel = jo_int(pStage, "channel", 0, model.argMap);
    string comment = jo_string(pStage, "comment", "", model.argMap);
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
        if (comment.size()) {
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

bool Pipeline::apply_PSNR(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    string path = jo_string(pStage, "path", "", model.argMap);
    string psnrSame = jo_string(pStage, "psnrSame", "SAME", model.argMap);
    double threshold = jo_double(pStage, "threshold", -1, model.argMap);
    const char *errMsg = NULL;
    Mat thatImage;

    if (path.empty()) {
        errMsg = "apply_PSNR() expected path for imread";
    } else {
        thatImage = imread(path.c_str(), CV_LOAD_IMAGE_COLOR);
        LOGTRACE2("apply_PSNR(%s) %s", path.c_str(), matInfo(thatImage).c_str());
        if (thatImage.data) {
            assert(model.image.cols == thatImage.cols);
            assert(model.image.rows == thatImage.rows);
            assert(model.image.channels() == thatImage.channels());
        } else {
            errMsg = "apply_PSNR() imread failed";
        }
    }

    if (!errMsg) {
        Mat s1;
        absdiff(model.image, thatImage, s1);  // |I1 - I2|
        s1.convertTo(s1, CV_32F);              // cannot make a square on 8 bits
        s1 = s1.mul(s1);                       // |I1 - I2|^2
        Scalar s = sum(s1);                   // sum elements per channel
        double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

#define SSE_THRESHOLD 1e-10
        if( sse > 1e-10) {
            double  mse =sse /(double)(model.image.channels() * model.image.total());
            double psnr = 10.0*log10((255*255)/mse);
            json_object_set(pStageModel, "PSNR", json_real(psnr));
            if (threshold >= 0) {
                if (psnr >= threshold) {
                    LOGTRACE2("apply_PSNR() threshold passed: %f >= %f", psnr, threshold);
                    json_object_set(pStageModel, "PSNR", json_string(psnrSame.c_str()));
                } else {
                    LOGTRACE2("apply_PSNR() threshold failed: %f < %f", psnr, threshold);
                }
            }
        } else if (sse == 0) {
            LOGTRACE("apply_PSNR() identical images: SSE == 0");
            json_object_set(pStageModel, "PSNR", json_string(psnrSame.c_str()));
        } else {
            LOGTRACE2("apply_PSNR() threshold passed: SSE %f < %f", sse, SSE_THRESHOLD);
            json_object_set(pStageModel, "PSNR", json_string(psnrSame.c_str()));
        }
    }

    return stageOK("apply_PSNR(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_Canny(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    float threshold1 = jo_float(pStage, "threshold1", 0, model.argMap);
    float threshold2 = jo_float(pStage, "threshold2", 50, model.argMap);
    int apertureSize = jo_int(pStage, "apertureSize", 3, model.argMap);
    bool L2gradient = jo_bool(pStage, "L2gradient", false);
    const char *errMsg = NULL;

    if (!errMsg) {
        Canny(model.image, model.image, threshold1, threshold2, apertureSize, L2gradient);
    }

    return stageOK("apply_Canny(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_absdiff(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    string img2_path = jo_string(pStage, "path", "", model.argMap);
    const char *errMsg = NULL;
    Mat img2;

    if (img2_path.empty()) {
        errMsg = "Expected path to image for absdiff";
    }

    if (!errMsg) {
        if (model.image.channels() == 1) {
            img2 = imread(img2_path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        } else {
            img2 = imread(img2_path.c_str(), CV_LOAD_IMAGE_COLOR);
        }
        if (img2.data) {
            LOGTRACE2("apply_absdiff() path:%s %s", img2_path.c_str(), matInfo(img2).c_str());
        } else {
            errMsg = "Could not read image from given path";
        }
    }

    if (!errMsg) {
        absdiff(model.image, img2, model.image);
    }

    return stageOK("apply_absdiff(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_threshold(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    string typeStr = jo_string(pStage, "type", "THRESH_BINARY", model.argMap);
    float maxval = jo_float(pStage, "maxval", 255, model.argMap);
    String otsu = jo_string(pStage, "thresh", "OTSU", model.argMap);
    bool isOtsu = otsu.compare("OTSU") == 0;
    float thresh = jo_float(pStage, "thresh", 128, model.argMap);
    bool gray = jo_bool(pStage, "gray", true, model.argMap);
    int type;
    const char *errMsg = NULL;

    if (typeStr.compare("THRESH_BINARY") == 0) {
        type = THRESH_BINARY;
    } else if (typeStr.compare("THRESH_BINARY_INV") == 0) {
        type = THRESH_BINARY_INV;
    } else if (typeStr.compare("THRESH_TRUNC") == 0) {
        type = THRESH_TRUNC;
    } else if (typeStr.compare("THRESH_TOZERO") == 0) {
        type = THRESH_TOZERO;
    } else if (typeStr.compare("THRESH_TOZERO_INV") == 0) {
        type = THRESH_TOZERO_INV;
    } else {
        errMsg = "Expected threshold type (e.g., THRESH_BINARY)";
    }
    if (!gray && isOtsu) {
        errMsg = "Otsu's method cannot be used with color images. Specify a thresh value for color images.";
    }
    if (!errMsg) {
        if (isOtsu) {
            type |= THRESH_OTSU;
        }
        if ((isOtsu || gray) && model.image.channels() > 1) {
            LOGTRACE("apply_threshold() converting image to grayscale");
            cvtColor(model.image, model.image, CV_BGR2GRAY, 0);
        }
        threshold(model.image, model.image, thresh, maxval, type);
    }

    return stageOK("apply_threshold(%s) %s", errMsg, pStage, pStageModel);
}


bool Pipeline::apply_transparent(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    Rect roi = jo_Rect(pStage, "roi", Rect(0, 0, model.image.cols, model.image.rows), model.argMap);
    float alphafg = jo_float(pStage, "alphafg", 1, model.argMap);
    float alphabg = jo_float(pStage, "alphabg", 0, model.argMap);
    vector<int> bgcolor = jo_vectori(pStage, "bgcolor", vector<int>(), model.argMap);
    const char *errMsg = NULL;

    int fgIntensity = 255 * alphafg;
    if (fgIntensity < 0 || 255 < fgIntensity) {
        errMsg = "Expected 0 < alphafg < 1";
    }

    int bgIntensity = 255 * alphabg;
    if (bgIntensity < 0 || 255 < bgIntensity) {
        errMsg = "Expected 0 < alphabg < 1";
    }

    bool isBgColor = bgcolor.size() == 3;
    if (bgcolor.size() != 0 && !isBgColor) {
        errMsg = "Expected JSON [B,G,R] array for bgcolor";
    }

    int roiRowStart = max(0, roi.y);
    int roiColStart = max(0, roi.x);
    int roiRowEnd = min(model.image.rows, roi.y+roi.height);
    int roiColEnd = min(model.image.cols, roi.x+roi.width);

    if (roiRowEnd <= 0 || model.image.rows <= roiRowStart ||
            roiColEnd <= 0 || model.image.cols <= roiColStart) {
        errMsg = "Region of interest is not in image";
    }

    if (!errMsg) {
        Mat imageAlpha;
        cvtColor(model.image, imageAlpha, CV_BGR2BGRA, 0);
        LOGTRACE1("apply_alpha() imageAlpha %s", matInfo(imageAlpha).c_str());

        int rows = imageAlpha.rows;
        int cols = imageAlpha.cols;
        int bgBlue = isBgColor ? bgcolor[0] : 255;
        int bgGreen = isBgColor ? bgcolor[1] : 255;
        int bgRed = isBgColor ? bgcolor[2] : 255;
        for (int r=roiRowStart; r < roiRowEnd; r++) {
            for (int c=roiColStart; c < roiColEnd; c++) {
                if (isBgColor) {
                    if ( bgBlue == imageAlpha.at<Vec4b>(r,c)[0] &&
                            bgGreen == imageAlpha.at<Vec4b>(r,c)[1] &&
                            bgRed == imageAlpha.at<Vec4b>(r,c)[2]) {
                        imageAlpha.at<Vec4b>(r,c)[3] = bgIntensity;
                    } else {
                        imageAlpha.at<Vec4b>(r,c)[3] = fgIntensity;
                    }
                } else {
                    imageAlpha.at<Vec4b>(r,c)[3] = fgIntensity;
                }
            }
        }
        model.image = imageAlpha;
    }

    return stageOK("apply_alpha(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_HoleRecognizer(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    float diamMin = jo_float(pStage, "diamMin", 0, model.argMap);
    float diamMax = jo_float(pStage, "diamMax", 0, model.argMap);
    int showMatches = jo_int(pStage, "show", 0, model.argMap);
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
        for (size_t i = 0; i < matches.size(); i++) {
            json_array_append(holes, matches[i].as_json_t());
        }
    }

    return stageOK("apply_HoleRecognizer(%s) %s", errMsg, pStage, pStageModel);
}

bool Pipeline::apply_HoughCircles(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    int diamMin = jo_int(pStage, "diamMin", 0, model.argMap);
    int diamMax = jo_int(pStage, "diamMax", 0, model.argMap);
    int showCircles = jo_int(pStage, "show", 0, model.argMap);
    // alg. parameters
    int bf_d = jo_int(pStage, "bilateralfilter_d", 15, model.argMap);
    double bf_sigmaColor = jo_double(pStage, "bilateralfilter_sigmaColor", 1000, model.argMap);
    double bf_sigmaSpace = jo_double(pStage, "bilateralfilter_sigmaSpace", 1000, model.argMap);
    double hc_dp = jo_double(pStage, "houghcircles_dp", 1, model.argMap);
    double hc_minDist = jo_double(pStage, "houghcircles_minDist", 10, model.argMap);
    double hc_param1 = jo_double(pStage, "houghcircles_param1", 80, model.argMap);
    double hc_param2 = jo_double(pStage, "houghcircles_param2", 10, model.argMap);

    const char *errMsg = NULL;

    if (diamMin <= 0 || diamMax <= 0 || diamMin > diamMax) {
        errMsg = "expected: 0 < diamMin < diamMax ";
    } else if (showCircles < 0) {
        errMsg = "expected: 0 < showCircles ";
    } else if (logLevel >= FIRELOG_TRACE) {
        char *pStageJson = json_dumps(pStage, 0);
        LOGTRACE1("apply_HoughCircles(%s)", pStageJson);
        free(pStageJson);
    }
    if (!errMsg) {
        vector<Circle> circles;
        HoughCircle hough_c(diamMin, diamMax);
        hough_c.setShowCircles(showCircles);
        hough_c.setFilterParams(bf_d, bf_sigmaColor, bf_sigmaSpace);
        hough_c.setHoughParams(hc_dp, hc_minDist, hc_param1, hc_param2);
        hough_c.scan(model.image, circles);
        json_t *circles_json = json_array();
        json_object_set(pStageModel, "circles", circles_json);
        for (size_t i = 0; i < circles.size(); i++) {
            json_array_append(circles_json, circles[i].as_json_t());
        }
    }

    return stageOK("apply_HoughCircles(%s) %s", errMsg, pStage, pStageModel);
}

Pipeline::Pipeline(const char *pDefinition, DefinitionType defType) {
    json_error_t jerr;
    string pipelineString = pDefinition;
    if (defType == PATH) {
        if (pDefinition && *pDefinition) {
            ifstream ifs(pDefinition);
            stringstream pipelineStream;
            pipelineStream << ifs.rdbuf();
            pipelineString = pipelineStream.str();
            if (pipelineString.size() < 10) {
                char msg[255];
                snprintf(msg, sizeof(msg), "Pipeline::Pipeline(%s, PATH) no JSON pipeline definition", pDefinition);
                LOGERROR(msg);
                throw msg;
            } else {
                LOGTRACE1("Pipeline::Pipeline(%s, PATH)", pDefinition);
            }
        } else {
            pipelineString = "[{\"op\":\"nop\"}]";
            LOGTRACE2("Pipeline::Pipeline(%s, PATH) => %s", pDefinition, pipelineString.c_str());
        }
    }
    pPipeline = json_loads(pipelineString.c_str(), 0, &jerr);

    if (!pPipeline) {
        LOGERROR3("Pipeline::process cannot parse json: %s src:%s line:%d", jerr.text, jerr.source, jerr.line);
        throw jerr;
    }
}

Pipeline::Pipeline(json_t *pJson) {
    pPipeline = json_incref(pJson);
}

Pipeline::~Pipeline() {
    if (pPipeline->refcount == 1) {
        LOGTRACE1("~Pipeline() pPipeline->refcount:%d", (int)pPipeline->refcount);
    } else {
        LOGERROR1("~Pipeline() pPipeline->refcount:%d EXPECTED 0", (int)pPipeline->refcount);
    }
    json_decref(pPipeline);
}

static bool logErrorMessage(const char *errMsg, const char *pName, json_t *pStage, json_t *pStageModel) {
    if (errMsg) {
        json_object_set(pStageModel, "ERROR", json_string(errMsg));
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

json_t *Pipeline::process(Mat &workingImage, ArgMap &argMap) {
    Model model(argMap);
    json_t *pModelJson = model.getJson(true);

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
    long long tickStart = cvGetTickCount();
    json_array_foreach(pPipeline, index, pStage) {
        string pOp = jo_string(pStage, "op", "", model.argMap);
        string pName = jo_string(pStage, "name");
        bool isSaveImage = true;
        if (pName.empty()) {
            char defaultName[100];
            snprintf(defaultName, sizeof(defaultName), "s%d", (int)index+1);
            pName = defaultName;
            isSaveImage = false;
        }
        string comment = jo_string(pStage, "comment", "", model.argMap);
        json_t *pStageModel = json_object();
        json_t *jmodel = model.getJson(false);
        json_object_set(jmodel, pName.c_str(), pStageModel);
        if (logLevel >= FIRELOG_DEBUG) {
            string stageDump = jo_object_dump(pStage, model.argMap);
            snprintf(debugBuf,sizeof(debugBuf), "process() %s %s",
                     matInfo(model.image).c_str(), stageDump.c_str());
        }
        if (strncmp(pOp.c_str(), "nop", 3)==0) {
            LOGDEBUG1("%s (NO ACTION TAKEN)", debugBuf);
        } else if (pName.compare("input")==0) {
            ok = logErrorMessage("\"input\" is the reserved stage name for the input image",
                                 pName.c_str(), pStage, pStageModel);
        } else {
            LOGDEBUG1("%s", debugBuf);
            try {
                const char *errMsg = dispatch(pName.c_str(), pOp.c_str(), pStage, pStageModel, model);
                ok = logErrorMessage(errMsg, pName.c_str(), pStage, pStageModel);
                if (isSaveImage) {
                    model.imageMap[pName.c_str()] = model.image.clone();
                }
            } catch (runtime_error &ex) {
                ok = logErrorMessage(ex.what(), pName.c_str(), pStage, pStageModel);
            } catch (cv::Exception &ex) {
                ok = logErrorMessage(ex.what(), pName.c_str(), pStage, pStageModel);
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

    float msElapsed = (cvGetTickCount() - tickStart)/cvGetTickFrequency()/1000;
    LOGDEBUG3("Pipeline::processModel(stages:%d) -> %s %.1fms",
              (int)json_array_size(pPipeline), matInfo(model.image).c_str(), msElapsed);

    return ok;
}

const char * Pipeline::dispatch(const char *pName, const char *pOp, json_t *pStage, json_t *pStageModel, Model &model) {
    bool ok = true;
    const char *errMsg = NULL;

    if (strcmp(pOp, "absdiff")==0) {
        ok = apply_absdiff(pStage, pStageModel, model);
    } else if (strcmp(pOp, "backgroundSubtractor")==0) {
        ok = apply_backgroundSubtractor(pStage, pStageModel, model);
    } else if (strcmp(pOp, "bgsub")==0) {
        ok = apply_backgroundSubtractor(pStage, pStageModel, model);
    } else if (strcmp(pOp, "blur")==0) {
        ok = apply_blur(pStage, pStageModel, model);
    } else if (strcmp(pOp, "calcHist")==0) {
        ok = apply_calcHist(pStage, pStageModel, model);
    } else if (strcmp(pOp, "calcOffset")==0) {
        ok = apply_calcOffset(pStage, pStageModel, model);
    } else if (strcmp(pOp, "circle")==0) {
        ok = apply_circle(pStage, pStageModel, model);
    } else if (strcmp(pOp, "convertTo")==0) {
        ok = apply_convertTo(pStage, pStageModel, model);
    } else if (strcmp(pOp, "cout")==0) {
        ok = apply_cout(pStage, pStageModel, model);
    } else if (strcmp(pOp, "crop")==0) {
        ok = apply_crop(pStage, pStageModel, model);
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
    } else if (strcmp(pOp, "FireSight")==0) {
        ok = apply_FireSight(pStage, pStageModel, model);
    } else if (strcmp(pOp, "HoleRecognizer")==0) {
        ok = apply_HoleRecognizer(pStage, pStageModel, model);
    } else if (strcmp(pOp, "HoughCircles")==0) {
        ok = apply_HoughCircles(pStage, pStageModel, model);
    } else if (strcmp(pOp, "points2resolution_RANSAC")==0) {
        ok = apply_points2resolution_RANSAC(pStage, pStageModel, model);
    } else if (strcmp(pOp, "imread")==0) {
        ok = apply_imread(pStage, pStageModel, model);
    } else if (strcmp(pOp, "imwrite")==0) {
        ok = apply_imwrite(pStage, pStageModel, model);
    } else if (strcmp(pOp, "Mat")==0) {
        ok = apply_Mat(pStage, pStageModel, model);
    } else if (strcmp(pOp, "matchGrid")==0) {
        ok = apply_matchGrid(pStage, pStageModel, model);
    } else if (strcmp(pOp, "matchTemplate")==0) {
        ok = apply_matchTemplate(pStage, pStageModel, model);
    } else if (strcmp(pOp, "meanStdDev")==0) {
        ok = apply_meanStdDev(pStage, pStageModel, model);
    } else if (strcmp(pOp, "minAreaRect")==0) {
        ok = apply_minAreaRect(pStage, pStageModel, model);
    } else if (strcmp(pOp, "model")==0) {
        ok = apply_model(pStage, pStageModel, model);
    } else if (strcmp(pOp, "morph")==0) {
        ok = apply_morph(pStage, pStageModel, model);
    } else if (strcmp(pOp, "MSER")==0) {
        ok = apply_MSER(pStage, pStageModel, model);
    } else if (strcmp(pOp, "normalize")==0) {
        ok = apply_normalize(pStage, pStageModel, model);
    } else if (strcmp(pOp, "PSNR")==0) {
        ok = apply_PSNR(pStage, pStageModel, model);
    } else if (strcmp(pOp, "proto")==0) {
        ok = apply_proto(pStage, pStageModel, model);
    } else if (strcmp(pOp, "putText")==0) {
        ok = apply_putText(pStage, pStageModel, model);
#ifdef LGPL2_1
    } else if (strcmp(pOp, "qrDecode")==0) {
        ok = apply_qrdecode(pStage, pStageModel, model);
#endif // LGPL2_1
    } else if (strcmp(pOp, "rectangle")==0) {
        ok = apply_rectangle(pStage, pStageModel, model);
    } else if (strcmp(pOp, "resize")==0) {
        ok = apply_resize(pStage, pStageModel, model);
    } else if (strcmp(pOp, "sharpness")==0) {
        ok = apply_sharpness(pStage, pStageModel, model);
    } else if (strcmp(pOp, "SimpleBlobDetector")==0) {
        ok = apply_SimpleBlobDetector(pStage, pStageModel, model);
    } else if (strcmp(pOp, "split")==0) {
        ok = apply_split(pStage, pStageModel, model);
    } else if (strcmp(pOp, "stageImage")==0) {
        ok = apply_stageImage(pStage, pStageModel, model);
    } else if (strcmp(pOp, "transparent")==0) {
        ok = apply_transparent(pStage, pStageModel, model);
    } else if (strcmp(pOp, "threshold")==0) {
        ok = apply_threshold(pStage, pStageModel, model);
    } else if (strcmp(pOp, "undistort")==0) {
        ok = apply_undistort(pName, pStage, pStageModel, model);
    } else if (strcmp(pOp, "warpAffine")==0) {
        ok = apply_warpAffine(pStage, pStageModel, model);
    } else if (strcmp(pOp, "warpRing")==0) {
        ok = apply_warpRing(pStage, pStageModel, model);
    } else if (strcmp(pOp, "warpPerspective")==0) {
        ok = apply_warpPerspective(pName, pStage, pStageModel, model);

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
