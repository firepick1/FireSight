#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "FireLog.h"
#include "Pipeline.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "jo_util.hpp"
#include "MatUtil.hpp"
#include "version.h"
#include "stages/Sharpness.h"
#include "gui.h"

#include "stages.h"

using namespace cv;
using namespace std;
using namespace firesight;

StageData::StageData(string stageName) {
    LOGTRACE1("StageData constructor %s", stageName.c_str());
}

StageData::~StageData() {
    LOGTRACE("StageData destructor");
}

bool Stage::stageOK(const char *fmt, const char *errMsg, json_t *pStage, json_t *pStageModel) {
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

void Stage::validateImage(Mat &image)
{
    if (image.cols == 0 || image.rows == 0) {
        image = Mat(100,100, CV_8UC3);
        putText(image, "FireSight:", Point(10,20), FONT_HERSHEY_PLAIN, 1, Scalar(128,255,255));
        putText(image, "No input", Point(10,40), FONT_HERSHEY_PLAIN, 1, Scalar(128,255,255));
        putText(image, "image?", Point(10,60), FONT_HERSHEY_PLAIN, 1, Scalar(128,255,255));
    }
}

// TODO remove call to Pipeline::stageOK
bool Pipeline::stageOK(const char *fmt, const char *errMsg, json_t *pStage, json_t *pStageModel) {
    return Stage::stageOK(fmt, errMsg, pStage, pStageModel);
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

bool Pipeline::apply_equalizeHist(json_t *pStage, json_t *pStageModel, Model &model) {
    const char *errMsg = NULL;

    if (!errMsg) {
        equalizeHist(model.image, model.image);
    }

    return stageOK("apply_equalizeHist(%s) %s", errMsg, pStage, pStageModel);
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

// TODO remove
void Pipeline::validateImage(Mat &image) {
    Stage::validateImage(image);
}

json_t *Pipeline::process(Input * input, ArgMap &argMap, Mat &output, bool gui) {
    Model model(argMap);
    json_t *pModelJson = model.getJson(true);

    model.image = (input->get()).clone();
    model.imageMap["input"] = model.image.clone();
    bool ok;
    if (gui)
        ok = processModelGUI(input, model);
    else
        ok = processModel(model);

    output = model.image;
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
    long long tickStart = cvGetTickCount();

    for(index = 0; index < json_array_size(pPipeline) && (pStage = json_array_get(pPipeline, index)); index++) {
        unique_ptr<Stage> stage = parseStage(index, pStage, model);

        json_t *pStageModel = json_object();
        json_t *jmodel = model.getJson(false);
        json_object_set(jmodel, stage->getName().c_str(), pStageModel);

        ok = stage->apply(pStageModel, model);

        if (!ok)
            break;
    } // json_array_foreach


    float msElapsed = (cvGetTickCount() - tickStart)/cvGetTickFrequency()/1000;
    LOGDEBUG3("Pipeline::processModel(stages:%d) -> %s %.1fms",
              (int)json_array_size(pPipeline), matInfo(model.image).c_str(), msElapsed);

    return ok;
}

bool Pipeline::processModelGUI(Input * input, Model &model) {
    if (!json_is_array(pPipeline)) {
        const char * errMsg = "Pipeline::process expected json array for pipeline definition";
        LOGERROR1(errMsg, "");
        throw errMsg;
    }

    PipelineViewer pv(500);

    bool ok = 1;
    size_t index;
    json_t *pStage;

    vector<unique_ptr<Stage> > stages;


    for(index = 0; index < json_array_size(pPipeline) && (pStage = json_array_get(pPipeline, index)); index++)
        stages.push_back(parseStage(index, pStage, model));

    bool go = true;
    int key = 0;
    int sel_stage = -1;
    int sel_param = -1;
    bool rerunPipeline;
    do {
        // get current image from input
        model.image = input->get().clone();
        model.imageMap["input"] = model.image.clone();
        rerunPipeline = false;
        Model workModel(model.argMap);

        workModel.image = model.image.clone();
        workModel.imageMap["input"] = model.image.clone();

        vector<Mat> history;
        history.push_back(workModel.image.clone());

        for(index = 0; index < json_array_size(pPipeline) && (pStage = json_array_get(pPipeline, index)); index++) {
            json_t *pStageModel = json_object();
            json_t *jmodel = workModel.getJson(false);
            json_object_set(jmodel, stages[index]->getName().c_str(), pStageModel);

//            stages[index]->print();
            try {
                ok = stages[index]->apply(pStageModel, workModel);
            } catch (std::exception &e) {
                workModel.image = Mat::zeros(workModel.image.size(), workModel.image.type());
            }

            // Print out returned model
//            json_t *pModel = workModel.getJson(true);
//            char *pModelStr = json_dumps(pModel, JSON_PRESERVE_ORDER|JSON_COMPACT|JSON_INDENT(2));
//            cout << pModelStr << endl;
//            free(pModelStr);

            history.push_back(workModel.image.clone());
        } // json_array_foreach

        do {
            pv.update(stages, history, sel_stage, sel_param);

            key = cv::waitKey(1);

            if (key >= '0' && key <= '9') {
                sel_stage = key - '0';
                if (sel_stage > stages.size())
                    sel_stage = stages.size();
                sel_param = 0;
                continue;
            }

            switch (key) {
            case 65362: // up
            case 'p':
                sel_param = max(0, sel_param-1);
                break;
            case 65364: // down
            case 'n':
                sel_param = min((int) stages[sel_stage-1]->getParams().size()-1, sel_param+1);
                break;
            case 65361: // left
            case '-':
            {
                map<string, Parameter*> params = stages[sel_stage-1]->getParams();
                int idx = 0;
                for (auto it : params) {
                    if (idx == sel_param) {
                        it.second->dec();
                        rerunPipeline = true;
                        break;
                    }
                    idx++;
                }
            }
                break;
            case 65363: // right
            case '+':
            {
                map<string, Parameter*> params = stages[sel_stage-1]->getParams();
                int idx = 0;
                for (auto it : params) {
                    if (idx == sel_param) {
                        it.second->inc();
                        rerunPipeline = true;
                        break;
                    }
                    idx++;
                }
            }
                break;
            case -1:
                rerunPipeline = true;
                break;
            case 27:
                go = false;
                break;
            }

//            printf("sel_stage = %i\nsel_param = %i\n", sel_stage, sel_param);

        } while (!rerunPipeline && go);//key != 27);
    } while (go);


    return ok;
}

unique_ptr<Stage> Pipeline::parseStage(int index, json_t * pStage, Model &model) {
    bool ok = 1;
    char debugBuf[255];
    unique_ptr<Stage> stage = nullptr;
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
            stage = StageFactory::getStage(pOp.c_str(), pStage, model, pName);
            if (!stage) {

                const char *errMsg = "Failed to parse stage";

                ok = logErrorMessage(errMsg, pName.c_str(), pStage, pStageModel);

            }
        } catch (runtime_error &ex) {
            ok = logErrorMessage(ex.what(), pName.c_str(), pStage, pStageModel);
        } catch (cv::Exception &ex) {
            ok = logErrorMessage(ex.what(), pName.c_str(), pStage, pStageModel);
        }
    } //if-else (pOp)

    return stage;

}

//bool Pipeline::processStage(int index, json_t * pStage, Model &model) {
//    bool ok = 1;
//    char debugBuf[255];
//    string pOp = jo_string(pStage, "op", "", model.argMap);
//    string pName = jo_string(pStage, "name");
//    bool isSaveImage = true;
//    if (pName.empty()) {
//        char defaultName[100];
//        snprintf(defaultName, sizeof(defaultName), "s%d", (int)index+1);
//        pName = defaultName;
//        isSaveImage = false;
//    }
//    string comment = jo_string(pStage, "comment", "", model.argMap);
//    json_t *pStageModel = json_object();
//    json_t *jmodel = model.getJson(false);
//    json_object_set(jmodel, pName.c_str(), pStageModel);
//    if (logLevel >= FIRELOG_DEBUG) {
//        string stageDump = jo_object_dump(pStage, model.argMap);
//        snprintf(debugBuf,sizeof(debugBuf), "process() %s %s",
//                 matInfo(model.image).c_str(), stageDump.c_str());
//    }
//    if (strncmp(pOp.c_str(), "nop", 3)==0) {
//        LOGDEBUG1("%s (NO ACTION TAKEN)", debugBuf);
//    } else if (pName.compare("input")==0) {
//        ok = logErrorMessage("\"input\" is the reserved stage name for the input image",
//                             pName.c_str(), pStage, pStageModel);
//    } else {
//        LOGDEBUG1("%s", debugBuf);
//        try {
//            Stage * stage = nullptr;
//            const char *errMsg = NULL;
//            stage = StageFactory::getStage(pOp.c_str(), pStage, model);
//            if (stage) {
//                ok = stage->apply(pStage, pStageModel, model);
//                if (!ok)
//                    errMsg = "Pipeline stage failed";
//            } else {
//                ok = false;
//                errMsg = "unknown stage";
//            }

//            ok = logErrorMessage(errMsg, pName.c_str(), pStage, pStageModel);
//            if (isSaveImage) {
//                model.imageMap[pName.c_str()] = model.image.clone();
//            }
//        } catch (runtime_error &ex) {
//            ok = logErrorMessage(ex.what(), pName.c_str(), pStage, pStageModel);
//        } catch (cv::Exception &ex) {
//            ok = logErrorMessage(ex.what(), pName.c_str(), pStage, pStageModel);
//        }
//    } //if-else (pOp)
//    if (!ok) {
//        LOGERROR("cancelled pipeline execution");
//        return false;
//    }
//    if (model.image.cols <=0 || model.image.rows<=0) {
//        LOGERROR2("Empty working image: %dr x %dc", model.image.rows, model.image.cols);
//        return false;
//    }

//    return ok;
//}

std::unique_ptr<Stage> StageFactory::getStage(const char *pOp, json_t *pStage, Model &model, string pName)
{
    static int id = 0;
    unique_ptr<Stage> stage = nullptr;
    try {
    if (strcmp(pOp, "absdiff")==0)
        stage = unique_ptr<Stage>(new AbsDiff(pStage, model, pName));
//    if (strcmp(pOp, "backgroundSubtractor")==0)
//        ok = apply_backgroundSubtractor(pStage, pStageModel, model);
//    if (strcmp(pOp, "bgsub")==0)
//        ok = apply_backgroundSubtractor(pStage, pStageModel, model);
    if (strcmp(pOp, "blur")==0)
        stage = unique_ptr<Stage>(new Blur(pStage, model, pName));
//    if (strcmp(pOp, "calcHist")==0)
//        ok = apply_calcHist(pStage, pStageModel, model);
//    if (strcmp(pOp, "calcOffset")==0)
//        ok = apply_calcOffset(pStage, pStageModel, model);
    if (strcmp(pOp, "circle")==0)
        stage = unique_ptr<Stage>(new DrawCircle(pStage, model, pName));
//    if (strcmp(pOp, "convertTo")==0)
//        ok = apply_convertTo(pStage, pStageModel, model);
//    if (strcmp(pOp, "cout")==0)
//        ok = apply_cout(pStage, pStageModel, model);
    if (strcmp(pOp, "Canny")==0)
        stage = unique_ptr<Stage>(new Canny(pStage, model, pName));
    if (strcmp(pOp, "cvtColor")==0)
        stage = unique_ptr<Stage>(new CvtColor(pStage, model, pName));
    if (strcmp(pOp, "dft")==0)
        stage = unique_ptr<Stage>(new DFT(pStage, model, pName));
    if (strcmp(pOp, "dftSpectrum")==0)
        stage = unique_ptr<Stage>(new DFTSpectrum(pStage, model, pName));
    if (strcmp(pOp, "dilate")==0)
        stage = unique_ptr<Stage>(new Dilate(pStage, model, pName));
    if (strcmp(pOp, "drawKeypoints")==0)
        stage = unique_ptr<Stage>(new DrawKeypoints(pStage, model, pName));
    if (strcmp(pOp, "drawRects")==0)
        stage = unique_ptr<Stage>(new DrawRects(pStage, model, pName));
//    if (strcmp(pOp, "equalizeHist")==0) {
//        ok = apply_equalizeHist(pStage, pStageModel, model);
    if (strcmp(pOp, "erode")==0)
        stage = unique_ptr<Stage>(new Erode(pStage, model, pName));
    if (strcmp(pOp, "FireSight")==0)
        stage = unique_ptr<Stage>(new FireSightStage(pStage, model, pName));
    if (strcmp(pOp, "HoleRecognizer")==0)
        stage = unique_ptr<Stage>(new HoleRecognizer(pStage, model, pName));
    if (strcmp(pOp, "HoughCircles")==0)
        stage = unique_ptr<Stage>(new HoughCircles(pStage, model, pName));
//    if (strcmp(pOp, "points2resolution_RANSAC")==0) {
//        ok = apply_points2resolution_RANSAC(pStage, pStageModel, model);
    if (strcmp(pOp, "imread")==0)
        stage = unique_ptr<Stage>(new ImRead(pStage, model, pName));
    if (strcmp(pOp, "imwrite")==0)
        stage = unique_ptr<Stage>(new ImWrite(pStage, model, pName));
//    if (strcmp(pOp, "Mat")==0) {
//        ok = apply_Mat(pStage, pStageModel, model);
//    if (strcmp(pOp, "matchGrid")==0) {
//        ok = apply_matchGrid(pStage, pStageModel, model);
    if (strcmp(pOp, "matchTemplate")==0)
        stage = unique_ptr<Stage>(new TemplateMatch(pStage, model, pName));
//    if (strcmp(pOp, "meanStdDev")==0) {
//        ok = apply_meanStdDev(pStage, pStageModel, model);
//    if (strcmp(pOp, "minAreaRect")==0) {
//        ok = apply_minAreaRect(pStage, pStageModel, model);
    if (strcmp(pOp, "model")==0)
        stage = unique_ptr<Stage>(new ModelStage(pStage, model, pName));
    if (strcmp(pOp, "morph")==0)
        stage = unique_ptr<Stage>(new Morph(pStage, model, pName));
//    if (strcmp(pOp, "MSER")==0) {
//        ok = apply_MSER(pStage, pStageModel, model);
//    if (strcmp(pOp, "normalize")==0) {
//        ok = apply_normalize(pStage, pStageModel, model);
//    if (strcmp(pOp, "PSNR")==0) {
//        ok = apply_PSNR(pStage, pStageModel, model);
//    if (strcmp(pOp, "proto")==0) {
//        ok = apply_proto(pStage, pStageModel, model);
    if (strcmp(pOp, "putText")==0)
        stage = unique_ptr<Stage>(new Text(pStage, model, pName));
//        ok = apply_putText(pStage, pStageModel, model);
//#ifdef LGPL2_1
//    if (strcmp(pOp, "qrDecode")==0) {
//        ok = apply_qrdecode(pStage, pStageModel, model);
//#endif // LGPL2_1
    if (strcmp(pOp, "rectangle")==0)
        stage = unique_ptr<Stage>(new DrawRectangle(pStage, model, pName));
//    if (strcmp(pOp, "resize")==0) {
//        ok = apply_resize(pStage, pStageModel, model);
//    if (strcmp(pOp, "sharpness")==0) {
//        ok = apply_sharpness(pStage, pStageModel, model);
    if (strcmp(pOp, "detectParts")==0)
        stage = unique_ptr<Stage>(new PartDetector(pStage, model, pName));
//        ok = apply_detectParts(pStage, pStageModel, model);
//    if (strcmp(pOp, "SimpleBlobDetector")==0) {
//        ok = apply_SimpleBlobDetector(pStage, pStageModel, model);
//    if (strcmp(pOp, "split")==0) {
//        ok = apply_split(pStage, pStageModel, model);
//    if (strcmp(pOp, "stageImage")==0) {
//        ok = apply_stageImage(pStage, pStageModel, model);
//    if (strcmp(pOp, "transparent")==0) {
//        ok = apply_transparent(pStage, pStageModel, model);
    if (strcmp(pOp, "threshold")==0)
        stage = unique_ptr<Stage>(new Threshold(pStage, model, pName));
//    if (strcmp(pOp, "undistort")==0) {
//        ok = apply_undistort(pName, pStage, pStageModel, model);
    if (strcmp(pOp, "warpAffine")==0)
        stage = unique_ptr<Stage>(new WarpAffine(pStage, model, pName));
    if (strcmp(pOp, "warpRing")==0)
        stage = unique_ptr<Stage>(new WarpRing(pStage, model, pName));
    if (strcmp(pOp, "warpPerspective")==0)
        stage = unique_ptr<Stage>(new WarpPerspective(pStage, model, pName));

//    if (strncmp(pOp, "nop", 3)==0) {
//        LOGDEBUG("Skipping nop...");

    } catch (invalid_argument &e) {
        string pName = jo_string(pStage, "name");
        json_t *pStageModel = json_object();
        json_t *jmodel = model.getJson(false);
        json_object_set(jmodel, pName.c_str(), pStageModel);

        logErrorMessage(e.what(), pName.c_str(), pStage, pStageModel);
    }

    return stage;
}
