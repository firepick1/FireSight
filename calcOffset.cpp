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

bool Pipeline::apply_calcOffset(json_t *pStage, json_t *pStageModel, Model &model) {
    validateImage(model.image);
    string tmpltPath = jo_string(pStage, "template", "", model.argMap);
    Scalar offsetColor(32,32,255);
    offsetColor = jo_Scalar(pStage, "offsetColor", offsetColor, model.argMap);
    int xtol = jo_int(pStage, "xtol", 32, model.argMap);
    int ytol = jo_int(pStage, "ytol", 32, model.argMap);
    vector<int> channels = jo_vectori(pStage, "channels", vector<int>(), model.argMap);
    assert(model.image.cols > 2*xtol);
    assert(model.image.rows > 2*ytol);
    Rect roi= jo_Rect(pStage, "roi", Rect(xtol, ytol, model.image.cols-2*xtol, model.image.rows-2*ytol), model.argMap);
    if (roi.x == -1) {
        roi.x = (model.image.cols - roi.width)/2;
    }
    if (roi.y == -1) {
        roi.y = (model.image.rows - roi.height)/2;
    }
    Rect roiScan = Rect(roi.x-xtol, roi.y-ytol, roi.width+2*xtol, roi.height+2*ytol);
    float minval = jo_float(pStage, "minval", 0.7f, model.argMap);
    float corr = jo_float(pStage, "corr", 0.99f);
    string outputStr = jo_string(pStage, "output", "current", model.argMap);
    string errMsg;
    int flags = INTER_LINEAR;
    int method = CV_TM_CCOEFF_NORMED;
    Mat tmplt;
    int borderMode = BORDER_REPLICATE;

	if (roiScan.x < 0 || roiScan.y < 0 || model.image.cols < roiScan.x+roiScan.width || model.image.rows < roiScan.y+roiScan.height) {
		errMsg = "ROI with and surrounding xtol,ytol region must be within image";
	}
    if (tmpltPath.empty()) {
        errMsg = "Expected template path for imread";
    } else {
        if (model.image.channels() == 1) {
            tmplt = imread(tmpltPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        } else {
            tmplt = imread(tmpltPath.c_str(), CV_LOAD_IMAGE_COLOR);
        }
        if (tmplt.data) {
            LOGTRACE2("apply_calcOffset(%s) %s", tmpltPath.c_str(), matInfo(tmplt).c_str());
            if (model.image.rows<tmplt.rows || model.image.cols<tmplt.cols) {
                errMsg = "Expected template smaller than image to match";
            }
        } else {
            errMsg = "imread failed: " + tmpltPath;
        }
    }
    if (errMsg.empty()) {
        if (model.image.channels() > 3) {
            errMsg = "Expected at most 3 channels for pipeline image";
        } else if (tmplt.channels() != model.image.channels()) {
            errMsg = "Template and pipeline image must have same number of channels";
        } else {
            for (int iChannel = 0; iChannel < channels.size(); iChannel++) {
                if (channels[iChannel] < 0 || model.image.channels() <= channels[iChannel]) {
                    errMsg = "Referenced channel is not in image";
                }
            }
        }
    }

    if (errMsg.empty()) {
        Mat result;
        Mat imagePlanes[] = { Mat(), Mat(), Mat() };
        Mat tmpltPlanes[] = { Mat(), Mat(), Mat() };
        if (channels.size() == 0) {
            channels.push_back(0);
            if (model.image.channels() == 1) {
                imagePlanes[0] = model.image;
                tmpltPlanes[0] = tmplt;
            } else {
                cvtColor(model.image, imagePlanes[0], CV_BGR2GRAY);
                cvtColor(tmplt, tmpltPlanes[0], CV_BGR2GRAY);
            }
        } else if (model.image.channels() == 1) {
            imagePlanes[0] = model.image;
            tmpltPlanes[0] = tmplt;
        } else {
            split(model.image, imagePlanes);
            split(tmplt, tmpltPlanes);
        }

        json_t *pRects = json_array();
        json_t *pChannels = json_object();
        json_object_set(pStageModel, "channels", pChannels);
        json_object_set(pStageModel, "rects", pRects);

        json_t *pRect = json_object();
        json_array_append(pRects, pRect);
        json_object_set(pRect, "x", json_integer(roiScan.x+roiScan.width/2));
        json_object_set(pRect, "y", json_integer(roiScan.y+roiScan.height/2));
        json_object_set(pRect, "width", json_integer(roiScan.width));
        json_object_set(pRect, "height", json_integer(roiScan.height));
        json_object_set(pRect, "angle", json_integer(0));
        json_t *pOffsetColor = NULL;

        for (int iChannel=0; iChannel<channels.size(); iChannel++) {
            int channel = channels[iChannel];
            Mat imageSource(imagePlanes[channel], roiScan);
            Mat tmpltSource(tmpltPlanes[channel], roi);

            matchTemplate(imageSource, tmpltSource, result, method);
            LOGTRACE4("apply_calcOffset() matchTemplate(%s,%s,%s,CV_TM_CCOEFF_NORMED) channel:%d",
                      matInfo(imageSource).c_str(), matInfo(tmpltSource).c_str(), matInfo(result).c_str(), channel);

            vector<Point> matches;
            float maxVal = *max_element(result.begin<float>(),result.end<float>());
            float rangeMin = corr * maxVal;
            float rangeMax = maxVal;
            matMaxima(result, matches, rangeMin, rangeMax);

            if (logLevel >= FIRELOG_TRACE) {
                for (size_t iMatch=0; iMatch<matches.size(); iMatch++) {
                    int mx = matches[iMatch].x;
                    int my = matches[iMatch].y;
                    float val = result.at<float>(my,mx);
                    if (val < minval) {
                        LOGTRACE4("apply_calcOffset() ignoring (%d,%d) val:%g corr:%g", mx, my, val, val/maxVal);
                    } else {
                        LOGTRACE4("apply_calcOffset() matched (%d,%d) val:%g corr:%g", mx, my, val, val/maxVal);
                    }
                }
            }
            json_t *pMatches = json_object();
            char key[10];
            snprintf(key, sizeof(key), "%d", channel);
            json_object_set(pChannels, key, pMatches);
            if (matches.size() == 1) {
                int mx = matches[0].x;
                int my = matches[0].y;
                float val = result.at<float>(my,mx);
                if (minval <= val) {
                    int dx = xtol - mx;
                    int dy = ytol - my;
                    json_object_set(pMatches, "dx", json_integer(dx));
                    json_object_set(pMatches, "dy", json_integer(dy));
                    json_object_set(pMatches, "match", json_float(val));
                    if (dx || dy) {
                        json_t *pOffsetRect = json_object();
                        json_array_append(pRects, pOffsetRect);
                        json_object_set(pOffsetRect, "x", json_integer(roi.x+roi.width/2-dx));
                        json_object_set(pOffsetRect, "y", json_integer(roi.y+roi.height/2-dy));
                        json_object_set(pOffsetRect, "width", json_integer(roi.width));
                        json_object_set(pOffsetRect, "height", json_integer(roi.height));
                        json_object_set(pOffsetRect, "angle", json_integer(0));
                        if (!pOffsetColor) {
                            pOffsetColor = json_array();
                            json_array_append(pOffsetColor, json_integer(offsetColor[0]));
                            json_array_append(pOffsetColor, json_integer(offsetColor[1]));
                            json_array_append(pOffsetColor, json_integer(offsetColor[2]));
                        }
                    }
                }
            }
        }

        json_t *pRoiRect = json_object();
        json_array_append(pRects, pRoiRect);
        json_object_set(pRoiRect, "x", json_integer(roi.x+roi.width/2));
        json_object_set(pRoiRect, "y", json_integer(roi.y+roi.height/2));
        json_object_set(pRoiRect, "width", json_integer(roi.width));
        json_object_set(pRoiRect, "height", json_integer(roi.height));
        json_object_set(pRoiRect, "angle", json_integer(0));
        if (pOffsetColor) {
            json_object_set(pRoiRect, "color", pOffsetColor);
        }

        normalize(result, result, 0, 255, NORM_MINMAX);
        result.convertTo(result, CV_8U);
        Mat corrInset = model.image.colRange(0,result.cols).rowRange(0,result.rows);
        switch (model.image.channels()) {
        case 3:
            cvtColor(result, corrInset, CV_GRAY2BGR);
            break;
        case 4:
            cvtColor(result, corrInset, CV_GRAY2BGRA);
            break;
        default:
            result.copyTo(corrInset);
            break;
        }
    }

    return stageOK("apply_calcOffset(%s) %s", errMsg.c_str(), pStage, pStageModel);
}
