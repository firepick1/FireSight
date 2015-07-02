#ifndef FOURIER_H
#define FOURIER_H

#include "Pipeline.h"
#include "jo_util.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;
using namespace std;


class DFT : public Stage
{
public:
    DFT(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        depthStr = jo_string(pStage, "depth", "CV_8U", model.argMap);
        _params["depth"] = new StringParameter(this, depthStr);

        json_t *pFlags = jo_object(pStage, "flags", model.argMap);
        flags = 0;

        if (json_is_array(pFlags)) {
          size_t index;
          json_t *pStr;
          json_array_foreach(pFlags, index, pStr) {
            const char *flag = json_string_value(pStr);
            if (!flag)
              throw std::invalid_argument("Expected array of flag name strings");

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
                throw std::invalid_argument("Unknown flag" + string(flag));
            }
          }
        }
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        const char *errMsg = NULL;

        switch (model.image.channels()) {
            case 4:
                LOGTRACE("apply_dft(): converting 4 channel image assuming CV_BGRA2GRAY");
                cvtColor(model.image, model.image, CV_BGRA2GRAY, 1);
                break;
            case 3:
                LOGTRACE("apply_dft(): converting 3 channel image assuming CV_BGR2GRAY");
                cvtColor(model.image, model.image, CV_BGR2GRAY, 1);
                break;
        }
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

        return stageOK("apply_dft(%s) %s", errMsg, pStage, pStageModel);
    }

    string depthStr;
    int flags;
};

class DFTSpectrum : public Stage {
public:
    enum ShowType {
        MAGNITUDE,
        PHASE,
        REAL,
        IMAGINARY
    };

    DFTSpectrum(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        delta = jo_int(pStage, "delta", 1, model.argMap);
        _params["delta"] = new IntParameter(this, delta);
        isShift = jo_bool(pStage, "shift", true);
        _params["shift"] = new BoolParameter(this, isShift);
        isLog = jo_bool(pStage, "log", true);
        _params["log"] = new BoolParameter(this, isLog);

        show = MAGNITUDE;
        mapShow[MAGNITUDE]			= "magnitude";
        mapShow[PHASE]              = "phase";
        mapShow[REAL]				= "real";
        mapShow[IMAGINARY]      	= "imaginary";
        string stype = jo_string(pStage, "show", "magnitude", model.argMap);
        auto findType = std::find_if(std::begin(mapShow), std::end(mapShow), [&](const std::pair<int, string> &pair)
        {
            return stype.compare(pair.second) == 0;
        });
        if (findType != std::end(mapShow)) {
            show = findType->first;
            _params["show"] = new EnumParameter(this, show, mapShow);
        } else
            throw std::invalid_argument("unknown 'show' parameter value");

        isMirror = jo_bool(pStage, "mirror", true);
        _params["mirror"] = new BoolParameter(this, isMirror);
        showStr = jo_string(pStage, "show", "magnitude", model.argMap);
        _params["show"] = new StringParameter(this, showStr);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);

        const char *errMsg = NULL;

        if (!errMsg) {
          if (show == REAL) {
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
            if (show == MAGNITUDE) {
              magnitude(planes[0], planes[1], model.image);
            } else if (show == PHASE) {
              phase(planes[0], planes[1], model.image);
            } else if (show == REAL) {
              model.image = planes[0];
            } else if (show == IMAGINARY) {
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
            dftShift(model.image);
          }
          if (isMirror) {
            dftMirror(model.image);
          }
        }
        return stageOK("apply_dftSpectrum(%s) %s", errMsg, pStage, pStageModel);
    }

    static void dftMirror(Mat &image) {
        int cx = image.cols/2;
        Mat imageL(image,Rect(0,0,cx,image.rows));
        Mat imageR(image,Rect(cx,0,cx,image.rows));
        flip(imageR, imageL, 1);
    }

    static void dftShift(Mat &image) {
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

    int show;
    map<int, string> mapShow;
    int delta;
    bool isShift;
    bool isLog;
    bool isMirror;
    string showStr;
};

class TemplateMatch : public Stage {
    enum Output {
        CURRENT,
        INPUT,
        CORR
    };

public:
    TemplateMatch(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        /* method */
        method = CV_TM_CCOEFF_NORMED;   // default
        mapMethod[CV_TM_SQDIFF]         = "CV_TM_SQDIFF";
        mapMethod[CV_TM_SQDIFF_NORMED]  = "CV_TM_SQDIFF_NORMED";
        mapMethod[CV_TM_CCORR]          = "CV_TM_CCORR";
        mapMethod[CV_TM_CCORR_NORMED]   = "CV_TM_CCORR_NORMED";
        mapMethod[CV_TM_CCOEFF]         = "CV_TM_CCOEFF";
        mapMethod[CV_TM_CCOEFF_NORMED]  = "CV_TM_CCOEFF_NORMED";
        string smethod = jo_string(pStage, "method", mapMethod[method].c_str(), model.argMap);
        auto findMethod = std::find_if(std::begin(mapMethod), std::end(mapMethod), [&](const std::pair<int, string> &pair)
        {
            return smethod.compare(pair.second) == 0;
        });
        if (findMethod != std::end(mapMethod))
            method = findMethod->first;
        _params["method"] = new EnumParameter(this, method, mapMethod);


        /* Border type */
        border = BORDER_REPLICATE;
        mapBorder[BORDER_DEFAULT]	= "Default";
        mapBorder[BORDER_CONSTANT]	= "Constant";
        mapBorder[BORDER_REPLICATE] = "Replicate";
        mapBorder[BORDER_ISOLATED]	= "Isolated";
        mapBorder[BORDER_REFLECT]	= "Reflect";
        mapBorder[BORDER_REFLECT_101] = "Reflect 101";
        mapBorder[BORDER_WRAP]		= "Wrap";
        string sborder = jo_string(pStage, "border", mapBorder[BORDER_DEFAULT].c_str(), model.argMap);
        auto findBorder = std::find_if(std::begin(mapBorder), std::end(mapBorder), [&](const std::pair<int, string> &pair)
        {
            return sborder.compare(pair.second) == 0;
        });
        if (findBorder != std::end(mapBorder))
            border = findBorder->first;
        _params["border"] = new EnumParameter(this, border, mapBorder);


        /* Path to template */
        tmpltPath = jo_string(pStage, "template", "", model.argMap);
        if (tmpltPath.empty())
            throw std::invalid_argument("Expected path to template ('template')");
        _params["tempalte"] = new StringParameter(this, tmpltPath);

        /* Threshold */
        threshold = jo_float(pStage, "threshold", 0.7f, model.argMap);
        _params["threshold"] = new FloatParameter(this, threshold);


        corr = jo_float(pStage, "corr", 0.85f, model.argMap);
        _params["corr"] = new FloatParameter(this, corr);

        output = CURRENT;
        mapOutput[CURRENT]          = "current";
        mapOutput[INPUT]            = "input";
        mapOutput[CORR]             = "corr";
        string soutput = jo_string(pStage, "output", mapOutput[output].c_str(), model.argMap);
        auto findOutput = std::find_if(std::begin(mapOutput), std::end(mapOutput), [&](const std::pair<int, string> &pair)
        {
            return soutput.compare(pair.second) == 0;
        });
        if (findOutput != std::end(mapOutput))
            output = findOutput->first;
        else
            throw std::invalid_argument("Expected \"output\" value: input, current, or corr");
        _params["output"] = new EnumParameter(this, output, mapOutput);


        // TODO parametrize
        angles = jo_vectorf(pStage, "angles", vector<float>(), model.argMap);
        if (angles.size() == 0) {
            angles = jo_vectorf(pStage, "angle", vector<float>(), model.argMap);
        }
        if (angles.size() == 0) {
            float angle = jo_float(pStage, "angle", 0, model.argMap);
            angles.push_back(angle);
        }
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        const char *errMsg = NULL;

        Mat tmplt;

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
            errMsg = "Failed to read template";
        }

        Mat warpedTmplt;
        if (!errMsg) {
            if (angles.size() > 0) {
                matWarpRing(tmplt, warpedTmplt, angles);
            } else {
                warpedTmplt = tmplt;
            }
        }

        if (!errMsg) {
            Mat result;
            Mat imageSource = output == CURRENT ? model.image.clone() : model.image;

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
                float rangeMin = max(threshold, corr * maxVal);
                float rangeMax = maxVal;
                matMaxima(result, matches, rangeMin, rangeMax);
            }

            int xOffset = output == CORR ? 0 : warpedTmplt.cols/2;
            int yOffset = output == CORR ? 0 : warpedTmplt.rows/2;
            modelMatches(Point(xOffset, yOffset), tmplt, result, angles, matches, pStageModel, maxVal, isMin);

            if (output == CORR) {
                LOGTRACE("apply_matchTemplate() normalize()");
                normalize(result, result, 0, 255, NORM_MINMAX);
                result.convertTo(model.image, CV_8U);
            } else if (output == INPUT) {
                LOGTRACE("apply_matchTemplate() clone input");
                model.image = model.imageMap["input"].clone();
            }
        }

        return stageOK("apply_matchTemplate(%s) %s", errMsg, pStage, pStageModel);
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
            } else {
                LOGTRACE1("Omitting angles (size:%d)", (int) angles.size());
            }
            json_object_set(pRect, "corr", json_float(val/maxVal));
            json_array_append(pRects, pRect);
        }
        json_object_set(pStageModel, "rects", pRects);
        json_object_set(pStageModel, "maxVal", json_float(maxVal));
        json_object_set(pStageModel, "matches", json_integer(matches.size()));
        LOGTRACE("modelMatches() end");
    }


    // method
    int method;
    map<int, string> mapMethod;
    // border
    int border;
    map<int, string> mapBorder;
    // path to template
    string tmpltPath;
    // threshold;
    float threshold;
    float corr;
    int output;
    map<int, string> mapOutput;
    vector<float> angles;
};

class CalcOffset : public Stage {
public:
    CalcOffset(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        tmpltPath = jo_string(pStage, "template", "", model.argMap);
        offsetColor = Scalar(32,32,255);
        offsetColor = jo_Scalar(pStage, "offsetColor", offsetColor, model.argMap);
        xtol = jo_int(pStage, "xtol", 32, model.argMap);
        ytol = jo_int(pStage, "ytol", 32, model.argMap);
        channels = jo_vectori(pStage, "channels", vector<int>(), model.argMap);
        minval = jo_float(pStage, "minval", 0.7f, model.argMap);
        corr = jo_float(pStage, "corr", 0.99f);
        outputStr = jo_string(pStage, "output", "current", model.argMap);
        flags = INTER_LINEAR;
        method = CV_TM_CCOEFF_NORMED;
        borderMode = BORDER_REPLICATE;
    }
protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        string errMsg;

        do {
            if (model.image.cols <= 2*xtol) {
                errMsg = "model.image.cols <= 2*xtol";
                break;
            }

            if (model.image.rows <= 2*ytol) {
                errMsg = "model.image.rows <= 2*ytol";
                break;
            }

            Rect roi= jo_Rect(pStage, "roi", Rect(xtol, ytol, model.image.cols-2*xtol, model.image.rows-2*ytol), model.argMap);
            if (roi.x == -1) {
                roi.x = (model.image.cols - roi.width)/2;
            }
            if (roi.y == -1) {
                roi.y = (model.image.rows - roi.height)/2;
            }
            Rect roiScan = Rect(roi.x-xtol, roi.y-ytol, roi.width+2*xtol, roi.height+2*ytol);

            Mat tmplt;

            if (roiScan.x < 0 || roiScan.y < 0 || model.image.cols < roiScan.x+roiScan.width || model.image.rows < roiScan.y+roiScan.height) {
                errMsg = "ROI with and surrounding xtol,ytol region must be within image";
                break;
            }

            if (tmpltPath.empty()) {
                errMsg = "Expected template path for imread";
                break;
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
                        break;
                    }
                } else {
                    errMsg = "imread failed: " + tmpltPath;
                    break;
                }
            }

            if (model.image.channels() > 3) {
                errMsg = "Expected at most 3 channels for pipeline image";
                break;
            } else if (tmplt.channels() != model.image.channels()) {
                errMsg = "Template and pipeline image must have same number of channels";
                break;
            } else {
                for (int iChannel = 0; iChannel < channels.size(); iChannel++) {
                    if (channels[iChannel] < 0 || model.image.channels() <= channels[iChannel]) {
                        errMsg = "Referenced channel is not in image";
                        break;
                    }
                }
            }

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
        } while (0);

        return stageOK("apply_calcOffset(%s) %s", errMsg.c_str(), pStage, pStageModel);
    }

    string tmpltPath;
    Scalar offsetColor;
    int xtol;
    int ytol;
    vector<int> channels;
    float minval;
    float corr;
    string outputStr;
    int flags;
    int method;
    int borderMode;
};

}

#endif // FOURIER_H
