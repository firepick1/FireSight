#ifndef DETECTOR_H
#define DETECTOR_H
/*
 * @Author  : Simon Fojtu
 * @Date    : 26.06.2015
 */

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class PartDetector : public Stage {
public:
    PartDetector(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
    }

private:
    vector<RotatedRect> detect(cv::Mat& image);

    bool apply_internal(json_t *pStageModel, Model &model) {
        const char *errMsg = NULL;

        if (model.image.channels() != 1) {
            errMsg = "PartDetector: single channel image required";
        }

        vector<RotatedRect> rects = PartDetector::detect(model.image);

        json_t *pRects = json_array();
        json_object_set(pStageModel, "rects", pRects);
        for (const auto& rect : rects) {
            json_t *pRect = json_object();
            json_object_set(pRect, "x", json_real(rect.center.x));
            json_object_set(pRect, "y", json_real(rect.center.y));
            json_object_set(pRect, "width", json_real(rect.size.width));
            json_object_set(pRect, "height", json_real(rect.size.height));
            json_object_set(pRect, "angle", json_real(rect.angle));
            json_array_append(pRects, pRect);
        }

        return stageOK("apply_detectParts(%s) %s", errMsg, pStage, pStageModel);
    }

};

class HoleRecognizer : public Stage {
public:
    /**
     * Update the working image to show MSER matches.
     * Image must have at least three channels representing RGB values.
     * @param show matched regions. Default is HOLE_SHOW_NONE
     */
    enum ShowMode {
        SHOW_NONE,      /* do not show matches */
        SHOW_MSER,      /* show all MSER matches */
        SHOW_MATCHES    /* only show MSER matches that meet hole criteria */
    };

    HoleRecognizer(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        float diamMin = jo_float(pStage, "diamMin", 0, model.argMap);
        _params["diamMin"] = new FloatParameter(this, diamMin);
        float diamMax = jo_float(pStage, "diamMax", 0, model.argMap);
        _params["diamMax"] = new FloatParameter(this, diamMax);

        showMatches = SHOW_NONE;
        mapShow[SHOW_NONE] =    "NONE";
        mapShow[SHOW_MSER] =    "MSER";
        mapShow[SHOW_MATCHES]=  "MATCHES";
        string show = jo_string(pStage, "show", "NONE", model.argMap);
        auto findType = std::find_if(std::begin(mapShow), std::end(mapShow), [&](const std::pair<int, string> &pair)
        {
            return show.compare(pair.second) == 0;
        });
        if (findType != std::end(mapShow)) {
            showMatches = findType->first;
            _params["show"] = new EnumParameter(this, showMatches, mapShow);
        } else
            throw std::invalid_argument("unknown 'show'");


        if (diamMin <= 0 || diamMax <= 0 || diamMin > diamMax)
            throw std::invalid_argument("expected: 0 < diamMin < diamMax ");
        if (showMatches < 0)
            throw std::invalid_argument("expected: 0 < showMatches ");

        if (logLevel >= FIRELOG_TRACE) {
            char *pStageJson = json_dumps(pStage, 0);
            LOGTRACE1("apply_HoleRecognizer(%s)", pStageJson);
            free(pStageJson);
        }

        // TODO parametrize?
        delta = 5;
        minArea = (int)(diamMin*diamMin*PI/4); // 60;
        maxArea = (int)(diamMax*diamMax*PI/4); // 14400;
        maxVariation = 0.25;
        minDiversity = (diamMax - diamMin)/(float)diamMin; // 0.2;
        max_evolution = 200;
        area_threshold = 1.01f;
        min_margin = .003f;
        edge_blur_size = 5;
        LOGTRACE3("HoleRecognizer() MSER(minArea:%d maxArea:%d minDiversity:%d/100)", minArea, maxArea, (int)(minDiversity*100+0.5));
        mser = MSER(delta, minArea, maxArea, maxVariation, minDiversity,
            max_evolution, area_threshold, min_margin, edge_blur_size);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        const char *errMsg = NULL;
        vector<MatchedRegion> matches;

        scan(model.image, matches);

        json_t *holes = json_array();
        json_object_set(pStageModel, "holes", holes);
        for (size_t i = 0; i < matches.size(); i++) {
            json_array_append(holes, matches[i].as_json_t());
        }

        return stageOK("apply_HoleRecognizer(%s) %s", errMsg, pStage, pStageModel);
    }

    void scan(Mat &matRGB, vector<MatchedRegion> &matches, float maxEllipse = 1.05, float maxCovar = 2.0);

    float diamMin;
    float diamMax;
    int showMatches;
    map<int, string> mapShow;

private:
//    int _showMatches;
    MSER mser;
//    float minDiam;
//    float maxDiam;
    int delta;
    int minArea;
    int maxArea;
    float maxVariation;
    float minDiversity;
    int max_evolution;
    float area_threshold;
    float min_margin;
    int edge_blur_size;
    void scanRegion(vector<Point> &pts, int i,
    Mat &matRGB, vector<MatchedRegion> &matches, float maxEllipse, float maxCovar);
};


struct Circle {
    float x;
    float y;
    float radius;

    Circle(float x, float y, float radius) {
        this->x = x;
        this->y = y;
        this->radius = radius;
    }
    string asJson() {
        json_t *pObj = as_json_t();
        char *pObjStr = json_dumps(pObj, JSON_PRESERVE_ORDER|JSON_COMPACT|JSON_INDENT(2));
        string result(pObjStr);
        return result;
    }
    json_t *as_json_t() {
        json_t *pObj = json_object();

        json_object_set(pObj, "x", json_real(x));
        json_object_set(pObj, "y", json_real(y));
        json_object_set(pObj, "radius", json_real(radius));

        return pObj;
    }
};

/**
 * @brief Use on grayscale images. Use some filter (bilateral) before this stage.
 */
class HoughCircles : public Stage {
public:
    HoughCircles(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        diamMin = jo_int(pStage, "diamMin", 0, model.argMap);
        // TODO parametrize
        diamMax = jo_int(pStage, "diamMax", 0, model.argMap);
        showCircles = jo_bool(pStage, "show", false, model.argMap);
        // alg. parameters

        hc_dp = jo_double(pStage, "houghcircles_dp", 1, model.argMap);
        hc_minDist = jo_double(pStage, "houghcircles_minDist", 10, model.argMap);
        hc_param1 = jo_double(pStage, "houghcircles_param1", 80, model.argMap);
        hc_param2 = jo_double(pStage, "houghcircles_param2", 10, model.argMap);

        if (diamMin <= 0 || diamMax <= 0 || diamMin > diamMax)
            throw std::invalid_argument("expected: 0 < diamMin < diamMax ");

        if (logLevel >= FIRELOG_TRACE) {
            char *pStageJson = json_dumps(pStage, 0);
            LOGTRACE1("apply_HoughCircles(%s)", pStageJson);
            free(pStageJson);
        }

        LOGTRACE2("HoughCircle() (maxDiam:%d minDiam:%d)", diamMax, diamMin);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);

        const char *errMsg = NULL;

        vector<Circle> circles;

        scan(model.image, circles);

        json_t *circles_json = json_array();
        json_object_set(pStageModel, "circles", circles_json);
        for (size_t i = 0; i < circles.size(); i++) {
            json_array_append(circles_json, circles[i].as_json_t());
        }

        return stageOK("apply_HoughCircles(%s) %s", errMsg, pStage, pStageModel);
    }

    void scan(Mat &image, vector<Circle> &circles);
    void show(Mat & image, vector<Circle> circles);

    int diamMin;
    int diamMax;
    bool showCircles;

    // alg. parameters
    double hc_dp;
    double hc_minDist;
    double hc_param1;
    double hc_param2;
};

}


#endif // DETECTOR_H
