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

}

#endif // DETECTOR_H
