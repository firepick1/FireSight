#ifndef MORPH_H
#define MORPH_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class Morph : public Stage
{
public:
    Morph(json_t *pStage, Model &model, string pName)
        : Stage(pStage, pName)
    {
        parseArgs(pStage, model);
    }

    void parseArgs(json_t *pStage, Model &model, int morph = MORPH_OPEN, string fmt_ = "apply_morph(%s) %s") {
        fmt = fmt_;
        /* Morph type */
        morphOp = morph;
        mapMorph[MORPH_ERODE]	= "MORPH_ERODE";
        mapMorph[MORPH_DILATE]	= "MORPH_DILATE";
        mapMorph[MORPH_OPEN] = "MORPH_OPEN";
        mapMorph[MORPH_CLOSE]	= "MORPH_CLOSE";
        mapMorph[MORPH_GRADIENT]	= "MORPH_GRADIENT";
        mapMorph[MORPH_TOPHAT] = "MORPH_TOPHAT";
        mapMorph[MORPH_BLACKHAT]		= "MORPH_BLACKHAT";
        string smop = jo_string(pStage, "mop", mapMorph[morphOp].c_str(), model.argMap);
        auto findMorph = std::find_if(std::begin(mapMorph), std::end(mapMorph), [&](const std::pair<int, string> &pair)
        {
            return smop.compare(pair.second) == 0;
        });
        if (findMorph != std::end(mapMorph))
            morphOp = findMorph->first;
        else
            throw std::invalid_argument("Unknown morphology operation: " + smop);
        _params["mop"] = new EnumParameter(this, morphOp, mapMorph);


        // TODO parametrize
        vector<int> ksize = jo_vectori(pStage, "ksize", vector<int>(2,3), model.argMap);
        if (ksize.size() == 1) {
          ksize.push_back(ksize[0]);
        }
        if (ksize.size() > 2)
            throw std::invalid_argument("Expected JSON [width,height] array for ksize");

        kwidth = jo_int(pStage, "ksize.width", ksize[0], model.argMap);
        kwidth = jo_int(pStage, "kwidth", kwidth, model.argMap);
        _params["kwidth"] = new IntParameter(this, kwidth);
        kheight = jo_int(pStage, "ksize.height", ksize[1], model.argMap);
        kheight = jo_int(pStage, "kheight", kheight, model.argMap);
        _params["kheight"] = new IntParameter(this, kheight);

        /* Morph shape */
        shape = MORPH_ELLIPSE;
        mapShape[MORPH_ELLIPSE] = "MORPH_ELLIPSE";
        mapShape[MORPH_CROSS] = "MORPH_CROSS";
        mapShape[MORPH_RECT] = "MORPH_RECT";
        string shapeStr = jo_string(pStage, "shape", "MORPH_ELLIPSE", model.argMap);
        auto findShape = std::find_if(std::begin(mapShape), std::end(mapShape), [&](const std::pair<int, string> &pair)
        {
            return shapeStr.compare(pair.second) == 0;
        });
        if (findShape != std::end(mapShape))
            morphOp = findMorph->first;
        else
            throw std::invalid_argument("Unknown morphology shape: " + shapeStr);
        _params["mop"] = new EnumParameter(this, morphOp, mapShape);

        iterations = jo_int(pStage, "iterations", 1, model.argMap);
        _params["iterations"] = new IntParameter(this, iterations);
        /* Anchor */
        int anchorx = jo_int(pStage, "anchor.x", -1, model.argMap);
        int anchory = jo_int(pStage, "anchor.y", -1, model.argMap);
        anchor = Point(anchorx, anchory);
        _params["anchor"] = new PointParameter(this, anchor);


    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        const char *errMsg = NULL;

        // TODO parametrize this
        vector<int> ksize = jo_vectori(pStage, "ksize", vector<int>(2,3), model.argMap);
        if (ksize.size() == 1) {
          ksize.push_back(ksize[0]);
        }
        if (ksize.size() > 2) {
          errMsg = "Expected JSON [width,height] array for ksize";
        }

        if (!errMsg) {
          if (shapeStr.compare("MORPH_ELLIPSE") == 0) {
            shape = MORPH_ELLIPSE;
          } else if (shapeStr.compare("MORPH_CROSS") == 0) {
            shape = MORPH_CROSS;
          } else if (shapeStr.compare("MORPH_RECT") == 0) {
            shape = MORPH_RECT;
          } else {
            shape = jo_shape(pStage, "shape", errMsg);
          }
        }

        if (!errMsg) {
          Mat structuringElement = getStructuringElement(shape, Size(kwidth, kheight));
          switch (morphOp) {
            case MORPH_ERODE:
          erode(model.image, model.image, structuringElement);
          break;
            case MORPH_DILATE:
          dilate(model.image, model.image, structuringElement);
          break;
            default:
          morphologyEx(model.image, model.image, morphOp, structuringElement, anchor, iterations);
          break;
          }
        }

        return stageOK(fmt.c_str(), errMsg, pStage, pStageModel);
    }

protected:
    string mop;
    int morphOp;
    map<int, string> mapMorph;

    int kwidth;
    int kheight;
    String shapeStr;
    int shape;
    map<int, string> mapShape;
    int iterations;
    Point anchor;

    string fmt;
};

class Erode : public Morph
{
public:
    Erode(json_t *pStage, Model &model, string pName) :
        Morph(pStage, model, pName)
    {
        parseArgs(pStage, model, MORPH_ERODE, "apply_erode(%s) %s");
    }


private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        Morph::apply_internal(pStageModel, model);
    }

protected:

};

class Dilate : public Morph
{
public:
    Dilate(json_t *pStage, Model &model, string pName) :
        Morph(pStage, model, pName)
    {
        parseArgs(pStage, model, MORPH_DILATE, "apply_dilate(%s) %s");
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        Morph::apply_internal(pStageModel, model);
    }

protected:

};

}

#endif // MORPH_H
