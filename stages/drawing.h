#ifndef DRAWING_H
#define DRAWING_H

/*
 * @Author  : Simon Fojtu
 * @Date    : 23.06.2015
 */

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class Text : public Stage
{
public:
    Text(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        text = jo_string(pStage, "text", "FireSight", model.argMap);
        _params["text"] = new StringParameter(this, text);
        color = jo_Scalar(pStage, "color", Scalar(0,255,0), model.argMap);
        _params["color"] = new ScalarParameter(this, color);
        fontFaceName = jo_string(pStage, "fontFace", "FONT_HERSHEY_PLAIN", model.argMap);
        _params["fontFace"] = new StringParameter(this, fontFaceName);
        thickness = jo_int(pStage, "thickness", 1, model.argMap);
        _params["thickness"] = new IntParameter(this, thickness);
        fontFace = FONT_HERSHEY_PLAIN;
        italic = jo_bool(pStage, "italic", false, model.argMap);
        _params["italic"] = new BoolParameter(this, italic);
        fontScale = jo_double(pStage, "fontScale", 1, model.argMap);
        _params["fontScale"] = new DoubleParameter(this, fontScale);
        org = jo_Point(pStage, "org", Point(5,-6), model.argMap);
        _params["org"] = new PointParameter(this, org);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model)
    {
        const char *errMsg = NULL;
        validateImage(model.image);

        Point org_(org);
        if (org_.y < 0) {
            org_.y = model.image.rows + org_.y;
        }

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
            putText(model.image, text.c_str(), org_, fontFace, fontScale, color, thickness);
        }

        return stageOK("apply_putText(%s) %s", errMsg, pStage, pStageModel);
    }

protected:
    string text;
    Scalar color;
    string fontFaceName;
    int thickness;
    int fontFace;
    bool italic;
    double fontScale;
    Point org;
};


class DrawRects : public Stage
{
public:
    DrawRects(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        color = jo_Scalar(pStage, "color", Scalar(-1,-1,-1,255), model.argMap);
        _params["color"] = new ScalarParameter(this, color);
        radius = jo_int(pStage, "radius", 0, model.argMap);
        _params["radius"] = new IntParameter(this, radius);
        thickness = jo_int(pStage, "thickness", 2, model.argMap);
        _params["thickness"] = new IntParameter(this, thickness);
        rectsModelName = jo_string(pStage, "model", "", model.argMap);
        _params["model"] = new StringParameter(this, rectsModelName);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model)
    {
        const char *errMsg = NULL;

        json_t *pRectsModel = json_object_get(model.getJson(false), rectsModelName.c_str());

        if (rectsModelName.empty()) {
            errMsg = "model: expected name of stage with rects";
        } else if (!json_is_object(pRectsModel)) {
            cout << "stage name:" << rectsModelName << endl;
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

protected:
    Scalar color;
    int radius;
    int thickness;
    string rectsModelName;
};

class DrawCircle : public Stage
{
public:
    DrawCircle(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        center = jo_Point(pStage, "center", Point(0,0), model.argMap);
        _params["center"] = new PointParameter(this, center);
        radius = jo_int(pStage, "radius", 0, model.argMap);
        _params["radius"] = new IntParameter(this, radius);
        color = jo_Scalar(pStage, "color", Scalar::all(0), model.argMap);
        _params["color"] = new ScalarParameter(this, color);
        thickness = jo_int(pStage, "thickness", 1, model.argMap);
        _params["thickness"] = new IntParameter(this, thickness);
        lineType = jo_int(pStage, "lineType", 8, model.argMap);
        _params["lineType"] = new IntParameter(this, lineType);
        fill = jo_Scalar(pStage, "fill", Scalar::all(-1), model.argMap);
        _params["fill"] = new ScalarParameter(this, fill);
        shift = jo_int(pStage, "shift", 0, model.argMap);
        _params["shift"] = new IntParameter(this, shift);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
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

    Point center;
    int radius;
    Scalar color;
    int thickness;
    int lineType;
    Scalar fill;
    int shift;
};

class DrawRectangle: public Stage
{
public:
    DrawRectangle(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        x = jo_int(pStage, "x", 0, model.argMap);
        _params["x"] = new IntParameter(this, x);
        y = jo_int(pStage, "y", 0, model.argMap);
        _params["y"] = new IntParameter(this, y);

        int defaultWidth = model.image.cols ? model.image.cols : 64;
        int defaultHeight = model.image.rows ? model.image.rows : 64;

        width = jo_int(pStage, "width", defaultWidth, model.argMap);
        _params["width"] = new IntParameter(this, width);
        height = jo_int(pStage, "height", defaultHeight, model.argMap);
        _params["height"] = new IntParameter(this, height);
        thickness = jo_int(pStage, "thickness", 1, model.argMap);
        _params["thickness"] = new IntParameter(this, thickness);
        lineType = jo_int(pStage, "lineType", 8, model.argMap);
        _params["lineType"] = new IntParameter(this, lineType);
        color = jo_Scalar(pStage, "color", Scalar::all(0), model.argMap);
        _params["color"] = new ScalarParameter(this, color);
        flood = jo_Scalar(pStage, "flood", Scalar::all(-1), model.argMap);
        _params["flood"] = new ScalarParameter(this, flood);
        fill = jo_Scalar(pStage, "fill", Scalar::all(-1), model.argMap);
        _params["fill"] = new ScalarParameter(this, fill);
        shift = jo_int(pStage, "shift", 0, model.argMap);
        _params["shift"] = new IntParameter(this, shift);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
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

    int x;
    int y;
    int width;
    int height;
    int thickness;
    int lineType;
    Scalar color;
    Scalar flood;
    Scalar fill;
    int shift;
};

class DrawKeypoints: public Stage
{
public:
    DrawKeypoints(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        color = jo_Scalar(pStage, "color", Scalar::all(-1), model.argMap);
        _params["color"] = new ScalarParameter(this, color);
        flags = jo_int(pStage, "flags", DrawMatchesFlags::DRAW_OVER_OUTIMG|DrawMatchesFlags::DRAW_RICH_KEYPOINTS, model.argMap);
        _params["flags"] = new IntParameter(this, flags);
        modelName = jo_string(pStage, "model", "", model.argMap);
        _params["model"] = new StringParameter(this, modelName);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);

        const char *errMsg = NULL;

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

    Scalar color;
    int flags;
    string modelName;


};


}

#endif // DRAWING_H
