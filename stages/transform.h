#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include "MatUtil.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class WarpAffine : public Stage
{
public:
    WarpAffine(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        scale = jo_float(pStage, "scale", 1, model.argMap);
        _params["scale"] = new FloatParameter(this, scale);
        angle = jo_float(pStage, "angle", 0, model.argMap);
        _params["angle"] = new FloatParameter(this, angle);
        dx = jo_int(pStage, "dx", (int)((scale-1)*model.image.cols/2.0+.5), model.argMap);
        _params["dx"] = new IntParameter(this, dx);
        dy = jo_int(pStage, "dy", (int)((scale-1)*model.image.rows/2.0+.5), model.argMap);
        _params["dy"] = new IntParameter(this, dy);
        reflect = jo_Point2f(pStage, "reflect", Point(0,0), model.argMap);

        /* Border type */
        border = BORDER_REPLICATE; //!< default value
        string sborder = jo_string(pStage, "borderMode", BorderTypeParser::get(border).c_str(), model.argMap);
        border = BorderTypeParser::get(sborder);
        mapBorder = BorderTypeParser::get();
        _params["Border"] = new EnumParameter(this, border, mapBorder);


        if (reflect.x && reflect.y && reflect.x != reflect.y) {
            throw std::invalid_argument("warpAffine only handles reflections around x- or y-axes");
        }


        if (scale <= 0) {
            throw std::invalid_argument("Expected 0<scale");
        }

        width = jo_int(pStage, "width", model.image.cols, model.argMap);
        _params["width"] = new IntParameter(this, width);
        height = jo_int(pStage, "height", model.image.rows, model.argMap);
        _params["height"] = new IntParameter(this, height);
        cx = jo_int(pStage, "cx", (int)(0.5+model.image.cols/2.0), model.argMap);
        _params["cx"] = new IntParameter(this, cx);
        cy = jo_int(pStage, "cy", (int)(0.5+model.image.rows/2.0), model.argMap);
        _params["cy"] = new IntParameter(this, cy);
        borderValue = jo_Scalar(pStage, "borderValue", Scalar::all(0), model.argMap);
        _params["borderValue"] = new ScalarParameter(this, borderValue);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model)
    {
        validateImage(model.image);
        const char *errMsg = NULL;

        Mat result;
        int flags=cv::INTER_LINEAR || WARP_INVERSE_MAP;
        matWarpAffine(model.image, result, Point(cx,cy), angle, scale, Point(dx,dy), Size(width,height),
            border, borderValue, reflect, flags);
        model.image = result;

        return stageOK("apply_warpAffine(%s) %s", errMsg, pStage, pStageModel);
    }

protected:
    float scale;
    float angle;
    int dx;
    int dy;
    Point2f reflect;
    int border; //< Border type
    map<int, string> mapBorder;
    int width;
    int height;
    int cx;
    int cy;
    Scalar borderValue;

};

class WarpRing : public Stage
{
public:
    WarpRing(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model)
    {
        validateImage(model.image);
        const char *errMsg = NULL;
        json_t *pAngles = jo_object(pStage, "angles", model.argMap);
        vector<float> angles;
        if (json_is_array(pAngles)) {
          size_t index;
          json_t *pAngle;
          json_array_foreach(pAngles, index, pAngle) {
            if (json_is_number(pAngle)) {
              angles.push_back((float) json_number_value(pAngle));
            } else if (json_is_string(pAngle)) {
              float angle = (float) atof(json_string_value(pAngle));
              angles.push_back(angle);
            } else {
              errMsg = "Expected angle values in degrees";
              break;
            }
          }
        } else if (pAngles == NULL) {
          // Ring
        } else {
          errMsg = "Expected JSON array of angles";
        }

        if (!errMsg) {
          Mat result;
          matWarpRing(model.image, result, angles);
          model.image = result;
          json_object_set(pStageModel, "width", json_integer(model.image.cols));
          json_object_set(pStageModel, "height", json_integer(model.image.rows));
        }

        return stageOK("apply_ring(%s) %s", errMsg, pStage, pStageModel);
    }

protected:

};

class WarpPerspective : public Stage
{
public:
    WarpPerspective(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        /* Border type */
        borderMode = BORDER_DEFAULT;
        mapBorder[BORDER_DEFAULT]	= "Default";
        mapBorder[BORDER_CONSTANT]	= "Constant";
        mapBorder[BORDER_REPLICATE] = "Replicate";
        mapBorder[BORDER_ISOLATED]	= "Isolated";
        mapBorder[BORDER_REFLECT]	= "Reflect";
        mapBorder[BORDER_REFLECT_101] = "Reflect 101";
        mapBorder[BORDER_WRAP]		= "Wrap";
        string sborder = jo_string(pStage, "borderMode", "BORDER_REPLICATE", model.argMap);
        auto findBorder = std::find_if(std::begin(mapBorder), std::end(mapBorder), [&](const std::pair<int, string> &pair)
        {
            return sborder.compare(pair.second) == 0;
        });
        if (findBorder != std::end(mapBorder))
            borderMode = findBorder->first;
        _params["border"] = new EnumParameter(this, borderMode, mapBorder);

        modelName = jo_string(pStage, "model", getName().c_str(), model.argMap);
        _params["model"] = new StringParameter(this, modelName);

        borderValue = jo_Scalar(pStage, "borderValue", Scalar::all(0), model.argMap);
        _params["borderValue"] = new ScalarParameter(this, borderValue);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        string errMsg;

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
            Mat result = Mat::zeros(model.image.rows, model.image.cols, model.image.type());
            warpPerspective(model.image, result, matrix, result.size(), cv::INTER_LINEAR, borderMode, borderValue );
            model.image = result;
        }

        return stageOK("apply_warpPerspective(%s) %s", errMsg.c_str(), pStage, pStageModel);
    }

protected:

    string modelName;
    int borderMode; //< Border type
    map<int, string> mapBorder;
    Scalar borderValue;

};

}

#endif // TRANSFORM_H
