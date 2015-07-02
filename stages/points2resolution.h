#ifndef POINTS2RESOLUTION_H
#define POINTS2RESOLUTION_H

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

typedef class Pt2Res {
    public:

        Pt2Res() {}
        double getResolution(double thr1, double thr2, double confidence, double separation, vector<XY> coords);
    private:
        static bool compare_XY_by_x(XY a, XY b);
        static bool compare_XY_by_y(XY a, XY b);
        int nsamples_RANSAC(size_t ninl, size_t xlen, unsigned int NSAMPL, double confidence);
        static double _RANSAC_line(XY * x, size_t nx, XY C);
        static double _RANSAC_pattern(XY * x, size_t nx, XY C);
        vector<XY> RANSAC_2D(unsigned int NSAMPL, vector<XY> coords, double thr, double confidence, double(*err_fun)(XY *, size_t, XY));
        void least_squares(vector<XY> xy, double * a, double * b);

} Pt2Res;

class Points2Resolution: public Stage {
public:
    Points2Resolution(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        // input parameters
        thr1 = jo_double(pStage, "threshold1", 1.2, model.argMap);
        _params["threshold1"] = new DoubleParameter(this, thr1);
        thr2 = jo_double(pStage, "threshold2", 1.2, model.argMap);
        _params["threshold2"] = new DoubleParameter(this, thr2);
        confidence = jo_double(pStage, "confidence", (1.0-1e-12), model.argMap);
        _params["confidence"] = new DoubleParameter(this, confidence);
        separation = jo_double(pStage, "separation", 4.0, model.argMap); // separation [mm]
        _params["separation"] = new DoubleParameter(this, separation);
        pointsModelName = jo_string(pStage, "model", "", model.argMap);
        _params["model"] = new StringParameter(this, pointsModelName);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {

        char *errMsg = NULL;

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

    // input parameters
    double thr1;
    double thr2;
    double confidence;
    double separation;

    string pointsModelName;

};

}

#endif // POINTS2RESOLUTION_H
