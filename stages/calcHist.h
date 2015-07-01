#ifndef CALCHIST_H
#define CALCHIST_H
#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

class CalcHist : public Stage {
public:
    CalcHist(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        rangeMin = jo_float(pStage, "rangeMin", 0, model.argMap);
        _params["rangeMin"] = new FloatParameter(this, rangeMin);
        rangeMax = jo_float(pStage, "rangeMax", 256, model.argMap);
        _params["rangeMax"] = new FloatParameter(this, rangeMax);
        binMin = jo_float(pStage, "binMin", 1, model.argMap);
        _params["binMin"] = new FloatParameter(this, binMin);
        binMax = jo_float(pStage, "binMax", 0, model.argMap);
        _params["binMax"] = new FloatParameter(this, binMax);
        dims = jo_int(pStage, "dims", 1, model.argMap);
        _params["dims"] = new IntParameter(this, dims);
        accumulate = jo_bool(pStage, "accumulate", false, model.argMap);
        _params["accumulate"] = new BoolParameter(this, accumulate);
        nChannels = model.image.channels();
        for (int i = 0; i < nChannels; i++) {
          defaultChannels.push_back(i);
        }
        // TODO parametrize
        histChannels = jo_vectori(pStage, "channels", defaultChannels, model.argMap);
        locations = jo_int(pStage, "locations", 0, model.argMap);
        _params["locations"] = new IntParameter(this, locations);
        bins = jo_int(pStage, "bins", (int)(rangeMax-rangeMin), model.argMap);
        _params["bins"] = new IntParameter(this, bins);
    }
protected:
    bool apply_internal(json_t *pStageModel, Model &model);

    float rangeMin;
    float rangeMax;
    float binMax;
    float binMin;
    int dims;
    bool accumulate;
    int nChannels;
    vector<int> defaultChannels;
    vector<int> histChannels;
    int locations;
    int bins;
};

}
#endif // CALCHIST_H
