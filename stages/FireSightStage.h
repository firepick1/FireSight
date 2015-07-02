#ifndef FIRESIGHTSTAGE_H
#define FIRESIGHTSTAGE_H
/*
 * @Author  : Simon Fojtu
 * @Date    : 27.06.2015
 */

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class FireSightStage : public Stage
{
public:
    FireSightStage(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) { }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        json_t *pFireSight = json_object();
        const char *errMsg = NULL;
        char version[100];
        snprintf(version, sizeof(version), "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
        json_object_set(pFireSight, "version", json_string(version));
        json_object_set(pFireSight, "url", json_string("https://github.com/firepick1/FireSight"));
        snprintf(version, sizeof(version), "%d.%d.%d", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
        json_object_set(pFireSight, "opencv", json_string(version));
        json_object_set(pStageModel, "FireSight", pFireSight);

        return stageOK("apply_FireSight(%s) %s", errMsg, pStage, pStageModel);
    }
};

}

#endif // FIRESIGHTSTAGE_H
