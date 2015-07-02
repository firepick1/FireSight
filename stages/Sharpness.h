/*
 * @Author  : Simon Fojtu
 * @Date    : 17.02.2015
 */

#ifndef __SHARPNESS_H_
#define __SHARPNESS_H_

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

class Sharpness : public Stage {
public:
    enum mapMethod {
        METHOD_GRAS,
        METHOD_LAPE,
        METHOD_LAPM
    };

    Sharpness(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        method = METHOD_GRAS;
        mapMethod[METHOD_GRAS] = "GRAS";
        mapMethod[METHOD_LAPE] = "LAPE";
        mapMethod[METHOD_LAPM] = "LAPM";
        string smethod = jo_string(pStage, "method", mapMethod[method].c_str(), model.argMap);
        auto findMethod = std::find_if(std::begin(mapMethod), std::end(mapMethod), [&](const std::pair<int, string> &pair)
        {
            return smethod.compare(pair.second) == 0;
        });
        if (findMethod != std::end(mapMethod))
            method = findMethod->first;
        else
            throw std::invalid_argument("Expected method value: {GRAS,LAPE,LAPM}");
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        const char *errMsg = NULL;

        /* Apply selected method */
        double sharpness = 0;
        switch (method) {
        case METHOD_GRAS:
            sharpness = Sharpness::GRAS(model.image);
            break;
        case METHOD_LAPE:
            sharpness = Sharpness::LAPE(model.image);
            break;
        case METHOD_LAPM:
            sharpness = Sharpness::LAPM(model.image);
            break;
        }

        json_object_set(pStageModel, "sharpness", json_real(sharpness));

        return stageOK("apply_sharpness(%s) %s", errMsg, pStage, pStageModel);
    }

    int method;
    map<int, string> mapMethod;

    /**
     * GRAS - Absolute squared gradient
     *
     * A.M. Eskicioglu and P. S. Fisher. Image quality measures and their performance. Communications, IEEE Transactions on, 43(12):2959–2965, 1995
     *
     * MATLAB:
     * Ix = diff(Image, 1, 2);
     * FM = Ix.^2;
     * FM = mean2(FM);
     */
    static double GRAS(cv::Mat & image);

    /**
     * LAPE - Energy of laplacian [Subbarao92a]
     *
     * MATLAB:
     * LAP = fspecial('laplacian');
     * FM = imfilter(Image, LAP, 'replicate', 'conv');
     * FM = mean2(FM.^2);
     */
    static double LAPE(cv::Mat & image);

    /**
     * LAPM - Modified Laplacian [Nayar89]
     *
     * MATLAB:
     * M = [-1 2 -1];        
     * Lx = imfilter(Image, M, 'replicate', 'conv');
     * Ly = imfilter(Image, M', 'replicate', 'conv');
     * FM = abs(Lx) + abs(Ly);
     * FM = mean2(FM);
     */
    static double LAPM(cv::Mat & image);
};

}

#endif

