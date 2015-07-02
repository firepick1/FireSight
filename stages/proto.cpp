#include "proto.h"

#include <string.h>
#include <iostream>
#include <math.h>
#include "FireLog.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "jo_util.hpp"
#include "MatUtil.hpp"

using namespace cv;
using namespace std;

namespace firesight {

bool Proto::apply_internal(json_t *pStageModel, Model &model)
{
    const char *errMsg = NULL;
    int tmpltWH = 2+max(width, height);
    int cx = tmpltWH/2;
    int cy = tmpltWH/2;
    cout << "cx:" << cx << " cy:" << cy << endl;
    Mat tmplt(tmpltWH, tmpltWH, CV_8U, Scalar(0));
    cout << matInfo(tmplt) << endl << tmplt << endl;
    cout << "cx2:" << cx-width/2 << " cy2:" << cy-height/2 << endl;
    tmplt(Rect(cx-width/2, cy-height/2, width, height)) += 127;
    cout << matInfo(tmplt) << endl << tmplt << endl;
    tmplt(Rect(cx-height/2, cy-width/2, height, width)) += 127;
    cout << matInfo(tmplt) << endl << tmplt << endl;

    if (!errMsg) {
        model.image = tmplt;
    }

    return stageOK("apply_proto(%s) %s", errMsg, pStage, pStageModel);
}

}
