#include <string.h>
#include <iostream>
#include <math.h>
#include "FireLog.h"
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "jo_util.hpp"
#include "MatUtil.hpp"

using namespace cv;
using namespace std;
using namespace FireSight;


bool Pipeline::apply_proto(json_t *pStage, json_t *pStageModel, Model &model) {
	int width = jo_int(pStage, "width", 14);
	int height = jo_int(pStage, "height", 21);
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

