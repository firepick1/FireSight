#include <string.h>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <boost/format.hpp>
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
  const char *tmpltPath = jo_string(pStage, "template", NULL);
	const char *errMsg = NULL;
	Mat tmplt;

	assert(0<model.image.rows && 0<model.image.cols);

	if (!tmpltPath) {
		errMsg = "Expected template path for imread";
	} else {
		tmplt = imread(tmpltPath, CV_LOAD_IMAGE_COLOR);
		if (tmplt.data) {
			LOGTRACE3("template %s %dx%d", tmpltPath, tmplt.rows, tmplt.cols);
			if (model.image.rows<tmplt.rows || model.image.cols<tmplt.cols) {
				errMsg = "Expected template smaller than image to match";
			}
		} else {
			errMsg = "imread failed";
		}
	}

	if (!errMsg) {
		// Compute template DFT
		if (tmplt.channels() == 3) {
			cvtColor(tmplt, tmplt, CV_BGR2GRAY, 0);
		}
		assert(tmplt.channels() == 1);
		flip(tmplt,tmplt,-1);
		assert(imwrite("target/flip.jpg", tmplt));
		Mat paddedTmplt;
		copyMakeBorder(tmplt, paddedTmplt, 0, model.image.cols - tmplt.rows, 0, model.image.cols - tmplt.cols, BORDER_CONSTANT, Scalar::all(0));
		assert(paddedTmplt.channels() == 1);
		Mat tmplt32F;
		paddedTmplt.convertTo(tmplt32F, CV_32FC1);
		assert(tmplt32F.channels() == 1);
		Mat dftTmplt32F;
		LOGTRACE("Taking dft of template");
		dft(tmplt32F, dftTmplt32F, DFT_SCALE|DFT_COMPLEX_OUTPUT);
		assert(dftTmplt32F.channels() == 2);

		// Compute image DFT
		if (model.image.channels() == 3) {
			cvtColor(model.image, model.image, CV_BGR2GRAY, 0);
		}
		assert(model.image.channels() == 1);
		Mat image32F;
		if (model.image.depth() == CV_32F) {
			image32F = model.image;
		} else {
			model.image.convertTo(image32F, CV_32F);
		}
		Mat dftImage32F;
		LOGTRACE("Taking dft of image");
		dft(image32F, dftImage32F, DFT_SCALE|DFT_COMPLEX_OUTPUT);

		LOGTRACE("Product for convolution");
		Mat dftProduct32F = dftImage32F * dftTmplt32F;

		model.image = dftProduct32F;
	}

	return stageOK("apply_proto(%s) %s", errMsg, pStage, pStageModel);
}

