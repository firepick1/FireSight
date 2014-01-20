
#include <string.h>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <boost/format.hpp>
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"

using namespace cv;
using namespace std;
using namespace FireSight;


int main(int argc, char *argv[])
{
	cout << "FireSightMain" << endl;
	Mat matRGB = imread("/dev/firefuse/cam.jpg", CV_LOAD_IMAGE_COLOR);
	cout << "matRGB " << matRGB.cols << "x" << matRGB.rows << endl;
	vector<MatchedRegion> matches;

	json_t *pNode = json_pack("{s:s}", "color", "blue");
	size_t dumpsFlags = 0;
	char *pJson = json_dumps(pNode, dumpsFlags);

	if (pJson) {
		cout << pJson << endl;
		free(pJson);
	} else {
		cout << "json_dumps() error: " << endl;
	}

	HoleRecognizer recognizer(26/1.15, 26*1.15);
  recognizer.scan(matRGB, matches);

	imwrite("/home/pi/camcv.bmp", matRGB);

	for (int i = 0; i < matches.size(); i++) {
		cout << "match " << i+1 << " = " << matches[i].asJson() << endl;
	}

  return 0;
}


