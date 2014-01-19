/*
FireSightMain.cpp https://github.com/firepick1/FireSight/wiki

Copyright (C) 2013,2014  Karl Lew, <karl@firepick.org>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#include <string.h>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <boost/format.hpp>
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
using namespace FireSight;


int main(int argc, char *argv[])
{
	cout << "FireSightMain" << endl;
	Mat matRGB = imread("/dev/firefuse/cam.jpg", CV_LOAD_IMAGE_COLOR);
	cout << "matRGB " << matRGB.cols << "x" << matRGB.rows << endl;
	vector<MatchedRegion> matches;

	HoleRecognizer recognizer(26/1.15, 26*1.15);
  recognizer.scan(matRGB, matches);

	imwrite("/home/pi/camcv.bmp", matRGB);

	for (int i = 0; i < matches.size(); i++) {
		cout << "match " << i+1 << " = " << matches[i].asJson() << endl;
	}

  return 0;
}


