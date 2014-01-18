/*
FireSightMain.cpp https://github.com/firepick1/FirePick/wiki

Copyright (C) 2013,2014  Karl Lew, <karl@firepick.org>

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
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


