/*
FireSight.cpp https://github.com/firepick1/FirePick/wiki

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
#include "FireLog.h"
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
using namespace FireSight;

float pi = boost::math::constants::pi<float>();

MatchedRegion::MatchedRegion(Range xRange, Range yRange, Point2f average, int pointCount, float covar) {
	this->xRange = xRange;
	this->yRange = yRange;
	this->average = average;
	this->pointCount = pointCount;
	this->ellipse = (xRange.end-xRange.start+1) * (yRange.end-yRange.start+1) * pi/4;
	this->covar = covar;
}

string MatchedRegion::asJson() {
	boost::format fmt(
		"{"
			" \"x\":{\"min\":%d,\"max\":%d,\"avg\":%.2f}"
			",\"y\":{\"min\":%d,\"max\":%d,\"avg\":%.2f}"
			",\"pts\":%d"
			",\"ellipse\":%.2f"
			",\"covar\":%.2f"
		" }"
	);
	fmt % xRange.start;
	fmt % xRange.end;
	fmt % average.x;
	fmt % yRange.start;
	fmt % yRange.end;
	fmt % average.y;
	fmt % pointCount;
	fmt % ellipse;
	fmt % covar;
	return fmt.str();
}

HoleRecognizer::HoleRecognizer(float minDiameter, float maxDiameter) {
	maxDiam = maxDiameter;
	minDiam = minDiameter;
  delta = 5;
	minArea = (int)(minDiameter*minDiameter*pi/4); // 60;
	maxArea = (int)(maxDiameter*maxDiameter*pi/4); // 14400;
	maxVariation = 0.25;
	minDiversity = (maxDiam - minDiam)/(float)minDiam; // 0.2;
	LOGDEBUG3("MSER minArea:%d maxArea:%d minDiversity:%d/100", minArea, maxArea, (int)(minDiversity*100+0.5));
	max_evolution = 200;
	area_threshold = 1.01;
	min_margin = .003;
	edge_blur_size = 5;
	mser = MSER(delta, minArea, maxArea, maxVariation, minDiversity,
		max_evolution, area_threshold, min_margin, edge_blur_size);
}

void HoleRecognizer::scanRegion(vector<Point> &pts, int i, 
	Mat &matRGB, vector<MatchedRegion> &matches, float maxEllipse, float maxCovar) 
{
	int nPts = pts.size();
	int minX = 0x7fff;
	int maxX = 0;
	int minY = 0x7fff;
	int maxY = 0;
	int totalX = 0;
	int totalY = 0;
	float totalXY = 0;
	for (int j = 0; j < nPts; j++) {
		if (pts[j].x < minX) { minX = pts[j].x; }
		if (pts[j].y < minY) { minY = pts[j].y; }
		if (pts[j].x > maxX) { maxX = pts[j].x; }
		if (pts[j].y > maxY) { maxY = pts[j].y; }
		totalX += pts[j].x;
		totalY += pts[j].y;
		totalXY += pts[j].x * pts[j].y;
	}
	float avgX = totalX / (float) nPts;
	float avgY = totalY / (float) nPts;
	float covar = abs(totalXY / nPts - avgX * avgY);
	MatchedRegion match(Range(minX, maxX), Range(minY, maxY), Point2f(avgX, avgY), nPts, covar);
	string json;
	int red = (i & 1) ? 0 : 255;
	int green = (i & 2) ? 128 : 192;
	int blue = (i & 1) ? 255 : 0;
	if (matRGB.channels() >= 3) {
		json = match.asJson();
		LOGTRACE2("HoleRecognizer pts[%d] %s", i, json.c_str());
	}

	int duplicate = 0;
	if (covar < maxCovar && maxX - minX < maxDiam && maxY - minY < maxDiam) {
		for (int j = 0; !duplicate && j < matches.size(); j++) {
			if (abs(match.average.x - matches[j].average.x) < maxDiam &&
					abs(match.average.y - matches[j].average.y) < maxDiam) 
			{ duplicate++; }
		}
		if (!duplicate && abs(match.ellipse-match.pointCount)/match.ellipse <= maxEllipse) {
			if (matRGB.channels() >= 3) {
				red = 255;
				green = 0;
				blue = 255;
				int n = matches.size() + 1;
				LOGDEBUG2("HoleRecognizer %d. %s", n, json.c_str());
			}
			matches.push_back(match);
		}
	}
	if (!duplicate && matRGB.channels() >= 3) {
		for (int j = 0; j < nPts; j++) {
			matRGB.at<Vec3b>(pts[j])[0] = red;
			matRGB.at<Vec3b>(pts[j])[1] = green;
			matRGB.at<Vec3b>(pts[j])[2] = blue;
		}
	}
}

void HoleRecognizer::scan(Mat &matRGB, vector<MatchedRegion> &matches, float maxEllipse, float maxCovar) {
	Mat matGray;
	if (matRGB.channels() == 1) {
		matRGB = matGray;
	} else {
		cvtColor(matRGB, matGray, CV_RGB2GRAY);
	}
	
	Mat mask;
	vector<vector<Point> > regions;
	mser(matGray, regions, mask);

	int nRegions = (int) regions.size();
	LOGTRACE1("HoleRecognizer::scan() -> matched %d regions", nRegions);
	for( int i = 0; i < nRegions; i++) {
		scanRegion(regions[i], i, matRGB, matches, maxEllipse, maxCovar);
	}
}

