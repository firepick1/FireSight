#include <string.h>
#include <math.h>
#include "FireLog.h"
#include "FireSight.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"

using namespace cv;
using namespace std;
using namespace FireSight;

HoleRecognizer::HoleRecognizer(float minDiameter, float maxDiameter) {
	maxDiam = maxDiameter;
	minDiam = minDiameter;
	_showMatches = HOLE_SHOW_NONE;
  delta = 5;
	minArea = (int)(minDiameter*minDiameter*M_PI/4); // 60;
	maxArea = (int)(maxDiameter*maxDiameter*M_PI/4); // 14400;
	maxVariation = 0.25;
	minDiversity = (maxDiam - minDiam)/(float)minDiam; // 0.2;
	max_evolution = 200;
	area_threshold = 1.01;
	min_margin = .003;
	edge_blur_size = 5;
	LOGTRACE3("HoleRecognizer() MSER(minArea:%d maxArea:%d minDiversity:%d/100)", minArea, maxArea, (int)(minDiversity*100+0.5));
	mser = MSER(delta, minArea, maxArea, maxVariation, minDiversity,
		max_evolution, area_threshold, min_margin, edge_blur_size);
}

void HoleRecognizer::scanRegion(vector<Point> &pts, int i, 
	Mat &image, vector<MatchedRegion> &matches, float maxEllipse, float maxCovar) 
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
	if (image.channels() >= 3) {
		json = match.asJson();
		LOGTRACE2("HoleRecognizer pts[%d] %s", i, json.c_str());
	}

	int duplicate = 0;
	bool matched = 0;
	if (covar < maxCovar && maxX - minX < maxDiam && maxY - minY < maxDiam) {
		for (int j = 0; !duplicate && j < matches.size(); j++) {
			if (abs(match.average.x - matches[j].average.x) < maxDiam &&
					abs(match.average.y - matches[j].average.y) < maxDiam) 
			{ duplicate++; }
		}
		if (!duplicate && abs(match.ellipse-match.pointCount)/match.ellipse <= maxEllipse) {
			if (image.channels() >= 3) {
				red = 255;
				green = 0;
				blue = 255;
				int n = matches.size() + 1;
				matched = 1;
				LOGDEBUG2("HoleRecognizer %d. %s", n, json.c_str());
			}
			matches.push_back(match);
		}
	}
	if (!duplicate && image.channels() >= 3) {
		if (matched && _showMatches==HOLE_SHOW_MATCHES || _showMatches==HOLE_SHOW_MSER) {
			for (int j = 0; j < nPts; j++) {
				image.at<Vec3b>(pts[j])[0] = red;
				image.at<Vec3b>(pts[j])[1] = green;
				image.at<Vec3b>(pts[j])[2] = blue;
			}
		}
	}
}

void HoleRecognizer::showMatches(int show) {
  _showMatches = show;
}

void HoleRecognizer::scan(Mat &image, vector<MatchedRegion> &matches, float maxEllipse, float maxCovar) {
	Mat matGray;
	if (image.channels() == 1) {
		image = matGray;
	} else {
		cvtColor(image, matGray, CV_RGB2GRAY);
	}
	
	Mat mask;
	vector<vector<Point> > regions;
	LOGTRACE1("HoleRecognizer::scan() mser()", NULL);
	mser(matGray, regions, mask);

	int nRegions = (int) regions.size();
	LOGTRACE1("HoleRecognizer::scan() -> matched %d regions", nRegions);
	for( int i = 0; i < nRegions; i++) {
		scanRegion(regions[i], i, image, matches, maxEllipse, maxCovar);
	}
}
