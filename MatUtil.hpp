#ifndef MATUTIL_HPP
#define MATUTIL_HPP
#include <string.h>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


std::string matInfo(const cv::Mat &m);

cv::Mat matRotateSize(cv::Size sizeIn, cv::Point2f center, double angle, double &minx, double &maxx, double &miny, double &maxy);

void matRing(const cv::Mat &image, cv::Mat &result, bool grow=1);

void matWarpAffine(
		const cv::Mat &image, 
		cv::Mat &result, 
		cv::Point2f center, 
		double angle, 
		double scale, 
		cv::Point2f offset, 
		cv::Size size, 
		int borderMode=cv::BORDER_REPLICATE, 
		cv::Scalar borderValue=cv::Scalar::all(0), 
		int flags=cv::INTER_LINEAR);

#endif
