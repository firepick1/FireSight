#ifndef MATUTIL_HPP
#define MATUTIL_HPP
#include <string.h>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


std::string matInfo(cv::Mat &m);

void matRing(const cv::Mat &image, cv::Mat &result);

void matWarpAffine(
		cv::Mat &image, 
		cv::Point center, double angle, double scale, 
		cv::Point offset, 
		cv::Size size, 
		cv::Scalar borderValue=cv::Scalar::all(0), 
		int borderMode=cv::BORDER_CONSTANT, 
		int flags=cv::INTER_LINEAR);

#endif
