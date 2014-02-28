#ifndef MATUTIL_HPP
#define MATUTIL_HPP
#include <string.h>
#include <vector>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


std::string matInfo(const cv::Mat &m);

cv::Mat matRotateSize(cv::Size sizeIn, cv::Point2f center, float angle, float &minx, float &maxx, float &miny, float &maxy);

void matRing(const cv::Mat &image, cv::Mat &result);

void matWarpRing(const cv::Mat &image, cv::Mat &result, std::vector<float> angles);

void matMaxima(const cv::Mat &mat, std::vector<cv::Point> &locations, float rangeMin=0, float rangeMax=FLT_MAX) ;
void matMinima(const cv::Mat &mat, std::vector<cv::Point> &locations, float rangeMin=0, float rangeMax=FLT_MAX) ;

void matWarpAffine(
		const cv::Mat &image, 
		cv::Mat &result, 
		cv::Point2f center, 
		float angle, 
		float scale, 
		cv::Point2f offset, 
		cv::Size size, 
		int borderMode=cv::BORDER_REPLICATE, 
		cv::Scalar borderValue=cv::Scalar::all(0), 
		int flags=cv::INTER_LINEAR);

#endif
