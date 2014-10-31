#ifndef MATUTIL_HPP
#define MATUTIL_HPP
#include <string.h>
#include <vector>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "winjunk.hpp"


CLASS_DECLSPEC std::string matInfo(const cv::Mat &m);

CLASS_DECLSPEC cv::Mat matRotateSize(cv::Size sizeIn, cv::Point2f center, float angle, float &minx, float &maxx, float &miny, float &maxy, float scale);

CLASS_DECLSPEC void matRing(const cv::Mat &image, cv::Mat &result);

CLASS_DECLSPEC void matWarpRing(const cv::Mat &image, cv::Mat &result, std::vector<float> angles);

CLASS_DECLSPEC void matMaxima(const cv::Mat &mat, std::vector<cv::Point> &locations, float rangeMin=0, float rangeMax=FLT_MAX) ;
CLASS_DECLSPEC void matMinima(const cv::Mat &mat, std::vector<cv::Point> &locations, float rangeMin=0, float rangeMax=FLT_MAX) ;

CLASS_DECLSPEC void matWarpAffine(
		const cv::Mat &image, 
		cv::Mat &result, 
		cv::Point2f center, 
		float angle, 
		float scale, 
		cv::Point2f offset, 
		cv::Size size, 
		int borderMode=cv::BORDER_REPLICATE, 
		cv::Scalar borderValue=cv::Scalar::all(0), 
		cv::Point2f reflect=cv::Point2f(0,0),
		int flags=cv::INTER_LINEAR);

#endif
