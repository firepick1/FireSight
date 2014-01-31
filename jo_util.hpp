#ifndef JO_UTIL_HPP
#define JO_UTIL_HPP

#include "opencv2/features2d/features2d.hpp"
#include <vector>
#include "jansson.h"

using namespace cv;
using namespace std;

namespace FireSight {

	const bool jo_bool(const json_t *pObj, const char *key, bool defaultValue=0) ;

	const int jo_int(const json_t *pObj, const char *key, int defaultValue=0) ;

	const double jo_double(const json_t *pObj, const char *key, double defaultValue=0) ;

	const char * jo_string(const json_t *pObj, const char *key, const char *defaultValue = NULL) ;

	Scalar jo_Scalar(const json_t *pObj, const char *key, const Scalar &defaultValue) ;

	int jo_shape(json_t *pStage, const char *key, const char *&errMsg) ;

} // namespace FireSight

#endif
