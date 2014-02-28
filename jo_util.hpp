#ifndef JO_UTIL_HPP
#define JO_UTIL_HPP

#include "opencv2/features2d/features2d.hpp"
#include <vector>
#include "jansson.h"
#include "FireSight.hpp"
#include "winjunk.hpp"

using namespace cv;
using namespace std;

namespace firesight {
	CLASS_DECLSPEC bool jo_bool(const json_t *pObj, const char *key, bool defaultValue=0, ArgMap &argMap=emptyMap) ;

	CLASS_DECLSPEC int jo_int(const json_t *pObj, const char *key, int defaultValue=0, ArgMap &argMap=emptyMap) ;

	CLASS_DECLSPEC double jo_double(const json_t *pObj, const char *key, double defaultValue=0, ArgMap &argMap=emptyMap) ;

	CLASS_DECLSPEC inline float jo_float(const json_t *pObj, const char *key, double defaultValue=0, ArgMap &argMap=emptyMap) {
			return (float) jo_double(pObj, key, defaultValue, argMap);
	}

	CLASS_DECLSPEC string jo_string(const json_t *pObj, const char *key, const char *defaultValue = "", ArgMap &argMap=emptyMap) ;

	CLASS_DECLSPEC Scalar jo_Scalar(const json_t *pObj, const char *key, const Scalar &defaultValue, ArgMap &argMap=emptyMap) ;

	CLASS_DECLSPEC int jo_shape(json_t *pStage, const char *key, const char *&errMsg, ArgMap &argMap=emptyMap) ;

  CLASS_DECLSPEC string jo_parse(const char * pSource, ArgMap &argMap=emptyMap);

	CLASS_DECLSPEC json_t *jo_object(const json_t *pStage, const char *key, ArgMap &argMap=emptyMap) ;

} // namespace firesight

#endif
