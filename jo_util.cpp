#include <string.h>
#include <math.h>
#include "FireLog.h"
#include "jansson.h"
#include "jo_util.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

#define START_DELIM "{{"
#define END_DELIM "}}"
#define DELIM_SIZE 2


namespace firesight {

ArgMap emptyMap;

string jo_parse(const char * pSource, ArgMap &argMap) {
	string result(pSource);
	if (argMap.size() == 0) {
		return result;
	}

	size_t varStart = 0;
	while ((varStart=result.find(START_DELIM,varStart)) != string::npos) {
		size_t nameEnd = result.find(END_DELIM,varStart);
		if (nameEnd < 0) {
			LOGERROR1("jo_parse(): Invalid variable specification: '%s'", pSource);
			break;
		} 
		int nameStart = varStart + DELIM_SIZE;
		string name = result.substr(nameStart,nameEnd-nameStart);
		const char * pRep = argMap[name.c_str()];
		if (!pRep) {
			LOGERROR1("jo_parse(): Undefined variable: %s", name.c_str());
			break;
		}
		int varEnd = nameEnd + DELIM_SIZE;
		result.replace(varStart, varEnd-varStart, pRep);
		varStart = varEnd;
	}

	LOGTRACE2("jo_parse(%s) => %s",  pSource, result.c_str());
	return result;
}


bool jo_bool(const json_t *pObj, const char *key, bool defaultValue, ArgMap &argMap) {
	json_t *pValue = json_object_get(pObj, key);
	bool result = pValue ? json_is_true(pValue) : defaultValue;
	LOGTRACE3("jo_bool(key:%s default:%d) -> %d", key, defaultValue, result);
	return result;
}

int jo_int(const json_t *pObj, const char *key, int defaultValue, ArgMap &argMap) {
	json_t *pValue = json_object_get(pObj, key);
	int result;
	if (json_is_integer(pValue)) {
		result = json_integer_value(pValue);
	} else if (json_is_string(pValue)) {
		sscanf(json_string_value(pValue), "%d", &result);
	} else {
		result = defaultValue;
	}
	LOGTRACE3("jo_int(key:%s default:%d) -> %d", key, defaultValue, result);
	return result;
}

double jo_double(const json_t *pObj, const char *key, double defaultValue, ArgMap &argMap) {
	json_t *pValue = json_object_get(pObj, key);
	double result = json_is_number(pValue) ? json_number_value(pValue) : defaultValue;
	LOGTRACE3("jo_double(key:%s default:%f) -> %f", key, defaultValue, result);
	return result;
}

string jo_string(const json_t *pObj, const char *key, const char *defaultValue, ArgMap &argMap) {
	json_t *pValue = json_object_get(pObj, key);
	const char * result = json_is_string(pValue) ? json_string_value(pValue) : defaultValue;
	LOGTRACE3("jo_string(key:%s default:%s) -> %s", key, defaultValue, result);

	return jo_parse(result, argMap);
}

Scalar jo_Scalar(const json_t *pObj, const char *key, const Scalar &defaultValue, ArgMap &argMap) {
	Scalar result = defaultValue;
	json_t *pValue = json_object_get(pObj, key);
	if (pValue) {
		if (!json_is_array(pValue)) {
			LOGERROR1("expected JSON array for %s", key);
		} else { 
			switch (json_array_size(pValue)) {
				case 1: 
					result = Scalar(json_number_value(json_array_get(pValue, 0)));
					break;
				case 2: 
					result = Scalar(json_number_value(json_array_get(pValue, 0)),
						json_number_value(json_array_get(pValue, 1)));
					break;
				case 3: 
					result = Scalar(json_number_value(json_array_get(pValue, 0)),
						json_number_value(json_array_get(pValue, 1)),
						json_number_value(json_array_get(pValue, 2)));
					break;
				case 4: 
					result = Scalar(json_number_value(json_array_get(pValue, 0)),
						json_number_value(json_array_get(pValue, 1)),
						json_number_value(json_array_get(pValue, 2)),
						json_number_value(json_array_get(pValue, 3)));
					break;
				default:
					LOGERROR1("expected JSON array with 1, 2, 3 or 4 integer values 0-255 for %s", key);
					return defaultValue;
			}
		}
	}
	if (pValue && logLevel >= FIRELOG_TRACE) {
		char buf[250];
		snprintf(buf, sizeof(buf), "jo_scalar(key:%s default:[%f %f %f %f]) -> [%f %f %f %f]", 
			key,
			defaultValue[0], defaultValue[1], defaultValue[2], defaultValue[3],
			result[0], result[1], result[2], result[3]);
		LOGTRACE1("%s", buf);
	}
	return result;
}

int jo_shape(json_t *pStage, const char *key, const char *&errMsg, ArgMap &argMap) {
	string shape = jo_string(pStage, key, "MORPH_ELLIPSE");
	int result = MORPH_ELLIPSE;

	if (shape.compare("MORPH_ELLIPSE")==0) {
		result = MORPH_ELLIPSE;
	} else if (shape.compare("MORPH_RECT")==0) {
		result = MORPH_RECT;
	} else if (shape.compare("MORPH_CROSS")==0) {
		result = MORPH_CROSS;
	} else {
		errMsg = "shape not supported";
	}
	if (errMsg) {
		LOGTRACE2("jo_shape(key:%s default:MORPH_ELLIPSE) -> %s", key, errMsg);
	} else {
		LOGTRACE2("jo_shape(key:%s default:MORPH_ELLIPSE) -> %d", key, result);
	}
	return result;
}

} // namespace firesight

