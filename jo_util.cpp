#include <string.h>
#include <math.h>
#include <iostream>
#include "FireLog.h"
#include "jansson.h"
#include "jo_util.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

#define START_DELIM "{{"
#define END_DELIM "}}"
#define DEFAULT_SEP "||"
#define SINGLE_SEP "|"


namespace firesight {

ArgMap emptyMap;

json_t *json_float(float value) {
  char buf[100];
  snprintf(buf, sizeof(buf), "%g", value);
  return json_string(buf);
}

string jo_parse(const char * pSource, ArgMap &argMap) {
  string result(pSource);
  size_t startDelim = 0;
  int substitutions = 0; 
  while ((startDelim=result.find(START_DELIM,startDelim)) != string::npos) {
    size_t defaultSep = result.find(DEFAULT_SEP,startDelim);
    if (defaultSep == string::npos) {
      size_t singleSep = result.find(SINGLE_SEP,startDelim);
      if (singleSep != string::npos) {
        LOGWARN("jo_parse() expected " DEFAULT_SEP " before default parameter value");
        result.replace(singleSep, sizeof(SINGLE_SEP)-1, DEFAULT_SEP);
        defaultSep = singleSep;
      }
    }
    size_t endDelim = result.find(END_DELIM,startDelim);
    if (endDelim == string::npos) {
      LOGERROR1("jo_parse(): Invalid variable specification: '%s'", pSource);
      break;
    } 
    size_t varEnd = endDelim + sizeof(END_DELIM)-1;
    substitutions++;
    size_t nameStart = startDelim + sizeof(START_DELIM)-1;
    size_t nameEnd = defaultSep == string::npos ? endDelim : defaultSep;
    string name = result.substr(nameStart,nameEnd-nameStart);
    const char * pRep = argMap[name.c_str()];
    if (pRep) { 
      result.replace(startDelim, varEnd-startDelim, pRep);
    } else { // scan for template default
      if (defaultSep == string::npos) {
        LOGERROR1("jo_parse(): variable has no provided or default value: %s", name.c_str());
        break;
      }
      size_t defaultStart = defaultSep + sizeof(DEFAULT_SEP)-1;
      string rep = result.substr(defaultStart, endDelim-defaultStart);
      result.replace(startDelim, varEnd-startDelim, rep);
    }
    startDelim = varEnd;
  }

  if (substitutions) {
    LOGTRACE2("jo_parse(%s) => %s",  pSource, result.c_str());
  }
  return result;
}

json_t *jo_object(const json_t *pObj, const char *key, ArgMap &argMap) {
  json_t *pVal = json_object_get(pObj, key);
  if (pVal) {
    if (json_is_string(pVal)) {
      string result = jo_parse(json_string_value(pVal), argMap);
      json_error_t error;
      pVal = json_loads(result.c_str(), JSON_DECODE_ANY, &error);
      if (!pVal) {
        LOGERROR1("Could not parse JSON: %s", result.c_str());
        throw error;
      }
    }
  }

  return pVal;
}

bool jo_bool(const json_t *pObj, const char *key, bool defaultValue, ArgMap &argMap) {
  json_t *pValue = json_object_get(pObj, key);
  bool result = defaultValue;;
  if (json_is_boolean(pValue)) {
    result = json_is_true(pValue);
  } else if (json_is_string(pValue)) {
    string valStr = jo_parse(json_string_value(pValue), argMap);
    result = valStr.compare("true") == 0;
  }
  LOGTRACE3("jo_bool(key:%s default:%d) -> %d", key, defaultValue, result);
  return result;
}

int jo_int(const json_t *pObj, const char *key, int defaultValue, ArgMap &argMap) {
  json_t *pValue = json_object_get(pObj, key);
  int result = defaultValue;
  if (pValue == NULL) {
    // default
  } else if (json_is_integer(pValue)) {
    result = (int)json_integer_value(pValue);
  } else if (json_is_number(pValue)) {
    result = (int)(0.5+json_number_value(pValue));
  } else if (json_is_string(pValue)) {
    string valStr = jo_parse(json_string_value(pValue), argMap);
    result = atoi(valStr.c_str());
  } else {
    LOGERROR1("jo_int() expected integer value for %s", key);
  }
  LOGTRACE3("jo_int(key:%s default:%d) -> %d", key, defaultValue, result);
  return result;
}

double jo_double(const json_t *pObj, const char *key, double defaultValue, ArgMap &argMap) {
  json_t *pValue = json_object_get(pObj, key);
  double result = defaultValue;
  if (pValue == NULL) {
    // default
  } else if (json_is_number(pValue)) {
    result = json_number_value(pValue);
  } else if (json_is_string(pValue)) {
    string valStr = jo_parse(json_string_value(pValue), argMap);
    result = atof(valStr.c_str());
  } else {
    LOGERROR1("jo_double() expected numeric value for %s", key);
  }

  LOGTRACE3("jo_double(key:%s default:%f) -> %f", key, defaultValue, result);
  return result;
}

string jo_string(const json_t *pObj, const char *key, const char *defaultValue, ArgMap &argMap) {
  json_t *pValue = json_object_get(pObj, key);
  const char * result = json_is_string(pValue) ? json_string_value(pValue) : defaultValue;
  LOGTRACE3("jo_string(key:%s default:%s) -> %s", key, defaultValue, result);

  return jo_parse(result, argMap);
}

Point jo_Point(const json_t *pObj, const char *key, const Point &defaultValue, ArgMap &argMap) {
  Point result = defaultValue;
  json_t *pValue = jo_object(pObj, key, argMap);
  if (pValue) {
    if (!json_is_array(pValue)) {
      LOGERROR1("expected JSON array for %s", key);
    } else { 
      switch (json_array_size(pValue)) {
        case 2: 
          result = Point(
            (size_t)json_integer_value(json_array_get(pValue, 0)),
            (size_t)json_integer_value(json_array_get(pValue, 1)));
          break;
        default:
          LOGERROR1("expected JSON array with 2 integer values for %s", key);
          return defaultValue;
      }
    }
  }
  if (pValue && logLevel >= FIRELOG_TRACE) {
    char buf[250];
    snprintf(buf, sizeof(buf), "jo_Point(key:%s default:[%d %d]) -> [%d %d]", 
      key, defaultValue.x, defaultValue.y, result.x, result.y);
    LOGTRACE1("%s", buf);
  }
  return result;
}

Scalar jo_Scalar(const json_t *pObj, const char *key, const Scalar &defaultValue, ArgMap &argMap) {
  Scalar result = defaultValue;
  json_t *pValue = jo_object(pObj, key, argMap);
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
  string shape = jo_string(pStage, key, "MORPH_ELLIPSE", argMap);
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

