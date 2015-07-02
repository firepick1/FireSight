#ifndef QRDECODE_H
#define QRDECODE_H

#ifdef LGPL2_1

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace firesight {

using namespace cv;

typedef struct QRPayload {
  double x, y;
  string text;
  json_t * as_json_t() {
      json_t *pObj = json_object();
      json_object_set(pObj, "x", json_real(x));
      json_object_set(pObj, "y", json_real(y));
      json_object_set(pObj, "text", json_string(text.c_str()));
      return pObj;
  }
  string asJson() {
      json_t *pObj = as_json_t();
      char *pObjStr = json_dumps(pObj, JSON_PRESERVE_ORDER|JSON_COMPACT|JSON_INDENT(2));
      string result(pObjStr);
      return result;
  }
} QRPayload;

typedef class ZbarQrDecode {
  public:
      ZbarQrDecode() {}
      vector<QRPayload> scan(Mat &img, int show);
} ZbarQrDecode;

class QrDecode: public Stage {
public:
    QrDecode(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        show = jo_int(pStage, "show", 0, model.argMap);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);
        char *errMsg = NULL;

        if (logLevel >= FIRELOG_TRACE) {
            char *pStageJson = json_dumps(pStage, 0);
            LOGTRACE1("apply_qrdecode(%s)", pStageJson);
            free(pStageJson);
        }

        try {
            ZbarQrDecode qr;
            vector<QRPayload> payload = qr.scan(model.image, show);

            json_t *payload_json = json_array();
            json_object_set(pStageModel, "qrdata", payload_json);
            for (size_t i = 0; i < payload.size(); i++) {
                json_array_append(payload_json, payload[i].as_json_t());
            }


        } catch (runtime_error &e) {
            errMsg = (char *) malloc(sizeof(char) * (strlen(e.what())+1));
            strcpy(errMsg, e.what());
        }

        return stageOK("apply_qrdecode(%s) %s", errMsg, pStage, pStageModel);
    }

    int show;

};

}
#endif

#endif // QRDECODE_H
