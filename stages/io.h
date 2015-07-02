#ifndef IO_H
#define IO_H

/*
 * @Author  : Simon Fojtu
 * @Date    : 17.06.2015
 */

#include "Pipeline.h"
#include "jo_util.hpp"

#include <map>
#include <string>
#include <stdexcept>

namespace firesight {

using namespace cv;

class ImWrite : public Stage
{
public:
    ImWrite(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        path = jo_string(pStage, "path");
        _params["path"] = new StringParameter(this, path);
    }

private:
    bool apply_internal(json_t *pStageModel, Model &model) {
        validateImage(model.image);

        const char *errMsg = NULL;

        if (path.empty()) {
            errMsg = "Expected path for imwrite";
        } else {
            bool result = imwrite(path.c_str(), model.image);
            json_object_set(pStageModel, "result", json_boolean(result));
        }

        return stageOK("apply_imwrite(%s) %s", errMsg, pStage, pStageModel);
    }

protected:
    string path;
};

class ImRead : public Stage
{
public:
    ImRead(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        path = jo_string(pStage, "path", "", model.argMap);
        _params["path"] = new StringParameter(this, path);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        const char *errMsg = NULL;

        if (path.empty()) {
            errMsg = "expected path for imread";
        } else {
            model.image = imread(path.c_str(), CV_LOAD_IMAGE_COLOR);
            if (model.image.data) {
                json_object_set(pStageModel, "rows", json_integer(model.image.rows));
                json_object_set(pStageModel, "cols", json_integer(model.image.cols));
            } else {
                LOGERROR1("imread(%s) failed", path.c_str());
                errMsg = "apply_imread() failed";
            }
        }

        return stageOK("apply_imread(%s) %s", errMsg, pStage, pStageModel);
    }

    string path;
};

class Cout : public Stage {
public:
    Cout(json_t *pStage, Model &model, string pName) : Stage(pStage, pName) {
        col = jo_int(pStage, "col", 0, model.argMap);
        _params["col"] = new IntParameter(this, col);
        row = jo_int(pStage, "row", 0, model.argMap);
        _params["row"] = new IntParameter(this, row);
        cols = jo_int(pStage, "cols", model.image.cols, model.argMap);
        _params["cols"] = new IntParameter(this, cols);
        rows = jo_int(pStage, "rows", model.image.rows, model.argMap);
        _params["rows"] = new IntParameter(this, rows);
        precision = jo_int(pStage, "precision", 1, model.argMap);
        _params["precision"] = new IntParameter(this, precision);
        width = jo_int(pStage, "width", 5, model.argMap);
        _params["width"] = new IntParameter(this, width);
        channel = jo_int(pStage, "channel", 0, model.argMap);
        _params["channel"] = new IntParameter(this, channel);
        comment = jo_string(pStage, "comment", "", model.argMap);
        _params["comment"] = new StringParameter(this, comment);
    }

protected:
    bool apply_internal(json_t *pStageModel, Model &model) {
        const char *errMsg = NULL;

        if (row<0 || col<0 || rows<=0 || cols<=0) {
            errMsg = "Expected 0<=row and 0<=col and 0<cols and 0<rows";
        }
        if (rows > model.image.rows) {
            rows = model.image.rows;
        }
        if (cols > model.image.cols) {
            cols = model.image.cols;
        }

        if (!errMsg) {
            int depth = model.image.depth();
            cout << matInfo(model.image);
            cout << " show:[" << row << "-" << row+rows-1 << "," << col << "-" << col+cols-1 << "]";
            if (comment.size()) {
                cout << " " << comment;
            }
            cout << endl;
            for (int r = row; r < row+rows; r++) {
                for (int c = col; c < col+cols; c++) {
                    cout.precision(precision);
                    cout.width(width);
                    if (model.image.channels() == 1) {
                        switch (depth) {
                        case CV_8S:
                        case CV_8U:
                            cout << (short) model.image.at<unsigned char>(r,c,channel) << " ";
                            break;
                        case CV_16U:
                            cout << model.image.at<unsigned short>(r,c) << " ";
                            break;
                        case CV_16S:
                            cout << model.image.at<short>(r,c) << " ";
                            break;
                        case CV_32S:
                            cout << model.image.at<int>(r,c) << " ";
                            break;
                        case CV_32F:
                            cout << std::fixed;
                            cout << model.image.at<float>(r,c) << " ";
                            break;
                        case CV_64F:
                            cout << std::fixed;
                            cout << model.image.at<double>(r,c) << " ";
                            break;
                        default:
                            cout << "UNSUPPORTED-CONVERSION" << " ";
                            break;
                        }
                    } else {
                        switch (depth) {
                        case CV_8S:
                        case CV_8U:
                            cout << (short) model.image.at<Vec2b>(r,c)[channel] << " ";
                            break;
                        case CV_16U:
                            cout << model.image.at<Vec2w>(r,c)[channel] << " ";
                            break;
                        case CV_16S:
                            cout << model.image.at<Vec2s>(r,c)[channel] << " ";
                            break;
                        case CV_32S:
                            cout << model.image.at<Vec2i>(r,c)[channel] << " ";
                            break;
                        case CV_32F:
                            cout << std::fixed;
                            cout << model.image.at<Vec2f>(r,c)[channel] << " ";
                            break;
                        case CV_64F:
                            cout << std::fixed;
                            cout << model.image.at<Vec2d>(r,c)[channel] << " ";
                            break;
                        default:
                            cout << "UNSUPPORTED-CONVERSION" << " ";
                            break;
                        }
                    }
                }
                cout << endl;
            }
        }

        return stageOK("apply_cout(%s) %s", errMsg, pStage, pStageModel);
    }

    int col;
    int row;
    int cols;
    int rows;
    int precision;
    int width;
    int channel;
    string comment;

};

}




#endif // IO_H
