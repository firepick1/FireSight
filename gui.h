#ifndef GUI_H
#define GUI_H

#include <string>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "console.h"

namespace firesight {

using namespace std;
using namespace cv;



struct PipelineViewer {
    PipelineViewer(const int width) :
        width(width)
    {
        namedWindow("pipeline", WINDOW_NORMAL);
        setMouseCallback("pipeline", CallBackFunc, this);
    }

    ~PipelineViewer() {
        destroyWindow("pipeline");
    }

    void update(vector<unique_ptr<Stage> >& stages, const vector<Mat>& images, int sel_stage, int sel_param) {
        if (images.size() == 0)
            return;

        height = width * images[0].rows/images[0].cols;

        showImages(images, stages, sel_stage, sel_param);
    }

    void static CallBackFunc(int event, int x, int y, int flags, void* userdata)
    {
        PipelineViewer *self = static_cast<PipelineViewer*>(userdata);
        self->doCallback(event, x, y, flags);
    }

    void doCallback(int event, int x, int y, int flags) {
        const int x_ = (x % width) / scale;
        const int y_ = (y % height) / scale;
        if  (event == cv::EVENT_LBUTTONDOWN)
        {
            x0 = x_;
            y0 = y_;
        }
        else if (event == cv::EVENT_LBUTTONUP)
        {
            if (roi.width == 0 || roi.height == 0) {
                roi = Rect(min(x_,x0), min(y_,y0), abs(x_-x0), abs(y_-y0));
            } else {
                // no nested ROIs
                roi = Rect(0,0,0,0);
            }
        }
        else if (event == EVENT_RBUTTONDOWN)
        {
            roi = Rect(0,0,0,0);
        }
    }

    // --------------------------------------------------------------
    // Function to draw several images to one image.
    // --------------------------------------------------------------
    void showImages(const vector<Mat>& imgs, vector<unique_ptr<Stage> >& stages, int sel_stage, int sel_param)
    {
        float nImgs=imgs.size();
        int imgsInRow=ceil(sqrt(nImgs));
        int imgsInCol=ceil(nImgs/imgsInRow);

        int resultImgW=width*imgsInRow;
        int resultImgH=height*imgsInCol;

        Mat resultImg=Mat::zeros(resultImgH,resultImgW,CV_8UC3);
        int ind=0;
        for(int i=0;i<imgsInCol;i++)
        {
            for(int j=0;j<imgsInRow;j++)
            {
                if(ind<imgs.size())
                {
                int cell_row=i*height;
                int cell_col=j*width;

                Mat tmp;
                if (imgs[ind].type() != resultImg.type())
                {
                    cvtColor(imgs[ind], tmp, CV_GRAY2RGB);
                }else{
                    imgs[ind].copyTo(tmp);
                }

                if (roi.width > 0 && roi.height > 0) {
                    Mat tmp_roi;
                    tmp(roi).copyTo(tmp_roi);
                    tmp = tmp_roi.clone();
                }

                scale = 1.0 * width / tmp.cols;

                resize(tmp, tmp, Size(width, height), 1,1, INTER_CUBIC);

                // print info into images
                if (ind == 0) {
                    putText(tmp, "input", Point(10, 10), FONT_HERSHEY_PLAIN, 0.8, cvScalar(255,255,255), (sel_stage == ind ? 2 : 1), CV_AA);
                } else {
                    vector<string> sinfo = stages[ind-1]->info();
                    sinfo[0] = to_string(ind) + ": " + sinfo[0];
                    Console con(tmp);
                    for (size_t i = 0; i < sinfo.size(); i++) {
                        con.print(sinfo[i], (sel_stage == ind && (i == 0 || sel_param == i-1) ? 2 : 1));
                    }
                    con.print("");
                    // display possible error message
                    con.print(stages[ind-1]->getErrorMessage(), 1, Scalar(0, 0, 255));
                }

                tmp.copyTo(resultImg(Range(cell_row,cell_row+tmp.rows),Range(cell_col,cell_col+tmp.cols)));

                }
                ind++;
            }
        }
        imshow("pipeline",resultImg);
    }

    int width; // set in constructor
    int height; // updated according to aspect ratio
    float scale; // how much is every image scaled to fit into width?
    Rect roi;
    int x0, y0;
};

}

#endif // GUI_H
