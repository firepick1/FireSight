/*
 * @Author  : Simon Fojtu
 * @Date    : 13.07.2014
 */

#ifdef LGPL2_1

#include "FireLog.h"
#include "jansson.h"

#include "opencv2/imgproc/imgproc.hpp"

#include <zbar.h>
#include <iostream>

#include "qrdecode.h"

using namespace std;
using namespace firesight;
using namespace zbar;

vector<QRPayload> ZbarQrDecode::scan(Mat &img, int show) {
    vector<QRPayload> result;

    // create a reader
    ImageScanner scanner;

    // configure the reader
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

    // wrap image data  
    Mat img_gray;
    if (img.channels() == 1) {
        img_gray = img;
    } else {
        cvtColor(img, img_gray, CV_BGR2GRAY);
    }
    int width = img_gray.cols;  
    int height = img_gray.rows;  
    uchar *raw = (uchar *)img_gray.data;  
    Image image(width, height, "Y800", raw, width * height);  

    // scan the image for barcodes
    int n = scanner.scan(image);

    // extract results
    
    for (Image::SymbolIterator symbol = image.symbol_begin();
        symbol != image.symbol_end();
        ++symbol) {
        QRPayload payload;

        if (symbol->get_location_size() == 4) {
            if (show) {
                line(img, Point(symbol->get_location_x(0), symbol->get_location_y(0)), Point(symbol->get_location_x(1), symbol->get_location_y(1)), Scalar(0, 255, 0), 2, 8, 0);
                line(img, Point(symbol->get_location_x(1), symbol->get_location_y(1)), Point(symbol->get_location_x(2), symbol->get_location_y(2)), Scalar(0, 255, 0), 2, 8, 0);
                line(img, Point(symbol->get_location_x(2), symbol->get_location_y(2)), Point(symbol->get_location_x(3), symbol->get_location_y(3)), Scalar(0, 255, 0), 2, 8, 0);
                line(img, Point(symbol->get_location_x(3), symbol->get_location_y(3)), Point(symbol->get_location_x(0), symbol->get_location_y(0)), Scalar(0, 255, 0), 2, 8, 0);
            }

            payload.x = (symbol->get_location_x(0) + symbol->get_location_x(2))/2.0;
            payload.y = (symbol->get_location_y(0) + symbol->get_location_y(2))/2.0;

        } else {
            payload.x = -1;
            payload.y = -1;
        }

        payload.text = symbol->get_data();

        result.push_back(payload);

    }

    // clean up
    image.set_data(NULL, 0);

    return result;
}

#endif
