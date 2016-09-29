#ifndef FIRESIGHT_HPP
#define FIRESIGHT_HPP

#include "opencv2/features2d/features2d.hpp"
#include <vector>
#include <map>
#ifdef _MSC_VER
#include "winjunk.hpp"
#else
#define CLASS_DECLSPEC
#endif
#include "jansson.h"

using namespace cv;
using namespace std;

#if __amd64__ || __x86_64__ || _WIN64 || _M_X64
#define FIRESIGHT_64_BIT
#define FIRESIGHT_PLATFORM_BITS 64
#else
#define FIRESIGHT_32_BIT
#define FIRESIGHT_PLATFORM_BITS 32
#endif

namespace firesight {

  CLASS_DECLSPEC typedef map<string,const char *> ArgMap;
  CLASS_DECLSPEC extern ArgMap emptyMap;

  typedef struct MatchedRegion {
    Range xRange;
    Range yRange;
    Point2f  average;
    int pointCount;
    float ellipse; // area of ellipse that covers {xRange, yRange}
    float covar;

    MatchedRegion(Range xRange, Range yRange, Point2f average, int pointCount, float covar);
    string asJson();
    json_t *as_json_t();
  } MatchedRegion;

  typedef class HoleRecognizer {
#define HOLE_SHOW_NONE 0 /* do not show matches */
#define HOLE_SHOW_MSER 1 /* show all MSER matches */
#define HOLE_SHOW_MATCHES 2 /* only show MSER matches that meet hole criteria */ 
    public: 
      HoleRecognizer(float minDiameter, float maxDiameter);

      /**
       * Update the working image to show MSER matches.
       * Image must have at least three channels representing RGB values.
       * @param show matched regions. Default is HOLE_SHOW_NONE
       */
      void showMatches(int show);

      void scan(Mat &matRGB, vector<MatchedRegion> &matches, float maxEllipse = 1.05, float maxCovar = 2.0);

    private:
      int _showMatches;
      MSER mser;
      float minDiam;
      float maxDiam;
      int delta;
      int minArea;
      int maxArea;
      float maxVariation;
      float minDiversity;
      int max_evolution;
      float area_threshold;
      float min_margin;
      int edge_blur_size;
      void scanRegion(vector<Point> &pts, int i, 
        Mat &matRGB, vector<MatchedRegion> &matches, float maxEllipse, float maxCovar);

  } MSER_holes;

    typedef struct Circle {
        float x;
        float y;
        float radius;

        Circle(float x, float y, float radius);
        string asJson();
        json_t *as_json_t();
    } Circle;

  typedef class HoughCircle {
#define CIRCLE_SHOW_NONE 0 /* do not show circles */
#define CIRCLE_SHOW_ALL 1  /* show all circles */
    public:
      HoughCircle(int minDiameter, int maxDiameter);

      /**
       * Update the working image to show detected circles.
       * Image must have at least three channels representing RGB values.
       * @param show matched regions. Default is CIRCLE_SHOW_NONE
       */
      void setShowCircles(int show);

      void scan(Mat &matRGB, vector<Circle> &circles);
      
      void setFilterParams( int d, double sigmaColor, double sigmaSpace);
      void setHoughParams(double dp, double minDist, double param1, double param2);

    private:
      int _showCircles;
      int minDiam;
      int maxDiam;
      vector<Circle> circles;

      void show(Mat & image, vector<Circle> circles);

      // bilateral filter parameters
      int bf_d;
      double bf_sigmaColor;
      double bf_sigmaSpace;

      // HoughCircle parameters
      double hc_dp;
      double hc_minDist;
      double hc_param1;
      double hc_param2;

  } HoughCircle;

  typedef struct XY {
      double x, y;
      XY(): x(0), y(0) {}
      XY(double x_, double y_): x(x_), y(y_) {}
  } XY;

  typedef class Pt2Res {
      public:

          Pt2Res() {}
          double getResolution(double thr1, double thr2, double confidence, double separation, vector<XY> coords);
      private:
          static bool compare_XY_by_x(XY a, XY b);
          static bool compare_XY_by_y(XY a, XY b);
          int nsamples_RANSAC(size_t ninl, size_t xlen, unsigned int NSAMPL, double confidence);
          static double _RANSAC_line(XY * x, size_t nx, XY C);
          static double _RANSAC_pattern(XY * x, size_t nx, XY C);
          vector<XY> RANSAC_2D(unsigned int NSAMPL, vector<XY> coords, double thr, double confidence, double(*err_fun)(XY *, size_t, XY));
          void least_squares(vector<XY> xy, double * a, double * b);

  } Pt2Res;

#ifdef LGPL2_1
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
#endif // LGPL2_1

  typedef class StageData {
    public:
      StageData(string stageName);
      ~StageData();
  } StageData, *StageDataPtr;

  typedef class Model {
    private:
      json_t *pJson;

    public: // methods
      Model(ArgMap &argMap=emptyMap);
      ~Model();

      /**
       * Return JSON root node of image recognition model.
       * Caller must json_decref() returned value.
       */
      inline json_t *getJson(bool incRef = true) { 
        if (incRef) {
          return json_incref(pJson);
        } else {
          return pJson;
        }
      };

    public: // fields
      Mat image;
      map<string, Mat> imageMap;
      map<string, StageDataPtr> stageDataMap;
      ArgMap argMap;
  } Model;

  typedef class CLASS_DECLSPEC Pipeline {
    protected:
      bool processModel(Model &model);
      bool stageOK(const char *fmt, const char *errMsg, json_t *pStage, json_t *pStageModel);
      KeyPoint _regionKeypoint(const vector<Point> &region);
      void _eigenXY(const vector<Point> &pts, Mat &eigenvectorsOut, Mat &meanOut, Mat &covOut);
      void _covarianceXY(const vector<Point> &pts, Mat &covOut, Mat &meanOut);
      bool morph(json_t *pStage, json_t *pStageModel, Model &model, String mop, const char * fmt) ;

      bool apply_absdiff(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_backgroundSubtractor(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_blur(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_matchTemplate(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_calcHist(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_calcOffset(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_Canny(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_circle(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_convertTo(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_cout(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_crop(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_cvtColor(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_dft(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_dftSpectrum(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_dilate(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_drawKeypoints(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_drawRects(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_equalizeHist(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_erode(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_FireSight(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_HoleRecognizer(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_HoughCircles(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_points2resolution_RANSAC(json_t *pStage, json_t *pStageModel, Model &model);
#ifdef LGPL2_1
      bool apply_qrdecode(json_t *pStage, json_t *pStageModel, Model &model);
#endif // LGPL2_1
      bool apply_imread(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_imwrite(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_log(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_Mat(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_matchGrid(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_meanStdDev(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_minAreaRect(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_model(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_morph(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_MSER(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_normalize(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_proto(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_PSNR(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_putText(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_rectangle(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_resize(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_threshold(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_warpRing(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_SimpleBlobDetector(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_sharpness(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_split(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_stageImage(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_transparent(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_undistort(const char *pName, json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_warpAffine(json_t *pStage, json_t *pStageModel, Model &model);
      bool apply_warpPerspective(const char *pName, json_t *pStage, json_t *pStageModel, Model &model);

      const char * dispatch(const char *pName, const char *pOp, json_t *pStage, json_t *pStageModel, Model &model);
      void detectKeypoints(json_t *pStageModel, vector<vector<Point> > &regions);
      void detectRects(json_t *pStageModel, vector<vector<Point> > &regions);
      int parseCvType(const char *typeName, const char *&errMsg);
      void validateImage(Mat &image);
      json_t *pPipeline;

    public: 
      enum DefinitionType { PATH, JSON };

      /**
       * Construct an image processing pipeline described by the given JSON array
       * that specifies a sequence of named processing stages.
       * @param pDefinition null terminated JSON string or file path
       * @param indicates whether definition is JSON string or file path
       */
      Pipeline(const char * pDefinition=NULL, DefinitionType defType=JSON );

      /**
       * Construct an image processing pipeline described by the given JSON array
       * that specifies a sequence of named processing stages.
       * @param pJson jansson array node 
       */
      Pipeline(json_t *pJson);

      ~Pipeline();

      /**
       * Process the given working image and return a JSON object that represents
       * the recognized model comprised of the individual stage models. 
       * @param mat initial and transformed working image
       * @return pointer to jansson root node of JSON object that has a field for each recognized stage model. E.g., {s1:{...}, s2:{...}, ... , sN:{...}}
       */
      json_t *process(Mat &mat, ArgMap &argMap);

  } Pipeline;

} // namespace firesight

#endif
