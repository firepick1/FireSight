#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "FireLog.h"
#include "Pipeline.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "jansson.h"
#include "MatUtil.hpp"
#include "stages/detector.h"

using namespace cv;
using namespace std;
using namespace firesight;

namespace firesight {
 typedef class TestPipeline : public Pipeline {
   public:
      KeyPoint regionKeypoint(const vector<Point> &region) {
         return MSERStage::_regionKeypoint(region);
      }

      TestPipeline(const char * pJson) : Pipeline(pJson) {
      }

  } TestPipeline;
}


static void assertKeypoint(const KeyPoint &keypoint, double x, double y, double size, double angle, double tolerance) {
  cout << "assertKeyPoint() x:" << keypoint.pt.x << " y:" << keypoint.pt.y ;
  cout << " size:" << keypoint.size << " angle:" << keypoint.angle;
  cout << endl;
  assert(x-tolerance <= keypoint.pt.x && keypoint.pt.x <= x+tolerance);
  assert(y-tolerance <= keypoint.pt.y && keypoint.pt.y <= y+tolerance);
  assert(size-tolerance <= keypoint.size && keypoint.size <= size+tolerance);
  assert(angle-tolerance <= keypoint.angle && keypoint.angle <= angle+tolerance);
}

void test_regionKeypoint() {
  TestPipeline pipeline("[]");
  KeyPoint keypoint;
  vector<Point> pts;
  double tolerance = .01;

  LOGTRACE("----------------------HORIZONTAL");

  pts.clear();
  pts.push_back(Point(0,10));
  pts.push_back(Point(10,10));
  pts.push_back(Point(20,10));
  cout << "horizontal pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 0, tolerance);

  pts.clear();
  pts.push_back(Point(0,11));
  pts.push_back(Point(10,10));
  pts.push_back(Point(20,9));
  cout << "horizontal slightly down pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 354.29, tolerance);

  pts.clear();
  pts.push_back(Point(0,9));
  pts.push_back(Point(10,10));
  pts.push_back(Point(20,11));
  cout << "horizontal slightly up pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 5.71, tolerance);
  
  LOGTRACE("----------------------VERTICAL");

  pts.clear();
  pts.push_back(Point(10,0));
  pts.push_back(Point(10,10));
  pts.push_back(Point(10,20));
  cout << "vertical pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 90, tolerance);

  pts.clear();
  pts.push_back(Point(11,0));
  pts.push_back(Point(10,10));
  pts.push_back(Point(9,20));
  cout << "vertical slightly left pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 95.71, tolerance);

  pts.clear();
  pts.push_back(Point(9,0));
  pts.push_back(Point(10,10));
  pts.push_back(Point(11,20));
  cout << "vertical slightly right pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 84.29, tolerance);

  LOGTRACE("----------------------DIAGONAL UP");

  pts.clear();
  pts.push_back(Point(0,0));
  pts.push_back(Point(10,10));
  pts.push_back(Point(20,20));
  cout << "diagonal up pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 45, tolerance);

  pts.clear();
  pts.push_back(Point(0,1));
  pts.push_back(Point(10,10));
  pts.push_back(Point(20,19));
  cout << "diagonal up- pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 41.99, tolerance);

  pts.clear();
  pts.push_back(Point(1,0));
  pts.push_back(Point(10,10));
  pts.push_back(Point(19,20));
  cout << "diagonal up+ pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 48.01, tolerance);

  LOGTRACE("----------------------DIAGONAL DOWN");

  pts.clear();
  pts.push_back(Point(20,0));
  pts.push_back(Point(10,10));
  pts.push_back(Point(0,20));
  cout << "diagonal down pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 315, tolerance);

  pts.clear();
  pts.push_back(Point(19,0));
  pts.push_back(Point(10,10));
  pts.push_back(Point(1,20));
  cout << "diagonal down+ pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 131.99, tolerance);

  pts.clear();
  pts.push_back(Point(20,1));
  pts.push_back(Point(10,10));
  pts.push_back(Point(0,19));
  cout << "diagonal down- pts:" << pts << endl;
  keypoint = pipeline.regionKeypoint(pts);
  assertKeypoint(keypoint, 10, 10, 1.954, 318.01, tolerance);

}
