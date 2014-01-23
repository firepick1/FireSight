#ifndef FIRESIGHT_HPP
#define FIRESIGHT_HPP

#include "opencv2/features2d/features2d.hpp"
#include <vector>
#include "jansson.h"

using namespace cv;
using namespace std;

namespace FireSight {
	typedef struct MatchedRegion {
		Range xRange;
		Range yRange;
		Point2f	average;
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
			double maxVariation;
			double minDiversity;
			int max_evolution;
			double area_threshold;
			double min_margin;
			int edge_blur_size;
			void scanRegion(vector<Point> &pts, int i, 
				Mat &matRGB, vector<MatchedRegion> &matches, float maxEllipse, float maxCovar);

	} MSER_holes;

	typedef class Pipeline {
	  public: 
			/**
			 * Construct an image processing pipeline described by the given JSON array
			 * that specifies a sequence of named processing stages.
			 * @param pJson null terminated JSON string
			 */
		  Pipeline(const char * pJson);

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
		  json_t *process(Mat &mat);

	  private:
			json_t *pPipeline;
	} Pipeline;

} // namespace FireSight

#endif
