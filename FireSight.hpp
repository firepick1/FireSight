#ifndef FIRESIGHT_HPP
#define FIRESIGHT_HPP

#include "opencv2/features2d/features2d.hpp"
#include <vector>

using namespace cv;
using namespace std;

#define TIME_NOW -1

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

	typedef class Analyzer {
	  public: 
			Mat workingImage;

			/**
			 * Constructor
			 * @param perceptionDepth number of images to cache for perceptual timeline
			 */
		  Analyzer(int perceptionDepth=1);

			/**
			 * Process the given json array as a pipeline of successive scanning commands for the given perceptual time
			 * @param json array of json command objects (e.g., {"op":"HoleRecognizer","aMin":25.2,"aMax":29.3})
			 * @param time specifies a point on a discrete abstract perceptual timeline
			 */
		  void process(const char* json, int time=TIME_NOW);

			#ifdef LATER
			/**
			 * Transform specified image 
			 * @param json specification for image source and analysis
			 * @param time specifies a point on a discrete abstract perceptual timeline
			 * @returns transformed OpenCV image 
			 */
			Mat processImage(const char* json, int time=TIME_NOW);

			/**
			 * Analyze image as specified by given json string 
			 * @param json specification for image source and analysis
			 * @param time specifies a point on a discrete abstract perceptual timeline
			 * @returns analyzed model json string that caller must free()
			 */
			const char * processModel(const char* json, int time=TIME_NOW);
			#endif

	  private:
		  int perceptionDepth;
			int currentTime;
			vector<Mat> sourceCache;
	} Analyzer;

} // namespace FireSight

#endif
