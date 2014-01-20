
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
		public: 
			HoleRecognizer(float minDiameter, float maxDiameter);
			void scan(Mat &matRGB, vector<MatchedRegion> &matches, float maxEllipse = 1.05, float maxCovar = 2.0);

		private:
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

	typedef class FireSight {
	  public: 
			/**
			 * Constructor
			 * @param perceptionDepth number of images to cache for analysis
			 */
		  FireSight(int perceptionDepth);

			/**
			 * Transform specified image 
			 * @param json specification for image source and analysis
			 * @param time used for perception synchronization
			 * @returns transformed OpenCV image 
			 */
			Mat processImage(const char* json, int time=TIME_NOW);

			/**
			 * Analyze image as specified by given json string 
			 * @param json specification for image source and analysis
			 * @param time used for perception synchronization
			 * @returns analyzed model json string that caller must free()
			 */
			const char * processModel(const char* json, int time=TIME_NOW);

	  private:
		  int perceptionDepth;
			int currentTime;
			vector<Mat> sourceCache;
	} FireSight;

} // namespace FireSight

#endif
