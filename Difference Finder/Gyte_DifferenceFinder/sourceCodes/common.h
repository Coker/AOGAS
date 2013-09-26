
#if !defined (__COMMON_H)
#define _COMMON_H

#define MAX_PATH_LENGTH 250

#define BLUE cv::Scalar(255, 0, 0)
#define GREEN cv::Scalar(0, 255, 0) 
#define RED cv::Scalar(0, 0, 255)

enum DifferenceFinderAlgorithm {
	NONE =0,
	HOG,
	SIFT,
	SURF,
	MSER,
	INSENTY
};

#endif