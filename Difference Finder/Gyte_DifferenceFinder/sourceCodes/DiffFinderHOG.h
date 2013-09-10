
#if !defined(_DIFFFINDER_HOG_H)
#define _DIFFFINDER_HOG_H

#include "DiffFinder.h"

namespace GYTE_DIFF_FINDER {

	class DiffFinderHOG : public DiffFinder {
	public:
		DiffFinderHOG();
		DiffFinderHOG(const cv::Mat& rgbMapPano, const cv::Mat& depthMapPano);
		
		// pure virtual function
		virtual void getDiff(AffRect registerCoordinate, const cv::Mat& rgbImage,
							 const char* const outputFolder =NULL) const;
		
		// pure virtual function
		virtual void getDiff(AffRect registerCoordinate, const cv::Mat& depthImage, const cv::Mat& rgbImage,
							 const char* const outputFolder =NULL) const;
	private:	
	};

} // end of namespace GYTE_DIFF_FINDER

#endif