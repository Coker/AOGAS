///////////////////////////////////////////////////////////
//  DiffFinder.h
//  Implementation of the Class DiffFinder
//  Created on:      25-Tem-2013 11:46:42
//  Original author: Coker
///////////////////////////////////////////////////////////

#if !defined(EA_3D58B721_9AA0_4417_B2AB_CF75FB53D3A7__INCLUDED_)
#define EA_3D58B721_9AA0_4417_B2AB_CF75FB53D3A7__INCLUDED_

#include <opencv\cv.h>
#include "AffRect.h"

#include "excelLib//ExcelFormat.h"

namespace GYTE_DIFF_FINDER {

	typedef struct {
		cv::Mat difImage1;
		cv::Mat difImage2;

		cv::Mat edgeMap1;
		cv::Mat edgeMap2;

		ExcelFormat::BasicExcel histogram;
	} difoutputs;

	class DiffFinder
	{

	public:
		DiffFinder();
		virtual ~DiffFinder();

		DiffFinder(const cv::Mat& depthMapPano, const cv::Mat& rgbMapPano);

		// pure virtual function
		virtual void getDiff(AffRect registerCoordinate, const cv::Mat& rgbImage,
							 const char* const outputFolder =NULL) const =0;
		
		// pure virtual function
		virtual void getDiff(AffRect registerCoordinate, const cv::Mat& depthImage, const cv::Mat& rgbImage,
							 const char* const outputFolder =NULL) const =0;

		void setRgbMapImage(cv::Mat rgbMapImage);
		void setDepthMapImage(cv::Mat depthMapImage);

	protected:
		cv::Mat depthMapPano;
		cv::Mat rgbMapPano;
	};

} // end of namespace GYTE_DIFF_FINDER


#endif // !defined(EA_3D58B721_9AA0_4417_B2AB_CF75FB53D3A7__INCLUDED_)
