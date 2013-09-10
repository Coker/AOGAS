///////////////////////////////////////////////////////////
//  AffRect.h
//  Implementation of the Class AffRect
//  Created on:      25-Tem-2013 11:43:55
//  Original author: Coker
///////////////////////////////////////////////////////////

#if !defined(EA_9BB3C1B3_5688_40ce_B405_C26EC080EF53__INCLUDED_)
#define EA_9BB3C1B3_5688_40ce_B405_C26EC080EF53__INCLUDED_

#include <opencv\cv.h>

namespace GYTE_DIFF_FINDER {
	class AffRect
	{

	public:
		AffRect();
		AffRect(cv::Point rightBottom, cv::Point leftBottom, cv::Point rightTop, cv::Point leftTop);
		~AffRect();
	
		inline cv::Point getLeftBottom(void);
		inline cv::Point getLeftTop(void);
		inline cv::Point getRightBottom(void);
		inline cv::Point getRightTop(void);
		void setLeftBottom(cv::Point leftBottom);
		void setLeftTop(cv::Point leftTop);
		void setRightBottom(cv::Point rightBottom);
		void setRightTop(cv::Point rightTop);

		bool operator ==(const AffRect& rtSide);

	private:
		cv::Point leftBottom;
		cv::Point leftTop;
		cv::Point rightBottom;
		cv::Point rightTop;

	};
} // end of namespace GYTE_DIFF_FINDER


#endif // !defined(EA_9BB3C1B3_5688_40ce_B405_C26EC080EF53__INCLUDED_)
