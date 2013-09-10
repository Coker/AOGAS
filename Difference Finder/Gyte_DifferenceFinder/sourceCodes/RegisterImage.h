///////////////////////////////////////////////////////////
//  RegisterImage.h
//  Implementation of the Class RegisterImage
//  Created on:      25-Tem-2013 11:46:57
//  Original author: Coker
///////////////////////////////////////////////////////////

#if !defined(EA_332DCCD2_6FC9_4757_BF58_C04B54CFF93D__INCLUDED_)
#define EA_332DCCD2_6FC9_4757_BF58_C04B54CFF93D__INCLUDED_

#include <opencv/cv.h>

#include "AffRect.h"

namespace GYTE_DIFF_FINDER {
	class RegisterImage
	{
	public:
		RegisterImage();
		RegisterImage(cv::Mat rgbPano);
		virtual ~RegisterImage();

		AffRect getAffineRect(cv::Mat rgbImage, char outputPath[] );
		void setPano(cv::Mat panoImage);

	private:
		cv::Mat rgbPano;

	};
} // end of GYTE_DIFF_FINDER namespace


#endif // !defined(EA_332DCCD2_6FC9_4757_BF58_C04B54CFF93D__INCLUDED_)
