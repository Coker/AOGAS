///////////////////////////////////////////////////////////
//  DiffFinder.cpp
//  Implementation of the Class DiffFinder
//  Created on:      25-Tem-2013 11:46:42
//  Original author: Coker
///////////////////////////////////////////////////////////

#include <opencv\cv.h>

#include "DiffFinder.h"
#include "AffRect.h"

GYTE_DIFF_FINDER::DiffFinder::DiffFinder() {
	this->rgbMapPano = cv::Mat();
	this->depthMapPano = cv::Mat();
}

GYTE_DIFF_FINDER::DiffFinder::DiffFinder(const cv::Mat& depthMapPano, const cv::Mat& rgbMapPano) {
	this->depthMapPano = depthMapPano;
	this->rgbMapPano = rgbMapPano;
}

GYTE_DIFF_FINDER::DiffFinder::~DiffFinder() {
	if ( !depthMapPano.empty() ) depthMapPano.release();
	if ( !rgbMapPano.empty() ) rgbMapPano.release();
}

void GYTE_DIFF_FINDER::DiffFinder::setRgbMapImage(cv::Mat rgbMapImage) {
	this->rgbMapPano = rgbMapImage;
}

void GYTE_DIFF_FINDER::DiffFinder::setDepthMapImage(cv::Mat depthMapImage) {
	this->depthMapPano = depthMapImage;
}

