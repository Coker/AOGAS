///////////////////////////////////////////////////////////
//  AffRect.cpp
//  Implementation of the Class AffRect
//  Created on:      25-Tem-2013 11:43:55
//  Original author: Coker
///////////////////////////////////////////////////////////

#include "AffRect.h"

// unnamed namespace
namespace {
	bool isSamePoint(cv::Point pt1, cv::Point pt2) {
		return (pt1.x == pt2.x) && (pt1.y == pt2.y);
	}
} // end of unnamed namespace

GYTE_DIFF_FINDER::AffRect::AffRect()
	:leftTop(cv::Point(0, 0)), leftBottom(cv::Point(0, 0)), rightTop(cv::Point(0, 0)), rightBottom(cv::Point(0, 0)) {
	/* deliberately empty */
}

GYTE_DIFF_FINDER::AffRect::~AffRect(){
	/* deliberately empty */
}

GYTE_DIFF_FINDER::AffRect::AffRect(cv::Point rightBottom, cv::Point leftBottom, cv::Point rightTop, cv::Point leftTop)
	:leftTop(leftTop), rightTop(rightTop), leftBottom(leftBottom), rightBottom(rightBottom) {
	/* deliberately empty */
}

cv::Point GYTE_DIFF_FINDER::AffRect::getLeftBottom(void) {
	return cv::Point(this->leftBottom);
}

cv::Point GYTE_DIFF_FINDER::AffRect::getLeftTop(void) {
	return cv::Point(this->leftTop);
}

cv::Point GYTE_DIFF_FINDER::AffRect::getRightBottom() {
	return cv::Point(this->rightBottom);
}

cv::Point GYTE_DIFF_FINDER::AffRect::getRightTop(void) {
	return  cv::Point(this->rightTop);
}

void GYTE_DIFF_FINDER::AffRect::setLeftBottom(cv::Point leftBottom) {
	this->leftBottom = leftBottom;
}

void GYTE_DIFF_FINDER::AffRect::setLeftTop(cv::Point leftTop) {
	this->leftTop = leftTop;
}

void GYTE_DIFF_FINDER::AffRect::setRightBottom(cv::Point rightBottom) {
	this->rightBottom = rightBottom;
}

void GYTE_DIFF_FINDER::AffRect::setRightTop(cv::Point rightTop) {
	this->rightTop = rightTop;
}

bool GYTE_DIFF_FINDER::AffRect::operator==(const GYTE_DIFF_FINDER::AffRect& rtSide) {
	return isSamePoint(this->leftBottom, rtSide.leftBottom) &&
		   isSamePoint(this->leftTop, rtSide.leftTop) &&
		   isSamePoint(this->rightBottom, rtSide.rightBottom) &&
		   isSamePoint(this->rightTop, rtSide.rightTop);
}