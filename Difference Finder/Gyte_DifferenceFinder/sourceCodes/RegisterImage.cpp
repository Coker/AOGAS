///////////////////////////////////////////////////////////
//  RegisterImage.cpp
//  Implementation of the Class RegisterImage
//  Created on:      25-Tem-2013 11:46:58
//  Original author: Coker
///////////////////////////////////////////////////////////

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\core\core.hpp>
#include "opencv2\calib3d\calib3d.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include "RegisterImage.h"

GYTE_DIFF_FINDER::RegisterImage::RegisterImage() :rgbPano(cv::Mat()) {	
	/* deliberately empty */
}

GYTE_DIFF_FINDER::RegisterImage::~RegisterImage() {
	this->rgbPano.release();
}

GYTE_DIFF_FINDER::RegisterImage::RegisterImage(cv::Mat rgbPano) :rgbPano(rgbPano) {
	/* deliberately empty */
}

GYTE_DIFF_FINDER::AffRect GYTE_DIFF_FINDER::RegisterImage::getAffineRect(cv::Mat rgbImage, char outputPath[] ) {
	GYTE_DIFF_FINDER::AffRect affRect;
	
	if (this->rgbPano.empty()) {
		std::cerr << "RgbPano hasn't set yet ! Please set fisrt\n";
		exit(-1);
	}

	if (rgbImage.empty()) {
		std::cerr << "RgbImage hasn't set yet ! Please set fisrt\n";
		exit(-1);
	}

	const int minHessian = 400;

	cv::Mat img1 = rgbImage.clone(),
			img2 = this->rgbPano.clone();

	cv::Mat image1, image2;
	cv::Mat kernel, res;
	kernel = cv::Mat::ones(5,5,CV_32F)/(float)(5*5);

	filter2D(img1,image1,-1,kernel);
	filter2D(img2,image2,-1,kernel);

	cv::namedWindow("img1", CV_WINDOW_NORMAL);
	cv::namedWindow("img2", CV_WINDOW_NORMAL);

	imshow("img1", img1), imshow("img2", img2);	cv::waitKey(0);

	cv::SurfFeatureDetector siftDetector(minHessian);
	cv::SiftDescriptorExtractor siftDescriptor;
	cv::Mat output_1, output_2;

	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	cv::Mat siftDescriptors_1, siftDescriptors_2;
	
	// Detect the keypoints			
	siftDetector.detect(image1, keypoints_1);
	siftDetector.detect(image2, keypoints_2);
	
	// Calculate descriptors (feature vectors)
	siftDescriptor.compute(image1, keypoints_1, siftDescriptors_1);
	siftDescriptor.compute(image2, keypoints_2, siftDescriptors_2);

	drawKeypoints(image1, keypoints_1, output_1, cv::Scalar(255, 0, 0));
	drawKeypoints(image2, keypoints_2, output_2, cv::Scalar(255, 0, 0));
	// cv::imshow("key Points 1", output_1); cv::imshow("key Points 2", output_2); cv::waitKey(0);
	
	cv::imwrite("rgbImageFeatures.jpg", output_1);
	cv::imwrite("rgbPanoFeatures.jpg", output_2);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector<cv::DMatch> matches;
	matcher.match( siftDescriptors_1, siftDescriptors_2, matches );

	double max_dist = 0,
		   min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < siftDescriptors_1.rows; i++ ) {
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	//-- PS.- radiusMatch can also be used here.
	std::vector<cv::DMatch> good_matches;

	for( int i = 0; i < siftDescriptors_1.rows; i++ ) {
		if( matches[i].distance < 1.5*min_dist )
			good_matches.push_back( matches[i]); 
		
	}

	//-- Draw only "good" matches
	cv::Mat img_matches;
	drawMatches( image1, keypoints_1, image2, keypoints_2,
				good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
				std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//-- Localize the object
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	std::cout << "size : " << good_matches.size() <<std:: endl;
	
	// cv::namedWindow("matches", CV_WINDOW_NORMAL);
	// cv::imshow("matches", img_matches); cv::waitKey(0);
	
	imwrite("image_matches.jpg", img_matches);

	if (good_matches.size() < 4)
		return GYTE_DIFF_FINDER::AffRect();

	for(unsigned int i = 0; i < good_matches.size(); i++ ) {
		//-- Get the keypoints from the good matches
		obj.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
		scene.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
	} // end of for statement

	cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC );

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0] = cv::Point(0,0); 
	obj_corners[1] = cv::Point( image1.cols, 0 );
	obj_corners[2] = cv::Point( image1.cols, image1.rows ); 
	obj_corners[3] = cv::Point( 0, image1.rows );
	std::vector<cv::Point2f> scene_corners(4);	

	cvNamedWindow("Result", CV_WINDOW_NORMAL);
	cvNamedWindow("Result_2", CV_WINDOW_NORMAL);
	
	cv::Mat wptImg_1, wptImg_2;
	
	warpPerspective(img1, wptImg_1, H, cv::Size(image2.cols, image2.rows), cv::INTER_CUBIC);
	
	// cv::imshow("Result", wptImg_1); cv::waitKey(0);

	cv::imwrite("warpedImage.jpg", wptImg_1);

	cv::perspectiveTransform( obj_corners, scene_corners, H);
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	cv::line( img_matches, scene_corners[0] + cv::Point2f( image1.cols, 0), scene_corners[1] + cv::Point2f( image1.cols, 0), cv::Scalar(0, 255, 0), 2 );
	cv::line( img_matches, scene_corners[1] + cv::Point2f( image1.cols, 0), scene_corners[2] + cv::Point2f( image1.cols, 0), cv::Scalar( 0, 255, 0), 2 );
	cv::line( img_matches, scene_corners[2] + cv::Point2f( image1.cols, 0), scene_corners[3] + cv::Point2f( image1.cols, 0), cv::Scalar( 0, 255, 0), 2 );
	cv::line( img_matches, scene_corners[3] + cv::Point2f( image1.cols, 0), scene_corners[0] + cv::Point2f( image1.cols, 0), cv::Scalar( 0, 255, 0), 2 );

	//-- Show detected matches
	imshow( "Matches", img_matches );
	
	cv::waitKey(0);
	
	// http://stackoverflow.com/questions/14365411/opencv-crop-image
	// http://stackoverflow.com/questions/14446643/crop-mat-image-in-opencv-2-4-3-ios
	
	if (scene_corners[0].x < 0 || scene_corners[0].y < 0 || scene_corners[0].x >= image2.cols || scene_corners[0].y >= image2.rows)
		return GYTE_DIFF_FINDER::AffRect();

	// cv::Rect rectangle(scene_corners[0].x, scene_corners[0].y, image2.cols - scene_corners[0].x, image2.rows - scene_corners[0].y);
	int widthRect = (int) MIN(image2.cols, scene_corners[2].x) - scene_corners[0].x,
		heigthRect = MIN(image2.rows, scene_corners[2].y)- scene_corners[0].y;

	// cv::Rect rectangle(scene_corners[0].x, scene_corners[0].y, scene_corners[2].x - scene_corners[0].x, scene_corners[2].y - scene_corners[0].y);
	cv::Rect rectangle(scene_corners[0].x, scene_corners[0].y, widthRect, heigthRect);

	printf("width = %d - height = %d\n", rectangle.width, rectangle.height);

	cv::Mat temp;

	temp = wptImg_1;
	cv::Mat regionOfInterest2 = temp(rectangle);
	
	temp = img2;
	cv::Mat regionOfInterest1 = temp(rectangle);
	
	// has different object
	std::string out(outputPath); 
	std::string output1 = out + "\\ImageMatch.jpg";
	std::string output2 = out + "\\PanoMatch.jpg";

	cv::imwrite("matchesResult1.jpg", regionOfInterest1);
	cv::imwrite("matchesResult2.jpg", regionOfInterest2);
	
	cv::imwrite(output1.data(), regionOfInterest1);
	cv::imwrite(output2.data(), regionOfInterest2);
	
	affRect.setLeftTop(scene_corners[0]);
	affRect.setRightTop(scene_corners[1]);
	affRect.setLeftBottom(scene_corners[2]);
	affRect.setRightBottom(scene_corners[3]);

	regionOfInterest1.release();
	regionOfInterest2.release();
	
	image1.release();
	image2.release();

	img1.release();
	img2.release();

	wptImg_1.release();
	wptImg_2.release();

	temp.release();
	
	output1.clear();
	output2.clear();

	return  GYTE_DIFF_FINDER::AffRect(affRect);
}

void GYTE_DIFF_FINDER::RegisterImage::setPano(cv::Mat panoImage) {
	if (panoImage.empty()) {
		std::cerr << "Rgb Pano Image Has NOT set yet !";
		return;
	}

	this->rgbPano = panoImage;
}