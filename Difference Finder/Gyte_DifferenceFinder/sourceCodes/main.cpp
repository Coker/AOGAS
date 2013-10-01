
/**
 * @author Mustafa ÇOKER
 * @date 25-07-2013
 * @filename main.cpp
 *
 */

#include <opencv2\highgui\highgui.hpp>

#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\core\core.hpp>
#include "opencv2\calib3d\calib3d.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include "AffRect.h"
#include "RegisterImage.h"
#include "DiffFinderHOG.h"
#include "common.h"

namespace {
	void configurePath(char** path, int size);
	void testSmooth(void);
	cv::Mat erade(const cv::Mat& src, const int eradeSize, const int eradeType);
	cv::Mat dilate(const cv::Mat& src, const int dilationSize, const int dilationType);
	cv::Mat scaleImage(const cv::Mat& image, const double scaleRatio);
}

int main(void)
{
	// cv::Mat im1 = cv::imread("C:\\Users\\Coker\\Desktop\\tests\\test5\\inputs\\im1.JPG");
	// cv::Mat im2 = cv::imread("C:\\Users\\Coker\\Desktop\\tests\\test5\\inputs\\im2.JPG");	

	/*
	cv::Mat im2 = cv::imread("ImageMatch.JPG");
	cv::Mat im1 = cv::imread("PanoMatch.JPG");

	char *temp = "C:\\Users\\Coker\\Desktop\\tests\\test5\\registered";
	GYTE_DIFF_FINDER::RegisterImage registeration(im1);
	registeration.getAffineRect(im2, temp);
	*/
	
	// testSmooth();
	// return 0;

	char *inputPath = new char[MAX_PATH_LENGTH];
	char *outputPath = new char[MAX_PATH_LENGTH];
	
	printf("Please Enter the input images path:\n");
	gets(inputPath);

	printf("Please Enter the output path:\n");
	gets(outputPath);

	configurePath(&inputPath, strlen(inputPath));
	configurePath(&outputPath, strlen(outputPath));

	std::string ImageMatchPath(inputPath), panoMatchPath(inputPath);

	ImageMatchPath += "\\ImageMatch.jpg";
	panoMatchPath += "\\PanoMatch.jpg";
	
	cv::Mat pano =cv::imread(ImageMatchPath.data());
	cv::Mat rgb =cv::imread(panoMatchPath.data());

	pano =scaleImage(pano, 2.0);
	rgb =scaleImage(rgb, 2.0);

	GYTE_DIFF_FINDER::DiffFinderHOG finder;
	finder.setRgbMapImage(pano);
	finder.getDiff(GYTE_DIFF_FINDER::AffRect(), rgb, outputPath);

	scanf("%*d");

	return 0;
}

namespace {
	
	cv::Mat scaleImage(const cv::Mat& image, const double scaleRatio) {
		
		if (image.empty() && scaleRatio < 0.001) {
			std::cerr << "Your parameter is not appropriate \n";
			scanf("%*c");
			std::exit(-1);
		}

		cv::Mat res(cv::Size(image.cols/scaleRatio, image.rows/scaleRatio), image.type());
		cv::resize(image, res, cv::Size(image.cols/scaleRatio, image.rows/scaleRatio));
		
		return res.clone();
	}

	cv::Mat erade(const cv::Mat& src, const int eradeSize, const int eradeType) {
		cv::Mat eroded;
		cv::Mat config;

		config = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*eradeSize, 2*eradeSize),
									   cv::Point(eradeSize, eradeSize));
		// cv::erode(src, eroded, config);
		cv::erode(src, eroded, cv::Mat(), cv::Point(-1, -1), eradeSize);

		return eroded;
	}

	cv::Mat dilate(const cv::Mat& src, const int dilationSize, const int dilationType) {
		cv::Mat dilated;
		cv::Mat config;

		config = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*dilationSize+1, 2*dilationSize+1), 
									       cv::Point(dilationSize, dilationSize));
		dilate(src, dilated, config);
		return dilated;
	}

	void testSmooth(void) {
		cv::Mat im1, im2, im1Smooth, im2Smooth;
		im1 =cv::imread("ImageMatch.JPG");
		im2 =cv::imread("PanoMatch.JPG");

		cv::Mat kernel;
		kernel = cv::Mat::ones(5,5,CV_32F)/(float)(5*5);

		cv::GaussianBlur(im1, im1Smooth, cv::Size(11, 11), 0,0);
		cv::GaussianBlur(im2, im2Smooth, cv::Size(11, 11), 0,0);

		cv::Mat res1 = im1Smooth - im1,
				res2 = im2 - im2Smooth;

		res1 = erade(res1, 1, 1);
		res2 = erade(res2, 1, 1);

		res1 = dilate(res1, 3, 3);
		res2 = dilate(res2, 3, 3);

		cv::GaussianBlur(res1, im1Smooth, cv::Size(11, 11), 0,0);
		cv::GaussianBlur(res2, im2Smooth, cv::Size(11, 11), 0,0);

		res1 = im1Smooth - res1;
		res2 = im2Smooth - res2;
	
		// res1 = dilate(res1, 3, 3);
		// res2 = dilate(res2, 3, 3);

		cv::namedWindow("ImageMatch", cv::WINDOW_NORMAL);
		cv::namedWindow("PanoMatch", cv::WINDOW_NORMAL);
		cv::namedWindow("Dif1", cv::WINDOW_NORMAL);
		cv::namedWindow("Dif2", cv::WINDOW_NORMAL);

		cv::imwrite("dif1.bmp", res1);
		cv::imwrite("dif2.bmp", res2);

		cv::imshow("ImageMatch", im1);
		cv::imshow("PanoMatch", im2);
		cv::imshow("Dif1", res1);
		cv::imshow("Dif2", res2);
		cv::waitKey();

	}

	void configurePath(char** path, int size) {
		char *res =new char[ sizeof(char)*MAX_PATH_LENGTH ];
		char *temp;
		char * const startOfTheResVariable = res;

		strcpy(res, (*path));

		temp = (*path);

		do {
			res = strchr(res, '\\' );

			if (NULL == res) break;
			memmove(res+2, res+1, strlen(res));
			
			res[1] = '\\';
			res += 2;

		} while(true);

		(*path) = startOfTheResVariable;
		
		return;
	}

}

#if 0
	#include <iostream>
	#include <vector>

	using namespace cv;
	using namespace std;

	Mat mat2gray(const cv::Mat& src)
	{
		Mat dst;
		normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

		return dst;
	}

	Mat orientationMap(const cv::Mat& mag, const cv::Mat& ori, double thresh = 1.0)
	{
		Mat oriMap = Mat::zeros(ori.size(), CV_8UC3);
		Vec3b red(0, 0, 255);
		Vec3b cyan(255, 255, 0);
		Vec3b green(0, 255, 0);
		Vec3b yellow(0, 255, 255);

		int arr[] = {0,0,0,0};
    
		for(int i = 0; i < mag.rows*mag.cols; i++) {
			float* magPixel = reinterpret_cast<float*>(mag.data + i*sizeof(float));
			if(*magPixel > thresh) {
				float* oriPixel = reinterpret_cast<float*>(ori.data + i*sizeof(float));
				Vec3b* mapPixel = reinterpret_cast<Vec3b*>(oriMap.data + i*3*sizeof(char));
				if(*oriPixel < 90.0) {
					*mapPixel = red;
					++arr[0];
				}
				else if(*oriPixel >= 90.0 && *oriPixel < 180.0) {
					*mapPixel = cyan;
					++arr[1];
				}
				else if(*oriPixel >= 180.0 && *oriPixel < 270.0) {
					*mapPixel = green;
					++arr[2];
				}
				else if(*oriPixel >= 270.0 && *oriPixel < 360.0) {
					*mapPixel = yellow;
					++arr[3];
				}
			}
		}

		FILE* edges = fopen("edges.txt", "a");

		fprintf(edges, "%d-%d-%d-%d\n", arr[0], arr[1], arr[2], arr[3]);
	
		fclose(edges);
	
		return oriMap;
	}

	int main(void)
	{
	
		// Mat image = Mat::zeros(Size(500, 240), CV_8UC1);
		// circle(image, Point(160, 120), 75, Scalar(155, 125, 155), -1, CV_AA);
		// rectangle(image, Rect(10, 10, 150, 150), Scalar(155,155,155), 10);
	
	
		// Mat image = cv::imread("C:\\Users\\Coker\\Documents\\Difference Finder\\Gyte_DifferenceFinder\\Resources\\1\\ImageMatch.jpg");
		Mat image = imread("image2.png");

		// imwrite("image.jpg", image);
		imshow("original", image);

		Mat Sx;
		Sobel(image, Sx, CV_32F, 2, 0, 3);
	
		Mat Sy;
		Sobel(image, Sy, CV_32F, 0, 2, 3);

		Mat mag, ori;
		magnitude(Sx, Sy, mag);
		phase(Sx, Sy, ori, true);

		Mat oriMap = orientationMap(mag, ori, 50.0);
	
		imshow("Sx", Sx);
		imshow("Sy", Sy);

		imshow("magnitude", mat2gray(mag));
		imshow("orientation", mat2gray(ori));
		imshow("orientation map", oriMap);
		waitKey();
	
		/*
		imwrite("Sx.jpg", Sx);
		imwrite("Sy.jpg", Sy);
		imwrite("magnitude.jpg", mag);
		imwrite("orientation.jpg", ori);
		imwrite("orimap.jpg", oriMap);
		*/

		return 0;
	}

#endif