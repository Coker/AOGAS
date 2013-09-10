
#include "DiffFinderHOG.h"

#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

// excel library include
#include "excelLib//ExcelFormat.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <crtdbg.h>

#include <iostream>
#include <string>

#include "common.h"

#include <ctime>

#define MAX_PATH_LENGTH 300
#define MINIMUM_GRAD_MAGNITUDE_FOR_ORIENTATION 0.001
#define PI 3.14159265

#define SHIFTER_AMOUNT 0.05

double ratio_1 =3, ratio_2 =3;
int threshold_1 =950, threshold_2 =1350;

#if defined(MIN)
	#undef MIN
	#define MIN(x, y) (y) + (((x) - (y)) & (((x) - (y)) >> (sizeof(int) * CHAR_BIT - 1)))
#elif
	#define MIN(x,y) (y) + (((x) - (y)) & (((x) - (y)) >> (sizeof(int) * CHAR_BIT - 1)))
#endif

#define EQUAL_HIST(image) cv::cvtColor((image), (image), CV_BGR2GRAY);  \
						  cv::equalizeHist((image), (image));
typedef struct {
	double i;
	double j;
} ImageShifterAmount;

namespace {
	// functions declerations
	void getHOGFeatures1(const cv::Mat& const InputImage, cv::Mat & Histogram);
	GYTE_DIFF_FINDER::difoutputs getDifference(const cv::Mat& const image1, const cv::Mat& const image2,
											   const int numberOfTile, int differecenceAlgo);
	int calcDiffHistogram(const cv::Mat& const Histogram1, const cv::Mat& const Histogram2);
	void printHistogramExcel(const cv::Mat& const hist1, const cv::Mat& const hist2,
							 ExcelFormat::BasicExcelWorksheet* sheet);
	cv::Mat getSpecificRegionOfTheImage(cv::Mat image, cv::Rect region);
	cv::Mat getEdgeImage(const cv::Mat& image, double ratio, double lowThreshold);
	cv::Mat smoothHist(const cv::Mat& hist, int width, double height);
	cv::Mat dilate(const cv::Mat& src, const int dilationSize, const int dilationType);
	cv::Mat erade(const cv::Mat& src, const int eradeSize, const int eradeType);
	void printHist2Screen(const cv::Mat& histogram);
	void findEdgeDetectorOptimumParameter(const cv::Mat& image1, const cv::Mat& image2,
					double& ratio_1, double& ratio_2, int& threshold_1, int& threshold_2);
} // end of unnamed namespace

GYTE_DIFF_FINDER::DiffFinderHOG::DiffFinderHOG() : DiffFinder() {
	/* Deliberately empty */
}

GYTE_DIFF_FINDER::DiffFinderHOG::DiffFinderHOG(const cv::Mat& rgMapbPano, const cv::Mat& depthMapPano) 
	: DiffFinder(depthMapPano, rgbMapPano)
{
	/* Deliberately empty */
}

void GYTE_DIFF_FINDER::DiffFinderHOG::getDiff(GYTE_DIFF_FINDER::AffRect registerCoordinate, 
					const cv::Mat& rgbImage, const char* const outputFolder) const {
	
	int tileNumber =5; 
	
	GYTE_DIFF_FINDER::difoutputs res;

	time_t before, after;
	struct tm newyear;
	double seconds;

	std::string outputDocTemp;

	time(&before);
	newyear = *localtime(&before);
	newyear.tm_hour = 0; newyear.tm_min = 0; newyear.tm_sec = 0;
	newyear.tm_mon = 0;  newyear.tm_mday = 1;

	res = getDifference(rgbMapPano, rgbImage, tileNumber, HOG);

	time(&after);
	seconds = difftime(before, after);

	if (res.difImage1.empty())
		return;

	// the output documents (images, excel files, text files, etc) writes appropriate folder
	outputDocTemp = std::string(outputFolder);
	outputDocTemp += "//edgeMap1.jpg";
	cv::imwrite(outputDocTemp.data(), res.edgeMap1);

	outputDocTemp = std::string(outputFolder);
	outputDocTemp += "//edgeMap2.jpg";
	cv::imwrite(outputDocTemp.data(), res.edgeMap2);

	outputDocTemp = std::string(outputFolder);
	outputDocTemp += "//diff1.jpg";
	cv::imwrite(outputDocTemp.data(), res.difImage1);

	outputDocTemp = std::string(outputFolder);
	outputDocTemp += "//diff2.jpg";
	cv::imwrite(outputDocTemp.data(), res.difImage2);

	outputDocTemp = std::string(outputFolder);
	outputDocTemp += "//histogram.xls";
	res.histogram.SaveAs(outputDocTemp.data());

	outputDocTemp = std::string(outputFolder);
	outputDocTemp += "//resDoc.txt";
	
	FILE* resDoc = fopen(outputDocTemp.data(), "w");

	if (NULL == resDoc) {
		fprintf(stderr, "File Not Created !");
		scanf("%*d");
		return;
	}

	fprintf(resDoc, "run time %f seconds", std::fabs(seconds));
	fclose(resDoc);
}

void GYTE_DIFF_FINDER::DiffFinderHOG::getDiff(GYTE_DIFF_FINDER::AffRect regist8erCoordinate, const cv::Mat& depthImage,
	 const cv::Mat& rgbImage, const char* const outputFolder) const {
	/* This function hasn't defined yet */
}

namespace {
	// functions definitions
	void printHist2Screen(const cv::Mat& histogram) {

		for (int i=0; i<histogram.cols; ++i) {
			fprintf(stdout, "%3d-", histogram.at<int>(0,i));
			
			if ( ((i+1)%18) == 0)
				printf("\n");
		}
			
	
		fprintf(stdout, "\n****** end of the Histogram ********\n");
	}

	// functions definitions
	cv::Mat getSpecificRegionOfTheImage(cv::Mat image, cv::Rect region) {
		cv::Mat specificReg = image(region);
		cv::imwrite("specificReg.bmp", specificReg);

		specificReg.release();
		specificReg = cv::imread("specificReg.bmp");
		cv::cvtColor(specificReg, specificReg, CV_RGB2GRAY);
		return specificReg;
	}

	cv::Mat getEdgeImage(const cv::Mat& image, double ratio, double lowThreshold) {
		cv::Mat src = image.clone();
		cv::Mat src_gray;
		cv::Mat dst, detected_edges;

		int kernel_size = 5;
		char* window_name = "Edge Map";

		if( !src.data ) { 
			std::cerr << "Edge Detector : Input Image is empty !";
			return cv::Mat(); 
		}

		/// Create a matrix of the same type and size as src (for dst)
		dst.create( src.size(), src.type() );

		/// Convert the image to grayscale
		cvtColor( src, src_gray, CV_BGR2GRAY );

		blur( src_gray, detected_edges, cv::Size(3,3) );

		/// Canny detector
		Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size, true);
		
		/// Using Canny's output as a mask, we display our result
		dst = cv::Scalar::all(0);

		src.copyTo( dst, detected_edges);
		// imshow( window_name, dst );

		return dst.clone();
	}

	cv::Mat smoothHist(const cv::Mat& hist, int width, double height) {
		cv::Mat smoothed = hist.clone();
		
		for (int i=2; i<hist.cols-2; ++i) {
			int val = hist.at<int>(0,i);
			int newVal = hist.at<int>(0,i-2)*0.1 + hist.at<int>(0,i-1)*0.2 + hist.at<int>(0,i)*0.4 + hist.at<int>(0,i+1)*0.2 + hist.at<int>(0,i+2)*0.1; 
			smoothed.at<int>(0,i) = newVal;
		}
		
		return smoothed;
	}

	cv::Mat dilate(const cv::Mat& src, const int dilationSize, const int dilationType) {
		cv::Mat dilated;
		cv::Mat config;

		config = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*dilationSize+1, 2*dilationSize+1), 
									       cv::Point(dilationSize, dilationSize));
		dilate(src, dilated, config);
		return dilated;
	}

	cv::Mat erade(const cv::Mat& src, const int eradeSize, const int eradeType) {
		cv::Mat eroded;
		cv::Mat config;

		config = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*eradeSize, 2*eradeSize),
									   cv::Point(eradeSize, eradeSize));
		cv::erode(src, eroded, config);

		return eroded;
	}

	void findEdgeDetectorOptimumParameter(const cv::Mat& image1, const cv::Mat& image2,
		double& ratio_1, double& ratio_2, int& threshold_1, int& threshold_2) {
		
		if (image1.empty() || image2.empty()) {
			fprintf(stderr, "findEdgeDetectorOptimumParameter: Inputs hasn't set\n");
			return;
		}



		return;
	}

	GYTE_DIFF_FINDER::difoutputs getDifference(const cv::Mat& const image1, const cv::Mat& const image2,
		const int numberOfTile, int differecenceAlgo) {
	
		GYTE_DIFF_FINDER::difoutputs outputs;
		
		outputs.difImage1 = cv::Mat();
		outputs.difImage2 = cv::Mat();
		outputs.edgeMap1 = cv::Mat();
		outputs.edgeMap2 = cv::Mat();

		if (image1.empty() || image2.empty()) {
			std::cerr << "Some Image File Have NOT set!\n Please set the Image\n";
			return outputs;
		}

		if (image1.cols != image2.cols || image1.rows != image2.rows) {
			std::cerr << "Images Not Registered !\n";
			return outputs;
		}

		cv::Mat gaussian1, gaussian2;

		cv::GaussianBlur(image1, gaussian1, cv::Size(5, 5), 0,0);
		cv::GaussianBlur(image2, gaussian2, cv::Size(5, 5), 0,0);

#ifdef DEBUG
		cv::namedWindow("image1", CV_WINDOW_NORMAL);
		cv::namedWindow("image2", CV_WINDOW_NORMAL);
#endif

		cv::Mat hist1;
		cv::Mat hist2;
		
		cv::Mat img1 =image1.clone();
		cv::Mat img2 =image2.clone();
		
		cv::Mat edgeImage1 =getEdgeImage(image1, ratio_1, threshold_1);
		cv::Mat edgeImage2 =getEdgeImage(image2, ratio_2, threshold_2);

		edgeImage1 = dilate(edgeImage1, 3, 3);
		edgeImage2 = dilate(edgeImage2, 3, 3);

		cv::imwrite("edgeMap1.bmp", edgeImage1);
		cv::imwrite("edgeMap2.bmp", edgeImage2);
		
		getHOGFeatures1(edgeImage1, hist1);
		getHOGFeatures1(edgeImage2, hist2);

		printf("Histogram dif of the Images %d\n", calcDiffHistogram(hist1, hist2));
		
		// edgeImage1 = dilate(edgeImage1, 3, 3);
		// edgeImage2 = dilate(edgeImage2, 3, 3);

		cv::cvtColor(edgeImage1, edgeImage1, CV_RGB2GRAY);
		cv::cvtColor(edgeImage2, edgeImage2, CV_RGB2GRAY);

		outputs.edgeMap1 = edgeImage1;
		outputs.edgeMap2 = edgeImage2;

#if defined (DEBUG)
		cv::imwrite("edgeImage1.jpg", edgeImage1);
		cv::imwrite("egdeImage2.jpg", edgeImage2);
		return outputs;
#endif		
		// Excel file config.
		ExcelFormat::BasicExcel xls;
		ExcelFormat::BasicExcelWorksheet* sheet;
		ExcelFormat::XLSFormatManager fmt_mgr(xls);

		ExcelFormat::ExcelFont font_bold;
		font_bold._weight = FW_BOLD; // 700

		ExcelFormat::CellFormat fmt_bold(fmt_mgr);
		fmt_bold.set_font(font_bold);
		// end of excel file config.

		char sheetName[50];
		char diffHist[25];
	     
		xls.New(numberOfTile*numberOfTile);

		int hTileSize = image1.rows;
		hTileSize /= numberOfTile;
	
		int wTileSize = image1.cols;
		wTileSize /= numberOfTile;

#if defined (DEBUG)
		cv::imshow("image1", img1);
		cv::imshow("image2", img2);
		cv::waitKey(0);
#endif

		for (int i=0; i<numberOfTile; ++i) { // form left to right
			for (int j=0; j<numberOfTile; ++j) { // from top to down
				
				printf("%d-%d processing\n", i,j);

				int diffOfHistogram =0;
				int minDiffOfHistogram =0;
			
				ImageShifterAmount curr, min;
				curr.i =0;
				curr.j =0;
				min = curr;
			
				cv::Rect roi(wTileSize*(i+curr.i), hTileSize*(j+curr.j), wTileSize, hTileSize);
				
				cv::Mat tile1 = getSpecificRegionOfTheImage(edgeImage1, roi);
				getHOGFeatures1(tile1, hist1);
				
				if (HOG == differecenceAlgo) {
					// default tile
					cv::Mat tile2 = getSpecificRegionOfTheImage(edgeImage2, roi);
					
					getHOGFeatures1(tile2, hist2);

					cv::imwrite("tiles//tile1.jpg", tile1);
				    cv::imwrite("tiles//tile2.jpg", tile2);

					diffOfHistogram = calcDiffHistogram(hist1, hist2);
					minDiffOfHistogram = diffOfHistogram;

					// apply HOG all possible neighboor to get min diff
					
					#define SET_DIF_VALUE roi = cv::Rect(wTileSize*(i+curr.i), hTileSize*(j+curr.j), wTileSize, hTileSize); \
										  cv::Mat tile2 = getSpecificRegionOfTheImage(edgeImage2, roi);						\
																															\
										  getHOGFeatures1(tile2, hist2);													\
										  diffOfHistogram = calcDiffHistogram(hist1, hist2);								\
																															\
																															\
										  if (diffOfHistogram < minDiffOfHistogram) {										\
												minDiffOfHistogram = diffOfHistogram;										\
												min = curr;																	\
												cv::imwrite("tiles//tile1.jpg", tile1);										\
												cv::imwrite("tiles//tile2.jpg", tile2);										\
										  }																					\
																															\
									      tile2.release();

					// -->
					if (i != numberOfTile-1) {
						curr.i = SHIFTER_AMOUNT;
						curr.j = 0;
						
						SET_DIF_VALUE
					}

					//    -->
					//   |
					//   V

					if ( !(i == numberOfTile-1 || j == numberOfTile-1) ) {
						curr.i = SHIFTER_AMOUNT;
						curr.j = SHIFTER_AMOUNT;
						
						SET_DIF_VALUE
					}

					//   |
					//   V
					if (j != numberOfTile-1) {
						curr.i = 0;
						curr.j = SHIFTER_AMOUNT;

						SET_DIF_VALUE
					}

					//  < --
					//     |
					//     V
					if ( !(i == 0 || j == numberOfTile-1) ) {
						curr.i = -SHIFTER_AMOUNT;
						curr.j = SHIFTER_AMOUNT;

						SET_DIF_VALUE
					}

					// <--
					if (0 != i) {
						curr.i = -SHIFTER_AMOUNT;
						curr.j = 0;

						SET_DIF_VALUE
					}

					//    ^
					//    |
					// <--
					if ( !(i == 0 || j == 0) ) {
						curr.i = -SHIFTER_AMOUNT;
						curr.j = -SHIFTER_AMOUNT;

						SET_DIF_VALUE
					}
					
					//  ^ 
					//  |
					if ( j != 0) {
						curr.i = 0;
						curr.j = -SHIFTER_AMOUNT;

						SET_DIF_VALUE
					}

					//  ^ 
					//  |__>
					if ( !(j == 0 || i == numberOfTile-1) ) {
						curr.i = SHIFTER_AMOUNT;
						curr.j = -SHIFTER_AMOUNT;

						SET_DIF_VALUE
					}

					#undef SET_DIF_VALUE
				} // end of The if (HOG == differenceAlgo)	

				sprintf(diffHist, "%d", minDiffOfHistogram);

				int lineSize = (img1.cols/900) * (img1.rows/900);
				if (minDiffOfHistogram > 1100) {
					cv::line(img1, cv::Point(wTileSize*i, hTileSize*j), cv::Point(wTileSize*(i+1), hTileSize*j),
							cv::Scalar(255,0,0), lineSize);
					cv::line(img1, cv::Point(wTileSize*(i+1), hTileSize*j), cv::Point(wTileSize*(i+1), hTileSize*(j+1)),
							cv::Scalar(255,0,0), lineSize);
					cv::line(img1, cv::Point(wTileSize*i, hTileSize*(j+1)), cv::Point(wTileSize*(i+1), hTileSize*(j+1)),
							cv::Scalar(255,0,0), lineSize);
					cv::line(img1, cv::Point(wTileSize*i, hTileSize*j), cv::Point(wTileSize*i, hTileSize*(j+1)),
							cv::Scalar(255,255,0), lineSize);

					cv::line(img2, cv::Point(wTileSize*(i+min.i), hTileSize*(j+min.j)), cv::Point(wTileSize*((i+min.i)+1), hTileSize*(j+min.j)),
							cv::Scalar(255,0,0), lineSize);
					cv::line(img2, cv::Point(wTileSize*((i+min.i)+1), hTileSize*(j+min.j)), cv::Point(wTileSize*((i+min.i)+1), hTileSize*((j+min.j)+1)),
							cv::Scalar(255,0,0), lineSize);
					cv::line(img2, cv::Point(wTileSize*(i+min.i), hTileSize*((j+min.j)+1)), cv::Point(wTileSize*((i+min.i)+1), hTileSize*((j+min.j)+1)),
							cv::Scalar(255,0,0), lineSize);
					cv::line(img2, cv::Point(wTileSize*(i+min.i), hTileSize*(j+min.j)), cv::Point(wTileSize*(i+min.i), hTileSize*((j+min.j)+1)),
							cv::Scalar(255,0,0), lineSize);
					
					putText(img1, diffHist, cv::Point(wTileSize*(i+0.1), hTileSize*(j+0.5)), cv::FONT_HERSHEY_COMPLEX_SMALL, lineSize+1,
							cv::Scalar(255,255,0), lineSize+1, CV_AA);
					putText(img2, diffHist, cv::Point(wTileSize*(i+0.1), hTileSize*(j+0.5)), cv::FONT_HERSHEY_COMPLEX_SMALL, lineSize+1, 
							cv::Scalar(255,255,0), lineSize+1, CV_AA);

				}
				
				printf("%d-%d done\n", i,j);
				sprintf(sheetName, "%d-%d", i, j);
  				sheet = xls.GetWorksheet(i*numberOfTile+j);
				ExcelFormat::XLSFormatManager fmt_mgr(xls);
				sheet->Rename(sheetName);
			
				(sheet->Cell(0,0))->SetFormat(fmt_bold);
				(sheet->Cell(0,1))->SetFormat(fmt_bold);
				(sheet->Cell(0,0))->Set("RgbPano");
				(sheet->Cell(0,1))->Set("RgbImage");

				// histogram data is printed to excel file
				printHistogramExcel(hist1, hist2, sheet);
				
				// cv::imshow("tile1", tile1); cv::imshow("tile2", tile2); 
				// cv::imshow("image1", img1); cv::imshow("image2", img2); cv::waitKey();
				
			}
		}


		outputs.difImage1 =img1;
		outputs.difImage2 =img2;
		outputs.histogram = xls;

#if defined( DEBUG )
		xls.SaveAs("histograms.xls");
		cv::imshow("image1", img1), cv::imshow("image2", img2); cv::waitKey(0);
		cv::imwrite("diff1.bmp", img1);
		cv::imwrite("diff2.bmp", img2);
#endif
		return outputs;
	}

	void printHistogramExcel(const cv::Mat& const hist1, const cv::Mat& const hist2,
							 ExcelFormat::BasicExcelWorksheet* sheet) {
	
		int hist1Val =0,
			hist2Val =0;

		ExcelFormat::BasicExcelCell* cellHist1;
		ExcelFormat::BasicExcelCell* cellHist2;
	     
		for (int i=1; i<=180; ++i) {
			hist1Val = hist1.at<int>(0, i-1);
			hist2Val = hist2.at<int>(0, i-1);

			cellHist1 = sheet->Cell(i, 0);
			cellHist2 = sheet->Cell(i, 1);

			cellHist1->Set(hist1Val);
			cellHist2->Set(hist2Val);
		}// end of for
	}

	int calcDiffHistogram(const cv::Mat& const Histogram1, const cv::Mat& const Histogram2) {
		if (Histogram1.cols != 180 || Histogram2.cols != 180) {
			std::cerr << "Histogram Reprensentation ERROR !\n";
			exit(-1);
		}	

		double diff =0;
		int localDiff;
	
		for (int i=0; i<180; ++i) {
			localDiff = std::abs(Histogram1.at<int>(0,i) - Histogram2.at<int>(0,i));

			diff += std::pow((double)localDiff, 2.0);
		}

		diff = std::sqrt((double)diff);

		return (int) diff;
	}

	void getHOGFeatures1(const cv::Mat& const InputImage, cv::Mat& Histogram) {
		cv::Mat gradH, gradV, imageO, imageM;

		cv::Sobel(InputImage, gradH, cv::DataType<float>::type, 1, 0, 3, 1.0, 0.0, cv::BORDER_DEFAULT);
		cv::Sobel(InputImage, gradV, cv::DataType<float>::type, 0, 1, 3, 1.0, 0.0, cv::BORDER_DEFAULT);
		
		imageM.create(InputImage.rows, InputImage.cols, cv::DataType<float>::type);
		imageO.create(InputImage.rows, InputImage.cols, cv::DataType<float>::type);

		// calculate magnitude and orientation images...
		float maxM = 0;
		int r, c;
		for (r=0;r<InputImage.rows;r++) {
			for (c=0;c<InputImage.cols;c++) {
				imageO.at<float>(r,c) = (float)(atan2(gradV.at<float>(r,c),gradH.at<float>(r,c)));
				imageM.at<float>(r,c) = gradH.at<float>(r,c)*gradH.at<float>(r,c) + gradV.at<float>(r,c)*gradV.at<float>(r,c);
				if (imageM.at<float>(r,c)>maxM)
					maxM = imageM.at<float>(r,c);
			}
		}

		
		// normalize magnitude image to 1...
		for (r=0;r<InputImage.rows;r++) {
			for (c=0;c<InputImage.cols;c++) {
				imageM.at<float>(r,c) /= maxM;
			}
		}

		// form the histogram - will get rid of small magnitude orientations
		Histogram.create(1, 180, cv::DataType<int>::type);
		for(c=0; c<Histogram.cols; c++) {
			Histogram.at<int>(0,c) = 0;
		}

		float stepSize = (float)(2.0*PI/(float)Histogram.cols);
		for (r=3;r<InputImage.rows-3;r++) {
			for (c=3;c<InputImage.cols-3;c++) {
				if (imageM.at<float>(r,c)>MINIMUM_GRAD_MAGNITUDE_FOR_ORIENTATION) {
					float theta = imageO.at<float>(r,c); // between -pi and pi...
					theta += (float)PI;
					int count = (int)(theta / stepSize);
					if (count>=0 && count<Histogram.cols) Histogram.at<int>(0,count) += 1;
				}
				else { 
				}
			}
		}

		// imshow("Orient Image", imageO); imshow("Magnit Image", imageM); cv::waitKey(0);
		Histogram = smoothHist(Histogram, 0, 0);
	} // end-getHOGFeatures1

} // end of unnamed namespace