
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
#include <ctime>
#include <vector>

#include "common.h"

#define MAX_PATH_LENGTH 300
#define MINIMUM_GRAD_MAGNITUDE_FOR_ORIENTATION 0.001
#define PI 3.14159265

#define SHIFTER_AMOUNT 0.05

#define DEFAULT_RATIO 3
#define DEFAULT_THRESHOLD 1350
#define DECREASE_SIZE 50

#define MAX_BUF_SIZE 250

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

typedef struct {
	int wPoint1;
	int hPoint1;
	int wPoint2;
	int hPoint2;

	int difference;
} Tile;

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
	// void findEdgeDetectorOptimumParameter(const cv::Mat& image1, const cv::Mat& image2, double& ratio_1, double& ratio_2, int& threshold_1, int& threshold_2);
	int processImageTileByTile (const cv::Mat& image1, const cv::Mat& image2, int numberOfTile,
								ExcelFormat::BasicExcel& xls, FILE* fPtr);
	int getSumOfTheHistogram(const cv::Mat& hist);
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

	int getSumOfTheHistogram(const cv::Mat& hist) {
		if (hist.empty()) {
			fprintf(stderr, "ERROR getSumOfTheHistogram : histogram is empty !\n");
			scanf("%*c");
			exit(-1);
		}
		
		if (hist.cols != 180) {
			fprintf(stderr, "ERROR getSumOfTheHistogram : Histogram Reprensentation ERROR !\n");
			scanf("%*c");
			exit(-1);
		}

		int sum =0;

		for (int i=0; i<hist.cols; ++i)
			sum += std::abs(hist.at<int>(0,i));

		return sum;
	}

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

	int processImageTileByTile (const cv::Mat& image1, const cv::Mat& image2, int numberOfTile,
						ExcelFormat::BasicExcel& xls, FILE* fPtr) {
		
		if (image1.empty() || image2.empty() || NULL == fPtr) {
			fprintf(stderr, "ERROR processImageTileByTile: inputs have NOT set !\n");
			return -1;
		}
		
		// file content information
		fprintf(fPtr, "wTileSize hTileSize\n\n");
		fprintf(fPtr, "wPoint1 hPoint1 wPoint2 hPoint2\n");
		fprintf(fPtr, "difference totalPoints\n\n");

		cv::Mat hist1,
				hist2;
		
		ExcelFormat::BasicExcelWorksheet* sheet;
		ExcelFormat::XLSFormatManager fmt_mgr(xls);

		ExcelFormat::ExcelFont font_bold;
		font_bold._weight = FW_BOLD; // 700

		ExcelFormat::CellFormat fmt_bold(fmt_mgr);
		fmt_bold.set_font(font_bold);
		// end of excel file config.

		char sheetName[50];
		char diffHist[25];

		int hTileSize = image1.rows/numberOfTile,
			wTileSize = image1.cols/numberOfTile;

		fprintf(fPtr, "%d %d\n\n", wTileSize, hTileSize);

		for (int i=0; i<numberOfTile; ++i) { // form left to right
			for (int j=0; j<numberOfTile; ++j) { // from top to down
				
				fprintf(stdout, "%d-%d processing\n", i,j);

				int diffOfHistogram =0,
					minDiffOfHistogram =0;

				int points1 =0,
					points2 =0,
					totalPoints =0;

				ImageShifterAmount curr, min;
				curr.i =0;
				curr.j =0;
				min = curr;
			
				cv::Rect roi(wTileSize*(i+curr.i), hTileSize*(j+curr.j), wTileSize, hTileSize);
				
				cv::Mat tile1 = getSpecificRegionOfTheImage(image1, roi);
				getHOGFeatures1(tile1, hist1);

				cv::Mat tile2 = getSpecificRegionOfTheImage(image2, roi);
					
				getHOGFeatures1(tile2, hist2);

				cv::imwrite("tiles//tile1.jpg", tile1);
				cv::imwrite("tiles//tile2.jpg", tile2);

				diffOfHistogram = calcDiffHistogram(hist1, hist2);
				minDiffOfHistogram = diffOfHistogram;
				
				points1 = getSumOfTheHistogram(hist1);
				points2 = getSumOfTheHistogram(hist2);

				// apply HOG all possible neighboor to get min diff
				#define SET_DIF_VALUE roi = cv::Rect(wTileSize*(i+curr.i), hTileSize*(j+curr.j), wTileSize, hTileSize); \
									cv::Mat tile2 = getSpecificRegionOfTheImage(image2, roi);							\
																														\
									getHOGFeatures1(tile2, hist2);														\
									diffOfHistogram = calcDiffHistogram(hist1, hist2);									\
																														\
																														\
									if (diffOfHistogram < minDiffOfHistogram) {											\
										minDiffOfHistogram = diffOfHistogram;											\
										min = curr;																		\
										points2 = getSumOfTheHistogram(hist2);											\
										cv::imwrite("tiles//tile1.jpg", tile1);											\
										cv::imwrite("tiles//tile2.jpg", tile2);											\
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

				fprintf(stdout, "%d-%d done\n", i,j);
				sprintf(sheetName, "%d-%d", i, j);
  				sheet = xls.GetWorksheet( i*numberOfTile+j );
				ExcelFormat::XLSFormatManager fmt_mgr( xls );
				sheet->Rename( sheetName );
			
				(sheet->Cell(0,0))->SetFormat( fmt_bold );
				(sheet->Cell(0,1))->SetFormat( fmt_bold );
				(sheet->Cell(0,0))->Set("RgbPano");
				(sheet->Cell(0,1))->Set("RgbImage");

				// histogram data is printed to excel file
				printHistogramExcel(hist1, hist2, sheet);

				int tile1StartPoint_w = wTileSize*i,
					tile1StartPoint_h = hTileSize*j;

				int tile2StartPoint_w = (int) wTileSize*(i+min.i),
					tile2StartPoint_h = (int) hTileSize*(j+min.j);

				// fprintf(stdout, "tile start point %d-%d\n", temp1, temp2);
				totalPoints = points1 + points2 - 3*minDiffOfHistogram;
				fprintf(fPtr, "%d %d %d %d\n", tile1StartPoint_w, tile1StartPoint_h, tile2StartPoint_w, tile2StartPoint_h);
				fprintf(fPtr, "%d %d\n\n", minDiffOfHistogram, totalPoints);

			} // end of for j
		} // end of for i

		return 0;
	}

	GYTE_DIFF_FINDER::difoutputs getDifference(const cv::Mat& const image1, const cv::Mat& const image2,
		const int numberOfTile, int differecenceAlgo) {
	
		GYTE_DIFF_FINDER::difoutputs outputs;
		
		outputs.difImage1 = cv::Mat();
		outputs.difImage2 = cv::Mat();
		outputs.edgeMap1 = cv::Mat();
		outputs.edgeMap2 = cv::Mat();

		if (image1.empty() || image2.empty()) {
			std::cerr << "ERROR getDifference: Some Image File Have NOT set!\n Please set the Image\n";
			scanf("%*c");
			return outputs;
		}

		if (image1.cols != image2.cols || image1.rows != image2.rows) {
			std::cerr << "ERROR getDifference: Images Not Registered !\n";
			scanf("%*c");
			return outputs;
		}

		cv::Mat gaussian1, gaussian2;

		cv::GaussianBlur(image1, gaussian1, cv::Size(5, 5), 0,0);
		cv::GaussianBlur(image2, gaussian2, cv::Size(5, 5), 0,0);

#ifdef DEBUG
		cv::namedWindow("image1", CV_WINDOW_NORMAL);
		cv::namedWindow("image2", CV_WINDOW_NORMAL);
#endif

		cv::Mat hist1,
				hist2;
		
		double ratio_1 =DEFAULT_RATIO,
			   ratio_2 =DEFAULT_RATIO;

		int threshold_1 =DEFAULT_THRESHOLD,
			threshold_2 =DEFAULT_THRESHOLD;

		cv::Mat img1 =image1.clone(),
				img2 =image2.clone();
		
		cv::Mat edgeImage1,
				edgeImage2;

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

		char sheetName[50],
			 diffHist[25];
	     
		xls.New(numberOfTile*numberOfTile);

		int hTileSize =0,
			wTileSize =0;
		
		FILE *resFilePtr = fopen("tileRes.txt", "w");

		if (NULL == resFilePtr) {
			fprintf(stderr, "ERROR getDifference function: resFile.txt Can Not Created !\n");
			scanf("%*c");
			return outputs;
		}

#if defined (DEBUG)
		cv::imshow("image1", img1);
		cv::imshow("image2", img2);
		cv::waitKey(0);
#endif
		
		char readedLine[MAX_BUF_SIZE];

		int wPoint1 =0,
			hPoint1 =0,
			wPoint2 =0,
			hPoint2 =0;

		int difference =0,
			totalDifference =0;

		int lineSize =(img1.cols/900) * (img1.rows/900);

		int numberOfTheDifferentHistogram =0;
		int points =0,
			totalPoints =0;

		for (threshold_2 =750; threshold_2<=1350; threshold_2 += 50) {
			for (threshold_1 =750; threshold_1<=1350; threshold_1 += 50) {
				totalPoints = 0;
				numberOfTheDifferentHistogram = totalDifference =0;

				edgeImage1 =getEdgeImage(image1, ratio_1, threshold_1);
				edgeImage2 =getEdgeImage(image2, ratio_2, threshold_2);

				cv::cvtColor(edgeImage1, edgeImage1, CV_RGB2GRAY);
				cv::cvtColor(edgeImage2, edgeImage2, CV_RGB2GRAY);

				edgeImage1 = dilate(edgeImage1, 3, 3);
				edgeImage2 = dilate(edgeImage2, 3, 3);

				resFilePtr = fopen("tileRes.txt", "w");

				if (NULL == resFilePtr) {
					fprintf(stderr, "ERROR getDifference function: resFile.txt Can Not Created !\n");
					scanf("%*c");
					return outputs;
				}

				cv::imwrite("edgeMap1.bmp", edgeImage1);
				cv::imwrite("edgeMap2.bmp", edgeImage2);

				processImageTileByTile(edgeImage1, edgeImage2, 5, xls, resFilePtr);
				fclose( resFilePtr );
				resFilePtr = fopen("tileRes.txt", "rt");

				for (int i=0; i<5; ++i)
					fgets(readedLine, MAX_BUF_SIZE, resFilePtr);

				fscanf(resFilePtr, "%d%d", &wTileSize, &hTileSize);
				fscanf(resFilePtr, "%*c");

				for (int i=0; i<numberOfTile*numberOfTile; ++i) {
					fscanf(resFilePtr, "%d%d%d%d", &wPoint1, &hPoint1, &wPoint2, &hPoint2);
					fscanf(resFilePtr, "%d%d", &difference, &points);

					totalDifference += difference;

					if (difference > 1100)
						++numberOfTheDifferentHistogram;

					totalPoints += points;
				}
				
				fprintf(stderr, "%d %d - %d\n", threshold_1, threshold_2, totalPoints);
				fclose( resFilePtr );	
			} // end of for (threshold_2)
		} // end of for (threshold 1)
		
		cv::imwrite("edgeMap1.bmp", edgeImage1);
		cv::imwrite("edgeMap2.bmp", edgeImage2);

		fclose( resFilePtr );
		resFilePtr = fopen("tileRes.txt", "rt");

		for (int i=0; i<5; ++i)
			fgets(readedLine, MAX_BUF_SIZE, resFilePtr);

		fscanf(resFilePtr, "%d%d", &wTileSize, &hTileSize);
		fscanf(resFilePtr, "%*c");

		for (int i=0; i<numberOfTile*numberOfTile; ++i) {
			fscanf(resFilePtr, "%d%d%d%d", &wPoint1, &hPoint1, &wPoint2, &hPoint2);
			fscanf(resFilePtr, "%d%d", &difference, &points);

			cv::line(img1, cv::Point(wPoint1, hPoint1), cv::Point(wPoint1+wTileSize, hPoint1),
						cv::Scalar(255, 255, 0), lineSize);
			cv::line(img1, cv::Point(wPoint1, hPoint1), cv::Point(wPoint1, hPoint1+hTileSize),
						cv::Scalar(255, 255, 0), lineSize);
			cv::line(img1, cv::Point(wPoint1+wTileSize, hPoint1), cv::Point(wPoint1+wTileSize, hPoint1+hTileSize),
						cv::Scalar(255, 255, 0), lineSize);
			cv::line(img1, cv::Point(wPoint1, hPoint1+hTileSize), cv::Point(wPoint1+wTileSize, hPoint1+hTileSize),
						cv::Scalar(255, 255, 0), lineSize);

			cv::line(img2, cv::Point(wPoint2, hPoint2), cv::Point(wPoint2, hPoint2+hTileSize),
						cv::Scalar(255,255,0), lineSize);
			cv::line(img2, cv::Point(wPoint2, hPoint2), cv::Point(wPoint2+wTileSize, hPoint2),
						cv::Scalar(255,255,0), lineSize);
			cv::line(img2, cv::Point(wPoint2, hPoint2+hTileSize), cv::Point(wPoint2+wTileSize, hPoint2+hTileSize),
						cv::Scalar(255,255,0), lineSize);
			cv::line(img2, cv::Point(wPoint2+wTileSize, hPoint2), cv::Point(wPoint2+wTileSize, hPoint2+hTileSize),
						cv::Scalar(255,255,0), lineSize);

			sprintf(diffHist, "%d", difference);

			if (difference>1100) { // it is fake if statement to see all result for now
				putText(img1, diffHist, cv::Point(wPoint1+wTileSize*0.1, hPoint1+hTileSize*0.5), cv::FONT_HERSHEY_COMPLEX_SMALL, lineSize+1,
						cv::Scalar(255, 255, 0), lineSize+1, CV_AA);

				putText(img2, diffHist, cv::Point(wPoint2+wTileSize*0.1, hPoint2+hTileSize*0.5), cv::FONT_HERSHEY_COMPLEX_SMALL, lineSize+1,
						cv::Scalar(255, 255, 0), lineSize+1, CV_AA);
			} // end of if (difference > 1100)
		} // end of for i

		fprintf(stderr, "average difference: %d\n", totalDifference/(numberOfTile*numberOfTile) );

		outputs.edgeMap1 =edgeImage1;
		outputs.edgeMap2 =edgeImage2;
		outputs.difImage1 =img1;
		outputs.difImage2 =img2;
		outputs.histogram =xls;
		fclose( resFilePtr );

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
		
		if (Histogram1.empty() || Histogram2.empty()) {
			fprintf(stderr, "ERROR calcDiffHistogram: Histogram empty !\n");
			scanf("%*c");
			exit(-1);
		}
		
		if (Histogram1.cols != 180 || Histogram2.cols != 180) {
			std::cerr << "ERROR calcDiffHistogram: Histogram Reprensentation ERROR !\n";
			scanf("%*c");
			exit(-1);
		}	

		double diff =0.0;
		int localDiff =0;
	
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