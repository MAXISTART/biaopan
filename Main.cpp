/*
This code is intended for academic use only.
You are free to use and modify the code, at your own risk.

If you use this code, or find it useful, please refer to the paper:

Michele Fornaciari, Andrea Prati, Rita Cucchiara,
A fast and effective ellipse detector for embedded vision applications
Pattern Recognition, Volume 47, Issue 11, November 2014, Pages 3693-3708, ISSN 0031-3203,
http://dx.doi.org/10.1016/j.patcog.2014.05.012.
(http://www.sciencedirect.com/science/article/pii/S0031320314001976)


The comments in the code refer to the abovementioned paper.
If you need further details about the code or the algorithm, please contact me at:

michele.fornaciari@unimore.it

last update: 23/12/2014
*/

#include <opencv2\opencv.hpp>
#include <algorithm>
#include "EllipseDetectorYaed.h"
#include <fstream>
#include <direct.h>

#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"

#define pi 3.1415926

using namespace std;
using namespace cv;




//void OnVideo()
//{
//	VideoCapture cap(0);
//	if(!cap.isOpened()) return;
//
//	CEllipseDetectorYaed yaed;
//
//	Mat1b gray;
//	while(true)
//	{	
//		Mat3b image;
//		cap >> image;
//		cvtColor(image, gray, CV_BGR2GRAY);	
//
//		vector<CEllipse> ellipses;
//
//		//Find Ellipses		
//		yaed.Detect(gray, ellipses);
//		yaed.DrawDetectedEllipses(image,ellipses);
//		imshow("Output", image);
//
//			
//		if(waitKey(10) >= 0) break;
//	}
//}


// Should be checked
void SaveEllipses(const string& workingDir, const string& imgName, const vector<Ellipse>& ellipses /*, const vector<double>& times*/)
{
	string path(workingDir + "/" + imgName + ".txt");
	ofstream out(path, ofstream::out | ofstream::trunc);
	if (!out.good())
	{
		cout << "Error saving: " << path << endl;
		return;
	}

	// Save execution time
	//out << times[0] << "\t" << times[1] << "\t" << times[2] << "\t" << times[3] << "\t" << times[4] << "\t" << times[5] << "\t" << "\n";

	unsigned n = ellipses.size();
	// Save number of ellipses
	out << n << "\n";

	// Save ellipses
	for (unsigned i = 0; i < n; ++i)
	{
		const Ellipse& e = ellipses[i];
		out << e._xc << "\t" << e._yc << "\t" << e._a << "\t" << e._b << "\t" << e._rad << "\t" << e._score << "\n";
	}
	out.close();
}

// Should be checked
bool LoadTest(vector<Ellipse>& ellipses, const string& sTestFileName, vector<double>& times, bool bIsAngleInRadians = true)
{
	ifstream in(sTestFileName);
	if (!in.good())
	{
		cout << "Error opening: " << sTestFileName << endl;
		return false;
	}

	times.resize(6);
	in >> times[0] >> times[1] >> times[2] >> times[3] >> times[4] >> times[5];

	unsigned n;
	in >> n;

	ellipses.clear();

	if (n == 0) return true;

	ellipses.reserve(n);

	while (in.good() && n--)
	{
		Ellipse e;
		in >> e._xc >> e._yc >> e._a >> e._b >> e._rad >> e._score;

		if (!bIsAngleInRadians)
		{
			e._rad = e._rad * float(CV_PI / 180.0);
		}

		e._rad = fmod(float(e._rad + 2.0*CV_PI), float(CV_PI));

		if ((e._a > 0) && (e._b > 0) && (e._rad >= 0))
		{
			ellipses.push_back(e);
		}
	}
	in.close();

	// Sort ellipses by decreasing score
	sort(ellipses.begin(), ellipses.end());

	return true;
}


void LoadGT(vector<Ellipse>& gt, const string& sGtFileName, bool bIsAngleInRadians = true)
{
	ifstream in(sGtFileName);
	if (!in.good())
	{
		cout << "Error opening: " << sGtFileName << endl;
		return;
	}

	unsigned n;
	in >> n;

	gt.clear();
	gt.reserve(n);

	while (in.good() && n--)
	{
		Ellipse e;
		in >> e._xc >> e._yc >> e._a >> e._b >> e._rad;

		if (!bIsAngleInRadians)
		{
			// convert to radians
			e._rad = float(e._rad * CV_PI / 180.0);
		}

		if (e._a < e._b)
		{
			float temp = e._a;
			e._a = e._b;
			e._b = temp;

			e._rad = e._rad + float(0.5*CV_PI);
		}

		e._rad = fmod(float(e._rad + 2.f*CV_PI), float(CV_PI));
		e._score = 1.f;
		gt.push_back(e);
	}
	in.close();
}

bool TestOverlap(const Mat1b& gt, const Mat1b& test, float th)
{
	float fAND = float(countNonZero(gt & test));
	float fOR = float(countNonZero(gt | test));
	float fsim = fAND / fOR;

	return (fsim >= th);
}

int Count(const vector<bool> v)
{
	int counter = 0;
	for (unsigned i = 0; i < v.size(); ++i)
	{
		if (v[i]) { ++counter; }
	}
	return counter;
}


// Should be checked !!!!!
std::tuple<float, float, float> Evaluate(const vector<Ellipse>& ellGT, const vector<Ellipse>& ellTest, const float th_score, const Mat3b& img)
{
	float threshold_overlap = 0.8f;
	//float threshold = 0.95f;

	unsigned sz_gt = ellGT.size();
	unsigned size_test = ellTest.size();

	unsigned sz_test = unsigned(min(1000, int(size_test)));

	vector<Mat1b> gts(sz_gt);
	vector<Mat1b> tests(sz_test);

	for (unsigned i = 0; i < sz_gt; ++i)
	{
		const Ellipse& e = ellGT[i];

		Mat1b tmp(img.rows, img.cols, uchar(0));
		ellipse(tmp, Point(e._xc, e._yc), Size(e._a, e._b), e._rad * 180.0 / CV_PI, 0.0, 360.0, Scalar(255), -1);
		gts[i] = tmp;
	}

	for (unsigned i = 0; i < sz_test; ++i)
	{
		const Ellipse& e = ellTest[i];

		Mat1b tmp(img.rows, img.cols, uchar(0));
		ellipse(tmp, Point(e._xc, e._yc), Size(e._a, e._b), e._rad * 180.0 / CV_PI, 0.0, 360.0, Scalar(255), -1);
		tests[i] = tmp;
	}

	Mat1b overlap(sz_gt, sz_test, uchar(0));
	for (int r = 0; r < overlap.rows; ++r)
	{
		for (int c = 0; c < overlap.cols; ++c)
		{
			overlap(r, c) = TestOverlap(gts[r], tests[c], threshold_overlap) ? uchar(255) : uchar(0);
		}
	}

	int counter = 0;

	vector<bool> vec_gt(sz_gt, false);

	for (int i = 0; i < sz_test; ++i)
	{
		const Ellipse& e = ellTest[i];
		for (int j = 0; j < sz_gt; ++j)
		{
			if (vec_gt[j]) { continue; }

			bool bTest = overlap(j, i) != 0;

			if (bTest)
			{
				vec_gt[j] = true;
				break;
			}
		}
	}

	int tp = Count(vec_gt);
	int fn = int(sz_gt) - tp;
	int fp = size_test - tp; // !!!!

	float pr(0.f);
	float re(0.f);
	float fmeasure(0.f);

	if (tp == 0)
	{
		if (fp == 0)
		{
			pr = 1.f;
			re = 0.f;
			fmeasure = (2.f * pr * re) / (pr + re);
		}
		else
		{
			pr = 0.f;
			re = 0.f;
			fmeasure = 0.f;
		}
	}
	else
	{
		pr = float(tp) / float(tp + fp);
		re = float(tp) / float(tp + fn);
		fmeasure = (2.f * pr * re) / (pr + re);
	}

	return make_tuple(pr, re, fmeasure);
}

void OnImage()
{
	string sWorkingDir = "C:/Users/miki/Pictures";
	string imagename = "Cloud.bmp";

	string filename = sWorkingDir + "/images/" + imagename;

	// Read image
	Mat3b image = imread(filename);
	Size sz = image.size();

	// Convert to grayscale
	Mat1b gray;
	cvtColor(image, gray, CV_BGR2GRAY);


	// Parameters Settings (Sect. 4.2)
	int		iThLength = 16;
	float	fThObb = 3.0f;
	float	fThPos = 1.0f;
	float	fTaoCenters = 0.05f;
	int 	iNs = 16;
	float	fMaxCenterDistance = sqrt(float(sz.width*sz.width + sz.height*sz.height)) * fTaoCenters;

	float	fThScoreScore = 0.4f;

	// Other constant parameters settings. 

	// Gaussian filter parameters, in pre-processing
	Size	szPreProcessingGaussKernelSize = Size(5, 5);
	double	dPreProcessingGaussSigma = 1.0;

	float	fDistanceToEllipseContour = 0.1f;	// (Sect. 3.3.1 - Validation)
	float	fMinReliability = 0.4f;	// Const parameters to discard bad ellipses


	// Initialize Detector with selected parameters
	CEllipseDetectorYaed yaed;
	yaed.SetParameters(szPreProcessingGaussKernelSize,
		dPreProcessingGaussSigma,
		fThPos,
		fMaxCenterDistance,
		iThLength,
		fThObb,
		fDistanceToEllipseContour,
		fThScoreScore,
		fMinReliability,
		iNs
		);


	// Detect
	vector<Ellipse> ellsYaed;
	Mat1b gray_clone = gray.clone();
	yaed.Detect(gray_clone, ellsYaed);

	vector<double> times = yaed.GetTimes();
	cout << "--------------------------------" << endl;

	cout << "Execution Time: " << endl;
	cout << "Edge Detection: \t" << times[0] << endl;
	cout << "Pre processing: \t" << times[1] << endl;
	cout << "Grouping:       \t" << times[2] << endl;
	cout << "Estimation:     \t" << times[3] << endl;
	cout << "Validation:     \t" << times[4] << endl;
	cout << "Clustering:     \t" << times[5] << endl;
	cout << "--------------------------------" << endl;
	cout << "Total:	         \t" << yaed.GetExecTime() << endl;
	cout << "--------------------------------" << endl;


	vector<Ellipse> gt;
	LoadGT(gt, sWorkingDir + "/gt/" + "gt_" + imagename + ".txt", true); // Prasad is in radians

	Mat3b resultImage = image.clone();

	// Draw GT ellipses
	for (unsigned i = 0; i < gt.size(); ++i)
	{
		Ellipse& e = gt[i];
		Scalar color(0, 0, 255);
		ellipse(resultImage, Point(cvRound(e._xc), cvRound(e._yc)), Size(cvRound(e._a), cvRound(e._b)), e._rad*180.0 / CV_PI, 0.0, 360.0, color, 3);
	}

	yaed.DrawDetectedEllipses(resultImage, ellsYaed);


	Mat3b res = image.clone();

	Evaluate(gt, ellsYaed, fThScoreScore, res);


	imshow("Yaed", resultImage);
	waitKey();
}

void OnDataset()
{
	string sWorkingDir = "D:\\data\\ellipse_dataset\\Random Images - Dataset #1\\";
	//string sWorkingDir = "D:\\data\\ellipse_dataset\\Prasad Images - Dataset Prasad\\";
	string out_folder = "D:\\data\\ellipse_dataset\\";

	vector<string> names;

	vector<float> prs;
	vector<float> res;
	vector<float> fms;
	vector<double> tms;

	//glob(sWorkingDir + "images\\" + "*.*", names);

	int counter = 0;
	for (const auto& image_name : names)
	{
		cout << double(counter++) / names.size() << "\n";

		string name_ext = image_name.substr(image_name.find_last_of("\\") + 1);
		string name = name_ext.substr(0, name_ext.find_last_of("."));

		Mat3b image = imread(image_name);
		Size sz = image.size();

		// Convert to grayscale
		Mat1b gray;
		cvtColor(image, gray, CV_BGR2GRAY);

		// Parameters Settings (Sect. 4.2)
		int		iThLength = 16;
		float	fThObb = 3.0f;
		float	fThPos = 1.0f;
		float	fTaoCenters = 0.05f;
		int 	iNs = 16;
		float	fMaxCenterDistance = sqrt(float(sz.width*sz.width + sz.height*sz.height)) * fTaoCenters;

		float	fThScoreScore = 0.72f;

		// Other constant parameters settings. 

		// Gaussian filter parameters, in pre-processing
		Size	szPreProcessingGaussKernelSize = Size(5, 5);
		double	dPreProcessingGaussSigma = 1.0;

		float	fDistanceToEllipseContour = 0.1f;	// (Sect. 3.3.1 - Validation)
		float	fMinReliability = 0.4;	// Const parameters to discard bad ellipses


		// Initialize Detector with selected parameters
		CEllipseDetectorYaed yaed;
		yaed.SetParameters(szPreProcessingGaussKernelSize,
			dPreProcessingGaussSigma,
			fThPos,
			fMaxCenterDistance,
			iThLength,
			fThObb,
			fDistanceToEllipseContour,
			fThScoreScore,
			fMinReliability,
			iNs
			);


		// Detect
		vector<Ellipse> ellsYaed;
		Mat1b gray_clone = gray.clone();
		yaed.Detect(gray_clone, ellsYaed);

		/*vector<double> times = yaed.GetTimes();
		cout << "--------------------------------" << endl;
		cout << "Execution Time: " << endl;
		cout << "Edge Detection: \t" << times[0] << endl;
		cout << "Pre processing: \t" << times[1] << endl;
		cout << "Grouping:       \t" << times[2] << endl;
		cout << "Estimation:     \t" << times[3] << endl;
		cout << "Validation:     \t" << times[4] << endl;
		cout << "Clustering:     \t" << times[5] << endl;
		cout << "--------------------------------" << endl;
		cout << "Total:	         \t" << yaed.GetExecTime() << endl;
		cout << "--------------------------------" << endl;*/

		tms.push_back(yaed.GetExecTime());


		vector<Ellipse> gt;
		LoadGT(gt, sWorkingDir + "gt\\" + "gt_" + name_ext + ".txt", false); // Prasad is in radians,set to true


		float pr, re, fm;
		std::tie(pr, re, fm) = Evaluate(gt, ellsYaed, fThScoreScore, image);

		prs.push_back(pr);
		res.push_back(re);
		fms.push_back(fm);

		Mat3b resultImage = image.clone();

		// Draw GT ellipses
		for (unsigned i = 0; i < gt.size(); ++i)
		{
			Ellipse& e = gt[i];
			Scalar color(0, 0, 255);
			ellipse(resultImage, Point(cvRound(e._xc), cvRound(e._yc)), Size(cvRound(e._a), cvRound(e._b)), e._rad*180.0 / CV_PI, 0.0, 360.0, color, 3);
		}

		yaed.DrawDetectedEllipses(resultImage, ellsYaed);

		//imwrite(out_folder + name + ".png", resultImage);
		//imshow("Yaed", resultImage);
		//waitKey();


		int dbg = 0;
	}

	float N = float(prs.size());
	float sumPR = accumulate(prs.begin(), prs.end(), 0.f);
	float sumRE = accumulate(res.begin(), res.end(), 0.f);
	float sumFM = accumulate(fms.begin(), fms.end(), 0.f);
	double sumTM = accumulate(tms.begin(), tms.end(), 0.0);

	float meanPR = sumPR / N;
	float meanRE = sumRE / N;
	float meanFM = sumFM / N;
	double meanTM = sumTM / N;

	float finalFM = (2.f * meanPR * meanRE) / (meanPR + meanRE);

	cout << "F-measure : " << finalFM << endl;
	cout << "Exec time : " << meanTM << endl;

	getchar();
}


int main2(int argc, char** argv)
{
	//OnVideo();
	//OnImage();
	OnDataset();

	return 0;
}





double iou(const Rect& r1, const Rect& r2)
{
	int x1 = std::max(r1.x, r2.x);
	int y1 = std::max(r1.y, r2.y);
	int x2 = std::min(r1.x + r1.width, r2.x + r2.width);
	int y2 = std::min(r1.y + r1.height, r2.y + r2.height);
	int w = std::max(0, (x2 - x1 + 1));
	int h = std::max(0, (y2 - y1 + 1));
	double inter = w * h;
	//double o = inter / (r1.area() + r2.area() - inter);
	// 如果后面的那个矩形的大部分面积都属于前面那个矩形的，那么后面的矩形可以删了
	double o = inter / r2.area();
	return (o >= 0) ? o : 0;
}

void nms(vector<Rect>& proposals, const double nms_threshold)
{
	// 这里的分数用的是面积
	vector<int> scores;
	for (auto i : proposals) scores.push_back(i.area());

	vector<int> index;
	for (int i = 0; i < scores.size(); ++i) {
		index.push_back(i);
	}

	sort(index.begin(), index.end(), [&](int a, int b) {
		return scores[a] > scores[b];
	});

	vector<bool> del(scores.size(), false);
	for (size_t i = 0; i < index.size(); i++) {
		if (!del[index[i]]) {
			for (size_t j = i + 1; j < index.size(); j++) {
				if (iou(proposals[index[i]], proposals[index[j]]) > nms_threshold) {
					del[index[j]] = true;
				}
			}
		}
	}

	vector<Rect> new_proposals;
	for (const auto i : index) {
		if (!del[i]) new_proposals.push_back(proposals[i]);
	}
	proposals = new_proposals;
}





// Mser目标检测 + nms
std::vector<cv::Rect> mser(cv::Mat srcImage)
{


	std::vector<std::vector<cv::Point> > regContours;
	
	// 创建MSER对象
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 10, 5000, 0.5, 0.3);


	std::vector<cv::Rect> boxes;
	// MSER检测
	mesr1->detectRegions(srcImage, regContours, boxes);
	// 存储矩形
	std::vector<Rect> keeps;


	cv::Mat mserMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);
	cv::Mat mserNegMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);

	for (int i = (int)regContours.size() - 1; i >= 0; i--)
	{
		// 根据检测区域点生成mser结果
		std::vector<Point> hull;
		convexHull(regContours[i], hull);
		Rect br = boundingRect(hull);
		// 宽高比例
		double wh_ratio = br.width / double(br.height);
		// 面积
		int b_size = br.width * br.height;
		// 不符合尺寸条件判断
		if (b_size < 600 && b_size > 150)
		keeps.push_back(br);
	}
	// 用nms抑制
	nms(keeps, 0.5);
	return  keeps;
}



// 计算点到直线的距离
float point2Line(Vec2f& point, Vec4f& line) 
{
	float x1 = line[0];
	float y1 = -line[1];
	float x2 = line[2];
	float y2 = -line[3];

	float A = (y1 - y2) / (x1 - x2);
	float B = -1;
	float C = y1 - x1 * A;
	return abs(A * point[0] - B * point[1] + C) / sqrt(pow(A, 2) + pow(B, 2));
}


// 计算点到点的距离
float point2point(float x1, float y1, float x2, float y2)
{
	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}
// 计算点到点的距离
float point2point(int x1, int y1, int x2, int y2)
{
	return sqrt(float(pow((x1 - x2), 2) + pow((y1 - y2), 2)));
}

// 计算点到点的距离
float point2point(Point point1, Point point2)
{
	return sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2));
}

// 计算点到点的距离
float point2point(Vec4f line)
{
	return sqrt(pow((line[0] - line[2]), 2) + pow((line[1] - line[3]), 2));
}

// 计算两直线间的夹角cos值
float line2lineAngleCos(Vec4f line1, Vec4f line2) 
{
	float leng1 = point2point(line1);
	float leng2 = point2point(line2);
	return ((line1[2] - line1[0]) * (line2[2] - line2[0]) + (line1[3] - line1[1]) * (line2[3] - line2[1])) / leng1 / leng2;
}


// 合并直线的操作

// 1. 右搜索
void backSearch(vector<bool>& isVisited, vector<int>& backs, vector<int>& goal_line)
{
	int right = goal_line[goal_line.size() - 1];
	isVisited[right] = true;
	if (right != backs[right])
	{
		goal_line.push_back(backs[right]);
		backSearch(isVisited, backs, goal_line);
	}
}


// 2. 左搜索
void frontSearch(vector<bool>& isVisited, vector<int>& fronts, vector<int>& goal_line)
{
	int left = goal_line[0];
	if (fronts[left] >= 0)
	{
		isVisited[fronts[left]] = true;
		goal_line.insert(goal_line.begin(), fronts[left]);
		frontSearch(isVisited, fronts, goal_line);
	}
}


// Test on single image
int main3()
{
	string images_folder = "D:\\VcProject\\biaopan\\imgs\\";
	string out_folder = "D:\\VcProject\\biaopan\\imgs\\";
	vector<string> names;

	//glob(images_folder + "Lo3my4.*", names);
	names.push_back("D:\\VcProject\\biaopan\\imgs\\003.jpg");
	int scaleSize = 8;
	for (const auto& image_name : names)
	{
		//string name = image_name.substr(image_name.find_last_of("\\") + 1);
		//name = name.substr(0, name.find_last_of("."));

		Mat3b image_1 = imread(image_name);
		
		Mat3b image;
		resize(image_1, image, Size(image_1.size[1] / scaleSize, image_1.size[0] / scaleSize));
		Size sz = image.size();
		
		// Convert to grayscale
		Mat1b gray;
		cvtColor(image, gray, CV_BGR2GRAY);
		
		// Parameters Settings (Sect. 4.2)
		int		iThLength = 16;
		float	fThObb = 3.0f;
		float	fThPos = 1.0f;
		float	fTaoCenters = 0.05f;
		int 	iNs = 16;
		float	fMaxCenterDistance = sqrt(float(sz.width*sz.width + sz.height*sz.height)) * fTaoCenters;

		float	fThScoreScore = 0.7f;

		// Other constant parameters settings. 

		// Gaussian filter parameters, in pre-processing
		Size	szPreProcessingGaussKernelSize = Size(5, 5);
		double	dPreProcessingGaussSigma = 1.0;

		float	fDistanceToEllipseContour = 0.1f;	// (Sect. 3.3.1 - Validation)
		float	fMinReliability = 0.5;	// Const parameters to discard bad ellipses


		// Initialize Detector with selected parameters
		CEllipseDetectorYaed yaed;
		yaed.SetParameters(szPreProcessingGaussKernelSize,
			dPreProcessingGaussSigma,
			fThPos,
			fMaxCenterDistance,
			iThLength,
			fThObb,
			fDistanceToEllipseContour,
			fThScoreScore,
			fMinReliability,
			iNs
			);


		// Detect
		vector<Ellipse> ellsYaed;
		Mat1b gray_clone = gray.clone();
		yaed.Detect(gray_clone, ellsYaed);
		



		vector<double> times = yaed.GetTimes();
		cout << "--------------------------------" << endl;
		cout << "Execution Time: " << endl;
		cout << "Edge Detection: \t" << times[0] << endl;
		cout << "Pre processing: \t" << times[1] << endl;
		cout << "Grouping:       \t" << times[2] << endl;
		cout << "Estimation:     \t" << times[3] << endl;
		cout << "Validation:     \t" << times[4] << endl;
		cout << "Clustering:     \t" << times[5] << endl;
		cout << "--------------------------------" << endl;
		cout << "Total:	         \t" << yaed.GetExecTime() << endl;
		cout << "--------------------------------" << endl;


		
		Mat3b resultImage = image.clone();
		yaed.DrawDetectedEllipses(resultImage, ellsYaed);
		cout << "detect ells number : " << ellsYaed.size() << endl;

		// 开展搜索范围，以长轴为直径的正方形区域
		int index = 0;
		Mat1b gray_clone2;
		cvtColor(image_1, gray_clone2, CV_BGR2GRAY);
		namedWindow("roi");
		int el_size = ellsYaed.size();

		// 选取至少有 35 个可能支持点的椭圆
		int min_vec_num = 35;
		// 存储目标椭圆
		Ellipse& el_dst = ellsYaed[0];
		// 存储目标区域
		Mat1b roi_zero = Mat::zeros(400, 400, CV_8UC1);
		Mat1b& roi_dst = roi_zero;
		// 存储目标的可能支持线
		vector<Vec4f> tLines;

		while(index < el_size ) {
			Ellipse& e = ellsYaed[index];
			int g = cvRound(e._score * 255.f);
			Scalar color(0, g, 0);
			// 找到长轴
			int long_a = e._a >= e._b ? cvRound(e._a) : cvRound(e._b);
			rectangle(resultImage, Rect(cvRound(e._xc)-long_a, cvRound(e._yc)-long_a, 2 * long_a, 2 * long_a), color, 1);

			// 找到在原图中的位置，然后先进行直方图均衡化最后再lsd
			// 这里由于精度问题，会出现 e._xc 略小于 long_a，或者矩形的范围超出了图片原来尺寸
			int r_x = max(0, (cvRound(e._xc) - long_a)) * scaleSize;
			int r_y = max(0, (cvRound(e._yc) - long_a)) * scaleSize;
			// 超出尺寸的话就适当缩小
			int r_mx = min(gray_clone2.cols, r_x + 2 * long_a * scaleSize);
			int r_my = min(gray_clone2.rows, r_y + 2 * long_a * scaleSize);
			int n_width = min(r_mx - r_x, r_my - r_y);
			Mat1b roi = gray_clone2(Rect(r_x, r_y, n_width, n_width));
			Mat1b roi_2;
			resize(roi, roi_2, Size(400, cvRound(float(roi.cols) / float(roi.rows) * 400)));
			// 同时也放缩一下椭圆
			float scaleFactor = float(400) / 2 / long_a;
			// 上面求区域的时候实际上就规定了最后在 400 x 400 的图里面椭圆中心就是(200, 200)
			e._xc = 200;
			e._yc = 200;
			e._a = e._a * scaleFactor;
			e._b = e._b * scaleFactor;

			Mat1b roi_3 = roi_2.clone();
			equalizeHist(roi_2, roi_2);


			// 运行LSD算法检测直线
			Canny(roi_2, roi_2, 50, 150, 3); // Apply canny edge//可选canny算子

			Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
			vector<Vec4f> lines_std;
			vector<Vec4f> lines_dst;

			// Detect the lines
			ls->detect(roi_2, lines_std);
			// Show found lines
			Mat drawnLines = roi_3.clone();
			

			Vec2f e_center = Vec2f(200, 200);
			cout << "lines_std.size() : " << lines_std.size() << endl;

			for (int j=0;j<lines_std.size();j++) 
			{
				// 筛选掉那些距离中心比较远的线
				float distance = point2Line(e_center, lines_std[j]);
				if (distance <= 10)
				{
					Vec4f l = lines_std[j];
					// 还要分头尾两点
					Vec4f dl;
					if (point2point(l[0], l[1], e_center[0], e_center[1]) >= point2point(l[2], l[3], e_center[0], e_center[1]))
					{
						dl = Vec4f(l[2], l[3], l[0], l[1]);
					}
					else 
					{
						dl = l;
					}
					lines_dst.push_back(dl);
					// 画出尾点
					circle(drawnLines, Point(dl[2], dl[3]), 2, color, -1);
				}
			}
			circle(drawnLines, Point(200, 200), 4, color, -1);

			// 画出所有朝向中点的线
			ls->drawSegments(drawnLines, lines_dst);
			cout << "size: " << lines_dst.size() << endl;

			
			imshow("roi", drawnLines);
			imshow("Yaed", resultImage);
			cout << "index : " << index << endl;
			waitKey();
			
			index += 1;


			// 从里面选出 可能支持点 超过 35的，并且面积是最小的
			cout << "目前的面积是： " << (el_dst._a * el_dst._b) << "   现在的面积是: " << (e._a * e._b) << endl;
			if (lines_dst.size() >= min_vec_num && (el_dst._a * el_dst._b) >= (e._a * e._b))
			{
				el_dst = e;
				roi_dst = roi_3;
				tLines = lines_dst;
			}
		}

		//imwrite(out_folder + name + ".png", resultImage);



		// 显示椭圆效果
		int g = cvRound(el_dst._score * 255.f);
		Scalar color(0, 255, 255);
		//ellipse(roi_dst, Point(cvRound(el_dst._xc), cvRound(el_dst._yc)), Size(cvRound(el_dst._a), cvRound(el_dst._b)), el_dst._rad*180.0 / CV_PI, 0.0, 360.0, color, 2);
		namedWindow("see", WINDOW_AUTOSIZE);
		

		// mser检测
		std::vector<cv::Rect> candidates;
		candidates = mser(roi_dst);
		// 区域显示
		for (int i = 0; i < candidates.size(); ++i)
		{
			//rectangle(roi_dst, candidates[i], color, 1);
		}

		//ellipse(roi_dst, Point(cvRound(el_dst._xc), cvRound(el_dst._yc)), Size(cvRound(el_dst._a), cvRound(el_dst._b)), el_dst._rad*180.0 / CV_PI, 0.0, 360.0, color, 2);


		imshow("see", roi_dst);
		imshow("Yaed", resultImage);





		// 1. 缩圈方法以及快速排出都结合起来，选取最优的椭圆（慢慢放缩得到最多的支持点时的个数最多，有个获取极值的过程，再放缩下去支持点就变少）

		// 2. 上面方法不行就用回以前的 bim+支持点方向分类 然后抽样获取椭圆

		// 3. hough圆检测获取精确中心，lsd获取最长直线（指针线）

		/**
			中心是(200, 200)，在这个中心以一定半径搜索精准的圆心
		**/
		int searchRadius = 25;
		vector<Vec3f> circles;
		Mat1b centerArea = roi_dst(Rect(200 - searchRadius, 200 - searchRadius, 2 * searchRadius, 2 * searchRadius));
		GaussianBlur(centerArea, centerArea, Size(3, 3), 1, 1);
		// 后面四个数字分别代表： 分辨率（可以理解为步进的倒数），两个圆之间最小距离，canny的低阈值，投票的累积次数（即有多少个点属于该圆），圆最小半径，圆最大半径
		HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 50, 10, 0, 6);
		// 寻找离形心最近的那个圆心作为我们的准心，绘制出圆
		Point ac_center = Point(200, 200);
		Point f_center = Point(searchRadius/2, searchRadius/2);
		float f2ac = 200;
		for (int i=circles.size()-1;i>=0;--i) 
		{
			Point mCenter = Point(circles[i][0], circles[i][1]);
			float mDistance = point2point(mCenter, f_center);
			if (mDistance < f2ac)
			{
				f2ac = mDistance;
				ac_center = Point(mCenter.x + 200 - searchRadius, mCenter.y + 200 - searchRadius);
			}
			circle(centerArea, mCenter, circles[i][2], color);
		}
		namedWindow("center", WINDOW_AUTOSIZE);
		imshow("center", centerArea);

		circle(roi_dst, ac_center, 3, color, -1);
		imshow("see", roi_dst);



		// 下面是获取指针线
		// 先是找出所有线的最近后接线
		float mRadius = 80;
		float angle = 10;
		float angelCos = cos(angle *  pi / 180);
		
		// 存储这些后线对
		vector<int> backs = vector<int>(tLines.size());
		// 存储这些前线对，方便之后的计算
		vector<int> fronts = vector<int>(tLines.size());

		// 初始化这些对的值，默认为-1
		for (int k=tLines.size()-1;k>=0;--k)
		{
			backs[k] = -1;
			fronts[k] = -1;
		}

		for (int i=tLines.size() - 1;i >= 0; --i) 
		{
			Vec4f& line1 = tLines[i];
			// 搜索半径内，首先是属于自己后面的线，然后是中点相连所成的角度不能大过某个值，并且取距离自己最短的那些线，最后是看这条线是不是已经成为了别人的后线了
			int mIndex = i;
			float mDis = 1000;
			for (int j = tLines.size() - 1; j >= 0; --j) 
			{
				if (i == j)
					continue;
				Vec4f& line2 = tLines[j];

				// 先判断是不是自己后面的线
				if (point2point(Point(line1[0], line1[1]), ac_center) > point2point(Point(line2[0], line2[1]), ac_center))
					continue;

				float dis = point2point(line1[2], line1[3], line2[0], line2[1]);
				
				if (dis <= min(mRadius, mDis))
				{
					Vec4f mLine = Vec4f((line1[0] + line1[2])/2, (line1[1] + line1[3]) / 2, (line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2);
					if (line2lineAngleCos(line1, mLine) >= angelCos && line2lineAngleCos(line2, mLine) >= angelCos)
					{
						mDis = dis;
						mIndex = j;
					}
				}

			}


			// 如果已经这个线已经是别人的后线了，那么要比较一下两个前线的长短，长的才有资格拿这个线
			if (fronts[mIndex] >= 0)
			{
				if (point2point(tLines[fronts[mIndex]]) < point2point(tLines[i])) {

					// 拿不到的人就back是自己
					backs[fronts[mIndex]] = fronts[mIndex];

					backs[i] = mIndex;
					if (i != mIndex)
						fronts[mIndex] = i;
				}
				else 
				{
					// 拿不到那你就没back了
					backs[i] = i;
				}
			}
			// 正常情况就直接确定前后关系即可
			else 
			{
				backs[i] = mIndex;
				if (i != mIndex)
					fronts[mIndex] = i;
			}
		}

		// 下面只是检验上面是否出错
		for (int i = tLines.size() - 1; i >= 0; --i)
		{
			cout << "backs-" << i << " : " << backs[i] << endl;
			cout << "fronts-" << i << " : " << fronts[i] << endl;
			int n = 0;
			for (int j = tLines.size() - 1; j >= 0; --j) {
				if (backs[j] == i)
					n++;
			}
			if (n > 2)
				cout << "线-" << i << "被超过1条线当做后线" << endl;
			// 正确的形式应该是，backs里面存储的是当前直线的后线，如果后线等于自己，说明自己没后线
			// 而fronts存储的是当前直线的前线，如果前线等于-1，说明自己没有前线
		}

		// 组合这些对并且得出最长的线
		// 创建一个记录节点是否被访问的数组
		vector<bool> isVisited = vector<bool>(backs.size());
		vector<vector<int>> goal_lines;
		for (int i = backs.size() - 1; i >= 0; --i)
		{
			if (!isVisited[i]) 
			{
				vector<int> goal_line;
				goal_line.push_back(i);
				if (i != backs[i])
					goal_line.push_back(backs[i]);

				backSearch(isVisited, backs, goal_line);
				frontSearch(isVisited, fronts, goal_line);
				goal_lines.push_back(goal_line);
			}
		}

		// 显示直线组合
		for (int i= goal_lines.size()-1;i>=0;--i)
		{
			vector<int>& a = goal_lines[i];
			cout << "[";
			for (int j = 0; j <= a.size() - 1; ++j)
			{
				cout << a[j] << ",";
			}
			cout << "]" << endl;
		}

		// 画出直线组合
		Scalar cc(0, 255, 255);
		for (int i = goal_lines.size() - 1; i >= 0; --i)
		{
			vector<int>& goal_line = goal_lines[i];
			for (int j = 0; j <= goal_line.size() - 1; ++j)
			{
				int li = goal_line[j];
				Vec4f ln = tLines[li];
				Point point1 = Point(ln[0], ln[1]);
				Point point2 = Point(ln[2], ln[3]);
				cv::line(roi_dst, point1, point2, cc, 2);
			}

			imshow("see", roi_dst);
			waitKey();
		}
		

		float maxLength = 0;
		vector<int>& maxLine = goal_lines[0];
		for (int i = goal_lines.size() - 1; i >= 1; --i)
		{
			vector<int>& goal_line = goal_lines[i];
			float total_length = 0;
			for (int j = 0; j <= goal_line.size() - 1; ++j)
			{
				int li = goal_line[j];
				total_length += point2point(tLines[li]);
			}
			if (total_length > maxLength) { maxLength = total_length; maxLine = goal_line; }
		}
		// 画出最长的线
		Scalar aa(255, 255, 255);
		for (int j = 0; j <= maxLine.size() - 1; ++j)
		{
			int li = maxLine[j];
			Vec4f ln = tLines[li];
			Point point1 = Point(ln[0], ln[1]);
			Point point2 = Point(ln[2], ln[3]);
			cv::line(roi_dst, point1, point2, aa, 2);
		}
		imshow("see", roi_dst);
		waitKey();

		// 拿最长的线的结尾当做指针线的尾点
		Vec4f last_part = tLines[maxLine[maxLine.size() - 1]];
		Point lastPoint = Point(last_part[2], last_part[3]);



		// 4. svm获取所有的数字区域（这里数据需要增强，比如镜像，颠倒），然后他们中心拟合出椭圆
		
		waitKey();	
	}

	int yghds = 0;
	return 0;
}





string int2str(const int &int_temp)
{
	stringstream stream;
	stream << int_temp;
	return stream.str();   //此处也可以用 stream>>string_temp  
}

vector<string> readTxt(string file)
{
	ifstream infile;
	infile.open(file.data());   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 

	vector<string> names;
	string s;
	while (getline(infile, s))
	{
		names.push_back(s);
	}
	infile.close();             //关闭文件输入流 
	return names;
}

// 分割字符串
vector<string> splitString(const std::string& s, const std::string& c)
{
	std::vector<std::string> v;
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
	return v;
}

int test() 
{
	string a = int2str(1);
	cout << "a: " << a << endl;

	string outputPath = "D:\\VcProject\\biaopan\\data\\goodImgs\\";
	string basePath = "D:\\VcProject\\biaopan\\data\\raw\\images\\" + int2str(1) + "\\";
	vector<string> goods = readTxt("D:\\VcProject\\biaopan\\data\\raw\\images\\" + int2str(1) + "\\goods.txt");
	int totalIndex = 0;
	for (int i=0;i<=goods.size()-1;i++) 
	{
		totalIndex += 1;
		string& s = goods[i];
		Mat3b image = imread(basePath + splitString(s, "\\")[4]);
		Mat1b gray;
		cvtColor(image, gray, CV_BGR2GRAY);
		imwrite(outputPath + int2str(totalIndex) + "\\" + "1.jpg", gray);
	}

	waitKey();
	return 0;
}




void writeImg(string imgReadPath, string dirPath)
{

	Mat3b image_1 = imread(imgReadPath);

	int scaleSize = 8;

	Mat3b image;
	resize(image_1, image, Size(image_1.size[1] / scaleSize, image_1.size[0] / scaleSize));
	Size sz = image.size();



	// Parameters Settings (Sect. 4.2)
	int		iThLength = 16;
	float	fThObb = 3.0f;
	float	fThPos = 1.0f;
	float	fTaoCenters = 0.05f;
	int 	iNs = 16;
	float	fMaxCenterDistance = sqrt(float(sz.width*sz.width + sz.height*sz.height)) * fTaoCenters;

	float	fThScoreScore = 0.7f;

	// Other constant parameters settings. 

	// Gaussian filter parameters, in pre-processing
	Size	szPreProcessingGaussKernelSize = Size(5, 5);
	double	dPreProcessingGaussSigma = 1.0;

	float	fDistanceToEllipseContour = 0.1f;	// (Sect. 3.3.1 - Validation)
	float	fMinReliability = 0.5;	// Const parameters to discard bad ellipses


	// Initialize Detector with selected parameters
	CEllipseDetectorYaed yaed;
	yaed.SetParameters(szPreProcessingGaussKernelSize,
		dPreProcessingGaussSigma,
		fThPos,
		fMaxCenterDistance,
		iThLength,
		fThObb,
		fDistanceToEllipseContour,
		fThScoreScore,
		fMinReliability,
		iNs
	);



	// Convert to grayscale
	Mat1b gray;
	cvtColor(image, gray, CV_BGR2GRAY);



	// Detect
	vector<Ellipse> ellsYaed;
	Mat1b gray_clone = gray.clone();
	yaed.Detect(gray_clone, ellsYaed);



	Mat3b resultImage = image.clone();
	yaed.DrawDetectedEllipses(resultImage, ellsYaed);


	// 开展搜索范围，以长轴为直径的正方形区域
	int index = 0;
	Mat1b gray_clone2;
	cvtColor(image_1, gray_clone2, CV_BGR2GRAY);
	int el_size = ellsYaed.size();

	// 选取至少有 35 个可能支持点的椭圆
	int min_vec_num = 35;
	// 存储目标椭圆
	Ellipse& el_dst = ellsYaed[0];
	// 存储目标区域
	Mat1b roi_zero = Mat::zeros(400, 400, CV_8UC1);
	Mat1b& roi_dst = roi_zero;
	// 存储目标的可能支持线
	vector<Vec4f> tLines;

	while (index < el_size) {
		Ellipse& e = ellsYaed[index];
		int g = cvRound(e._score * 255.f);
		Scalar color(0, g, 0);
		// 找到长轴
		int long_a = e._a >= e._b ? cvRound(e._a) : cvRound(e._b);
		rectangle(resultImage, Rect(cvRound(e._xc) - long_a, cvRound(e._yc) - long_a, 2 * long_a, 2 * long_a), color, 1);

		// 找到在原图中的位置，然后先进行直方图均衡化最后再lsd
		// 这里由于精度问题，会出现 e._xc 略小于 long_a，或者矩形的范围超出了图片原来尺寸
		int r_x = max(0, (cvRound(e._xc) - long_a)) * scaleSize;
		int r_y = max(0, (cvRound(e._yc) - long_a)) * scaleSize;
		// 超出尺寸的话就适当缩小
		int r_mx = min(gray_clone2.cols, r_x + 2 * long_a * scaleSize);
		int r_my = min(gray_clone2.rows, r_y + 2 * long_a * scaleSize);
		int n_width = min(r_mx - r_x, r_my - r_y);
		Mat1b roi = gray_clone2(Rect(r_x, r_y, n_width, n_width));
		Mat1b roi_2;
		resize(roi, roi_2, Size(400, cvRound(float(roi.cols) / float(roi.rows) * 400)));
		// 同时也放缩一下椭圆
		float scaleFactor = float(400) / 2 / long_a;
		// 上面求区域的时候实际上就规定了最后在 400 x 400 的图里面椭圆中心就是(200, 200)
		e._xc = 200;
		e._yc = 200;
		e._a = e._a * scaleFactor;
		e._b = e._b * scaleFactor;

		Mat1b roi_3 = roi_2.clone();
		equalizeHist(roi_2, roi_2);


		// 运行LSD算法检测直线
		Canny(roi_2, roi_2, 50, 150, 3); // Apply canny edge//可选canny算子

		Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
		vector<Vec4f> lines_std;
		vector<Vec4f> lines_dst;

		// Detect the lines
		ls->detect(roi_2, lines_std);


		Vec2f e_center = Vec2f(200, 200);

		for (int j = 0; j < lines_std.size(); j++)
		{
			// 筛选掉那些距离中心比较远的线
			float distance = point2Line(e_center, lines_std[j]);
			if (distance <= 10)
			{
				Vec4f l = lines_std[j];
				// 还要分头尾两点
				Vec4f dl;
				if (point2point(l[0], l[1], e_center[0], e_center[1]) >= point2point(l[2], l[3], e_center[0], e_center[1]))
				{
					dl = Vec4f(l[2], l[3], l[0], l[1]);
				}
				else
				{
					dl = l;
				}
				lines_dst.push_back(dl);
			}
		}

		index += 1;


		// 从里面选出 可能支持点 超过 35的，并且面积是最小的
		if (lines_dst.size() >= min_vec_num && (el_dst._a * el_dst._b) >= (e._a * e._b))
		{
			el_dst = e;
			roi_dst = roi_3;
			tLines = lines_dst;
		}
	}


	// mser检测
	std::vector<cv::Rect> candidates;

	candidates = mser(roi_dst);
	// 区域显示
	int file_index = 0;
	for (int i = 0; i < candidates.size(); ++i)
	{
		file_index += 1;
		if (candidates[i].x <= 0 || candidates[i].y <= 0 || candidates[i].width <= 0 || candidates[i].height <= 0)
			continue;
		Mat1b mser_item = roi_dst(candidates[i]);
		
		_mkdir(dirPath.c_str());
		// 写最终结果
		imwrite(dirPath + "\\" + int2str(file_index) + ".jpg", mser_item);
	}
}



int main4()
{
	vector<string> aa;

	aa.push_back("024");
	aa.push_back("034");
	aa.push_back("035");
	for (int i=0;i<=aa.size()-1;i++) 
	{
		cout << "当前准备的是： " << aa[i] << endl;
		writeImg("D:\\VcProject\\biaopan\\data\\raw\\images\\3\\"+aa[i]+".jpg", "D:\\VcProject\\biaopan\\data\\test\\1");
	}
	return 0;
}



// 这里仅仅是用作制作数据
int main()
{


	//glob(images_folder + "Lo3my4.*", names);

	string outputPath = "D:\\VcProject\\biaopan\\data\\goodImgs\\";
	string basePath = "D:\\VcProject\\biaopan\\data\\raw\\images\\";
	vector<string> names;

	
	int folderIndex = 54;
	for (int b_i=3;b_i<=13;b_i++) 
	{
		vector<string> goods = readTxt(basePath + int2str(b_i) + "\\goods.txt");
		for (int b_j = 0; b_j <= goods.size() - 1; b_j++)
		{
			// 一张图片一个folder
			folderIndex += 1;
			string& s = goods[b_j];
			string imgReadPath = basePath + int2str(b_i) + "\\" + splitString(s, "\\")[4];
			string dirPath = outputPath + int2str(folderIndex);
			writeImg(imgReadPath, dirPath);
		}
	}

	int yghds = 0;
	return 0;
}