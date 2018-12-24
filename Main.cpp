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
#include <unordered_map>
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/ml/ml.hpp>


#include<time.h>
#define random(x) (rand()%x)+1


#define DATA_DIR "D:\\OpenCV\\bin\\toy_data\\"
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


// 计算点到直线的距离，这里输入的点是图像坐标的点
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


// 计算点到直线的距离，这里的点是标准坐标的点
float point2Line(float x, float y, float x1, float y1, float x2, float y2)
{

	float A = (y1 - y2) / (x1 - x2);
	float B = -1;
	float C = y1 - x1 * A;
	float dis = abs(A * x + B * y + C) / sqrt(pow(A, 2) + pow(B, 2));
	return dis;
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

// 加个别的项进去
void nms2(vector<Rect>& proposals,vector<int>& indexes, const double nms_threshold)
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
	vector<int> new_indexes;
	for (const auto i : index) {
		if (!del[i])
		{
			new_proposals.push_back(proposals[i]);
			new_indexes.push_back(i);
		}

	}
	proposals = new_proposals;
	indexes = new_indexes;
}




// 画出旋转矩阵
void drawRotatedRect(Mat& drawer, RotatedRect& rrect)
{
	// 获取旋转矩形的四个顶点
	cv::Point2f* vertices = new cv::Point2f[4];
	rrect.points(vertices);
	//逐条边绘制
	for (int j = 0; j < 4; j++)
	{
		// 画出他的四个点的边角，黑色是p[3]，白色是p[1]
		circle(drawer, vertices[0], 2, Scalar(225, 225, 225), -1);
		circle(drawer, vertices[3], 2, Scalar(0, 0, 0), -1);
		cv::line(drawer, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 0, 0));
	}
}

// Mser目标检测 + nms，其中rrects存储每个Rect中的那个旋转矩阵
std::vector<cv::Rect> mser(cv::Mat srcImage, vector<cv::RotatedRect>& rrects)
{

	std::vector<std::vector<cv::Point> > regContours;
	
	// 创建MSER对象
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 5, 800, 0.5, 0.3);


	std::vector<cv::Rect> boxes;
	// MSER检测
	mesr1->detectRegions(srcImage, regContours, boxes);
	// 存储矩形
	std::vector<Rect> keeps;


	cv::Mat mserMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);
	cv::Mat mserNegMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);

	Mat mser_show = srcImage.clone();

	vector<RotatedRect> rrecs;
	for (int i = (int)regContours.size() - 1; i >= 0; i--)
	{

		// 根据检测区域点生成mser结果
		std::vector<Point> hull;
		// 计算凸包
		convexHull(regContours[i], hull);

		// 下面是为了检测角度而设的。
		// mser检测出来的是连通域的所有点
		// 画出凸包或者原来的点，画出来会发现这个结果是挺不错的，但是数字粘连的情况也出现了，所以需要计算角度然后通过投影来分离
		// polylines(mser_show, regContours[i], 1, Scalar(255, 0, 0));
		// 获取最小包围矩形
		RotatedRect rotatedRect = minAreaRect(regContours[i]);
		rrecs.push_back(rotatedRect);

		drawRotatedRect(mser_show, rotatedRect);



		Rect br = boundingRect(hull);
		// 宽高比例
		double wh_ratio = br.height / double(br.width);
		// 面积
		int b_size = br.width * br.height;
		// 不符合尺寸条件判断
		if (b_size < 800 && b_size > 50)
		{
			// 实验证明，往外扩张的时候识别效果更好，
			br = Rect(br.x - 3, br.y - 3, br.width + 6, br.height + 6);
			keeps.push_back(br);
		}
		// 稍微让方框往外扩一点
		
		//keeps.push_back(br);
	}

	imshow("mser_show", mser_show);
	waitKey(0);
	// 用nms抑制
	nms(keeps, 0.7);


	mser_show = srcImage.clone();
	// 找出每个keep中的矩形所包含的旋转矩阵，
	vector<cv::RotatedRect> rrects_;
	for (int j = 0; j < keeps.size(); j++)
	{
		float karea = keeps[j].width * keeps[j].height;
		rectangle(mser_show, keeps[j], Scalar(255, 255, 255), 2);
		RotatedRect krec;
		float max_size = 0.2;
		for (int i = 0; i < rrecs.size(); i++)
		{
			
			//获取旋转矩形的四个顶点
			cv::Point2f* vertices = new cv::Point2f[4];
			rrecs[i].points(vertices);
			
			// 计算两个矩形的重叠率
			//float area = point2point(vertices[0], vertices[1]) * point2point(vertices[0], vertices[3]);

			if (rrecs[i].center.x >= keeps[j].x && rrecs[i].center.y >= keeps[j].y 
				&& rrecs[i].center.x <= keeps[j].x+ keeps[j].width && rrecs[i].center.y <= keeps[j].y + keeps[j].height)
			{
				//float area = point2point(vertices[0], vertices[1]) * point2point(vertices[0], vertices[3]);
				float area = rrecs[i].size.height * rrecs[i].size.width;
				// 取面积最大且有一定阈值的作为判断方向的准则
				if (area / karea >= max_size && area / karea <= 0.8)
				{
					max_size = area / karea; 
					krec = rrecs[i];
				}
			}
		}
		// 即使没有新的矩形，也要放进去一个已经有的。
		rrects_.push_back(krec);
		// 如果max_size没有变化，说明没有满足要求的内接旋转矩阵
		if (max_size == 0.2) { continue; }
		drawRotatedRect(mser_show, krec);
		//imshow("mser_show", mser_show);
		//waitKey(0);
	}
	rrects = rrects_;

	imshow("mser_show", mser_show);
	waitKey(0);
	return  keeps;
}





string int2str(const int &int_temp)
{
	stringstream stream;
	stream << int_temp;
	return stream.str();   //此处也可以用 stream>>string_temp  
}

int str2int(const string &string_temp)
{
	stringstream ss;
	ss << string_temp;
	int i;
	ss >> i;
	return i;
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



// 获取一个cell的向量，dn是指有多少个方向
vector<float> getCellData(Mat& mag, Mat& angle, int r, int c, int cellSize, int dn)
{
	vector<float> cell(dn, 0);
	float tangle = 360 / (float)dn;

	for (int k = r; k < r + cellSize; k++)
	{
		// 每一行图像的指针
		const float* magData = mag.ptr<float>(k);
		const float* angleData = angle.ptr<float>(k);
		for (int i = c; i < c + cellSize; i++)
		{
			// floor 是向上取整
			// cout << angleData[i] << endl;		
			// cout << magData[i] << endl;
			cell[floor(angleData[i] / tangle)] += magData[i];
		}
	}
	return cell;
}


// 获取hog向量
vector<float> getHogData(Mat& originImg)
{
	Mat img;
	// 进行resize
	resize(originImg, img, Size(18, 36));
	// 这里可以考虑归一化，也就是第三个参数可以设置为1/255，使0~255映射成0到1
	img.convertTo(img, CV_32F, 1);
	Mat gx, gy;
	Sobel(img, gx, CV_32F, 1, 0, 1);
	Sobel(img, gy, CV_32F, 0, 1, 1);

	Mat mag, angle;
	cartToPolar(gx, gy, mag, angle, 1);


	// 对每个cell都进行直方图统计
	vector<vector<float>> cells;
	int cellSize = 9;
	int directionNum = 12;
	for (int i = 0; i < 4; i++)
	{
		cells.push_back(getCellData(mag, angle, i * 9, 0, cellSize, directionNum));
		cells.push_back(getCellData(mag, angle, 9, i * 9, cellSize, directionNum));
	}

	// 把3个block都整合成一个vector
	vector<float> hogData;
	// 每四个cell做一个block，最后串联起来
	// 第一层是控制第几个block
	for (int i = 0; i < 3; i++)
	{
		// 存储每个block的vector
		vector<float> v;
		float total = 0;
		// 第二层是控制block中的第几个cell
		for (int j = i * 2; j < i * 2 + 4; j++)
		{
			// 控制每个cell里面的每个float值
			for (int k = 0; k < cells[j].size(); k++)
			{
				// 计算一个block里面的L2模外还需要把四个vector整合在一起
				total += pow(cells[j][k], 2);
				v.push_back(cells[j][k]);
			}
		}
		// 根据之前算出来block里面的L2模，对之前push进去的进行归一化
		for (int e = 0; e < v.size(); e++)
		{
			hogData.push_back(v[e] / sqrt(total));
		}
	}
	return hogData;
}







// 计算坐标旋转后的点，这里输入的坐标是标准坐标系，输出的也是标准坐标系
Point rotate(float theta, float x, float y)
{
	return Point(cos(theta)*x - sin(theta)*y, sin(theta)*x + cos(theta)*y);
}

// 切换到椭圆坐标，这里输入的坐标是图像坐标系，输出的是标准坐标系（先平移后旋转）
Point origin2el(Point2f& center, float theta, Point& origin)
{
	float x = origin.x;
	float y = -origin.y;
	return rotate(theta, x-center.x, y+center.y);
}


// 切换到图像坐标，这里输入的坐标是标准坐标系，输出的是图像坐标系（先旋转后平移）
Point el2origin(Point2f& center, float theta, Point& el)
{
	Point origin = rotate(theta, el.x, el.y);
	float x = origin.x;
	float y = -origin.y;
	return Point(x + center.x, y + center.y);
}


// 仅仅是测试旋转坐标
int testRoute()
{
	Mat a(400, 400, CV_8UC1, Scalar(0, 0, 0));
	Point2f center = Point(100, 200);
	ellipse(a, center, Size(50, 100), 30, 0, 360, Scalar(225, 225, 225), 1, 8);
	
	
	for (int i = 0;i < 5; i++)
	{
		Point x = Point(120, 200);
		Point newx = origin2el(center, 30 / (float)180 * pi, x);
		float eq = pow(newx.x, 2) / pow(50, 2) + pow(newx.y, 2) / pow(100, 2);
		circle(a, x, 2, Scalar(225, 225, 225), -1);
		cout << eq << endl;
		imshow("a", a);
		waitKey();
	}
	

	return 0;
}



// 计算出一个向量与椭圆的, a是椭圆的第一个轴长，b是椭圆的第二个轴长，theta是椭圆的倾斜角，xx1 和 xx2代表与椭圆相交的直线，最终返回的是 与xx1xx2向量同方向的交点(p1是出发点)
Point anchor_on_el_line(float a, float b, float theta, Point2f& center, Point& xx1, Point& xx2)
{

	Point newx1 = origin2el(center, theta, xx1);
	Point newx2 = origin2el(center, theta, xx2);
	float k = (newx1.y - newx2.y) / (float)(newx1.x - newx2.x);
	float c = newx2.y - k * newx2.x;
	float A = pow(b, 2) + pow(a, 2) * pow(k, 2);
	float B = 2 * k * c * pow(a, 2);
	float C = pow(a, 2) * pow(c, 2) - pow(a, 2) * pow(b, 2);

	float delta = pow(B, 2) - 4 * A * C;
	// if (abs(delta) <= 0.01) { delta = 0; }
	Point origin;
	Point el;
	if (delta == 0)
	{
		float x = -B / 2 / A;		
		float y = k * x + c;
		el = Point(x, y);
		// 最后要转为图像坐标输出
		origin = el2origin(center, -theta, el);
	}
	if (delta > 0)
	{

		float x1 = (-B + sqrt(delta)) / 2 / A;
		float x2 = (-B - sqrt(delta)) / 2 / A;
		float y1 = (k * x1 + c);
		float y2 = (k * x2 + c);

		// p1p2向量，下面判断是否同方向
		Vec2f v0 = Vec2f(newx2.x - newx1.x, newx2.y - newx1.y);
		Vec2f v1 = Vec2f(x1 - newx1.x, y1 - newx1.y);
		Vec2f v2 = Vec2f(x2 - newx1.x, y2 - newx1.y);
		float d1 = abs((v0[0] * v1[0] + v0[1] * v1[1]) / sqrt(pow(v0[0], 2) + pow(v0[1], 2)) / sqrt(pow(v1[0], 2) + pow(v1[1], 2)) - 1);
		float d2 = abs((v0[0] * v2[0] + v0[1] * v2[1]) / sqrt(pow(v0[0], 2) + pow(v0[1], 2)) / sqrt(pow(v2[0], 2) + pow(v2[1], 2)) - 1);
		if (d1 <= d2)
		{
			el = Point(x1, y1);
			// 最后要转为图像坐标输出
			origin = el2origin(center, -theta, el);
		}
		else
		{
			el = Point(x2, y2);
			// 最后要转为图像坐标输出
			origin = el2origin(center, -theta, el);
		}


	}
	return origin;
}




// 测试椭圆交线
int test_anchor_on_el_line()
{
	Mat a(400, 400, CV_8UC3, Scalar(0, 0, 0));
	Point x = Point(200, 300);
	Point2f center = Point2f(100, 200);
	Point center_ = Point(100, 200);
	ellipse(a, center, Size(50, 100), 75, 0, 360, Scalar(225, 225, 225), 1, 8);

	Point xxxx = anchor_on_el_line(50, 100, 75 / (float)180 * pi, center, center_, x);
	circle(a, center, 2, Scalar(0, 225, 225), -1);
	circle(a, x, 2, Scalar(0, 225, 225), -1);
	circle(a, xxxx, 2, Scalar(255, 0, 0), -1);
	imshow("a", a);
	waitKey(0);
	return 0;
}


// 试试光用mser和nms能否提取出所有的刻度线
int main__()
{
	string image_name = "D:\\VcProject\\biaopan\\data\\test\\2\\aa2.jpg";
	Mat image_1 = imread(image_name);

	// mser检测
	std::vector<cv::Rect> candidates;
	vector<RotatedRect> rrects;
	candidates = mser(image_1, rrects);
	for (int i = 0; i < candidates.size(); ++i)
	{
		rectangle(image_1, candidates[i], Scalar(255, 255, 255), 1);	
	}
	imshow("test", image_1);
	waitKey(0);
	return 0;
}


/* ------------------------------------ */
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

/* ------------------------------------ */



/* ------------------------------------ */
// 合并数字区域的操作

// 1. 连接点搜索
void joinSearch(vector<bool>& isVisited, vector<int>& goal_set, vector<vector<int>>& joinTable)
{
	vector<int>::iterator it;
	int i = goal_set[goal_set.size()-1];
	isVisited[i] = true;
	// -2表示对角线（自己连自己），-1表示没有连接，大于等于0的值表示这条连接的直线的倾斜角
	for (int j = 0; j < joinTable.size(); j++)
	{
		if (i == j) { continue; }
		if (joinTable[i][j] > -1) 
		{
			it = find(goal_set.begin(), goal_set.end(), j);
			// 如果goal_set里面原本就有这个点的就不用管这个点了，继续寻找其他需要的点
			if (it != goal_set.end()) { continue; }
			// 否则把这个点添加进来，同时让他去找下一个点
			goal_set.push_back(j);
			joinSearch(isVisited, goal_set, joinTable);
		}
	}
	

}

// 2.存储确认是数字的区域以及他们的中心点还有他们的响应
struct SingleArea
{
	// 存储响应值
	float response;
	// 存储里面的candidates的序号
	int cc_index;
	// 存储中心
	Point center;
};

// 3.存储融合的数字
struct MergeArea
{
	// 存储响应值
	float response;
	// 存储与中心的旋转角度(以三四象限分割线为开始旋转的轴)
	float angle;
	// 存储里面的candidates的序号
	vector<int> cc_indexs;
	// 存储中心
	Point center;

};

// 4.判断角度，输入的标准坐标系下向量的 y 与 x，输出的是 以标准坐标系中的三四象限中间轴为0度轴，顺时针旋转的角度
float getVecAngle(float dy, float dx)
{
	float vecAngle = atan2(dy, dx) * 180 / pi;
	if (vecAngle <= -90) { vecAngle = -vecAngle - 90; }
	else if (vecAngle >= 0) { vecAngle = -vecAngle + 270; }
	else if (vecAngle < 0 && vecAngle > -90) { vecAngle = -vecAngle + 270; }
	return vecAngle;
}

// 4.vector<SingleArea>根据center.x值来排序
bool SortByX(SingleArea &v1, SingleArea &v2)//注意：本函数的参数的类型一定要与vector中元素的类型一致  
{
	//升序排列  
	return v1.center.x < v2.center.x;
}


// 5.vector<MergeArea>根据angle值来排序
bool SortByAngle(MergeArea &v1, MergeArea &v2)//注意：本函数的参数的类型一定要与vector中元素的类型一致  
{
	//升序排列  
	return v1.angle < v2.angle;
}

// 6.vector<MergeArea>根据response值来排序，降序
bool SortByRes(MergeArea &v1, MergeArea &v2)//注意：本函数的参数的类型一定要与vector中元素的类型一致  
{
	//降序排列  
	return v1.response > v2.response;
}

/* ------------------------------------ */



/* ------------------------------------ */
// vector<Ellipse>根据椭圆的面积大小值来排序，降序
bool SortByEllipseArea(Ellipse &e1, Ellipse &e2)//注意：本函数的参数的类型一定要与vector中元素的类型一致  
{
	//降序排列  
	return (e1._a*e1._b) > (e2._a*e2._b);
}



// 筛选椭圆的工作放到这里来，最后一行参数是输出。
void getGoodElls(vector<Ellipse>& ellsYaed, float& el_ab_p, Mat& resultImage, 
	int& scaleSize, Mat& gray_clone2, Ptr<LineSegmentDetector>& ls, Vec2f& e_center, float& dis2e_center, int& min_vec_num,
	Ellipse& el_dst, Mat& roi_dst, vector<Vec4f>& tLines, Mat& bl_drawLines)
{

	if (ellsYaed.size() == 0) { return; }
	// 先按照面积大小排序这些椭圆
	// 然后再从大的找起，寻找满足一定要求的椭圆
	sort(ellsYaed.begin(), ellsYaed.end(), SortByEllipseArea);

	int index = 0;
	while (index < ellsYaed.size()) {
		
		Ellipse& e = ellsYaed[index];
		index += 1;
		int g = cvRound(e._score * 255.f);
		Scalar color(0, g, 0);
		// 找到长轴短轴
		int long_a = e._a >= e._b ? cvRound(e._a) : cvRound(e._b);
		int short_b = e._a < e._b ? cvRound(e._a) : cvRound(e._b);

		// 如果长宽比过小则舍去
		cout << "比例是：" << short_b / (float)long_a << endl;
		if (short_b / (float)long_a < el_ab_p) { continue; }

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
		// imwrite("D:\\VcProject\\biaopan\\data\\test\\2\\aa1.jpg", roi_2);
		equalizeHist(roi_2, roi_2);
		Mat1b roi_2_add = roi_2.clone();
		// imwrite("D:\\VcProject\\biaopan\\data\\test\\2\\aa2.jpg", roi_2);

		// 运行LSD算法检测直线
		// 实验发现用自动阈值比canny更好得到刻度线，这里的 9=3*3模板，25=5*5模板，这里也发现 ADAPTIVE_THRESH_MEAN_C 比 ADAPTIVE_THRESH_GAUSSIAN_C的更容易检测到直线
		//adaptiveThreshold(roi_2, roi_2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 9, 0);


		adaptiveThreshold(roi_2_add, roi_2_add, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);
		threshold(roi_2, roi_2, 0, 255, CV_THRESH_OTSU);
		imshow("thresh_result_1", roi_2);
		imshow("thresh_result_2", roi_2_add);
		// 但是实际上，对于表盘应用，上面两个相减会得到并集的效果
		imshow("thresh_result_3", roi_2_add - roi_2);
		waitKey(0);


		// gray_clone, gray_edge, 3, 9, 3
		// Canny(roi_2, roi_2, 3, 9, 3); // Apply canny edge//可选canny算子


		vector<Vec4f> lines_std;
		vector<Vec4f> lines_dst;

		// Detect the lines
		ls->detect(roi_2_add - roi_2, lines_std);
		// Show found lines
		Mat drawnLines = roi_3.clone();



		cout << "lines_std.size() : " << lines_std.size() << endl;

		for (int j = 0; j < lines_std.size(); j++)
		{
			// 筛选掉那些距离中心比较远的线
			float distance = point2Line(e_center, lines_std[j]);
			if (distance <= dis2e_center)
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



		


		// 从里面选出 可能支持点 超过 35 的，一旦找到了就不需要再继续找了，因为前面已经确保这是足够大的椭圆了（按照面积排序过了）

		if (lines_dst.size() >= min_vec_num)
		{
			min_vec_num = lines_dst.size();
			el_dst = e;
			roi_dst = roi_3;
			tLines = lines_dst;
			bl_drawLines = drawnLines;
			break;
		}

		//imshow("drawnLines", drawnLines);
		imshow("Yaed", resultImage);
		cout << "index : " << index << endl;
		waitKey();
	}
}


/* ------------------------------------ */







/* ------------------------------------ */
// Mser目标检测 + nms
std::vector<cv::Rect> mser2(cv::Mat srcImage)
{


	std::vector<std::vector<cv::Point> > regContours;

	// 创建MSER对象
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 300, 10000, 0.5, 0.3);


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
		keeps.push_back(br);

		//keeps.push_back(br);
	}
	// 用nms抑制
	nms(keeps, 0.5);
	return  keeps;
}
/* ------------------------------------ */



/* ------------------------------------ */
// 下面的是各种噪声的生成，用于测试环境
//生成高斯噪声
double generateGaussianNoise(double mu, double sigma)
{
	//定义小值
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假构造高斯随机变量X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//构造随机变量
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真构造高斯随机变量
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0 * sigma + mu;
}

//为图像加入高斯噪声
Mat addGaussianNoise(Mat &srcImag)
{
	Mat dstImage = srcImag.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols*channels;
	//推断图像的连续性
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//加入高斯噪声
			int val = dstImage.ptr<uchar>(i)[j] +
				generateGaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			dstImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return dstImage;
}

/* ------------------------------------ */




int randomTest()
{
	string image_name = "F:\\chrome下载\\作业\\工业视觉\\dataset\\circle\\2.jpg";
	
	Mat3b image_1 = imread(image_name);
	vector<Rect> candidates;
	Mat3b image_2;
	resize(image_1, image_2, Size(image_1.cols/4, image_1.rows/4));
	candidates = mser2(image_1);
	for (int i=0;i<candidates.size();i++)
	{
		rectangle(image_2, candidates[i], Scalar(0, 255, 0), 1);
	}
	imshow("image_2", image_2);
	waitKey();
	return 0;
}


// 检测一张图片的各种参数，main函数
// main
int main()
{
	// 给随机种子
	srand((unsigned)time(NULL));
	string images_folder = "D:\\VcProject\\biaopan\\imgs\\";
	string out_folder = "D:\\VcProject\\biaopan\\imgs\\";
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";
	// 加载svm
	Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
	vector<string> names;

	//glob(images_folder + "Lo3my4.*", names);
	// 1 85是重要因素


	string picName = "4 54.jpg";
	names.push_back("D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\4\\" + picName);

	//string picName = "16 43.jpg";
	//names.push_back("D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\16\\" + picName);
	//string picName = "0001.jpg";
	//names.push_back("D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\test\\" + picName);
	//string picName = "12 14.jpg";
	//names.push_back("D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\12\\" + picName);
	//string picName = "013.jpg";
	//names.push_back("D:\\VcProject\\biaopan\\imgs\\013.jpg");
	int scaleSize = 5;
	for (const auto& image_name : names)
	{
		//string name = image_name.substr(image_name.find_last_of("\\") + 1);
		//name = name.substr(0, name.find_last_of("."));

		Mat3b image_1 = imread(image_name);

		// 对图片进行压缩
		//resize(image_1, image_1, Size(image_1.size[1] / 2.5, image_1.size[0] / 2.5));
		//resize(image_1, image_1, Size(image_1.size[1] / 8, image_1.size[0] / 8));
		blur(image_1, image_1, Size(7, 7));
		imshow("blur_image", image_1);
		waitKey();
		// 给图片添加噪声
		//image_1 = addGaussianNoise(image_1);

		// 有些图片需要翻转
		// flip(image_1, image_1, -1);
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
		Mat1b gray_clone_add = gray.clone();

		imshow("gray_clone", gray_clone);
		waitKey();
		


		//Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
		//filter2D(gray_clone, gray_clone_add, CV_8UC3, kernel);
		//imshow("gray_clone", gray_clone_add);
		//waitKey();

		Mat1b gray_edge;
		Canny(gray_clone, gray_edge, 3, 9, 3);
		imshow("gray_clone", gray_edge);
		waitKey();

		yaed.Detect(gray_edge, ellsYaed);
		



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
		imshow("resultImage", resultImage);
		waitKey();



		// 开展搜索范围，以长轴为直径的正方形区域
		
		Mat1b gray_clone2;
		cvtColor(image_1, gray_clone2, CV_BGR2GRAY);

		// 选取至少有 35 个可能支持点的椭圆
		int min_vec_num = 25;
		// 存储目标椭圆
		Ellipse el_dst;
		// 存储目标区域
		Mat1b roi_zero = Mat::zeros(400, 400, CV_8UC1);
		Mat1b& roi_dst = roi_zero;
		// 存储目标的可能支持线
		vector<Vec4f> tLines;
		// 允许直线距离e_center的距离
		float dis2e_center = 30;
		// 允许椭圆的最小长宽比
		float el_ab_p = 0.75;

		Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
		Mat bl_drawLines;
		Vec2f e_center = Vec2f(200, 200);

		getGoodElls(ellsYaed, el_ab_p, resultImage,scaleSize, gray_clone2, ls, e_center, dis2e_center, min_vec_num,
			el_dst, roi_dst, tLines, bl_drawLines);

		// 如果上面的canny算子获取不到椭圆就用下面的方法获取
		if (ellsYaed.size() == 0 || tLines.size() == 0)
		{
			adaptiveThreshold(gray_clone_add, gray_clone_add, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 9, 0);
			imshow("gray_clone", gray_clone_add);
			waitKey();
			yaed.Detect(gray_clone_add, ellsYaed);
			yaed.DrawDetectedEllipses(resultImage, ellsYaed);
			imshow("resultImage", resultImage);
			waitKey();
			getGoodElls(ellsYaed, el_ab_p, resultImage, scaleSize, gray_clone2, ls, e_center, dis2e_center, min_vec_num,
				el_dst, roi_dst, tLines, bl_drawLines);
		}

		if (ellsYaed.size() == 0)
		{
			cout << "检测不到椭圆，请按任意键退出程序" << endl;
			cout << "--------------------------------" << endl;
			system("pause");
			return 0;
		}

		
		// 这里显示的是最好的那个椭圆的画线
		imshow("drawnLines", bl_drawLines);


		// 显示椭圆效果
		int g = cvRound(el_dst._score * 255.f);
		Scalar color(0, 255, 255);
		//ellipse(roi_dst, Point(cvRound(el_dst._xc), cvRound(el_dst._yc)), Size(cvRound(el_dst._a), cvRound(el_dst._b)), el_dst._rad*180.0 / CV_PI, 0.0, 360.0, color, 2);
		
		Mat roi_center = roi_dst.clone();
		Mat roi_line = roi_dst.clone();
		Mat roi_mser = roi_dst.clone();
		Mat roi_merge = roi_dst.clone();


		// 下面测试用 自动阈值 来mser，经过试验发现，这个方法切割比直接mser好
		Mat roi_thresh = roi_dst.clone();
		Mat roi_thresh_otsu = roi_dst.clone();
		Mat roi_thresh_mser = roi_dst.clone();
		// 实验证明 ADAPTIVE_THRESH_GAUSSIAN_C 比 ADAPTIVE_THRESH_MEAN_C 分割的更好，边缘更加光滑
		adaptiveThreshold(roi_thresh, roi_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);
		//erode(roi_thresh, roi_thresh, Mat(2, 2, CV_8U), Point(0, 0), 1);
		//medianBlur(roi_thresh, roi_thresh, 3);


		//threshold(roi_thresh_otsu, roi_thresh_otsu, 0, 255, CV_THRESH_OTSU);
		//roi_thresh = roi_thresh - roi_thresh_otsu;

		//获取自定义核
		Mat e_element = getStructuringElement(MORPH_ELLIPSE, Size(4, 4)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
		// Mat d_element = getStructuringElement(MORPH_ELLIPSE, Size(2, 2)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
		// 膨胀把所有的数字都抠出来
		//dilate(roi_thresh, roi_thresh, e_element);

		imshow("roi_thresh", roi_thresh);
		waitKey(0);

		
		//erode(roi_thresh, roi_thresh, e_element);
		//dilate(roi_thresh, roi_thresh, e_element);
		// 开操作
		//morphologyEx(roi_thresh, roi_thresh, MORPH_OPEN, e_element);


		// 下面尝试腐蚀多次，得出我们想要的指针的直线，而不是通过之前的步骤做到
		Mat d_element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		Mat line_roi;
		vector<Vec4f> tLines2;
		vector<Vec4f> tLines1;

		erode(roi_thresh, line_roi, d_element);

		erode(line_roi, line_roi, d_element);
		erode(line_roi, line_roi, d_element);
		ls->detect(line_roi, tLines1);
		ls->drawSegments(roi_line, tLines1);
		imshow("roi_line11", roi_line);
		imshow("line_roi11", line_roi);
		
		// 给他区分尾点和非尾点
		for (int j = 0; j < tLines1.size(); j++)
		{
			// 筛选掉那些距离中心比较远的线
			float distance = point2Line(e_center, tLines1[j]);

			if (distance <= dis2e_center)
			{
				Vec4f l = tLines1[j];
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
				tLines2.push_back(dl);
			}
		}
		ls->drawSegments(roi_line, tLines2);
		imshow("roi_line11", roi_line);
		waitKey(0);


		//存储里面的旋转矩阵。
		vector<RotatedRect> rrects;
		vector<Rect> ccs = mser(roi_dst, rrects);
		for (int cci=0;cci < ccs.size();cci++) 
		{
			rectangle(roi_thresh_mser, ccs[cci], Scalar(255,255,255), 1);
			// 然后再把矩形框往外框一点，就能把所有数字区域都得到了

		}
		imshow("roi_thresh_mser", roi_thresh_mser);
		waitKey(0);





		// 2. hough圆检测获取精确中心，lsd获取最长直线（指针线）

		/**
			中心是(200, 200)，在这个中心以一定半径搜索精准的圆心
		**/
		int searchRadius = 35;
		vector<Vec3f> circles;
		Mat1b centerArea = roi_dst(Rect(200 - searchRadius, 200 - searchRadius, 2 * searchRadius, 2 * searchRadius)).clone();
		Mat1b centerArea2 = centerArea.clone();
		// Mat3b centerArea3 = Mat::zeros(centerArea2.cols, centerArea2.rows, CV_8UC3);


		GaussianBlur(centerArea, centerArea, Size(3, 3), 1, 1);
		//Mat1b centerArea_add = centerArea.clone();
		//threshold(centerArea_add, centerArea_add, 0, 255, CV_THRESH_OTSU);

		// 这个局部阈值能使中间部分完全变白，从而可以探测得到
		adaptiveThreshold(centerArea, centerArea, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 9, 0);
		imshow("center", centerArea);
		waitKey(0);
		// 下面这种做法是比较冒险的。。因为不保证最后的输出是模糊还是清晰，同时无法确保这个阈值30在各个场景均适用
		// HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 30, 8, 0, 6);
		// 后面四个数字分别代表： 分辨率（可以理解为步进的倒数），两个圆之间最小距离，canny的低阈值，投票的累积次数（即有多少个点属于该圆），圆最小半径，圆最大半径
		// 一般投票值在 15 到 20间，如果拍摄视角特别偏，把中间圆变成椭圆，就需要降低投票值了，目前检测视角比较正常，所以只需要保持20
		HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 150, 18, 10, 25);


		// 实际上上面是可以继续优化的，比如搜索不到任何圆的情况下就尝试扩大搜索域，减少投票次数，放大圆最大半径

		// 寻找离形心最近的那个圆心作为我们的准心，绘制出圆
		Point ac_center = Point(200, 200);
		Point f_center = Point(searchRadius/2, searchRadius/2);
		float f2ac = 200;
		for (int i=circles.size()-1;i>=0;--i) 
		{
			Point mCenter = Point(circles[i][0], circles[i][1]);
			// 下面两种办法，一种是取距离f_center最短，一种是面积最小,先采用第二种
			//float mDistance = point2point(mCenter, f_center);
			//if (mDistance < f2ac)
			//{
			//	f2ac = mDistance;
			//	ac_center = Point(mCenter.x + 200 - searchRadius, mCenter.y + 200 - searchRadius);
			//}
			if (circles[i][2] < f2ac)
			{
				f2ac = circles[i][2];
				ac_center = Point(mCenter.x + 200 - searchRadius, mCenter.y + 200 - searchRadius);
			}
			circle(centerArea2, mCenter, circles[i][2], Scalar(255, 255, 255));
		}

		imshow("center", centerArea);
		imshow("center2", centerArea2);
		circle(roi_center, ac_center, 3, color, -1);
		imshow("roi_center", roi_center);



		// 下面是获取指针线
		// 先是找出所有线的最近后接线
		float mRadius = 80;
		float angle = 4;
		float angelCos = cos(angle *  pi / 180);
		
		// 注意这里读取的 Lines 是 上面 通过  腐蚀得到的那个线，而不是原先的那个线，但是拟合椭圆的时候用的是原先的线

		// 存储这些后线对
		vector<int> backs = vector<int>(tLines2.size());
		// 存储这些前线对，方便之后的计算
		vector<int> fronts = vector<int>(tLines2.size());

		// 初始化这些对的值，默认为-1
		for (int k= tLines2.size()-1;k>=0;--k)
		{
			backs[k] = -1;
			fronts[k] = -1;
		}

		for (int i=tLines2.size() - 1;i >= 0; --i)
		{
			Vec4f& line1 = tLines2[i];
			// 搜索半径内，首先是属于自己后面的线，然后是中点相连所成的角度不能大过某个值，并且取距离自己最短的那些线，最后是看这条线是不是已经成为了别人的后线了
			int mIndex = i;
			float mDis = 1000;
			for (int j = tLines2.size() - 1; j >= 0; --j)
			{
				if (i == j)
					continue;
				Vec4f& line2 = tLines2[j];

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
		for (int i = tLines2.size() - 1; i >= 0; --i)
		{
			//cout << "backs-" << i << " : " << backs[i] << endl;
			//cout << "fronts-" << i << " : " << fronts[i] << endl;
			int n = 0;
			for (int j = tLines2.size() - 1; j >= 0; --j) {
				if (backs[j] == i)
					n++;
			}
			if (n > 2)
			{
				//cout << "线-" << i << "被超过1条线当做后线" << endl;
			}
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
			//cout << "[";
			for (int j = 0; j <= a.size() - 1; ++j)
			{
				//cout << a[j] << ",";
			}
			//cout << "]" << endl;
		}

		// 画出直线组合
		Scalar cc(0, 255, 255);
		for (int i = goal_lines.size() - 1; i >= 0; --i)
		{
			vector<int>& goal_line = goal_lines[i];
			for (int j = 0; j <= goal_line.size() - 1; ++j)
			{
				int li = goal_line[j];
				Vec4f ln = tLines2[li];
				Point point1 = Point(ln[0], ln[1]);
				Point point2 = Point(ln[2], ln[3]);
				cv::line(roi_line, point1, point2, cc, 2);
			}

			imshow("roi_line", roi_line);
			//waitKey();
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
				// 这里要进行平方，因为越长的占比应该越大
				total_length += point2point(tLines2[li]);
			}
			if (total_length > maxLength) { maxLength = total_length; maxLine = goal_line; }
		}
		// 画出最长的线
		Scalar aa(255, 255, 255);
		for (int j = 0; j <= maxLine.size() - 1; ++j)
		{
			int li = maxLine[j];
			Vec4f ln = tLines2[li];
			Point point1 = Point(ln[0], ln[1]);
			Point point2 = Point(ln[2], ln[3]);
			cv::line(roi_line, point1, point2, aa, 2);
		}
		imshow("roi_line", roi_line);
		waitKey();

		// 拿最长的线的结尾当做指针线的尾点
		Vec4f last_part = tLines2[maxLine[maxLine.size() - 1]];
		Point lastPoint = Point(last_part[2], last_part[3]);


		// 3. 上面方法不行就用回以前的 bim+支持点方向分类 然后抽样获取椭圆
		// 给tLine进行分类（按角度分类），比如按照40度一个扇区，就有9个扇区 
		// 这个 tLine 和用来识别指针的 line是不一样的。
		int sangle = 40;
		vector<vector<int>> sangles_(360 / sangle);
		for (int ki = tLines.size() - 1; ki >= 0; --ki)
		{
			
			// 计算夹角的cos值
			int xd = tLines[ki][2] - ac_center.x;
			int yd = ac_center.y - tLines[ki][3];
			// 值域在 0~360之间
			float vangle = fastAtan2(yd, xd);
			sangles_[(int)vangle / sangle].push_back(ki);
		}
		// 这里要过滤要一些没有点的扇区
		vector<vector<int>> sangles;
		for (int ki = 0; ki < sangles_.size(); ki++)
		{
			vector<int> sangel_ = sangles_[ki];
			if (sangel_.size() > 0) 
			{
				sangles.push_back(sangel_);
			}
		}


		int asize = sangles.size();
		int ranTimes = 1000;
		// 允许椭圆拟合中支持点最大的远离程度，这里的程度是乘上椭圆长轴的
		float acceptThresh = 8;
		// 存储当前的支持点个数
		int nowSnum = 0;
		// 存储最多支持点的那个椭圆的外接矩形
		RotatedRect bestEl;
		// 存储最终的支持点
		vector<int> bestSupportPoints;
		// 这里设置随机的次数
		for (int ri=0;ri<ranTimes;ri++)
		{
			// 存储要选取的扇区，从5个扇区中取值
			vector<int> ks;
			// 给定初始的点
			int kindex = random(asize - 1);
			ks.push_back(kindex);
			for (int ki = 0; ki < 4; ki++)
			{
				// 产生1~2的随机数，也就是所选取的扇区，后选的扇区比前选的扇区最多隔两个位置，保证有一定的角度
				kindex = (kindex + random(2)) % asize;
				ks.push_back(kindex);
			}
			// 从每个所选的扇区中随机选取一个点进行椭圆拟合
			vector<Point> candips;
			for (int ki = 0; ki < 5; ki++)
			{
				// ks[ki]指随机选取到的扇区的序号，sangles[扇区序号]拿到的就是这个扇区里面的所有点的序号，所以这里最终随机产生的就是扇区里面的第几个点
				vector<int> shanqu = sangles[ks[ki]];
				int sii = random(shanqu.size() - 1);
				Vec4f vvvv = tLines[shanqu[sii]];
				Point ssss = Point(vvvv[2], vvvv[3]);
				candips.push_back(ssss);
			}
			// 拟合椭圆后计算支持点个数
			RotatedRect rect = fitEllipse(candips);

			//Mat forellipse = roi_foresee.clone();
			//ellipse(forellipse, rect, Scalar(255, 255, 255));
			//imshow("forellipse", forellipse);
			//int aaaa = waitKey(0);
			//if (aaaa == 100) { continue; }

			// 存储支持点
			vector<int> support_points;
			for (int ki=0;ki<tLines.size();ki++)
			{
				Point sp = Point(tLines[ki][2], tLines[ki][3]);
				Point newsp = origin2el(rect.center, rect.angle / (float)180 * pi, sp);
				//circle(forellipse, Point(tLines[ki][2], tLines[ki][3]), 3, Scalar(0, 0, 0), -1);
				// 这里还需要考虑上拟合圆的大小
				float edistance = abs(sqrt(pow(newsp.x, 2) / pow(rect.size.width/2, 2) + pow(newsp.y, 2) / pow(rect.size.height/2, 2)) - 1) * max(rect.size.width / 2, rect.size.height / 2);
				//cout << "edistance： " << edistance << endl;
				if (edistance <= acceptThresh)
				{
					support_points.push_back(ki);
				}
				//cout << "support_points： " << support_points.size() << endl;
				//cout << "当前点数： " << ki << endl;
				//imshow("forellipse", forellipse);
				//waitKey(0);
			}

			//cout << "支持点为： " << endl;
			//waitKey(0);
			//forellipse = roi_foresee.clone();
			//for (int ki = 0; ki < support_points.size(); ki++)
			//{
			//	int tLineIndex = support_points[ki];
			//	circle(forellipse, Point(tLines[tLineIndex][2], tLines[tLineIndex][3]), 3, Scalar(0, 0, 0), -1);
			//}
			//imshow("forellipse", forellipse);
			//waitKey(0);

			if (support_points.size() >= nowSnum)
			{
				nowSnum = support_points.size();
				bestSupportPoints = support_points;
				bestEl = rect;
			}
				
		}
		// 经过上面的循环就能得出支持点最多的那个外接椭圆，下面显示出来

		// 对拿到的支持点继续进行拟合
		vector<Point> supporters;
		for (int ki = 0; ki < bestSupportPoints.size(); ki++)
		{
			int tLineIndex = bestSupportPoints[ki];
			Point supporter = Point(tLines[tLineIndex][2], tLines[tLineIndex][3]);
			supporters.push_back(supporter);
			circle(roi_line, supporter, 3, Scalar(0, 0, 0), -1);
		}
		bestEl = fitEllipse(supporters);
		// 一般的出来的椭圆不能是太扁，也就是长轴比短轴是有个限制的，所以需要把上面的椭圆方法单独抽出成为一个方法，
		// 实现递归调用，而且每次传送一个计数器，计数器表示第几次递归，不能递归太多次导致无限循环
		// 有条件的话还可以拿一开始拿到的椭圆进行 比例比对，如果比例失衡打到一定限度，那就是弄错了，需要递归一次，如果最终递归还是错误，那么椭圆就是一开始拿到的椭圆(就是检测表盘区域的椭圆)
		ellipse(roi_line, bestEl, Scalar(255, 255, 255));
		RotatedRect bestEl_scale = RotatedRect(bestEl.center, Size2f(bestEl.size.width * 0.86, bestEl.size.height * 0.86), bestEl.angle);
		ellipse(roi_line, bestEl_scale, Scalar(0, 0, 0));
		imshow("roi_line", roi_line);
		waitKey(0);
		waitKey(0);






		// mser检测
		std::vector<cv::Rect> candidates;
		imshow("roi_mser", roi_dst);
		waitKey(0);
		// candidates = mser(roi_dst);
		// 实验证明，thresh后的mser比直接mser更好
		candidates = ccs;


		vector<SingleArea> sas;

		// 存储要拟合各个数字的椭圆
		vector<Point> numberAreaCenters;


		// 投票选择哪个角度才是正确的（只有内圈的mser才能投票），投票数多的就是表盘旋转的角度，然后复原旋转前的样子进行分割后再判别。
		// 允许相同角度之间的差值
		int rect_angle_err = 5;
		// 把属于同一个角度的都加起来，最后要计算他的平均值。
		unordered_map<int, int> rect_angle_sums;
		unordered_map<int, int> rect_angle_nums;
		// 存储每个角度对应的那些旋转矩形的id
		unordered_map<int, vector<int>> rect_goal_ixs;
		// 抽出他的迭代器
		unordered_map<int, int>::iterator  rect_iter;
		unordered_map<int, int>::iterator  rect_sums_iter;
		unordered_map<int, int>::iterator  rect_goal_ixs_iter;
		
		// 存储符合条件的那些矩形的index
		vector<int> ris;
		// 区域显示
		for (int i = 0; i < candidates.size(); ++i)
		{
			// 先筛选掉拟合圆外面的点
			Point ncenter = Point(candidates[i].x + candidates[i].width / 2, candidates[i].y + candidates[i].height / 2);
			Point newsc = origin2el(bestEl.center, bestEl.angle / (float)180 * pi, ncenter);
			float ndistance = sqrt(pow(newsc.x, 2) / pow(bestEl.size.width / 2, 2) + pow(newsc.y, 2) / pow(bestEl.size.height / 2, 2));
			// 这里的椭圆需要往内部缩一下，去除过多的咋点，比如刻度线
			if (ndistance >= 0.86) { continue; }

			ris.push_back(i);

			// 先把所有的都标出来
			rectangle(roi_mser, candidates[i], Scalar(255, 255, 255), 1);
			drawRotatedRect(roi_mser, rrects[i]);
			cout << "w=" << candidates[i].width << ",h=" << candidates[i].height << ",w-h=" << candidates[i].width - candidates[i].height << endl;
			//cout << "角度为： " << rrects[i].angle << endl;


			// 统计投票
			int rrect_angle = rrects[i].angle;
			bool is_new_r_angle = true;
			for (rect_iter = rect_angle_nums.begin(); rect_iter != rect_angle_nums.end(); rect_iter++)
			{
				// 与存在的angle进行比较，如果相差不大，就放进那个angle里面
				// iter->first是key，iter->second是value
				int compare_angle = rect_iter->first;
				if (abs(rrect_angle - compare_angle) <= rect_angle_err)
				{
					is_new_r_angle = false;
					rect_angle_nums[compare_angle] = rect_iter->second + 1;
					rect_angle_sums[compare_angle] += rrect_angle;
					rect_goal_ixs[compare_angle].push_back(i);
					break;
				}
			}
			if (is_new_r_angle) 
			{ 
				rect_angle_nums[rrect_angle] = 1; 
				rect_angle_sums[rrect_angle] = 0; 
				vector<int> emptyv;
				rect_goal_ixs[rrect_angle] = emptyv;
			}
			



			// 把符合数字的区域框选出来
			Mat mser_item = roi_dst(candidates[i]);



			vector<float> v = getHogData(mser_item);
			Mat testData(1, 144, CV_32FC1, v.data());
			int response = svm->predict(testData);

			// 如果发现是黏连的，就用腐蚀把他分割然后弄出来


			// 显示出来
			if (response >= 0)
			{
				rectangle(roi_mser, candidates[i], Scalar(255, 255, 255), 2);
				circle(roi_mser, ncenter, 2, color, -1);

				// 为了之后的运算
				sas.push_back({ float(response), i, ncenter });
				numberAreaCenters.push_back(ncenter);
				// 这里需要修改，只是方便测试
				// if (response == 3) { response = 2; }

				//cout << "标签为： " << response << endl;
				//imshow("roi_mser", roi_mser);
				//waitKey(0);
			}
		}

		imshow("roi_mser", roi_mser);
		waitKey(0);


		int max_rect_angle = 0;
		int max_rect_angle_nums = 0;
		int max_rect_angle_sum = 0;
		// 找出票数最多的旋转角度，然后复原旋转图像，分割数字，识别
		for (rect_iter = rect_angle_nums.begin(); rect_iter != rect_angle_nums.end(); rect_iter++)
		{
			if (rect_iter->second > max_rect_angle_nums)
			{
				max_rect_angle_nums = rect_iter->second;
				max_rect_angle = rect_iter->first;
			}
		}

		for (rect_sums_iter = rect_angle_sums.begin(); rect_sums_iter != rect_angle_sums.end(); rect_sums_iter++)
		{
			if (rect_sums_iter->first == max_rect_angle)
			{
				// rect_sums_iter->second拿到的是投票最多的角度的和
				max_rect_angle = rect_sums_iter->first;
				max_rect_angle_sum = rect_sums_iter->second;
				break;
			}
		}
		

		// 下面判断这个矩形是向右转还是向左转
		// 假设这个矩形处于0度旋转，如果在0度时他的长宽比和他的外围长宽比并不是一致，说明侧边和底边与正常方向理解是反过来的。
		// 如果旋转矩形的旋转边是侧边，那么说明矩形往右倾，如果旋转边是底边，说明矩形往左倾。
		// 通过统计长宽比一致情况，来确保是哪种方向
		vector<int> rids = rect_goal_ixs[max_rect_angle];
		int is_same_num = 0;
		for (int ri = 0; ri < rids.size(); ri++)
		{
			int ix = rids[ri];
			RotatedRect rrect = rrects[ix];

			// 获取旋转矩形的四个顶点
			cv::Point2f* vertices = new cv::Point2f[4];
			rrect.points(vertices);
			// 上面的边长应该是 p[0]与p[3]的距离，侧面的边长应该是p[1]与p[0]的距离
			float rw = point2point(vertices[0], vertices[3]);
			float rh = point2point(vertices[0], vertices[1]);

			float is_same_shape = (candidates[ix].width - candidates[ix].height) * (rw - rh);
			if (is_same_shape >= 0) { is_same_num++; }
		}

		// 最后这个旋转角要平均一下，但是发现不平均的话效果更好
		//max_rect_angle = max_rect_angle_sum / max_rect_angle_nums;
		// 构造旋转矩阵
		
		Mat rotationMatrix;
		// 下面这个参数表明的是大多数的角度边到底应该是底边还是侧边
		bool is_b_or_s = false;
		float rangle = 0;
		if (is_same_num / rids.size() >= 0.5) 
		{
			// 在getRotationMatrix2D中，角度为负，顺时针；角度为正，逆时针。第三个参数默认不用管
			// 角度边（也就是p[0]~p[3]的那条边）是底边，说明往左倾
			is_b_or_s = true;
			rangle = -abs(max_rect_angle);
			rotationMatrix = getRotationMatrix2D(Point(200, 200), rangle, 1);//计算旋转的仿射变换矩阵 
		}
		else 
		{
			// 角度边（也就是p[0]~p[3]的那条边）是侧边，说明往右倾
			is_b_or_s = false;
			rangle = (90 - abs(max_rect_angle));
			rotationMatrix = getRotationMatrix2D(Point(200, 200), rangle, 1);//计算旋转的仿射变换矩阵 
		}
		Mat rMat;
		warpAffine(roi_dst, rMat, rotationMatrix, Size(roi_dst.cols, roi_dst.rows));//仿射变换  
		imshow("rMat", rMat);
		waitKey(0);



		// 上面只是发现旋转角而已，下面就是在roi_dst中的每个小mser窗口中进行旋转，旋转中心是旋转矩形的中心。
		// 然后再进行数字分割，然后再交给识别器去识别数字。


		Mat roi_dst_clone_ = roi_dst.clone();
		// 画出按照最大角变化的所有旋转矩形
		for (int ri = 0;ri < ris.size();ri++)
		{
			int ix = ris[ri];
			RotatedRect rrect = rrects[ix];
			Rect rect = candidates[ix];
			Point rect_center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
			// 实验发现把中心换成 mser出来的矩形的中心，中心更准确，同时扩张一下范围
			rrect.center = rect_center;
			rrect.size = Size(rrect.size.width + 4, rrect.size.height + 4);
			// 获取旋转矩形的四个顶点
			cv::Point2f* vertices = new cv::Point2f[4];
			rrect.points(vertices);
			// 上面的边长应该是 p[0]与p[3]的距离，侧面的边长应该是p[1]与p[0]的距离
			int rw = point2point(vertices[0], vertices[3]);
			int rh = point2point(vertices[0], vertices[1]);
			int temp_ = 0;
			float is_same_shape = (candidates[ix].width - candidates[ix].height) * (rw - rh);

			// 他的角度边是底边
			if (is_same_shape >= 0) 
			{
				// 与实际角度边一致
				if (is_b_or_s) { rrect.angle = max_rect_angle; }
				// 实际应该是侧边作为角度边才对
				else { rrect.angle = 270 - abs(max_rect_angle); }
			}
			else 
				// 角度边是侧边
			{ 
				// 与实际角度边一致
				if (!is_b_or_s) { rrect.angle = max_rect_angle; }
				// 实际应该是底边作为角度边才对
				else { rrect.angle = 90 + abs(max_rect_angle); }
				// 角度边是侧边，因此，他的rw实际上是h，rh实际上是w
				temp_ = rw; rw = rh; rh = temp_;
			}

			
			drawRotatedRect(roi_dst_clone_, rrect);

			cout << "rrect.angle = " << rrect.angle << endl;
			// 下面单独提取这些旋转矩阵区域
			// 先把区域放大到足够大，两倍于原来图像，然后围绕中心旋转，之后再提取roi
			Mat scaleMat = roi_dst(Rect(rrect.center.x - rect.width, rrect.center.y - rect.height, 2 * rect.width, 2 * rect.height));
			// 在放大的区域中，中心是 Point(rect.width, rect.height)
			rotationMatrix = getRotationMatrix2D(Point(rect.width, rect.height), rangle, 1);//计算旋转的仿射变换矩阵 
			Mat scaleMat2;
			warpAffine(scaleMat, scaleMat2, rotationMatrix, Size(scaleMat.rows, scaleMat.cols));//仿射变换  
			// 提取目标区域
			int x = max(0, rect.width - rw / 2);
			int y = max(0, rect.height - rh / 2);
			int width = min(scaleMat2.cols - x, rw);
			int height = min(scaleMat2.rows - y, rh);
			Rect roi_area = Rect(x, y, width, height);
			Mat final_mser_roi = scaleMat2(roi_area);
			imshow("mmmser", final_mser_roi);
			waitKey();
		}
		imshow("roi_dst_clone_", roi_dst_clone_);
		waitKey();



		for (int ri = 0; ri < ris.size(); ri++)
		{
			int ix = ris[ri];
			RotatedRect rrect = rrects[ix];
			Rect rect = candidates[ix];
			Point rect_center = Point(rect.width / 2, rect.height / 2);
			rotationMatrix = getRotationMatrix2D(rect_center, rangle, 1);
			Mat roi_rect = roi_dst(rect);
			Mat mser_r;

			// 获取旋转矩形的四个顶点
			cv::Point2f* vertices = new cv::Point2f[4];
			rrect.points(vertices);
			// 上面的边长应该是 p[0]与p[3]的距离，侧面的边长应该是p[1]与p[0]的距离
			int rw = point2point(vertices[0], vertices[3]);
			int rh = point2point(vertices[0], vertices[1]);

			float is_same_shape = (candidates[ix].width - candidates[ix].height) * (rw - rh);

			warpAffine(roi_rect, mser_r, rotationMatrix, Size(roi_rect.rows, roi_rect.cols));//仿射变换  



			imshow("mmmser", mser_r);
			waitKey(0);
		}


		imshow("roi_mser", roi_mser);
		waitKey(0);





		// 4. 数值读取
		// 先找出变换点，也就是能作为透视变换的点，然后是变换
		// 从有数字的里面搜寻成对存在的
		// 存储这些合并后的区域以及他们的响应值




		

		// 中间辅助的容器
		vector<MergeArea> mas;
		// 规定融合的最短距离
		float merge_distance = 30;
		float min_merge_distance = 10;
		// 允许与angle 的误差
		int angle_error = 12;
		int max_likely_angle = -1;
		// 按照一定步进慢慢缩小 merge_distance
		float merge_distance_scale_step = 3;
		// 记录连接情况的表，这是个二维数组
		vector<vector<int>> joinTable(sas.size());
		// 初始化二维数组

		// 统计连线出现的所有角度的对应的边的数量
		unordered_map<int, int> line_angle_nums;
		// 抽出他的迭代器
		unordered_map<int, int>::iterator  iter;


		
		// 最终决定多大比例的线是要占最多的
		float max_portion = 0.5;


		// 计算统计好的那些连线
		// 如果统计里面最多的那个连线不超过总数的80%，那么就降低阈值，重复寻找最佳边表

		while (true)
		{
			int total_join_line_num = 0;
			int max_join_line_num = 0;
			int max_join_line_angle = -1;
			for (iter = line_angle_nums.begin(); iter != line_angle_nums.end(); iter++)
			{
				if (iter->second > max_join_line_num)
				{
					max_join_line_num = iter->second;
					max_join_line_angle = iter->first;
				}
				total_join_line_num += iter->second;
			}
			// 记录下在最优阈值下的最多的那个angle
			max_likely_angle = max_join_line_angle;
			if (total_join_line_num == 0 || max_join_line_num / (float)total_join_line_num < max_portion)
			{
				merge_distance -= merge_distance_scale_step;
				if (merge_distance < min_merge_distance)
				{
					// 如果阈值过低，那么就可以暂停
					break;
				}
				else
				{
					// 重复操作，以下操作就是寻找边并更改 边表 以及 line_angle_nums

					// 初始化 边表 以及 line_angle_nums
					for (int kii = 0; kii < sas.size(); kii++)
					{
						joinTable[kii] = vector<int>(sas.size());
						// -2表示对角线（自己连自己），-1表示没有连接，大于等于0的值表示这条连接的直线的倾斜角
						for (int kjj = 0; kjj < sas.size(); kjj++)
						{
							if (kii == kjj) { joinTable[kii][kjj] = -2; }
							else { joinTable[kii][kjj] = -1; }
						}
					}
					line_angle_nums = unordered_map<int, int>();

					// 根据阈值把点变成边。
					for (int kii = 0; kii < sas.size(); kii++)
					{
						float response = sas[kii].response;
						// 要么直接把属于融合的那个类10直接渲染成 merge_area 输出，要么就把他当做和单个区域一样的东西
						if (response > 9)
						{
							//csets.push_back();
							//continue;
						}
						Point sacenter = sas[kii].center;
						// 寻找能融合的那个点，也就是距离他中心最短的那个点
						float min_dd = merge_distance;
						int target_sa = -1;
						for (int kjj = 0; kjj < sas.size(); kjj++)
						{
							// 这里需要查看kjj是否已经连接过kii，已经连过的就跳过(-1表示未连接，-2表示自己连自己)
							if (joinTable[kii][kjj] ==-2 || joinTable[kii][kjj] > -1 || sas[kjj].response > 9) { continue; }
							float jj_ii_distance = point2point(sacenter, sas[kjj].center);
							if (min_dd > jj_ii_distance) { min_dd = jj_ii_distance; target_sa = kjj; }
						}
						// 如果找到融合点，就修改table，若没有就不用管
						if (target_sa >= 0)
						{
							// 计算出这条直线的倾斜角
							double dy = sas[kii].center.y - sas[target_sa].center.y;
							double dx = sas[kii].center.x - sas[target_sa].center.x;
							// 注意，这里不用atan2，因为我们只需要在一四象限判断方向（因为只是看看斜率）,因为的到的是-90到90，向上偏移90，保证-1和-2是其他用途。
							int s_line_angle = int(atan(dy / dx) * 180 / pi) + 90;


							// 从统计中寻找跟这个角度类似的，如果没有就添加该项的统计，如果有就加1

							bool is_new_angle = true;
							for (iter = line_angle_nums.begin(); iter != line_angle_nums.end(); iter++)
							{
								// 与存在的angle进行比较，如果相差不大，就放进那个angle里面
								// iter->first是key，iter->second是value
								int compare_angle = iter->first;
								if (abs(s_line_angle - compare_angle) <= angle_error) 
								{
									is_new_angle = false; 
									line_angle_nums[compare_angle] = iter->second + 1; 
									joinTable[kii][target_sa] = compare_angle;
									joinTable[target_sa][kii] = compare_angle;
									break; 
								}
							}
							if (is_new_angle) 
							{
								line_angle_nums[s_line_angle] = 1; 
								joinTable[target_sa][kii] = s_line_angle;
								joinTable[kii][target_sa] = s_line_angle;
							}

						}
					}
				}
			}
			// 如果已经达到要求，就退出循环
			else { break; }
		}



		// 如果统计里面最多的超过80%，把那些连线角度不是最多的那些连线(根据angle判断)，在table中全部置为断开状态（即0）
		for (int kii = 0; kii < sas.size(); kii++)
		{
			// -2表示对角线（自己连自己），-1表示没有连接，大于等于0的值表示这条连接的直线的倾斜角
			for (int kjj = 0; kjj < sas.size(); kjj++)
			{
				if (kjj == kii) { continue; }
				if (joinTable[kii][kjj] != max_likely_angle) { joinTable[kii][kjj] = -1; }
			}
		}


		// 创建一个记录节点是否被访问的数组
		vector<bool> set_is_visited = vector<bool>(sas.size());
		vector<vector<int>> goal_sets;
		for (int kii = 0; kii < sas.size(); kii++)
		{
			if (!set_is_visited[kii])
			{
				vector<int> goal_set;
				goal_set.push_back(kii);
				joinSearch(set_is_visited, goal_set, joinTable);
				goal_sets.push_back(goal_set);
			}
		}

		
		vector<MergeArea> merge_areas;
		// 显示数字融合效果，同时渲染成merge_area
		for (int kii = 0; kii < goal_sets.size(); kii++)
		{
			vector<int> goal_set = goal_sets[kii];
			int goal_set_size = goal_set.size();
			vector<SingleArea> goal_set_areas;
			vector<int> cc_indexs;
			float x_ = 0;
			float y_ = 0;
			for (int kjj = 0; kjj < goal_set_size; kjj++)
			{
				int sas_index = goal_set[kjj];
				circle(roi_merge, sas[sas_index].center, 1, Scalar(0, 0, 0), -1);
				// 先把所有的都标出来
				rectangle(roi_merge, candidates[sas[sas_index].cc_index], Scalar(255, 255, 255), 1);
				cc_indexs.push_back(sas[sas_index].cc_index);
				goal_set_areas.push_back(sas[sas_index]);
				x_ += sas[sas_index].center.x;
				y_ += sas[sas_index].center.y;
			}
			// 计算出新的中心
			Point merge_center = Point(int(x_ / goal_set_size), int(y_ / goal_set_size));
			// 画出中心以及中心与ac_center的连线
			circle(roi_merge, merge_center, 2, Scalar(255, 255, 255), -1);

			// 按照x对goal_set_areas排序，并得出他的总response
			sort(goal_set_areas.begin(), goal_set_areas.end(), SortByX);
			float merge_response = 0;
			for (int kjj = 0; kjj < goal_set_size; kjj++)
			{
				merge_response += goal_set_areas[kjj].response * pow(10, -kjj);
			}

			// 计算与三四象限分割线的轴的夹角并 渲染成 merge_area
			float ddy = -merge_center.y + ac_center.y;
			float ddx = merge_center.x - ac_center.x;
			float merge_angle = getVecAngle(ddy, ddx);
			merge_areas.push_back({ merge_response, merge_angle, cc_indexs, merge_center});

			cout << "merge_angle: " << merge_angle << endl;
			cout << "merge_response: " << merge_response << endl;
			imshow("roi_merge", roi_merge);
			waitKey();

		}







		/* 
			下面是对这些 merge_area 进行筛选，把有些响应值奇怪的点去除掉

			方法是把merge_area看成一个点，其中response是y值，angle是x值，下面还有详细说明

		*/

		// 存储异值点,false表示该点是非奇异值，true表示该点为奇异值，一开始全部都归结为非奇异值
		vector<bool> singulars(merge_areas.size(),false);

		// 允许与拟合直线的距离
		float dis_error = 0.0125;
		// 判断是否奇异点的那个阈值，是一个比例值,这里稍微处理下，如果点比较密集的话，可以放松这个限制
		float singular_thresh = 0.5;
		if (merge_areas.size() >= 8) { singular_thresh = 0.35; }
		if (merge_areas.size() <= 5) { singular_thresh = 0.6; dis_error = 0.04; }


		// 先对merge_areas按照响应值进行排序，响应值大的排前面
		// 然后采用类似RANSAC算法，不同的是，每次归一化都不一样，每次都是把max线性投射到1，如果这个max点是正常点，那么归一化结果没改变，
		// 但是如果这个max点本身就是个异值，说明他过大了，导致其他点看起来响应很小，就需要把她给排出掉，然后对其他点迭代性的进行归一化
		// 归一化后就采用遍历的方法（不是随机，因为我们的点比较少），每次找模型最合理的那条线（穿过的点最多）
		sort(merge_areas.begin(), merge_areas.end(), SortByRes);

		// 下面是把不符合条件的点都放到 singulars 中
		for (int kii = 0; kii < merge_areas.size(); kii++)
		{
			if (kii == 11) 
			{
				cout << "in" << endl;
			}
			// 穿过点最多的那条线所 穿过了多少个点
			int best_accept_num = 0;
			// 对每个点都统计他与其他点所连成的直线是否穿过足够多的点，而是否穿过这个阈值的判定需要依据归一化
			for (int kjj = 0; kjj < merge_areas.size(); kjj++) 
			{
				// 不能连接已经是奇异值的点
				if (kii == kjj || singulars[kjj]) { continue; }
				// 找出目前非奇异值中最大的那个响应值，用它来进行归一化
				float max_res = -1.0;
				for (int kdd = 0; kdd < singulars.size(); kdd++) { if (!singulars[kdd]) { max_res = merge_areas[kdd].response; break; } }

				// 计算与kjj相连的直线穿过其他点的数量，本身直线就已经穿过两个点了，那两个点不用计算
				int accept_num = 2;
				for (int kdd = 0; kdd < singulars.size(); kdd++)
				{
					// 奇异点不在考虑范围
					if (kjj == kdd || kii == kdd || singulars[kdd]) { continue; }
					float ddx = merge_areas[kdd].angle / 360; float ddy = merge_areas[kdd].response / max_res;
					float ddx1 = merge_areas[kii].angle / 360; float ddy1 = merge_areas[kii].response / max_res;
					float ddx2 = merge_areas[kjj].angle / 360; float ddy2 = merge_areas[kjj].response / max_res;
					float m_dis = point2Line(ddx, ddy, ddx1, ddy1, ddx2, ddy2);
					if (m_dis <= dis_error) { accept_num++; };
				}
				// 如果这条连线穿过的点比之前还好，需要替换一下
				if (accept_num >= best_accept_num) { best_accept_num = accept_num; }
			}
			// 统计并判断best_accept_num是否真的满足一定比例，不满足那么他就是奇异值点
			if (best_accept_num < singular_thresh * singulars.size()) { singulars[kii] = true; }
		}



		// 拿出符合条件的所有的点
		vector<MergeArea> merges_1;
		vector<MergeArea> merges;

		cout << "第一次确认后" << endl;
		for (int kii = 0; kii < singulars.size(); kii++)
		{
			if (singulars[kii]) { continue; }
			cout << "满足的响应值为： " << merge_areas[kii].response << endl;
			merges_1.push_back(merge_areas[kii]);
		}


		if (merges_1.size()==0) 
		{
			cout << "!!!!!!!!!!检测角度存在问题，请调整新角度继续检测!!!!!!!!!!!" << endl;
			continue;
		}


		// 按照角度给所有mergeArea排序
		sort(merges_1.begin(), merges_1.end(), SortByAngle);


		// 这里再做多一次修正，sort完之后一定是严格单调递增的，如果出现递减或者不变的情况，那么就需要去除掉了
		merges.push_back(merges_1[0]);
		int merges_pos = 0;
		cout << "第二次确认后" << endl;
		cout << "满足的响应值为： " << merges_1[0].response << endl;
		for (int kii = 1; kii < merges_1.size(); kii++)
		{
			if (merges_1[kii].response <= merges[merges_pos].response) { continue; }
			else { merges.push_back(merges_1[kii]); merges_pos++; cout << "满足的响应值为： " << merges_1[kii].response << endl;}
		}



		// 这里可以添加算法来计算dis_error是否需要再次减小，这里判断下异值点常见的现象，比如是否存在相等的点


		// 消除merge_areas
		// delete merge_areas;

		/*
			给直线所指刻度找范围，如果范围确定在nonsingulars里面，那么就寻找最接近的那个范围，
			如果在nonsingulars左侧（即角度小于已知范围），则用最左侧的gradient去求他应该的数值
			同理右侧
		*/
		// 因为上面已经对merge_areas做角度排序了，所以merge_areas的第一个元素就是角度最小的，最后一个元素就是角度最大的
		float ddy = -lastPoint.y + ac_center.y;
		float ddx = lastPoint.x - ac_center.x;
		float pointerAngle = getVecAngle(ddy, ddx);
		float pointerValue = 0;

		int merge_last = merges.size() - 1;
		if (pointerAngle < merges[0].angle)
		{
			// 寻找最近的那个gradient
			float delta_value = (merges[1].response - merges[0].response) / (merges[1].angle - merges[0].angle) * (merges[0].angle - pointerAngle);
			pointerValue = max(float(0), merges[0].response - delta_value);
		}
		else if (pointerAngle > merges[merge_last].angle)
		{
			int last_0 = merge_last - 1;
			float delta_value = (merges[merge_last].response - merges[last_0].response) / (merges[merge_last].angle - merges[last_0].angle) * (pointerAngle - merges[merge_last].angle);
			pointerValue = merges[merge_last].response + delta_value;
		}
		else 
		{
			// 寻找刚好吻合的那个区域
			for (int kii = 0; kii < merges.size(); kii++)
			{
				if (pointerAngle <= merges[kii].angle)
				{
					float delta_value = (merges[kii].response - merges[kii-1].response) / (merges[kii].angle - merges[kii-1].angle) * (pointerAngle - merges[kii-1].angle);
					pointerValue = merges[kii-1].response + delta_value;
					break;
				}
			}
		}

		cout << "读取得到的数值为： " << pointerValue << endl;
		imshow("roi_merge", roi_merge);
		waitKey();	
		waitKey();
		waitKey();



	}






	int yghds = 0;
	return 0;
}






// 做svmdata用于训练，里面可以考虑翻转增强泛性
// train
int train()
{
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";
	string labelPath = "D:\\VcProject\\biaopan\\data\\labels.txt";
	// 存储 图片存储路径 还有 对应的label
	vector<string> raws = readTxt(labelPath);
	// 存储每张图片的特征向量以及label
	vector<vector<float>> trainingData;

	vector<int> labels;
	// 字符串的分割标识
	const string spliter = "===";
	// 训练和测试过程是完全不一样的，训练是全部扎在一起训练，但是测试的时候需要按照 一个origin一份mser集合，多个origin取平均准确率
	for (int i=0;i< raws.size();i++)
	{
		// 对于一张图片
		vector<string> raw = splitString(raws[i], spliter);
		string src = raw[0];
		int label = str2int(raw[1]);
		Mat mat = imread(src, IMREAD_GRAYSCALE);
		// 稍微给点模糊
		blur(mat, mat, Size(3, 3));
		trainingData.push_back(getHogData(mat));
		labels.push_back(label);
		// 这里做数据增强，比如镜像，翻转（原因是可能会出现这些情况）
		//for (int flipCode=-1; flipCode <2; flipCode++)
		//{
		//	Mat flipMat;
		//	flip(mat, flipMat, flipCode);
		//	trainingData.push_back(getHogData(mat));
		//	labels.push_back(label);
		//}
	}

	//设置支持向量机的参数（Set up SVM's parameters）

	Ptr<cv::ml::SVM> svm = ml::SVM::create();

	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	// svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1200, FLT_EPSILON));
	// svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 200, 1e-6));
	// svm->setDegree(1.0);
	svm->setC(2.67);
	svm->setGamma(5.83);


	// 准备好数据
	// 一张图片的特征向量是144维
	const int feature_length{ 144 };
	const int samples_count{ (int)trainingData.size() };
	if (labels.size() != trainingData.size())
	{
		cout << "数据导出有问题" << endl;
		return 0;
	}

	vector<float> data(samples_count * feature_length, 0.f);
	for (int i = 0; i < samples_count; ++i) {
		for (int j = 0; j < feature_length; ++j) {
			data[i*feature_length + j] = trainingData[i][j];
		}
	}

	Mat trainingDataMat(samples_count, feature_length, CV_32FC1, data.data());
	Mat labelsMat((int)samples_count, 1, CV_32S, (int*)labels.data());

	cout << "开始训练" << endl;
	// 训练
	svm->train(trainingDataMat, ml::ROW_SAMPLE, labelsMat);
	// 生成模型文件
	svm->save(modelPath);

	// 测试
	
	Mat s1 = imread("D:\\VcProject\\biaopan\\data\\goodImgs\\457\\33.jpg", IMREAD_GRAYSCALE);
	imshow("s1", s1);
	waitKey(0);
	vector<float> ss = getHogData(s1);
	Mat testData(1, feature_length, CV_32FC1, ss.data());
	int response = svm->predict(testData);
	cout << "结果是 : " << response << endl;


	return 0;
}

// 下面单体测试结果
// singleTest
int singleTest()
{
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";
	Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
	Mat s1 = imread("D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\msers\\positives\\no_rotation\\120\\10\\7.jpg", IMREAD_GRAYSCALE);

	vector<float> ss = getHogData(s1);
	Mat testData(1, 144, CV_32FC1, ss.data());
	int response = svm->predict(testData);
	cout << "结果是 : " << response << endl;
	cout << "end" << endl;
	return 0;
}

// 算出准确率
int batchTest()
{
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";
	string labelPath = "D:\\VcProject\\biaopan\\data\\labels.txt";

	Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
	// 存储 图片存储路径 还有 对应的label
	vector<string> raws = readTxt(labelPath);

	vector<int> labels;
	// 字符串的分割标识
	const string spliter = "===";
	// 这里只训练28800个数据，其余用来测试
	int totalNum = 0;
	int correct = 0;
	for (int i = 28800; i < raws.size(); i++)
	{
		// 对于一张图片
		vector<string> raw = splitString(raws[i], spliter);
		string src = raw[0];
		int label = str2int(raw[1]);
		Mat mat = imread(src, IMREAD_GRAYSCALE);
		// 存储一个hog向量
		// vector<float> descriptors;//HOG描述子向量
		// descriptors = getHogData(mat);
		// descriptors;
		vector<float> ss = getHogData(mat);
		Mat testData(1, 144, CV_32FC1, ss.data());
		int response = svm->predict(testData);
		if (response == label)
			correct += 1;
		totalNum += 1;
	}
	cout << "准确率是 : " << correct / (float)totalNum << endl;
	cout << "end" << endl;
	return 0;
}



// 同样是做单体数据的测试
int randomTest2() 
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




void writeImg(string imgReadPath, string dirPath, bool if_flip = false)
{

	Mat3b image_1 = imread(imgReadPath);
	// 对图片进行压缩
	resize(image_1, image_1, Size(image_1.size[1] / 2.5, image_1.size[0] / 2.5));
	// 有些图片需要翻转
	if (if_flip) { flip(image_1, image_1, -1); }
	int scaleSize = 2;

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
	Mat1b gray_clone_add = gray_clone.clone();


	Mat1b gray_edge;
	Canny(gray_clone, gray_edge, 3, 9, 3);

	yaed.Detect(gray_edge, ellsYaed);


	// 如果上面的canny算子获取不到椭圆就用下面的方法获取
	if (ellsYaed.size() <= 1)
	{
		adaptiveThreshold(gray_clone_add, gray_clone_add, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 9, 0);
		yaed.Detect(gray_clone_add, ellsYaed);
	}

	if (ellsYaed.size() == 0)
	{
	// 检测不到椭圆，直接跳过
		return;
	}

	Mat3b resultImage = image.clone();



	// 开展搜索范围，以长轴为直径的正方形区域
	int index = 0;
	Mat1b gray_clone2;
	cvtColor(image_1, gray_clone2, CV_BGR2GRAY);
	int el_size = ellsYaed.size();



	// 选取至少有 35 个可能支持点的椭圆
	int min_vec_num = 25;
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
		Mat1b roi_2_add = roi_2.clone();

		adaptiveThreshold(roi_2_add, roi_2_add, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);
		threshold(roi_2, roi_2, 0, 255, CV_THRESH_OTSU);

		Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
		vector<Vec4f> lines_std;
		vector<Vec4f> lines_dst;

		// Detect the lines
		ls->detect(roi_2_add - roi_2, lines_std);

		Vec2f e_center = Vec2f(200, 200);
		// cout << "lines_std.size() : " << lines_std.size() << endl;

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

		if (lines_dst.size() >= min_vec_num && (el_dst._a * el_dst._b) <= (e._a * e._b))
		{
			min_vec_num = lines_dst.size();
			el_dst = e;
			roi_dst = roi_3;
			tLines = lines_dst;
		}
	}




	// 2. hough圆检测获取精确中心，lsd获取最长直线（指针线）

	/**
		中心是(200, 200)，在这个中心以一定半径搜索精准的圆心
	**/
	int searchRadius = 25;
	vector<Vec3f> circles;
	Mat1b centerArea = roi_dst(Rect(200 - searchRadius, 200 - searchRadius, 2 * searchRadius, 2 * searchRadius)).clone();
	Mat1b centerArea2 = centerArea.clone();

	GaussianBlur(centerArea, centerArea, Size(3, 3), 1, 1);

	// 这个局部阈值能使中间部分完全变白，从而可以探测得到
	adaptiveThreshold(centerArea, centerArea, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 9, 0);

	// 下面这种做法是比较冒险的。。因为不保证最后的输出是模糊还是清晰，同时无法确保这个阈值30在各个场景均适用
	// HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 30, 8, 0, 6);
	// 后面四个数字分别代表： 分辨率（可以理解为步进的倒数），两个圆之间最小距离，canny的低阈值，投票的累积次数（即有多少个点属于该圆），圆最小半径，圆最大半径
	// 一般投票值在 15 到 20间，如果拍摄视角特别偏，把中间圆变成椭圆，就需要降低投票值了，目前检测视角比较正常，所以只需要保持20
	HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 150, 18, 10, 25);


	// 实际上上面是可以继续优化的，比如搜索不到任何圆的情况下就尝试扩大搜索域，减少投票次数，放大圆最大半径

	// 寻找离形心最近的那个圆心作为我们的准心，绘制出圆
	Point ac_center = Point(200, 200);
	Point f_center = Point(searchRadius / 2, searchRadius / 2);
	float f2ac = 200;
	for (int i = circles.size() - 1; i >= 0; --i)
	{
		Point mCenter = Point(circles[i][0], circles[i][1]);
		if (circles[i][2] < f2ac)
		{
			f2ac = circles[i][2];
			ac_center = Point(mCenter.x + 200 - searchRadius, mCenter.y + 200 - searchRadius);
		}
	}


	// 3. 上面方法不行就用回以前的 bim+支持点方向分类 然后抽样获取椭圆
	// 给tLine进行分类（按角度分类），比如按照40度一个扇区，就有9个扇区 
	int sangle = 40;
	vector<vector<int>> sangles_(360 / sangle);
	for (int ki = tLines.size() - 1; ki >= 0; --ki)
	{

		// 计算夹角的cos值
		int xd = tLines[ki][2] - ac_center.x;
		int yd = ac_center.y - tLines[ki][3];
		// 值域在 0~360之间
		float vangle = fastAtan2(yd, xd);
		sangles_[(int)vangle / sangle].push_back(ki);
	}
	// 这里要过滤要一些没有点的扇区
	vector<vector<int>> sangles;
	for (int ki = 0; ki < sangles_.size(); ki++)
	{
		vector<int> sangel_ = sangles_[ki];
		if (sangel_.size() > 0)
		{
			sangles.push_back(sangel_);
		}
	}


	int asize = sangles.size();
	int ranTimes = 1000;
	// 允许椭圆拟合中支持点最大的远离程度，这里的程度是乘上椭圆长轴的
	float acceptThresh = 8;
	// 存储当前的支持点个数
	int nowSnum = 0;
	// 存储最多支持点的那个椭圆的外接矩形
	RotatedRect bestEl;
	// 存储最终的支持点
	vector<int> bestSupportPoints;
	// 这里设置随机的次数
	for (int ri = 0; ri < ranTimes; ri++)
	{
		// 存储要选取的扇区，从5个扇区中取值
		vector<int> ks;
		// 给定初始的点
		int kindex = random(asize - 1);
		ks.push_back(kindex);
		for (int ki = 0; ki < 4; ki++)
		{
			// 产生1~2的随机数，也就是所选取的扇区，后选的扇区比前选的扇区最多隔两个位置，保证有一定的角度
			kindex = (kindex + random(2)) % asize;
			ks.push_back(kindex);
		}
		// 从每个所选的扇区中随机选取一个点进行椭圆拟合
		vector<Point> candips;
		for (int ki = 0; ki < 5; ki++)
		{
			// ks[ki]指随机选取到的扇区的序号，sangles[扇区序号]拿到的就是这个扇区里面的所有点的序号，所以这里最终随机产生的就是扇区里面的第几个点
			vector<int> shanqu = sangles[ks[ki]];
			int sii = random(shanqu.size() - 1);
			Vec4f vvvv = tLines[shanqu[sii]];
			Point ssss = Point(vvvv[2], vvvv[3]);
			candips.push_back(ssss);
		}
		// 拟合椭圆后计算支持点个数
		RotatedRect rect = fitEllipse(candips);

		// 存储支持点
		vector<int> support_points;
		for (int ki = 0; ki < tLines.size(); ki++)
		{
			Point sp = Point(tLines[ki][2], tLines[ki][3]);
			Point newsp = origin2el(rect.center, rect.angle / (float)180 * pi, sp);

			float edistance = abs(sqrt(pow(newsp.x, 2) / pow(rect.size.width / 2, 2) + pow(newsp.y, 2) / pow(rect.size.height / 2, 2)) - 1) * max(rect.size.width / 2, rect.size.height / 2);

			if (edistance <= acceptThresh)
			{
				support_points.push_back(ki);
			}
		}


		if (support_points.size() >= nowSnum)
		{
			nowSnum = support_points.size();
			bestSupportPoints = support_points;
			bestEl = rect;
		}

	}
	// 经过上面的循环就能得出支持点最多的那个外接椭圆，下面显示出来

	// 对拿到的支持点继续进行拟合
	vector<Point> supporters;
	for (int ki = 0; ki < bestSupportPoints.size(); ki++)
	{
		int tLineIndex = bestSupportPoints[ki];
		Point supporter = Point(tLines[tLineIndex][2], tLines[tLineIndex][3]);
		supporters.push_back(supporter);
	}
	bestEl = fitEllipse(supporters);




	// 下面测试用 自动阈值 来mser，经过试验发现，这个方法切割比直接mser好
	Mat roi_thresh = roi_dst.clone();
	Mat roi_thresh_mser = roi_dst.clone();
	adaptiveThreshold(roi_thresh, roi_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);


	//获取自定义核
	Mat e_element = getStructuringElement(MORPH_ELLIPSE, Size(1, 1)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的

	erode(roi_thresh, roi_thresh, e_element);



	// mser检测
	vector<RotatedRect> rrects;
	std::vector<cv::Rect> candidates = mser(roi_thresh, rrects);

	int file_index = 0;
	for (int i = 0; i < candidates.size(); ++i)
	{

		// 检测合法性
		if (candidates[i].x <= 0 || candidates[i].y <= 0 || candidates[i].width <= 0 || candidates[i].height <= 0 
			|| candidates[i].x + candidates[i].width > roi_dst.cols || candidates[i].y + candidates[i].height > roi_dst.rows)
		{
			continue;
		}

		// 筛选掉拟合圆外面的点
		Point ncenter = Point(candidates[i].x + candidates[i].width / 2, candidates[i].y + candidates[i].height / 2);
		Point newsc = origin2el(bestEl.center, bestEl.angle / (float)180 * pi, ncenter);
		float ndistance = sqrt(pow(newsc.x, 2) / pow(bestEl.size.width / 2, 2) + pow(newsc.y, 2) / pow(bestEl.size.height / 2, 2));
		// 这里的椭圆需要往内部缩一下，去除过多的咋点，比如刻度线
		if (ndistance >= 0.86) { continue; }



		// 把符合的区域存储起来

		Mat mser_item = roi_dst(candidates[i]);
		file_index += 1;
		_mkdir(dirPath.c_str());
		// 写最终结果
		imwrite(dirPath + "\\" + int2str(file_index) + ".jpg", mser_item);
		// 把原图也给写进去
		imwrite(dirPath + "\\" + "origin.jpg", image_1);
	}

}


// 制作数据的单体测试
// writeSingleImg
int writeSingleImg()
{
	vector<string> aa;

	aa.push_back("13 04");

	for (int i=0;i<=aa.size()-1;i++) 
	{
		cout << "当前准备的是： " << aa[i] << endl;
		writeImg("D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\13\\"+aa[i]+".jpg", "D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\msers\\1");
	}
	return 0;
}



// 制作mser数据
// makeMserData
int makeMserData()
{

	
	//glob(images_folder + "Lo3my4.*", names);

	string outputPath = "D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\msers\\";
	string basePath = "D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\";
	string sorted_imgs_path = "D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\sorted_imgs.txt";
	// 下面的路径存储的是正样本的数据，这是为了自己标定时候更加方便而做的
	string positive_base = "D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\msers\\positives\\";
	vector<string> names;

	ofstream out(outputPath + "break_points.txt", ios::app);

	vector<string> alls = readTxt(sorted_imgs_path);
	int rotation_folderIndex = 0;
	int no_rotation_folderIndex = 0;
	int noise_folderIndex = 0;
	// 595前的都是正常，595 到 913 后面的一部分是要翻转的
	for (int i = 0; i < alls.size(); i++)
	{

		string& s = alls[i];
		// D:\VcProject\biaopan\data\raw\newData\images\newData\4\4 48.jpg===1
		vector<string> ss = splitString(s, "===");
		string imgReadPath = ss[0];
		int label = str2int(ss[1]);
		string dirPath = outputPath;
		string addtion_part = "";
		if (label == -1)
		{
			no_rotation_folderIndex += 1;
			addtion_part = "no_rotation\\" + int2str(no_rotation_folderIndex) + "\\";
		}
		if (label == 1)
		{
			rotation_folderIndex += 1;
			addtion_part = "rotation\\" + int2str(rotation_folderIndex) + "\\";

		}
		if (label == 2)
		{
			noise_folderIndex += 1;
			addtion_part = "noise\\" + int2str(noise_folderIndex) + "\\";
		}
		cout << "======================================================" << endl;
		cout << "如果需要更改请修改以下数值" << endl;
		cout << "当前是i是 -- " << i << endl;
		cout << "当前是no_rotation_folderIndex是 -- " << no_rotation_folderIndex << endl;
		cout << "当前是noise_folderIndex是 -- " << noise_folderIndex << endl;
		cout << "当前是rotation_folderIndex是 -- " << rotation_folderIndex << endl;
		cout << "当前是path是 -- " << imgReadPath << endl;

		// 向断点文件中写入断点

		if (out.is_open())
		{
			out << "======================================================" << endl;
			out << "rotation_folderIndex = " << int2str(rotation_folderIndex) << endl;
			out << "no_rotation_folderIndex = " << int2str(no_rotation_folderIndex) << endl;
			out << "noise_folderIndex = " << int2str(noise_folderIndex) << endl;
			out << "i = " << int2str(i) << endl;
		}
		dirPath = dirPath + addtion_part;
		// 先创建9个文件夹的父亲文件夹
		_mkdir((positive_base + addtion_part).c_str());
		// 创建9个空文件夹
		for (int b_i = 0; b_i < 10; b_i++)
		{
			string folder_a_name = positive_base + addtion_part + int2str(b_i);
			cout << "folder_a_name : " << folder_a_name << endl;
			_mkdir(folder_a_name.c_str());
		}
		if (i >= 595 && i <= 913)
		{
			writeImg(imgReadPath, dirPath, true);
		}
		else 
		{
			writeImg(imgReadPath, dirPath);
		}
	}
	out.close();
	return 0;
}