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


// ����㵽ֱ�ߵľ��룬��������ĵ���ͼ������ĵ�
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


// ����㵽ֱ�ߵľ��룬����ĵ��Ǳ�׼����ĵ�
float point2Line(float x, float y, float x1, float y1, float x2, float y2)
{

	float A = (y1 - y2) / (x1 - x2);
	float B = -1;
	float C = y1 - x1 * A;
	float dis = abs(A * x + B * y + C) / sqrt(pow(A, 2) + pow(B, 2));
	return dis;
}


// ����㵽��ľ���
float point2point(float x1, float y1, float x2, float y2)
{
	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}
// ����㵽��ľ���
float point2point(int x1, int y1, int x2, int y2)
{
	return sqrt(float(pow((x1 - x2), 2) + pow((y1 - y2), 2)));
}

// ����㵽��ľ���
float point2point(Point point1, Point point2)
{
	return sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2));
}

// ����㵽��ľ���
float point2point(Vec4f line)
{
	return sqrt(pow((line[0] - line[2]), 2) + pow((line[1] - line[3]), 2));
}

// ������ֱ�߼�ļн�cosֵ
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
	// ���������Ǹ����εĴ󲿷����������ǰ���Ǹ����εģ���ô����ľ��ο���ɾ��
	double o = inter / r2.area();
	return (o >= 0) ? o : 0;
}

void nms(vector<Rect>& proposals, const double nms_threshold)
{
	// ����ķ����õ������
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

// �Ӹ�������ȥ
void nms2(vector<Rect>& proposals,vector<int>& indexes, const double nms_threshold)
{
	// ����ķ����õ������
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




// ������ת����
void drawRotatedRect(Mat& drawer, RotatedRect& rrect)
{
	// ��ȡ��ת���ε��ĸ�����
	cv::Point2f* vertices = new cv::Point2f[4];
	rrect.points(vertices);
	//�����߻���
	for (int j = 0; j < 4; j++)
	{
		// ���������ĸ���ı߽ǣ���ɫ��p[3]����ɫ��p[1]
		circle(drawer, vertices[0], 2, Scalar(225, 225, 225), -1);
		circle(drawer, vertices[3], 2, Scalar(0, 0, 0), -1);
		cv::line(drawer, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 0, 0));
	}
}

// MserĿ���� + nms������rrects�洢ÿ��Rect�е��Ǹ���ת����
std::vector<cv::Rect> mser(cv::Mat srcImage, vector<cv::RotatedRect>& rrects)
{

	std::vector<std::vector<cv::Point> > regContours;
	
	// ����MSER����
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 5, 800, 0.5, 0.3);


	std::vector<cv::Rect> boxes;
	// MSER���
	mesr1->detectRegions(srcImage, regContours, boxes);
	// �洢����
	std::vector<Rect> keeps;


	cv::Mat mserMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);
	cv::Mat mserNegMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);

	Mat mser_show = srcImage.clone();

	vector<RotatedRect> rrecs;
	for (int i = (int)regContours.size() - 1; i >= 0; i--)
	{

		// ���ݼ�����������mser���
		std::vector<Point> hull;
		// ����͹��
		convexHull(regContours[i], hull);

		// ������Ϊ�˼��Ƕȶ���ġ�
		// mser������������ͨ������е�
		// ����͹������ԭ���ĵ㣬�������ᷢ����������ͦ����ģ���������ճ�������Ҳ�����ˣ�������Ҫ����Ƕ�Ȼ��ͨ��ͶӰ������
		// polylines(mser_show, regContours[i], 1, Scalar(255, 0, 0));
		// ��ȡ��С��Χ����
		RotatedRect rotatedRect = minAreaRect(regContours[i]);
		rrecs.push_back(rotatedRect);

		drawRotatedRect(mser_show, rotatedRect);



		Rect br = boundingRect(hull);
		// ��߱���
		double wh_ratio = br.height / double(br.width);
		// ���
		int b_size = br.width * br.height;
		// �����ϳߴ������ж�
		if (b_size < 800 && b_size > 50)
		{
			// ʵ��֤�����������ŵ�ʱ��ʶ��Ч�����ã�
			br = Rect(br.x - 3, br.y - 3, br.width + 6, br.height + 6);
			keeps.push_back(br);
		}
		// ��΢�÷���������һ��
		
		//keeps.push_back(br);
	}

	imshow("mser_show", mser_show);
	waitKey(0);
	// ��nms����
	nms(keeps, 0.7);


	mser_show = srcImage.clone();
	// �ҳ�ÿ��keep�еľ�������������ת����
	vector<cv::RotatedRect> rrects_;
	for (int j = 0; j < keeps.size(); j++)
	{
		float karea = keeps[j].width * keeps[j].height;
		rectangle(mser_show, keeps[j], Scalar(255, 255, 255), 2);
		RotatedRect krec;
		float max_size = 0.2;
		for (int i = 0; i < rrecs.size(); i++)
		{
			
			//��ȡ��ת���ε��ĸ�����
			cv::Point2f* vertices = new cv::Point2f[4];
			rrecs[i].points(vertices);
			
			// �����������ε��ص���
			//float area = point2point(vertices[0], vertices[1]) * point2point(vertices[0], vertices[3]);

			if (rrecs[i].center.x >= keeps[j].x && rrecs[i].center.y >= keeps[j].y 
				&& rrecs[i].center.x <= keeps[j].x+ keeps[j].width && rrecs[i].center.y <= keeps[j].y + keeps[j].height)
			{
				//float area = point2point(vertices[0], vertices[1]) * point2point(vertices[0], vertices[3]);
				float area = rrecs[i].size.height * rrecs[i].size.width;
				// ȡ����������һ����ֵ����Ϊ�жϷ����׼��
				if (area / karea >= max_size && area / karea <= 0.8)
				{
					max_size = area / karea; 
					krec = rrecs[i];
				}
			}
		}
		// ��ʹû���µľ��Σ�ҲҪ�Ž�ȥһ���Ѿ��еġ�
		rrects_.push_back(krec);
		// ���max_sizeû�б仯��˵��û������Ҫ����ڽ���ת����
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
	return stream.str();   //�˴�Ҳ������ stream>>string_temp  
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
	infile.open(file.data());   //���ļ����������ļ��������� 
	assert(infile.is_open());   //��ʧ��,�����������Ϣ,����ֹ�������� 

	vector<string> names;
	string s;
	while (getline(infile, s))
	{
		names.push_back(s);
	}
	infile.close();             //�ر��ļ������� 
	return names;
}

// �ָ��ַ���
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



// ��ȡһ��cell��������dn��ָ�ж��ٸ�����
vector<float> getCellData(Mat& mag, Mat& angle, int r, int c, int cellSize, int dn)
{
	vector<float> cell(dn, 0);
	float tangle = 360 / (float)dn;

	for (int k = r; k < r + cellSize; k++)
	{
		// ÿһ��ͼ���ָ��
		const float* magData = mag.ptr<float>(k);
		const float* angleData = angle.ptr<float>(k);
		for (int i = c; i < c + cellSize; i++)
		{
			// floor ������ȡ��
			// cout << angleData[i] << endl;		
			// cout << magData[i] << endl;
			cell[floor(angleData[i] / tangle)] += magData[i];
		}
	}
	return cell;
}


// ��ȡhog����
vector<float> getHogData(Mat& originImg)
{
	Mat img;
	// ����resize
	resize(originImg, img, Size(18, 36));
	// ������Կ��ǹ�һ����Ҳ���ǵ�����������������Ϊ1/255��ʹ0~255ӳ���0��1
	img.convertTo(img, CV_32F, 1);
	Mat gx, gy;
	Sobel(img, gx, CV_32F, 1, 0, 1);
	Sobel(img, gy, CV_32F, 0, 1, 1);

	Mat mag, angle;
	cartToPolar(gx, gy, mag, angle, 1);


	// ��ÿ��cell������ֱ��ͼͳ��
	vector<vector<float>> cells;
	int cellSize = 9;
	int directionNum = 12;
	for (int i = 0; i < 4; i++)
	{
		cells.push_back(getCellData(mag, angle, i * 9, 0, cellSize, directionNum));
		cells.push_back(getCellData(mag, angle, 9, i * 9, cellSize, directionNum));
	}

	// ��3��block�����ϳ�һ��vector
	vector<float> hogData;
	// ÿ�ĸ�cell��һ��block�����������
	// ��һ���ǿ��Ƶڼ���block
	for (int i = 0; i < 3; i++)
	{
		// �洢ÿ��block��vector
		vector<float> v;
		float total = 0;
		// �ڶ����ǿ���block�еĵڼ���cell
		for (int j = i * 2; j < i * 2 + 4; j++)
		{
			// ����ÿ��cell�����ÿ��floatֵ
			for (int k = 0; k < cells[j].size(); k++)
			{
				// ����һ��block�����L2ģ�⻹��Ҫ���ĸ�vector������һ��
				total += pow(cells[j][k], 2);
				v.push_back(cells[j][k]);
			}
		}
		// ����֮ǰ�����block�����L2ģ����֮ǰpush��ȥ�Ľ��й�һ��
		for (int e = 0; e < v.size(); e++)
		{
			hogData.push_back(v[e] / sqrt(total));
		}
	}
	return hogData;
}







// ����������ת��ĵ㣬��������������Ǳ�׼����ϵ�������Ҳ�Ǳ�׼����ϵ
Point rotate(float theta, float x, float y)
{
	return Point(cos(theta)*x - sin(theta)*y, sin(theta)*x + cos(theta)*y);
}

// �л�����Բ���꣬���������������ͼ������ϵ��������Ǳ�׼����ϵ����ƽ�ƺ���ת��
Point origin2el(Point2f& center, float theta, Point& origin)
{
	float x = origin.x;
	float y = -origin.y;
	return rotate(theta, x-center.x, y+center.y);
}


// �л���ͼ�����꣬��������������Ǳ�׼����ϵ���������ͼ������ϵ������ת��ƽ�ƣ�
Point el2origin(Point2f& center, float theta, Point& el)
{
	Point origin = rotate(theta, el.x, el.y);
	float x = origin.x;
	float y = -origin.y;
	return Point(x + center.x, y + center.y);
}


// �����ǲ�����ת����
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



// �����һ����������Բ��, a����Բ�ĵ�һ���᳤��b����Բ�ĵڶ����᳤��theta����Բ����б�ǣ�xx1 �� xx2��������Բ�ཻ��ֱ�ߣ����շ��ص��� ��xx1xx2����ͬ����Ľ���(p1�ǳ�����)
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
		// ���ҪתΪͼ���������
		origin = el2origin(center, -theta, el);
	}
	if (delta > 0)
	{

		float x1 = (-B + sqrt(delta)) / 2 / A;
		float x2 = (-B - sqrt(delta)) / 2 / A;
		float y1 = (k * x1 + c);
		float y2 = (k * x2 + c);

		// p1p2�����������ж��Ƿ�ͬ����
		Vec2f v0 = Vec2f(newx2.x - newx1.x, newx2.y - newx1.y);
		Vec2f v1 = Vec2f(x1 - newx1.x, y1 - newx1.y);
		Vec2f v2 = Vec2f(x2 - newx1.x, y2 - newx1.y);
		float d1 = abs((v0[0] * v1[0] + v0[1] * v1[1]) / sqrt(pow(v0[0], 2) + pow(v0[1], 2)) / sqrt(pow(v1[0], 2) + pow(v1[1], 2)) - 1);
		float d2 = abs((v0[0] * v2[0] + v0[1] * v2[1]) / sqrt(pow(v0[0], 2) + pow(v0[1], 2)) / sqrt(pow(v2[0], 2) + pow(v2[1], 2)) - 1);
		if (d1 <= d2)
		{
			el = Point(x1, y1);
			// ���ҪתΪͼ���������
			origin = el2origin(center, -theta, el);
		}
		else
		{
			el = Point(x2, y2);
			// ���ҪתΪͼ���������
			origin = el2origin(center, -theta, el);
		}


	}
	return origin;
}




// ������Բ����
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


// ���Թ���mser��nms�ܷ���ȡ�����еĿ̶���
int main__()
{
	string image_name = "D:\\VcProject\\biaopan\\data\\test\\2\\aa2.jpg";
	Mat image_1 = imread(image_name);

	// mser���
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
// �ϲ�ֱ�ߵĲ���

// 1. ������
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


// 2. ������
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
// �ϲ���������Ĳ���

// 1. ���ӵ�����
void joinSearch(vector<bool>& isVisited, vector<int>& goal_set, vector<vector<int>>& joinTable)
{
	vector<int>::iterator it;
	int i = goal_set[goal_set.size()-1];
	isVisited[i] = true;
	// -2��ʾ�Խ��ߣ��Լ����Լ�����-1��ʾû�����ӣ����ڵ���0��ֵ��ʾ�������ӵ�ֱ�ߵ���б��
	for (int j = 0; j < joinTable.size(); j++)
	{
		if (i == j) { continue; }
		if (joinTable[i][j] > -1) 
		{
			it = find(goal_set.begin(), goal_set.end(), j);
			// ���goal_set����ԭ�����������ľͲ��ù�������ˣ�����Ѱ��������Ҫ�ĵ�
			if (it != goal_set.end()) { continue; }
			// ������������ӽ�����ͬʱ����ȥ����һ����
			goal_set.push_back(j);
			joinSearch(isVisited, goal_set, joinTable);
		}
	}
	

}

// 2.�洢ȷ�������ֵ������Լ����ǵ����ĵ㻹�����ǵ���Ӧ
struct SingleArea
{
	// �洢��Ӧֵ
	float response;
	// �洢�����candidates�����
	int cc_index;
	// �洢����
	Point center;
};

// 3.�洢�ںϵ�����
struct MergeArea
{
	// �洢��Ӧֵ
	float response;
	// �洢�����ĵ���ת�Ƕ�(���������޷ָ���Ϊ��ʼ��ת����)
	float angle;
	// �洢�����candidates�����
	vector<int> cc_indexs;
	// �洢����
	Point center;

};

// 4.�жϽǶȣ�����ı�׼����ϵ�������� y �� x��������� �Ա�׼����ϵ�е����������м���Ϊ0���ᣬ˳ʱ����ת�ĽǶ�
float getVecAngle(float dy, float dx)
{
	float vecAngle = atan2(dy, dx) * 180 / pi;
	if (vecAngle <= -90) { vecAngle = -vecAngle - 90; }
	else if (vecAngle >= 0) { vecAngle = -vecAngle + 270; }
	else if (vecAngle < 0 && vecAngle > -90) { vecAngle = -vecAngle + 270; }
	return vecAngle;
}

// 4.vector<SingleArea>����center.xֵ������
bool SortByX(SingleArea &v1, SingleArea &v2)//ע�⣺�������Ĳ���������һ��Ҫ��vector��Ԫ�ص�����һ��  
{
	//��������  
	return v1.center.x < v2.center.x;
}


// 5.vector<MergeArea>����angleֵ������
bool SortByAngle(MergeArea &v1, MergeArea &v2)//ע�⣺�������Ĳ���������һ��Ҫ��vector��Ԫ�ص�����һ��  
{
	//��������  
	return v1.angle < v2.angle;
}

// 6.vector<MergeArea>����responseֵ�����򣬽���
bool SortByRes(MergeArea &v1, MergeArea &v2)//ע�⣺�������Ĳ���������һ��Ҫ��vector��Ԫ�ص�����һ��  
{
	//��������  
	return v1.response > v2.response;
}

/* ------------------------------------ */



/* ------------------------------------ */
// vector<Ellipse>������Բ�������Сֵ�����򣬽���
bool SortByEllipseArea(Ellipse &e1, Ellipse &e2)//ע�⣺�������Ĳ���������һ��Ҫ��vector��Ԫ�ص�����һ��  
{
	//��������  
	return (e1._a*e1._b) > (e2._a*e2._b);
}



// ɸѡ��Բ�Ĺ����ŵ������������һ�в����������
void getGoodElls(vector<Ellipse>& ellsYaed, float& el_ab_p, Mat& resultImage, 
	int& scaleSize, Mat& gray_clone2, Ptr<LineSegmentDetector>& ls, Vec2f& e_center, float& dis2e_center, int& min_vec_num,
	Ellipse& el_dst, Mat& roi_dst, vector<Vec4f>& tLines, Mat& bl_drawLines)
{

	if (ellsYaed.size() == 0) { return; }
	// �Ȱ��������С������Щ��Բ
	// Ȼ���ٴӴ������Ѱ������һ��Ҫ�����Բ
	sort(ellsYaed.begin(), ellsYaed.end(), SortByEllipseArea);

	int index = 0;
	while (index < ellsYaed.size()) {
		
		Ellipse& e = ellsYaed[index];
		index += 1;
		int g = cvRound(e._score * 255.f);
		Scalar color(0, g, 0);
		// �ҵ��������
		int long_a = e._a >= e._b ? cvRound(e._a) : cvRound(e._b);
		int short_b = e._a < e._b ? cvRound(e._a) : cvRound(e._b);

		// �������ȹ�С����ȥ
		cout << "�����ǣ�" << short_b / (float)long_a << endl;
		if (short_b / (float)long_a < el_ab_p) { continue; }

		rectangle(resultImage, Rect(cvRound(e._xc) - long_a, cvRound(e._yc) - long_a, 2 * long_a, 2 * long_a), color, 1);

		// �ҵ���ԭͼ�е�λ�ã�Ȼ���Ƚ���ֱ��ͼ���⻯�����lsd
		// �������ھ������⣬����� e._xc ��С�� long_a�����߾��εķ�Χ������ͼƬԭ���ߴ�
		int r_x = max(0, (cvRound(e._xc) - long_a)) * scaleSize;
		int r_y = max(0, (cvRound(e._yc) - long_a)) * scaleSize;
		// �����ߴ�Ļ����ʵ���С
		int r_mx = min(gray_clone2.cols, r_x + 2 * long_a * scaleSize);
		int r_my = min(gray_clone2.rows, r_y + 2 * long_a * scaleSize);
		int n_width = min(r_mx - r_x, r_my - r_y);
		Mat1b roi = gray_clone2(Rect(r_x, r_y, n_width, n_width));
		Mat1b roi_2;
		resize(roi, roi_2, Size(400, cvRound(float(roi.cols) / float(roi.rows) * 400)));
		// ͬʱҲ����һ����Բ
		float scaleFactor = float(400) / 2 / long_a;
		// �����������ʱ��ʵ���Ͼ͹涨������� 400 x 400 ��ͼ������Բ���ľ���(200, 200)
		e._xc = 200;
		e._yc = 200;
		e._a = e._a * scaleFactor;
		e._b = e._b * scaleFactor;

		Mat1b roi_3 = roi_2.clone();
		// imwrite("D:\\VcProject\\biaopan\\data\\test\\2\\aa1.jpg", roi_2);
		equalizeHist(roi_2, roi_2);
		Mat1b roi_2_add = roi_2.clone();
		// imwrite("D:\\VcProject\\biaopan\\data\\test\\2\\aa2.jpg", roi_2);

		// ����LSD�㷨���ֱ��
		// ʵ�鷢�����Զ���ֵ��canny���õõ��̶��ߣ������ 9=3*3ģ�壬25=5*5ģ�壬����Ҳ���� ADAPTIVE_THRESH_MEAN_C �� ADAPTIVE_THRESH_GAUSSIAN_C�ĸ����׼�⵽ֱ��
		//adaptiveThreshold(roi_2, roi_2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 9, 0);


		adaptiveThreshold(roi_2_add, roi_2_add, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);
		threshold(roi_2, roi_2, 0, 255, CV_THRESH_OTSU);
		imshow("thresh_result_1", roi_2);
		imshow("thresh_result_2", roi_2_add);
		// ����ʵ���ϣ����ڱ���Ӧ�ã��������������õ�������Ч��
		imshow("thresh_result_3", roi_2_add - roi_2);
		waitKey(0);


		// gray_clone, gray_edge, 3, 9, 3
		// Canny(roi_2, roi_2, 3, 9, 3); // Apply canny edge//��ѡcanny����


		vector<Vec4f> lines_std;
		vector<Vec4f> lines_dst;

		// Detect the lines
		ls->detect(roi_2_add - roi_2, lines_std);
		// Show found lines
		Mat drawnLines = roi_3.clone();



		cout << "lines_std.size() : " << lines_std.size() << endl;

		for (int j = 0; j < lines_std.size(); j++)
		{
			// ɸѡ����Щ�������ıȽ�Զ����
			float distance = point2Line(e_center, lines_std[j]);
			if (distance <= dis2e_center)
			{
				Vec4f l = lines_std[j];
				// ��Ҫ��ͷβ����
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
				// ����β��
				circle(drawnLines, Point(dl[2], dl[3]), 2, color, -1);
			}
		}
		circle(drawnLines, Point(200, 200), 4, color, -1);

		// �������г����е����
		ls->drawSegments(drawnLines, lines_dst);
		cout << "size: " << lines_dst.size() << endl;



		


		// ������ѡ�� ����֧�ֵ� ���� 35 �ģ�һ���ҵ��˾Ͳ���Ҫ�ټ������ˣ���Ϊǰ���Ѿ�ȷ�������㹻�����Բ�ˣ��������������ˣ�

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
// MserĿ���� + nms
std::vector<cv::Rect> mser2(cv::Mat srcImage)
{


	std::vector<std::vector<cv::Point> > regContours;

	// ����MSER����
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 300, 10000, 0.5, 0.3);


	std::vector<cv::Rect> boxes;
	// MSER���
	mesr1->detectRegions(srcImage, regContours, boxes);
	// �洢����
	std::vector<Rect> keeps;


	cv::Mat mserMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);
	cv::Mat mserNegMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);

	for (int i = (int)regContours.size() - 1; i >= 0; i--)
	{
		// ���ݼ�����������mser���
		std::vector<Point> hull;
		convexHull(regContours[i], hull);
		Rect br = boundingRect(hull);
		// ��߱���
		double wh_ratio = br.width / double(br.height);
		// ���
		int b_size = br.width * br.height;
		// �����ϳߴ������ж�
		keeps.push_back(br);

		//keeps.push_back(br);
	}
	// ��nms����
	nms(keeps, 0.5);
	return  keeps;
}
/* ------------------------------------ */



/* ------------------------------------ */
// ������Ǹ������������ɣ����ڲ��Ի���
//���ɸ�˹����
double generateGaussianNoise(double mu, double sigma)
{
	//����Сֵ
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flagΪ�ٹ����˹�������X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//�����������
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flagΪ�湹���˹�������
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0 * sigma + mu;
}

//Ϊͼ������˹����
Mat addGaussianNoise(Mat &srcImag)
{
	Mat dstImage = srcImag.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols*channels;
	//�ƶ�ͼ���������
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//�����˹����
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
	string image_name = "F:\\chrome����\\��ҵ\\��ҵ�Ӿ�\\dataset\\circle\\2.jpg";
	
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


// ���һ��ͼƬ�ĸ��ֲ�����main����
// main
int main()
{
	// ���������
	srand((unsigned)time(NULL));
	string images_folder = "D:\\VcProject\\biaopan\\imgs\\";
	string out_folder = "D:\\VcProject\\biaopan\\imgs\\";
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";
	// ����svm
	Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
	vector<string> names;

	//glob(images_folder + "Lo3my4.*", names);
	// 1 85����Ҫ����


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

		// ��ͼƬ����ѹ��
		//resize(image_1, image_1, Size(image_1.size[1] / 2.5, image_1.size[0] / 2.5));
		//resize(image_1, image_1, Size(image_1.size[1] / 8, image_1.size[0] / 8));
		blur(image_1, image_1, Size(7, 7));
		imshow("blur_image", image_1);
		waitKey();
		// ��ͼƬ�������
		//image_1 = addGaussianNoise(image_1);

		// ��ЩͼƬ��Ҫ��ת
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



		// ��չ������Χ���Գ���Ϊֱ��������������
		
		Mat1b gray_clone2;
		cvtColor(image_1, gray_clone2, CV_BGR2GRAY);

		// ѡȡ������ 35 ������֧�ֵ����Բ
		int min_vec_num = 25;
		// �洢Ŀ����Բ
		Ellipse el_dst;
		// �洢Ŀ������
		Mat1b roi_zero = Mat::zeros(400, 400, CV_8UC1);
		Mat1b& roi_dst = roi_zero;
		// �洢Ŀ��Ŀ���֧����
		vector<Vec4f> tLines;
		// ����ֱ�߾���e_center�ľ���
		float dis2e_center = 30;
		// ������Բ����С�����
		float el_ab_p = 0.75;

		Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
		Mat bl_drawLines;
		Vec2f e_center = Vec2f(200, 200);

		getGoodElls(ellsYaed, el_ab_p, resultImage,scaleSize, gray_clone2, ls, e_center, dis2e_center, min_vec_num,
			el_dst, roi_dst, tLines, bl_drawLines);

		// ��������canny���ӻ�ȡ������Բ��������ķ�����ȡ
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
			cout << "��ⲻ����Բ���밴������˳�����" << endl;
			cout << "--------------------------------" << endl;
			system("pause");
			return 0;
		}

		
		// ������ʾ������õ��Ǹ���Բ�Ļ���
		imshow("drawnLines", bl_drawLines);


		// ��ʾ��ԲЧ��
		int g = cvRound(el_dst._score * 255.f);
		Scalar color(0, 255, 255);
		//ellipse(roi_dst, Point(cvRound(el_dst._xc), cvRound(el_dst._yc)), Size(cvRound(el_dst._a), cvRound(el_dst._b)), el_dst._rad*180.0 / CV_PI, 0.0, 360.0, color, 2);
		
		Mat roi_center = roi_dst.clone();
		Mat roi_line = roi_dst.clone();
		Mat roi_mser = roi_dst.clone();
		Mat roi_merge = roi_dst.clone();


		// ��������� �Զ���ֵ ��mser���������鷢�֣���������и��ֱ��mser��
		Mat roi_thresh = roi_dst.clone();
		Mat roi_thresh_otsu = roi_dst.clone();
		Mat roi_thresh_mser = roi_dst.clone();
		// ʵ��֤�� ADAPTIVE_THRESH_GAUSSIAN_C �� ADAPTIVE_THRESH_MEAN_C �ָ�ĸ��ã���Ե���ӹ⻬
		adaptiveThreshold(roi_thresh, roi_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);
		//erode(roi_thresh, roi_thresh, Mat(2, 2, CV_8U), Point(0, 0), 1);
		//medianBlur(roi_thresh, roi_thresh, 3);


		//threshold(roi_thresh_otsu, roi_thresh_otsu, 0, 255, CV_THRESH_OTSU);
		//roi_thresh = roi_thresh - roi_thresh_otsu;

		//��ȡ�Զ����
		Mat e_element = getStructuringElement(MORPH_ELLIPSE, Size(4, 4)); //��һ������MORPH_RECT��ʾ���εľ���ˣ���Ȼ������ѡ����Բ�εġ������͵�
		// Mat d_element = getStructuringElement(MORPH_ELLIPSE, Size(2, 2)); //��һ������MORPH_RECT��ʾ���εľ���ˣ���Ȼ������ѡ����Բ�εġ������͵�
		// ���Ͱ����е����ֶ��ٳ���
		//dilate(roi_thresh, roi_thresh, e_element);

		imshow("roi_thresh", roi_thresh);
		waitKey(0);

		
		//erode(roi_thresh, roi_thresh, e_element);
		//dilate(roi_thresh, roi_thresh, e_element);
		// ������
		//morphologyEx(roi_thresh, roi_thresh, MORPH_OPEN, e_element);


		// ���波�Ը�ʴ��Σ��ó�������Ҫ��ָ���ֱ�ߣ�������ͨ��֮ǰ�Ĳ�������
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
		
		// ��������β��ͷ�β��
		for (int j = 0; j < tLines1.size(); j++)
		{
			// ɸѡ����Щ�������ıȽ�Զ����
			float distance = point2Line(e_center, tLines1[j]);

			if (distance <= dis2e_center)
			{
				Vec4f l = tLines1[j];
				// ��Ҫ��ͷβ����
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


		//�洢�������ת����
		vector<RotatedRect> rrects;
		vector<Rect> ccs = mser(roi_dst, rrects);
		for (int cci=0;cci < ccs.size();cci++) 
		{
			rectangle(roi_thresh_mser, ccs[cci], Scalar(255,255,255), 1);
			// Ȼ���ٰѾ��ο������һ�㣬���ܰ������������򶼵õ���

		}
		imshow("roi_thresh_mser", roi_thresh_mser);
		waitKey(0);





		// 2. houghԲ����ȡ��ȷ���ģ�lsd��ȡ�ֱ�ߣ�ָ���ߣ�

		/**
			������(200, 200)�������������һ���뾶������׼��Բ��
		**/
		int searchRadius = 35;
		vector<Vec3f> circles;
		Mat1b centerArea = roi_dst(Rect(200 - searchRadius, 200 - searchRadius, 2 * searchRadius, 2 * searchRadius)).clone();
		Mat1b centerArea2 = centerArea.clone();
		// Mat3b centerArea3 = Mat::zeros(centerArea2.cols, centerArea2.rows, CV_8UC3);


		GaussianBlur(centerArea, centerArea, Size(3, 3), 1, 1);
		//Mat1b centerArea_add = centerArea.clone();
		//threshold(centerArea_add, centerArea_add, 0, 255, CV_THRESH_OTSU);

		// ����ֲ���ֵ��ʹ�м䲿����ȫ��ף��Ӷ�����̽��õ�
		adaptiveThreshold(centerArea, centerArea, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 9, 0);
		imshow("center", centerArea);
		waitKey(0);
		// �������������ǱȽ�ð�յġ�����Ϊ����֤���������ģ������������ͬʱ�޷�ȷ�������ֵ30�ڸ�������������
		// HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 30, 8, 0, 6);
		// �����ĸ����ֱַ���� �ֱ��ʣ��������Ϊ�����ĵ�����������Բ֮����С���룬canny�ĵ���ֵ��ͶƱ���ۻ����������ж��ٸ������ڸ�Բ����Բ��С�뾶��Բ���뾶
		// һ��ͶƱֵ�� 15 �� 20�䣬��������ӽ��ر�ƫ�����м�Բ�����Բ������Ҫ����ͶƱֵ�ˣ�Ŀǰ����ӽǱȽ�����������ֻ��Ҫ����20
		HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 150, 18, 10, 25);


		// ʵ���������ǿ��Լ����Ż��ģ��������������κ�Բ������¾ͳ������������򣬼���ͶƱ�������Ŵ�Բ���뾶

		// Ѱ��������������Ǹ�Բ����Ϊ���ǵ�׼�ģ����Ƴ�Բ
		Point ac_center = Point(200, 200);
		Point f_center = Point(searchRadius/2, searchRadius/2);
		float f2ac = 200;
		for (int i=circles.size()-1;i>=0;--i) 
		{
			Point mCenter = Point(circles[i][0], circles[i][1]);
			// �������ְ취��һ����ȡ����f_center��̣�һ���������С,�Ȳ��õڶ���
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



		// �����ǻ�ȡָ����
		// �����ҳ������ߵ���������
		float mRadius = 80;
		float angle = 4;
		float angelCos = cos(angle *  pi / 180);
		
		// ע�������ȡ�� Lines �� ���� ͨ��  ��ʴ�õ����Ǹ��ߣ�������ԭ�ȵ��Ǹ��ߣ����������Բ��ʱ���õ���ԭ�ȵ���

		// �洢��Щ���߶�
		vector<int> backs = vector<int>(tLines2.size());
		// �洢��Щǰ�߶ԣ�����֮��ļ���
		vector<int> fronts = vector<int>(tLines2.size());

		// ��ʼ����Щ�Ե�ֵ��Ĭ��Ϊ-1
		for (int k= tLines2.size()-1;k>=0;--k)
		{
			backs[k] = -1;
			fronts[k] = -1;
		}

		for (int i=tLines2.size() - 1;i >= 0; --i)
		{
			Vec4f& line1 = tLines2[i];
			// �����뾶�ڣ������������Լ�������ߣ�Ȼ�����е��������ɵĽǶȲ��ܴ��ĳ��ֵ������ȡ�����Լ���̵���Щ�ߣ�����ǿ��������ǲ����Ѿ���Ϊ�˱��˵ĺ�����
			int mIndex = i;
			float mDis = 1000;
			for (int j = tLines2.size() - 1; j >= 0; --j)
			{
				if (i == j)
					continue;
				Vec4f& line2 = tLines2[j];

				// ���ж��ǲ����Լ��������
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


			// ����Ѿ�������Ѿ��Ǳ��˵ĺ����ˣ���ôҪ�Ƚ�һ������ǰ�ߵĳ��̣����Ĳ����ʸ��������
			if (fronts[mIndex] >= 0)
			{
				if (point2point(tLines[fronts[mIndex]]) < point2point(tLines[i])) {

					// �ò������˾�back���Լ�
					backs[fronts[mIndex]] = fronts[mIndex];

					backs[i] = mIndex;
					if (i != mIndex)
						fronts[mIndex] = i;
				}
				else 
				{
					// �ò��������ûback��
					backs[i] = i;
				}
			}
			// ���������ֱ��ȷ��ǰ���ϵ����
			else 
			{
				backs[i] = mIndex;
				if (i != mIndex)
					fronts[mIndex] = i;
			}
		}

		// ����ֻ�Ǽ��������Ƿ����
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
				//cout << "��-" << i << "������1���ߵ�������" << endl;
			}
			// ��ȷ����ʽӦ���ǣ�backs����洢���ǵ�ǰֱ�ߵĺ��ߣ�������ߵ����Լ���˵���Լ�û����
			// ��fronts�洢���ǵ�ǰֱ�ߵ�ǰ�ߣ����ǰ�ߵ���-1��˵���Լ�û��ǰ��
		}

		// �����Щ�Բ��ҵó������
		// ����һ����¼�ڵ��Ƿ񱻷��ʵ�����
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

		// ��ʾֱ�����
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

		// ����ֱ�����
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
				// ����Ҫ����ƽ������ΪԽ����ռ��Ӧ��Խ��
				total_length += point2point(tLines2[li]);
			}
			if (total_length > maxLength) { maxLength = total_length; maxLine = goal_line; }
		}
		// ���������
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

		// ������ߵĽ�β����ָ���ߵ�β��
		Vec4f last_part = tLines2[maxLine[maxLine.size() - 1]];
		Point lastPoint = Point(last_part[2], last_part[3]);


		// 3. ���淽�����о��û���ǰ�� bim+֧�ֵ㷽����� Ȼ�������ȡ��Բ
		// ��tLine���з��ࣨ���Ƕȷ��ࣩ�����簴��40��һ������������9������ 
		// ��� tLine ������ʶ��ָ��� line�ǲ�һ���ġ�
		int sangle = 40;
		vector<vector<int>> sangles_(360 / sangle);
		for (int ki = tLines.size() - 1; ki >= 0; --ki)
		{
			
			// ����нǵ�cosֵ
			int xd = tLines[ki][2] - ac_center.x;
			int yd = ac_center.y - tLines[ki][3];
			// ֵ���� 0~360֮��
			float vangle = fastAtan2(yd, xd);
			sangles_[(int)vangle / sangle].push_back(ki);
		}
		// ����Ҫ����ҪһЩû�е������
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
		// ������Բ�����֧�ֵ�����Զ��̶ȣ�����ĳ̶��ǳ�����Բ�����
		float acceptThresh = 8;
		// �洢��ǰ��֧�ֵ����
		int nowSnum = 0;
		// �洢���֧�ֵ���Ǹ���Բ����Ӿ���
		RotatedRect bestEl;
		// �洢���յ�֧�ֵ�
		vector<int> bestSupportPoints;
		// ������������Ĵ���
		for (int ri=0;ri<ranTimes;ri++)
		{
			// �洢Ҫѡȡ����������5��������ȡֵ
			vector<int> ks;
			// ������ʼ�ĵ�
			int kindex = random(asize - 1);
			ks.push_back(kindex);
			for (int ki = 0; ki < 4; ki++)
			{
				// ����1~2���������Ҳ������ѡȡ����������ѡ��������ǰѡ��������������λ�ã���֤��һ���ĽǶ�
				kindex = (kindex + random(2)) % asize;
				ks.push_back(kindex);
			}
			// ��ÿ����ѡ�����������ѡȡһ���������Բ���
			vector<Point> candips;
			for (int ki = 0; ki < 5; ki++)
			{
				// ks[ki]ָ���ѡȡ������������ţ�sangles[�������]�õ��ľ������������������е����ţ���������������������ľ�����������ĵڼ�����
				vector<int> shanqu = sangles[ks[ki]];
				int sii = random(shanqu.size() - 1);
				Vec4f vvvv = tLines[shanqu[sii]];
				Point ssss = Point(vvvv[2], vvvv[3]);
				candips.push_back(ssss);
			}
			// �����Բ�����֧�ֵ����
			RotatedRect rect = fitEllipse(candips);

			//Mat forellipse = roi_foresee.clone();
			//ellipse(forellipse, rect, Scalar(255, 255, 255));
			//imshow("forellipse", forellipse);
			//int aaaa = waitKey(0);
			//if (aaaa == 100) { continue; }

			// �洢֧�ֵ�
			vector<int> support_points;
			for (int ki=0;ki<tLines.size();ki++)
			{
				Point sp = Point(tLines[ki][2], tLines[ki][3]);
				Point newsp = origin2el(rect.center, rect.angle / (float)180 * pi, sp);
				//circle(forellipse, Point(tLines[ki][2], tLines[ki][3]), 3, Scalar(0, 0, 0), -1);
				// ���ﻹ��Ҫ���������Բ�Ĵ�С
				float edistance = abs(sqrt(pow(newsp.x, 2) / pow(rect.size.width/2, 2) + pow(newsp.y, 2) / pow(rect.size.height/2, 2)) - 1) * max(rect.size.width / 2, rect.size.height / 2);
				//cout << "edistance�� " << edistance << endl;
				if (edistance <= acceptThresh)
				{
					support_points.push_back(ki);
				}
				//cout << "support_points�� " << support_points.size() << endl;
				//cout << "��ǰ������ " << ki << endl;
				//imshow("forellipse", forellipse);
				//waitKey(0);
			}

			//cout << "֧�ֵ�Ϊ�� " << endl;
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
		// ���������ѭ�����ܵó�֧�ֵ������Ǹ������Բ��������ʾ����

		// ���õ���֧�ֵ�����������
		vector<Point> supporters;
		for (int ki = 0; ki < bestSupportPoints.size(); ki++)
		{
			int tLineIndex = bestSupportPoints[ki];
			Point supporter = Point(tLines[tLineIndex][2], tLines[tLineIndex][3]);
			supporters.push_back(supporter);
			circle(roi_line, supporter, 3, Scalar(0, 0, 0), -1);
		}
		bestEl = fitEllipse(supporters);
		// һ��ĳ�������Բ������̫�⣬Ҳ���ǳ���ȶ������и����Ƶģ�������Ҫ���������Բ�������������Ϊһ��������
		// ʵ�ֵݹ���ã�����ÿ�δ���һ������������������ʾ�ڼ��εݹ飬���ܵݹ�̫��ε�������ѭ��
		// �������Ļ���������һ��ʼ�õ�����Բ���� �����ȶԣ��������ʧ���һ���޶ȣ��Ǿ���Ū���ˣ���Ҫ�ݹ�һ�Σ�������յݹ黹�Ǵ�����ô��Բ����һ��ʼ�õ�����Բ(���Ǽ������������Բ)
		ellipse(roi_line, bestEl, Scalar(255, 255, 255));
		RotatedRect bestEl_scale = RotatedRect(bestEl.center, Size2f(bestEl.size.width * 0.86, bestEl.size.height * 0.86), bestEl.angle);
		ellipse(roi_line, bestEl_scale, Scalar(0, 0, 0));
		imshow("roi_line", roi_line);
		waitKey(0);
		waitKey(0);






		// mser���
		std::vector<cv::Rect> candidates;
		imshow("roi_mser", roi_dst);
		waitKey(0);
		// candidates = mser(roi_dst);
		// ʵ��֤����thresh���mser��ֱ��mser����
		candidates = ccs;


		vector<SingleArea> sas;

		// �洢Ҫ��ϸ������ֵ���Բ
		vector<Point> numberAreaCenters;


		// ͶƱѡ���ĸ��ǶȲ�����ȷ�ģ�ֻ����Ȧ��mser����ͶƱ����ͶƱ����ľ��Ǳ�����ת�ĽǶȣ�Ȼ��ԭ��תǰ�����ӽ��зָ�����б�
		// ������ͬ�Ƕ�֮��Ĳ�ֵ
		int rect_angle_err = 5;
		// ������ͬһ���ǶȵĶ������������Ҫ��������ƽ��ֵ��
		unordered_map<int, int> rect_angle_sums;
		unordered_map<int, int> rect_angle_nums;
		// �洢ÿ���Ƕȶ�Ӧ����Щ��ת���ε�id
		unordered_map<int, vector<int>> rect_goal_ixs;
		// ������ĵ�����
		unordered_map<int, int>::iterator  rect_iter;
		unordered_map<int, int>::iterator  rect_sums_iter;
		unordered_map<int, int>::iterator  rect_goal_ixs_iter;
		
		// �洢������������Щ���ε�index
		vector<int> ris;
		// ������ʾ
		for (int i = 0; i < candidates.size(); ++i)
		{
			// ��ɸѡ�����Բ����ĵ�
			Point ncenter = Point(candidates[i].x + candidates[i].width / 2, candidates[i].y + candidates[i].height / 2);
			Point newsc = origin2el(bestEl.center, bestEl.angle / (float)180 * pi, ncenter);
			float ndistance = sqrt(pow(newsc.x, 2) / pow(bestEl.size.width / 2, 2) + pow(newsc.y, 2) / pow(bestEl.size.height / 2, 2));
			// �������Բ��Ҫ���ڲ���һ�£�ȥ�������զ�㣬����̶���
			if (ndistance >= 0.86) { continue; }

			ris.push_back(i);

			// �Ȱ����еĶ������
			rectangle(roi_mser, candidates[i], Scalar(255, 255, 255), 1);
			drawRotatedRect(roi_mser, rrects[i]);
			cout << "w=" << candidates[i].width << ",h=" << candidates[i].height << ",w-h=" << candidates[i].width - candidates[i].height << endl;
			//cout << "�Ƕ�Ϊ�� " << rrects[i].angle << endl;


			// ͳ��ͶƱ
			int rrect_angle = rrects[i].angle;
			bool is_new_r_angle = true;
			for (rect_iter = rect_angle_nums.begin(); rect_iter != rect_angle_nums.end(); rect_iter++)
			{
				// ����ڵ�angle���бȽϣ�������󣬾ͷŽ��Ǹ�angle����
				// iter->first��key��iter->second��value
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
			



			// �ѷ������ֵ������ѡ����
			Mat mser_item = roi_dst(candidates[i]);



			vector<float> v = getHogData(mser_item);
			Mat testData(1, 144, CV_32FC1, v.data());
			int response = svm->predict(testData);

			// �������������ģ����ø�ʴ�����ָ�Ȼ��Ū����


			// ��ʾ����
			if (response >= 0)
			{
				rectangle(roi_mser, candidates[i], Scalar(255, 255, 255), 2);
				circle(roi_mser, ncenter, 2, color, -1);

				// Ϊ��֮�������
				sas.push_back({ float(response), i, ncenter });
				numberAreaCenters.push_back(ncenter);
				// ������Ҫ�޸ģ�ֻ�Ƿ������
				// if (response == 3) { response = 2; }

				//cout << "��ǩΪ�� " << response << endl;
				//imshow("roi_mser", roi_mser);
				//waitKey(0);
			}
		}

		imshow("roi_mser", roi_mser);
		waitKey(0);


		int max_rect_angle = 0;
		int max_rect_angle_nums = 0;
		int max_rect_angle_sum = 0;
		// �ҳ�Ʊ��������ת�Ƕȣ�Ȼ��ԭ��תͼ�񣬷ָ����֣�ʶ��
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
				// rect_sums_iter->second�õ�����ͶƱ���ĽǶȵĺ�
				max_rect_angle = rect_sums_iter->first;
				max_rect_angle_sum = rect_sums_iter->second;
				break;
			}
		}
		

		// �����ж��������������ת��������ת
		// ����������δ���0����ת�������0��ʱ���ĳ���Ⱥ�������Χ����Ȳ�����һ�£�˵����ߺ͵ױ���������������Ƿ������ġ�
		// �����ת���ε���ת���ǲ�ߣ���ô˵�����������㣬�����ת���ǵױߣ�˵�����������㡣
		// ͨ��ͳ�Ƴ����һ���������ȷ�������ַ���
		vector<int> rids = rect_goal_ixs[max_rect_angle];
		int is_same_num = 0;
		for (int ri = 0; ri < rids.size(); ri++)
		{
			int ix = rids[ri];
			RotatedRect rrect = rrects[ix];

			// ��ȡ��ת���ε��ĸ�����
			cv::Point2f* vertices = new cv::Point2f[4];
			rrect.points(vertices);
			// ����ı߳�Ӧ���� p[0]��p[3]�ľ��룬����ı߳�Ӧ����p[1]��p[0]�ľ���
			float rw = point2point(vertices[0], vertices[3]);
			float rh = point2point(vertices[0], vertices[1]);

			float is_same_shape = (candidates[ix].width - candidates[ix].height) * (rw - rh);
			if (is_same_shape >= 0) { is_same_num++; }
		}

		// ��������ת��Ҫƽ��һ�£����Ƿ��ֲ�ƽ���Ļ�Ч������
		//max_rect_angle = max_rect_angle_sum / max_rect_angle_nums;
		// ������ת����
		
		Mat rotationMatrix;
		// ������������������Ǵ�����ĽǶȱߵ���Ӧ���ǵױ߻��ǲ��
		bool is_b_or_s = false;
		float rangle = 0;
		if (is_same_num / rids.size() >= 0.5) 
		{
			// ��getRotationMatrix2D�У��Ƕ�Ϊ����˳ʱ�룻�Ƕ�Ϊ������ʱ�롣����������Ĭ�ϲ��ù�
			// �Ƕȱߣ�Ҳ����p[0]~p[3]�������ߣ��ǵױߣ�˵��������
			is_b_or_s = true;
			rangle = -abs(max_rect_angle);
			rotationMatrix = getRotationMatrix2D(Point(200, 200), rangle, 1);//������ת�ķ���任���� 
		}
		else 
		{
			// �Ƕȱߣ�Ҳ����p[0]~p[3]�������ߣ��ǲ�ߣ�˵��������
			is_b_or_s = false;
			rangle = (90 - abs(max_rect_angle));
			rotationMatrix = getRotationMatrix2D(Point(200, 200), rangle, 1);//������ת�ķ���任���� 
		}
		Mat rMat;
		warpAffine(roi_dst, rMat, rotationMatrix, Size(roi_dst.cols, roi_dst.rows));//����任  
		imshow("rMat", rMat);
		waitKey(0);



		// ����ֻ�Ƿ�����ת�Ƕ��ѣ����������roi_dst�е�ÿ��Сmser�����н�����ת����ת��������ת���ε����ġ�
		// Ȼ���ٽ������ַָȻ���ٽ���ʶ����ȥʶ�����֡�


		Mat roi_dst_clone_ = roi_dst.clone();
		// �����������Ǳ仯��������ת����
		for (int ri = 0;ri < ris.size();ri++)
		{
			int ix = ris[ri];
			RotatedRect rrect = rrects[ix];
			Rect rect = candidates[ix];
			Point rect_center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
			// ʵ�鷢�ְ����Ļ��� mser�����ľ��ε����ģ����ĸ�׼ȷ��ͬʱ����һ�·�Χ
			rrect.center = rect_center;
			rrect.size = Size(rrect.size.width + 4, rrect.size.height + 4);
			// ��ȡ��ת���ε��ĸ�����
			cv::Point2f* vertices = new cv::Point2f[4];
			rrect.points(vertices);
			// ����ı߳�Ӧ���� p[0]��p[3]�ľ��룬����ı߳�Ӧ����p[1]��p[0]�ľ���
			int rw = point2point(vertices[0], vertices[3]);
			int rh = point2point(vertices[0], vertices[1]);
			int temp_ = 0;
			float is_same_shape = (candidates[ix].width - candidates[ix].height) * (rw - rh);

			// ���ĽǶȱ��ǵױ�
			if (is_same_shape >= 0) 
			{
				// ��ʵ�ʽǶȱ�һ��
				if (is_b_or_s) { rrect.angle = max_rect_angle; }
				// ʵ��Ӧ���ǲ����Ϊ�Ƕȱ߲Ŷ�
				else { rrect.angle = 270 - abs(max_rect_angle); }
			}
			else 
				// �Ƕȱ��ǲ��
			{ 
				// ��ʵ�ʽǶȱ�һ��
				if (!is_b_or_s) { rrect.angle = max_rect_angle; }
				// ʵ��Ӧ���ǵױ���Ϊ�Ƕȱ߲Ŷ�
				else { rrect.angle = 90 + abs(max_rect_angle); }
				// �Ƕȱ��ǲ�ߣ���ˣ�����rwʵ������h��rhʵ������w
				temp_ = rw; rw = rh; rh = temp_;
			}

			
			drawRotatedRect(roi_dst_clone_, rrect);

			cout << "rrect.angle = " << rrect.angle << endl;
			// ���浥����ȡ��Щ��ת��������
			// �Ȱ�����Ŵ��㹻��������ԭ��ͼ��Ȼ��Χ��������ת��֮������ȡroi
			Mat scaleMat = roi_dst(Rect(rrect.center.x - rect.width, rrect.center.y - rect.height, 2 * rect.width, 2 * rect.height));
			// �ڷŴ�������У������� Point(rect.width, rect.height)
			rotationMatrix = getRotationMatrix2D(Point(rect.width, rect.height), rangle, 1);//������ת�ķ���任���� 
			Mat scaleMat2;
			warpAffine(scaleMat, scaleMat2, rotationMatrix, Size(scaleMat.rows, scaleMat.cols));//����任  
			// ��ȡĿ������
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

			// ��ȡ��ת���ε��ĸ�����
			cv::Point2f* vertices = new cv::Point2f[4];
			rrect.points(vertices);
			// ����ı߳�Ӧ���� p[0]��p[3]�ľ��룬����ı߳�Ӧ����p[1]��p[0]�ľ���
			int rw = point2point(vertices[0], vertices[3]);
			int rh = point2point(vertices[0], vertices[1]);

			float is_same_shape = (candidates[ix].width - candidates[ix].height) * (rw - rh);

			warpAffine(roi_rect, mser_r, rotationMatrix, Size(roi_rect.rows, roi_rect.cols));//����任  



			imshow("mmmser", mser_r);
			waitKey(0);
		}


		imshow("roi_mser", roi_mser);
		waitKey(0);





		// 4. ��ֵ��ȡ
		// ���ҳ��任�㣬Ҳ��������Ϊ͸�ӱ任�ĵ㣬Ȼ���Ǳ任
		// �������ֵ�������Ѱ�ɶԴ��ڵ�
		// �洢��Щ�ϲ���������Լ����ǵ���Ӧֵ




		

		// �м丨��������
		vector<MergeArea> mas;
		// �涨�ںϵ���̾���
		float merge_distance = 30;
		float min_merge_distance = 10;
		// ������angle �����
		int angle_error = 12;
		int max_likely_angle = -1;
		// ����һ������������С merge_distance
		float merge_distance_scale_step = 3;
		// ��¼��������ı����Ǹ���ά����
		vector<vector<int>> joinTable(sas.size());
		// ��ʼ����ά����

		// ͳ�����߳��ֵ����нǶȵĶ�Ӧ�ıߵ�����
		unordered_map<int, int> line_angle_nums;
		// ������ĵ�����
		unordered_map<int, int>::iterator  iter;


		
		// ���վ���������������Ҫռ����
		float max_portion = 0.5;


		// ����ͳ�ƺõ���Щ����
		// ���ͳ�����������Ǹ����߲�����������80%����ô�ͽ�����ֵ���ظ�Ѱ����ѱ߱�

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
			// ��¼����������ֵ�µ������Ǹ�angle
			max_likely_angle = max_join_line_angle;
			if (total_join_line_num == 0 || max_join_line_num / (float)total_join_line_num < max_portion)
			{
				merge_distance -= merge_distance_scale_step;
				if (merge_distance < min_merge_distance)
				{
					// �����ֵ���ͣ���ô�Ϳ�����ͣ
					break;
				}
				else
				{
					// �ظ����������²�������Ѱ�ұ߲����� �߱� �Լ� line_angle_nums

					// ��ʼ�� �߱� �Լ� line_angle_nums
					for (int kii = 0; kii < sas.size(); kii++)
					{
						joinTable[kii] = vector<int>(sas.size());
						// -2��ʾ�Խ��ߣ��Լ����Լ�����-1��ʾû�����ӣ����ڵ���0��ֵ��ʾ�������ӵ�ֱ�ߵ���б��
						for (int kjj = 0; kjj < sas.size(); kjj++)
						{
							if (kii == kjj) { joinTable[kii][kjj] = -2; }
							else { joinTable[kii][kjj] = -1; }
						}
					}
					line_angle_nums = unordered_map<int, int>();

					// ������ֵ�ѵ��ɱߡ�
					for (int kii = 0; kii < sas.size(); kii++)
					{
						float response = sas[kii].response;
						// Ҫôֱ�Ӱ������ںϵ��Ǹ���10ֱ����Ⱦ�� merge_area �����Ҫô�Ͱ��������͵�������һ���Ķ���
						if (response > 9)
						{
							//csets.push_back();
							//continue;
						}
						Point sacenter = sas[kii].center;
						// Ѱ�����ںϵ��Ǹ��㣬Ҳ���Ǿ�����������̵��Ǹ���
						float min_dd = merge_distance;
						int target_sa = -1;
						for (int kjj = 0; kjj < sas.size(); kjj++)
						{
							// ������Ҫ�鿴kjj�Ƿ��Ѿ����ӹ�kii���Ѿ������ľ�����(-1��ʾδ���ӣ�-2��ʾ�Լ����Լ�)
							if (joinTable[kii][kjj] ==-2 || joinTable[kii][kjj] > -1 || sas[kjj].response > 9) { continue; }
							float jj_ii_distance = point2point(sacenter, sas[kjj].center);
							if (min_dd > jj_ii_distance) { min_dd = jj_ii_distance; target_sa = kjj; }
						}
						// ����ҵ��ںϵ㣬���޸�table����û�оͲ��ù�
						if (target_sa >= 0)
						{
							// ���������ֱ�ߵ���б��
							double dy = sas[kii].center.y - sas[target_sa].center.y;
							double dx = sas[kii].center.x - sas[target_sa].center.x;
							// ע�⣬���ﲻ��atan2����Ϊ����ֻ��Ҫ��һ�������жϷ�����Ϊֻ�ǿ���б�ʣ�,��Ϊ�ĵ�����-90��90������ƫ��90����֤-1��-2��������;��
							int s_line_angle = int(atan(dy / dx) * 180 / pi) + 90;


							// ��ͳ����Ѱ�Ҹ�����Ƕ����Ƶģ����û�о���Ӹ����ͳ�ƣ�����оͼ�1

							bool is_new_angle = true;
							for (iter = line_angle_nums.begin(); iter != line_angle_nums.end(); iter++)
							{
								// ����ڵ�angle���бȽϣ�������󣬾ͷŽ��Ǹ�angle����
								// iter->first��key��iter->second��value
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
			// ����Ѿ��ﵽҪ�󣬾��˳�ѭ��
			else { break; }
		}



		// ���ͳ���������ĳ���80%������Щ���߽ǶȲ���������Щ����(����angle�ж�)����table��ȫ����Ϊ�Ͽ�״̬����0��
		for (int kii = 0; kii < sas.size(); kii++)
		{
			// -2��ʾ�Խ��ߣ��Լ����Լ�����-1��ʾû�����ӣ����ڵ���0��ֵ��ʾ�������ӵ�ֱ�ߵ���б��
			for (int kjj = 0; kjj < sas.size(); kjj++)
			{
				if (kjj == kii) { continue; }
				if (joinTable[kii][kjj] != max_likely_angle) { joinTable[kii][kjj] = -1; }
			}
		}


		// ����һ����¼�ڵ��Ƿ񱻷��ʵ�����
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
		// ��ʾ�����ں�Ч����ͬʱ��Ⱦ��merge_area
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
				// �Ȱ����еĶ������
				rectangle(roi_merge, candidates[sas[sas_index].cc_index], Scalar(255, 255, 255), 1);
				cc_indexs.push_back(sas[sas_index].cc_index);
				goal_set_areas.push_back(sas[sas_index]);
				x_ += sas[sas_index].center.x;
				y_ += sas[sas_index].center.y;
			}
			// ������µ�����
			Point merge_center = Point(int(x_ / goal_set_size), int(y_ / goal_set_size));
			// ���������Լ�������ac_center������
			circle(roi_merge, merge_center, 2, Scalar(255, 255, 255), -1);

			// ����x��goal_set_areas���򣬲��ó�������response
			sort(goal_set_areas.begin(), goal_set_areas.end(), SortByX);
			float merge_response = 0;
			for (int kjj = 0; kjj < goal_set_size; kjj++)
			{
				merge_response += goal_set_areas[kjj].response * pow(10, -kjj);
			}

			// �������������޷ָ��ߵ���ļнǲ� ��Ⱦ�� merge_area
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
			�����Ƕ���Щ merge_area ����ɸѡ������Щ��Ӧֵ��ֵĵ�ȥ����

			�����ǰ�merge_area����һ���㣬����response��yֵ��angle��xֵ�����滹����ϸ˵��

		*/

		// �洢��ֵ��,false��ʾ�õ��Ƿ�����ֵ��true��ʾ�õ�Ϊ����ֵ��һ��ʼȫ�������Ϊ������ֵ
		vector<bool> singulars(merge_areas.size(),false);

		// ���������ֱ�ߵľ���
		float dis_error = 0.0125;
		// �ж��Ƿ��������Ǹ���ֵ����һ������ֵ,������΢�����£������Ƚ��ܼ��Ļ������Է����������
		float singular_thresh = 0.5;
		if (merge_areas.size() >= 8) { singular_thresh = 0.35; }
		if (merge_areas.size() <= 5) { singular_thresh = 0.6; dis_error = 0.04; }


		// �ȶ�merge_areas������Ӧֵ����������Ӧֵ�����ǰ��
		// Ȼ���������RANSAC�㷨����ͬ���ǣ�ÿ�ι�һ������һ����ÿ�ζ��ǰ�max����Ͷ�䵽1��������max���������㣬��ô��һ�����û�ı䣬
		// ����������max�㱾����Ǹ���ֵ��˵���������ˣ����������㿴������Ӧ��С������Ҫ�������ų�����Ȼ�������������ԵĽ��й�һ��
		// ��һ����Ͳ��ñ����ķ����������������Ϊ���ǵĵ�Ƚ��٣���ÿ����ģ�������������ߣ������ĵ���ࣩ
		sort(merge_areas.begin(), merge_areas.end(), SortByRes);

		// �����ǰѲ����������ĵ㶼�ŵ� singulars ��
		for (int kii = 0; kii < merge_areas.size(); kii++)
		{
			if (kii == 11) 
			{
				cout << "in" << endl;
			}
			// ������������������ �����˶��ٸ���
			int best_accept_num = 0;
			// ��ÿ���㶼ͳ�����������������ɵ�ֱ���Ƿ񴩹��㹻��ĵ㣬���Ƿ񴩹������ֵ���ж���Ҫ���ݹ�һ��
			for (int kjj = 0; kjj < merge_areas.size(); kjj++) 
			{
				// ���������Ѿ�������ֵ�ĵ�
				if (kii == kjj || singulars[kjj]) { continue; }
				// �ҳ�Ŀǰ������ֵ�������Ǹ���Ӧֵ�����������й�һ��
				float max_res = -1.0;
				for (int kdd = 0; kdd < singulars.size(); kdd++) { if (!singulars[kdd]) { max_res = merge_areas[kdd].response; break; } }

				// ������kjj������ֱ�ߴ��������������������ֱ�߾��Ѿ������������ˣ��������㲻�ü���
				int accept_num = 2;
				for (int kdd = 0; kdd < singulars.size(); kdd++)
				{
					// ����㲻�ڿ��Ƿ�Χ
					if (kjj == kdd || kii == kdd || singulars[kdd]) { continue; }
					float ddx = merge_areas[kdd].angle / 360; float ddy = merge_areas[kdd].response / max_res;
					float ddx1 = merge_areas[kii].angle / 360; float ddy1 = merge_areas[kii].response / max_res;
					float ddx2 = merge_areas[kjj].angle / 360; float ddy2 = merge_areas[kjj].response / max_res;
					float m_dis = point2Line(ddx, ddy, ddx1, ddy1, ddx2, ddy2);
					if (m_dis <= dis_error) { accept_num++; };
				}
				// ����������ߴ����ĵ��֮ǰ���ã���Ҫ�滻һ��
				if (accept_num >= best_accept_num) { best_accept_num = accept_num; }
			}
			// ͳ�Ʋ��ж�best_accept_num�Ƿ��������һ����������������ô����������ֵ��
			if (best_accept_num < singular_thresh * singulars.size()) { singulars[kii] = true; }
		}



		// �ó��������������еĵ�
		vector<MergeArea> merges_1;
		vector<MergeArea> merges;

		cout << "��һ��ȷ�Ϻ�" << endl;
		for (int kii = 0; kii < singulars.size(); kii++)
		{
			if (singulars[kii]) { continue; }
			cout << "�������ӦֵΪ�� " << merge_areas[kii].response << endl;
			merges_1.push_back(merge_areas[kii]);
		}


		if (merges_1.size()==0) 
		{
			cout << "!!!!!!!!!!���Ƕȴ������⣬������½Ƕȼ������!!!!!!!!!!!" << endl;
			continue;
		}


		// ���սǶȸ�����mergeArea����
		sort(merges_1.begin(), merges_1.end(), SortByAngle);


		// ����������һ��������sort��֮��һ�����ϸ񵥵������ģ�������ֵݼ����߲�����������ô����Ҫȥ������
		merges.push_back(merges_1[0]);
		int merges_pos = 0;
		cout << "�ڶ���ȷ�Ϻ�" << endl;
		cout << "�������ӦֵΪ�� " << merges_1[0].response << endl;
		for (int kii = 1; kii < merges_1.size(); kii++)
		{
			if (merges_1[kii].response <= merges[merges_pos].response) { continue; }
			else { merges.push_back(merges_1[kii]); merges_pos++; cout << "�������ӦֵΪ�� " << merges_1[kii].response << endl;}
		}



		// �����������㷨������dis_error�Ƿ���Ҫ�ٴμ�С�������ж�����ֵ�㳣�������󣬱����Ƿ������ȵĵ�


		// ����merge_areas
		// delete merge_areas;

		/*
			��ֱ����ָ�̶��ҷ�Χ�������Χȷ����nonsingulars���棬��ô��Ѱ����ӽ����Ǹ���Χ��
			�����nonsingulars��ࣨ���Ƕ�С����֪��Χ��������������gradientȥ����Ӧ�õ���ֵ
			ͬ���Ҳ�
		*/
		// ��Ϊ�����Ѿ���merge_areas���Ƕ������ˣ�����merge_areas�ĵ�һ��Ԫ�ؾ��ǽǶ���С�ģ����һ��Ԫ�ؾ��ǽǶ�����
		float ddy = -lastPoint.y + ac_center.y;
		float ddx = lastPoint.x - ac_center.x;
		float pointerAngle = getVecAngle(ddy, ddx);
		float pointerValue = 0;

		int merge_last = merges.size() - 1;
		if (pointerAngle < merges[0].angle)
		{
			// Ѱ��������Ǹ�gradient
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
			// Ѱ�Ҹպ��Ǻϵ��Ǹ�����
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

		cout << "��ȡ�õ�����ֵΪ�� " << pointerValue << endl;
		imshow("roi_merge", roi_merge);
		waitKey();	
		waitKey();
		waitKey();



	}






	int yghds = 0;
	return 0;
}






// ��svmdata����ѵ����������Կ��Ƿ�ת��ǿ����
// train
int train()
{
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";
	string labelPath = "D:\\VcProject\\biaopan\\data\\labels.txt";
	// �洢 ͼƬ�洢·�� ���� ��Ӧ��label
	vector<string> raws = readTxt(labelPath);
	// �洢ÿ��ͼƬ�����������Լ�label
	vector<vector<float>> trainingData;

	vector<int> labels;
	// �ַ����ķָ��ʶ
	const string spliter = "===";
	// ѵ���Ͳ��Թ�������ȫ��һ���ģ�ѵ����ȫ������һ��ѵ�������ǲ��Ե�ʱ����Ҫ���� һ��originһ��mser���ϣ����originȡƽ��׼ȷ��
	for (int i=0;i< raws.size();i++)
	{
		// ����һ��ͼƬ
		vector<string> raw = splitString(raws[i], spliter);
		string src = raw[0];
		int label = str2int(raw[1]);
		Mat mat = imread(src, IMREAD_GRAYSCALE);
		// ��΢����ģ��
		blur(mat, mat, Size(3, 3));
		trainingData.push_back(getHogData(mat));
		labels.push_back(label);
		// ������������ǿ�����羵�񣬷�ת��ԭ���ǿ��ܻ������Щ�����
		//for (int flipCode=-1; flipCode <2; flipCode++)
		//{
		//	Mat flipMat;
		//	flip(mat, flipMat, flipCode);
		//	trainingData.push_back(getHogData(mat));
		//	labels.push_back(label);
		//}
	}

	//����֧���������Ĳ�����Set up SVM's parameters��

	Ptr<cv::ml::SVM> svm = ml::SVM::create();

	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	// svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1200, FLT_EPSILON));
	// svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 200, 1e-6));
	// svm->setDegree(1.0);
	svm->setC(2.67);
	svm->setGamma(5.83);


	// ׼��������
	// һ��ͼƬ������������144ά
	const int feature_length{ 144 };
	const int samples_count{ (int)trainingData.size() };
	if (labels.size() != trainingData.size())
	{
		cout << "���ݵ���������" << endl;
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

	cout << "��ʼѵ��" << endl;
	// ѵ��
	svm->train(trainingDataMat, ml::ROW_SAMPLE, labelsMat);
	// ����ģ���ļ�
	svm->save(modelPath);

	// ����
	
	Mat s1 = imread("D:\\VcProject\\biaopan\\data\\goodImgs\\457\\33.jpg", IMREAD_GRAYSCALE);
	imshow("s1", s1);
	waitKey(0);
	vector<float> ss = getHogData(s1);
	Mat testData(1, feature_length, CV_32FC1, ss.data());
	int response = svm->predict(testData);
	cout << "����� : " << response << endl;


	return 0;
}

// ���浥����Խ��
// singleTest
int singleTest()
{
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";
	Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
	Mat s1 = imread("D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\msers\\positives\\no_rotation\\120\\10\\7.jpg", IMREAD_GRAYSCALE);

	vector<float> ss = getHogData(s1);
	Mat testData(1, 144, CV_32FC1, ss.data());
	int response = svm->predict(testData);
	cout << "����� : " << response << endl;
	cout << "end" << endl;
	return 0;
}

// ���׼ȷ��
int batchTest()
{
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";
	string labelPath = "D:\\VcProject\\biaopan\\data\\labels.txt";

	Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
	// �洢 ͼƬ�洢·�� ���� ��Ӧ��label
	vector<string> raws = readTxt(labelPath);

	vector<int> labels;
	// �ַ����ķָ��ʶ
	const string spliter = "===";
	// ����ֻѵ��28800�����ݣ�������������
	int totalNum = 0;
	int correct = 0;
	for (int i = 28800; i < raws.size(); i++)
	{
		// ����һ��ͼƬ
		vector<string> raw = splitString(raws[i], spliter);
		string src = raw[0];
		int label = str2int(raw[1]);
		Mat mat = imread(src, IMREAD_GRAYSCALE);
		// �洢һ��hog����
		// vector<float> descriptors;//HOG����������
		// descriptors = getHogData(mat);
		// descriptors;
		vector<float> ss = getHogData(mat);
		Mat testData(1, 144, CV_32FC1, ss.data());
		int response = svm->predict(testData);
		if (response == label)
			correct += 1;
		totalNum += 1;
	}
	cout << "׼ȷ���� : " << correct / (float)totalNum << endl;
	cout << "end" << endl;
	return 0;
}



// ͬ�������������ݵĲ���
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
	// ��ͼƬ����ѹ��
	resize(image_1, image_1, Size(image_1.size[1] / 2.5, image_1.size[0] / 2.5));
	// ��ЩͼƬ��Ҫ��ת
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


	// ��������canny���ӻ�ȡ������Բ��������ķ�����ȡ
	if (ellsYaed.size() <= 1)
	{
		adaptiveThreshold(gray_clone_add, gray_clone_add, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 9, 0);
		yaed.Detect(gray_clone_add, ellsYaed);
	}

	if (ellsYaed.size() == 0)
	{
	// ��ⲻ����Բ��ֱ������
		return;
	}

	Mat3b resultImage = image.clone();



	// ��չ������Χ���Գ���Ϊֱ��������������
	int index = 0;
	Mat1b gray_clone2;
	cvtColor(image_1, gray_clone2, CV_BGR2GRAY);
	int el_size = ellsYaed.size();



	// ѡȡ������ 35 ������֧�ֵ����Բ
	int min_vec_num = 25;
	// �洢Ŀ����Բ
	Ellipse& el_dst = ellsYaed[0];
	// �洢Ŀ������
	Mat1b roi_zero = Mat::zeros(400, 400, CV_8UC1);
	Mat1b& roi_dst = roi_zero;
	// �洢Ŀ��Ŀ���֧����
	vector<Vec4f> tLines;

	while (index < el_size) {
		Ellipse& e = ellsYaed[index];
		int g = cvRound(e._score * 255.f);
		Scalar color(0, g, 0);
		// �ҵ�����
		int long_a = e._a >= e._b ? cvRound(e._a) : cvRound(e._b);
		rectangle(resultImage, Rect(cvRound(e._xc) - long_a, cvRound(e._yc) - long_a, 2 * long_a, 2 * long_a), color, 1);

		// �ҵ���ԭͼ�е�λ�ã�Ȼ���Ƚ���ֱ��ͼ���⻯�����lsd
		// �������ھ������⣬����� e._xc ��С�� long_a�����߾��εķ�Χ������ͼƬԭ���ߴ�
		int r_x = max(0, (cvRound(e._xc) - long_a)) * scaleSize;
		int r_y = max(0, (cvRound(e._yc) - long_a)) * scaleSize;
		// �����ߴ�Ļ����ʵ���С
		int r_mx = min(gray_clone2.cols, r_x + 2 * long_a * scaleSize);
		int r_my = min(gray_clone2.rows, r_y + 2 * long_a * scaleSize);
		int n_width = min(r_mx - r_x, r_my - r_y);
		Mat1b roi = gray_clone2(Rect(r_x, r_y, n_width, n_width));
		Mat1b roi_2;
		resize(roi, roi_2, Size(400, cvRound(float(roi.cols) / float(roi.rows) * 400)));
		// ͬʱҲ����һ����Բ
		float scaleFactor = float(400) / 2 / long_a;
		// �����������ʱ��ʵ���Ͼ͹涨������� 400 x 400 ��ͼ������Բ���ľ���(200, 200)
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
			// ɸѡ����Щ�������ıȽ�Զ����
			float distance = point2Line(e_center, lines_std[j]);
			if (distance <= 10)
			{
				Vec4f l = lines_std[j];
				// ��Ҫ��ͷβ����
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




	// 2. houghԲ����ȡ��ȷ���ģ�lsd��ȡ�ֱ�ߣ�ָ���ߣ�

	/**
		������(200, 200)�������������һ���뾶������׼��Բ��
	**/
	int searchRadius = 25;
	vector<Vec3f> circles;
	Mat1b centerArea = roi_dst(Rect(200 - searchRadius, 200 - searchRadius, 2 * searchRadius, 2 * searchRadius)).clone();
	Mat1b centerArea2 = centerArea.clone();

	GaussianBlur(centerArea, centerArea, Size(3, 3), 1, 1);

	// ����ֲ���ֵ��ʹ�м䲿����ȫ��ף��Ӷ�����̽��õ�
	adaptiveThreshold(centerArea, centerArea, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 9, 0);

	// �������������ǱȽ�ð�յġ�����Ϊ����֤���������ģ������������ͬʱ�޷�ȷ�������ֵ30�ڸ�������������
	// HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 30, 8, 0, 6);
	// �����ĸ����ֱַ���� �ֱ��ʣ��������Ϊ�����ĵ�����������Բ֮����С���룬canny�ĵ���ֵ��ͶƱ���ۻ����������ж��ٸ������ڸ�Բ����Բ��С�뾶��Բ���뾶
	// һ��ͶƱֵ�� 15 �� 20�䣬��������ӽ��ر�ƫ�����м�Բ�����Բ������Ҫ����ͶƱֵ�ˣ�Ŀǰ����ӽǱȽ�����������ֻ��Ҫ����20
	HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 150, 18, 10, 25);


	// ʵ���������ǿ��Լ����Ż��ģ��������������κ�Բ������¾ͳ������������򣬼���ͶƱ�������Ŵ�Բ���뾶

	// Ѱ��������������Ǹ�Բ����Ϊ���ǵ�׼�ģ����Ƴ�Բ
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


	// 3. ���淽�����о��û���ǰ�� bim+֧�ֵ㷽����� Ȼ�������ȡ��Բ
	// ��tLine���з��ࣨ���Ƕȷ��ࣩ�����簴��40��һ������������9������ 
	int sangle = 40;
	vector<vector<int>> sangles_(360 / sangle);
	for (int ki = tLines.size() - 1; ki >= 0; --ki)
	{

		// ����нǵ�cosֵ
		int xd = tLines[ki][2] - ac_center.x;
		int yd = ac_center.y - tLines[ki][3];
		// ֵ���� 0~360֮��
		float vangle = fastAtan2(yd, xd);
		sangles_[(int)vangle / sangle].push_back(ki);
	}
	// ����Ҫ����ҪһЩû�е������
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
	// ������Բ�����֧�ֵ�����Զ��̶ȣ�����ĳ̶��ǳ�����Բ�����
	float acceptThresh = 8;
	// �洢��ǰ��֧�ֵ����
	int nowSnum = 0;
	// �洢���֧�ֵ���Ǹ���Բ����Ӿ���
	RotatedRect bestEl;
	// �洢���յ�֧�ֵ�
	vector<int> bestSupportPoints;
	// ������������Ĵ���
	for (int ri = 0; ri < ranTimes; ri++)
	{
		// �洢Ҫѡȡ����������5��������ȡֵ
		vector<int> ks;
		// ������ʼ�ĵ�
		int kindex = random(asize - 1);
		ks.push_back(kindex);
		for (int ki = 0; ki < 4; ki++)
		{
			// ����1~2���������Ҳ������ѡȡ����������ѡ��������ǰѡ��������������λ�ã���֤��һ���ĽǶ�
			kindex = (kindex + random(2)) % asize;
			ks.push_back(kindex);
		}
		// ��ÿ����ѡ�����������ѡȡһ���������Բ���
		vector<Point> candips;
		for (int ki = 0; ki < 5; ki++)
		{
			// ks[ki]ָ���ѡȡ������������ţ�sangles[�������]�õ��ľ������������������е����ţ���������������������ľ�����������ĵڼ�����
			vector<int> shanqu = sangles[ks[ki]];
			int sii = random(shanqu.size() - 1);
			Vec4f vvvv = tLines[shanqu[sii]];
			Point ssss = Point(vvvv[2], vvvv[3]);
			candips.push_back(ssss);
		}
		// �����Բ�����֧�ֵ����
		RotatedRect rect = fitEllipse(candips);

		// �洢֧�ֵ�
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
	// ���������ѭ�����ܵó�֧�ֵ������Ǹ������Բ��������ʾ����

	// ���õ���֧�ֵ�����������
	vector<Point> supporters;
	for (int ki = 0; ki < bestSupportPoints.size(); ki++)
	{
		int tLineIndex = bestSupportPoints[ki];
		Point supporter = Point(tLines[tLineIndex][2], tLines[tLineIndex][3]);
		supporters.push_back(supporter);
	}
	bestEl = fitEllipse(supporters);




	// ��������� �Զ���ֵ ��mser���������鷢�֣���������и��ֱ��mser��
	Mat roi_thresh = roi_dst.clone();
	Mat roi_thresh_mser = roi_dst.clone();
	adaptiveThreshold(roi_thresh, roi_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);


	//��ȡ�Զ����
	Mat e_element = getStructuringElement(MORPH_ELLIPSE, Size(1, 1)); //��һ������MORPH_RECT��ʾ���εľ���ˣ���Ȼ������ѡ����Բ�εġ������͵�

	erode(roi_thresh, roi_thresh, e_element);



	// mser���
	vector<RotatedRect> rrects;
	std::vector<cv::Rect> candidates = mser(roi_thresh, rrects);

	int file_index = 0;
	for (int i = 0; i < candidates.size(); ++i)
	{

		// ���Ϸ���
		if (candidates[i].x <= 0 || candidates[i].y <= 0 || candidates[i].width <= 0 || candidates[i].height <= 0 
			|| candidates[i].x + candidates[i].width > roi_dst.cols || candidates[i].y + candidates[i].height > roi_dst.rows)
		{
			continue;
		}

		// ɸѡ�����Բ����ĵ�
		Point ncenter = Point(candidates[i].x + candidates[i].width / 2, candidates[i].y + candidates[i].height / 2);
		Point newsc = origin2el(bestEl.center, bestEl.angle / (float)180 * pi, ncenter);
		float ndistance = sqrt(pow(newsc.x, 2) / pow(bestEl.size.width / 2, 2) + pow(newsc.y, 2) / pow(bestEl.size.height / 2, 2));
		// �������Բ��Ҫ���ڲ���һ�£�ȥ�������զ�㣬����̶���
		if (ndistance >= 0.86) { continue; }



		// �ѷ��ϵ�����洢����

		Mat mser_item = roi_dst(candidates[i]);
		file_index += 1;
		_mkdir(dirPath.c_str());
		// д���ս��
		imwrite(dirPath + "\\" + int2str(file_index) + ".jpg", mser_item);
		// ��ԭͼҲ��д��ȥ
		imwrite(dirPath + "\\" + "origin.jpg", image_1);
	}

}


// �������ݵĵ������
// writeSingleImg
int writeSingleImg()
{
	vector<string> aa;

	aa.push_back("13 04");

	for (int i=0;i<=aa.size()-1;i++) 
	{
		cout << "��ǰ׼�����ǣ� " << aa[i] << endl;
		writeImg("D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\13\\"+aa[i]+".jpg", "D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\msers\\1");
	}
	return 0;
}



// ����mser����
// makeMserData
int makeMserData()
{

	
	//glob(images_folder + "Lo3my4.*", names);

	string outputPath = "D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\msers\\";
	string basePath = "D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\";
	string sorted_imgs_path = "D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\sorted_imgs.txt";
	// �����·���洢���������������ݣ�����Ϊ���Լ��궨ʱ����ӷ��������
	string positive_base = "D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\msers\\positives\\";
	vector<string> names;

	ofstream out(outputPath + "break_points.txt", ios::app);

	vector<string> alls = readTxt(sorted_imgs_path);
	int rotation_folderIndex = 0;
	int no_rotation_folderIndex = 0;
	int noise_folderIndex = 0;
	// 595ǰ�Ķ���������595 �� 913 �����һ������Ҫ��ת��
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
		cout << "�����Ҫ�������޸�������ֵ" << endl;
		cout << "��ǰ��i�� -- " << i << endl;
		cout << "��ǰ��no_rotation_folderIndex�� -- " << no_rotation_folderIndex << endl;
		cout << "��ǰ��noise_folderIndex�� -- " << noise_folderIndex << endl;
		cout << "��ǰ��rotation_folderIndex�� -- " << rotation_folderIndex << endl;
		cout << "��ǰ��path�� -- " << imgReadPath << endl;

		// ��ϵ��ļ���д��ϵ�

		if (out.is_open())
		{
			out << "======================================================" << endl;
			out << "rotation_folderIndex = " << int2str(rotation_folderIndex) << endl;
			out << "no_rotation_folderIndex = " << int2str(no_rotation_folderIndex) << endl;
			out << "noise_folderIndex = " << int2str(noise_folderIndex) << endl;
			out << "i = " << int2str(i) << endl;
		}
		dirPath = dirPath + addtion_part;
		// �ȴ���9���ļ��еĸ����ļ���
		_mkdir((positive_base + addtion_part).c_str());
		// ����9�����ļ���
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