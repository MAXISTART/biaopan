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





// MserĿ���� + nms
std::vector<cv::Rect> mser(cv::Mat srcImage)
{


	std::vector<std::vector<cv::Point> > regContours;
	
	// ����MSER����
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 10, 500, 0.5, 0.3);


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
		if (b_size < 500 && b_size > 50)
		{
			br = Rect(br.x - 3, br.y - 3, br.width + 6, br.height + 6);
			keeps.push_back(br);
		}
		// ��΢�÷���������һ��
		
		//keeps.push_back(br);
	}
	// ��nms����
	nms(keeps, 0.7);
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





// ����㵽ֱ�ߵľ���
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
	candidates = mser(image_1);
	for (int i = 0; i < candidates.size(); ++i)
	{
		rectangle(image_1, candidates[i], Scalar(255, 255, 255), 1);	
	}
	imshow("test", image_1);
	waitKey(0);
	return 0;
}



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

int zhujhanshu ()
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
	string picName = "16 28.jpg";
	names.push_back("D:\\VcProject\\biaopan\\data\\raw\\newData\\images\\newData\\16\\" + picName);
	//names.push_back("D:\\VcProject\\biaopan\\imgs\\002.jpg");
	int scaleSize = 8;
	for (const auto& image_name : names)
	{
		//string name = image_name.substr(image_name.find_last_of("\\") + 1);
		//name = name.substr(0, name.find_last_of("."));

		Mat3b image_1 = imread(image_name);
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


		imshow("gray_clone", gray_clone);
		waitKey();
		Mat1b gray_clone_add = gray_clone.clone();
		threshold(gray_clone, gray_clone, 0, 255, CV_THRESH_OTSU);
		adaptiveThreshold(gray_clone_add, gray_clone_add, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 9, 0);
		imshow("gray_clone", gray_clone_add - gray_clone);
		waitKey();
		waitKey();

		gray_clone = gray_clone_add - gray_clone;
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


		if (ellsYaed.size() == 0)
		{
			cout << "��ⲻ����Բ���밴������˳�����" << endl;
			cout << "--------------------------------" << endl;
			system("pause");
			return 0;
		}

		// ��չ������Χ���Գ���Ϊֱ��������������
		int index = 0;
		Mat1b gray_clone2;
		cvtColor(image_1, gray_clone2, CV_BGR2GRAY);
		namedWindow("roi");
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





		while(index < el_size ) {
			Ellipse& e = ellsYaed[index];
			int g = cvRound(e._score * 255.f);
			Scalar color(0, g, 0);
			// �ҵ�����
			int long_a = e._a >= e._b ? cvRound(e._a) : cvRound(e._b);
			rectangle(resultImage, Rect(cvRound(e._xc)-long_a, cvRound(e._yc)-long_a, 2 * long_a, 2 * long_a), color, 1);

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
			// Canny(roi_2, roi_2, 50, 150, 3); // Apply canny edge//��ѡcanny����

			Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
			vector<Vec4f> lines_std;
			vector<Vec4f> lines_dst;

			// Detect the lines
			ls->detect(roi_2_add - roi_2, lines_std);
			// Show found lines
			Mat drawnLines = roi_3.clone();
			

			Vec2f e_center = Vec2f(200, 200);
			cout << "lines_std.size() : " << lines_std.size() << endl;

			for (int j=0;j<lines_std.size();j++) 
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
					// ����β��
					circle(drawnLines, Point(dl[2], dl[3]), 2, color, -1);
				}
			}
			circle(drawnLines, Point(200, 200), 4, color, -1);

			// �������г����е����
			ls->drawSegments(drawnLines, lines_dst);
			cout << "size: " << lines_dst.size() << endl;

			
			
			index += 1;


			// ������ѡ�� ����֧�ֵ� ���� 35 �ģ�ͬʱ�����ģ�Ȼ�� ���Ҳ�����޶�һ�£�һ��ѡȡ�����󣬳�������Բ��
			cout << "Ŀǰ������ǣ� " << (el_dst._a * el_dst._b) << "   ���ڵ������: " << (e._a * e._b) << endl;
			if (lines_dst.size() >= min_vec_num && (el_dst._a * el_dst._b) <= (e._a * e._b))
			{
				min_vec_num = lines_dst.size();
				el_dst = e;
				roi_dst = roi_3;
				tLines = lines_dst;
			}

			imshow("roi", drawnLines);
			imshow("Yaed", resultImage);
			cout << "index : " << index << endl;
			waitKey();
		}


		//imwrite(out_folder + name + ".png", resultImage);



		// ��ʾ��ԲЧ��
		int g = cvRound(el_dst._score * 255.f);
		Scalar color(0, 255, 255);
		//ellipse(roi_dst, Point(cvRound(el_dst._xc), cvRound(el_dst._yc)), Size(cvRound(el_dst._a), cvRound(el_dst._b)), el_dst._rad*180.0 / CV_PI, 0.0, 360.0, color, 2);
		
		Mat roi_center = roi_dst.clone();
		Mat roi_line = roi_dst.clone();
		Mat roi_mser = roi_dst.clone();



		// ��������� �Զ���ֵ ��mser���������鷢�֣��������û��ֱ��mser��Ч
		Mat roi_thresh = roi_dst.clone();
		Mat roi_thresh_mser = roi_dst.clone();
		// ʵ��֤�� ADAPTIVE_THRESH_GAUSSIAN_C �� ADAPTIVE_THRESH_MEAN_C �ָ�ĸ��ã���Ե���ӹ⻬
		adaptiveThreshold(roi_thresh, roi_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);
		//erode(roi_thresh, roi_thresh, Mat(2, 2, CV_8U), Point(0, 0), 1);
		//medianBlur(roi_thresh, roi_thresh, 3);
		imshow("roi_thresh", roi_thresh);
		waitKey(0);
		

		vector<Rect> ccs = mser(roi_thresh);
		for (int cci=0;cci < ccs.size();cci++) 
		{
			// ��ɸѡ�����Բ����ĵ㣬�������mser����һ��

			rectangle(roi_thresh_mser, ccs[cci], Scalar(255,255,255), 1);
			// Ȼ���ٰѾ��ο������һ�㣬���ܰ������������򶼵õ���
		}
		imshow("roi_thresh_mser", roi_thresh_mser);
		waitKey(0);





		// 2. houghԲ����ȡ��ȷ���ģ�lsd��ȡ�ֱ�ߣ�ָ���ߣ�

		/**
			������(200, 200)�������������һ���뾶������׼��Բ��
		**/
		int searchRadius = 25;
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
		
		// �洢��Щ���߶�
		vector<int> backs = vector<int>(tLines.size());
		// �洢��Щǰ�߶ԣ�����֮��ļ���
		vector<int> fronts = vector<int>(tLines.size());

		// ��ʼ����Щ�Ե�ֵ��Ĭ��Ϊ-1
		for (int k=tLines.size()-1;k>=0;--k)
		{
			backs[k] = -1;
			fronts[k] = -1;
		}

		for (int i=tLines.size() - 1;i >= 0; --i) 
		{
			Vec4f& line1 = tLines[i];
			// �����뾶�ڣ������������Լ�������ߣ�Ȼ�����е��������ɵĽǶȲ��ܴ��ĳ��ֵ������ȡ�����Լ���̵���Щ�ߣ�����ǿ��������ǲ����Ѿ���Ϊ�˱��˵ĺ�����
			int mIndex = i;
			float mDis = 1000;
			for (int j = tLines.size() - 1; j >= 0; --j) 
			{
				if (i == j)
					continue;
				Vec4f& line2 = tLines[j];

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
		for (int i = tLines.size() - 1; i >= 0; --i)
		{
			//cout << "backs-" << i << " : " << backs[i] << endl;
			//cout << "fronts-" << i << " : " << fronts[i] << endl;
			int n = 0;
			for (int j = tLines.size() - 1; j >= 0; --j) {
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
				Vec4f ln = tLines[li];
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
				total_length += point2point(tLines[li]);
			}
			if (total_length > maxLength) { maxLength = total_length; maxLine = goal_line; }
		}
		// ���������
		Scalar aa(255, 255, 255);
		for (int j = 0; j <= maxLine.size() - 1; ++j)
		{
			int li = maxLine[j];
			Vec4f ln = tLines[li];
			Point point1 = Point(ln[0], ln[1]);
			Point point2 = Point(ln[2], ln[3]);
			cv::line(roi_line, point1, point2, aa, 2);
		}
		imshow("roi_line", roi_line);
		waitKey();

		// ������ߵĽ�β����ָ���ߵ�β��
		Vec4f last_part = tLines[maxLine[maxLine.size() - 1]];
		Point lastPoint = Point(last_part[2], last_part[3]);


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

		imshow("roi_line", roi_line);
		waitKey(0);
		waitKey(0);






		// mser���
		std::vector<cv::Rect> candidates;
		imshow("roi_mser", roi_dst);
		waitKey(0);
		candidates = mser(roi_dst);
		// �洢ȷ�������ֵ������Լ����ǵ����ĵ㻹�����ǵ���Ӧ
		vector<Rect> numberAreas;
		vector<Point> numberAreaCenters;
		vector<int> numberAreaRes;


		// ������ʾ
		for (int i = 0; i < candidates.size(); ++i)
		{
			// ��ɸѡ�����Բ����ĵ�
			Point ncenter = Point(candidates[i].x + candidates[i].width / 2, candidates[i].y + candidates[i].height / 2);
			Point newsc = origin2el(bestEl.center, bestEl.angle / (float)180 * pi, ncenter);
			float ndistance = sqrt(pow(newsc.x, 2) / pow(bestEl.size.width / 2, 2) + pow(newsc.y, 2) / pow(bestEl.size.height / 2, 2)) - 1 ;
			if (ndistance >= 0) { continue; }


			rectangle(roi_mser, candidates[i], color, 1);
			// �ѷ������ֵ������ѡ����
			Mat mser_item = roi_dst(candidates[i]);
			vector<float> v = getHogData(mser_item);
			Mat testData(1, 144, CV_32FC1, v.data());
			int response = svm->predict(testData);
			// ��ʾ����
			if (response >= 0)
			{
				rectangle(roi_mser, candidates[i], Scalar(255, 255, 255), 2);
				circle(roi_mser, ncenter, 2, color, -1);
				numberAreaCenters.push_back(ncenter);
				numberAreas.push_back(candidates[i]);
				// ������Ҫ�޸ģ�ֻ�Ƿ������
				if (response == 3) { response = 2; }
				numberAreaRes.push_back(response);
				cout << "��ǩΪ�� " << response << endl;
				imshow("roi_mser", roi_mser);
				waitKey(0);
			}
		}
		RotatedRect box = fitEllipse(numberAreaCenters);
		ellipse(roi_mser, box, Scalar(255, 255, 255), 1, CV_AA);
		ellipse(roi_mser, Point(cvRound(el_dst._xc), cvRound(el_dst._yc)), Size(cvRound(el_dst._a), cvRound(el_dst._b)), el_dst._rad*180.0 / CV_PI, 0.0, 360.0, color, 2);

		imwrite("D:\\VcProject\\biaopan\\data\\temp\\1\\" + picName, roi_mser);
		imshow("roi_mser", roi_mser);
		waitKey(0);





		// 4. ��ֵ��ȡ
		// ���ҳ��任�㣬Ҳ��������Ϊ͸�ӱ任�ĵ㣬Ȼ���Ǳ任
		// �������ֵ�������Ѱ�ɶԴ��ڵ�
		// �洢��Щ�ϲ���������Լ����ǵ���Ӧֵ
		struct MergeArea
		{
			// ����index�洢��candidate�е����
			int aIndex;
			int bIndex;
			// �洢�ܹ�����Ӧֵ
			float response;
			// �洢���ǵ�����
			Point center;
		};
		vector<MergeArea> mms;
		
		for (int kii=0;kii<numberAreaRes.size();kii++) 
		{
			int response = numberAreaRes[kii];
			// ��������0�ĵ�
			if (response == 0) { continue; }
			Point numcenter = numberAreaCenters[kii];
			Point newNumCenter;
			// ������벻��̫��
			float min_distance = 30;
			// �洢��õ������̵���Ч�㣬���û�еľ���-1�����Ļ��
			int cc = -1;
			Mat forellipse_1 = roi_mser.clone();
			rectangle(forellipse_1, numberAreas[kii], Scalar(255, 255, 255), 2);
			// cout << "mm.kjj: " << response << endl;
			imshow("forellipse_1", forellipse_1);
			waitKey(0);
			for (int kjj = 0; kjj < numberAreaRes.size(); kjj++)
			{
				if (numberAreaRes[kjj] > 0) { continue; }
				float kjj_distance = point2point(numcenter, numberAreaCenters[kjj]);
				// �����������Ϊ���Ļ��
				// cout << "mm.kjj_distance: " << kjj_distance << endl;
				// cout << "mm.min_distance: " << min_distance << endl;
				// cout << "mm.kjj: " << numberAreaRes[kjj] << endl;
				if (kjj_distance < min_distance )
				{
					min_distance = kjj_distance;
					cc = kjj;
				}
				rectangle(forellipse_1, numberAreas[kjj], Scalar(255, 255, 255), 2);
				imshow("forellipse_1", forellipse_1);
				waitKey(0);

			}
			Point mm_center;
			if (cc >= 0)
			{
				mm_center = Point((numcenter.x + numberAreaCenters[cc].x) / 2, (numcenter.y + numberAreaCenters[cc].y) / 2);
			}
			else if (cc < 0)
			{
				mm_center = numcenter;
			}
			MergeArea mm = MergeArea{ kii, cc, response / (float)10, mm_center };
			cout << "==========================" << endl;
			cout << "mm.response: " << mm.response << endl;
			mms.push_back(mm);
		}	
		waitKey();	



		// �ó��任��
		vector<Point2f> elPoints;
		vector<float> resNums;
		Mat forellipse_2 = roi_mser.clone();
		ellipse(forellipse_2, bestEl, Scalar(255, 255, 255));
		for (int kii = 0; kii < mms.size(); kii++)
		{
			Point center_ = Point(bestEl.center.x, bestEl.center.y);
			Point elPoint = anchor_on_el_line(bestEl.size.width / 2, bestEl.size.height / 2, bestEl.angle / (float)180 * pi, bestEl.center, center_, mms[kii].center);
			elPoints.push_back(elPoint);
			resNums.push_back(mms[kii].response);
			circle(forellipse_2, elPoint, 3, Scalar(0, 0, 0), -1);
			imshow("forellipse_2", forellipse_2);
			waitKey(0);
		}


		//// ����
		//Point test_point = Point(350, 350);
		//cout << "�뾶�� " << point2point(ac_center, test_point) << endl;
		//line(forellipse_2, ac_center, test_point, Scalar(255, 255, 255));
		//for (int dgree=0;dgree <= 360; dgree++) 
		//{
		//	Point2f ac_center_ = Point2f(ac_center.x, ac_center.y);
		//	Point newp = origin2el(ac_center_, dgree / (float)180 * pi, test_point);
		//	Point nnp = Point(ac_center.x + newp.x, ac_center.y - newp.y);
		//	cout << "�Ƕȣ� " << dgree << endl;
		//	line(forellipse_2, ac_center, nnp, Scalar(255, 255, 255));
		//	imshow("forellipse_2", forellipse_2);
		//	waitKey(0);
		//}

		// ��������ĳ�һ�����ۣ����� ÿ�������������45�ȣ��м����Ķ���Ϊ95�ȡ��뾶��Լ��180�ź������ǿ��Թ�����һ��ͼ����ͬʱ����֪��������ľ���λ��
		vector<Point2f> originPoints(elPoints.size());
		Point pr_center = Point(200, 200);
		int rrr = 160;
		Point pointer = Point(pr_center.x, pr_center.y + rrr);
		int e_pointnums = 7;
		int empty_degree = 90;
		float singgle_d_distance = (360 - empty_degree) / float(e_pointnums-1);
		Point2f pr_center_ = Point2f(pr_center.x, pr_center.y);


		Mat forellipse_3 = roi_mser.clone();
		ellipse(forellipse_3, bestEl, Scalar(255, 255, 255));

		for (int iii = e_pointnums-1;iii >= 0; iii--)
		{
			float ss = singgle_d_distance;
			if (iii == e_pointnums - 1) { ss = empty_degree / 2; }
			for (int kkk = 0; kkk < resNums.size();kkk++) 
			{
				if (iii / (float)10 == resNums[kkk])
				{
					Point e_point_ = origin2el(pr_center_, (e_pointnums - iii) * ss / (float)180 * pi, pointer);
					Point e_point = Point(pr_center.x + e_point_.x, pr_center.y - e_point_.y);
					originPoints[kkk] = e_point;
					circle(forellipse_3, e_point, 3, Scalar(0, 0, 0), -1);
				}
			}
		}
		imshow("forellipse_3", forellipse_3);
		waitKey(0);



		elPoints.push_back(ac_center);
		originPoints.push_back(pr_center);
		Mat forellipse_4 = roi_mser.clone();
		circle(forellipse_4, lastPoint, 3, Scalar(0, 0, 0), -1);
		Mat transform = getPerspectiveTransform(elPoints, originPoints);


		Mat warped;
		warpPerspective(forellipse_4, warped, transform, warped.size(), INTER_LINEAR, BORDER_CONSTANT);

		
		for (int kkk = 0; kkk < originPoints.size(); kkk++)
		{
			circle(warped, originPoints[kkk], 3, Scalar(0, 0, 0), -1);	
		}
		
		imshow("img_trans", warped);
		waitKey(0);

	}






	int yghds = 0;
	return 0;
}






// ��svmdata����ѵ��
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
	// ����ֻѵ��28800�����ݣ�������������
	for (int i=0;i<28800;i++)
	{
		// ����һ��ͼƬ
		vector<string> raw = splitString(raws[i], spliter);
		string src = raw[0];
		int label = str2int(raw[1]);
		Mat mat = imread(src, IMREAD_GRAYSCALE);
		trainingData.push_back(getHogData(mat));
		labels.push_back(label);
		// ������������ǿ�����羵�񣬷�ת��ԭ���ǿ��ܻ������Щ�����
		for (int flipCode=-1; flipCode <2; flipCode++)
		{
			Mat flipMat;
			flip(mat, flipMat, flipCode);
			trainingData.push_back(getHogData(mat));
			labels.push_back(label);
		}
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
int singleTest()
{
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";
	Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
	Mat s1 = imread("D:\\VcProject\\biaopan\\data\\goodImgs\\454\\66.jpg", IMREAD_GRAYSCALE);

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
	// ��ЩͼƬ��Ҫ��ת
	if (if_flip) { flip(image_1, image_1, -1); }
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
	Mat1b gray_clone_add = gray_clone.clone();
	threshold(gray_clone, gray_clone, 0, 255, CV_THRESH_OTSU);
	adaptiveThreshold(gray_clone_add, gray_clone_add, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 9, 0);
	gray_clone = gray_clone_add - gray_clone;

	yaed.Detect(gray_clone, ellsYaed);


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







	// mser���
	std::vector<cv::Rect> candidates;
	candidates = mser(roi_dst);

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
		float ndistance = sqrt(pow(newsc.x, 2) / pow(bestEl.size.width / 2, 2) + pow(newsc.y, 2) / pow(bestEl.size.height / 2, 2)) - 1;
		if (ndistance >= 0) { continue; }



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
int main()
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
	// 595ǰ�Ķ���������595�����һ������Ҫ��ת��
	for (int i = 0; i < 596; i++)
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
		writeImg(imgReadPath, dirPath);
	}
	out.close();
	return 0;
}