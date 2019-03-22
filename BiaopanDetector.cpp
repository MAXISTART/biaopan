
#include <algorithm>
#include "EllipseDetectorYaed.h"
#include <fstream>
#include <direct.h>
#include <unordered_map>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "BiaopanDetector.h"

#define random(x) (rand()%x)+1

using namespace cv::ximgproc;
using namespace std;


// ��ʼ��һЩ����
void BiaopanDetector::initBiaopan()
{
	// ����svm
	svm = cv::ml::SVM::load(modelPath);
	// ��ʼ��lsd
	int    length_threshold = 10;
	float  distance_threshold = 1.41421356f;
	double canny_th1 = 50.0;
	double canny_th2 = 50.0;
	int    canny_aperture_size = 3;
	bool   do_merge = true;
	fld = createFastLineDetector(
		length_threshold,
		distance_threshold,
		canny_th1,
		canny_th2,
		canny_aperture_size,
		do_merge);
}



/*������*/
// ����㵽ֱ�ߵľ��룬��������ĵ���ͼ������ĵ�
float BiaopanDetector::point2Line(Point point, Vec4f& line)
{
	float x1 = line[0];
	float y1 = -line[1];
	float x2 = line[2];
	float y2 = -line[3];

	float A = (y1 - y2) / (x1 - x2);
	float B = -1;
	float C = y1 - x1 * A;
	return abs(A * point.x - B * point.y + C) / sqrt(pow(A, 2) + pow(B, 2));
}
// ����㵽ֱ�ߵľ��룬����ĵ��Ǳ�׼����ĵ�
float BiaopanDetector::point2Line(float x, float y, float x1, float y1, float x2, float y2)
{

	float A = (y1 - y2) / (x1 - x2);
	float B = -1;
	float C = y1 - x1 * A;
	float dis = abs(A * x + B * y + C) / sqrt(pow(A, 2) + pow(B, 2));
	return dis;
}
// ����㵽��ľ���
float BiaopanDetector::point2point(float x1, float y1, int x2, int y2)
{
	return sqrt(pow((x1 - (float)x2), 2) + pow((y1 - (float)y2), 2));
}
float BiaopanDetector::point2point(float x1, float y1, float x2, float y2)
{
	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}
float BiaopanDetector::point2point(int x1, int y1, int x2, int y2)
{
	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}
float BiaopanDetector::point2point(Point point1, Point point2)
{
	return sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2));
}
float BiaopanDetector::point2point(Vec4f line)
{
	return sqrt(pow((line[0] - line[2]), 2) + pow((line[1] - line[3]), 2));
}


/* ------------------------------------ */
// �ϲ�ֱ�ߵĲ���

// 1. ������
void BiaopanDetector::backSearch(vector<bool>& isVisited, vector<int>& backs, vector<int>& goal_line)
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
void BiaopanDetector::frontSearch(vector<bool>& isVisited, vector<int>& fronts, vector<int>& goal_line)
{
	int left = goal_line[0];
	if (fronts[left] >= 0)
	{
		isVisited[fronts[left]] = true;
		goal_line.insert(goal_line.begin(), fronts[left]);
		frontSearch(isVisited, fronts, goal_line);
	}
}


// 3. ������ֱ�߼�ļн�cosֵ
float BiaopanDetector::line2lineAngleCos(Vec4f line1, Vec4f line2)
{
	float leng1 = point2point(line1);
	float leng2 = point2point(line2);
	return ((line1[2] - line1[0]) * (line2[2] - line2[0]) + (line1[3] - line1[1]) * (line2[3] - line2[1])) / leng1 / leng2;
}

/* ------------------------------------ */
// ��ֵ�����һЩ����


// ������ת����
void BiaopanDetector::drawRotatedRect(Mat& drawer, RotatedRect& rrect)
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
std::vector<cv::Rect> BiaopanDetector::rotateMser(cv::Mat& srcImage, vector<cv::RotatedRect>& rrects)
{

	std::vector<std::vector<cv::Point> > regContours;

	// ����MSER����
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 5, 800, 0.5, 0.3);

	// �޶������
	int max_width = srcImage.cols;
	int max_height = srcImage.rows;

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
			int topx = max(0, br.x - 3);
			int topy = max(0, br.y - 3);
			int maxw = min(br.width + 6, max_width - topx);
			int maxh = min(br.height + 6, max_height - topy);
			br = Rect(br.x - 3, br.y - 3, br.width + 6, br.height + 6);
			keeps.push_back(br);
		}
		// ��΢�÷���������һ��

		//keeps.push_back(br);
	}

	imshow("rotateMser", mser_show);
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
				&& rrecs[i].center.x <= keeps[j].x + keeps[j].width && rrecs[i].center.y <= keeps[j].y + keeps[j].height)
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

// MserĿ���� + nms������Ǽ����mser��������ת����
vector<Rect> BiaopanDetector::simpleMser(Mat& srcImage)
{
	Mat show = srcImage.clone();
	// �޶������
	int max_width = srcImage.cols;
	int max_height = srcImage.rows;

	vector<vector<Point> > regContours;

	// ����MSER����
	Ptr<MSER> mesr1 = MSER::create(2, 5, 800, 0.5, 0.3);


	vector<cv::Rect> boxes;
	// MSER���
	mesr1->detectRegions(srcImage, regContours, boxes);
	// �洢����
	vector<Rect> keeps;


	Mat mserMapMat = Mat::zeros(srcImage.size(), CV_8UC1);
	Mat mserNegMapMat = Mat::zeros(srcImage.size(), CV_8UC1);

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
		if (b_size < 800 && b_size > 50)
		{
			// ʵ��֤�����������ŵ�ʱ��ʶ��Ч�����ã�
			int topx = max(0, br.x - mserRadius);
			int topy = max(0, br.y - mserRadius);
			int maxw = min(br.width, max_width - topx + mserRadius * 2);
			int maxh = min(br.height, max_height - topy + mserRadius * 2);
			br = Rect(br.x, br.y, br.width, br.height);
			keeps.push_back(br);	
		}
	}
	
	// ��nms����
	nms(keeps, 0.7);

	for (int i=0;i<keeps.size();i++)
	{
		rectangle(show, keeps[i], Scalar(255, 255, 255), 1);
	}
	//imshow("simpleMser", show);
	return  keeps;
}
/* ------------------------------------ */


// vector<ranglevoter>����ranglevoter�е�voteNum����
bool BiaopanDetector::SortByVote(ranglevoter &v1, ranglevoter &v2)
{
	//��������  
	return v1.voteNum > v2.voteNum;
}

// vector<int>����vector�е�ֵ������
bool BiaopanDetector::SortByUp(upper &v1, upper &v2)
{
	//��������  
	return v1.y > v2.y;
}

// Ѱ��������
void BiaopanDetector::vertical_projection(Mat& input_src, vector<upper>& uppers)
{

	int width = input_src.cols;
	int height = input_src.rows;
	int perPixelValue;//ÿ�����ص�ֵ
	// ��ʼ��
	vector<int> projectValArry(width, 0);

	// ƽ������
	int smooth_thresh = 4;

	// Ѱ����������
	// last�洢��һ�е�ֵ�����統ǰ����ǰһ�������󣨱��統ǰ��Ϊ�գ��ģ�
	// ��ô�ȼ�������̽�����������ĵ㣬�������û̽��������ôȥ��һ��ǰһ�У����ֺ�ǰһ��һ���������Ǳ�֤��������ͻȻ�Ͽ�
	int last = 0;
	for (int col = 0; col < width; ++col)
	{
		projectValArry[col] = last;
		for (int row = 0; row < height; ++row)
		{
			perPixelValue = input_src.at<uchar>(row, col);
			if (perPixelValue > 0 && abs(row - last) <= smooth_thresh)
			{
				projectValArry[col] = row;
				break;
			}
		}
		last = projectValArry[col];
	}

	// ǰ�����Щ0��Ҫ��ƽ����
	for (int col = 0; col < width; ++col)
	{
		if (projectValArry[col] > 0)
		{
			for (int i = 0; i < col; ++i)
			{
				projectValArry[i] = projectValArry[col];
			}
			break;
		}
	}


	/*�½�һ��Mat���ڴ���ͶӰֱ��ͼ����������Ϊ��ɫ*/
	Mat verticalProjectionMat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			perPixelValue = 255;  //��������Ϊ��ɫ��   
			verticalProjectionMat.at<uchar>(i, j) = perPixelValue;
		}
	}

	/*��ֱ��ͼ��������Ϊ��ɫ*/
	//cout << "projectValArry: [";
	for (int i = 0; i < width; i++)
	{
		perPixelValue = 0;  //ֱ��ͼ����Ϊ��ɫ  
		verticalProjectionMat.at<uchar>(projectValArry[i], i) = perPixelValue;
		//cout << projectValArry[i] << ",";

		// װ��uppers���淵��
		uppers.push_back(upper{ i, projectValArry[i] });
	}
	//cout << "]" << endl;
	//imshow("��ȡ������ͼ", input_src);
	//imshow("����������", verticalProjectionMat);
	//waitKey();
}


/* ------------------------------------ */
// �ϲ���ֵ����Ĳ���
// 1. ���ӵ�����
void BiaopanDetector::joinSearch(vector<bool>& isVisited, vector<int>& goal_set, vector<vector<int>>& joinTable)
{
	vector<int>::iterator it;
	int i = goal_set[goal_set.size() - 1];
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


// 2.�жϽǶȣ�����ı�׼����ϵ�������� y �� x��������� �Ա�׼����ϵ�е����������м���Ϊ0���ᣬ˳ʱ����ת�ĽǶ�
float BiaopanDetector::getVecAngle(float dy, float dx)
{
	float vecAngle = atan2(dy, dx) * 180 / pi;
	if (vecAngle <= -90) { vecAngle = -vecAngle - 90; }
	else if (vecAngle >= 0) { vecAngle = -vecAngle + 270; }
	else if (vecAngle < 0 && vecAngle > -90) { vecAngle = -vecAngle + 270; }
	return vecAngle;
}

// 3.vector<SingleArea>����center.xֵ������
bool BiaopanDetector::SortByX(SingleArea &v1, SingleArea &v2)//ע�⣺�������Ĳ���������һ��Ҫ��vector��Ԫ�ص�����һ��  
{
	//��������  
	return v1.center.x < v2.center.x;
}


// 4.vector<MergeArea>����angleֵ������
bool BiaopanDetector::SortByAngle(MergeArea &v1, MergeArea &v2)//ע�⣺�������Ĳ���������һ��Ҫ��vector��Ԫ�ص�����һ��  
{
	//��������  
	return v1.angle < v2.angle;
}

// 5.vector<MergeArea>����responseֵ�����򣬽���
bool BiaopanDetector::SortByRes(MergeArea &v1, MergeArea &v2)//ע�⣺�������Ĳ���������һ��Ҫ��vector��Ԫ�ص�����һ��  
{
	//��������  
	return v1.response > v2.response;
}



// ��ȡһ��cell��������dn��ָ�ж��ٸ�����
vector<float> BiaopanDetector::getCellData(Mat& mag, Mat& angle, int r, int c, int cellSize, int dn)
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
vector<float> BiaopanDetector::getHogData(Mat& originImg)
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

/* ------------------------------------ */



/* ------------------------------------ */
// vector<Ellipse>������Բ�������Сֵ�����򣬽���
bool BiaopanDetector::SortByEllipseArea(Ellipse& e1, Ellipse& e2)
{
	//��������  
	return (e1._a*e1._b) > (e2._a*e2._b);
}

/* ------------------------------------ */
// ����iou+nms�����Բ�ķ������⣬������Բ��Ӿ��ε�iou������
double BiaopanDetector::iou(const Rect& r1, const Rect& r2)
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

void BiaopanDetector::nms(vector<Rect>& proposals, const double nms_threshold)
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


// ����������ת��ĵ㣬��������������Ǳ�׼����ϵ�������Ҳ�Ǳ�׼����ϵ,�ǵ�����ʱ����ת
Point BiaopanDetector::rotate(float theta, Point a)
{
	float x = a.x;
	float y = a.y;
	return Point(int(cos(theta)*x - sin(theta)*y), int(sin(theta)*x + cos(theta)*y));
}

// ����������ת��ĵ㣬��������������Ǳ�׼����ϵ�������Ҳ�Ǳ�׼����ϵ,�ǵ�����ʱ����ת
Point BiaopanDetector::rotate(float theta, float x, float y)
{
	return Point(cos(theta)*x - sin(theta)*y, sin(theta)*x + cos(theta)*y);
}


// �л�����Բ���꣬���������������ͼ������ϵ��������Ǳ�׼����ϵ����ƽ�ƺ���ת��
Point BiaopanDetector::origin2el(Point2f& center, float theta, Point& origin)
{
	float x = origin.x;
	float y = -origin.y;
	return rotate(theta, x - center.x, y + center.y);
}

// �л���ͼ�����꣬��������������Ǳ�׼����ϵ���������ͼ������ϵ������ת��ƽ�ƣ�
Point BiaopanDetector::el2origin(Point2f& center, float theta, Point& el)
{
	Point origin = rotate(theta, el.x, el.y);
	float x = origin.x;
	float y = -origin.y;
	return Point(x + center.x, y + center.y);
}


// ��Բ��⣬����ԭͼ��ͼ�����ŵı����Լ�ģ������ĳ̶ȣ�Ҫ����Ķ�������ֻ����Բ
vector<Ellipse> BiaopanDetector::getElls(Mat& img, int& scaleSize, int blurSize)
{

	vector<Ellipse> output;

	Size sz = img.size();
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

	Mat1b edge;
	Mat1b image;

	// �Ƚ�������
	resize(img, image, Size(img.size[1] / scaleSize, img.size[0] / scaleSize));
	// �ȱ���һ��Ԥ��ͼ
	Mat3b image_show;
	cvtColor(image, image_show, COLOR_GRAY2BGR);
	// �ٽ���һ��ģ��
	GaussianBlur(image, image, Size(blurSize, blurSize), 3, 3);

	Canny(image, edge, 3, 9, 3);

	yaed.Detect(edge, ellsYaed);


	// ������� iou+nms ����Բ�ֿ�


	if (ellsYaed.size() == 0) { return output; }
	// �����Բ����Ӿ���
	vector<Rect> rects;

	for (int index = 0; index < ellsYaed.size(); index++)
	{
		Ellipse& e = ellsYaed[index];
		// �ҵ��������
		int long_a = e._a >= e._b ? cvRound(e._a) : cvRound(e._b);
		int short_b = e._a < e._b ? cvRound(e._a) : cvRound(e._b);
		rects.push_back(Rect(cvRound(e._xc) - long_a, cvRound(e._yc) - long_a, 2 * long_a, 2 * long_a));
	}
	nms(rects, 0.7);
	// �����ƺ�ľ�����Ѱ�ҷ���Ҫ�����Բ
	vector<Ellipse> ells;
	for (int j = 0; j < rects.size(); j++)
	{
		float karea = rects[j].width * rects[j].height;
		float max_size = -1;
		Ellipse max_el;
		for (int i = 0; i < ellsYaed.size(); i++)
		{
			Ellipse& e = ellsYaed[i];
			// �ҵ��������
			int long_a = e._a >= e._b ? cvRound(e._a) : cvRound(e._b);
			int short_b = e._a < e._b ? cvRound(e._a) : cvRound(e._b);
			// �������ȹ�С����ȥ
			if (short_b / (float)long_a < el_ab_p) { continue; }

			if (ellsYaed[i]._xc >= rects[j].x && ellsYaed[i]._yc >= rects[j].y
				&& ellsYaed[i]._xc <= rects[j].x + rects[j].width && ellsYaed[i]._yc <= rects[j].y + rects[j].height)
			{
				float area = long_a * short_b;
				if (area / karea >= max_size)
				{
					max_size = area / karea;
					max_el = ellsYaed[i];
				}
			}
		}
		// ��� max_size ��Ȼ����-1˵�������Ӿ�������û������Ҫ�����Բ
		if (max_size > 0) { ells.push_back(max_el); }
	}
	// ���������� std ���У�Ҫ��Ȼ�͵���cv��sort�ˣ�ͬʱ�ȽϷ��������Ǹ�static�ķ���
	std::sort(ells.begin(), ells.end(), SortByEllipseArea);



	// ������Բ��ԭ��ȥ
	for (int i = 0; i < ells.size(); i++)
	{
		Ellipse e = ells[i];
		e._xc *= scaleSize;
		e._yc *= scaleSize;
		e._a *= scaleSize;
		e._b *= scaleSize;
		output.push_back(e);
	}

	// ������
	yaed.DrawDetectedEllipses(image_show, ells);
	//imshow("ellipse detect", image_show);
	//waitKey();

	// ��Ŀ����Բ����ȥ
	return output;

}

// ���ݱ���ID ��ȡ �������� roi_dst���������ԭʼͼ�񣨾����ҶȻ��������̵�id��������� һ������ roi_dst �Լ� ��ת�Ƕ�
Mat BiaopanDetector::getRoiDst(Mat& img, string& id, float& rotation)
{

	//����QRDetector����ͼƬ
	vector<BCoder> bcs = qrdetect.detect(img);
	Mat roi_dst;
	for (int i=0;i<bcs.size();i++)
	{
		BCoder bc = bcs[i];
		// ��ȡλ����ת�Ƕ�
		rotation = bc.rotation;
		Point center = bc.center;


		// �����ǽ�����������Ϣ���ж����ǲ���Ҫ�ҵı��̣�������Ǿͻ���һ����ά����continue������Ǿͼ���
		//.......//
		// ��������һ��֮��Ϳ���ȷ�������ά���Ǵ���������Ҫ�ҵı��̣����Խ�����ľ��Ƕ�ȡ���̵�����



		// �Ƚ�����ת
		Mat rotationMatrix;
		// ��getRotationMatrix2D�У��Ƕ�Ϊ����˳ʱ�룻�Ƕ�Ϊ������ʱ�롣����������Ĭ�ϲ��ù�
		rotationMatrix = getRotationMatrix2D(center, -rotation, 1);//������ת�ķ���任���� 

		// ��Ȼ����Բ��������㣬��ȻҲ������ �����ά��������ȷ�� ���ؾ��룬����Ϊ���̴�Ŵ�СҲ�ǿ�֪�ģ��Ϳ����Ƴ����̵Ĵ���λ�ã����ǲ�����
		int scaleSize = initScaleSize;
		// Ĭ����ģ����Ϊ7�������������һ��
		vector<Ellipse> ells = getElls(img, scaleSize, 7);
		// Ѱ�Ұ�����ά�����ĵ���Ǹ���Բ���ҵ��ĵ�һ����break
		Ellipse el_dst;
		bool detectEllSuccess = false;
		for (int j=0;j<ells.size();j++)
		{
			Ellipse e = ells[j];
			Point2f el_center = Point2f(e._xc, e._yc);
			Point newsc = origin2el(el_center, e._rad, center);
			float ndistance = sqrt(pow(newsc.x, 2) / pow(e._a, 2) + pow(newsc.y, 2) / pow(e._b, 2));
			cout << "ndistance: " << ndistance << endl;
			if (ndistance <= 1)
			{
				detectEllSuccess = true;
				el_dst = e;
				break;
			}
		}
		if (!detectEllSuccess) 
		{
			cout << "�ڼ��IDΪ��..(Ŀǰ��δ�����Ϣ����)�Ķ�ά�� ������Բ���ʧ�ܣ�����ǶȻ��߹���" << endl;
			continue;
		}

		// ����Ŀ����Բ����roi_dst���͵�ָ����ģ����ȥ

		/* ����Ĳ���ֻ����������roi_dst */
		int long_a = el_dst._a >= el_dst._b ? cvRound(el_dst._a) : cvRound(el_dst._b);
		int short_b = el_dst._a < el_dst._b ? cvRound(el_dst._a) : cvRound(el_dst._b);
		int r_x = max(0, (cvRound(el_dst._xc) - long_a));
		int r_y = max(0, (cvRound(el_dst._yc) - long_a));
		// �����ߴ�Ļ����ʵ���С
		int r_mx = min(img.cols, r_x + 2 * long_a);
		int r_my = min(img.rows, r_y + 2 * long_a);
		int n_width = min(r_mx - r_x, r_my - r_y);
		// ��ȡĿ������
		roi_dst = img(Rect(r_x, r_y, n_width, n_width));

		
		resize(roi_dst, roi_dst, Size(roi_width, cvRound(float(roi_dst.cols) / float(roi_dst.rows) * roi_width)));
		//imshow("roi_dst", roi_dst);
		//waitKey();
		/* ----------------------------------------------------------  */


		// ���һ��mat�����mat����ԭ����ά����ĵ���Ϣ
		cvtColor(img, BiaopanDetector::rs1, COLOR_GRAY2BGR);
		circle(rs1, bc.a, 10, Scalar(100, 0, 0), -1);
		circle(rs1, bc.b, 10, Scalar(0, 200, 0), -1);
		circle(rs1, bc.c, 10, Scalar(0, 0, 120), -1);
		circle(rs1, bc.d, 10, Scalar(50, 150, 150), -1);
		rs1 = rs1(Rect(r_x, r_y, n_width, n_width));
		resize(rs1, rs1, Size(roi_width, cvRound(float(rs1.cols) / float(rs1.rows) * roi_width)));
		

		// ��Ϊ������̾���Ҫ�ҵģ����Ժ���Ķ�ά��Ҳ�����ټ����
		break;
	}

	return roi_dst;
}


// ֱ�Ӿ�������̵�Բ������ͬʱ�ḳֵac_center
RotatedRect BiaopanDetector::getAcOutline(Mat& roi_dst)
{


	int scaleSize = 1;
	vector<Ellipse> ells = getElls(roi_dst, scaleSize, 5);


	RotatedRect acEl;
	acEl.angle = ells[0]._rad / pi * 180;
	acEl.center = Point(ells[0]._xc, ells[0]._yc);
	acEl.size = Size2f(2 * ells[0]._a, 2 * ells[0]._b);

	
	BiaopanDetector::ac_el = acEl;
	return acEl;
}



// �������ļ�⣬�������roi_dst��������Ǳ�������λ��
Point BiaopanDetector::getAcCenter(Mat& roi_dst)
{
	Point ac_center = Point(roi_width / 2, roi_width / 2);

	vector<Vec3f> circles;
	Mat centerArea = roi_dst(Rect(roi_width / 2 - center_search_radius, 
		roi_width / 2 - center_search_radius, 2 * center_search_radius, 2 * center_search_radius)).clone();


	GaussianBlur(centerArea, centerArea, Size(3, 3), 1, 1);
	// ����ֲ���ֵ��ʹ�м䲿����ȫ��ף��Ӷ�����̽��õ�
	adaptiveThreshold(centerArea, centerArea, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 9, 0);
	// �������������ǱȽ�ð�յġ�����Ϊ����֤���������ģ������������ͬʱ�޷�ȷ�������ֵ30�ڸ�������������
	// HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 30, 8, 0, 6);
	// �����ĸ����ֱַ���� �ֱ��ʣ��������Ϊ�����ĵ�����������Բ֮����С���룬canny�ĵ���ֵ��ͶƱ���ۻ����������ж��ٸ������ڸ�Բ����Բ��С�뾶��Բ���뾶
	// һ��ͶƱֵ�� 15 �� 20�䣬��������ӽ��ر�ƫ�����м�Բ�����Բ������Ҫ����ͶƱֵ�ˣ�Ŀǰ����ӽǱȽ�����������ֻ��Ҫ����20
	HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 150, 18, 10, 25);
	Point f_center = Point(center_search_radius / 2, center_search_radius / 2);
	float f2ac = 200;
	for (int i = circles.size() - 1; i >= 0; --i)
	{
		Point mCenter = Point(circles[i][0], circles[i][1]);
		// �������ְ취��һ����ȡ����f_center��̣�һ���������С,�Ȳ��õڶ���
		//float mDistance = point2point(mCenter, f_center);
		//if (mDistance < f2ac)
		//{
		//	f2ac = mDistance;
		//	ac_center = Point(mCenter.x + 200 - searchRadius, mCenter.y + 200 - searchRadius);
		//}
		// Ҳ������� f2ac �洢����Բ�������
		if (circles[i][2] < f2ac)
		{
			f2ac = circles[i][2];
			ac_center = Point(mCenter.x + roi_width / 2 - center_search_radius, mCenter.y + roi_width / 2 - center_search_radius);
		}
	}

	Mat show = roi_dst.clone();
	circle(show, ac_center, 3, Scalar(255, 255, 255), -1);

	// ����rs1��
	circle(BiaopanDetector::rs1, ac_center, 3, Scalar(255, 255, 255), -1);

	//imshow("center", show);
	//waitKey();
	return ac_center;
}


// ƴ��ֱ�ߣ�����ֱ��Ⱥ��vec4f��ǰ����ֵΪͷ�㣬������ֵΪβ�㣩���������ֱ��Ⱥ��������
vector<vector<int>> BiaopanDetector::groupLines(vector<Vec4f> lines, Point& ac_center)
{
	float angelCos = cos(lineMaxAngle *  pi / 180);

	// �洢��Щ���߶�
	vector<int> backs = vector<int>(lines.size());
	// �洢��Щǰ�߶ԣ�����֮��ļ���
	vector<int> fronts = vector<int>(lines.size());

	// ��ʼ����Щ�Ե�ֵ��Ĭ��Ϊ-1
	for (int i = lines.size() - 1; i >= 0; --i)
	{
		backs[i] = -1;
		fronts[i] = -1;
	}
	// ��һ���Ƕ�ÿ�����ж�ǰ����
	for (int i = lines.size() - 1; i >= 0; --i)
	{
		Vec4f& line1 = lines[i];
		// �����뾶�ڣ������������Լ�������ߣ�Ȼ�����е��������ɵĽǶȲ��ܴ��ĳ��ֵ������ȡ�����Լ���̵���Щ�ߣ�����ǿ��������ǲ����Ѿ���Ϊ�˱��˵ĺ�����
		int mIndex = i;
		float mDis = 1000;
		for (int j = lines.size() - 1; j >= 0; --j)
		{
			if (i == j)
				continue;
			Vec4f& line2 = lines[j];

			// ���ж��ǲ����Լ��������
			if (point2point(Point(line1[0], line1[1]), ac_center) > point2point(Point(line2[0], line2[1]), ac_center))
				continue;

			float dis = point2point(line1[2], line1[3], line2[0], line2[1]);

			if (dis <= min(line_search_radius, mDis))
			{
				Vec4f mLine = Vec4f((line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2, (line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2);
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
			if (point2point(lines[fronts[mIndex]]) < point2point(lines[i])) {

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

	// �ݹ������Щ�Բ��ҵó������
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

	return goal_lines;
}

// ָ���⣬�������roi_dst���������ģ�������� ָ��ĽǶ� 
float BiaopanDetector::getPointer(Mat& roi_dst, Point& ac_center)
{
	Mat roi_thresh = roi_dst.clone();
	Mat show;
	cvtColor(roi_dst, show, COLOR_GRAY2BGR);

	adaptiveThreshold(roi_thresh, roi_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);
	// ���波�Ը�ʴ��Σ��ó�������Ҫ��ָ���ֱ�ߣ�������ͨ��֮ǰ�Ĳ�������
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	// �洢��һ�μ�⵽��ֱ��
	vector<Vec4f> tLines_detect;
	// ��ֱ�����ֺ�ͷ���β��
	vector<Vec4f> tLines_pack;

	// ��ʴ
	for (int i=erode_times;i>0;i--)
	{
		erode(roi_thresh, roi_thresh, element);
	}

	// ���������
	imwrite("1.jpg", roi_thresh);
	fld->detect(roi_thresh, tLines_detect);
	// ��������β��ͷ�β��
	for (int i = 0; i < tLines_detect.size(); i++)
	{
		// ɸѡ����Щ�������ıȽ�Զ����
		float distance = point2Line(ac_center, tLines_detect[i]);

		if (distance <= center2line)
		{
			Vec4f l = tLines_detect[i];
			// ��Ҫ��ͷβ����
			Vec4f dl;
			if (point2point(l[0], l[1], ac_center.x, ac_center.y) >= point2point(l[2], l[3], ac_center.x, ac_center.y))
			{
				dl = Vec4f(l[2], l[3], l[0], l[1]);
			}
			else
			{
				dl = l;
			}
			tLines_pack.push_back(dl);

			// ������
			circle(show, Point(dl[0], dl[1]), 2, Scalar(255, 0, 255), -1);
			circle(show, Point(dl[2], dl[3]), 2, Scalar(0, 255, 255), -1);

			// ����rs1��
			circle(BiaopanDetector::rs1, Point(dl[0], dl[1]), 2, Scalar(255, 0, 255), -1);
			circle(BiaopanDetector::rs1, Point(dl[2], dl[3]), 2, Scalar(0, 255, 255), -1);
		}
	}
	fld->drawSegments(show, tLines_pack);
	fld->drawSegments(BiaopanDetector::rs1, tLines_pack);


	// ����ƴ��ֱ�߲���Ѱ������
	vector<vector<int>> groups = groupLines(tLines_pack, ac_center);

	// �ҳ�����Ǹ�ֱ����ϣ���ʾ����
	float maxLength = 0;
	vector<int>& maxLine = groups[0];
	for (int i = groups.size() - 1; i >= 1; --i)
	{
		vector<int>& group = groups[i];

		float total_length = 0;
		for (int j = 0; j <= group.size() - 1; ++j)
		{
			int li = group[j];
			// ����Ҫ����ƽ������ΪԽ����ռ��Ӧ��Խ��
			total_length += point2point(tLines_pack[li]);
		}
		if (total_length > maxLength) { maxLength = total_length; maxLine = group; }
	}
	// ���������
	Scalar cc(255, 255, 255);
	for (int j = 0; j <= maxLine.size() - 1; ++j)
	{
		int li = maxLine[j];
		Vec4f ln = tLines_pack[li];
		Point point1 = Point(ln[0], ln[1]);
		Point point2 = Point(ln[2], ln[3]);
		line(show, point1, point2, cc, 2);
		// �����rs1��
		line(BiaopanDetector::rs1, point1, point2, cc, 2);
	}


	// ������ߵĽ�β����ָ���ߵ�β��
	Vec4f last_part = tLines_pack[maxLine[maxLine.size() - 1]];
	Point lastPoint = Point(last_part[2], last_part[3]);


	// ����ָ����ת��
	float dx = last_part[2] - ac_center.x;
	float dy = -last_part[3] + ac_center.y;
	// �������������޷ָ��ߵ���ļн�
	float vecAngle = atan2(dy, dx) * 180 / pi;
	if (vecAngle <= -90) { vecAngle = -vecAngle - 90; }
	else if (vecAngle >= 0) { vecAngle = -vecAngle + 270; }
	else if (vecAngle < 0 && vecAngle > -90) { vecAngle = -vecAngle + 270; }

	cout << "ָ��Ƕ�Ϊ�� " << vecAngle << endl;

	//imshow("pointer", show);
	//waitKey(0);

	return vecAngle;
}


// ��Բ��ϣ��������roi_dst����Բ������ģ����Բ��Ǿ�ȷ�����ģ���������� ��Բ��Ӿ���
RotatedRect BiaopanDetector::fitELL(Mat& roi_dst, Point& ac_center)
{
	Mat show;
	cvtColor(roi_dst, show, COLOR_GRAY2BGR);
	cvtColor(roi_dst, BiaopanDetector::rs2, COLOR_GRAY2BGR);
	Mat equalizer;
	Mat otsu_thresh;
	Mat ada_thresh;

	float roi_area = roi_dst.cols * roi_dst.rows;
	// ����ֵ��������������Ƿ���� ֱ��ͼ���⻯
	equalizeHist(roi_dst, equalizer);
	adaptiveThreshold(equalizer, ada_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);
	threshold(equalizer, otsu_thresh, 0, 255, CV_THRESH_OTSU);

	vector<Vec4f> tLines_detect;
	vector<Vec4f> tLines_pack;
	// ���������
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
	ls->detect(ada_thresh - otsu_thresh, tLines_detect);
	// ��������β��ͷ�β��
	for (int i = 0; i < tLines_detect.size(); i++)
	{
		// ɸѡ����Щ�������ıȽ�Զ����
		float distance = point2Line(ac_center, tLines_detect[i]);

		if (distance <= center2line)
		{
			Vec4f l = tLines_detect[i];
			// ��Ҫ��ͷβ����
			Vec4f dl;
			if (point2point(l[0], l[1], ac_center.x, ac_center.y) >= point2point(l[2], l[3], ac_center.x, ac_center.y))
			{
				dl = Vec4f(l[2], l[3], l[0], l[1]);
			}
			else
			{
				dl = l;
			}
			tLines_pack.push_back(dl);

			// ������
			circle(show, Point(dl[0], dl[1]), 2, Scalar(255, 0, 255), -1);
			circle(show, Point(dl[2], dl[3]), 2, Scalar(0, 255, 255), -1);

			circle(BiaopanDetector::rs2, Point(dl[0], dl[1]), 2, Scalar(255, 0, 255), -1);
			circle(BiaopanDetector::rs2, Point(dl[2], dl[3]), 2, Scalar(0, 255, 255), -1);

		}
	}
	// �������г����е����
	ls->drawSegments(show, tLines_pack);
	ls->drawSegments(BiaopanDetector::rs2, tLines_pack);


	// bim+֧�ֵ㷽����� Ȼ�������ȡ��Բ
	// ��tLine���з��ࣨ���Ƕȷ��ࣩ�����簴��40��һ������������9������ 
	// ��� tLine ������ʶ��ָ��� line�ǲ�һ���ġ�
	vector<vector<int>> sangles_(360 / sangle);
	for (int i = tLines_pack.size() - 1; i >= 0; --i)
	{
		// ����нǵ�cosֵ
		int xd = tLines_pack[i][2] - ac_center.x;
		int yd = ac_center.y - tLines_pack[i][3];
		// ֵ���� 0~360֮��
		float vangle = fastAtan2(yd, xd);
		sangles_[(int)vangle / sangle].push_back(i);
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
	// �洢��ǰ��֧�ֵ����
	int nowSnum = 5;
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
			Vec4f vvvv = tLines_pack[shanqu[sii]];
			Point ssss = Point(vvvv[2], vvvv[3]);
			candips.push_back(ssss);
		}
		// �����Բ�����֧�ֵ����
		RotatedRect rect = fitEllipse(candips);



		// �����Բ��Ҫ����ϵ���Բ�����ĺ�acc_center��������Ҳռ��roi_dstһ������
		float area = pi * rect.size.width*rect.size.height / 4;
		float a = rect.size.width > rect.size.height ? rect.size.width : rect.size.height;
		float b = rect.size.width < rect.size.height ? rect.size.width : rect.size.height;

		if (point2point(rect.center, ac_center) <= ell_accept_distance && (area / roi_area) >= ellsize2roisize && (b / a) >= ell_ab_p)
		{
			// �洢֧�ֵ�
			vector<int> support_points;
			for (int ki = 0; ki < tLines_pack.size(); ki++)
			{
				Point sp = Point(tLines_pack[ki][2], tLines_pack[ki][3]);
				Point newsp = origin2el(rect.center, rect.angle / (float)180 * pi, sp);
				float edistance = abs(sqrt(pow(newsp.x, 2) / pow(rect.size.width / 2, 2) + pow(newsp.y, 2) / pow(rect.size.height / 2, 2)) - 1) * max(rect.size.width / 2, rect.size.height / 2);
				if (edistance <= ell_accept_thresh)
				{
					support_points.push_back(ki);
				}
			}

			if (support_points.size() >= nowSnum)
			{
				bestSupportPoints = support_points;
				bestEl = rect;
				nowSnum = support_points.size();
			}
		}

	}

	// ���õ���֧�ֵ�����������
	vector<Point> supporters;



	for (int ki = 0; ki < bestSupportPoints.size(); ki++)
	{
		int tLineIndex = bestSupportPoints[ki];
		Point supporter = Point(tLines_pack[tLineIndex][2], tLines_pack[tLineIndex][3]);
		supporters.push_back(supporter);
		// ������ϵ�
		circle(show, supporter, 3, Scalar(0, 0, 0), -1);
		circle(BiaopanDetector::rs2, supporter, 3, Scalar(0, 0, 0), -1);
	}

	if (supporters.size() < 5)
	{
		cout << "��Բ���ʧ�ܣ�����Ҫ�����ϵ㲻�㣬Ҫô�������ԲɸѡҪ��Ҫô������ֵ��ʽ��" << endl;
		return bestEl;
	}

	bestEl = fitEllipse(supporters);
	ellipse(show, bestEl, Scalar(255, 255, 255));

	float area = pi * bestEl.size.width * bestEl.size.height / 4;
	cout << "area pro: " << area / roi_area << endl;
	cout << "dis pro: " << point2point(bestEl.center, ac_center) << endl;

	//imshow("fitEllipse", show);
	//waitKey();
	return bestEl;
}


// ��������������ʶ�����֣�������ǵ���mser���������response
vector<float> BiaopanDetector::readMser(Mat& mser)
{
	// otsu��ȡ���֣�Ȼ�󿪲���ȥ���ӵ㣬����Ƿָ�ʶ��
	Mat mser_roi_thresh;
	threshold(mser, mser_roi_thresh, 0, 255, CV_THRESH_OTSU);
	mser_roi_thresh = 255 - mser_roi_thresh;
	Mat e_element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	// ������ȥ���ӵ�
	morphologyEx(mser_roi_thresh, mser_roi_thresh, MORPH_OPEN, e_element);
	// ǰ���rect�Ŵ���һ�㣬����������ͳ�����Ŀ�߱�
	float wihi = (mser_roi_thresh.cols) / (float)(mser_roi_thresh.rows);

	vector<upper> uppers;
	vertical_projection(mser_roi_thresh, uppers);

	// ����Ѱ�Ҷ��ٸ��ָ��ߣ����ݱ�����������ȻҲ������ȫ����������㷨�Զ�Ѱ�ң�������㷨Ѱ�ҵ����еķָ���Ҫȥͷβ��������ķָ��
	// �ȶ����е�ֵ���дӴ�С��������Ȼ���ͷ��ʼ������Ѱ��ĳ���㣬����7�����Ǹ�͹����(��ΪѰ�������������б�֤�˺����������ģ����Բ�������ֵ��)
	vector<upper> uppers_sort;
	for (int ei = 0; ei < uppers.size(); ei++) { uppers_sort.push_back(uppers[ei]); }
	// ��ʾ������
	//waitKey();
	sort(uppers_sort.begin(), uppers_sort.end(), SortByUp);
	// �洢�и�㣬anchors��ŵ��Ƿָ�����ţ������������ֵ
	vector<int> anchors;
	// ���anchors��Ӧ�ļ�ֵ��
	vector<extremer> extremers;
	int usize = uppers.size();

	if (wihi < 0.6) { usize = 0; }

	for (int ei = 0; ei < uppers_sort.size(); ei++)
	{
		// ����ռ��Ϊ�����ߣ����������ߣ�Ѱ�ҵ���С��0�������㣬����ҵĵ���ô�������Ǽ�Сֵ��
		int uindex = uppers_sort[ei].x;
		int res = uppers_sort[ei].y;
		bool left_all_small = true;
		int left_gradient = 2;
		bool right_all_small = true;
		int right_gradient = 2;
		int lindex = uindex - 1;
		int rindex = uindex + 1;
		for (; lindex >= 0; lindex--)
		{
			if ((uppers[lindex].y - uppers[lindex + 1].y) < 0) { left_gradient--; if (left_gradient == 0) break; }
			if ((uppers[lindex].y - uppers[lindex + 1].y) > 0) { left_all_small = false; break; }

		}
		for (; rindex < usize; rindex++)
		{
			if ((uppers[rindex].y - uppers[rindex - 1].y) < 0) { right_gradient--; if (right_gradient == 0) break; }
			if ((uppers[rindex].y - uppers[rindex - 1].y) > 0) { right_all_small = false; break; }
		}

		if (left_all_small && right_all_small && (left_gradient == 0) && (right_gradient == 0))
		{
			anchors.push_back(uindex);
			// ��Ž� ��ֵ�� vector
			extremers.push_back({ uppers_sort[ei], lindex, rindex });
		}
	}

	vector<float> responses;
	bool is_singular = false;

	if (anchors.size() > 0)
	{
		// ����һ�ηָ�㣬�ָ������ĵ���һ���ָ�㣬true_anchors��ŵ��Ƿָ�������
		// һ��ָ�������3�ָΪ�������ݴ��ԣ��������ٸ�2����Ҫ����Ϊ���ֿ��ܱȽϼ��У�
		int close_thresh = uppers.size() / 3 - 2;
		//cout << "close_thresh: " << close_thresh << endl;
		vector<int> true_anchors;

		// �洢���������ļ�ֵ������
		vector<int> true_extremers;
		true_anchors.push_back(0);
		int pp = 0;
		// ����ָ�㣬������ķָ���Ϊһ���ָ�㣨�Ϻ�ķָ�����м䣩
		for (int ei = 0; ei < anchors.size(); ei++)
		{
			// Ҫ��ֹ���ָ�
			int ax = uppers[anchors[ei]].x;
			if (abs(ax - true_anchors[pp]) > close_thresh)
			{
				true_anchors.push_back(ax); pp++;
				//max_xs.push_back(ax); min_xs.push_back(ax);
				true_extremers.push_back(ei);
			}
		}
		for (int ei = 1; ei < true_anchors.size(); ei++)
		{
			//true_anchors[ei] = min_xs[ei - 1] + (max_xs[ei - 1] - min_xs[ei - 1]) / 2;
			int emIndex = true_extremers[ei - 1];
			true_anchors[ei] = (extremers[emIndex].lindex + extremers[emIndex].rindex) / 2;
		}
		true_anchors.push_back(mser.cols);
		// ���մ�С��������ָ��
		sort(true_anchors.begin(), true_anchors.end());

		// ���true_anchors
		//cout << endl;
		//cout << "��һ��--true_anchors: [";
		//for (int ei = 0; ei < true_anchors.size(); ei++) { cout << true_anchors[ei] << ","; }
		//cout << "]" << endl;

		// ������Σ���󻹵���һ�ι��ָ��飬���������ڵ�����ָ���ں�
		vector<int> temp_anchors;
		temp_anchors.push_back(0);
		// �����Ǽ�¼ÿ����ֵ������ұ߽�
		vector<int> max_xs;
		vector<int> min_xs;
		int pp2 = 0;
		for (int eii = 1; eii < true_anchors.size() - 1; eii++)
		{
			int ax = true_anchors[eii];
			if (abs(ax - temp_anchors[pp2]) > close_thresh)
			{
				temp_anchors.push_back(ax); pp2++;
				max_xs.push_back(ax); min_xs.push_back(ax);
			}
			else
			{
				// ������һ�����������ĵ������ſ��Կ�ʼ����max��min
				if (pp2 >= 1)
				{
					if (ax > max_xs[pp2 - 1]) { max_xs[pp2 - 1] = ax; }
					if (ax < min_xs[pp2 - 1]) { min_xs[pp2 - 1] = ax; }
				}
			}
		}
		for (int eii = 1; eii < temp_anchors.size() - 1; eii++)
		{
			temp_anchors[eii] = (max_xs[eii - 1] + min_xs[eii - 1]) / 2;
		}
		temp_anchors.push_back(mser.cols);
		true_anchors = temp_anchors;

		// ���true_anchors
		//cout << endl;
		//cout << "�ڶ���--true_anchors: [";
		//for (int ei = 0; ei < true_anchors.size(); ei++) { cout << true_anchors[ei] << ","; }
		//cout << "]" << endl;
		// ��������ķָ��ָ�����Ȼ��ʶ��������һ��
		// ע�������anchor�����ͼƬ�������˵�


		for (int ei = 0; ei < true_anchors.size() - 1; ei++)
		{
			//Mat mmser = final_mser_roi.colRange(max(0, true_anchors[ei]-1), min(true_anchors[ei + 1]+1, width));
			Mat mmser = mser.colRange(max(0, true_anchors[ei]), min(true_anchors[ei + 1], mser.cols));
			vector<float> v = getHogData(mmser);
			Mat testData(1, 144, CV_32FC1, v.data());
			int response = svm->predict(testData);
			//imshow("[�ָ���Сͼ]", mmser);
			//waitKey();
			// response = -1��ʾ����������֣�response = 10��ʾ����������ں�����
			if (response < 0 || response == 10) { is_singular = true; break; }
			else { responses.push_back(response); }

		}
	}

	else
	{
		// ֱ��Ԥ��
		vector<float> v = getHogData(mser);
		Mat testData(1, 144, CV_32FC1, v.data());
		int response = svm->predict(testData);
		// response = -1��ʾ����������֣�response = 10��ʾ����������ں�����
		if (response < 0 || response == 10) { is_singular = true; }
		else { responses.push_back(response); }
	}


	if (is_singular) 
	{ 
		//cout << "is_singular: true" << endl; 
		return responses; 
	}
	else
	{
		//cout << "merge_response: ";
		//for (int ei = 0; ei < responses.size(); ei++)
		//{
		//	cout << responses[ei];
		//}
		//cout << endl;

		// ��ŵ�singleArea���ú�������ں�
		Point ncenter;
		// Ϊ��֮�������
		return responses;
	}


}

// �ںϣ��������singleArea�ļ��ϣ�ac_center
vector<MergeArea> BiaopanDetector::mergeSingleArea(vector<SingleArea>& sas, Point& ac_center)
{
	// �涨�ںϵ���̾���
	float merge_distance = 30;
	float min_merge_distance = 10;
	// ������angle �����
	int angle_error = 15;
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

					Point sacenter = sas[kii].center;
					// Ѱ�����ںϵ��Ǹ��㣬Ҳ���Ǿ�����������̵��Ǹ���
					float min_dd = merge_distance;
					int target_sa = -1;
					for (int kjj = 0; kjj < sas.size(); kjj++)
					{
						// ������Ҫ�鿴kjj�Ƿ��Ѿ����ӹ�kii���Ѿ������ľ�����(-1��ʾδ���ӣ�-2��ʾ�Լ����Լ�)
						if (joinTable[kii][kjj] == -2 || joinTable[kii][kjj] > -1) { continue; }
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
			cc_indexs.push_back(sas[sas_index].cc_index);
			goal_set_areas.push_back(sas[sas_index]);
			x_ += sas[sas_index].center.x;
			y_ += sas[sas_index].center.y;
		}
		// ������µ�����
		Point merge_center = Point(int(x_ / goal_set_size), int(y_ / goal_set_size));


		// ����x��goal_set_areas���򣬲��ó�������response
		sort(goal_set_areas.begin(), goal_set_areas.end(), SortByX);
		float merge_response = 0;
		// ����� merge_response ��ǰ�治ͬ����Ϊǰ���singleArea�к���һЩ�Ѿ��ںϵģ�Ӧ�����ж�singleArea�Ƕ���λ�ġ�Ȼ��������ȥ���µ� merge_response
		vector<float> responses;

		for (int kjj = 0; kjj < goal_set_size; kjj++)
		{
			vector<float> mres = goal_set_areas[kjj].response;
			for (int kjjj = 0; kjjj < mres.size(); kjjj++)
			{
				responses.push_back(mres[kjjj]);
			}
		}

		// �ж��Ƿ��¶ȼƣ�������¶ȼ���Ҫ�ñ�ļ������
		if (responses.size() >= 2 && responses[0] > 0 && responses[1] == 0)
		{
			// ����ĳ�ֹ���������response
			merge_response = responses[0] * pow(10, responses.size() - 1);
		}
		else
		{
			for (int kjj = 0; kjj < responses.size(); kjj++)
			{
				// ����ĳ�ֹ���������response
				merge_response += responses[kjj] * pow(10, -kjj);
			}
		}

		// �������������޷ָ��ߵ���ļнǲ� ��Ⱦ�� merge_area
		float ddy = -merge_center.y + ac_center.y;
		float ddx = merge_center.x - ac_center.x;
		float merge_angle = getVecAngle(ddy, ddx);

		// ɸѡ���Ƕȹ���Ŀ��ɵ�
		if (merge_angle > 320 || merge_angle < 30) { continue; }

		merge_areas.push_back({ merge_response, merge_angle, cc_indexs, merge_center });

	}

	return merge_areas;
}

// ��ֵ ��⣬������� MergeArea�ļ��ϣ�ac_center��������Ƿ���������ͬʱ�Ѿ��źýǶȵ� MergeArea�ļ���
vector<MergeArea> BiaopanDetector::removeSingular(vector<MergeArea>& merge_areas, Point& ac_center)
{

	/*
		�����Ƕ���Щ merge_area ����ɸѡ������Щ��Ӧֵ��ֵĵ�ȥ����

		�����ǰ�merge_area����һ���㣬����response��yֵ��angle��xֵ�����滹����ϸ˵��

	*/

	// �洢��ֵ��,false��ʾ�õ��Ƿ�����ֵ��true��ʾ�õ�Ϊ����ֵ��һ��ʼȫ�������Ϊ������ֵ
	vector<bool> singulars(merge_areas.size(), false);

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

	//cout << "��һ��ȷ�Ϻ�" << endl;
	for (int kii = 0; kii < singulars.size(); kii++)
	{
		if (singulars[kii]) { continue; }
		//cout << "�������ӦֵΪ�� " << merge_areas[kii].response << endl;
		merges_1.push_back(merge_areas[kii]);
	}


	if (merges_1.size() == 0)
	{
		cout << "!!!!!!!!!!���Ƕȴ������⣬������½Ƕȼ������!!!!!!!!!!!" << endl;
		return merges;
	}


	// ���սǶȸ�����mergeArea����
	sort(merges_1.begin(), merges_1.end(), SortByAngle);


	// ����������һ��������sort��֮��һ�����ϸ񵥵������ģ�������ֵݼ����߲�����������ô����Ҫȥ������
	merges.push_back(merges_1[0]);



	int merges_pos = 0;
	//cout << "�ڶ���ȷ�Ϻ�" << endl;
	cout << "�������ӦֵΪ�� " << merges_1[0].response << endl;
	for (int kii = 1; kii < merges_1.size(); kii++)
	{
		if (merges_1[kii].response <= merges[merges_pos].response) { continue; }
		else { merges.push_back(merges_1[kii]); merges_pos++; cout << "�������ӦֵΪ�� " << merges_1[kii].response << endl; }
	}

	return merges;
	
}


// ���յ�ʾ����ȡ��������� MergeArea�ļ��ϣ����̵�ָ��Ƕȣ�������� ���ն�ȡʾ��
float BiaopanDetector::readAngle(vector<MergeArea>& merges, float pointerAngle)
{
	float pointerValue = -1;

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
				float delta_value = (merges[kii].response - merges[kii - 1].response) / (merges[kii].angle - merges[kii - 1].angle) * (pointerAngle - merges[kii - 1].angle);
				pointerValue = merges[kii - 1].response + delta_value;
				break;
			}
		}
	}
	return pointerValue;
}

// ʾ����ȡ���������roi_dst����Բ���ģ���������mser������������̰뾶�ļнǣ���ָ��нǣ��Ƕ���Ϣ���Ƿ�ͶƱ��ȡ�Ƕȣ��������ʾ��
// ֮ǰ������Բ��ϣ����������Բ��ϣ����ñ��̾���������
float BiaopanDetector::readRoi(Mat& roi_dst, Point& ac_center, float pointerAngle, float mserAngle, bool isVoted)
{

	// ����ac_centerΪ���Ľ��нǶȽ���

	// rotationMatrix�ĵڶ�����������ʱ����ת�Ƕȣ��պ����ǵ�mserAngle��ʾ���Ǽ�⵽���̵�˳ʱ����ת�Ƕȣ�Ҫ��������Ҫ��ʱ����ת
	Mat rotationMatrix = getRotationMatrix2D(ac_center, mserAngle, 1);//������ת�ķ���任���� 
	Mat rMat;
	warpAffine(roi_dst, rMat, rotationMatrix, Size(roi_dst.cols, roi_dst.rows));//����任  


	// չʾ��Ҳ���������
	Mat show;
	cvtColor(rMat, show, COLOR_GRAY2BGR);

	// ������Ƿ���� ֱ��ͼ ���⻯
	Mat roi_thresh;
	equalizeHist(rMat, roi_thresh);

	// ֱ���ñ���������
	RotatedRect bestEl = ac_el;
	// ���ǻ�ҪӦ����ת
	bestEl.angle = bestEl.angle - mserAngle;

	// ������������������
	RotatedRect bestEl_scale0 = RotatedRect(bestEl.center, Size2f(bestEl.size.width, bestEl.size.height), bestEl.angle);
	RotatedRect bestEl_scale1 = RotatedRect(bestEl.center, Size2f(bestEl.size.width * 0.75, bestEl.size.height * 0.75), bestEl.angle);
	RotatedRect bestEl_scale2 = RotatedRect(bestEl.center, Size2f(bestEl.size.width * 0.4, bestEl.size.height * 0.4), bestEl.angle);
	// ��������
	circle(show, Point(bestEl.center.x, bestEl.center.y), 3, Scalar(0, 0, 0), -1);
	ellipse(show, bestEl_scale1, Scalar(200, 100, 150));
	ellipse(show, bestEl_scale2, Scalar(110, 30, 250));
	ellipse(show, bestEl_scale0, Scalar(0, 0, 0), 3);


	// mser���
	vector<Rect> candidates = simpleMser(rMat);
	// �洢��δ����Ƿ���ֵ��mser���������ֵģ��ʹ��
	vector<SingleArea> sas;

	// ����ֻ�ǰѷ���������candidate����ѡ���������������
	for (int i = 0; i < candidates.size(); ++i)
	{
		// �ȱ����е�mser
		rectangle(show, candidates[i], Scalar(200, 100, 30), 1);

		// ��ɸѡ�����Բ����ĵ�
		Point ncenter = Point(candidates[i].x + candidates[i].width / 2, candidates[i].y + candidates[i].height / 2);
		Point newsc = origin2el(bestEl.center, bestEl.angle / (float)180 * pi, ncenter);
		float ndistance = sqrt(pow(newsc.x, 2) / pow(bestEl.size.width / 2, 2) + pow(newsc.y, 2) / pow(bestEl.size.height / 2, 2));
		// �������Բ��Ҫ���ڲ���һ�£�ȥ�������զ�㣬����̶���
		// ͬʱ�ֲ���̫���棬��Ϊ������������һ��������ġ�
		if (ndistance >= 0.75 || ndistance < 0.4) { continue; }



		
		Rect mserRoi = Rect(candidates[i].x, candidates[i].y, candidates[i].width, candidates[i].height);
		Mat mser = rMat(mserRoi);
		Point mserCenter = Point(mserRoi.x + mserRoi.width/2, mserRoi.y + mserRoi.height/2);

		//imshow("mserRoi", mser);
		//imshow("readRoi", show);
		//waitKey();

		// ��ȡʾ��
		vector<float> response = readMser(mser);
		// ��Ⱦ��������ʽ
		if (response.size() > 0)
		{
			// ����ֻ����������������ڵ�mser����ʶ��ɹ���
			rectangle(show, candidates[i], Scalar(255, 255, 0), 1);

			// ��Ⱦ��singleArea
			sas.push_back({ response, i, mserCenter });
		}

	}

	vector<MergeArea> merges = mergeSingleArea(sas, ac_center);

	if (merges.size() == 0)
	{
		cout << "��ȡʾ��ʧ�ܣ�����roi�Ƿ���ȷ" << endl;
		return -1;
	}
	// ����merge
	for (int i = 0; i < merges.size(); ++i) 
	{
		MergeArea merge = merges[i];
		Point mergeCenter = merge.center;
		vector<int> ccs = merge.cc_indexs;
		for (int j = 0; j < ccs.size(); ++j) 
		{
			rectangle(show, candidates[ccs[j]], Scalar(255, 255, 255), 1);
		}
		circle(show, mergeCenter, 2, Scalar(255, 255, 255), -1);
	}


	merges = removeSingular(merges, ac_center);

	if (merges.size() <= 3)
	{
		cout << "!!!!!!!!!!���Ƕȴ������⣬������½Ƕȼ������!!!!!!!!!!!" << endl;
		return -1;
	}
	// ����Ҫע�⣬��Ϊ���Ǽ���ͼ�Ǿ������ǽ�����ͼ������ָ���ڽ������ͼ�ĽǶ�����Ҫ��ȥ�����ĽǶȣ�
	// ����Ϊ���˳ʱ����ת�ǶȺܴ󣬱�ʾ����ʵ����ʱ����ת�Ƕȣ�������Ҫ����һ�£����Ƴ��������
	float pv = pointerAngle - mserAngle;
	if (pv < 0) { pv = pv + 360; }
	float finalValue = readAngle(merges, pv);

	if (finalValue < 0)
	{
		cout << "!!!!!!!!!!���Ƕȴ������⣬������½Ƕȼ������!!!!!!!!!!!" << endl;
		return -1;
	}

	//imshow("readRoi", show);
	//waitKey();

	rs2 = show;

	return finalValue;
}



// �ܵ���ڷ���
float BiaopanDetector::detect(Mat& mat, string& id)
{
	// ����Ƿ�ڰף�������Ǻڰ׾�ת�ɺڰ�
	Mat img;
	if (mat.channels() > 1)
	{
		cvtColor(mat, img, COLOR_BGR2GRAY);
	}
	else
	{
		img = mat;
	}
	// ��������ʹ��roi֮ǰ�ȼ���roi�Ƿ���ɹ�
	float rotation;
	Mat roi_dst = getRoiDst(img, id, rotation);
	if (roi_dst.cols > 0)
	{
		// ��������
		RotatedRect acEl = getAcOutline(roi_dst);
		if (!acEl.angle) { cout << "����������ʶ������������Դ���߽Ƕ� " << endl; return -2; }
		// ��������
		Point ac_center = Point(acEl.center.x, acEl.center.y);
		// ����ָ��Ƕ�
		float pointerAngle = getPointer(roi_dst, ac_center);
		// ��ȡ����
		float value = readRoi(roi_dst, ac_center, pointerAngle, rotation, false);
		return value;
	}
	return -1;
}



void main_test()
{
	string img_path = "D:\\VcProject\\biaopan\\biaopan1\\data\\2.jpg";
	img_path = "D:\\VcProject\\MvProj\\Samples\\VC\\VS\\test\\6.jpg";

	Mat img = imread(img_path, 0);
	BiaopanDetector detector;
	detector.initBiaopan();

	string id = "��һ������";
	// ��������ʹ��roi֮ǰ�ȼ���roi�Ƿ���ɹ�
	float rotation;
	Mat roi_dst = detector.getRoiDst(img, id, rotation);
	if (roi_dst.cols > 0)
	{
		// ��������
		RotatedRect acEl = detector.getAcOutline(roi_dst);
		if (!acEl.angle) { cout << "����������ʶ������������Դ���߽Ƕ� " << endl; return; }
		// ��������
		Point ac_center = Point(acEl.center.x, acEl.center.y);
		// ����ָ��Ƕ�
		float pointerAngle = detector.getPointer(roi_dst, ac_center);
		// ��ȡ����
		float value = detector.readRoi(roi_dst, ac_center, pointerAngle, rotation, false);

		cout << "ʾ���ǣ� " << value << endl;
	}
	imshow("rs1", detector.rs1);
	imshow("rs2", detector.rs2);
	waitKey();
}

// codeLen �ĳ����� 
void  BiaopanDetector::testThreshDetect(Mat& img, Point& ac_pos, float& ac_distance, float& ac_angle, float& angle_given, float& codeLen, float& dd)
{
	BiaopanDetector detector;
	detector.initBiaopan();
	Point img_center = Point(img.cols / 2, -img.rows / 2);
	// ��������������ĽǶȽ���ͼ�����
	// rotationMatrix�ĵڶ�����������ʱ����ת�Ƕȣ��պ����ǵ�mserAngle��ʾ���Ǽ�⵽���̵�˳ʱ����ת�Ƕȣ�Ҫ��������Ҫ��ʱ����ת
	QRDetector qr;
	vector<BCoder> coders = qr.detect(img);
	BCoder bc = coders[0];
	// �õ��Ƕ�
	ac_angle = bc.rotation - angle_given;
	// �õ��߳�
	float dis_a = detector.point2point(bc.a, bc.c);
	float dis_b = detector.point2point(bc.b, bc.d);

	// �õ�ʵ�ʾ��������ؾ����ֵ
	float ratio =  (dis_a + dis_b) / (2 * sqrt(2) * codeLen);
	// ����������꣬����תΪ��׼����ϵ����Ϊ����angle_given����Ҫ����תһ������
	Point rPos = Point(bc.center.x - img_center.x, -bc.center.y - img_center.y);



	rPos = detector.rotate(angle_given, rPos);
	ac_pos = Point((float)rPos.x / ratio, (float)rPos.y / ratio);
	
	
	
	float kk = (float)detector.point2point(bc.a, bc.b) / ratio;

	double ss = dis_a * dis_b;
	
	double biaoding_ss = 230281;
	double dis = sqrt(600 * 600 * biaoding_ss / ss);


	circle(img, rPos, 10, Scalar(255, 255, 255));
	circle(img, img_center, 10, Scalar(0, 0, 0));

}



void fldTest()
{
	Mat image = imread("D:\\VcProject\\MvProj\\Samples\\VC\\VS\\test\\3.jpg", 0);
	// Create FLD detector
	// Param               Default value   Description
	// length_threshold    10            - Segments shorter than this will be discarded
	// distance_threshold  1.41421356    - A point placed from a hypothesis line
	//                                     segment farther than this will be
	//                                     regarded as an outlier
	// canny_th1           50            - First threshold for
	//                                     hysteresis procedure in Canny()
	// canny_th2           50            - Second threshold for
	//                                     hysteresis procedure in Canny()
	// canny_aperture_size 3             - Aperturesize for the sobel
	//                                     operator in Canny()
	// do_merge            false         - If true, incremental merging of segments
	//                                     will be perfomred
	int    length_threshold = 10;
	float  distance_threshold = 1.41421356f;
	double canny_th1 = 50.0;
	double canny_th2 = 50.0;
	int    canny_aperture_size = 3;
	bool   do_merge = true;
	Ptr<FastLineDetector> fld = createFastLineDetector(
		length_threshold,
		distance_threshold,
		canny_th1,
		canny_th2,
		canny_aperture_size,
		do_merge);
	vector<Vec4f> lines_fld;

	fld->detect(image, lines_fld);
	// Show found lines with FLD
	Mat line_image_fld(image);
	fld->drawSegments(line_image_fld, lines_fld);
	imshow("FLD result", line_image_fld);
	waitKey();


}


int tt()
{
	Mat roi_thresh = imread("D:\\VcProject\\MvProj\\Samples\\VC\\VS\\test\\3.jpg", 0);
	
	vector<Vec4f> tLines_detect;
	
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector();
	ls->detect(roi_thresh, tLines_detect);
	cout << tLines_detect.size() << endl;
	waitKey();
	return 0;
}
