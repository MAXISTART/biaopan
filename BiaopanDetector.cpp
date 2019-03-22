
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


// 初始化一些东西
void BiaopanDetector::initBiaopan()
{
	// 加载svm
	svm = cv::ml::SVM::load(modelPath);
	// 初始化lsd
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



/*工具类*/
// 计算点到直线的距离，这里输入的点是图像坐标的点
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
// 计算点到直线的距离，这里的点是标准坐标的点
float BiaopanDetector::point2Line(float x, float y, float x1, float y1, float x2, float y2)
{

	float A = (y1 - y2) / (x1 - x2);
	float B = -1;
	float C = y1 - x1 * A;
	float dis = abs(A * x + B * y + C) / sqrt(pow(A, 2) + pow(B, 2));
	return dis;
}
// 计算点到点的距离
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
// 合并直线的操作

// 1. 右搜索
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


// 2. 左搜索
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


// 3. 计算两直线间的夹角cos值
float BiaopanDetector::line2lineAngleCos(Vec4f line1, Vec4f line2)
{
	float leng1 = point2point(line1);
	float leng2 = point2point(line2);
	return ((line1[2] - line1[0]) * (line2[2] - line2[0]) + (line1[3] - line1[1]) * (line2[3] - line2[1])) / leng1 / leng2;
}

/* ------------------------------------ */
// 数值区域的一些方法


// 画出旋转矩阵
void BiaopanDetector::drawRotatedRect(Mat& drawer, RotatedRect& rrect)
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
std::vector<cv::Rect> BiaopanDetector::rotateMser(cv::Mat& srcImage, vector<cv::RotatedRect>& rrects)
{

	std::vector<std::vector<cv::Point> > regContours;

	// 创建MSER对象
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 5, 800, 0.5, 0.3);

	// 限定长宽高
	int max_width = srcImage.cols;
	int max_height = srcImage.rows;

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
			int topx = max(0, br.x - 3);
			int topy = max(0, br.y - 3);
			int maxw = min(br.width + 6, max_width - topx);
			int maxh = min(br.height + 6, max_height - topy);
			br = Rect(br.x - 3, br.y - 3, br.width + 6, br.height + 6);
			keeps.push_back(br);
		}
		// 稍微让方框往外扩一点

		//keeps.push_back(br);
	}

	imshow("rotateMser", mser_show);
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
				&& rrecs[i].center.x <= keeps[j].x + keeps[j].width && rrecs[i].center.y <= keeps[j].y + keeps[j].height)
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

// Mser目标检测 + nms，这个是简洁版的mser，不带旋转矩阵
vector<Rect> BiaopanDetector::simpleMser(Mat& srcImage)
{
	Mat show = srcImage.clone();
	// 限定长宽高
	int max_width = srcImage.cols;
	int max_height = srcImage.rows;

	vector<vector<Point> > regContours;

	// 创建MSER对象
	Ptr<MSER> mesr1 = MSER::create(2, 5, 800, 0.5, 0.3);


	vector<cv::Rect> boxes;
	// MSER检测
	mesr1->detectRegions(srcImage, regContours, boxes);
	// 存储矩形
	vector<Rect> keeps;


	Mat mserMapMat = Mat::zeros(srcImage.size(), CV_8UC1);
	Mat mserNegMapMat = Mat::zeros(srcImage.size(), CV_8UC1);

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
		if (b_size < 800 && b_size > 50)
		{
			// 实验证明，往外扩张的时候识别效果更好，
			int topx = max(0, br.x - mserRadius);
			int topy = max(0, br.y - mserRadius);
			int maxw = min(br.width, max_width - topx + mserRadius * 2);
			int maxh = min(br.height, max_height - topy + mserRadius * 2);
			br = Rect(br.x, br.y, br.width, br.height);
			keeps.push_back(br);	
		}
	}
	
	// 用nms抑制
	nms(keeps, 0.7);

	for (int i=0;i<keeps.size();i++)
	{
		rectangle(show, keeps[i], Scalar(255, 255, 255), 1);
	}
	//imshow("simpleMser", show);
	return  keeps;
}
/* ------------------------------------ */


// vector<ranglevoter>根据ranglevoter中的voteNum排序
bool BiaopanDetector::SortByVote(ranglevoter &v1, ranglevoter &v2)
{
	//降序排列  
	return v1.voteNum > v2.voteNum;
}

// vector<int>根据vector中的值来排序
bool BiaopanDetector::SortByUp(upper &v1, upper &v2)
{
	//降序排列  
	return v1.y > v2.y;
}

// 寻找上轮廓
void BiaopanDetector::vertical_projection(Mat& input_src, vector<upper>& uppers)
{

	int width = input_src.cols;
	int height = input_src.rows;
	int perPixelValue;//每个像素的值
	// 初始化
	vector<int> projectValArry(width, 0);

	// 平滑参数
	int smooth_thresh = 4;

	// 寻找上轮廓，
	// last存储上一列的值，比如当前列与前一列相差过大（比如当前列为空）的，
	// 那么先继续向下探索符合条件的点，如果还是没探索到，那么去看一看前一列，保持和前一列一样，这样是保证函数不会突然断开
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

	// 前面的那些0需要被平滑掉
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


	/*新建一个Mat用于储存投影直方图并将背景置为白色*/
	Mat verticalProjectionMat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			perPixelValue = 255;  //背景设置为白色。   
			verticalProjectionMat.at<uchar>(i, j) = perPixelValue;
		}
	}

	/*将直方图的曲线设为黑色*/
	//cout << "projectValArry: [";
	for (int i = 0; i < width; i++)
	{
		perPixelValue = 0;  //直方图设置为黑色  
		verticalProjectionMat.at<uchar>(projectValArry[i], i) = perPixelValue;
		//cout << projectValArry[i] << ",";

		// 装进uppers里面返回
		uppers.push_back(upper{ i, projectValArry[i] });
	}
	//cout << "]" << endl;
	//imshow("读取轮廓的图", input_src);
	//imshow("【上轮廓】", verticalProjectionMat);
	//waitKey();
}


/* ------------------------------------ */
// 合并数值区域的操作
// 1. 连接点搜索
void BiaopanDetector::joinSearch(vector<bool>& isVisited, vector<int>& goal_set, vector<vector<int>>& joinTable)
{
	vector<int>::iterator it;
	int i = goal_set[goal_set.size() - 1];
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


// 2.判断角度，输入的标准坐标系下向量的 y 与 x，输出的是 以标准坐标系中的三四象限中间轴为0度轴，顺时针旋转的角度
float BiaopanDetector::getVecAngle(float dy, float dx)
{
	float vecAngle = atan2(dy, dx) * 180 / pi;
	if (vecAngle <= -90) { vecAngle = -vecAngle - 90; }
	else if (vecAngle >= 0) { vecAngle = -vecAngle + 270; }
	else if (vecAngle < 0 && vecAngle > -90) { vecAngle = -vecAngle + 270; }
	return vecAngle;
}

// 3.vector<SingleArea>根据center.x值来排序
bool BiaopanDetector::SortByX(SingleArea &v1, SingleArea &v2)//注意：本函数的参数的类型一定要与vector中元素的类型一致  
{
	//升序排列  
	return v1.center.x < v2.center.x;
}


// 4.vector<MergeArea>根据angle值来排序
bool BiaopanDetector::SortByAngle(MergeArea &v1, MergeArea &v2)//注意：本函数的参数的类型一定要与vector中元素的类型一致  
{
	//升序排列  
	return v1.angle < v2.angle;
}

// 5.vector<MergeArea>根据response值来排序，降序
bool BiaopanDetector::SortByRes(MergeArea &v1, MergeArea &v2)//注意：本函数的参数的类型一定要与vector中元素的类型一致  
{
	//降序排列  
	return v1.response > v2.response;
}



// 获取一个cell的向量，dn是指有多少个方向
vector<float> BiaopanDetector::getCellData(Mat& mag, Mat& angle, int r, int c, int cellSize, int dn)
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
vector<float> BiaopanDetector::getHogData(Mat& originImg)
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

/* ------------------------------------ */



/* ------------------------------------ */
// vector<Ellipse>根据椭圆的面积大小值来排序，降序
bool BiaopanDetector::SortByEllipseArea(Ellipse& e1, Ellipse& e2)
{
	//降序排列  
	return (e1._a*e1._b) > (e2._a*e2._b);
}

/* ------------------------------------ */
// 引用iou+nms解决椭圆的分类问题，根据椭圆外接矩形的iou来计算
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
	// 如果后面的那个矩形的大部分面积都属于前面那个矩形的，那么后面的矩形可以删了
	double o = inter / r2.area();
	return (o >= 0) ? o : 0;
}

void BiaopanDetector::nms(vector<Rect>& proposals, const double nms_threshold)
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


// 计算坐标旋转后的点，这里输入的坐标是标准坐标系，输出的也是标准坐标系,记得是逆时针旋转
Point BiaopanDetector::rotate(float theta, Point a)
{
	float x = a.x;
	float y = a.y;
	return Point(int(cos(theta)*x - sin(theta)*y), int(sin(theta)*x + cos(theta)*y));
}

// 计算坐标旋转后的点，这里输入的坐标是标准坐标系，输出的也是标准坐标系,记得是逆时针旋转
Point BiaopanDetector::rotate(float theta, float x, float y)
{
	return Point(cos(theta)*x - sin(theta)*y, sin(theta)*x + cos(theta)*y);
}


// 切换到椭圆坐标，这里输入的坐标是图像坐标系，输出的是标准坐标系（先平移后旋转）
Point BiaopanDetector::origin2el(Point2f& center, float theta, Point& origin)
{
	float x = origin.x;
	float y = -origin.y;
	return rotate(theta, x - center.x, y + center.y);
}

// 切换到图像坐标，这里输入的坐标是标准坐标系，输出的是图像坐标系（先旋转后平移）
Point BiaopanDetector::el2origin(Point2f& center, float theta, Point& el)
{
	Point origin = rotate(theta, el.x, el.y);
	float x = origin.x;
	float y = -origin.y;
	return Point(x + center.x, y + center.y);
}


// 椭圆检测，传入原图像，图像缩放的比例以及模糊处理的程度，要输出的东西，就只有椭圆
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

	// 先进行缩放
	resize(img, image, Size(img.size[1] / scaleSize, img.size[0] / scaleSize));
	// 先保存一个预览图
	Mat3b image_show;
	cvtColor(image, image_show, COLOR_GRAY2BGR);
	// 再进行一层模糊
	GaussianBlur(image, image, Size(blurSize, blurSize), 3, 3);

	Canny(image, edge, 3, 9, 3);

	yaed.Detect(edge, ellsYaed);


	// 下面进行 iou+nms 把椭圆分开


	if (ellsYaed.size() == 0) { return output; }
	// 存放椭圆的外接矩形
	vector<Rect> rects;

	for (int index = 0; index < ellsYaed.size(); index++)
	{
		Ellipse& e = ellsYaed[index];
		// 找到长轴短轴
		int long_a = e._a >= e._b ? cvRound(e._a) : cvRound(e._b);
		int short_b = e._a < e._b ? cvRound(e._a) : cvRound(e._b);
		rects.push_back(Rect(cvRound(e._xc) - long_a, cvRound(e._yc) - long_a, 2 * long_a, 2 * long_a));
	}
	nms(rects, 0.7);
	// 从抑制后的矩形中寻找符合要求的椭圆
	vector<Ellipse> ells;
	for (int j = 0; j < rects.size(); j++)
	{
		float karea = rects[j].width * rects[j].height;
		float max_size = -1;
		Ellipse max_el;
		for (int i = 0; i < ellsYaed.size(); i++)
		{
			Ellipse& e = ellsYaed[i];
			// 找到长轴短轴
			int long_a = e._a >= e._b ? cvRound(e._a) : cvRound(e._b);
			int short_b = e._a < e._b ? cvRound(e._a) : cvRound(e._b);
			// 如果长宽比过小则舍去
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
		// 如果 max_size 仍然等于-1说明这个外接矩形里面没有满足要求的椭圆
		if (max_size > 0) { ells.push_back(max_el); }
	}
	// 这里必须加上 std 才行，要不然就调用cv的sort了，同时比较方法必须是个static的方法
	std::sort(ells.begin(), ells.end(), SortByEllipseArea);



	// 最后把椭圆还原回去
	for (int i = 0; i < ells.size(); i++)
	{
		Ellipse e = ells[i];
		e._xc *= scaleSize;
		e._yc *= scaleSize;
		e._a *= scaleSize;
		e._b *= scaleSize;
		output.push_back(e);
	}

	// 画出来
	yaed.DrawDetectedEllipses(image_show, ells);
	//imshow("ellipse detect", image_show);
	//waitKey();

	// 把目标椭圆传出去
	return output;

}

// 根据表盘ID 获取 表盘区域 roi_dst，传入的是原始图像（经过灰度化），表盘的id，输出的是 一定规格的 roi_dst 以及 旋转角度
Mat BiaopanDetector::getRoiDst(Mat& img, string& id, float& rotation)
{

	//调用QRDetector解析图片
	vector<BCoder> bcs = qrdetect.detect(img);
	Mat roi_dst;
	for (int i=0;i<bcs.size();i++)
	{
		BCoder bc = bcs[i];
		// 获取位置旋转角度
		rotation = bc.rotation;
		Point center = bc.center;


		// 首先是解析出他的信息，判断他是不是要找的表盘，如果不是就换下一个二维码检测continue。如果是就继续
		//.......//
		// 从上面那一步之后就可以确定这个二维码是代表着我们要找的表盘，所以接下面的就是读取表盘的问题



		// 先进行旋转
		Mat rotationMatrix;
		// 在getRotationMatrix2D中，角度为负，顺时针；角度为正，逆时针。第三个参数默认不用管
		rotationMatrix = getRotationMatrix2D(center, -rotation, 1);//计算旋转的仿射变换矩阵 

		// 仍然用椭圆检测来计算，当然也可以用 计算二维码的面积来确定 像素距离，又因为表盘大概大小也是可知的，就可以推出表盘的大致位置，但是不好用
		int scaleSize = initScaleSize;
		// 默认是模糊度为7，根据情况调大一点
		vector<Ellipse> ells = getElls(img, scaleSize, 7);
		// 寻找包含二维码中心点的那个椭圆，找到的第一个就break
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
			cout << "在检测ID为：..(目前还未结合信息来做)的二维码 表盘椭圆检测失败，请检查角度或者光照" << endl;
			continue;
		}

		// 根据目标椭圆生成roi_dst，送到指针检测模块中去

		/* 下面的部分只是缩放生成roi_dst */
		int long_a = el_dst._a >= el_dst._b ? cvRound(el_dst._a) : cvRound(el_dst._b);
		int short_b = el_dst._a < el_dst._b ? cvRound(el_dst._a) : cvRound(el_dst._b);
		int r_x = max(0, (cvRound(el_dst._xc) - long_a));
		int r_y = max(0, (cvRound(el_dst._yc) - long_a));
		// 超出尺寸的话就适当缩小
		int r_mx = min(img.cols, r_x + 2 * long_a);
		int r_my = min(img.rows, r_y + 2 * long_a);
		int n_width = min(r_mx - r_x, r_my - r_y);
		// 提取目标区域
		roi_dst = img(Rect(r_x, r_y, n_width, n_width));

		
		resize(roi_dst, roi_dst, Size(roi_width, cvRound(float(roi_dst.cols) / float(roi_dst.rows) * roi_width)));
		//imshow("roi_dst", roi_dst);
		//waitKey();
		/* ----------------------------------------------------------  */


		// 输出一个mat，这个mat包含原来二维码的四点信息
		cvtColor(img, BiaopanDetector::rs1, COLOR_GRAY2BGR);
		circle(rs1, bc.a, 10, Scalar(100, 0, 0), -1);
		circle(rs1, bc.b, 10, Scalar(0, 200, 0), -1);
		circle(rs1, bc.c, 10, Scalar(0, 0, 120), -1);
		circle(rs1, bc.d, 10, Scalar(50, 150, 150), -1);
		rs1 = rs1(Rect(r_x, r_y, n_width, n_width));
		resize(rs1, rs1, Size(roi_width, cvRound(float(rs1.cols) / float(rs1.rows) * roi_width)));
		

		// 因为这个表盘就是要找的，所以后面的二维码也不用再检测了
		break;
	}

	return roi_dst;
}


// 直接具体检测表盘的圆轮廓，同时会赋值ac_center
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



// 表盘中心检测，传入的是roi_dst，输出的是表盘中心位置
Point BiaopanDetector::getAcCenter(Mat& roi_dst)
{
	Point ac_center = Point(roi_width / 2, roi_width / 2);

	vector<Vec3f> circles;
	Mat centerArea = roi_dst(Rect(roi_width / 2 - center_search_radius, 
		roi_width / 2 - center_search_radius, 2 * center_search_radius, 2 * center_search_radius)).clone();


	GaussianBlur(centerArea, centerArea, Size(3, 3), 1, 1);
	// 这个局部阈值能使中间部分完全变白，从而可以探测得到
	adaptiveThreshold(centerArea, centerArea, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 9, 0);
	// 下面这种做法是比较冒险的。。因为不保证最后的输出是模糊还是清晰，同时无法确保这个阈值30在各个场景均适用
	// HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 30, 8, 0, 6);
	// 后面四个数字分别代表： 分辨率（可以理解为步进的倒数），两个圆之间最小距离，canny的低阈值，投票的累积次数（即有多少个点属于该圆），圆最小半径，圆最大半径
	// 一般投票值在 15 到 20间，如果拍摄视角特别偏，把中间圆变成椭圆，就需要降低投票值了，目前检测视角比较正常，所以只需要保持20
	HoughCircles(centerArea, circles, CV_HOUGH_GRADIENT, 1, 2, 150, 18, 10, 25);
	Point f_center = Point(center_search_radius / 2, center_search_radius / 2);
	float f2ac = 200;
	for (int i = circles.size() - 1; i >= 0; --i)
	{
		Point mCenter = Point(circles[i][0], circles[i][1]);
		// 下面两种办法，一种是取距离f_center最短，一种是面积最小,先采用第二种
		//float mDistance = point2point(mCenter, f_center);
		//if (mDistance < f2ac)
		//{
		//	f2ac = mDistance;
		//	ac_center = Point(mCenter.x + 200 - searchRadius, mCenter.y + 200 - searchRadius);
		//}
		// 也就是这个 f2ac 存储的是圆的面积。
		if (circles[i][2] < f2ac)
		{
			f2ac = circles[i][2];
			ac_center = Point(mCenter.x + roi_width / 2 - center_search_radius, mCenter.y + roi_width / 2 - center_search_radius);
		}
	}

	Mat show = roi_dst.clone();
	circle(show, ac_center, 3, Scalar(255, 255, 255), -1);

	// 画到rs1上
	circle(BiaopanDetector::rs1, ac_center, 3, Scalar(255, 255, 255), -1);

	//imshow("center", show);
	//waitKey();
	return ac_center;
}


// 拼合直线，传入直线群（vec4f中前两个值为头点，后两个值为尾点），输出的是直线群的序号组合
vector<vector<int>> BiaopanDetector::groupLines(vector<Vec4f> lines, Point& ac_center)
{
	float angelCos = cos(lineMaxAngle *  pi / 180);

	// 存储这些后线对
	vector<int> backs = vector<int>(lines.size());
	// 存储这些前线对，方便之后的计算
	vector<int> fronts = vector<int>(lines.size());

	// 初始化这些对的值，默认为-1
	for (int i = lines.size() - 1; i >= 0; --i)
	{
		backs[i] = -1;
		fronts[i] = -1;
	}
	// 这一步是对每条线判断前后线
	for (int i = lines.size() - 1; i >= 0; --i)
	{
		Vec4f& line1 = lines[i];
		// 搜索半径内，首先是属于自己后面的线，然后是中点相连所成的角度不能大过某个值，并且取距离自己最短的那些线，最后是看这条线是不是已经成为了别人的后线了
		int mIndex = i;
		float mDis = 1000;
		for (int j = lines.size() - 1; j >= 0; --j)
		{
			if (i == j)
				continue;
			Vec4f& line2 = lines[j];

			// 先判断是不是自己后面的线
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

		// 如果已经这个线已经是别人的后线了，那么要比较一下两个前线的长短，长的才有资格拿这个线
		if (fronts[mIndex] >= 0)
		{
			if (point2point(lines[fronts[mIndex]]) < point2point(lines[i])) {

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

	// 递归组合这些对并且得出最长的线
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

	return goal_lines;
}

// 指针检测，传入的是roi_dst，表盘中心，输出的是 指针的角度 
float BiaopanDetector::getPointer(Mat& roi_dst, Point& ac_center)
{
	Mat roi_thresh = roi_dst.clone();
	Mat show;
	cvtColor(roi_dst, show, COLOR_GRAY2BGR);

	adaptiveThreshold(roi_thresh, roi_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);
	// 下面尝试腐蚀多次，得出我们想要的指针的直线，而不是通过之前的步骤做到
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	// 存储第一次检测到的直线
	vector<Vec4f> tLines_detect;
	// 给直线区分好头点和尾点
	vector<Vec4f> tLines_pack;

	// 腐蚀
	for (int i=erode_times;i>0;i--)
	{
		erode(roi_thresh, roi_thresh, element);
	}

	// 创建检测器
	imwrite("1.jpg", roi_thresh);
	fld->detect(roi_thresh, tLines_detect);
	// 给他区分尾点和非尾点
	for (int i = 0; i < tLines_detect.size(); i++)
	{
		// 筛选掉那些距离中心比较远的线
		float distance = point2Line(ac_center, tLines_detect[i]);

		if (distance <= center2line)
		{
			Vec4f l = tLines_detect[i];
			// 还要分头尾两点
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

			// 画出来
			circle(show, Point(dl[0], dl[1]), 2, Scalar(255, 0, 255), -1);
			circle(show, Point(dl[2], dl[3]), 2, Scalar(0, 255, 255), -1);

			// 画到rs1上
			circle(BiaopanDetector::rs1, Point(dl[0], dl[1]), 2, Scalar(255, 0, 255), -1);
			circle(BiaopanDetector::rs1, Point(dl[2], dl[3]), 2, Scalar(0, 255, 255), -1);
		}
	}
	fld->drawSegments(show, tLines_pack);
	fld->drawSegments(BiaopanDetector::rs1, tLines_pack);


	// 下面拼合直线并且寻找最大的
	vector<vector<int>> groups = groupLines(tLines_pack, ac_center);

	// 找出最长的那个直线组合，显示出来
	float maxLength = 0;
	vector<int>& maxLine = groups[0];
	for (int i = groups.size() - 1; i >= 1; --i)
	{
		vector<int>& group = groups[i];

		float total_length = 0;
		for (int j = 0; j <= group.size() - 1; ++j)
		{
			int li = group[j];
			// 这里要进行平方，因为越长的占比应该越大
			total_length += point2point(tLines_pack[li]);
		}
		if (total_length > maxLength) { maxLength = total_length; maxLine = group; }
	}
	// 画出最长的线
	Scalar cc(255, 255, 255);
	for (int j = 0; j <= maxLine.size() - 1; ++j)
	{
		int li = maxLine[j];
		Vec4f ln = tLines_pack[li];
		Point point1 = Point(ln[0], ln[1]);
		Point point2 = Point(ln[2], ln[3]);
		line(show, point1, point2, cc, 2);
		// 输出到rs1上
		line(BiaopanDetector::rs1, point1, point2, cc, 2);
	}


	// 拿最长的线的结尾当做指针线的尾点
	Vec4f last_part = tLines_pack[maxLine[maxLine.size() - 1]];
	Point lastPoint = Point(last_part[2], last_part[3]);


	// 计算指针旋转角
	float dx = last_part[2] - ac_center.x;
	float dy = -last_part[3] + ac_center.y;
	// 计算与三四象限分割线的轴的夹角
	float vecAngle = atan2(dy, dx) * 180 / pi;
	if (vecAngle <= -90) { vecAngle = -vecAngle - 90; }
	else if (vecAngle >= 0) { vecAngle = -vecAngle + 270; }
	else if (vecAngle < 0 && vecAngle > -90) { vecAngle = -vecAngle + 270; }

	cout << "指针角度为： " << vecAngle << endl;

	//imshow("pointer", show);
	//waitKey(0);

	return vecAngle;
}


// 椭圆拟合，传入的是roi_dst，椭圆大概中心（可以不是精确的中心），输出的是 椭圆外接矩形
RotatedRect BiaopanDetector::fitELL(Mat& roi_dst, Point& ac_center)
{
	Mat show;
	cvtColor(roi_dst, show, COLOR_GRAY2BGR);
	cvtColor(roi_dst, BiaopanDetector::rs2, COLOR_GRAY2BGR);
	Mat equalizer;
	Mat otsu_thresh;
	Mat ada_thresh;

	float roi_area = roi_dst.cols * roi_dst.rows;
	// 先阈值处理，看情况决定是否进行 直方图均衡化
	equalizeHist(roi_dst, equalizer);
	adaptiveThreshold(equalizer, ada_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 0);
	threshold(equalizer, otsu_thresh, 0, 255, CV_THRESH_OTSU);

	vector<Vec4f> tLines_detect;
	vector<Vec4f> tLines_pack;
	// 创建检测器
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
	ls->detect(ada_thresh - otsu_thresh, tLines_detect);
	// 给他区分尾点和非尾点
	for (int i = 0; i < tLines_detect.size(); i++)
	{
		// 筛选掉那些距离中心比较远的线
		float distance = point2Line(ac_center, tLines_detect[i]);

		if (distance <= center2line)
		{
			Vec4f l = tLines_detect[i];
			// 还要分头尾两点
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

			// 画出来
			circle(show, Point(dl[0], dl[1]), 2, Scalar(255, 0, 255), -1);
			circle(show, Point(dl[2], dl[3]), 2, Scalar(0, 255, 255), -1);

			circle(BiaopanDetector::rs2, Point(dl[0], dl[1]), 2, Scalar(255, 0, 255), -1);
			circle(BiaopanDetector::rs2, Point(dl[2], dl[3]), 2, Scalar(0, 255, 255), -1);

		}
	}
	// 画出所有朝向中点的线
	ls->drawSegments(show, tLines_pack);
	ls->drawSegments(BiaopanDetector::rs2, tLines_pack);


	// bim+支持点方向分类 然后抽样获取椭圆
	// 给tLine进行分类（按角度分类），比如按照40度一个扇区，就有9个扇区 
	// 这个 tLine 和用来识别指针的 line是不一样的。
	vector<vector<int>> sangles_(360 / sangle);
	for (int i = tLines_pack.size() - 1; i >= 0; --i)
	{
		// 计算夹角的cos值
		int xd = tLines_pack[i][2] - ac_center.x;
		int yd = ac_center.y - tLines_pack[i][3];
		// 值域在 0~360之间
		float vangle = fastAtan2(yd, xd);
		sangles_[(int)vangle / sangle].push_back(i);
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
	// 存储当前的支持点个数
	int nowSnum = 5;
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
			Vec4f vvvv = tLines_pack[shanqu[sii]];
			Point ssss = Point(vvvv[2], vvvv[3]);
			candips.push_back(ssss);
		}
		// 拟合椭圆后计算支持点个数
		RotatedRect rect = fitEllipse(candips);



		// 检测椭圆的要求，拟合的椭圆的中心和acc_center相近且面积也占有roi_dst一定比例
		float area = pi * rect.size.width*rect.size.height / 4;
		float a = rect.size.width > rect.size.height ? rect.size.width : rect.size.height;
		float b = rect.size.width < rect.size.height ? rect.size.width : rect.size.height;

		if (point2point(rect.center, ac_center) <= ell_accept_distance && (area / roi_area) >= ellsize2roisize && (b / a) >= ell_ab_p)
		{
			// 存储支持点
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

	// 对拿到的支持点继续进行拟合
	vector<Point> supporters;



	for (int ki = 0; ki < bestSupportPoints.size(); ki++)
	{
		int tLineIndex = bestSupportPoints[ki];
		Point supporter = Point(tLines_pack[tLineIndex][2], tLines_pack[tLineIndex][3]);
		supporters.push_back(supporter);
		// 画出拟合点
		circle(show, supporter, 3, Scalar(0, 0, 0), -1);
		circle(BiaopanDetector::rs2, supporter, 3, Scalar(0, 0, 0), -1);
	}

	if (supporters.size() < 5)
	{
		cout << "椭圆拟合失败，符合要求的拟合点不足，要么降低拟合圆筛选要求，要么更改阈值方式！" << endl;
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


// 上轮廓分析包括识别数字，传入的是单个mser，输出的是response
vector<float> BiaopanDetector::readMser(Mat& mser)
{
	// otsu提取文字，然后开操作去除杂点，最后是分割识别
	Mat mser_roi_thresh;
	threshold(mser, mser_roi_thresh, 0, 255, CV_THRESH_OTSU);
	mser_roi_thresh = 255 - mser_roi_thresh;
	Mat e_element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	// 开操作去除杂点
	morphologyEx(mser_roi_thresh, mser_roi_thresh, MORPH_OPEN, e_element);
	// 前面把rect放大了一点，现在缩回来统计他的宽高比
	float wihi = (mser_roi_thresh.cols) / (float)(mser_roi_thresh.rows);

	vector<upper> uppers;
	vertical_projection(mser_roi_thresh, uppers);

	// 具体寻找多少个分割线，根据比例而定，当然也可以完全交给下面的算法自动寻找，下面的算法寻找到所有的分割点后要去头尾两个多余的分割部分
	// 先对所有的值进行从大到小进行排序，然后从头开始遍历，寻找某个点，他的7邻域是个凸曲线(因为寻找上轮廓过程中保证了函数是连续的，所以不会有奇值点)
	vector<upper> uppers_sort;
	for (int ei = 0; ei < uppers.size(); ei++) { uppers_sort.push_back(uppers[ei]); }
	// 显示上轮廓
	//waitKey();
	sort(uppers_sort.begin(), uppers_sort.end(), SortByUp);
	// 存储切割点，anchors存放的是分割点的序号，并不是坐标和值
	vector<int> anchors;
	// 存放anchors对应的极值点
	vector<extremer> extremers;
	int usize = uppers.size();

	if (wihi < 0.6) { usize = 0; }

	for (int ei = 0; ei < uppers_sort.size(); ei++)
	{
		// 领域空间分为两边走，往左往右走，寻找导数小于0的两个点，如果找的到那么这个点就是极小值点
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
			// 存放进 极值点 vector
			extremers.push_back({ uppers_sort[ei], lindex, rindex });
		}
	}

	vector<float> responses;
	bool is_singular = false;

	if (anchors.size() > 0)
	{
		// 整理一次分割点，分割点过近的当做一个分割点，true_anchors存放的是分割点的坐标
		// 一般分割最多就是3分割，为了增加容错性，给他再少个2（主要是因为数字可能比较集中）
		int close_thresh = uppers.size() / 3 - 2;
		//cout << "close_thresh: " << close_thresh << endl;
		vector<int> true_anchors;

		// 存储符合条件的极值点的序号
		vector<int> true_extremers;
		true_anchors.push_back(0);
		int pp = 0;
		// 整理分割点，把相近的分割点合为一个分割点（合后的分割点在中间）
		for (int ei = 0; ei < anchors.size(); ei++)
		{
			// 要防止过分割
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
		// 按照从小到大排序分割点
		sort(true_anchors.begin(), true_anchors.end());

		// 输出true_anchors
		//cout << endl;
		//cout << "第一次--true_anchors: [";
		//for (int ei = 0; ei < true_anchors.size(); ei++) { cout << true_anchors[ei] << ","; }
		//cout << "]" << endl;

		// 无论如何，最后还得来一次过分割检查，把在邻域内的相近分割点融合
		vector<int> temp_anchors;
		temp_anchors.push_back(0);
		// 下面是记录每个极值点的左右边界
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
				// 至少有一个符合条件的点进来后才可以开始计算max和min
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

		// 输出true_anchors
		//cout << endl;
		//cout << "第二次--true_anchors: [";
		//for (int ei = 0; ei < true_anchors.size(); ei++) { cout << true_anchors[ei] << ","; }
		//cout << "]" << endl;
		// 按照上面的分割点分割区域然后识别，最后合在一起
		// 注意上面给anchor添加了图片的两个端点


		for (int ei = 0; ei < true_anchors.size() - 1; ei++)
		{
			//Mat mmser = final_mser_roi.colRange(max(0, true_anchors[ei]-1), min(true_anchors[ei + 1]+1, width));
			Mat mmser = mser.colRange(max(0, true_anchors[ei]), min(true_anchors[ei + 1], mser.cols));
			vector<float> v = getHogData(mmser);
			Mat testData(1, 144, CV_32FC1, v.data());
			int response = svm->predict(testData);
			//imshow("[分割后的小图]", mmser);
			//waitKey();
			// response = -1表示这个不是数字，response = 10表示这个数字是融合数字
			if (response < 0 || response == 10) { is_singular = true; break; }
			else { responses.push_back(response); }

		}
	}

	else
	{
		// 直接预测
		vector<float> v = getHogData(mser);
		Mat testData(1, 144, CV_32FC1, v.data());
		int response = svm->predict(testData);
		// response = -1表示这个不是数字，response = 10表示这个数字是融合数字
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

		// 存放到singleArea中让后面继续融合
		Point ncenter;
		// 为了之后的运算
		return responses;
	}


}

// 融合，输入的是singleArea的集合，ac_center
vector<MergeArea> BiaopanDetector::mergeSingleArea(vector<SingleArea>& sas, Point& ac_center)
{
	// 规定融合的最短距离
	float merge_distance = 30;
	float min_merge_distance = 10;
	// 允许与angle 的误差
	int angle_error = 15;
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

					Point sacenter = sas[kii].center;
					// 寻找能融合的那个点，也就是距离他中心最短的那个点
					float min_dd = merge_distance;
					int target_sa = -1;
					for (int kjj = 0; kjj < sas.size(); kjj++)
					{
						// 这里需要查看kjj是否已经连接过kii，已经连过的就跳过(-1表示未连接，-2表示自己连自己)
						if (joinTable[kii][kjj] == -2 || joinTable[kii][kjj] > -1) { continue; }
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
			cc_indexs.push_back(sas[sas_index].cc_index);
			goal_set_areas.push_back(sas[sas_index]);
			x_ += sas[sas_index].center.x;
			y_ += sas[sas_index].center.y;
		}
		// 计算出新的中心
		Point merge_center = Point(int(x_ / goal_set_size), int(y_ / goal_set_size));


		// 按照x对goal_set_areas排序，并得出他的总response
		sort(goal_set_areas.begin(), goal_set_areas.end(), SortByX);
		float merge_response = 0;
		// 这里的 merge_response 和前面不同，因为前面的singleArea中含有一些已经融合的，应该先判断singleArea是多少位的。然后根据这个去求新的 merge_response
		vector<float> responses;

		for (int kjj = 0; kjj < goal_set_size; kjj++)
		{
			vector<float> mres = goal_set_areas[kjj].response;
			for (int kjjj = 0; kjjj < mres.size(); kjjj++)
			{
				responses.push_back(mres[kjjj]);
			}
		}

		// 判断是否温度计，如果是温度计需要用别的计算规则
		if (responses.size() >= 2 && responses[0] > 0 && responses[1] == 0)
		{
			// 按照某种规则计算这个response
			merge_response = responses[0] * pow(10, responses.size() - 1);
		}
		else
		{
			for (int kjj = 0; kjj < responses.size(); kjj++)
			{
				// 按照某种规则计算这个response
				merge_response += responses[kjj] * pow(10, -kjj);
			}
		}

		// 计算与三四象限分割线的轴的夹角并 渲染成 merge_area
		float ddy = -merge_center.y + ac_center.y;
		float ddx = merge_center.x - ac_center.x;
		float merge_angle = getVecAngle(ddy, ddx);

		// 筛选掉角度过大的可疑点
		if (merge_angle > 320 || merge_angle < 30) { continue; }

		merge_areas.push_back({ merge_response, merge_angle, cc_indexs, merge_center });

	}

	return merge_areas;
}

// 异值 检测，输入的是 MergeArea的集合，ac_center，输出的是符合条件的同时已经排好角度的 MergeArea的集合
vector<MergeArea> BiaopanDetector::removeSingular(vector<MergeArea>& merge_areas, Point& ac_center)
{

	/*
		下面是对这些 merge_area 进行筛选，把有些响应值奇怪的点去除掉

		方法是把merge_area看成一个点，其中response是y值，angle是x值，下面还有详细说明

	*/

	// 存储异值点,false表示该点是非奇异值，true表示该点为奇异值，一开始全部都归结为非奇异值
	vector<bool> singulars(merge_areas.size(), false);

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

	//cout << "第一次确认后" << endl;
	for (int kii = 0; kii < singulars.size(); kii++)
	{
		if (singulars[kii]) { continue; }
		//cout << "满足的响应值为： " << merge_areas[kii].response << endl;
		merges_1.push_back(merge_areas[kii]);
	}


	if (merges_1.size() == 0)
	{
		cout << "!!!!!!!!!!检测角度存在问题，请调整新角度继续检测!!!!!!!!!!!" << endl;
		return merges;
	}


	// 按照角度给所有mergeArea排序
	sort(merges_1.begin(), merges_1.end(), SortByAngle);


	// 这里再做多一次修正，sort完之后一定是严格单调递增的，如果出现递减或者不变的情况，那么就需要去除掉了
	merges.push_back(merges_1[0]);



	int merges_pos = 0;
	//cout << "第二次确认后" << endl;
	cout << "满足的响应值为： " << merges_1[0].response << endl;
	for (int kii = 1; kii < merges_1.size(); kii++)
	{
		if (merges_1[kii].response <= merges[merges_pos].response) { continue; }
		else { merges.push_back(merges_1[kii]); merges_pos++; cout << "满足的响应值为： " << merges_1[kii].response << endl; }
	}

	return merges;
	
}


// 最终的示数读取，输入的是 MergeArea的集合，表盘的指针角度，输出的是 最终读取示数
float BiaopanDetector::readAngle(vector<MergeArea>& merges, float pointerAngle)
{
	float pointerValue = -1;

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
				float delta_value = (merges[kii].response - merges[kii - 1].response) / (merges[kii].angle - merges[kii - 1].angle) * (pointerAngle - merges[kii - 1].angle);
				pointerValue = merges[kii - 1].response + delta_value;
				break;
			}
		}
	}
	return pointerValue;
}

// 示数读取，传入的是roi_dst，椭圆中心（用来计算mser区域中心与表盘半径的夹角），指针夹角，角度信息，是否投票获取角度，输出的是示数
// 之前是用椭圆拟合，这里放弃椭圆拟合，改用表盘具体外轮廓
float BiaopanDetector::readRoi(Mat& roi_dst, Point& ac_center, float pointerAngle, float mserAngle, bool isVoted)
{

	// 先以ac_center为中心进行角度矫正

	// rotationMatrix的第二个参数是逆时针旋转角度，刚好我们的mserAngle表示的是检测到表盘的顺时针旋转角度，要矫正就需要逆时针旋转
	Mat rotationMatrix = getRotationMatrix2D(ac_center, mserAngle, 1);//计算旋转的仿射变换矩阵 
	Mat rMat;
	warpAffine(roi_dst, rMat, rotationMatrix, Size(roi_dst.cols, roi_dst.rows));//仿射变换  


	// 展示的也是修正后的
	Mat show;
	cvtColor(rMat, show, COLOR_GRAY2BGR);

	// 看情况是否加入 直方图 均衡化
	Mat roi_thresh;
	equalizeHist(rMat, roi_thresh);

	// 直接用表盘外轮廓
	RotatedRect bestEl = ac_el;
	// 但是还要应用旋转
	bestEl.angle = bestEl.angle - mserAngle;

	// 画出满足条件的区域
	RotatedRect bestEl_scale0 = RotatedRect(bestEl.center, Size2f(bestEl.size.width, bestEl.size.height), bestEl.angle);
	RotatedRect bestEl_scale1 = RotatedRect(bestEl.center, Size2f(bestEl.size.width * 0.75, bestEl.size.height * 0.75), bestEl.angle);
	RotatedRect bestEl_scale2 = RotatedRect(bestEl.center, Size2f(bestEl.size.width * 0.4, bestEl.size.height * 0.4), bestEl.angle);
	// 画出中心
	circle(show, Point(bestEl.center.x, bestEl.center.y), 3, Scalar(0, 0, 0), -1);
	ellipse(show, bestEl_scale1, Scalar(200, 100, 150));
	ellipse(show, bestEl_scale2, Scalar(110, 30, 250));
	ellipse(show, bestEl_scale0, Scalar(0, 0, 0), 3);


	// mser检测
	vector<Rect> candidates = simpleMser(rMat);
	// 存储还未检测是否异值的mser，供检测异值模块使用
	vector<SingleArea> sas;

	// 这里只是把符合条件的candidate给挑选出来，给后面操作
	for (int i = 0; i < candidates.size(); ++i)
	{
		// 先标所有的mser
		rectangle(show, candidates[i], Scalar(200, 100, 30), 1);

		// 先筛选掉拟合圆外面的点
		Point ncenter = Point(candidates[i].x + candidates[i].width / 2, candidates[i].y + candidates[i].height / 2);
		Point newsc = origin2el(bestEl.center, bestEl.angle / (float)180 * pi, ncenter);
		float ndistance = sqrt(pow(newsc.x, 2) / pow(bestEl.size.width / 2, 2) + pow(newsc.y, 2) / pow(bestEl.size.height / 2, 2));
		// 这里的椭圆需要往内部缩一下，去除过多的咋点，比如刻度线
		// 同时又不能太里面，因为数字是在中心一定距离外的。
		if (ndistance >= 0.75 || ndistance < 0.4) { continue; }



		
		Rect mserRoi = Rect(candidates[i].x, candidates[i].y, candidates[i].width, candidates[i].height);
		Mat mser = rMat(mserRoi);
		Point mserCenter = Point(mserRoi.x + mserRoi.width/2, mserRoi.y + mserRoi.height/2);

		//imshow("mserRoi", mser);
		//imshow("readRoi", show);
		//waitKey();

		// 读取示数
		vector<float> response = readMser(mser);
		// 渲染成特殊形式
		if (response.size() > 0)
		{
			// 这里只标出被包含在区域内的mser并且识别成功的
			rectangle(show, candidates[i], Scalar(255, 255, 0), 1);

			// 渲染成singleArea
			sas.push_back({ response, i, mserCenter });
		}

	}

	vector<MergeArea> merges = mergeSingleArea(sas, ac_center);

	if (merges.size() == 0)
	{
		cout << "读取示数失败，请检查roi是否正确" << endl;
		return -1;
	}
	// 画出merge
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
		cout << "!!!!!!!!!!检测角度存在问题，请调整新角度继续检测!!!!!!!!!!!" << endl;
		return -1;
	}
	// 这里要注意，因为我们检测的图是经过我们矫正的图，所以指针在矫正后的图的角度是需要减去矫正的角度，
	// 又因为这个顺时针旋转角度很大，表示的其实是逆时针旋转角度，所以需要处理一下（类似除余操作）
	float pv = pointerAngle - mserAngle;
	if (pv < 0) { pv = pv + 360; }
	float finalValue = readAngle(merges, pv);

	if (finalValue < 0)
	{
		cout << "!!!!!!!!!!检测角度存在问题，请调整新角度继续检测!!!!!!!!!!!" << endl;
		return -1;
	}

	//imshow("readRoi", show);
	//waitKey();

	rs2 = show;

	return finalValue;
}



// 总的入口方法
float BiaopanDetector::detect(Mat& mat, string& id)
{
	// 检测是否黑白，如果不是黑白就转成黑白
	Mat img;
	if (mat.channels() > 1)
	{
		cvtColor(mat, img, COLOR_BGR2GRAY);
	}
	else
	{
		img = mat;
	}
	// 表盘区域，使用roi之前先检测该roi是否检测成功
	float rotation;
	Mat roi_dst = getRoiDst(img, id, rotation);
	if (roi_dst.cols > 0)
	{
		// 表盘轮廓
		RotatedRect acEl = getAcOutline(roi_dst);
		if (!acEl.angle) { cout << "表盘外轮廓识别错误，请调整光源或者角度 " << endl; return -2; }
		// 表盘中心
		Point ac_center = Point(acEl.center.x, acEl.center.y);
		// 表盘指针角度
		float pointerAngle = getPointer(roi_dst, ac_center);
		// 读取表盘
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

	string id = "第一个表盘";
	// 表盘区域，使用roi之前先检测该roi是否检测成功
	float rotation;
	Mat roi_dst = detector.getRoiDst(img, id, rotation);
	if (roi_dst.cols > 0)
	{
		// 表盘轮廓
		RotatedRect acEl = detector.getAcOutline(roi_dst);
		if (!acEl.angle) { cout << "表盘外轮廓识别错误，请调整光源或者角度 " << endl; return; }
		// 表盘中心
		Point ac_center = Point(acEl.center.x, acEl.center.y);
		// 表盘指针角度
		float pointerAngle = detector.getPointer(roi_dst, ac_center);
		// 读取表盘
		float value = detector.readRoi(roi_dst, ac_center, pointerAngle, rotation, false);

		cout << "示数是： " << value << endl;
	}
	imshow("rs1", detector.rs1);
	imshow("rs2", detector.rs2);
	waitKey();
}

// codeLen 的长度是 
void  BiaopanDetector::testThreshDetect(Mat& img, Point& ac_pos, float& ac_distance, float& ac_angle, float& angle_given, float& codeLen, float& dd)
{
	BiaopanDetector detector;
	detector.initBiaopan();
	Point img_center = Point(img.cols / 2, -img.rows / 2);
	// 根据摄像机给定的角度进行图像矫正
	// rotationMatrix的第二个参数是逆时针旋转角度，刚好我们的mserAngle表示的是检测到表盘的顺时针旋转角度，要矫正就需要逆时针旋转
	QRDetector qr;
	vector<BCoder> coders = qr.detect(img);
	BCoder bc = coders[0];
	// 得到角度
	ac_angle = bc.rotation - angle_given;
	// 得到边长
	float dis_a = detector.point2point(bc.a, bc.c);
	float dis_b = detector.point2point(bc.b, bc.d);

	// 得到实际距离与像素距离比值
	float ratio =  (dis_a + dis_b) / (2 * sqrt(2) * codeLen);
	// 计算相对坐标，并且转为标准坐标系，因为存在angle_given，需要再旋转一下坐标
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
