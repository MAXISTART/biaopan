#pragma once

#include <opencv2\opencv.hpp>
#include <algorithm>
#include "EllipseDetectorYaed.h"
#include <fstream>
#include <direct.h>
#include <unordered_map>


#include "QRDetector.h" 

#include<time.h>
#define random(x) (rand()%x)+1


#define DATA_DIR "D:\\OpenCV\\bin\\toy_data\\"
#define pi 3.1415926

using namespace std;




// 存储旋转矩形角的信息（投票信息）
struct ranglevoter
{
	// 有多少票
	int voteNum;
	// 这些投票的角度加起来是多少（用来给后面平均的）
	int sum;
	// 这个角度是多少
	int rangle;
};


// 存储上轮廓值以及坐标
struct upper
{
	// 存储其x值
	int x;
	// 存储其y值
	int y;

};

// 存储极值点（一个极值点不仅包含中心点，还有其左右边界（山谷的左右两个边界））
struct extremer
{
	upper center;
	int lindex;
	int rindex;
};

// 存储确认是数字的区域以及他们的中心点还有他们的响应
struct SingleArea
{
	// 存储响应值
	vector<float> response;
	// 存储里面的candidates的序号
	int cc_index;
	// 存储中心
	Point center;
};

// 存储融合的数字
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

class BiaopanDetector
{


public:
	/*一些参量*/
	float el_ab_p = 0.4;						// 筛选椭圆的条件，椭圆的长短轴之比
	int min_vec_num;							// 检测表盘时，表盘里面至少需要多少条面向中心的直线
	int roi_width = 400;						// 检测区域 roi_dst 的大小。

	int center_search_radius = 35;				// 表盘中心检测过程中，表盘检测范围的半径	
	float center2line = 30;						// 表盘中心检测过程中，允许直线距离表盘中心的距离

	int erode_times = 3;						// 指针检测前图像腐蚀的次数，腐蚀越多次，检测到的直线更少
	float line_search_radius = 80;				// 指针检测过程中，合并直线时，每个直线自己的搜索范围
	float lineMaxAngle = 4;						// 指针检测过程中，合并直线时，直线与直线之间最大的夹角

	int sangle = 40;							// 一椭圆拟合过程中，一个扇区的角度大小。
	int ranTimes = 1500;						// 一次椭圆拟合过程中，随机的最大次数
	float ell_accept_thresh = 8;				// 允许椭圆拟合中支持点最大的远离程度，这里的程度是乘上椭圆长轴的（具体看算法）
	float ell_accept_distance = 20;				// 椭圆拟合中，允许拟合圆与 ac_center 的最大距离
	float ellsize2roisize = 0.35;				// 椭圆拟合之中，允许拟合圆占 roi_dst 的最小比例
	float ell_ab_p = 0.85;						// 椭圆拟合中，除了面积要求，还要有长短轴比的要求，而且这个比椭圆检测的要求要高



		
	int rect_angle_err = 5;						// 读取示数时，判定两个mser旋转矩形角度是否相同的允许差值，用来进行角度投票
	double minMserArea = 20;					// 读取示数时，允许最小的mser区域的面积
	int mserRadius = 3;							// mser时分割到的矩阵再往外扩张的距离，这个距离对识别非常重要，尽量在数字3以上


	Mat rs1;									// roi检测+二维码检测结果+center+指针，roi肯定包含了椭圆检测的信息
	Mat rs2;									// 椭圆拟合+矫正
	Mat rs3;									// mser结果以及merge


	Ptr<cv::ml::SVM> svm;
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";

	QRDetector qrdetect;


	// 初始化一些参数
	void initBiaopan();

	/*一些计算方法*/

	// 计算点到直线的距离，这里输入的点是图像坐标的点
	float point2Line(Point point, Vec4f& line);
	// 计算点到直线的距离，这里的点是标准坐标的点
	float point2Line(float x, float y, float x1, float y1, float x2, float y2);
	// 计算点到点的距离
	float point2point(float x1, float y1, float x2, float y2);
	// 计算点到点的距离
	float point2point(float x1, float y1, int x2, int y2);
	// 计算点到点的距离
	float point2point(int x1, int y1, int x2, int y2);
	// 计算点到点的距离
	float point2point(Point point1, Point point2);
	// 计算点到点的距离
	float point2point(Vec4f line);
	// 计算两直线间的夹角cos值
	float line2lineAngleCos(Vec4f line1, Vec4f line2);
	// 重叠面积率计算
	double iou(const Rect& r1, const Rect& r2);
	void nms(vector<Rect>& proposals, const double nms_threshold);



	// 根据表盘的id获取表盘目标区域
	Mat getRoiDst(Mat& img, string& id, float& rotation);
	// 仅仅是获取椭圆，需要输入预处理的模糊程度
	vector<Ellipse> getElls(Mat& img, int& scaleSize, int blurSize);
	// 获取表盘中心
	Point getAcCenter(Mat& roi_dst);
	// 获取表盘指针
	float getPointer(Mat& roi_dst, Point& ac_center);
	// 拼合直线，传入直线群（vec4f中前两个值为头点，后两个值为尾点），输出的是直线群的序号组合
	vector<vector<int>> groupLines(vector<Vec4f> lines, Point& ac_center);
	// 椭圆拟合，传入的是roi_dst，椭圆大概中心（可以不是精确的中心），输出的是 椭圆外接矩形
	RotatedRect fitELL(Mat& roi_dst, Point& ac_center);
	// 读取示数
	float readRoi(Mat& roi_dst, Point& ac_center, float pointerAngle, float mserAngle, bool isVoted);
	// 读取单个mser，里面调用了上轮廓分割，供 readRoi 调用
	vector<float> readMser(Mat& mser);
	// 融合单体检测值。
	vector<MergeArea> mergeSingleArea(vector<SingleArea>& sas, Point& ac_center);
	// 异值 检测，输入的是 MergeArea的集合，ac_center，输出的是符合条件的 MergeArea的集合
	vector<MergeArea> removeSingular(vector<MergeArea>& sas, Point& ac_center);
	// 最终的示数读取，输入的是 MergeArea的集合，表盘的指针角度，输出的是 最终读取示数
	float readAngle(vector<MergeArea>& merge_areas, float pointerAngle);
	// vector<int>根据vector中的值来排序
	static bool SortByUp(upper &v1, upper &v2);



	// 获取一个cell的向量，dn是指有多少个方向
	vector<float> getCellData(Mat& mag, Mat& angle, int r, int c, int cellSize, int dn);
	// 获取hog向量
	vector<float> getHogData(Mat& originImg);



	// 计算坐标旋转后的点，这里输入的坐标是标准坐标系，输出的也是标准坐标系
	Point rotate(float theta, float x, float y);
	// 切换到椭圆坐标，这里输入的坐标是图像坐标系，输出的是标准坐标系（先平移后旋转）
	Point origin2el(Point2f& center, float theta, Point& origin);
	// 切换到图像坐标，这里输入的坐标是标准坐标系，输出的是图像坐标系（先旋转后平移）
	Point el2origin(Point2f& center, float theta, Point& el);


	// 计算出一个向量与椭圆的, a是椭圆的第一个轴长，b是椭圆的第二个轴长，theta是椭圆的倾斜角，xx1 和 xx2代表与椭圆相交的直线，最终返回的是 与xx1xx2向量同方向的交点(p1是出发点)
	Point anchor_on_el_line(float a, float b, float theta, Point2f& center, Point& xx1, Point& xx2);



	// 合并直线的操作

	// 1. 右搜索
	void backSearch(vector<bool>& isVisited, vector<int>& backs, vector<int>& goal_line);
	// 2. 左搜索
	void frontSearch(vector<bool>& isVisited, vector<int>& fronts, vector<int>& goal_line);

	// 合并数字区域的操作

	// 1. 连接点搜索
	void joinSearch(vector<bool>& isVisited, vector<int>& goal_set, vector<vector<int>>& joinTable);
	// 2.判断角度，输入的标准坐标系下向量的 y 与 x，输出的是 以标准坐标系中的三四象限中间轴为0度轴，顺时针旋转的角度
	float getVecAngle(float dy, float dx);
	// 3.vector<SingleArea>根据center.x值来排序
	static bool SortByX(SingleArea &v1, SingleArea &v2);
	// 4.vector<MergeArea>根据angle值来排序
	static bool SortByAngle(MergeArea &v1, MergeArea &v2);
	// 5.vector<MergeArea>根据response值来排序，降序
	static bool SortByRes(MergeArea &v1, MergeArea &v2);


	// 画出旋转矩阵
	void drawRotatedRect(Mat& drawer, RotatedRect& rrect);
	// Mser目标检测 + nms，其中rrects存储每个Rect中的那个旋转矩阵
	vector<Rect> rotateMser(Mat& srcImage, vector<cv::RotatedRect>& rrects);
	// Mser目标检测 + nms，这个是简洁版的mser，不带旋转矩阵
	vector<Rect> simpleMser(Mat& srcImage);
	// vector<ranglevoter>根据ranglevoter中的voteNum排序
	static bool SortByVote(ranglevoter &v1, ranglevoter &v2);
	// 寻找上轮廓
	void vertical_projection(Mat& input_src, vector<upper>& uppers);
	// 总的方法
	float detect(Mat& mat, string& id);


	// vector<Ellipse>根据椭圆的面积大小值来排序，降序
	static bool SortByEllipseArea(Ellipse& e1, Ellipse& e2);


	// 工具方法
	string int2str(const int &int_temp);
	int str2int(const string &string_temp);
	vector<string> readTxt(string file);
	// 分割字符串
	vector<string> splitString(const std::string& s, const std::string& c);



};