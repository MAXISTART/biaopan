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




// �洢��ת���νǵ���Ϣ��ͶƱ��Ϣ��
struct ranglevoter
{
	// �ж���Ʊ
	int voteNum;
	// ��ЩͶƱ�ĽǶȼ������Ƕ��٣�����������ƽ���ģ�
	int sum;
	// ����Ƕ��Ƕ���
	int rangle;
};


// �洢������ֵ�Լ�����
struct upper
{
	// �洢��xֵ
	int x;
	// �洢��yֵ
	int y;

};

// �洢��ֵ�㣨һ����ֵ�㲻���������ĵ㣬���������ұ߽磨ɽ�ȵ����������߽磩��
struct extremer
{
	upper center;
	int lindex;
	int rindex;
};

// �洢ȷ�������ֵ������Լ����ǵ����ĵ㻹�����ǵ���Ӧ
struct SingleArea
{
	// �洢��Ӧֵ
	vector<float> response;
	// �洢�����candidates�����
	int cc_index;
	// �洢����
	Point center;
};

// �洢�ںϵ�����
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

class BiaopanDetector
{


public:
	/*һЩ����*/
	float el_ab_p = 0.4;						// ɸѡ��Բ����������Բ�ĳ�����֮��
	int min_vec_num;							// ������ʱ����������������Ҫ�������������ĵ�ֱ��
	int roi_width = 400;						// ������� roi_dst �Ĵ�С��

	int center_search_radius = 35;				// �������ļ������У����̼�ⷶΧ�İ뾶	
	float center2line = 30;						// �������ļ������У�����ֱ�߾���������ĵľ���

	int erode_times = 3;						// ָ����ǰͼ��ʴ�Ĵ�������ʴԽ��Σ���⵽��ֱ�߸���
	float line_search_radius = 80;				// ָ��������У��ϲ�ֱ��ʱ��ÿ��ֱ���Լ���������Χ
	float lineMaxAngle = 4;						// ָ��������У��ϲ�ֱ��ʱ��ֱ����ֱ��֮�����ļн�

	int sangle = 40;							// һ��Բ��Ϲ����У�һ�������ĽǶȴ�С��
	int ranTimes = 1500;						// һ����Բ��Ϲ����У������������
	float ell_accept_thresh = 8;				// ������Բ�����֧�ֵ�����Զ��̶ȣ�����ĳ̶��ǳ�����Բ����ģ����忴�㷨��
	float ell_accept_distance = 20;				// ��Բ����У��������Բ�� ac_center ��������
	float ellsize2roisize = 0.35;				// ��Բ���֮�У��������Բռ roi_dst ����С����
	float ell_ab_p = 0.85;						// ��Բ����У��������Ҫ�󣬻�Ҫ�г�����ȵ�Ҫ�󣬶����������Բ����Ҫ��Ҫ��



		
	int rect_angle_err = 5;						// ��ȡʾ��ʱ���ж�����mser��ת���νǶ��Ƿ���ͬ�������ֵ���������нǶ�ͶƱ
	double minMserArea = 20;					// ��ȡʾ��ʱ��������С��mser��������
	int mserRadius = 3;							// mserʱ�ָ�ľ������������ŵľ��룬��������ʶ��ǳ���Ҫ������������3����


	Mat rs1;									// roi���+��ά������+center+ָ�룬roi�϶���������Բ������Ϣ
	Mat rs2;									// ��Բ���+����
	Mat rs3;									// mser����Լ�merge


	Ptr<cv::ml::SVM> svm;
	string modelPath = "D:\\VcProject\\biaopan\\data\\model.txt";

	QRDetector qrdetect;


	// ��ʼ��һЩ����
	void initBiaopan();

	/*һЩ���㷽��*/

	// ����㵽ֱ�ߵľ��룬��������ĵ���ͼ������ĵ�
	float point2Line(Point point, Vec4f& line);
	// ����㵽ֱ�ߵľ��룬����ĵ��Ǳ�׼����ĵ�
	float point2Line(float x, float y, float x1, float y1, float x2, float y2);
	// ����㵽��ľ���
	float point2point(float x1, float y1, float x2, float y2);
	// ����㵽��ľ���
	float point2point(float x1, float y1, int x2, int y2);
	// ����㵽��ľ���
	float point2point(int x1, int y1, int x2, int y2);
	// ����㵽��ľ���
	float point2point(Point point1, Point point2);
	// ����㵽��ľ���
	float point2point(Vec4f line);
	// ������ֱ�߼�ļн�cosֵ
	float line2lineAngleCos(Vec4f line1, Vec4f line2);
	// �ص�����ʼ���
	double iou(const Rect& r1, const Rect& r2);
	void nms(vector<Rect>& proposals, const double nms_threshold);



	// ���ݱ��̵�id��ȡ����Ŀ������
	Mat getRoiDst(Mat& img, string& id, float& rotation);
	// �����ǻ�ȡ��Բ����Ҫ����Ԥ�����ģ���̶�
	vector<Ellipse> getElls(Mat& img, int& scaleSize, int blurSize);
	// ��ȡ��������
	Point getAcCenter(Mat& roi_dst);
	// ��ȡ����ָ��
	float getPointer(Mat& roi_dst, Point& ac_center);
	// ƴ��ֱ�ߣ�����ֱ��Ⱥ��vec4f��ǰ����ֵΪͷ�㣬������ֵΪβ�㣩���������ֱ��Ⱥ��������
	vector<vector<int>> groupLines(vector<Vec4f> lines, Point& ac_center);
	// ��Բ��ϣ��������roi_dst����Բ������ģ����Բ��Ǿ�ȷ�����ģ���������� ��Բ��Ӿ���
	RotatedRect fitELL(Mat& roi_dst, Point& ac_center);
	// ��ȡʾ��
	float readRoi(Mat& roi_dst, Point& ac_center, float pointerAngle, float mserAngle, bool isVoted);
	// ��ȡ����mser������������������ָ�� readRoi ����
	vector<float> readMser(Mat& mser);
	// �ںϵ�����ֵ��
	vector<MergeArea> mergeSingleArea(vector<SingleArea>& sas, Point& ac_center);
	// ��ֵ ��⣬������� MergeArea�ļ��ϣ�ac_center��������Ƿ��������� MergeArea�ļ���
	vector<MergeArea> removeSingular(vector<MergeArea>& sas, Point& ac_center);
	// ���յ�ʾ����ȡ��������� MergeArea�ļ��ϣ����̵�ָ��Ƕȣ�������� ���ն�ȡʾ��
	float readAngle(vector<MergeArea>& merge_areas, float pointerAngle);
	// vector<int>����vector�е�ֵ������
	static bool SortByUp(upper &v1, upper &v2);



	// ��ȡһ��cell��������dn��ָ�ж��ٸ�����
	vector<float> getCellData(Mat& mag, Mat& angle, int r, int c, int cellSize, int dn);
	// ��ȡhog����
	vector<float> getHogData(Mat& originImg);



	// ����������ת��ĵ㣬��������������Ǳ�׼����ϵ�������Ҳ�Ǳ�׼����ϵ
	Point rotate(float theta, float x, float y);
	// �л�����Բ���꣬���������������ͼ������ϵ��������Ǳ�׼����ϵ����ƽ�ƺ���ת��
	Point origin2el(Point2f& center, float theta, Point& origin);
	// �л���ͼ�����꣬��������������Ǳ�׼����ϵ���������ͼ������ϵ������ת��ƽ�ƣ�
	Point el2origin(Point2f& center, float theta, Point& el);


	// �����һ����������Բ��, a����Բ�ĵ�һ���᳤��b����Բ�ĵڶ����᳤��theta����Բ����б�ǣ�xx1 �� xx2��������Բ�ཻ��ֱ�ߣ����շ��ص��� ��xx1xx2����ͬ����Ľ���(p1�ǳ�����)
	Point anchor_on_el_line(float a, float b, float theta, Point2f& center, Point& xx1, Point& xx2);



	// �ϲ�ֱ�ߵĲ���

	// 1. ������
	void backSearch(vector<bool>& isVisited, vector<int>& backs, vector<int>& goal_line);
	// 2. ������
	void frontSearch(vector<bool>& isVisited, vector<int>& fronts, vector<int>& goal_line);

	// �ϲ���������Ĳ���

	// 1. ���ӵ�����
	void joinSearch(vector<bool>& isVisited, vector<int>& goal_set, vector<vector<int>>& joinTable);
	// 2.�жϽǶȣ�����ı�׼����ϵ�������� y �� x��������� �Ա�׼����ϵ�е����������м���Ϊ0���ᣬ˳ʱ����ת�ĽǶ�
	float getVecAngle(float dy, float dx);
	// 3.vector<SingleArea>����center.xֵ������
	static bool SortByX(SingleArea &v1, SingleArea &v2);
	// 4.vector<MergeArea>����angleֵ������
	static bool SortByAngle(MergeArea &v1, MergeArea &v2);
	// 5.vector<MergeArea>����responseֵ�����򣬽���
	static bool SortByRes(MergeArea &v1, MergeArea &v2);


	// ������ת����
	void drawRotatedRect(Mat& drawer, RotatedRect& rrect);
	// MserĿ���� + nms������rrects�洢ÿ��Rect�е��Ǹ���ת����
	vector<Rect> rotateMser(Mat& srcImage, vector<cv::RotatedRect>& rrects);
	// MserĿ���� + nms������Ǽ����mser��������ת����
	vector<Rect> simpleMser(Mat& srcImage);
	// vector<ranglevoter>����ranglevoter�е�voteNum����
	static bool SortByVote(ranglevoter &v1, ranglevoter &v2);
	// Ѱ��������
	void vertical_projection(Mat& input_src, vector<upper>& uppers);
	// �ܵķ���
	float detect(Mat& mat, string& id);


	// vector<Ellipse>������Բ�������Сֵ�����򣬽���
	static bool SortByEllipseArea(Ellipse& e1, Ellipse& e2);


	// ���߷���
	string int2str(const int &int_temp);
	int str2int(const string &string_temp);
	vector<string> readTxt(string file);
	// �ָ��ַ���
	vector<string> splitString(const std::string& s, const std::string& c);



};