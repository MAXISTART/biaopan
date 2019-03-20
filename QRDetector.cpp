#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include "opencv2/opencv.hpp"  
#include <iostream>
#include "zbar.h"  
#include "BCoder.h"
#include "QRDetector.h"
#include <iostream>

#define pi 3.1415926

using namespace zbar;
using namespace cv;
using namespace std;



QRDetector::QRDetector()
{
	
}


vector<BCoder> QRDetector::detect(Mat& img)
{
	// 调用zbar检测QR二维码并且封装成多个BCoder传出去
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

	vector<BCoder> coders;

	Mat img_show;
	cvtColor(img, img_show, CV_GRAY2RGB);
	int width = img.cols;
	int height = img.rows;

	uchar *raw = (uchar *)img.data;
	// wrap image data  
	Image image(width, height, "Y800", raw, width * height);
	// scan the image for qrcodes  
	int n = scanner.scan(image);


	// extract results  
	for (Image::SymbolIterator symbol = image.symbol_begin();
		symbol != image.symbol_end();
		++symbol) {
		vector<Point> vp;
		// do something useful with results  
		int n = symbol->get_location_size();
		for (int i = 0; i < n; i++) {
			vp.push_back(Point(symbol->get_location_x(i), symbol->get_location_y(i)));
		}
		// 如果vp不是4个，说明识别的不是QR码，或者识别不全，相当于识别不到，直接跳过
		if (vp.size() != 4)
		{
			continue;
		}

		// 获取中心点以及角度
		BCoder coder;
		coder.a = vp[0];
		coder.b = vp[1];
		coder.c = vp[2];
		coder.d = vp[3];

		int k1 = -vp[2].y + vp[0].y;
		int k2 = vp[2].x - vp[0].x;
		int k3 = -vp[3].y + vp[1].y;
		int k4 = vp[3].x - vp[1].x;
		int y = (k1*k3*(vp[1].x-vp[0].x)-k2*k3*vp[0].y+k1*k4*vp[1].y) / (k1*k4 - k2 * k3);
		int x = (k1*vp[0].x+k2*(-y+vp[0].y)) / k1;
		Point center = Point(x, y);
		coder.center = center;



		vector<Point> que;
		que.push_back(vp[1]);
		que.push_back(vp[0]);
		que.push_back(vp[3]);
		que.push_back(vp[2]);

		// 获取角度方向及角度
		cout << "以下五个旋转值理应相差不大，如果相差很大， 那就是有问题了" << endl;

		vector<float> vecAngles;
		for (int i = 0; i < 4; i++)
		{
			float dx = que[i].x - x;
			float dy = -que[i].y + y;
			// 计算与三四象限分割线的轴的夹角
			float vecAngle = atan2(dy, dx) * 180 / pi;
			if (vecAngle <= -90) { vecAngle = -vecAngle - 90; }
			else if (vecAngle >= 0) { vecAngle = -vecAngle + 270; }
			else if (vecAngle < 0 && vecAngle > -90) { vecAngle = -vecAngle + 270; }

			vecAngle -= (45 + i * 90);
			if (vecAngle < 0) { vecAngle = 360 + vecAngle; }
			vecAngles.push_back(vecAngle);
		}

		// 这里需要修正下，防止四个角度差距过大
		float reference = vecAngles[0];
		cout << reference << endl;
		float total = 0;
		for (int i = 1; i < 4; i++)
		{
			float dis = abs(vecAngles[i] - reference);
			if (dis < 10) { total += dis; }
			// 如果差距太大，有可能是相差了一个周期而已
			else
			{
				dis = abs(dis - 360);
				total += dis;
			}
			cout << reference + dis << endl;
		}

		float average = total / 3;

		coder.rotation = reference + average;

		cout << reference + average << endl;
		cout << "请确保二维码贴的方向是正确的" << endl;
		cout << "===================================" << endl;

		coders.push_back(coder);
		// 这是用来演示识别效果的
		vector<Scalar> ss;
		ss.push_back(Scalar(100, 0, 0));
		ss.push_back(Scalar(0, 200, 0));
		ss.push_back(Scalar(0, 0, 120));
		ss.push_back(Scalar(50, 150, 150));

		for (int i = 0; i < vp.size(); i++) {
			circle(img_show, vp[i], 8, ss[i], -1);
		}
		circle(img_show, center, 8, Scalar(0, 100, 90), -1);

		resize(img_show, img_show, Size(400, cvRound(float(img_show.rows) / float(img_show.cols) * 400)));

		//imshow("img_show", img_show);
		//waitKey();
	}
	return coders;

}


int test()
{
	string img_path = "D:\\VcProject\\biaopan\\biaopan1\\data\\2.jpg";

	Mat img = imread(img_path, 0);

	QRDetector qr;
	qr.detect(img);

	return 1;
}