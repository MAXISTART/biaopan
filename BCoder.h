#pragma once

#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include "opencv2/opencv.hpp"  


using namespace cv;

class BCoder
{
public:
	Point a;
	Point b;
	Point c;
	Point d;
	Point center;
	// 顺时针旋转角度
	float rotation;

	// 存储这个二维码的信息

};
