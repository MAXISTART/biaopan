#pragma once
#include "BCoder.h"
#include "zbar.h"  

using namespace zbar;
using namespace std;

class QRDetector
{

public:
	vector<BCoder> detect(Mat& img);
	QRDetector();

private :
	ImageScanner scanner;
};