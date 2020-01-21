#include "bladecv.hpp"
#include <stdio.h>
#include <math.h>
#include "openai_io.hpp"

using namespace fcv;

using namespace std;

int main()
{
	string filename = "a15.jpg";
	Mat src_img = cv::imread(filename);  

	cv::Size src_sz = src_img.size();  
	cv::Size dst_sz(src_sz.height, src_sz.width);  
	int len = std::max(src_img.cols, src_img.rows);  
	printf("%d,%d\n",src_img.cols, src_img.rows);
	Point p = Point(len / 2., len / 2.);

	//use cv::mat
	cv::Mat cvmat2 = imread(filename, 1);
		
	/* Show fcvMat */	
	mark_info info;
	info.type = TEXT;
	strcpy(info.textstr, "OPEN AI LAB");
	info.font_size = 20;
	info.width = 150;
	info.height = 50;
	info.index = 1;
	info.x = 10;
	info.y = 20;
	namedWindow("OPEN AI LAB", WINDOW_NORMAL);
	imshow("OPEN AI LAB", cvmat2, &info);
	waitkey(1000);
	//destroyWindow("OPEN AI LAB");
	
	return 0;
}
