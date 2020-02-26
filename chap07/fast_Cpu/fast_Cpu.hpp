#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
namespace cv_Cpu
{

	template<int patternSize>
	int cornerScore_Cpu(const uchar* ptr, const int pixel[], int threshold);
}
