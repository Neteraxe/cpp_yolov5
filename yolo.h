#ifdef USECPPLIBRARY
#ifdef CPPLIBRARY_EXPORTS
#define CPPAPI __declspec(dllexport)
#endif
#else
#define CPPAPI
#endif

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

#define EOF (-1)

namespace Deepideal
{
	class YOLO
	{
	public:
		YOLO(float confThreshold, float nmsThreshold, float objThreshold, string netname) : confThreshold(confThreshold),
																							nmsThreshold(nmsThreshold),
																							objThreshold(objThreshold),
																							netname(netname)
		{
			ifstream ifs(this->classesFile);
			string line;
			while (getline(ifs, line))
				this->classes.push_back(line);

			this->net = readNetFromONNX(this->netname);
			// TODO
			// change type to speed up
			this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		};
		string detect(Mat &src, Mat &frame);
		string inference(string &image, void *userData);
		~YOLO()
		{
			delete[] anchors;
			delete[] stride;
			//delete[] result;
		};

	private:
		const float anchors[3][6] = {{10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0}, {116.0, 90.0, 156.0, 198.0, 373.0, 326.0}};
		const float stride[3] = {8.0, 16.0, 32.0};
		const string classesFile = "coco.names";
		const int inpWidth = 512;
		const int inpHeight = 512;
		//int inpWidth = 892;
		//int inpHeight = 892;
		float confThreshold;
		float nmsThreshold;
		float objThreshold;
		string netname;
		string result;

		vector<string> classes;
		Net net;
		void drawPred(int &classId, float &conf, int &left, int &top, int &right, int &bottom, Mat &frame);
		void sigmoid(Mat *out, int &length);
	};

	float sigmoid_x(float &x)
	{
		return (float)(1.f / (1.f + exp(-x)));
	}
}
