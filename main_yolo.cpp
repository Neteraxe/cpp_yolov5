#include "yolo.h"

// for file list
#include <filesystem>
namespace fs = std::filesystem;

clock_t t0, t1;

namespace Deepideal
{
	void YOLO::drawPred(int &classId, float &conf, int &left, int &top, int &right, int &bottom, Mat &frame) // Draw the predicted bounding box
	{
		//Draw a rectangle displaying the bounding box
		rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

		//Get the label for the class name and its confidence
		string label = format("%.2f", conf);
		label = this->classes[classId] + ":" + label;

		//Display the label at the top of the bounding box
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	}

	void YOLO::sigmoid(Mat *out, int &length)
	{
		float *pdata = (float *)(out->data);
		int i = 0;
		for (i = 0; i < length; i++)
		{
			pdata[i] = 1.0 / (1 + expf(-pdata[i]));
		}
	}

	string YOLO::detect(Mat &src, Mat &frame)
	{
		// it is the most consumed place
		Mat blob;
		blobFromImage(frame, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
		this->net.setInput(blob);

		vector<Mat> outs;

		std::vector<String> &names = this->net.getUnconnectedOutLayersNames();

		// 1. 减少图片大小
		// 2.
		//this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
		this->net.forward(outs, names);

		//0.0005s
		/////generate proposals
		vector<int> classIds;
		vector<float> confidences;
		vector<Rect> boxes;

		float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
		int n = 0, q = 0, i = 0, j = 0, nout = this->classes.size() + 5, row_ind = 0;

		if (!outs.size())
		{
			cout << "model doesn't detect even one box……\n";
			return "[]";
		}

		for (n = 0; n < 3; n++) ///3 layers
		{
			int num_grid_x = (int)(this->inpWidth / this->stride[n]);
			int num_grid_y = (int)(this->inpHeight / this->stride[n]);
			for (q = 0; q < 3; q++) ///reshape and permute
			{
				const float anchor_w = this->anchors[n][q * 2];
				const float anchor_h = this->anchors[n][q * 2 + 1];
				for (i = 0; i < num_grid_y; i++)
				{
					for (j = 0; j < num_grid_x; j++)
					{
						float *pdata = (float *)outs[0].data + row_ind * nout;
						float box_score = sigmoid_x(pdata[4]);
						if (box_score > this->objThreshold)
						{
							Mat scores = outs[0].row(row_ind).colRange(5, outs[0].cols);
							Point classIdPoint;
							double max_class_socre;
							// Get the value and location of the maximum score
							minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);

							float temp_mcs = (float)max_class_socre;
							max_class_socre = sigmoid_x(temp_mcs);
							if (max_class_socre > this->confThreshold)
							{
								float cx = (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * this->stride[n]; ///cx
								float cy = (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * this->stride[n]; ///cy
								float w = powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;			 ///w
								float h = powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h;			 ///h

								int left = (cx - 0.5 * w) * ratiow;
								int top = (cy - 0.5 * h) * ratioh; ///���껹ԭ��ԭͼ��

								classIds.push_back(classIdPoint.x);
								confidences.push_back(max_class_socre);
								boxes.push_back(Rect(left, top, (int)(w * ratiow), (int)(h * ratioh)));
							}
						}
						row_ind++;
					}
				}
			}
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		vector<int> indices;
		NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

		for (size_t i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];
			Rect box = boxes[idx];

			int right = box.x + box.width;
			int bottom = box.y + box.height;
			this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
						   right, bottom, frame);
		}

		std::stringstream result;
		Rect box;
		for (size_t i = 0; i < indices.size(); ++i)
		{
			box = boxes[i];
			int x1 = box.tl().x;
			int y1 = box.tl().y;
			int x2 = box.br().x;
			int y2 = box.br().y;

			result << "[" << x1 << "," << y1 << "," << x2 << "," << y2 << "]";
		}
		return result.str();
	}

	// must correctly signs
	string YOLO::inference(string &image, void *userData)
	{
		Mat srcimg = imread(image);
		Mat detimg = srcimg;

		t0 = clock();
		string result = this->detect(srcimg, detimg);
		t1 = clock();

		string kWinName = to_string((double)(t1 - t0) / CLOCKS_PER_SEC).append("s");
		cout << "time spent: " << kWinName << endl;

		namedWindow(kWinName, WINDOW_NORMAL);
		imshow(kWinName, srcimg);
		waitKey(0);
		destroyAllWindows();

		return result;
	}
	/*
	int main()
	{
		YOLO yolo_model(0.8, 0.5, 0.5, "yolov5s.onnx");

		string imgpath = "bus.jpg";
		Mat srcimg = imread(imgpath);
		yolo_model.detect(srcimg);

		string kWinName = to_string((double)(t1 - t0) / CLOCKS_PER_SEC * 1000).append("ms") + to_string((double)(t2 - tt) / CLOCKS_PER_SEC * 1000).append("ms");

		namedWindow(kWinName, WINDOW_NORMAL);
		imshow(kWinName, srcimg);
		waitKey(0);
		destroyAllWindows();
	}
	*/
}

int main(int argc, char *argv[])
{
	// yolo.exe *.onnx *.img
	if (argc < 3)
	{
		cout << "(1) .onnx (2)imgs/ .img";
		return -1;
	}
	Deepideal::YOLO yolo(0.5, 0.5, 0.5, argv[1]);
	std::string path = argv[2];
	int userdata[] = {1, 2, 3};
	if ((fs::is_directory(path)))
	{
		t0 = clock();
		for (const auto &entry : fs::directory_iterator(path))
		{
			std::cout << entry.path() << "\n";
			// read path, because path is more extensive
			string result = yolo.inference(entry.path().string(), userdata);
			cout << result << "\n";
		}
		t1 = clock();
		cout << to_string((double)(t1 - t0) / CLOCKS_PER_SEC).append("s");
	}
	else
	{
		t0 = clock();
		string result = yolo.inference(path, userdata);
		t1 = clock();
		cout << to_string((double)(t1 - t0) / CLOCKS_PER_SEC).append("s");
	}
}
