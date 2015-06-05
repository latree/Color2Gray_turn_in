#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


string type2str(int type);
void nayatani(Mat & src);
int clamp(int f, int min, int max);
void local_adjust(Mat & src, Mat & src_lab, double p, double k);
//void local_adjust(Mat & src, Mat & src_lab, double p, double k[]);

int main(int argc, char** argv)
{	
	cout << "Have " << argc << " arguments:" << endl;
	for (int i = 0; i < argc; ++i) {
		cout << argv[i] << endl;
	}
	
	//IplImage* image = cvLoadImage("C:\\Users\\latree\\Dropbox\\CS510-Computational_Photography\\project\\firstTry\\firstTry\\apple.png");
	//Mat img = Mat(image);
	Mat img = imread(argv[1]);
	Mat luv_img;
	Mat lab_img;
	Mat gamma_mapping_img;
	double p = 0.25;
	double k = 0.5;
	//double k[4] = { 0.5, 0.5, 0.5, 0.5};

	// lab color space
	cvtColor(img, lab_img, CV_BGR2Lab);

	// luv color space
	cvtColor(img, luv_img, CV_BGR2Luv);

	
	gamma_mapping_img = luv_img;


	//calculate the chromatic object lightness channel
	nayatani(gamma_mapping_img);

	// trasform color image to grayscale image
	cvtColor(gamma_mapping_img, gamma_mapping_img, CV_BGR2GRAY);

	// local adjustment help to improve the quality of greyscale result we get from above function
	local_adjust(gamma_mapping_img, lab_img, p, k);

    //cvtColor(img, img, CV_BGR2GRAY);
	imshow("Display window: ", gamma_mapping_img);
	waitKey(0);
	
	system("pause");
	return 0;
	
}

// type checking function for testing purpose 
string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {

	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

// testing function for changing rgb values in mat
Mat change_rgb(Mat resource){
	for (int x = 0; x > resource.rows; x++)
	{
		for (int y = 0; y > resource.cols; y++){

			Vec3b color = resource.at<Vec3b>(Point(x, y));
			if (color[0] > 10 && color[1] > 10 && color[2]> 10)
			{
				color[0] = 0;
				color[1] = 0;
				color[2] = 0;

				resource.at<Vec3b>(Point(x, y)) = color;
			}
		}

	}
	return resource;
}

// get the correct value between 0 - 255
int clamp(int f, int min, int max){
	return (f) < (min) ? (min) : (f) > (max) ? (max) : (f);
}

//Calculate chromatic object lightness channel
//Map to the grayscale

void nayatani(Mat & src){
	double hue, qhue, kbr, suv, gamma;
	double adaptlum = 20.0;
	double u_white = 0.20917;
	double v_white = 0.48810;

	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			src.at<Vec3b>(i, j)[1] = (int) (src.at<Vec3b>(i, j)[1] - u_white);
			src.at<Vec3b>(i, j)[2] = (int)(src.at<Vec3b>(i, j)[2] - v_white);
			Vec3b luv = src.at<Vec3b>(i, j);


			hue = atan2(luv[2], luv[1]);
			qhue = -0.01585 - 0.03016*cos(hue) - 0.04556*cos(2 * hue) - 0.02667*cos(3 * hue) - 0.00295*cos(4 * hue) + 0.14592*sin(hue) + 0.05084*sin(2 * hue) - 0.01900*sin(3 * hue) - 0.00764*sin(4 * hue);
			kbr = 0.2717*(6.469 + 6.362*pow(adaptlum, 0.4495)) / (6.469 + pow(adaptlum, 0.4495));

			suv = 13 * pow(pow(luv[1], 2) + pow(luv[2], 2), 0.5);
			gamma = 1 + (-0.1340*qhue + 0.0872*kbr)*suv;
			
			src.at<Vec3b>(i, j)[0] = clamp((int)(gamma*src.at<Vec3b>(i, j)[0] * 2.5599), 0, 255);
		}
	}
	return;
}

//Decompose image into several bandpass
//Increase local contrast by amount of lambda for each scale in laplacian pyramid
//Sum all the scales together and add it to grayscale value from gamma mapping

void local_adjust(Mat & src, Mat & src_lab, double p, double k){
	double gray_l_cntrst = 0.0;
	double gray_cntrst = 0.0; 
	double color_cntrst = 0.0;
	double lambda = 0.0;
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			if ((j + 3) < src.cols){
				int temp_j1 = j + 1;
				int temp_j2 = j + 2;
				int temp_j3 = j + 3;
				gray_l_cntrst = (double)(src_lab.at<Vec3b>(i, j)[0] -src_lab.at<Vec3b>(i, temp_j1)[0]);
				gray_cntrst = pow(pow(gray_l_cntrst, 2), 0.5);
				color_cntrst = pow(pow(src_lab.at<Vec3b>(i, j)[0] - src_lab.at<Vec3b>(i, temp_j1)[0], 2) + pow(src_lab.at<Vec3b>(i, j)[1] - src_lab.at<Vec3b>(i, temp_j1)[1], 2) + pow(src_lab.at<Vec3b>(i, j)[2] - src_lab.at<Vec3b>(i,temp_j2)[2], 2), 0.5);
				if (gray_cntrst == 0)
					lambda = color_cntrst / 0.00001;
				else
					lambda = color_cntrst / gray_cntrst;

				lambda = pow(lambda, p);

				src.at<uchar>(i,j) = src.at<uchar>(i,j) + (k*lambda*gray_l_cntrst);
			}
		}
	}
	return;
}



/*
void local_adjust(Mat & src, Mat & src_lab, double p, double k[]){
	double gray_l_cntrst[4];
	double gray_cntrst[4];
	double color_cntrst[4];
	double lambda[4];
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			if ((j + 3) < src.cols){
				int temp_j1 = j + 1;
				int temp_j2 = j + 2;
				int temp_j3 = j + 3;

					gray_l_cntrst[0] = (double)(src_lab.at<Vec3b>(i, j)[0] - src_lab.at<Vec3b>(i, temp_j1)[0]);
					gray_cntrst[0] = pow(pow(gray_l_cntrst[0], 2), 0.5);
					color_cntrst[0] = pow(pow(src_lab.at<Vec3b>(i, j)[0] - src_lab.at<Vec3b>(i, temp_j1)[0], 2) + pow(src_lab.at<Vec3b>(i, j)[1] - src_lab.at<Vec3b>(i, temp_j1)[1], 2) + pow(src_lab.at<Vec3b>(i, j)[2] - src_lab.at<Vec3b>(i, temp_j1)[2], 2), 0.5);
					if (gray_cntrst[0] == 0)
						lambda[0] = color_cntrst[0] / 0.00001;
					else
						lambda[0] = color_cntrst[0] / gray_cntrst[0];

					lambda[0] = pow(lambda[0], p);

					/////////
					gray_l_cntrst[1] = (double)(src_lab.at<Vec3b>(i, j)[0] - src_lab.at<Vec3b>(i+1, j)[0]);
					gray_cntrst[1] = pow(pow(gray_l_cntrst[1], 2), 0.5);
					color_cntrst[1] = pow(pow(src_lab.at<Vec3b>(i, j)[0] - src_lab.at<Vec3b>(i+1, j)[0], 2) + pow(src_lab.at<Vec3b>(i, j)[1] - src_lab.at<Vec3b>(i+1, j)[1], 2) + pow(src_lab.at<Vec3b>(i, j)[2] - src_lab.at<Vec3b>(i+1, j)[2], 2), 0.5);
					if (gray_cntrst[1] == 0)
						lambda[1] = color_cntrst[1] / 0.00001;
					else
						lambda[1] = color_cntrst[1] / gray_cntrst[1];

					lambda[1] = pow(lambda[1], p);
					
					/////////

					gray_l_cntrst[2] = (double)(src_lab.at<Vec3b>(i, j)[0] - src_lab.at<Vec3b>(i-1, j)[0]);
					gray_cntrst[2] = pow(pow(gray_l_cntrst[2], 2), 0.5);
					color_cntrst[2] = pow(pow(src_lab.at<Vec3b>(i, j)[0] - src_lab.at<Vec3b>(i-1, j)[0], 2) + pow(src_lab.at<Vec3b>(i, j)[1] - src_lab.at<Vec3b>(i-1, j)[1], 2) + pow(src_lab.at<Vec3b>(i, j)[2] - src_lab.at<Vec3b>(i-1, j)[2], 2), 0.5);
					if (gray_cntrst[2] == 0)
						lambda[2] = color_cntrst[2] / 0.00001;
					else
						lambda[2] = color_cntrst[2] / gray_cntrst[1];

					lambda[2] = pow(lambda[2], p);

					///////////
					gray_l_cntrst[3] = (double)(src_lab.at<Vec3b>(i, j)[0] - src_lab.at<Vec3b>(i, j-1)[0]);
					gray_cntrst[3] = pow(pow(gray_l_cntrst[3], 2), 0.5);
					color_cntrst[3] = pow(pow(src_lab.at<Vec3b>(i, j)[0] - src_lab.at<Vec3b>(i, j-1)[0], 2) + pow(src_lab.at<Vec3b>(i, j)[1] - src_lab.at<Vec3b>(i, j-1)[1], 2) + pow(src_lab.at<Vec3b>(i, j)[2] - src_lab.at<Vec3b>(i, j-1)[2], 2), 0.5);
					if (gray_cntrst[3] == 0)
						lambda[3] = color_cntrst[3] / 0.00001;
					else
						lambda[3] = color_cntrst[3] / gray_cntrst[3];

					lambda[3] = pow(lambda[3], p);
					

					src.at<uchar>(i, j) = src.at<uchar>(i, j) + (k[0] * lambda[0] * gray_l_cntrst[0] + k[1] * lambda[1] * gray_l_cntrst[1] + k[2] * lambda[2] * gray_l_cntrst[2] + k[3] * lambda[3] * gray_l_cntrst[3]);
			}
		}
	}
	return;
}
*/