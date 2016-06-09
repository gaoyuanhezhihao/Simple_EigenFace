#ifndef IMGPROCESS_H
#define IMGPROCESS_H

#include "opencv2\core\core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2\imgproc\imgproc.hpp"
using namespace std;
using namespace cv;

bool concatenate_image_row(vector<Mat> & image_vector, Mat& train_imags, Size& im_size);
bool normlize_mat(Mat & src, Size max_size, Mat & normed_im);
Size calc_new_size(vector<Mat>::iterator & it_ref, int rows_max, int cols_max);
Mat norm_0_255(const Mat& src);
bool minus_average_face(Mat & imgs, Mat& average_face);
#endif //IMGPROCESS_H