#include "opencv2\core\core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2\imgproc\imgproc.hpp"
#include "Config.h"
#include <map>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

bool concatenate_image_row(vector<Mat> & image_vector, Mat& train_imags, Size& im_size);
Size calc_new_size(vector<Mat>::iterator & it_ref, int rows_max, int cols_max);

int main(int argc, char ** argv) {
	vector<Mat> train_im_vec;
	map<string, universal_type> config_map;
	Size im_size;
	read_config_file(argv[1], config_map);
	int image_count = 0;
	image_count << config_map["image count"];
	string image_dir;
	image_dir << config_map["image dir"];
	//cout << image_dir << '\n' << image_count << endl;
	train_im_vec.reserve(image_count);
	//read image.
	int i = 0;
	string im_id;
	Mat train_imgs;
	Mat im;
	for (i = 1; i <= image_count; i++) {
		im_id = i<10? '0'+to_string(i): to_string(i);
		//cout << image_dir + '\\' + im_id +".jpg" << endl;
		train_im_vec.push_back(imread(image_dir + '\\' + im_id + ".jpg", IMREAD_GRAYSCALE));
	}
	concatenate_image_row(train_im_vec, train_imgs, im_size);
	im = imread(image_dir + '\\' + to_string(11) + ".jpg", IMREAD_GRAYSCALE);
	imshow("original image", im);
	Mat rebuild_im = train_imgs.row(11-1).clone();
	imshow("rebuild image", rebuild_im.reshape(0, im_size.height));
	waitKey(0);
	return 0;
}
/* resize the images to the same size. Press them to an row vector. Concatenate them into a matrix.
* -------------------------------------------------------------------------------------------------
* @image_vector: vector containin the original images.
* @train_imags: concatenated image will be contained here.
*/
bool concatenate_image_row(vector<Mat> & image_vector, Mat& train_imags, Size& im_size) {
	// find the max row size and max column size.
	int rows_max = 0, column_max = 0;
	vector<Mat>::const_iterator it_im_vec = image_vector.cbegin();
	for (; it_im_vec != image_vector.cend(); ++it_im_vec) {
		if (it_im_vec->rows > rows_max) {
			rows_max = it_im_vec->rows;
		}
		if (it_im_vec->cols > column_max) {
			column_max = it_im_vec->cols;
		}
	}
	assert(rows_max != 0 && column_max != 0);
	im_size.width = column_max;
	im_size.height = rows_max;
	// resize every image to the max size and concatenate them.
	vector<Mat>::iterator it = image_vector.begin();
	Mat im_resized=Mat::zeros(rows_max,column_max,image_vector[0].type());
	Size new_size;
	for (; it != image_vector.end(); ++it) {
		new_size = calc_new_size(it, rows_max, column_max);
		im_resized = Mat::zeros(rows_max, column_max, image_vector[0].type());
		resize(*it, im_resized(Rect(0, 0, new_size.width, new_size.height)), new_size);
		//it->copyTo(im_resized(Rect(0, 0, it->cols, it->rows)));
		train_imags.push_back(im_resized.reshape(0, 1));
	}
	return true;
}

Size calc_new_size(vector<Mat>::iterator & it_ref, int rows_max, int cols_max) {
	Size new_size;
	if (it_ref->rows * cols_max > it_ref->cols * rows_max) {
		new_size.height = rows_max;
		new_size.width = it_ref->cols * rows_max / it_ref->rows;
		new_size.width = new_size.width > cols_max ? cols_max : new_size.width;
	} 
	else {
		new_size.width = cols_max;
		new_size.height = it_ref->rows * cols_max / it_ref->cols;
		new_size.height = new_size.height > rows_max ? rows_max : new_size.height;
	}
	return new_size;
}

