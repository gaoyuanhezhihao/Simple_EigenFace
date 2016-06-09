#include "ImgProcess.h"

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
	Mat normed_im;
	Size new_size;
	for (; it != image_vector.end(); ++it) {
		//new_size = calc_new_size(it, rows_max, column_max);
		//resize(*it, im_resized, new_size);
		//im_resized.copyTo(im_std_size(Rect(0, 0, new_size.width, new_size.height)));
		normlize_mat(*it, im_size, normed_im);
		//it->copyTo(im_resized(Rect(0, 0, it->cols, it->rows)));
		train_imags.push_back(normed_im.reshape(0, 1));
	}
	return true;
}

/* resize the image to the max_size. normalize the image.
*---------------------------------------------------------
* @src: the original image.
* @max_size: maximum size of the images.
* @normed_im: result.
*/
bool normlize_mat(Mat & src, Size max_size, Mat & normed_im) {
	Mat im_resized = Mat::zeros(max_size.height, max_size.width, CV_64FC1);
	resize(src, im_resized, max_size);
	normed_im = im_resized;
	//normalize(im_resized, normed_im, 100);
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




// Normalizes a given image into a value range between 0 and 255.
Mat norm_0_255(const Mat& src) {
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

bool minus_average_face(Mat & imgs, Mat& average_face) {
	average_face = Mat::zeros(1, imgs.cols, imgs.type());
	int i = 0;
	for (i = 0; i < imgs.rows; ++i) {
		average_face += imgs.row(i);
	}
	average_face /= imgs.rows;
	for (i = 0; i < imgs.rows; ++i) {
		 imgs.row(i) -=average_face;
	}
	return true;
}