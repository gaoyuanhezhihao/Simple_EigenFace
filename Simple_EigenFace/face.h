#ifndef FACE_H
#define FACE_H

#include "opencv2\core\core.hpp"
using namespace std;
using namespace cv;
struct face
{
	double possibility;// the possibility of the image belongs to a real face.
	Mat ori_im;
	Mat	normed_im;// resized and normed image.
	Mat feature;
	int real_id;
};

bool build_face_base(const Mat & train_imags, const Mat & eigen_faces, Mat & face_base);
bool project_to_eigen_face(const Mat & eigen_faces, const Mat & face_img, Mat & feature);
int recognize_face(Mat & im_tested, const Mat & eigen_faces, Size std_size, Mat & face_base, Mat & average_face);
int evaluate_eigen_face_number(const Mat & train_imgs, const Mat & eigenvectors, \
	const string test_image_dir, const int test_image_count, \
	Size std_im_size, Mat average_face, int num_eigen_face);
#endif