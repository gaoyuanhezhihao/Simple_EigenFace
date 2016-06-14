#ifndef FACE_H
#define FACE_H

#include "opencv2\core\core.hpp"
#include <list>
#include<limits>
using namespace std;
using namespace cv;
struct face_location
{
	double error;// the possibility of the image belongs to a real face.
	Rect face_rect;
	face_location() :error(std::numeric_limits<double>::infinity()), face_rect(Rect(0,0,0,0)){}
};

double dist_to_face_mean(Mat & face_base_mean, Mat & feature_id);
double calc_mse(Mat &test_img, Mat &img_rebuild);
bool back_project_img(Mat & feature_id, Mat & img_rebuild, Mat & eigen_faces, const Mat & average_face, const Size std_size);
bool build_face_base(const Mat & train_imags, const Mat & eigen_faces, Mat & face_base);
bool project_to_eigen_face(const Mat & eigen_faces, const Mat & face_img, Mat & feature);
int recognize_face(Mat & im_tested, const Mat & eigen_faces, Size std_size, Mat & face_base, Mat & average_face);
int evaluate_eigen_face_number(const Mat & train_imgs, const Mat & eigenvectors, \
	const string test_image_dir, const int test_image_count, \
	Size std_im_size, Mat average_face, int num_eigen_face);
double calc_bias_to_center(Mat & face_base_mean, Mat & feature_id);
double analize_face_base(Mat &face_base, Mat &mean_face_base);
bool findFace(Mat& test_image, double min_scale, double max_scale, \
	double scale_step, Mat &eigen_faces, Size std_im_size, \
	Mat &face_base, int number_faces, string result_save_path, Mat & average_face);
double update_face_location(face_location & new_face_location, list<face_location>& face_found);
bool draw_rect_to_face(Mat &test_im, list<face_location>& face_found);
bool check_overlap(list<face_location>& face_found, face_location & new_face_location);
#endif