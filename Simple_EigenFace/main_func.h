#include "opencv2\core\core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2\imgproc\imgproc.hpp"
#include "ImgProcess.h"
#include "Config.h"
#include "face.h"
#include <map>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;
using namespace cv;

void recognize_test(const Mat & train_imgs, const Mat & eigenvectors, \
					const string test_image_dir, const int test_image_count, \
					Size std_im_size, Mat average_face, \
					string result_save_filename, int train_im_count) {
	std::ofstream model_eval_result;
	model_eval_result.open(result_save_filename);
	model_eval_result << "number of eigen face | success rate" << endl;
	int success_count = 0;
	int i = 0;
	for (i = 1; i <= train_im_count; ++i) {
		success_count = evaluate_eigen_face_number(train_imgs, eigenvectors, test_image_dir, test_image_count, std_im_size, average_face, i);
		model_eval_result << i << '|' << (double)success_count / (double)train_im_count << endl;
	}
	char wait;
	cin >> wait;
}

void print_help() {
	cout << "-r: recognize test" << '\n' << \
		"-f: find image" << endl;
}

bool find_face_test(map<string, universal_type> &config_map, const Mat & eigenvectors,\
					Mat &average_face, Size std_im_size, const Mat & train_imgs) {
	string path_group_image;
	path_group_image << config_map["image to find face"];
	int num_eigen_faces;
	num_eigen_faces << config_map["number of eigen faces"];
	Mat test_im_8bit = imread(path_group_image, IMREAD_GRAYSCALE);
	Mat test_im_64bit;
	test_im_8bit.convertTo(test_im_64bit, CV_64FC1);

	double min_scale = 0;
	min_scale << config_map["min scale"];
	double max_scale = 0;
	max_scale << config_map["max scale"];
	double scale_step = 0;
	scale_step << config_map["scale step"];
	// copy the eigen face from eigen vector.
	int i = 0;
	Mat sub_eigen_faces;
	for (i = 0; i < num_eigen_faces; ++i){
		sub_eigen_faces.push_back(eigenvectors.row(i));
	}
	// build face base.
	Mat face_base;
	build_face_base(train_imgs, sub_eigen_faces, face_base);
	// find the face
	int number_faces = 0;
	number_faces << config_map["number of faces in the image"];
	string result_save_path;
	result_save_path << config_map["findFace result path"];
	findFace(test_im_64bit, min_scale, max_scale, scale_step, sub_eigen_faces,
			std_im_size, face_base, number_faces, result_save_path, average_face);
	return true;
}