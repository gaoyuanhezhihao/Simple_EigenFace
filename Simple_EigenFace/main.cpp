#include "opencv2\core\core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2\imgproc\imgproc.hpp"
#include "ImgProcess.h"
#include "Config.h"
#include "face.h"
#include <map>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;



int main(int argc, char ** argv) {
	vector<Mat> train_im_vec, test_im_vec;
	map<string, universal_type> config_map;
	Size std_im_size;
	read_config_file(argv[1], config_map);
	int image_count = 0;
	int test_image_count = 0;
	image_count << config_map["image count"];
	test_image_count << config_map["test image count"];
	string train_image_dir;
	train_image_dir << config_map["train image dir"];
	string test_image_dir;
	test_image_dir << config_map["test image dir"];
	train_im_vec.reserve(image_count);
	//read train image.
	int i = 0;
	string im_id;
	Mat train_imgs;
	Mat train_im_8bit;
	Mat train_im_64bit;
	for (i = 1; i <= image_count; i++) {
		im_id = i<10? '0'+to_string(i): to_string(i);
		//cout << train_image_dir + '\\' + im_id +".jpg" << endl;
		train_im_8bit = imread(train_image_dir + '\\' + im_id + ".jpg", IMREAD_GRAYSCALE);
		train_im_8bit.convertTo(train_im_64bit, CV_64FC1);
		train_im_vec.push_back(train_im_64bit);
	}
	Mat average_face;
	concatenate_image_row(train_im_vec, train_imgs, std_im_size);
	minus_average_face(train_imgs, average_face);
	// pca
	// Number of components to keep for the PCA:
	int num_components = 10;

	// Perform a PCA:
	PCA pca(train_imgs, Mat(), CV_PCA_DATA_AS_ROW, num_components);

	// And copy the PCA results:
	Mat mean = pca.mean.clone();
	Mat eigenvalues = pca.eigenvalues.clone();
	Mat eigenvectors = pca.eigenvectors.clone();

	// The mean face:
	imwrite("average face.jpg", average_face.reshape(1,std_im_size.height));

	// The first three eigenfaces:
	imwrite("pc1.jpg", norm_0_255(pca.eigenvectors.row(0)).reshape(1, std_im_size.height));
	imwrite("pc2.jpg", norm_0_255(pca.eigenvectors.row(1)).reshape(1, std_im_size.height));
	imwrite("pc3.jpg", norm_0_255(pca.eigenvectors.row(2)).reshape(1, std_im_size.height));

	// build face base.
	Mat face_base;
	build_face_base(train_imgs, eigenvectors, face_base);

	// test the image.
	Mat test_im;
	int success_count = 0;
	int prodict_id = 0;
	Mat test_im_64bit;
	for (i = 1; i <= test_image_count; ++i){
		im_id = i<10 ? '0' + to_string(i) : to_string(i);
		//cout << train_image_dir + '\\' + im_id +".jpg" << endl;
		test_im = imread(test_image_dir + '\\' + im_id + ".jpg", IMREAD_GRAYSCALE);
		test_im.convertTo(test_im_64bit, CV_64FC1);
		prodict_id = recognize_face(test_im_64bit, eigenvectors, std_im_size, face_base, average_face);
		if (prodict_id == i - 1) {
			++success_count;
		}
		cout << i - 1 << "-th face -->" << prodict_id << endl;
	}
	cout << "the success count is: " << success_count << endl;
	char wait;
	cout << "input any char to exit" << endl;
	cin >> wait;
	////debug
	//im = imread(train_image_dir + '\\' + to_string(11) + ".jpg", IMREAD_GRAYSCALE);
	//imwrite("original image.jpg", im);
	//Mat rebuild_im = train_imgs.row(11-1).clone();
	//imwrite("rebuild image.jpg", rebuild_im.reshape(0, std_im_size.height));
	//waitKey(0);
	return 0;
}
