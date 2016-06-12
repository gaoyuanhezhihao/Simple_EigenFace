#include "face.h" 
#include "ImgProcess.h"
#include <limits>
#include <iostream>


/* build face base using eigen faces.
* --------------------------------------
* @train_images:[K*N] matrix. with each row represent one image.
* @eigen_faces: [M*N] matrix. with each row represent one eigen face.
* @face_base: [K*M] matrix. with each row represent the feature of one image.
*/
bool build_face_base(const Mat & train_imags, const Mat & eigen_faces, Mat & face_base){
	
	int id = 0;
	Mat face_feature = Mat::zeros(1, eigen_faces.rows, CV_64FC1);
	for (id = 0; id < train_imags.rows; ++id) {
		project_to_eigen_face(eigen_faces, train_imags.row(id), face_feature);
		face_base.push_back(face_feature);
	}
	return true;
}

/* project the face_img to eigen_faces space. a feature will be generated.
* ------------------------------------------------------------------------
* @eigen_faces: [M*N] matrix with M eigen faces and each face is a (1*N) vector.
* @face_img: [1*N] matrix.
* @feature: [1*M] matrix.
*/
bool project_to_eigen_face(const Mat & eigen_faces, const Mat & face_img, Mat & feature) {
	int r = 0;
	feature = eigen_faces * face_img.t();
	transpose(feature, feature);
	return true;
}


/* recognize a face from the face_base space.Return the id of the similiariest
*-----------------------------------------------------------------------------
* @im_tested: image to be reconized.
* @eigen_faces: eigen face space.
* @Size std_size: standard size the tested image should be transformed to.
* @face_base: features of the train face.
* @average_face: average face of the train sample.
*-----------------------------------------------------------------------------
* return: id the face similiest to the tested image.
*/
int recognize_face(Mat & im_tested, const Mat & eigen_faces, Size std_size, Mat & face_base, Mat & average_face) {
	Mat im_test_new;
	Mat feature_test=Mat::zeros(1,eigen_faces.cols,CV_64FC1);
	normlize_mat(im_tested, std_size, im_test_new);
	im_test_new = im_test_new.reshape(0, 1);
	im_test_new -= average_face;
	project_to_eigen_face(eigen_faces, im_test_new, feature_test);
	int i = 0;
	int best_id = 0;
	//double max_similarity = 0;
	//double simi = 0;
	double min_dif = std::numeric_limits<double>::infinity();
	double dif = 0;
	// compare every base image one by one.
	for (i = 0; i < face_base.rows; ++i) {
		//simi = face_base.row(i).dot(feature_test);
		dif = norm(face_base.row(i), feature_test, NORM_L2);
		if (min_dif>dif) {
			min_dif = dif;
			best_id = i;
		}
	}
	cout << "test face feature: " << feature_test << endl;
	cout << "best : " << face_base.row(best_id) << endl;
	return best_id;
}

/* evalute the model to different number of eigen faces.
*-------------------------------------------------------
* @train_imags: the concatenated faces.
* @eigenvectors:
* @test_image_dir: the directory containing the test images.
* @test_image_count: counts of the test images.
* @std_im_size: the size of the stand image size. every image should be resize to it.
* @average_face: average face of the train faces.
* @num_eigen_face: number of the eigen faces used in this model.
*---------------------------------------------------------------
* return: the success count
*/
int evaluate_eigen_face_number(const Mat & train_imgs, const Mat & eigenvectors,\
								const string test_image_dir, const int test_image_count,\
								Size std_im_size, Mat average_face, int num_eigen_face) {
	// copy the eigen face from eigen vector.
	int i = 0;
	Mat sub_eigen_faces;
	for (i = 0; i < num_eigen_face; ++i){
		sub_eigen_faces.push_back(eigenvectors.row(i));
	}
	// build face base.
	Mat face_base;
	build_face_base(train_imgs, sub_eigen_faces, face_base);

	// test the image.
	Mat test_im;
	int success_count = 0;
	int prodict_id = 0;
	Mat test_im_64bit;
	string im_id;
	for (i = 1; i <= test_image_count; ++i){
		im_id = i<10 ? '0' + to_string(i) : to_string(i);
		//cout << train_image_dir + '\\' + im_id +".jpg" << endl;
		test_im = imread(test_image_dir + '\\' + im_id + ".jpg", IMREAD_GRAYSCALE);
		test_im.convertTo(test_im_64bit, CV_64FC1);
		prodict_id = recognize_face(test_im_64bit, sub_eigen_faces, std_im_size, face_base, average_face);
		if (prodict_id == i - 1) {
			++success_count;
		}
		cout << i - 1 << "-th face -->" << prodict_id << endl;
	}
	cout << "the success count is: " << success_count << endl;
	return success_count;
}
