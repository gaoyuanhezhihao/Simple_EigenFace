#include "face.h" 
#include "ImgProcess.h"
#include <limits>
#include <iostream>
#include <algorithm>

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

/* calculate the unpossibility of the test_img to be a face.
* ----------------------------------------------------------
* @test_img: images in stand size[R*C].
* @face_base: [M*N] matrix. M is the number of the train image.
				N is the number of dimension of the feature.
* @eigen_faces: [N*S] matrix. N is the number of eigen faces used.
				S=R*C is the number of the pixels per standard image.
* @face_base_mean: [1*N] matrix.
* @face_base_bias_variance: the variance of the bias to the center of face base.
* @average face:
*-------------------------------------------------------------------------
* return: the error of the image to be a face.
*/
double is_face(Mat &test_img, Mat & face_base, Mat &eigen_faces, Mat &face_base_mean,\
				double face_base_bias_variance, const Mat &average_face, const Size std_size) {
	Mat feature_id;
	Mat test_im_array = test_img.reshape(0, 1);
	test_im_array -= average_face;
	project_to_eigen_face(eigen_faces, test_im_array, feature_id);
	Mat img_rebuild;
	back_project_img(feature_id, img_rebuild, eigen_faces, average_face, std_size);
	double mse = calc_mse(test_img, img_rebuild);
	double dist_to_face_mean = calc_bias_to_center(feature_id, face_base_mean);
	return mse*dist_to_face_mean / face_base_bias_variance;
}

double calc_bias_to_center(Mat & face_base_mean, Mat & feature_id) {
	return norm(face_base_mean, feature_id, NORM_L2);
}

double calc_mse(Mat &test_img, Mat &img_rebuild) {
	return sum(abs(test_img - img_rebuild))[0]/(test_img.rows * test_img.cols);
}

/*	back project the feature id to image. rebuild the "face" image from the vie of face space.
* --------------------------------------------------------------------------------------------
* @feature_id: [1*N] matrix. N is the number of dimension of the feature.
* @img_rebuild: [1*S] matrix. S=R*C is the number of the pixels per standard image.
* @eigen_faces: [N*S] matrix. N is the number of eigen faces used.
				S=R*C is the number of the pixels per standard image.
* @average_face:[1*S] matrix.
* --------------------------------------------------------------------------------------------
*/
bool back_project_img(Mat & feature_id, Mat & img_rebuild, Mat & eigen_faces,
						const Mat & average_face, const Size std_size) {
	img_rebuild = average_face + feature_id * eigen_faces;
	img_rebuild = img_rebuild.reshape(1, std_size.height);
	return true;
}
/* get the mean of the face base and its variance.
* ------------------------------------------------
* @face_base: [M*N] matrix. M is the number of the train image.
				N is the number of dimension of the feature.
* @mean_face_base: [1*N] matrix. It is the mean of the face base.
* -------------------------------------------------
* return: the variance.
*/
double analize_face_base(Mat &face_base, Mat &mean_face_base) {
	//double denom = 1.0/ (double)(face_base.rows);
	//Mat denom_array(1, face_base.cols, face_base.type(), Scalar(denom));
	//mean_face_base = 
	//mean_face_base = denom_array*face_base;
	reduce(face_base, mean_face_base, 0, REDUCE_AVG);
	int i = 0;
	double variance = 0;
	for (i = 0; i < face_base.rows; ++i){
		variance += norm(face_base.row(i) - mean_face_base, NORM_L2);
	}
	variance /= face_base.rows;
	return variance;
}

/*find same faces from the test image.
* --------------------------------------
* @test_image: image to be found faces.
* @min_scale: the minimum scale factor the original image be resized to.
* @max_scale: the maximum scale factor the original image be resized to.
* @scale_step: the step of the iteration of the scale factor.
* @eigen_faces: [N*S] matrix. N is the number of eigen faces used.
				S=R*C is the number of the pixels per standard image.
* @std_im_size: standard size of the face image.
* @face_base: [M*N] matrix. M is the number of the train image.
				N is the number of dimension of the feature.
* @number_faces: number of the faces hidden in the test image.
* @result_save_path: the path to save the result.
* @average_face: [1*S] matrix. S=R*C is the number of the pixels per standard image.
* ----------------------------------------------------------------------------------
*/
bool findFace(Mat& test_image, double min_scale, double max_scale, \
				double scale_step, Mat &eigen_faces, Size std_im_size, \
				Mat &face_base, int number_faces, string result_save_path, Mat & average_face) {
	list<face_location> face_found(8);
	double error_thres = std::numeric_limits<double>::infinity();
	double tmp_error;
	face_location new_face_location;
	Rect img_block;
	int x = 0;
	int y = 0;
	Mat img_part;
	Mat img_ROI;
	Mat error_mat = Mat::zeros(test_image.size(), test_image.type());
	// get the mean and variance of the face base.
	Mat mean_face_base;
	double variance = analize_face_base(face_base, mean_face_base);
	for (x = 0; x + std_im_size.width < test_image.cols; ++x){
		for (y = 0; y + std_im_size.height < test_image.rows; ++y) {
			img_block.x = x;
			img_block.y = y;
			img_block.width = std_im_size.width;
			img_block.height = std_im_size.height;
			img_ROI = test_image(img_block);
			img_ROI.copyTo(img_part);
			tmp_error = is_face(img_part, face_base, eigen_faces, mean_face_base, variance, average_face, std_im_size);
			error_mat.at<double>(y, x) = tmp_error;
			if (tmp_error < error_thres){
				new_face_location.face_rect.x = x;
				new_face_location.face_rect.y = y;
				new_face_location.face_rect.width = std_im_size.width;
				new_face_location.face_rect.height = std_im_size.height;
				new_face_location.error = tmp_error;
				error_thres = update_face_location(new_face_location,face_found);
			}
		}
	}
	draw_rect_to_face(test_image, face_found);
	imwrite(result_save_path, test_image);
	imwrite("./error_image.jpg", error_mat);
	return true;
}

double update_face_location(face_location & new_face_location, list<face_location>& face_found) {
	if (check_overlap(face_found, new_face_location)) {
		// redundant face.
		return face_found.back().error;
	}
	auto it = face_found.begin();
	for (; it != face_found.end(); ++it){
		if (it->error > new_face_location.error) {
			face_found.insert(it, new_face_location);
			break;
		}
	}
	face_found.pop_back();
	// return the error of the last element. It is the maximum error in current "faces".
	return face_found.back().error;
}

bool check_overlap(list<face_location>& face_found, face_location & new_face_location) {
	auto it = face_found.begin();
	int old_lenth = face_found.size();
	while (it != face_found.end()) {
		if ((it->face_rect &new_face_location.face_rect).area() != 0){
			if (it->error > new_face_location.error) {
				it = face_found.erase(it);
				continue;
			}
			else{
				// the new face is redundant.
				return true;
			}
		}
		++it;
	}
	int i = 0;
	int new_lenth = face_found.size();
	for (; i < (old_lenth - new_lenth); ++i) {
		face_found.push_back(face_location());
	}
	return false;
}

bool draw_rect_to_face(Mat &test_im, list<face_location>& face_found){
	auto it = face_found.cbegin();
	for (; it != face_found.cend(); ++it) {
		rectangle(test_im, it->face_rect, Scalar(255, 0, 0));
	}
	return true;
}
