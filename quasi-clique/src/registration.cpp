#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <direct.h>
#include <iostream>
#include <string>
#include <algorithm>
//#include <io.h>
#include <omp.h>
#include "../include/Eva.h"
#include <stdarg.h>
#include <chrono>
#include "../include/Util.h"
#include "../include/args.hxx"
#include "../include/FastQC.h"
#include "../include/Corepruning.h"
#include "../include/global_counter.h"
//#include <windows.h>
//#include <io.h>
using namespace Eigen;
using namespace std;
// igraph 0.9.9
extern bool add_overlap;
extern bool low_inlieratio;
extern bool no_logs;
int global_counter = 0;
std::mutex mtx;
static int execution_count  = 0;

void savePointClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_des, int execution_count);
void savePointCloudssuccess(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_des, int execution_count);
void saveClique(const std::vector<Corre_3DMatch> &selected, const std::vector<Corre_3DMatch> &all_selected, const Eigen::Matrix4d &transformation, int execution_count);
void saveCliquesuccess(const std::vector<Corre_3DMatch> &selected, const std::vector<Corre_3DMatch> &all_selected,  const Eigen::Matrix4d &transformation, int execution_count);
void savetransformation(const Eigen::Matrix4d &transformation,const string &gt_mat);
void savetransformationsuccess(const Eigen::Matrix4d &transformation,const string &gt_mat);
bool CopyFile(const std::string& srcPath, const std::string& destPath);


double calculateInlierRatio(const vector<Corre_3DMatch>& correspondences, const Eigen::Matrix4d& GTmat, double threshold) {
    int inliers = 0;
    for (const auto& match : correspondences) {
        Eigen::Vector4d src_point(match.src.x, match.src.y, match.src.z, 1.0);
        Eigen::Vector4d transformed_src_point = GTmat * src_point;
        double distance = std::sqrt(std::pow(transformed_src_point.x() - match.des.x, 2) +
                                    std::pow(transformed_src_point.y() - match.des.y, 2) +
                                    std::pow(transformed_src_point.z() - match.des.z, 2));
        if (distance < threshold) {
            inliers++;
        }
    }
    return static_cast<double>(inliers) / correspondences.size();
}

Eigen::MatrixXd load_csv(const std::string &path){
	std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    int rows = 0;

    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(values.data(), rows, values.size() / rows);
}
std::string getFileName(const std::string& prefix) {
    std::stringstream ss;
    ss << prefix << global_counter;
    return ss.str();
}
std::string getFileName(const std::string& base_path, const std::string& suffix) {
    std::lock_guard<std::mutex> lock(mtx);
    std::stringstream ss;
    ss << base_path << "file_" << global_counter << suffix;
    return ss.str();
}
void calculate_gt_overlap(vector<Corre_3DMatch>&corre, PointCloudPtr &src, PointCloudPtr &tgt, Eigen::Matrix4d &GTmat,  bool ind, double GT_thresh, double &max_corr_weight){
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src, *src_trans, GTmat);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src_trans, kdtree_des;
    kdtree_src_trans.setInputCloud(src_trans);
    kdtree_des.setInputCloud(tgt);
    vector<int>src_ind(1), des_ind(1);
    vector<float>src_dis(1), des_dis(1);
    PointCloudPtr src_corr(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr src_corr_trans(new pcl::PointCloud<pcl::PointXYZ>);
    if(!ind){
        for(auto & i :corre){
            src_corr->points.push_back(i.src);
        }
        pcl::transformPointCloud(*src_corr, *src_corr_trans, GTmat);
        src_corr.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }
    for(int i  = 0; i < corre.size(); i++){
        pcl::PointXYZ src_query, des_query;
        if(!ind){
            src_query = src_corr_trans->points[i];
            des_query = corre[i].des;
        }
        else{
            src_query = src->points[corre[i].src_index];
            des_query = tgt->points[corre[i].des_index];
        }
        kdtree_des.nearestKSearch(src_query, 1, des_ind, src_dis);
        kdtree_src_trans.nearestKSearch(des_query, 1, src_ind, des_dis);
        int src_ov_score = src_dis[0] > pow(GT_thresh,2) ? 0 : 1; //square dist  <= GT_thresh
        int des_ov_score = des_dis[0] > pow(GT_thresh,2) ? 0 : 1;
        if(src_ov_score && des_ov_score){
            corre[i].score = 1;
            max_corr_weight = 1;
        }
        else{
            corre[i].score = 0;
        }
    }
    src_corr_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

bool registration(const string &name, string src_pointcloud, string des_pointcloud,const string &corr_path, const string &label_path, const string &ov_label, const string &gt_mat, const string &folderPath, double& RE, double& TE, double& inlier_num, double& total_num, double& inlier_ratio, double& success_num, double& total_estimate, const string &descriptor, vector<double>& time_consumption, double& IR, double& FMR) {
	bool sc2 = true;
	bool Corr_select = false;
	bool GT_cmp_mode = false;
	int max_est_num = INT_MAX;
	bool ransc_original = false;
    bool instance_equal = true;
	// string metric = "MAE";
	string metric = "Chamfer";
	execution_count++;

	success_num = 0;
	if (!no_logs && access(folderPath.c_str(), 0))
	{
		if (mkdir(folderPath.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
			cout << " 创建数据项目录失败 " << endl;
			exit(-1);
		}
	}
	cout << folderPath << endl;
	string dataPath = corr_path.substr(0, corr_path.rfind("/"));
	string item_name = folderPath.substr(folderPath.rfind("/") + 1, folderPath.length());
	string ref_points_c_path, src_points_c_path, ref_feats_c_path, src_feats_c_path, gt_node_corr_indices_path, node_corr_indices_path;
	string ref_points_f_path, src_points_f_path, ref_feats_f_path, src_feats_f_path, corr_indices_path;

	size_t at_index = corr_path.find("@");
	if (at_index != string::npos)
	{
//		ref_points_c_path = corr_path.substr(0, at_index) + "@ref_points_c.csv";
//		src_points_c_path = corr_path.substr(0, at_index) + "@src_points_c.csv";
//		ref_feats_c_path = corr_path.substr(0, at_index) + "@ref_feats_c.csv";
//		src_feats_c_path = corr_path.substr(0, at_index) + "@src_feats_c.csv";
//      gt_node_corr_indices_path = corr_path.substr(0, at_index) + "@gt_node_corr_indices.csv";
//		node_corr_indices_path = corr_path.substr(0, at_index) + "@merged_node_corr_indices.csv";
		ref_points_f_path = corr_path.substr(0, at_index) + "@ref_points_f.csv";
		src_points_f_path = corr_path.substr(0, at_index) + "@src_points_f.csv";
		ref_feats_f_path = corr_path.substr(0, at_index) + "@ref_feats_f.csv";
		src_feats_f_path = corr_path.substr(0, at_index) + "@src_feats_f.csv";
		corr_indices_path = corr_path.substr(0, at_index) + "@corr_indices_geotransformer.txt";
	}

//	FILE* ref_points_c_file = fopen(ref_points_c_path.c_str(), "r");
//	FILE* src_points_c_file = fopen(src_points_c_path.c_str(), "r");
//	FILE* ref_feats_c_file = fopen(ref_feats_c_path.c_str(), "r");
//	FILE* src_feats_c_file = fopen(src_feats_c_path.c_str(), "r");
//	FILE* gt_node_corr_indices_file = fopen(gt_node_corr_indices_path.c_str(), "r");
//	FILE* node_corr_indices_file = fopen(node_corr_indices_path.c_str(), "r");
	FILE* ref_points_f_file = fopen(ref_points_f_path.c_str(), "r");
	FILE* src_points_f_file = fopen(src_points_f_path.c_str(), "r");
	FILE* ref_feats_f_file = fopen(ref_feats_f_path.c_str(), "r");
	FILE* src_feats_f_file = fopen(src_feats_f_path.c_str(), "r");
	FILE* corr_indices_file = fopen(corr_indices_path.c_str(), "r");
	if (!ref_points_f_file || !src_points_f_file || !ref_feats_f_file || !src_feats_f_file || !corr_indices_file)
	{
		std::cerr << "Failed to open one or more files." << std::endl;
		exit(-1);
	}

	// 验证文件是否成功打开
//	if (!ref_points_c_file || !src_points_c_file || !ref_feats_c_file || !src_feats_c_file || !gt_node_corr_indices_file || !node_corr_indices_file)
//	{
//                std::cerr << "Failed to open one or more files." << std::endl;
//		exit(-1);
//	}
	// 用于存储特征向量的容器
	// std::vector<std::vector<float>> ref_feats_c;
	// std::vector<std::vector<float>> src_feats_c;
//	std::vector<Eigen::VectorXd> ref_feats_c;
//	std::vector<Eigen::VectorXd> src_feats_c;
//	std::vector<pcl::PointXYZ> ref_points_c;
//	std::vector<pcl::PointXYZ> src_points_c;
	std::vector<Eigen::VectorXd> ref_feats_f;
	std::vector<Eigen::VectorXd> src_feats_f;
	std::vector<pcl::PointXYZ> ref_points_f; 
	std::vector<pcl::PointXYZ> src_points_f;
	// // 读取坐标文件
	// while (!feof(ref_points_c_file))
	// {
	// 	float x, y, z;
	// 	if (fscanf(ref_points_c_file, "%f,%f,%f\n", &x, &y, &z) == 3)
	// 	{
	// 		pcl::PointXYZ pt(x, y, z);
	// 		ref_points_c.push_back(pt);
	// 	}
	// }
	// while (!feof(src_points_c_file))
	// {
	// 	float x, y, z;
	// 	if (fscanf(src_points_c_file, "%f,%f,%f\n", &x, &y, &z) == 3)
	// 	{
	// 		pcl::PointXYZ pt(x, y, z);
	// 		src_points_c.push_back(pt);
	// 	}
	// }
	while (!feof(ref_points_f_file))
	{
		float x, y, z;
		if (fscanf(ref_points_f_file, "%f,%f,%f\n", &x, &y, &z) == 3)
		{
			pcl::PointXYZ pt(x, y, z);
			ref_points_f.push_back(pt);
		}
	}
	while (!feof(src_points_f_file))
	{
		float x, y, z;
		if (fscanf(src_points_f_file, "%f,%f,%f\n", &x, &y, &z) == 3)
		{
			pcl::PointXYZ pt(x, y, z);
			src_points_f.push_back(pt);
		}
	}
	// // 读取特征文件
	// while (!feof(ref_feats_c_file))
	// {
	// 	float feature;
	// 	Eigen::VectorXd feat_vec(256); // 特征向量大小为256
	// 	feat_vec.setZero(); // 初始化为0
	// 	int feature_count = 0;
	// 	while (feature_count < 256 && fscanf(ref_feats_c_file, "%f,", &feature) == 1)
	// 	{
	// 		feat_vec(feature_count++) = feature;
	// 	}
	// 	if (feature_count == 256)
	// 	{
	// 		ref_feats_c.push_back(feat_vec);
	// 	}
	// }
	// while (!feof(src_feats_c_file))
	// {
	// 	float feature;
	// 	Eigen::VectorXd feat_vec(256); // 特征向量大小为256
	// 	feat_vec.setZero(); // 初始化为0
	// 	int feature_count = 0;
	// 	while (feature_count < 256 && fscanf(src_feats_c_file, "%f,", &feature) == 1)
	// 	{
	// 		feat_vec(feature_count++) = feature;
	// 	}
	// 	if (feature_count == 256)
	// 	{
	// 		src_feats_c.push_back(feat_vec);
	// 	}
	// }
	while (!feof(ref_feats_f_file))
	{
		float feature;
		Eigen::VectorXd feat_vec(256); // 特征向量大小为256
		feat_vec.setZero(); // 初始化为0
		int feature_count = 0;
		while (feature_count < 256 && fscanf(ref_feats_f_file, "%f,", &feature) == 1)
		{
			feat_vec(feature_count++) = feature;
		}
		if (feature_count == 256)
		{
			ref_feats_f.push_back(feat_vec);
		}
	}
	while (!feof(src_feats_f_file))
	{
		float feature;
		Eigen::VectorXd feat_vec(256); // 特征向量大小为256
		feat_vec.setZero(); // 初始化为0
		int feature_count = 0;
		while (feature_count < 256 && fscanf(src_feats_f_file, "%f,", &feature) == 1)
		{
			feat_vec(feature_count++) = feature;
		}
		if (feature_count == 256)
		{
			src_feats_f.push_back(feat_vec);
		}
	}
	// //首先读取超点pair序号

	// std::vector<std::pair<int, int>> gt_node_corr_indices;

	// while (!feof(gt_node_corr_indices_file))
	// {
	// 	double ref_value, src_value;
	// 	if (fscanf(gt_node_corr_indices_file, "%lf,%lf\n", &ref_value, &src_value) == 2)
	// 	{
	// 		int ref_index = static_cast<int>(ref_value);
	// 		int src_index = static_cast<int>(src_value);
	// 		gt_node_corr_indices.emplace_back(ref_index, src_index);
	// 	}
	// }
	// //粗匹配关系
	// std::vector<std::pair<int, int>> node_corr_indices;
	// while (!feof(node_corr_indices_file))
	// {
	// 	double ref_value, src_value;
	// 	if (fscanf(node_corr_indices_file, "%lf,%lf\n", &ref_value, &src_value) == 2)
	// 	{
	// 		int ref_index = static_cast<int>(ref_value);
	// 		int src_index = static_cast<int>(src_value);
	// 		node_corr_indices.emplace_back(ref_index, src_index);
	// 	}
	// }
	
	std::vector<std::pair<int, int>> corr_indices;
	int ref_value, src_value;
	while (fscanf(corr_indices_file, "%d %d", &src_value, &ref_value) == 2)
	{
		corr_indices.emplace_back(ref_value, src_value);
	}
	fclose(corr_indices_file);

	FILE* corr, * gt;
	corr = fopen(corr_path.c_str(), "r");
	gt = fopen(label_path.c_str(), "r");
	// if (corr == NULL) {
	// 	std::cout << " error in loading correspondence data. " << std::endl;
    //     cout << corr_path << endl;
	// 	exit(-1);
	// }
	if (gt == NULL) {
		std::cout << " error in loading ground truth label data. " << std::endl;
        cout << label_path << endl;
		exit(-1);
	}

	FILE* ov;
	vector<double>ov_corr_label;
    double max_corr_weight = 0;
	if (add_overlap && ov_label != "NULL")
	{
		ov = fopen(ov_label.c_str(), "r");
		if (ov == NULL) {
			std::cout << " error in loading overlap data. " << std::endl;
			exit(-1);
		}
		while (!feof(ov))
		{
			double value;
			fscanf(ov, "%lf\n", &value);
            if(value > max_corr_weight){
                max_corr_weight = value;
            }
			ov_corr_label.push_back(value);

		}
		fclose(ov);
		cout << "load overlap data finished." << endl;
	}

	//PointCloudPtr Overlap_src(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr Raw_src(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr Raw_des(new pcl::PointCloud<pcl::PointXYZ>);
	float raw_des_resolution = 0;
	float raw_src_resolution = 0;
	//pcl::KdTreeFLANN<pcl::PointXYZ>kdtree_Overlap_des, kdtree_Overlap_src;

	PointCloudPtr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr cloud_des(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr normal_src(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr normal_des(new pcl::PointCloud<pcl::Normal>);
	vector<Corre_3DMatch>correspondence;
	vector<int>true_corre;
	inlier_num = 0;
	float resolution = 0;
	bool kitti = false;
    Eigen::Matrix4d GTmat;

    //GTMatRead(gt_mat, GTmat);
    FILE* fp = fopen(gt_mat.c_str(), "r");
    if (fp == NULL)
    {
        printf("Mat File can't open!\n");
        return -1;
    }
    fscanf(fp, "%lf %lf %lf %lf\n", &GTmat(0, 0), &GTmat(0, 1), &GTmat(0, 2), &GTmat(0, 3));
    fscanf(fp, "%lf %lf %lf %lf\n", &GTmat(1, 0), &GTmat(1, 1), &GTmat(1, 2), &GTmat(1, 3));
    fscanf(fp, "%lf %lf %lf %lf\n", &GTmat(2, 0), &GTmat(2, 1), &GTmat(2, 2), &GTmat(2, 3));
    fscanf(fp, "%lf %lf %lf %lf\n", &GTmat(3, 0), &GTmat(3, 1), &GTmat(3, 2), &GTmat(3, 3));
    fclose(fp);
	if (low_inlieratio)
	{
		if (pcl::io::loadPCDFile(src_pointcloud.c_str(), *cloud_src) < 0) {
			std::cout << "Error in loading source pointcloud." << std::endl;
			exit(-1);
		}

		if (pcl::io::loadPCDFile(des_pointcloud.c_str(), *cloud_des) < 0) {
			std::cout << "Error in loading target pointcloud." << std::endl;
			exit(-1);
		}
        while (!feof(corr)) {
            Corre_3DMatch t;
            pcl::PointXYZ src, des;
            fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
            t.src = src;
            t.des = des;
            correspondence.push_back(t);
        }
        if(add_overlap && ov_label == "NULL") { // GT overlap
            cout << "load gt overlap" << endl;
            calculate_gt_overlap(correspondence, cloud_src, cloud_des, GTmat, false, 0.0375, max_corr_weight);
        }
        else if (add_overlap && ov_label != "NULL"){
            for(int i  = 0; i < correspondence.size(); i++){
                correspondence[i].score = ov_corr_label[i];
                if(ov_corr_label[i] > max_corr_weight){
                    max_corr_weight = ov_corr_label[i];
                }
            }
        }
		fclose(corr);
	}
	else {
		if (name == "KITTI")//KITTI
		{
            int idx = 0;
			kitti = true;
			while (!feof(corr))
			{
				Corre_3DMatch t;
				pcl::PointXYZ src, des;
				fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
				t.src = src;
				t.des = des;
                if (add_overlap)
                {
                    t.score = ov_corr_label[idx];
                }
                else
                {
                    t.score = 0;
                }
				correspondence.push_back(t);
                idx++;
			}
			fclose(corr);
		}
		else if (name == "U3M") {
			XYZorPly_Read(src_pointcloud.c_str(), cloud_src);
			XYZorPly_Read(des_pointcloud.c_str(), cloud_des);
			float resolution_src = MeshResolution_mr_compute(cloud_src);
			float resolution_des = MeshResolution_mr_compute(cloud_des);
			resolution = (resolution_des + resolution_src) / 2;
            int idx = 0;
			while (!feof(corr))
			{
				Corre_3DMatch t;
				pcl::PointXYZ src, des;
				fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
				t.src = src;
				t.des = des;
                if (add_overlap)
                {
                    t.score = ov_corr_label[idx];
                }
                else
                {
                    t.score = 0;
                }
				correspondence.push_back(t);
                idx++;
			}
			fclose(corr);
		}
		else if (name == "3dmatch" || name == "3dlomatch") {

			if (!(src_pointcloud == "NULL" && des_pointcloud == "NULL"))
			{
				if (pcl::io::loadPLYFile(src_pointcloud.c_str(), *cloud_src) < 0) {
					std::cout << " error in loading source pointcloud. " << std::endl;
					exit(-1);
				}

				if (pcl::io::loadPLYFile(des_pointcloud.c_str(), *cloud_des) < 0) {
					std::cout << " error in loading target pointcloud. " << std::endl;
					exit(-1);
				}
				float resolution_src = MeshResolution_mr_compute(cloud_src);
				float resolution_des = MeshResolution_mr_compute(cloud_des);
				resolution = (resolution_des + resolution_src) / 2;

				int idx = 0;
				// 根据ground truth建立带有坐标和特征的对应关系
				for (const auto& index_pair : corr_indices)
				{
					size_t ref_index = index_pair.first;
					size_t src_index = index_pair.second;
					// 创建Corre_3DMatch结构体
					Corre_3DMatch t;
					// t.src = src_points_c[src_index];
					// t.des = ref_points_c[ref_index];
					// t.src_features = src_feats_c[src_index];
					// t.des_features = ref_feats_c[ref_index];
					t.src = src_points_f[src_index];
					t.des = ref_points_f[ref_index];
					t.src_features = src_feats_f[src_index];
					t.des_features = ref_feats_f[ref_index];

					// 添加到correspondence容器
					correspondence.push_back(t);
					t.inlier_weight = 0;
					idx++;
				}
				// std::cout << "gt_node_corr_indices size: " << gt_node_corr_indices.size() << std::endl;
				// std::cout << "node_corr_indices size: " << node_corr_indices.size() << std::endl;
				// while (!feof(corr))
				// {
				// 	Corre_3DMatch t;
				// 	pcl::PointXYZ src, des;
				// 	fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
				// 	t.src = src;
				// 	t.des = des;
				// 	//t.src_index = src_ind[0];
				// 	//t.des_index = des_ind[0];
				// 	//t.src_norm = src_vector;
				// 	//t.des_norm = des_vector;
                //     if (add_overlap && ov_label != "NULL")
                //     {
                //         t.score = ov_corr_label[idx];
                //     }
                //     else{
                //         t.score = 0;
                //     }
				// 	t.inlier_weight = 0;
				// 	correspondence.push_back(t);
                //     idx ++;
				// }
				// fclose(corr);
				//src_ind.clear(); des_ind.clear();
				//src_dis.clear(); des_dis.clear();
                if(add_overlap && ov_label == "NULL"){
                    cout << "load gt overlap" << endl;
                    calculate_gt_overlap(correspondence, cloud_src, cloud_des, GTmat, false, 0.0375, max_corr_weight);
                }
			}
			else {
				int idx = 0;
				while (!feof(corr))
				{
					Corre_3DMatch t;
					pcl::PointXYZ src, des;
					fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
					t.src = src;
					t.des = des;
					t.inlier_weight = 0;
					if (add_overlap)
					{
						t.score = ov_corr_label[idx];
					}
					else
					{
						t.score = 0;
					}
					correspondence.push_back(t);
					idx++;
				}
				fclose(corr);
					}
					}
		else {
			exit(-1);
		}
	}
	
	total_num = correspondence.size();
	while (!feof(gt))
	{
		int value;
		fscanf(gt, "%d\n", &value);
		true_corre.push_back(value);
		if (value == 1)
		{
			inlier_num++;
		}
	}
	fclose(gt);

	inlier_ratio = 0;
	if (inlier_num == 0)
	{
		cout << " NO INLIERS！ " << endl;
	}
	inlier_ratio = inlier_num / (total_num / 1.0);

	double RE_thresh, TE_thresh, inlier_thresh;
	if (name == "KITTI")
	{
		RE_thresh = 5;
		TE_thresh = 60;
		inlier_thresh = 0.6;
	}
	else if (name == "3dmatch" || name == "3dlomatch")
	{
		RE_thresh = 15;
		TE_thresh = 30;
		inlier_thresh = 0.1;
	}
	else if (name == "U3M") {
		inlier_thresh = 5 * resolution;
	}
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_time, total_time;

	if (ransc_original)
	{
		Eigen::Matrix4f Mat;
		float RANSAC_inlier_judge_thresh = 0.1;
		float score = 0;
		bool found = false;
		pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
		for (int i = 0; i < correspondence.size(); i++)
		{
			pcl::PointXYZ point_s, point_t;
			point_s = correspondence[i].src;
			point_t = correspondence[i].des;
			source_match_points->points.push_back(point_s);
			target_match_points->points.push_back(point_t);
		}
		//
		total_estimate = max_est_num;
		int Iterations = max_est_num;
		int Match_Idx1, Match_Idx2, Match_Idx3;
		double re, te;

#pragma omp parallel for
		for (int Rand_seed = Iterations; Rand_seed > 0; Rand_seed--)
		{
			Rand_3(Rand_seed, correspondence.size(), Match_Idx1, Match_Idx2, Match_Idx3);
			pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
			point_s1 = correspondence[Match_Idx1].src;
			point_s2 = correspondence[Match_Idx2].src;
			point_s3 = correspondence[Match_Idx3].src;
			point_t1 = correspondence[Match_Idx1].des;
			point_t2 = correspondence[Match_Idx2].des;
			point_t3 = correspondence[Match_Idx3].des;
			//
			Eigen::Matrix4f Mat_iter;
			RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
			float score_iter = Score_est(source_match_points, target_match_points, Mat_iter, RANSAC_inlier_judge_thresh, "inlier");
			//Eigen::MatrixXd Mat_1 = Mat_iter.cast<double>();
			//bool success = evaluation_est(Mat_1, GTmat, 15, 30, re, te);
//#pragma omp critical
//			{
//				success_num = success ? success_num + 1 : success_num;
//				//找到最佳
//				if (success && re < RE && te < TE)
//				{
//					RE = re;
//					TE = te;
//					Mat = Mat_iter;
//					score = score_iter;
//					found = true;
//				}
//			}

#pragma omp critical
			{
				if (score < score_iter)
				{
					score = score_iter;
					Mat = Mat_iter;
				}
			}
		}
		//cout << success_num << " : " << max_est_num << endl;

		Eigen::MatrixXd Mat_1 = Mat.cast<double>();
		found = evaluation_est(Mat_1, GTmat, 15, 30, RE, TE);
		for (size_t i = 0; i < 4; i++)
		{
			time_consumption.push_back(0);
		}

		//保存匹配到txt
		//savetxt(correspondence, folderPath + "/corr.txt");
		//savetxt(selected, folderPath + "/selected.txt");
		string save_est = folderPath + "/est.txt";
		//string save_gt = folderPath + "/GTmat.txt";
		ofstream outfile(save_est, ios::trunc);
		outfile.setf(ios::fixed, ios::floatfield);
		outfile << setprecision(10) << Mat_1;
		outfile.close();
		//CopyFile(gt_mat.c_str(), save_gt.c_str(), false);
		//string save_label = folderPath + "/label.txt";
		//CopyFile(label_path.c_str(), save_label.c_str(), false);

		//保存ply
		//string save_src_cloud = folderPath + "/source.ply";
		//string save_tgt_cloud = folderPath + "/target.ply";
		//CopyFile(src_pointcloud.c_str(), save_src_cloud.c_str(), false);
		//CopyFile(des_pointcloud.c_str(), save_tgt_cloud.c_str(), false);
		cout << "RE=" << RE << " " << "TE=" << TE << endl;
		if (found)
		{
			cout << Mat_1 << endl;
			return true;
		}
		return false;
	}
	std::string base_path = dataPath + "/" + item_name + "@";
	start = std::chrono::system_clock::now();
    // MatrixXd ref_points = load_csv(base_path + "ref_points.csv");
    // MatrixXd src_points = load_csv(base_path + "src_points.csv");
    // MatrixXd ref_points_f = load_csv(base_path + "ref_points_f.csv");
    // MatrixXd src_points_f = load_csv(base_path + "src_points_f.csv");
    // MatrixXd ref_points_c = load_csv(base_path + "ref_points_c.csv");
    // MatrixXd src_points_c = load_csv(base_path + "src_points_c.csv");
    // MatrixXd ref_feats_c = load_csv(base_path + "ref_feats_c.csv");
    // MatrixXd src_feats_c = load_csv(base_path + "src_feats_c.csv");
    // MatrixXd ref_node_corr_indices = load_csv(base_path + "ref_node_corr_indices.csv");
    // MatrixXd src_node_corr_indices = load_csv(base_path + "src_node_corr_indices.csv");
    // MatrixXd ref_corr_points = load_csv(base_path + "ref_corr_points.csv");
    // MatrixXd src_corr_points = load_csv(base_path + "src_corr_points.csv");
    // MatrixXd corr_scores = load_csv(base_path + "corr_scores.csv");
    // MatrixXd gt_node_corr_indices = load_csv(base_path + "gt_node_corr_indices.csv");
    // MatrixXd gt_node_corr_overlaps = load_csv(base_path + "gt_node_corr_overlaps.csv");
    // MatrixXd estimated_transform = load_csv(base_path + "estimated_transform.csv");
    // MatrixXd transform = load_csv(base_path + "transform.csv");
	std::cout << "loading complete." << endl;
	Eigen::MatrixXf Graph = Graph_construction(correspondence, resolution, sc2, name, descriptor, inlier_thresh);
	end = std::chrono::system_clock::now();
	elapsed_time = end - start;
	time_consumption.push_back(elapsed_time.count());
	total_time += elapsed_time;
	cout << " graph construction: " << elapsed_time.count() << endl; 
	if (Graph.norm() == 0) {
        cout << "Graph is disconnected." << endl;
		return false;
	}
	/*MatD sorted_Graph;
	MatrixXi sort_index;
	sort_row(Graph, sorted_Graph, sort_index);*/

	vector<int>degree(total_num, 0);
	vector<Vote_exp> pts_degree;
	for (int i = 0; i < total_num; i++)
	{
		Vote_exp t;
		t.true_num = 0;
		vector<int> corre_index;
		for (int j = 0; j < total_num; j++)
		{
			if (i != j && Graph(i, j)) {
				degree[i] ++;
				corre_index.push_back(j);
				if (true_corre[j])
				{
					t.true_num++;
				}
			}
		}
		t.index = i;
		t.degree = degree[i];
		t.corre_index = corre_index;
		pts_degree.push_back(t);
	}

	//evaluate graph
	vector<Vote> cluster_factor;
	double sum_fenzi = 0;
	double sum_fenmu = 0;
	for (int i = 0; i < total_num; i++) {
		Vote t;
		double wijk = 0;
		int index_size = pts_degree[i].corre_index.size();
		const std::vector<int>& corre_index = pts_degree[i].corre_index;

		for (int j = 0; j < index_size; j++) {
			int a = corre_index[j];
			for (int k = j + 1; k < index_size; k++) {
				int b = corre_index[k];
				if (Graph(a, b)) {
					double graph_ia = Graph(i, a);
					double graph_ib = Graph(i, b);
					double graph_ab = Graph(a, b);
					wijk += std::pow(graph_ia * graph_ib * graph_ab, 1.0 / 3);
				}
			}
		}

		if (degree[i] > 1) {
			double f1 = wijk;
			double f2 = degree[i] * (degree[i] - 1) * 0.5;
			sum_fenzi += f1;
			sum_fenmu += f2;
			double factor = f1 / f2;
			t.index = i;
			t.score = factor;
		} else {
			t.index = i;
			t.score = 0;
		}
		cluster_factor.push_back(t);
	}
	end = std::chrono::system_clock::now();
	elapsed_time = end - start;
	cout << " coefficient computation: " << elapsed_time.count() << endl;
	double average_factor = 0;
	for (size_t i = 0; i < cluster_factor.size(); i++)
	{
		average_factor += cluster_factor[i].score;
	}
	average_factor /= cluster_factor.size();

	double total_factor = sum_fenzi / sum_fenmu;

	vector<Vote_exp> pts_degree_bac;
	vector<Vote>cluster_factor_bac;
	pts_degree_bac.assign(pts_degree.begin(), pts_degree.end());
	cluster_factor_bac.assign(cluster_factor.begin(), cluster_factor.end());

	sort(cluster_factor.begin(), cluster_factor.end(), compare_vote_score);
	sort(pts_degree.begin(), pts_degree.end(), compare_vote_degree);

    if(!no_logs){
        string point_degree = folderPath + "/degree.txt";
        string cluster = folderPath + "/cluster.txt";
        FILE* exp = fopen(point_degree.c_str(), "w");
        for (size_t i = 0; i < total_num; i++)
        {
            fprintf(exp, "%d : %d ", pts_degree[i].index, pts_degree[i].degree);
            if (true_corre[pts_degree[i].index])
            {
                fprintf(exp, "1 ");
            }
            else {
                fprintf(exp, "0 ");
            }
            fprintf(exp, "%d\n", pts_degree[i].true_num);
        }
        fclose(exp);
        exp = fopen(cluster.c_str(), "w");
        for (size_t i = 0; i < total_num; i++)
        {
            fprintf(exp, "%d : %f ", cluster_factor[i].index, cluster_factor[i].score);
            if (true_corre[cluster_factor[i].index])
            {
                fprintf(exp, "1 ");
            }
            else {
                fprintf(exp, "0 ");
            }
            fprintf(exp, "%d\n", pts_degree_bac[cluster_factor[i].index].true_num);
        }
        fclose(exp);
    }

	Eigen::VectorXd cluster_coefficients;
	cluster_coefficients.resize(cluster_factor.size());
	for (size_t i = 0; i < cluster_factor.size(); i++)
	{
		cluster_coefficients[i] = cluster_factor[i].score;
	}

	int cnt = 0;
	double OTSU = 0;
	if (cluster_factor[0].score != 0)
	{
		OTSU = OTSU_thresh(cluster_coefficients); 
	}
	double cluster_threshold = min(OTSU, min(average_factor, total_factor)); 

	cout << cluster_threshold << "->min(" << average_factor << " " << total_factor << " " << OTSU << ")" << endl;
	cout << " inliers: " << inlier_num << "\ttotal num: " << total_num << "\tinlier ratio: " << inlier_ratio << endl;
	
	double weight_thresh = cluster_threshold; 

	if (add_overlap)
	{
        cout << "Max weight: " << max_corr_weight << endl;
        if(max_corr_weight > 0.5){
            weight_thresh = 0.5;
            //internal_selection = true;
        }
        else {
             cout << "internal selection is unused." << endl;
            weight_thresh = 0;
            if(max_corr_weight == 0){
                instance_equal = true;
            }
        }
	}
	else {
		weight_thresh = 0;
	}

	if (!add_overlap || instance_equal)
	{
		for (size_t i = 0; i < total_num; i++)
		{
			correspondence[i].score = cluster_factor_bac[i].score;
		}
	}

		//GTM 筛选
		vector<int>Match_inlier;
		if (Corr_select)
		{
			//GTM_corre_select(100, resolution, cloud_src, cloud_des, correspondence, Match_inlier);
			Geometric_consistency(pts_degree, Match_inlier);
		}
		/*****************************************igraph**************************************************/
		igraph_t g;
		igraph_matrix_t g_mat;
		igraph_vector_t weights;
		igraph_vector_init(&weights, Graph.rows() * (Graph.cols() - 1) / 2);
		igraph_matrix_init(&g_mat, Graph.rows(), Graph.cols());

		if (Corr_select)
		{
			if (cluster_threshold > 3) {
				double f = 10;
				while (1)
				{
					if (f * max(OTSU, total_factor) > cluster_factor[49].score)
					{
						f -= 0.05;
					}
					else {
						break;
					}
				}
				for (int i = 0; i < Graph.rows(); i++)
				{
					if (Match_inlier[i] && cluster_factor_bac[i].score > f * max(OTSU, total_factor))
					{
						for (int j = i + 1; j < Graph.cols(); j++)
						{
							if (Match_inlier[j] && cluster_factor_bac[j].score > f * max(OTSU, total_factor))
							{
								MATRIX(g_mat, i, j) = Graph(i, j);
							}
						}
					}
				}
			}
			else
			{
				for (int i = 0; i < Graph.rows(); i++)
				{
					if (Match_inlier[i])
					{
						for (int j = i + 1; j < Graph.cols(); j++)
						{
							if (Match_inlier[j])
							{
								MATRIX(g_mat, i, j) = Graph(i, j);
							}
						}
					}
				}
			}

		}
		else {
			if (cluster_threshold > 3 && correspondence.size() > 50/*max(OTSU, total_factor) > 0.3*/) //reduce the graph size
			{
				double f = 10;
				while (1)
				{
					if (f * max(OTSU, total_factor) > cluster_factor[49].score)
					{
						f -= 0.05;
					}
					else {
						break;
					}
				}
				for (int i = 0; i < Graph.rows(); i++)
				{
					if (cluster_factor_bac[i].score > f * max(OTSU, total_factor))
					{
						for (int j = i + 1; j < Graph.cols(); j++)
						{
							if (cluster_factor_bac[j].score > f * max(OTSU, total_factor))
							{
								MATRIX(g_mat, i, j) = Graph(i, j);
							}
						}
					}
				}
			}
			else {
				for (int i = 0; i < Graph.rows(); i++)
				{
					for (int j = i + 1; j < Graph.cols(); j++)
					{
						if (Graph(i, j))
						{
							MATRIX(g_mat, i, j) = Graph(i, j);
						}
					}
				}
			}
		}

		igraph_set_attribute_table(&igraph_cattribute_table);
		igraph_weighted_adjacency(&g, &g_mat, IGRAPH_ADJ_UNDIRECTED, 0, 1);
		const char* att = "weight";
		EANV(&g, att, &weights);
		resetIterationCount();
		//find all maximal cliques
		igraph_vector_ptr_t cliques;
		igraph_vector_ptr_init(&cliques, 0);
		start = std::chrono::system_clock::now();
		// igraph_maximal_cliques(&g, &cliques, 3, 0); //3dlomatch 4 3dmatch; 3 Kitti  4
		// int clique_num = igraph_vector_ptr_size(&cliques);

		Util util;
		double gamma_input = 0.70;

		int theta = 3;
		int quiete = 0;
		// int *degree_quasi=NULL;
    	// int **Graph_quasi=NULL;
		// std::vector<int> original_indices;
		std::vector<std::vector<int>> Graph_quasi;
		std::vector<int> degree_quasi;
		int graph_size = util.ReadGraph(g, Graph_quasi, degree_quasi);
		
		CoreLocate Core(Graph_quasi, degree_quasi, graph_size, std::ceil(gamma_input * (theta - 1)));
		int num_of_nodes_lower_than_k = Core.Coredecompose();
		// new_g_order.assign(Core.G_order,Core.G_order+std::size_of(Core.G_order)/4);
		// std::copy(Core.G_order, Core.G_order + Core.getGraphSize(), new_g_order.begin());

		Core.GetMaxcore();
		// int **pG, *pd, pgs;
		std::vector<std::vector<int>> pG;
		std::vector<int> pd;
		int pgs = Core.CorePrune(pG, pd);
		int **pG_new = new int*[pG.size()];
		for (size_t i = 0; i < pG.size(); ++i){
			pG_new[i] = pG[i].data();
		}
		set<int> *setG = new set<int>[pgs];
		for (int i = 0; i < pgs; ++i) {
			for (int j = pd[i] - 1; j >= 0; --j)
				setG[i].insert(pG_new[i][j]);
		}
		// int **pG_new = pG.data();


		int *pd_new = pd.data();
		FastQC miner(pG_new, pd_new, pgs, gamma_input, theta, setG, quiete);
		time_t s1 = clock();
		cliques = miner.DCStrategy(Core.order_G, num_of_nodes_lower_than_k); // 传递原始节点编号映射
		time_t s2 = clock();

		// std::string base_path = "/home/public/fyc/multi-clique-worktree/visualization/";
		// std::string filename = getFileName(base_path, "_aftercliques.txt");
		// ofstream in_aftercliques;
		// in_aftercliques.open(filename);
		// // 输出映射后的结果
		// for (int i = 0; i < igraph_vector_ptr_size(&cliques); ++i) {
		// 	igraph_vector_t *clique = (igraph_vector_t*)VECTOR(cliques)[i];
		// 	in_aftercliques << "Clique " << i << ": ";
		// 	for (int j = 0; j < igraph_vector_size(clique); ++j) {
		// 		in_aftercliques << (int)VECTOR(*clique)[j] << " ";
		// 	}
		// 	in_aftercliques << std::endl;
		// }
		{
			std::lock_guard<std::mutex> lock(mtx);
			global_counter++;
		}
		std::cout << "# of returned QCs: " << miner.res_num << std::endl;
		std::cout << "Running Time: " << (double)(s2 - s1) / CLOCKS_PER_SEC << "s" << std::endl;
		int clique_num = miner.res_num;
		if (((double)(s2 - s1) / CLOCKS_PER_SEC)>10 || clique_num <  100)
		{
			igraph_maximal_cliques(&g, &cliques, 3, 0); //3dlomatch 4 3dmatch; 3 Kitti  4
			clique_num = igraph_vector_ptr_size(&cliques);
		}


		// igraph_largest_cliques(&g, &cliques);
		end = std::chrono::system_clock::now();
		elapsed_time = end - start;
		time_consumption.push_back(elapsed_time.count());
		total_time += elapsed_time;
		//print_and_destroy_cliques(&cliques);
		// int clique_num = igraph_vector_ptr_size(&cliques);
		if (clique_num == 0) {
			cout << " NO CLIQUES! " << endl;
		}
		cout << " clique computation: " << elapsed_time.count() << endl;

		//clear useless data
		igraph_destroy(&g);
		igraph_matrix_destroy(&g_mat);
		igraph_vector_destroy(&weights);

		vector<int>remain;
		start = std::chrono::system_clock::now();
		for (int i = 0; i < clique_num; i++)
		{
			remain.push_back(i);
		}
		node_cliques* N_C = new node_cliques[(int)total_num];
		// find_largest_clique_of_node(Graph, &cliques, correspondence, N_C, remain, total_num, max_est_num, descriptor);
		select_more_quasi_cliques(Graph, &cliques, correspondence, N_C, remain, total_num, max_est_num, descriptor);
		end = std::chrono::system_clock::now();
		elapsed_time = end - start;
		time_consumption.push_back(elapsed_time.count());
		total_time += elapsed_time;
		cout << " clique selection: " << elapsed_time.count() << endl;

		PointCloudPtr src_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
		PointCloudPtr des_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
		for (size_t i = 0; i < correspondence.size(); i++)
        {
            src_corr_pts->push_back(correspondence[i].src);
            des_corr_pts->push_back(correspondence[i].des);
		}
		
		/******************************************registraion***************************************************/
		cout << "Number of cliques:" << clique_num << endl;
		cout << "After selection:" << remain.size() << endl;

		stringstream ss;
		ss << "visualization/cliques_info_" << execution_count << ".txt";
		ofstream clique_file(ss.str());
		clique_file << "Number of cliques: " << clique_num << endl;

		// for (int i = 0; i < clique_num; ++i) {
        // igraph_vector_t *clique = (igraph_vector_t*)VECTOR(cliques)[i];
        // if (clique != nullptr) {
        //     if (clique->stor_begin != NULL && clique->end != NULL) {
        //         clique_file << "Clique " << i << ": ";
        //         int j_size = igraph_vector_size(clique);
        //         for (int j = 0; j < j_size; ++j) {
        //             clique_file << VECTOR(*clique)[j] << " ";
        //         }
        //         clique_file << endl;
		// 		}
		// 	}
		// }
		// clique_file.close();
		
		RE = RE_thresh;
		TE = TE_thresh;
		Eigen::Matrix4d best_est;
		bool found = false;
		double best_score = 0;
		vector<Corre_3DMatch>selected;
		vector<int>corre_index;

		std::vector<Corre_3DMatch> all_selected;
		std::vector<int> all_selected_index;

		start = std::chrono::system_clock::now();
		total_estimate = remain.size();
		#pragma omp parallel for
		for (int i = 0; i < remain.size(); i++) {
			vector<Corre_3DMatch> Group;
			vector<int> selected_index;
			igraph_vector_t* v = (igraph_vector_t*)VECTOR(cliques)[remain[i]];
			int group_size = igraph_vector_size(v);

			for (int j = 0; j < group_size; j++) {
				Corre_3DMatch C = correspondence[VECTOR(*v)[j]];
				Group.push_back(C);
				selected_index.push_back(VECTOR(*v)[j]);
			}

			Eigen::Matrix4d est_trans;
			double score = evaluation_trans(Group, correspondence, src_corr_pts, des_corr_pts, weight_thresh, est_trans, inlier_thresh, metric, raw_des_resolution, instance_equal);
			double ir, fmr;
			if (GT_cmp_mode) {
				if (score > 0) {
					double re, te;
					bool success = evaluation_est(est_trans, GTmat, 15, 30, re, te);

					// 计算 Inlier Ratio
					double ir = calculateInlierRatio(Group, GTmat, 0.1);  // 0.1m 阈值
					// 假设在 GT_cmp_mode 计算单对 FMR，传入 5% 作为阈值
					double fmr = ir > 0.05 ? 1.0 : 0.0; // 单对的 FMR (符合条件记为1.0)

		#pragma omp critical
					{
						success_num = success ? success_num + 1 : success_num;
						if (success && re < RE && te < TE) {
							RE = re;
							TE = te;
							best_est = est_trans;
							best_score = score;
							selected = Group;
							corre_index = selected_index;
							found = true;
							IR = ir;
							FMR = fmr;
						}

						all_selected.insert(all_selected.end(), Group.begin(), Group.end());
						all_selected_index.insert(all_selected_index.end(), selected_index.begin(), selected_index.end());
					}
				}
			} else {
				if (true) {
		#pragma omp critical
					{
						if (best_score < score) {
							best_score = score;
							best_est = est_trans;
							selected = Group;
							corre_index = selected_index;
						}
						 // 将当前 Group 和 selected_index 添加到 all_selected 中
						all_selected.insert(all_selected.end(), Group.begin(), Group.end());
						all_selected_index.insert(all_selected_index.end(), selected_index.begin(), selected_index.end());

					}
				}
			}

			Group.clear();
			Group.shrink_to_fit();
			selected_index.clear();
			selected_index.shrink_to_fit();
		}
		end = std::chrono::system_clock::now();
		elapsed_time = end - start;
		time_consumption.push_back(elapsed_time.count());
		total_time += elapsed_time;
		cout << " hypothesis generation & evaluation: " << elapsed_time.count() << endl;
		igraph_vector_ptr_destroy(&cliques);
		cout << success_num << " : " << total_estimate << " : " << clique_num << endl;
		Eigen::MatrixXd tmp_best;
		bool vis_or_not = false;
		if (name == "U3M")
		{
			RE = RMSE_compute(cloud_src, cloud_des, best_est, GTmat, resolution);
			TE = 0;
		}
		else {
			if (!found)
			{
				found = evaluation_est(best_est, GTmat, RE_thresh, TE_thresh, RE, TE);
				IR = calculateInlierRatio(selected, GTmat, 0.1);
				FMR = IR > 0.05 ? 1.0 : 0.0;
			}
			tmp_best = best_est;
			post_refinement(correspondence, src_corr_pts, des_corr_pts, best_est, best_score, inlier_thresh, 20, "MAE");
		}

		cout << selected.size() << " " << best_score << endl;
	
	// 	if (GT_cmp_mode)
	// 		{
	// 			if (score > 0)
	// 			{
	// 				double re, te;
	// 				bool success = evaluation_est(est_trans, GTmat, 15, 30, re, te);
	// #pragma omp critical
	// 				{
	// 					success_num = success ? success_num + 1 : success_num;
	// 					if (success && re < RE && te < TE)
	// 					{
	// 						RE = re;
	// 						TE = te;
	// 						best_est = est_trans;
	// 						best_score = score;
	// 						selected = Group;
	// 						corre_index = selected_index;
	// 						found = true;
	// 					}
	// 				}
	// 			}
	// 		}
	// 		else {
	// 			if (score > 0)
	// 			{
	// #pragma omp critical
	// 				{
	// 					if (best_score < score)
	// 					{
	// 						best_score = score;
	// 						best_est = est_trans;
	// 						selected = Group;
	// 						corre_index = selected_index;
	// 					}
	// 				}
	// 			}
	// 		}
	// 		Group.clear();
	// 		Group.shrink_to_fit();
	// 		selected_index.clear();
	// 		selected_index.shrink_to_fit();
	// 	}

		for (int i = 0; i < selected.size(); i++)
		{
			cout << selected[i].score << " ";
		}
		cout << endl;

        // if(!no_logs){
        //     //保存匹配到txt
        //     //savetxt(correspondence, folderPath + "/corr.txt");
        //     //savetxt(selected, folderPath + "/selected.txt");
        //     string save_est = folderPath + "/est.txt";
        //     //string save_gt = folderPath + "/GTmat.txt";
        //     ofstream outfile(save_est, ios::trunc);
        //     outfile.setf(ios::fixed, ios::floatfield);
        //     outfile << setprecision(10) << best_est;
        //     outfile.close();
        // }

           if (name == "U3M")
		{
			if (RE <= 5)
			{
				cout << RE << endl;
				cout << best_est << endl;
				if (vis_or_not)
				{
					savePointCloudssuccess(cloud_src, cloud_des, execution_count);
					saveCliquesuccess(selected, all_selected, best_est, execution_count);
					savetransformationsuccess(best_est,gt_mat);
				}
				// 重置部分内容
				correspondence.clear();
				correspondence.shrink_to_fit();
				ov_corr_label.clear();
				ov_corr_label.shrink_to_fit();
				true_corre.clear();
				true_corre.shrink_to_fit();
				degree.clear();
				degree.shrink_to_fit();
				pts_degree.clear();
				pts_degree.shrink_to_fit();
				pts_degree_bac.clear();
				pts_degree_bac.shrink_to_fit();
				cluster_factor.clear();
				cluster_factor.shrink_to_fit();
				cluster_factor_bac.clear();
				cluster_factor_bac.shrink_to_fit();
				delete[] N_C;
				remain.clear();
				remain.shrink_to_fit();
				selected.clear();
				selected.shrink_to_fit();
				corre_index.clear();
				corre_index.shrink_to_fit();
				src_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
				des_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
				cloud_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
				cloud_des.reset(new pcl::PointCloud<pcl::PointXYZ>);
				normal_src.reset(new pcl::PointCloud<pcl::Normal>);
				normal_des.reset(new pcl::PointCloud<pcl::Normal>);
				Raw_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
				Raw_des.reset(new pcl::PointCloud<pcl::PointXYZ>);
				return true;
			}
			else {
				// 保存配准失败的点云
				if(vis_or_not){
					savePointClouds(cloud_src, cloud_des, execution_count);
					// 保存使用到的团
					saveClique(selected,all_selected, best_est, execution_count);
					savetransformation(best_est,gt_mat);
				}
				// 重置部分内容
				correspondence.clear();
				correspondence.shrink_to_fit();
				ov_corr_label.clear();
				ov_corr_label.shrink_to_fit();
				true_corre.clear();
				true_corre.shrink_to_fit();
				degree.clear();
				degree.shrink_to_fit();
				pts_degree.clear();
				pts_degree.shrink_to_fit();
				pts_degree_bac.clear();
				pts_degree_bac.shrink_to_fit();
				cluster_factor.clear();
				cluster_factor.shrink_to_fit();
				cluster_factor_bac.clear();
				cluster_factor_bac.shrink_to_fit();
				delete[] N_C;
				remain.clear();
				remain.shrink_to_fit();
				selected.clear();
				selected.shrink_to_fit();
				corre_index.clear();
				corre_index.shrink_to_fit();
				src_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
				des_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
				cloud_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
				cloud_des.reset(new pcl::PointCloud<pcl::PointXYZ>);
				normal_src.reset(new pcl::PointCloud<pcl::Normal>);
				normal_des.reset(new pcl::PointCloud<pcl::Normal>);
				Raw_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
				Raw_des.reset(new pcl::PointCloud<pcl::PointXYZ>);
				return false;
			}
		}
		else {
			if (found)
			{
				double new_re, new_te;
				evaluation_est(best_est, GTmat, RE_thresh, TE_thresh, new_re, new_te);
				IR = calculateInlierRatio(selected, GTmat, 0.1);
				FMR = IR > 0.05 ? 1.0 : 0.0;
				if (new_re < RE && new_te < TE)
				{
					cout << "est_trans updated!!!" << endl;
					cout << "RE=" << new_re << " " << "TE=" << new_te << endl;
					cout << "IR=" << IR << " " << "FMR=" << FMR << endl;
					cout << best_est << endl;
				}
				else {
					best_est = tmp_best;
					cout << "RE=" << RE << " " << "TE=" << TE << endl;
					cout << "IR=" << IR << " " << "FMR=" << FMR << endl;
					cout << best_est << endl;
				}
				RE = new_re;
				TE = new_te;
				if(vis_or_not){
					savePointCloudssuccess(cloud_src, cloud_des, execution_count);
					saveCliquesuccess(selected, all_selected, best_est, execution_count);
					savetransformationsuccess(best_est,gt_mat);
				}
				// 重置部分内容
				correspondence.clear();
				correspondence.shrink_to_fit();
				ov_corr_label.clear();
				ov_corr_label.shrink_to_fit();
				true_corre.clear();
				true_corre.shrink_to_fit();
				degree.clear();
				degree.shrink_to_fit();
				pts_degree.clear();
				pts_degree.shrink_to_fit();
				pts_degree_bac.clear();
				pts_degree_bac.shrink_to_fit();
				cluster_factor.clear();
				cluster_factor.shrink_to_fit();
				cluster_factor_bac.clear();
				cluster_factor_bac.shrink_to_fit();
				delete[] N_C;
				remain.clear();
				remain.shrink_to_fit();
				selected.clear();
				selected.shrink_to_fit();
				corre_index.clear();
				corre_index.shrink_to_fit();
				src_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
				des_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
				cloud_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
				cloud_des.reset(new pcl::PointCloud<pcl::PointXYZ>);
				normal_src.reset(new pcl::PointCloud<pcl::Normal>);
				normal_des.reset(new pcl::PointCloud<pcl::Normal>);
				Raw_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
				Raw_des.reset(new pcl::PointCloud<pcl::PointXYZ>);
				return true;
			}
			else {
				if(vis_or_not){
					// 保存配准失败的点云
					savePointClouds(cloud_src, cloud_des, execution_count);
					// 保存使用到的团
					saveClique(selected, all_selected, best_est, execution_count);
					savetransformation(best_est,gt_mat);
				}
				double new_re, new_te, new_ir, new_fmr;
				found = evaluation_est(best_est, GTmat, RE_thresh, TE_thresh, new_re, new_te);
				IR = calculateInlierRatio(selected, GTmat, 0.1);
				FMR = IR > 0.05 ? 1.0 : 0.0;
				if (found)
				{
					RE = new_re;
					TE = new_te;
					cout << "est_trans corrected!!!" << endl;
					cout << "RE=" << RE << " " << "TE=" << TE << endl;
					cout << "IR=" << IR << " " << "FMR=" << FMR << endl;
					cout << best_est << endl;

				}

				cout << "RE=" << RE << " " << "TE=" << TE << endl;
				cout << "IR=" << IR << " " << "FMR=" << FMR << endl;

				// 重置部分内容
				correspondence.clear();
				correspondence.shrink_to_fit();
				ov_corr_label.clear();
				ov_corr_label.shrink_to_fit();
				true_corre.clear();
				true_corre.shrink_to_fit();
				degree.clear();
				degree.shrink_to_fit();
				pts_degree.clear();
				pts_degree.shrink_to_fit();
				pts_degree_bac.clear();
				pts_degree_bac.shrink_to_fit();
				cluster_factor.clear();
				cluster_factor.shrink_to_fit();
				cluster_factor_bac.clear();
				cluster_factor_bac.shrink_to_fit();
				delete[] N_C;
				remain.clear();
				remain.shrink_to_fit();
				selected.clear();
				selected.shrink_to_fit();
				corre_index.clear();
				corre_index.shrink_to_fit();
				src_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
				des_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
				cloud_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
				cloud_des.reset(new pcl::PointCloud<pcl::PointXYZ>);
				normal_src.reset(new pcl::PointCloud<pcl::Normal>);
				normal_des.reset(new pcl::PointCloud<pcl::Normal>);
				Raw_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
				Raw_des.reset(new pcl::PointCloud<pcl::PointXYZ>);
				return false;
			}
		}
}



const std::string failed_save_folder = "/home/public/fyc/multi-clique-worktree/failed";
const std::string success_save_folder = "/home/public/fyc/multi-clique-worktree/success";

// 保存点云的函数
void savePointClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_des, int execution_count) {
    std::string pointCloudFolder = failed_save_folder;
    if (access(pointCloudFolder.c_str(), 0) != 0 && mkdir(pointCloudFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
        std::cout << "创建点云保存目录失败" << std::endl;
        exit(-1);
    }

    pcl::io::savePCDFileASCII(pointCloudFolder + "/failed_source_" + std::to_string(execution_count) + ".pcd", *cloud_src);
    pcl::io::savePCDFileASCII(pointCloudFolder + "/failed_target_" + std::to_string(execution_count) + ".pcd", *cloud_des);
    std::cout << "Saved failed registration point clouds to " << pointCloudFolder << std::endl;
}
void savePointCloudssuccess(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_des, int execution_count) {
    std::string pointCloudFolder = success_save_folder;
    if (access(pointCloudFolder.c_str(), 0) != 0 && mkdir(pointCloudFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
        std::cout << "创建点云保存目录失败" << std::endl;
        exit(-1);
    }

    pcl::io::savePCDFileASCII(pointCloudFolder + "/success_source_" + std::to_string(execution_count) + ".pcd", *cloud_src);
    pcl::io::savePCDFileASCII(pointCloudFolder + "/success_target_" + std::to_string(execution_count) + ".pcd", *cloud_des);
    std::cout << "Saved successful registration point clouds to " << pointCloudFolder << std::endl;
}

// 保存团的函数
void saveClique(const std::vector<Corre_3DMatch> &selected, 
               const std::vector<Corre_3DMatch> &all_selected, 
               const Eigen::Matrix4d &transformation, 
               int execution_count) {
    std::string cliqueFolder = failed_save_folder;
    if (access(cliqueFolder.c_str(), 0) != 0 && mkdir(cliqueFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
        std::cout << "创建团保存目录失败" << std::endl;
        exit(-1);
    }

    // 保存 selected
    std::ofstream cliqueFileSelected(cliqueFolder + "/failed_clique_selected_" + std::to_string(execution_count) + ".txt");
    if (!cliqueFileSelected.is_open()) {
        std::cerr << "无法打开文件以保存 selected clique" << std::endl;
        return;
    }
    for (const auto &match : selected) {
        Eigen::Vector4d src_point(match.src.x, match.src.y, match.src.z, 1.0);
        Eigen::Vector4d transformed_src_point = transformation * src_point;
        cliqueFileSelected << transformed_src_point.x() << " " << transformed_src_point.y() << " " << transformed_src_point.z() << " "
                           << match.des.x << " " << match.des.y << " " << match.des.z << std::endl;
    }
    cliqueFileSelected.close();

    // 保存 all_selected
    std::ofstream cliqueFileAll(cliqueFolder + "/failed_clique_all_selected_" + std::to_string(execution_count) + ".txt");
    if (!cliqueFileAll.is_open()) {
        std::cerr << "无法打开文件以保存 all_selected clique" << std::endl;
        return;
    }
    for (const auto &match : all_selected) {
        Eigen::Vector4d src_point(match.src.x, match.src.y, match.src.z, 1.0);
        Eigen::Vector4d transformed_src_point = transformation * src_point;
        cliqueFileAll << transformed_src_point.x() << " " << transformed_src_point.y() << " " << transformed_src_point.z() << " "
                     << match.des.x << " " << match.des.y << " " << match.des.z << std::endl;
    }
    cliqueFileAll.close();

    std::cout << "Saved selected clique and all selected cliques to " << cliqueFolder << std::endl;
}

// 修改后的 saveCliquesuccess 函数
void saveCliquesuccess(const std::vector<Corre_3DMatch> &selected, 
                       const std::vector<Corre_3DMatch> &all_selected, 
                       const Eigen::Matrix4d &transformation, 
                       int execution_count) {
    std::string cliqueFolder = success_save_folder;
    if (access(cliqueFolder.c_str(), 0) != 0 && mkdir(cliqueFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
        std::cout << "创建团保存目录失败" << std::endl;
        exit(-1);
    }

    // 保存 selected
    std::ofstream cliqueFileSelected(cliqueFolder + "/success_clique_selected_" + std::to_string(execution_count) + ".txt");
    if (!cliqueFileSelected.is_open()) {
        std::cerr << "无法打开文件以保存 selected success clique" << std::endl;
        return;
    }
    for (const auto &match : selected) {
        Eigen::Vector4d src_point(match.src.x, match.src.y, match.src.z, 1.0);
        Eigen::Vector4d transformed_src_point = transformation * src_point;
        cliqueFileSelected << transformed_src_point.x() << " " 
                           << transformed_src_point.y() << " " 
                           << transformed_src_point.z() << " "
                           << match.des.x << " " 
                           << match.des.y << " " 
                           << match.des.z << std::endl;
    }
    cliqueFileSelected.close();

    // 保存 all_selected
    std::ofstream cliqueFileAll(cliqueFolder + "/success_clique_all_selected_" + std::to_string(execution_count) + ".txt");
    if (!cliqueFileAll.is_open()) {
        std::cerr << "无法打开文件以保存 all_selected success clique" << std::endl;
        return;
    }
    for (const auto &match : all_selected) {
        Eigen::Vector4d src_point(match.src.x, match.src.y, match.src.z, 1.0);
        Eigen::Vector4d transformed_src_point = transformation * src_point;
        cliqueFileAll << transformed_src_point.x() << " " 
                     << transformed_src_point.y() << " " 
                     << transformed_src_point.z() << " "
                     << match.des.x << " " 
                     << match.des.y << " " 
                     << match.des.z << std::endl;
    }
    cliqueFileAll.close();

    std::cout << "Saved selected success clique and all selected success cliques to " << cliqueFolder << std::endl;
}

void savetransformation(const Eigen::Matrix4d &transformation,const string &gt_mat){
	string save_est =  "/home/public/fyc/multi-clique-worktree/failed/est"+std::to_string(execution_count)+".txt";
	//string save_gt = folderPath + "/GTmat.txt";
	string save_gt =  "/home/public/fyc/multi-clique-worktree/failed/GTmat"+std::to_string(execution_count)+".txt";
	ofstream outfile(save_est, ios::trunc);
	outfile.setf(ios::fixed, ios::floatfield);
	outfile << setprecision(10) << transformation;
	outfile.close();
	CopyFile(gt_mat.c_str(), save_gt.c_str());
}
void savetransformationsuccess(const Eigen::Matrix4d &transformation,const string &gt_mat){
	string save_est =  "/home/public/fyc/multi-clique-worktree/success/est"+std::to_string(execution_count)+".txt";
	//string save_gt = folderPath + "/GTmat.txt";
	string save_gt =  "/home/public/fyc/multi-clique-worktree/success/GTmat"+std::to_string(execution_count)+".txt";
	ofstream outfile(save_est, ios::trunc);
	outfile.setf(ios::fixed, ios::floatfield);
	outfile << setprecision(10) << transformation;
	outfile.close();
	CopyFile(gt_mat.c_str(), save_gt.c_str());
}

bool CopyFile(const std::string& srcPath, const std::string& destPath) {
    std::ifstream srcFile(srcPath, std::ios::binary);
    if (!srcFile.is_open()) {
        std::cerr << "Error opening source file: " << srcPath << std::endl;
        return false;
    }

    std::ofstream destFile(destPath, std::ios::binary);
    if (!destFile.is_open()) {
        std::cerr << "Error opening destination file: " << destPath << std::endl;
        srcFile.close();
        return false;
    }

    // 使用缓冲区进行文件复制
    char buffer[4096]; // 可以调整缓冲区大小以优化性能
    while (srcFile.read(buffer, sizeof(buffer))) {
        destFile.write(buffer, srcFile.gcount());
    }

    // 复制剩余的数据
    destFile.write(buffer, srcFile.gcount());

    srcFile.close();
    destFile.close();

    return true;
}
