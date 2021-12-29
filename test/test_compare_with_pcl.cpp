/*** 
 * @Author: JoeyforJoy
 * @Date: 2021-05-06 20:57:30
 * @LastEditTime: 2021-10-22 23:51:20
 * @LastEditors: JoeyforJoy
 * @Description: 
 */

#include "kdtree.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <pcl/common/common_headers.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

using namespace NNSearch;
using namespace std;
using namespace std::chrono;

Eigen::MatrixXd readPtCloudFromOFF (const std::string &filename) {
    std::ifstream fin(filename);
    if (!fin) std::cerr << "OFF file read failure!\n";
    std::string tmp; getline(fin, tmp);

    int vertexNum = 0;
    fin >> vertexNum;

    // init point cloud
    Eigen::MatrixXd cloud = Eigen::MatrixXd::Zero(vertexNum, 3);
    int tmp_i;
    fin >> tmp_i; fin >> tmp_i;
    for (int i = 0; i < vertexNum; i++) {
        // read point location
        double x, y, z;
        fin >> x; fin >> y; fin >> z;
        
        // add to point cloud
        cloud(i, 0) = x;
        cloud(i, 1) = y;
        cloud(i, 2) = z;
    }
    return cloud;
}

int main (int argc, char **argv) {
    // read point cloud
    // string file_path = "../data/airplane_0001.off";
    string file_path = __FILE__;
    file_path = file_path.substr(0, file_path.find_last_of("/")) + "/data/airplane_0001.off";
    Eigen::MatrixXd cloud_eigen = readPtCloudFromOFF(file_path);
    cout << "Point Read Successfully!\n";
    int iter_num = 100;

    // ====== test my kd-tree ======
    cout << "=== Testing my kd-tree ===\n";
    // build kd-tree
    const int leaf_size = 5;
    auto t1 = chrono::steady_clock::now();
    KDTree kdtree(cloud_eigen, leaf_size); // build kdtree
    cout << "KD Tree builded! Takes " 
        << duration_cast<milliseconds>(chrono::steady_clock::now()-t1).count() << "ms\n";
    // knn search
    Eigen::Vector3d point;
    point << 0, 1 ,2;
    std::vector<int> pts_idx;
    std::vector<double> pts_dist;
    int k = 5;
    t1 = chrono::steady_clock::now();
    for (int i = 0; i < iter_num; i++)
         kdtree.knnSearch(point, k, pts_idx, pts_dist); // knn search
    cout << "knn search finish. Takes "
         << duration_cast<microseconds>(chrono::steady_clock::now()-t1).count() / iter_num << "us\n";
    // radius search
    int radius = 5;
    t1 = chrono::steady_clock::now();
    for (int i = 0; i < iter_num; i++)
        kdtree.radiusSearch(point, radius, pts_idx, pts_dist);
    cout << "radius search finish. Takes "
         << duration_cast<microseconds>(chrono::steady_clock::now()-t1).count() / iter_num << "us\n";

    // ====== test pcl kd-tree ======
    cout << "=== Testing pcl's kd-tree ===\n";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcl(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < cloud_eigen.rows(); i++) {
        pcl::PointXYZ pt(cloud_eigen(i, 0), cloud_eigen(i, 1), cloud_eigen(i, 2));
        cloud_pcl->push_back(pt);
    }
    // build kd-tree
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_pcl;
    t1 = chrono::steady_clock::now();
    kdtree_pcl.setInputCloud(cloud_pcl); // build kdtree using pcl interface
    cout << "KD Tree builded, using pcl interface! Takes " 
         << duration_cast<milliseconds>(chrono::steady_clock::now()-t1).count() << "ms\n";
    // knn search
    std::vector<int> pointIdxNKNSearch(k);
    std::vector<float> pointNKNSquaredDistance(k);
    pcl::PointXYZ searchPoint = pcl::PointXYZ(0, 1, 2);
    t1 = chrono::steady_clock::now();
    for (int i = 0; i < iter_num; i++) 
        kdtree_pcl.nearestKSearch (searchPoint, k, pointIdxNKNSearch, pointNKNSquaredDistance);
    cout << "pcl knn search finish, using pcl interface!. Takes "
         << duration_cast<microseconds>(chrono::steady_clock::now()-t1).count() / iter_num << "us\n";
    // radius search
    t1 = chrono::steady_clock::now();
    for (int i = 0; i < iter_num; i++)
        kdtree_pcl.radiusSearch (searchPoint, radius, 
                              pointIdxNKNSearch, pointNKNSquaredDistance);
    cout << "pcl radius search finish, using pcl interface!. Takes "
         << duration_cast<microseconds>(chrono::steady_clock::now()-t1).count() / iter_num << "us\n";

}