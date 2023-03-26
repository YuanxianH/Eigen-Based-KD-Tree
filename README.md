# KD Tree
A simple and efficient c++ KD-Tree implementation.
This code was written when I was learning c++11. So I deliberately wrote it all in c++ style and used some c++11 characteristics, like smart pointer and many other STL tools. 

If you are interested in following features, you can try this project for fun:
- **KNN Search** and **Radius Search** are surpported and have comparable performance with PCL.
- Simple. Only 2 **header-only** file.
- **Eigen-Based**. Point cloud is organized by Eigen, and can be extended to any dimension.
- **Low Dependency**. All you need is Eigen.
- **Smart Pointer**. Not worry about memory leak when use it. 

## Requirements
- Eigen
- cmake 2.8+
## Quick Start
```bash
mkdir build
cd build
cmake ..
make 
./main # test kdtree
```
## Usage
After include `kdtree.hpp`, you can use it as following.
```c++
Eigen::MatrixXd cloud_eigen; // point cloud
cloud_eigen = ... ; // assign point cloud for cloud_eigen
int leaf_size = 5; // the minimum number of points in the leaf nodes
KDTree kdtree(cloud_eigen, leaf_size); // build kdtree
// knn search
std::vector<std::size_t> pts_idx; // indices of result points
std::vector<double> pts_dist; // distances of result points
int k = 5;
kdtree.knnSearch(point, k, pts_idx, pts_dist);
// radius search
double radius = 5;
kdtree.radiusSearch(point, radius, pts_idx, pts_dist);
```