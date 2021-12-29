/*** 
 * @Author: JoeyforJoy
 * @Date: 2021-05-06 21:55:55
 * @LastEditTime: 2021-10-23 00:13:45
 * @LastEditors: JoeyforJoy
 * @Description: A simple KD Tree implementation
 * @Usage:
 *      Eigen::MatrixXd cloud_eigen; 
 *      cloud_eigen = ... ; // assign point cloud for cloud_eigen
 *      int leaf_size = 5; // the minimum number of points in the leaf nodes
 *      KDTree kdtree(cloud_eigen, leaf_size); // build kdtree
 *      // knn search
 *      std::vector<int> pts_idx; // indices of result points
 *      std::vector<double> pts_dist; // distance of result points
 *      int k = 5;
 *      kdtree.knnSearch(point, k, pts_idx, pts_dist);
 *      // radius search
 *      int radius = 5;
 *      kdtree.radiusSearch(point, radius, pts_idx, pts_dist);
 */

#ifndef KDTREE_HPP
#define KDTREE_HPP
#include <memory>
#include <numeric>
#include <Eigen/Core>

#include "resultSet.hpp"

namespace NNSearch {
    struct KDNode {
        typedef std::shared_ptr<KDNode> Ptr;
        public:
            int dim; // 0 for X, 1 for Y, 2 for X
            double value;
            KDNode::Ptr left;
            KDNode::Ptr right;
            bool isLeaf;
            std::vector<int> points_idx;
        public:
            KDNode(): dim(0), value(0), isLeaf(false), left(nullptr), right(nullptr){}
            KDNode(int dim_): dim(dim_), value(0), left(nullptr), right(nullptr), isLeaf(false) {}
    };

    class KDTree {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        public:
            KDTree(): _root(nullptr), _leaf_size(0) {}
            // copy constructor
            KDTree(const KDTree& tree): _root(tree.getRoot()), _leaf_size(tree._leaf_size) {}
            // copy assign
            KDTree& operator=(const KDTree& tree) {_root = tree.getRoot(); _leaf_size = tree._leaf_size;}
            // build KDTree
            KDTree(Eigen::MatrixXd &points, int leaf_size): _points(points), _leaf_size(leaf_size) {
                std::vector<int> pts_idx(points.rows(), 0);
                std::iota(pts_idx.begin(), pts_idx.end(), 0);
                _root = _buildKDTree(pts_idx, 0, pts_idx.size(), 0);
            }

            ~KDTree() {}

            KDNode::Ptr getRoot() const {return _root;}
            int getLeafSize() const {return _leaf_size;}

            // knn search
            void knnSearch(const Eigen::Vector3d &point, int k, std::vector<int> &pts_idx, std::vector<double> &pts_dist) {
                // KNN search
                KNNResultSet result_set(k);
                _SearchIdxDist(_root, point, result_set);
                // unpack result
                result_set.unpackResultSet(pts_idx, pts_dist);
            }

            // radius search
            void radiusSearch(const Eigen::Vector3d &point, int radius, std::vector<int> &pts_idx, std::vector<double> &pts_dist) {
                // radius search
                RadiusResultSet result_set(radius);
                _SearchIdxDist(_root, point, result_set);
                // unpack result
                result_set.unpackResultSet(pts_idx, pts_dist);
            }

        private:
            void _SearchIdxDist(KDNode::Ptr root, const Eigen::Vector3d &point, AbstractResultSet &result_set) {
                if (!root) return;
                if (root->isLeaf) {
                    // process leaf
                    for (int idx: root->points_idx) {
                        double dist = (_points.row(idx) - point.transpose()).norm();
                        result_set.addOnePoint(idx, dist);
                    }
                } else {
                    if (point(root->dim) <= root->value) { // If the value is less than the root, search the left.
                        _SearchIdxDist(root->left, point, result_set);
                        // If the point is too far, don't search the right tree.
                        if (fabs(point(root->dim) - root->value) <= result_set.worst_dist()) {
                            _SearchIdxDist(root->right, point, result_set);
                        }
                    } else {
                        _SearchIdxDist(root->right, point, result_set);
                        // If the point is too far, don't search the left tree.
                        if (fabs(point(root->dim) - root->value) <= result_set.worst_dist()) {
                            _SearchIdxDist(root->left, point, result_set);
                        }
                    }
                }
            }

            // 构造 KD 树
            KDNode::Ptr _buildKDTree(std::vector<int> &pts_idx, 
                                    int left, int right, int dim) {
                if (right - left <= 0) return nullptr;
                KDNode::Ptr root(new KDNode(dim));
                if (right - left <= _leaf_size ) {
                    root->isLeaf = true;
                    root->points_idx.assign(pts_idx.begin()+left, pts_idx.begin()+right);
                } else {
                    // current splitting axis
                    auto min_max_x = std::minmax_element(
                        pts_idx.begin() + left,
                        pts_idx.begin() + right,
                        [&](int lhs, int rhs) { return _points(lhs, 0) < _points(rhs, 0); });
                    auto min_max_y = std::minmax_element(
                        pts_idx.begin() + left,
                        pts_idx.begin() + right,
                        [&](int lhs, int rhs) { return _points(lhs, 1) < _points(rhs, 1); });
                    auto min_max_z = std::minmax_element(
                        pts_idx.begin() + left,
                        pts_idx.begin() + right,
                        [&](int lhs, int rhs) { return _points(lhs, 2) < _points(rhs, 2); });
                    auto dx = _points((*min_max_x.second),0) - _points((*min_max_x.first),0);
                    auto dy = _points((*min_max_x.second),1) - _points((*min_max_x.first),1);
                    auto dz = _points((*min_max_x.second),2) - _points((*min_max_x.first),2);
                    dim = (dx > dy ? (dx > dz ? 0 : 2) : (dy > dz ? 1 : 2));

                    // find middle o(NlogN)
                    std::nth_element (
                        pts_idx.begin() + left,
                        pts_idx.begin() + (right - left) / 2 + left,
                        pts_idx.begin() + right,
                        [&]
                        (int lhs, int rhs) 
                        { return _points(lhs, dim) < _points(rhs, dim);});

                    // find the middle and split
                    int mid = (right - left) / 2 + left;

                    root->value = (_points(pts_idx[mid], dim) + _points(pts_idx[mid+1], dim) + 1e-3) /2 ;
                    root->left = _buildKDTree(pts_idx, left, mid, dim);
                    root->right = _buildKDTree(pts_idx, mid, right, dim);
                }
                return root;
            }

        private:
            Eigen::MatrixXd _points;
            KDNode::Ptr _root;
            int _leaf_size;
    };
}
#endif