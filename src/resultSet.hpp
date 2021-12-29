/*** 
 * @Author: JoeyforJoy
 * @Date: 2021-05-06 21:31:36
 * @LastEditTime: 2021-10-22 23:31:53
 * @LastEditors: JoeyforJoy
 * @Description: Result containers.
 */
#ifndef RESULT_SET_H
#define RESULT_SET_H
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>

namespace NNSearch {
    /*** 
     * @description: point's index and distance
     */
    struct IdxDist {
        IdxDist(int idx_, double dist_): idx(idx_), dist(dist_) {}
        int idx;
        double dist;

        bool operator<(const IdxDist & rhs) const {
            return dist < rhs.dist;
        }
    };

    /*** 
     * @description: Abstract result set
     */
    class AbstractResultSet {
        public:
            /*** 
             * @description: return the worst dist oh current points
             * @return: current worst dist of the points in the set
             */
            virtual double worst_dist() const = 0;
            
            /*** 
             * @description: if the point meet the conditions
             * @param {int} point's index
             * @param {double} point's distance
             */
            virtual void addOnePoint(int idx, double dist) = 0;
            
            /*** 
             * @description: 
             * @param {vector<int>} output indices
             * @param {std::vector<double>} output distances
             * @return {*}
             */
            virtual void unpackResultSet(std::vector<int> &pts_idx, std::vector<double> &pts_dist) = 0;

            virtual ~AbstractResultSet() {}
    };

    /*** 
     * @description: Store the result of KNN search.
     */
    class KNNResultSet: public AbstractResultSet {
        public:
            KNNResultSet(int k):capacity_(k) {}
            ~KNNResultSet() {}
            double worst_dist() const final {
                if (max_heap.empty())
                    return std::numeric_limits<double>::max();
                else
                    return max_heap.top().dist;
            }

            void addOnePoint(int idx, double dist) final {
                if (max_heap.size() < capacity_) {
                    max_heap.push(IdxDist(idx, dist));
                } else {
                    if (max_heap.top().dist > dist) {
                        max_heap.pop();
                        max_heap.push(IdxDist(idx, dist));
                    }
                }
            }

            void unpackResultSet(std::vector<int> &pts_idx, std::vector<double> &pts_dist) final {
                pts_idx.clear();
                pts_idx.reserve(max_heap.size());
                pts_dist.clear();
                pts_dist.reserve(max_heap.size());
                while (!max_heap.empty()) {
                    IdxDist idx_dist = max_heap.top();
                    pts_idx.push_back(idx_dist.idx);
                    pts_dist.push_back(idx_dist.dist);
                    max_heap.pop();
                }
            }

            inline int get_capacity() const { return capacity_;}

            inline bool isfull() const { return max_heap.size() >= capacity_;}

            inline bool empty() const { return max_heap.empty();}

            inline IdxDist top() const { return max_heap.top(); }

            inline void pop() { max_heap.pop();}
            
        private:
            int capacity_;
            // use max heap to maintain the k nearest points
            std::priority_queue<IdxDist> max_heap;
    };

    /*** 
     * @description: Store the result of radius search.
     */
    class RadiusResultSet: public AbstractResultSet {
        public:
            RadiusResultSet(double r):radius(r) {}

            double worst_dist() const final {
                return radius;
            }

            void addOnePoint(int idx, double dist) final {
                if (dist < radius) {
                    result_points.push_back(IdxDist(idx, dist));
                }
            }

            void unpackResultSet(std::vector<int> &pts_idx, std::vector<double> &pts_dist) final {
                sort(result_points.begin(), result_points.end(), 
                        [] (const IdxDist &lhs, const IdxDist &rhs) {
                            return lhs.dist < rhs.dist;
                        });
                pts_idx.clear();
                pts_idx.reserve(result_points.size());
                pts_dist.clear();
                pts_dist.reserve(result_points.size());
                for (const auto &idx_dist: result_points) {
                    pts_idx.push_back(idx_dist.idx);
                    pts_dist.push_back(idx_dist.dist);
                }
            }

        private:
            std::vector<IdxDist> result_points;
            double radius;
    };
}
#endif