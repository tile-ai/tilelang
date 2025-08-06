#ifndef TVM_TL_TRANSFORM_COMMON_UNION_FIND_H_
#define TVM_TL_TRANSFORM_COMMON_UNION_FIND_H_

#include <unordered_map>
#include <vector>

namespace tvm {
namespace tl {

template <typename T> class UnionFind {
public:
  void MakeSet(const T &x) {
    if (parent_.find(x) == parent_.end()) {
      parent_[x] = x;
      rank_[x] = 0;
    }
  }

  T Find(const T &x) {
    if (parent_[x] != x) {
      parent_[x] = Find(parent_[x]); // Path compression
    }
    return parent_[x];
  }

  void Union(const T &x, const T &y) {
    T x_root = Find(x);
    T y_root = Find(y);

    if (x_root == y_root)
      return;

    // Union by rank
    if (rank_[x_root] < rank_[y_root]) {
      parent_[x_root] = y_root;
    } else if (rank_[x_root] > rank_[y_root]) {
      parent_[y_root] = x_root;
    } else {
      parent_[y_root] = x_root;
      rank_[x_root]++;
    }
  }

private:
  std::unordered_map<T, T> parent_;
  std::unordered_map<T, int> rank_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_UNION_FIND_H_
