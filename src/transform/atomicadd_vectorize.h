/*!
 * \file atomicadd_vectorize.h
 * \brief A tool to automatically vectorize a for atomicadd
 */

#ifndef TVM_TL_ATOMICADD_VECTORIZE_H_
#define TVM_TL_ATOMICADD_VECTORIZE_H_

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/builtin.h"
#include "arith/int_operator.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_vectorization_utils.h"
#include <numeric>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <utility>

namespace tvm {
namespace tl {

using namespace tir;

For VectorizeAtomicAdd(const For &for_node, int compute_capability);

struct AtomicAddVectorizePlanResult {
  int vector_size;
  bool dynamic;
  PrimExpr condition;
};

class AtomicAddVectorizePlanner : public arith::IRVisitorWithAnalyzer {
public:
  AtomicAddVectorizePlanner();

  AtomicAddVectorizePlanResult Plan(const For &node, int compute_capability);

private:
  void VisitStmt_(const ForNode *node) final;
  void VisitExpr_(const CallNode *node) final;

  int GetVectorizeSizeMax(int compute_capability, DataType dtype);
  void UpdateVectorSize(const Array<PrimExpr> &indices, const Buffer &buffer);

  const ForNode *inner_for_ = nullptr;
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 4;
  int max_vector_size = 1;
  bool dynamic_ = false;
  PrimExpr condition_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ATOMICADD_VECTORIZE_H_
