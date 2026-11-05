/*!
 * \file tl/rocm/op/atomic_add.cc
 * \brief ROCm implementation for tl.atomic_add lowering.
 */

#include "op/atomic_add.h"
#include "backend/common/op/atomic_reduce.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>

#include "backend/common/target_utils.h"
#include "layout/layout.h"
#include "op/utils.h"
#include "rocm/op/builtin.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;

namespace rocm {

namespace {

bool UseTMA(const AtomicAddNode &op) {
  if (auto value = op.annotations.Get("use_tma");
      value && GetBoolAnnotation(*value)) {
    ICHECK(!op.src_value.has_value())
        << "TMA is not supported when using TiledAtomicAdd with PrimExpr as "
           "value.";
    return true;
  }
  return false;
}

} // namespace

struct AtomicAdd {
  static Stmt LowerSIMT(const AtomicAddNode &op, const LowerArgs &lower_args,
                        sym::AnalyzerObj *analyzer) {
    AtomicAddNode scalar_op = op;
    scalar_op.annotations.erase("use_tma");
    return backend::AtomicReduce::Lower(scalar_op, lower_args, analyzer);
  }

  static LayoutMap InferLayout(const AtomicAddNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    ICHECK(!UseTMA(op))
        << "TMA atomic_add is only supported by the CUDA backend";
    return backend::AtomicReduce::InferLayout(op, layout_args, level);
  }

  static Stmt Lower(const AtomicAddNode &op, const LowerArgs &lower_args,
                    sym::AnalyzerObj *analyzer) {
    ICHECK(!UseTMA(op))
        << "TMA atomic_add is only supported by the CUDA backend";
    return LowerSIMT(op, lower_args, analyzer);
  }
};

} // namespace rocm

namespace {

bool MatchROCmAtomicAddTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmAtomicAdd() {
  RegisterAtomicAddImpl(AtomicAddImpl{
      "rocm.AtomicAdd",
      MatchROCmAtomicAddTarget,
      rocm::AtomicAdd::InferLayout,
      rocm::AtomicAdd::Lower,
  });
  return true;
}

const bool rocm_atomic_add_registered = RegisterROCmAtomicAdd();

} // namespace

} // namespace tl
} // namespace tvm
