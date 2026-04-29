/*!
 * \file detect_mutable_descriptors.cc
 * \brief Detect TMA descriptors that are mutated via tensormap.replace + cp_fence_release.
 *
 * Runs before LowerHopperIntrin.  Scans the PrimFunc body for
 * tensormap_cp_fence_release calls; each such call's first argument names a
 * descriptor that will be written back to global memory at runtime and must
 * therefore use a per-block GMEM workspace slot instead of the read-only
 * __grid_constant__ parameter.
 *
 * Sets PrimFunc attributes:
 *   "mutable_tma_descriptor_count" (Integer) – total mutable descriptors
 *   "mutable_tma_descriptor_names"  (Array<StringImm>) – Var name_hints
 */

#include <string>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

class MutableDescDetector : public StmtExprVisitor {
public:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tensormap_cp_fence_release()) && op->args.size() >= 1) {
      if (auto var = op->args[0].as<VarNode>()) {
        mutable_desc_names_.insert(var->name_hint);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::vector<std::string> GetSortedNames() const {
    std::vector<std::string> names(mutable_desc_names_.begin(),
                                   mutable_desc_names_.end());
    std::sort(names.begin(), names.end());
    return names;
  }

private:
  std::unordered_set<std::string> mutable_desc_names_;
};

tvm::transform::Pass DetectMutableDescriptors() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    MutableDescDetector detector;
    detector(f->body);
    auto names = detector.GetSortedNames();

    if (!names.empty()) {
      Array<StringImm> name_arr;
      for (const auto &n : names) {
        name_arr.push_back(StringImm(n));
      }
      f = WithAttr(std::move(f), "mutable_tma_descriptor_names", name_arr);
      f = WithAttr(std::move(f), "mutable_tma_descriptor_count",
                   Integer(static_cast<int>(names.size())));
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.DetectMutableDescriptors", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.DetectMutableDescriptors",
                        DetectMutableDescriptors);
}

} // namespace tl
} // namespace tvm
