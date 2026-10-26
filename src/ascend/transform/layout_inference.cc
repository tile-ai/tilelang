/*!
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include "transform/common/loop_aware_mutator.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

#include "ascend/op/utils.h"
#include "ascend/transform/vf_regions.h"
#include "op/gemm_blockscaled.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/layout_inference/layout_inference.h"

namespace tvm::tl::ascend {
namespace {

class ParallelLoopFuserSkipSimdVF : public ParallelLoopFuser {
public:
  explicit ParallelLoopFuserSkipSimdVF(sym::Analyzer analyzer)
      : ParallelLoopFuser(std::move(analyzer)) {}

private:
  bool PreserveParallelLoopNest(const ForNode *node) const final {
    if (!inside_simt_vf_)
      return false;
    bool has_reducer_update = false;
    PostOrderVisit(node->body, [&](const ffi::ObjectRef &object) {
      if (const auto *call = object.as<CallNode>())
        has_reducer_update |= call->op.same_as(reducer_update());
    });
    return has_reducer_update;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    auto name = VFRegionName(node);
    if (name == "SIMD_VF")
      return ffi::Unchanged{};
    bool previous = inside_simt_vf_;
    if (name == "SIMT_VF")
      inside_simt_vf_ = true;
    auto result = LoopAwareMutator::Mutate_(node, mode);
    inside_simt_vf_ = previous;
    return result;
  }

  bool inside_simt_vf_{false};
};

} // namespace

tvm::transform::Pass AscendLayoutInference() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (!function->body)
      return function;
    auto bindings = CollectL0SFBindings(function->body.value());
    auto fuser = ffi::make_object<ParallelLoopFuserSkipSimdVF>(sym::Analyzer());
    auto original = function->body.value();
    function.CopyOnWrite()->body =
        fuser->Mutate(original).ValueOrUnchanged(original);
    LayoutInferenceOptions options;
    options.fuse_parallel_loops = false;
    options.opaque_region = [](const RegionStmtNode *region) {
      return VFRegionName(region) == "SIMD_VF";
    };
    options.validate_operator = [bindings](const TileOperator &operation,
                                           sym::Analyzer analyzer) {
      if (auto gemm = operation.as<GemmBlockScaled>())
        ValidateL0SFGemm(gemm.value(), bindings, std::move(analyzer));
    };
    return InferLayouts(std::move(function), std::move(options));
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.AscendLayoutInference");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendLayoutInference",
                                   AscendLayoutInference);
}

} // namespace tvm::tl::ascend
