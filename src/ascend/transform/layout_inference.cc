/*!
 * \file ascend/transform/layout_inference.cc
 * \brief Ascend region policy over the shared native layout inference engine.
 */
#include "ascend/ir/block.h"
#include "ascend/op/utils.h"
#include "backend/common/target_utils.h"
#include "ir/analyzer.h"
#include "ir/annotations.h"
#include "op/gemm_blockscaled.h"
#include "op/reducer.h"
#include "transform/layout_inference/policy.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>

#include "ir/stmt_functor.h"
#include <string_view>
#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl::ascend {
namespace {
using namespace tirx;

bool ScopeIs(const RegionStmtObj *region, const char *name) {
  if (IsAscendBlock(region)) {
    const auto block = AscendBlockName(region);
    return (std::string_view(name) == "tl.simdvf_scope" &&
            block == "SIMD_VF") ||
           (std::string_view(name) == "tl.simtvf_scope" && block == "SIMT_VF");
  }
  if (region->op->name != "tl.attr")
    return false;
  auto fields = ReadAnnotations(region);
  return fields.at("key").cast<ffi::String>() == name &&
         GetIntegerAnnotation(fields.at("value")) != 0;
}

class SFGemmValidator : public StmtExprVisitorWithAnalyzer {
public:
  explicit SFGemmValidator(L0SFBindings bindings)
      : bindings_(std::move(bindings)) {}
  using StmtExprVisitorWithAnalyzer::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (op->op.same_as(GemmBlockScaled::Get())) {
      auto decoded = DecodeTileOpArguments(op);
      ValidateL0SFGemm(GemmBlockScaled(decoded.operands, decoded.annotations),
                       bindings_, GetSyncedAnalyzer());
    }
    return StmtExprVisitorWithAnalyzer::Visit_(op);
  }

private:
  L0SFBindings bindings_;
};
} // namespace

tvm::transform::Pass AscendLayoutInference() {
  auto validate = tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        auto target = function->GetAttr<Target>(tvm::attr::kTarget);
        TVM_FFI_CHECK(target && TargetIsAscend(*target), ValueError)
            << "AscendLayoutInference requires an Ascend target";
        if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false))
          ffi::make_object<SFGemmValidator>(CollectL0SFBindings(function->body))
              ->Visit(function->body);
        return function;
      },
      0, "tl.ValidateAscendLayoutOperands");
  LayoutInferenceOptions options;
  options.opaque_region = [](const RegionStmtObj *region) {
    return ScopeIs(region, "tl.simdvf_scope");
  };
  options.preserve_parallel_nest = [](const ForObj *loop,
                                      const ffi::Array<RegionStmt> &regions) {
    bool simt = false;
    for (const auto &region : regions)
      simt |= ScopeIs(region.get(), "tl.simtvf_scope");
    if (!simt)
      return false;
    bool reducer = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        loop->body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
          reducer |= call->op.same_as(reducer_update());
          return ffi::WalkResult::Advance();
        });
    return reducer;
  };
  return tvm::transform::Sequential(
      {validate, LayoutInferenceWithPolicy(std::move(options))},
      "tl.AscendLayoutInference");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendLayoutInference",
                                   AscendLayoutInference);
}
} // namespace tvm::tl::ascend
