/*!
 * \file ascend/transform/layout_inference.cc
 * \brief Ascend region policy over the shared native layout inference engine.
 */
#include "ascend/op/utils.h"
#include "backend/common/target_utils.h"
#include "ir/analyzer.h"
#include "op/gemm_blockscaled.h"
#include "op/reducer.h"
#include "transform/common/native_layout_utils.h"
#include "transform/layout_inference/native_policy.h"

#include "ir/stmt_functor.h"
#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl::ascend {
namespace {
using namespace tirx;

bool ScopeIs(const RegionStmtObj *region, const char *name) {
  if (region->op->name != "tl.attr")
    return false;
  auto fields = ReadNativeAnnotations(region);
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
  NativeLayoutInferenceOptions options;
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
    VisitNativePostOrder(loop->body, [&](const ffi::ObjectRef &node) {
      if (const auto *call = node.as<CallObj>())
        reducer |= call->op.same_as(reducer_update());
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
