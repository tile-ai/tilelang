/*!
 * \file testing/cpp/native_operator_probe.cc
 * \brief Test-only bridge to real TileLang operator inference and lowering.
 */
#include "cuda/transform/ptx_async_copy_injector.h"
#include "op/operator.h"
#include "rocm/transform/async_copy_injector.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {
namespace {

class AccessProbeNode final : public TileOperatorNode {
public:
  AccessProbeNode(TensorRegion region, int mask)
      : region_(region), mask_(mask) {
    SetAccessRegions({{region_, mask_}});
  }
  Stmt Lower(const LowerArgs &, sym::AnalyzerObj *) const final {
    TVM_FFI_THROW(InternalError)
        << "Access-region queries must not lower the probe";
  }
  LayoutMap InferLayout(const LayoutInferArgs &, InferLevel) const final {
    TVM_FFI_THROW(InternalError)
        << "Access-region queries must not infer probe layouts";
  }
  TileOperator Clone() const final {
    return TileOperator(ffi::make_object<AccessProbeNode>(region_, mask_));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.testing.AccessProbe", AccessProbeNode,
                                    TileOperatorNode);

private:
  TensorRegion region_;
  int mask_;
};

ffi::Map<ffi::String, ffi::Any>
LowerNativeOperatorForTest(Call call, Target target, LayoutMap layouts) {
  sym::Analyzer analyzer;
  TileOperator operation = ParseOperator(call);
  TVM_FFI_CHECK(operation.defined(), ValueError)
      << "The test bridge requires a registered TileLang tile operator";
  LayoutInferArgs infer{target};
  infer.layout_map = std::move(layouts);
  infer.analyzer = analyzer.get();
  for (InferLevel level :
       {InferLevel::kCommon, InferLevel::kStrict, InferLevel::kFree}) {
    for (const auto &[tensor, layout] : operation->InferLayout(infer, level)) {
      infer.layout_map.Set(tensor, layout);
    }
  }
  LowerArgs lower{target};
  lower.layout_map = infer.layout_map;
  const AccessRegions accesses = operation->GetAccessRegions();
  Stmt statement = operation->Lower(lower, analyzer.get());
  return {{"statement", statement},
          {"layouts", infer.layout_map},
          {"reads", accesses.reads},
          {"writes", accesses.writes},
          {"operator", operation}};
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterTileOperator(
      "tl.testing.access_probe",
      [](ffi::Array<Expr> operands, BlockAnnotations) {
        TVM_FFI_CHECK_EQ(operands.size(), 1U, ValueError);
        return TileOperator(ffi::make_object<AccessProbeNode>(
            operands[0].as_or_throw<TensorRegion>(), kAccessRead));
      },
      OpBlockAnnotationHandlerFunc([](TileOperator operation,
                                      BlockAnnotations annotations) {
        auto original = operation->GetAccessRegions().reads[0];
        int mask =
            annotations.Get("testing.access_mask").value_or(1).cast<int>();
        return TileOperator(ffi::make_object<AccessProbeNode>(original, mask));
      }));
  ffi::reflection::GlobalDef()
      .def("tl.testing.InjectPTXAsyncCopy",
           [](Stmt body, bool explicit_async) {
             auto result = InjectPTXAsyncCopy(body, explicit_async);
             return ffi::Map<ffi::String, ffi::Any>{
                 {"statement", result.stmt},
                 {"injected", result.injected_ptx_async_copy}};
           })
      .def("tl.testing.InjectROCmAsyncCopy",
           [](Stmt body, bool explicit_async) {
             auto result = InjectROCmAsyncCopy(body, explicit_async);
             return ffi::Map<ffi::String, ffi::Any>{
                 {"statement", result.stmt},
                 {"injected", result.injected_rocm_async_copy}};
           })
      .def("tl.testing.LowerNativeOperator", LowerNativeOperatorForTest)
      .def("tl.testing.ParseNativeOperator", [](Call call) {
        TileOperator operation = ParseOperator(call);
        TVM_FFI_CHECK(operation.defined(), ValueError)
            << "The test bridge requires a registered TileLang tile operator";
        return operation;
      });
}
} // namespace tvm::tl
