/*!
 * \file verify_lowered_annotations.cc
 * \brief Check that annotation consumers have finished before code generation.
 */
#include "ir/annotations.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

class LoweredAnnotationVerifier : public tirx::StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (op->op->name == "tl.attr") {
      auto fields =
          DecodeAnnotations(
              op->args[0].as_or_throw<Tuple>(),
              op->attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema")
                  .value())
              .cast<ffi::Map<ffi::String, ffi::Any>>();
      auto key = fields.at("key").cast<ffi::String>();
      TVM_FFI_CHECK(IsCodegenAnnotation(key), ValueError)
          << "Unconsumed TileLang region `" << key
          << "` at VerifyLoweredAnnotations; run its owning lowering pass "
             "first";
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (const auto *intrinsic = op->op.as<OpObj>()) {
      if (IsAnnotationCall(op)) {
        auto values = ReadAnnotations(op);
        TVM_FFI_THROW(ValueError)
            << "Unconsumed TileLang call `" << intrinsic->name
            << "` at "
               "VerifyLoweredAnnotations: "
            << values << "; run its owning lowering pass first";
      }
      TVM_FFI_CHECK(intrinsic->name.find("tl.tileop.") != 0, ValueError)
          << "Unconsumed TileLang call `" << intrinsic->name
          << "` at VerifyLoweredAnnotations; run its owning lowering pass "
             "first";
    }
    if (op->attrs) {
      if (auto attrs = op->attrs.value().as<DictAttrs>()) {
        for (const char *key :
             {"operand_count", "annotation_schema", "tl.source_block_alloc"}) {
          TVM_FFI_CHECK(!attrs.value()->dict.count(key), ValueError)
              << "Unconsumed call annotation `" << key
              << "` at VerifyLoweredAnnotations; run its owning lowering pass "
                 "first";
        }
      }
    }
    return StmtExprVisitor::Visit_(op);
  }
};

} // namespace

namespace transform {
tvm::transform::Pass VerifyLoweredAnnotations() {
  auto verify = [](tirx::Function function, IRModule,
                   tvm::transform::PassContext) {
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      // Inspect executable bodies, not archived source modules or the final
      // pointer/parameter-index alias contracts stored in Function attrs.
      auto verifier = ffi::make_object<LoweredAnnotationVerifier>();
      verifier->Visit(function->body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(verify, 0,
                                             "tl.VerifyLoweredAnnotations");
}
} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.VerifyLoweredAnnotations",
                                   transform::VerifyLoweredAnnotations);
}
} // namespace tvm::tl
