/*!
 * \file lower_ascend_annotations.cc
 * \brief Prepare Ascend execution-region contracts for code generation.
 */

#include "ascend/ir/block.h"
#include "ir/annotations.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl::ascend {
namespace {

using ffi::Any;
using ffi::Map;
using ffi::String;

class AscendAnnotationLowerer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  explicit AscendAnnotationLowerer(const tirx::Function &function) {
    if (auto target = function->GetAttr<Target>(tvm::attr::kTarget))
      is_ascend_ = target.value()->kind->name == "ascend";
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (IsAscendBlock(op)) {
      auto name = AscendBlockName(op);
      TVM_FFI_CHECK(is_ascend_, ValueError)
          << "Ascend execution region " << name << " requires an Ascend target";
      auto annotations = AscendBlockAnnotations(op);
      PrepareHeader(name, &annotations);
      auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
      return MakeAscendBlock(name, annotations, body, op->loc);
    }
    if (op->op->name == "tl.attr") {
      auto fields = ReadAnnotations(op);
      const auto key = fields.at("key").cast<String>();
      if (key == "tl.simtvf_scope" || key == "tl.simdvf_scope")
        TVM_FFI_CHECK(is_ascend_, ValueError)
            << "Ascend VF scope " << key << " requires an Ascend target";
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

private:
  static void PrepareHeader(const String &name, Map<String, Any> *annotations) {
    const bool vf = name == "SIMT_VF" || name == "SIMD_VF";
    for (const auto &[key, value] : *annotations) {
      if (key == "lexical_alloc_scope")
        continue;
      const bool vector_count = name == "VECTOR" && key == "vector_count";
      const bool source_index = vf && key == "tl.vf_source_index";
      const bool latency = vf && key == "tl.vf_latency";
      TVM_FFI_CHECK(vector_count || source_index || latency, ValueError)
          << "Unconsumed structured block annotation `" << key
          << "` at LowerAscendAnnotations; run its owning lowering pass first";
      const auto *literal = value.as<IntImmObj>();
      TVM_FFI_CHECK(literal || value.try_cast<int64_t>().has_value(),
                    ValueError)
          << "Ascend execution-region annotation " << key
          << " must be a constant integer";
      const int64_t count = literal ? static_cast<int64_t>(literal->value)
                                    : value.cast<int64_t>();
      if (vector_count) {
        TVM_FFI_CHECK(count == 1 || count == 2, ValueError)
            << "VECTOR vector_count must be a constant integer 1 or 2";
      } else if (source_index) {
        TVM_FFI_CHECK_GE(count, 0, ValueError)
            << "VF source index must be nonnegative";
      }
    }
    // This boundary also runs when scheduling was disabled. Validate before
    // retiring latency, while keeping the source index needed by emitters.
    if (vf)
      annotations->erase("tl.vf_latency");
  }

  bool is_ascend_{false};
};

} // namespace
} // namespace tvm::tl::ascend

namespace tvm::tl::transform {

tvm::transform::Pass LowerAscendAnnotations() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto lowerer = ffi::make_object<ascend::AscendAnnotationLowerer>(function);
    auto body = function->body;
    function.CopyOnWrite()->body = lowerer->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.LowerAscendAnnotations");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerAscendAnnotations",
                                   LowerAscendAnnotations);
}

} // namespace tvm::tl::transform
