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
    auto result = StmtExprMutator::Mutate_(op, mode);
    if (!IsAscendBlock(op))
      return result;
    TVM_FFI_CHECK(is_ascend_, ValueError)
        << "Ascend execution region requires an Ascend target";
    auto region = std::move(result)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                      .as_or_throw<RegionStmt>();
    if (region->attrs->dict.count("tl.vf_latency")) {
      auto attrs = region->attrs->dict;
      attrs.erase("tl.vf_latency");
      region.CopyOnWrite()->attrs = DictAttrs(attrs);
    }
    return region;
  }

private:
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
