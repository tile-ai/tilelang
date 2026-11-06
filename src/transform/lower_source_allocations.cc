/*!
 * \file lower_source_allocations.cc
 * \brief Retire source allocation origins after their lifetime owner ran.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

class SourceAllocationLowerer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  explicit SourceAllocationLowerer(const tirx::Function &function) {
    if (auto target = function->GetAttr<Target>(tvm::attr::kTarget))
      target_kind_ = target.value()->kind->name;
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    if (!op->op.same_as(tirx::alloc_tensor_op()) || !op->attrs)
      return StmtExprMutator::Mutate_(op, mode);
    auto attrs = op->attrs.value().as<DictAttrs>();
    if (!attrs || !attrs.value()->dict.count("tl.source_block_alloc"))
      return StmtExprMutator::Mutate_(op, mode);
    auto origin = attrs.value()->dict.at("tl.source_block_alloc").as<IntImm>();
    TVM_FFI_CHECK(origin && origin.value()->value == 1, ValueError)
        << "tl.source_block_alloc must be the source allocation marker";
    TVM_FFI_CHECK(!target_kind_.empty(), ValueError)
        << "LowerSourceAllocations requires a bound target for source "
           "allocation origins";
    TVM_FFI_CHECK(target_kind_ != "c" && target_kind_ != "llvm", ValueError)
        << "CPU source allocation origins require LowerCPUKernelLaunch "
           "before LowerSourceAllocations";

    // CPU owns placement. Other targets keep the native allocation exactly
    // where it is, including its type, placement operands and other attrs.
    auto rewritten = StmtExprMutator::Mutate_(op, mode)
                         .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                         .as_or_throw<Call>();
    auto values = rewritten->attrs.value().as_or_throw<DictAttrs>()->dict;
    values.erase("tl.source_block_alloc");
    return Call(rewritten->ty, rewritten->op, rewritten->args,
                DictAttrs(values), rewritten->ty_args, rewritten->loc);
  }

private:
  ffi::String target_kind_;
};

} // namespace

namespace transform {

tvm::transform::Pass LowerSourceAllocations() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto lowerer = ffi::make_object<SourceAllocationLowerer>(function);
    auto body = function->body;
    function.CopyOnWrite()->body = lowerer->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.LowerSourceAllocations");
}

} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerSourceAllocations",
                                   transform::LowerSourceAllocations);
}

} // namespace tvm::tl
