/*!
 * \file mark_scalar_dcache_bypass.cc
 * \brief Bypass scalar GM dcache for physical storage that is written.
 */
#include "ascend/op/builtin.h"
#include "ir/alias_analysis.h"
#include "ir/annotations.h"
#include "op/builtin.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

bool IsGlobal(const Expr &expression) {
  if (const auto *tensor = expression->ty.as<TensorTypeObj>())
    return tensor->scope == runtime::MemoryScope::Global();
  if (const auto *pointer = expression->ty.as<PtrTypeObj>())
    return pointer->scope == runtime::MemoryScope::Global();
  return false;
}

class ScalarDcacheBypassRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit ScalarDcacheBypassRewriter(const tirx::Function &function)
      : roots_(function) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *store = object.as<TensorStoreObj>())
            RecordWrite(store->dest);
          if (const auto *call = object.as<CallObj>()) {
            if (call->op.same_as(ascend_copy_ubuf_to_gm()) ||
                call->op.same_as(ascend_copy_matrix_cc_to_gm()) ||
                call->op.same_as(ascend_write_gm_bypass_dcache())) {
              auto operands = DecodeCallOperands(call);
              if (!operands.empty())
                RecordWrite(operands[0]);
            }
          }
          return ffi::WalkResult::Advance();
        });
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    // Skipping the whole VF also handles nested scopes without leaking a
    // boolean state into later siblings. VF memory has its dedicated path.
    if (ascend::GetVFKind(op) != ascend::VFKind::kNone)
      return Unchanged();
    return StmtExprMutator::Mutate_(op, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                InplaceMode mode) final {
    auto original = GetRef<PrimExpr>(op);
    auto load = StmtExprMutator::Mutate_(op, mode)
                    .ValueOrUnchanged(original)
                    .as_or_throw<TensorLoad>();
    if (load->indices.size() == 1 && load.ty().lanes() == 1 &&
        ShouldBypass(load->source)) {
      return Call(load->ty, ascend_read_gm_bypass_dcache(), {Address(load)}, {},
                  {}, load->loc)
          .as_or_throw<PrimExpr>();
    }
    return load;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op, InplaceMode mode) final {
    auto original = GetRef<Stmt>(op);
    auto store = StmtExprMutator::Mutate_(op, mode)
                     .ValueOrUnchanged(original)
                     .as_or_throw<TensorStore>();
    if (store->indices.size() == 1 && store->value.ty().lanes() == 1 &&
        ShouldBypass(store->dest)) {
      TensorLoad destination(store->dest, store->indices, store->loc);
      return Evaluate(Call(PrimType::Void(), ascend_write_gm_bypass_dcache(),
                           {Address(destination), store->value}, {}, {},
                           store->loc),
                      store->loc);
    }
    return store;
  }

  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    if ((op->op.same_as(tirx::address_of_op()) ||
         op->op.same_as(tl::access_ptr())) &&
        !op->args.empty()) {
      if (const auto *load = op->args[0].as<TensorLoadObj>()) {
        // An address marker does not read the tensor. Its index expressions
        // still execute normally, so nested scalar loads in indices do rewrite.
        auto args = op->args;
        auto source = Mutate(load->source, mode).ValueOrUnchanged(load->source);
        Array<PrimExpr> indices;
        for (const auto &index : load->indices)
          indices.push_back(Mutate(index, mode).ValueOrUnchanged(index));
        args.Set(0, TensorLoad(source, indices, load->loc));
        for (size_t i = 1; i < args.size(); ++i)
          args.Set(i, Mutate(args[i], mode).ValueOrUnchanged(args[i]));
        return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->loc);
      }
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

private:
  void RecordWrite(const Expr &expression) {
    if (!IsGlobal(expression))
      return;
    auto root = roots_.Root(expression);
    if (root)
      writable_.insert(root.value().get());
    else
      unknown_write_ = true;
  }
  bool ShouldBypass(const Expr &expression) const {
    if (!IsGlobal(expression))
      return false;
    auto root = roots_.Root(expression);
    return unknown_write_ || (root ? writable_.count(root.value().get()) != 0
                                   : !writable_.empty());
  }
  static Expr Address(const TensorLoad &load) {
    auto tensor_type = load->source->ty.as_or_throw<TensorType>();
    return Call(tensor_type->DataPtrType(), tirx::address_of_op(), {load}, {},
                {}, load->loc);
  }
  PhysicalRootAnalysis roots_;
  std::unordered_set<const VarObj *> writable_;
  bool unknown_write_{false};
};
} // namespace

tvm::transform::Pass MarkScalarDcacheBypass() {
  auto pass = [](tirx::Function function, const IRModule &,
                 const tvm::transform::PassContext &) {
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      auto rewriter = ffi::make_object<ScalarDcacheBypassRewriter>(function);
      function.CopyOnWrite()->body =
          rewriter->Mutate(function->body).ValueOrUnchanged(function->body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.MarkScalarDcacheBypass");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MarkScalarDcacheBypass",
                                   MarkScalarDcacheBypass);
}
} // namespace tvm::tl
