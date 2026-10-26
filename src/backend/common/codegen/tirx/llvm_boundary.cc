#include "backend/common/codegen/native_codegen.h"
#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "compat/region.h"
#include "op/dtype.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl {
namespace {
using namespace tvm::tirx;
using namespace tvm::prim;

class ContinueUpdater : public tvm::tl::StmtExprMutator {
public:
  explicit ContinueUpdater(Stmt update) : update_(std::move(update)) {}
  using tvm::tl::StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const ContinueNode *node, InplaceMode) final {
    return SeqStmt({update_, GetRef<Stmt>(node)}, node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *, InplaceMode) final {
    return ffi::Unchanged();
  }

  UnchangedOr<Stmt> Mutate_(const WhileNode *, InplaceMode) final {
    return ffi::Unchanged();
  }

private:
  Stmt update_;
};

class LLVMCodegenNormalizer : public tvm::tl::StmtExprMutator {
public:
  explicit LLVMCodegenNormalizer(bool packed) : packed_(packed) {}
  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode mode) final {
    if (auto type = value.as<PrimType>();
        type && ::tvm::tl::IsCustomDType(*type))
      return ffi::Any(::tvm::tl::CustomDTypeStorage(*type));
    return tvm::tl::StmtExprMutator::Mutate(value, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const CastNode *node, InplaceMode mode) final {
    TVM_FFI_CHECK(
        !::tvm::tl::IsCustomDType(node->value.ty()) &&
            !::tvm::tl::IsCustomDType(node->ty.as_or_throw<PrimType>()),
        ValueError)
        << "Custom dtype casts require a backend numeric lowering";
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const FloatImmNode *node,
                                InplaceMode mode) final {
    TVM_FFI_CHECK(!::tvm::tl::IsCustomDType(node->ty.as_or_throw<PrimType>()),
                  ValueError)
        << "Custom dtype constants require a backend numeric lowering";
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node, InplaceMode mode) final {
    Call call = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(GetRef<Expr>(node))
                    .as_or_throw<Call>();
    if (call->op.same_as(alloc_tensor_op()) && !call->attrs)
      call.CopyOnWrite()->attrs = DictAttrs();
    return call;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    RegionStmt region = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                            .ValueOrUnchanged(GetRef<Stmt>(node))
                            .as_or_throw<RegionStmt>();
    if (region->op.same_as(::tvm::tl::scope_op())) {
      TVM_FFI_CHECK(region->args.empty() && region->body_params.empty() &&
                        region->result_vars.empty(),
                    ValueError)
          << "A TileLang lexical scope cannot carry operands or results";
      return region->body;
    }
    return region;
  }

  UnchangedOr<Stmt> Mutate_(const ReturnNode *node, InplaceMode mode) final {
    Expr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    if (const auto *tuple = value.as<TupleNode>();
        tuple && tuple->fields.empty())
      return Return(IntImm::Int32(0), node->loc);
    auto scalar = value.as<PrimExpr>();
    if (scalar && IsZero(*scalar))
      return Return(value, node->loc);
    TVM_FFI_CHECK(packed_ && scalar && scalar->ty() == PrimType::Int(32),
                  ValueError)
        << "TileLang LLVM codegen requires MakePackedAPI before nonzero "
           "returns";
    Stmt failure = Evaluate(Call(PrimType::Int(32), throw_last_error_op(), {}));
    return SeqStmt({If(*scalar != 0, failure), Return(IntImm::Int32(0))},
                   node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *node, InplaceMode mode) final {
    For loop = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                   .ValueOrUnchanged(GetRef<Stmt>(node))
                   .as_or_throw<For>();
    TVM_FFI_CHECK(!IsZero(loop->step), ValueError)
        << "For loop step must be nonzero";
    if (tl::arith::CanProve(analyzer_, loop->step > 0) &&
        SideEffect(loop->begin) == CallEffectKind::kPure &&
        SideEffect(loop->end) == CallEffectKind::kPure &&
        SideEffect(loop->step) == CallEffectKind::kPure)
      return loop;
    TVM_FFI_CHECK(loop->kind != ForKind::kParallel &&
                      loop->kind != ForKind::kVectorized,
                  ValueError)
        << "Non-forward LLVM loops must be serialized before code generation";
    PrimType type = loop->var.ty();
    IntImm zero = IntImm::Int32(0);
    PrimVar begin("loop_begin", type), end("loop_end", type),
        step("loop_step", type);
    TensorVar counter(
        "loop_counter",
        TensorType({IntImm::Int32(1)}, type, runtime::MemoryScope("local")));
    auto current = [&]() -> PrimExpr { return TensorLoad(counter, {zero}); };
    Stmt update = TensorStore(counter, {zero}, loop->var + step);
    Stmt body = ffi::make_object<ContinueUpdater>(update)
                    ->Mutate(loop->body, InplaceMode::kDisallow)
                    .ValueOrUnchanged(loop->body);
    PrimExpr condition =
        (step > 0 && current() < end) || (step < 0 && current() > end);
    return SeqStmt(
        {Bind(begin, loop->begin), Bind(end, loop->end), Bind(step, loop->step),
         Bind(counter, Call(counter.ty(), alloc_tensor_op(), {}, DictAttrs(),
                            {counter.ty()})),
         TensorStore(counter, {zero}, begin),
         While(condition, SeqStmt({Bind(loop->var, current()), body, update}))},
        loop->loc);
  }

private:
  bool packed_;
  sym::Analyzer analyzer_;
};

ffi::Module BuildNativeLLVM(IRModule module, Target target) {
  for (const auto &[global, base_function] : module->functions) {
    auto function = base_function.as_or_throw<Function>();
    bool packed =
        function->GetAttr(tvm::attr::kCallingConv)
            .value_or(CallingConv::kDefault) == CallingConv::kCPackedFunc;
    auto normalizer = ffi::make_object<LLVMCodegenNormalizer>(packed);
    auto rewritten = normalizer->Mutate(function, InplaceMode::kDisallow)
                         .ValueOrUnchanged(function);
    module.CopyOnWrite()->functions.Set(global,
                                        rewritten.as_or_throw<Function>());
  }
  return tvm::codegen::Build(std::move(module), std::move(target));
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.codegen.build_llvm", BuildNativeLLVM)
      .def("target.build.tilelang_llvm", [](IRModule module, Target target) {
        return BuildNativeCodegen(std::move(module), std::move(target));
      });
}

} // namespace tvm::tl
