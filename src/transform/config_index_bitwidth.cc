/*!
 * \file config_index_bitwidth.cc
 * \brief Preserve public types while legalizing native address arithmetic.
 */
#include "config_index_bitwidth.h"
#include "../ir/analyzer.h"
#include "../op/builtin.h"
#include "common/int64_promoter.h"

#include "ir/stmt_functor.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

class IndexLegalizer : public tl::StmtExprMutatorWithAnalyzer {
public:
  using StmtExprMutatorWithAnalyzer::Mutate;
  using StmtExprMutatorWithAnalyzer::Mutate_;
  explicit IndexLegalizer(bool force_wide,
                          sym::Analyzer analyzer = CreateTileLangAnalyzer())
      : StmtExprMutatorWithAnalyzer(analyzer),
        analyzer_owner_(std::move(analyzer)), force_wide_(force_wide) {}

  tirx::Function Rewrite(tirx::Function function) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    MarkTensorParamShapes(function);
    const auto body = function->body;
    auto result = Mutate(body).ValueOrUnchanged(body);
    if (!result.same_as(body))
      function.CopyOnWrite()->body = std::move(result);
    return function;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                     InplaceMode mode) final {
    const auto load = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                          .ValueOrUnchanged(ffi::GetRef<TensorLoad>(op))
                          .as_or_throw<TensorLoad>();
    return TensorLoad(load->source, RewriteIndices(load->indices), load->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op,
                                 InplaceMode mode) final {
    const auto store = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                           .ValueOrUnchanged(ffi::GetRef<TensorStore>(op))
                           .as_or_throw<TensorStore>();
    return TensorStore(store->dest, RewriteIndices(store->indices),
                       store->value, store->loc);
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    auto call = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                    .ValueOrUnchanged(ffi::GetRef<Call>(op))
                    .as_or_throw<Call>();
    // tl.ptr_access still counts logical elements. Its offset must be wide
    // before LowerAccessPtr scales it to bytes, just like a native byte offset.
    if (call->op.same_as(tirx::ptr_byte_offset_op()) ||
        call->op.same_as(Op::Get("tl.ptr_access"))) {
      auto args = call->args;
      args.Set(1, RewriteIndex(args[1].as_or_throw<PrimExpr>()));
      return Call(call->ty, call->op, args, call->attrs, call->ty_args,
                  call->loc);
    }
    return call;
  }

private:
  PrimExpr RewriteIndex(const PrimExpr &index) {
    const PrimType type = index.ty();
    if (!type.MatchesCode(kDLInt) || type.bits() >= 64)
      return index;
    PrimExpr wide = Int64Promoter()(index);
    if (force_wide_)
      return wide;
    // Inspect the mathematical address before narrowing would hide overflow.
    // A bound on the original int32 expression alone can only describe its
    // truncated result and therefore cannot justify a narrow address.
    if (type.IsScalar()) {
      try {
        auto bound = GetSyncedAnalyzer()->const_int_bound(wide);
        const int64_t limit = int64_t{1} << (type.bits() - 1);
        if (bound->min_value >= -limit && bound->max_value < limit)
          return index;
      } catch (const std::exception &) {
        // An unproved signed address needs the wider representation.
      }
    }
    return wide;
  }

  ffi::Array<PrimExpr> RewriteIndices(const ffi::Array<PrimExpr> &indices) {
    return indices.Map(
        [&](const PrimExpr &value) { return RewriteIndex(value); });
  }
  // The native base borrows the analyzer pointer. Keep its owner alive for
  // every scoped bound and proof made during this traversal.
  sym::Analyzer analyzer_owner_;
  bool force_wide_;
};

} // namespace
namespace transform {

tvm::transform::Pass LegalizeIndexArithmetic(bool force_wide) {
  auto rewrite = [force_wide](tirx::Function function, IRModule,
                              tvm::transform::PassContext) {
    return ffi::make_object<IndexLegalizer>(force_wide)
        ->Rewrite(std::move(function));
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0,
                                             "tl.LegalizeIndexArithmetic");
}

tvm::transform::Pass ConfigIndexBitwidth() {
  auto pass = [](IRModule module, tvm::transform::PassContext context) {
    auto requested = context->GetConfig<int64_t>(kConfigIndexBitwidth);
    TVM_FFI_CHECK(!requested || requested.value() == 32 ||
                      requested.value() == 64,
                  ValueError)
        << "tl.config_index_bitwidth must be 32 or 64";
    module = LegalizeIndexArithmetic(requested.value_or(32) == 64)(module);
    // A requested 32-bit index width is used only when native range analysis
    // proves it safe. It never changes tensor element dtypes or the public ABI.
    if (requested && requested.value() == 32)
      module = tirx::transform::NarrowDataType(32)(module);
    return module;
  };
  return tvm::transform::CreateModulePass(pass, 0, "tl.ConfigIndexBitwidth");
}

} // namespace transform
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.ConfigIndexBitwidth",
                                   transform::ConfigIndexBitwidth);
}
} // namespace tvm::tl
