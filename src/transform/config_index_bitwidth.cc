#include "../op/builtin.h"
#include "common/index_legalizer.h"
#include "common/int64_promoter.h"
#include "compat/expr.h"
#include "compat/visit.h"
#include "support/check.h"

#include <tvm/tirx/data_type_rewriter.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

class IndexLegalizer : public tirx::StmtExprMutatorWithAnalyzer {
public:
  explicit IndexLegalizer(sym::Analyzer analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer),
        analyzer_context_(std::move(analyzer)) {}

private:
  ffi::Array<PrimExpr> Legalize(const ffi::Array<PrimExpr> &indices) {
    return indices.Map([&](const PrimExpr &index) {
      bool promote = false;
      PostOrderVisit(index, [&](const ffi::ObjectRef &object) {
        auto expression = object.as<PrimExpr>();
        if (!expression || promote)
          return;
        PrimType type = expression.value().ty();
        if (!type.MatchesCode(kDLInt) || type.bits() >= 64 || !type.IsScalar())
          return;
        auto bounds = analyzer_->const_int_bound(expression.value());
        int64_t maximum = (int64_t{1} << (type.bits() - 1)) - 1;
        int64_t minimum = -(int64_t{1} << (type.bits() - 1));
        promote = bounds->max_value > maximum || bounds->min_value < minimum;
      });
      return promote ? Int64Promoter()(index) : index;
    });
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                     ffi::InplaceMode mode) final {
    TensorLoad load = StmtExprMutatorWithAnalyzer::Mutate_(node, mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(node))
                          .as_or_throw<TensorLoad>();
    load.CopyOnWrite()->indices = Legalize(load->indices);
    return load;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                                 ffi::InplaceMode mode) final {
    TensorStore store = StmtExprMutatorWithAnalyzer::Mutate_(node, mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(node))
                            .as_or_throw<TensorStore>();
    store.CopyOnWrite()->indices = Legalize(store->indices);
    return store;
  }

  sym::Analyzer analyzer_context_;
};

tirx::Function LegalizeIndexArithmetic(tirx::Function function) {
  function.CopyOnWrite()->body =
      ffi::make_object<IndexLegalizer>(sym::Analyzer())
          ->Mutate(function->body, ffi::InplaceMode::kDisallow)
          .ValueOrUnchanged(function->body);
  return function;
}

tvm::transform::Pass ConfigIndexBitwidth() {
  auto transform = [](tirx::Function function, const IRModule &,
                      const tvm::transform::PassContext &context) {
    if (auto width = context->GetConfig<int64_t>(kConfigIndexBitwidth)) {
      TVM_FFI_CHECK(width.value() == 32 || width.value() == 64, ValueError)
          << "tl.config_index_bitwidth must be 32 or 64";
      function = ffi::make_object<tirx::IndexDataTypeNormalizer>(
                     PrimType::Int(width.value()))
                     ->Rewrite(function);
    }
    return LegalizeIndexArithmetic(std::move(function));
  };
  return tirx::transform::CreateFunctionPass(transform, 0,
                                             "tl.ConfigIndexBitwidth");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.ConfigIndexBitwidth",
                                   ConfigIndexBitwidth);
}

} // namespace tvm::tl
