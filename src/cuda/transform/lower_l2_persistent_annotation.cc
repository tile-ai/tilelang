/*!
 * \file lower_l2_persistent_annotation.cc
 * \brief Consume native L2 metadata without hiding tensor uses in attributes.
 */
#include "transform/common/native_layout_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

class L2PersistentLowerer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *statement,
                            InplaceMode mode) final {
    const auto *call = statement->value.as<CallObj>();
    if (!call || !call->op.same_as(Op::Get("tl.annotation")))
      return StmtExprMutator::Mutate_(statement, mode);
    auto annotations = ReadNativeAnnotations(call);
    auto hints = annotations.Get("l2_hit_ratio_map");
    if (!hints)
      return StmtExprMutator::Mutate_(statement, mode);
    for (const auto &[value, ratio] : hints->cast<Map<Any, Any>>()) {
      TensorVar tensor = value.cast<TensorVar>();
      TVM_FFI_CHECK(tensor.dtype().IsScalar(), ValueError)
          << "L2 persistent windows require scalar tensor elements";
      auto hit_ratio = ratio.cast<FloatImm>();
      TVM_FFI_CHECK(hit_ratio->value >= 0 && hit_ratio->value <= 1, ValueError)
          << "L2 persistent hit ratio must be in [0, 1]";
      auto known = names_.find(tensor->name);
      TVM_FFI_CHECK(known == names_.end() || known->second == tensor.get(),
                    ValueError)
          << "L2 persistent annotations require unambiguous tensor names";
      names_.insert_or_assign(tensor->name, tensor.get());
      // Preserve the source annotation's logical byte-window calculation.
      // The native Hopper host consumer independently adds the typed origin.
      PrimExpr bytes =
          IntImm(PrimType::Int(64), (tensor.dtype().bits() + 7) / 8);
      for (const PrimExpr &extent : tensor.shape())
        bytes = bytes * prim::cast(extent, PrimType::Int(64));
      entries.Set(tensor->name, {hit_ratio, bytes});
    }
    annotations.erase("l2_hit_ratio_map");
    changed = true;
    if (annotations.empty())
      return SeqStmt({}, statement->loc);
    return MakeNativeAnnotation(annotations, statement->loc);
  }

  Map<String, Array<Expr>> entries;
  bool changed{false};

private:
  std::unordered_map<std::string, const VarObj *> names_;
};
} // namespace

tvm::transform::Pass LowerL2Persistent() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto lowerer = make_object<L2PersistentLowerer>();
    SeqStmt body = function->body;
    body = lowerer->Mutate(body).ValueOrUnchanged(body);
    if (!lowerer->changed)
      return function;
    function.CopyOnWrite()->body = body;
    if (!lowerer->entries.empty()) {
      function =
          WithAttr(std::move(function), "l2_persistent_map", lowerer->entries);
      function = WithoutAttr(std::move(function), "tl.l2_policy_materialized");
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0,
                                             "tl.LowerL2Persistent");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.LowerL2Persistent",
                                   LowerL2Persistent);
}
} // namespace tvm::tl
