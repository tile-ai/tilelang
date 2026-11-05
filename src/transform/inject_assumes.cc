/*!
 * \file inject_assumes.cc
 * \brief Carry native shape facts and checked user assumptions to their
 * consumers.
 */

#include "../ir/annotations.h"
#include "../ir/expr_utils.h"
#include <sstream>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

class AssumeInjector : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  SeqStmt Run(const tirx::Function &function) {
    Array<Stmt> body;
    for (const Var &parameter : function->params) {
      if (auto tensor = parameter.as<tirx::TensorVar>())
        AppendTensorFacts(tensor.value(), &body, parameter->loc);
    }
    body.push_back(Mutate(function->body).ValueOrUnchanged(function->body));
    return SeqStmt(body, function->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    auto rewritten = StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(
        ffi::GetRef<Stmt>(op));
    if (auto tensor = op->var.as<tirx::TensorVar>()) {
      Array<Stmt> body{rewritten};
      AppendTensorFacts(tensor.value(), &body, op->loc);
      return SeqStmt(body, op->loc);
    }
    return rewritten;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    SeqStmt suffix(Array<Stmt>{}, op->loc);
    for (size_t i = op->seq.size(); i-- > 0;) {
      const Stmt &statement = op->seq[i];
      const auto *evaluate = statement.as<EvaluateObj>();
      const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
      if (call && call->op.same_as(prim::assume_op())) {
        PrimExpr condition = call->args[0].as_or_throw<PrimExpr>();
        TVM_FFI_CHECK(condition.ty().MatchesCode(kDLBool) &&
                          condition.ty().IsScalar(),
                      TypeError)
            << "T.assume requires a scalar boolean condition";
        // The checked marker precedes the optimizer fact. Repeated
        // simplification therefore cannot prove the check from itself.
        std::ostringstream message;
        message << "Assume: " << condition;
        auto encoded = EncodeAnnotations(Map<String, Any>{
            {"node", RenewExpressionDefs(condition)},
            {"key", String("tl.assume_requires_runtime_check")},
            {"value", String(message.str())}});
        suffix = SeqStmt(RegionStmt(
            Op::Get("tl.attr"), {encoded.payload}, {},
            DictAttrs(
                {{"schema_version", 1}, {"annotation_schema", encoded.schema}}),
            SeqStmt({statement, suffix}, statement->loc), {}, statement->loc));
      } else {
        suffix = SeqStmt(
            {Mutate(statement, mode).ValueOrUnchanged(statement), suffix},
            op->loc);
      }
    }
    return suffix;
  }

private:
  static void AppendTensorFacts(const tirx::TensorVar &tensor,
                                Array<Stmt> *body, const Location &loc) {
    for (const PrimExpr &extent : tensor.shape()) {
      if (extent.as<IntImmObj>())
        continue;
      body->push_back(
          Evaluate(prim::assume(extent >= IntImm(extent.ty(), 0), loc), loc));
    }
    int element_bits = tensor.dtype().bits() * tensor.dtype().lanes();
    if (element_bits >= 8)
      return;
    TVM_FFI_CHECK(element_bits > 0 && 8 % element_bits == 0, ValueError);
    auto pattern = tirx::MatchShapeStridePattern(tensor.ty());
    if (!pattern)
      return;
    const auto &strides = pattern.value().get<1>();
    for (size_t i = 0; i + 1 < strides.size(); ++i) {
      const PrimExpr &stride = strides[i];
      if (stride.as<IntImmObj>())
        continue;
      auto fact =
          tvm::truncmod(stride, IntImm(stride.ty(), 8 / element_bits)) ==
          IntImm(stride.ty(), 0);
      body->push_back(Evaluate(prim::assume(fact, loc), loc));
    }
  }
};
} // namespace

tvm::transform::Pass InjectAssumes() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false) ||
        function->GetAttr<bool>("tl.assumes_injected").value_or(false))
      return function;
    auto injector = ffi::make_object<AssumeInjector>();
    SeqStmt body = injector->Run(function);
    function.CopyOnWrite()->body = body;
    return WithAttr(function, "tl.assumes_injected", true);
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.InjectAssumes");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.InjectAssumes", InjectAssumes);
}
} // namespace tvm::tl
