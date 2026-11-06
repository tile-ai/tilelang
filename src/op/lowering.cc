/*!
 * \file tl/op/lowering.cc
 * \brief Native helper inlining with explicit captures and type remapping.
 */
#include "lowering.h"
#include "builtin.h"
#include "ir/stmt_functor.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl {
namespace {

bool IsVoid(const Type &type) {
  return IsVoidType(type) ||
         (type.as<PrimType>() && type.as_or_throw<PrimType>().IsVoid());
}

// A helper's trailing void return denotes normal completion. An early return
// cannot be spliced into its caller without changing control flow.
SeqStmt HelperBody(const SeqStmt &body) {
  ffi::Array<Stmt> statements = body->seq;
  if (!statements.empty()) {
    if (const auto *ret = statements.back().as<ReturnObj>()) {
      TVM_FFI_CHECK(IsVoid(ret->value->ty), ValueError)
          << "A tile-op lowering helper must return void";
      statements.pop_back();
    }
  }
  SeqStmt result(statements, body->loc);
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(result, [](const ReturnObj *) {
    TVM_FFI_THROW(ValueError)
        << "A tile-op lowering helper cannot return from inside its body";
    return ffi::WalkResult::Advance();
  });
  return result;
}

} // namespace

Stmt InlineTileOpFunction(const ffi::Any &lowered, const ffi::String &name) {
  const auto pair = lowered.as_or_throw<ffi::Array<ffi::Any>>();
  TVM_FFI_CHECK_EQ(pair.size(), 2U, ValueError)
      << "A tile-op lowering callback returns (Function, capture_args)";
  tirx::Function function = pair[0].as_or_throw<tirx::Function>();
  const auto arguments = pair[1].as_or_throw<ffi::Array<Expr>>();
  TVM_FFI_CHECK(
      IsVoid(function->ret_type) &&
          (!function->GetAttr(tvm::attr::kDeclaration).value_or(false)),
      ValueError)
      << "A tile-op lowering helper must have a body and a void return type";
  TVM_FFI_CHECK_EQ(function->params.size(), arguments.size(), ValueError)
      << "Tile-op lowering capture arity mismatch";
  TVM_FFI_CHECK(UndefinedVars(function).empty(), ValueError)
      << "Tile-op lowering Function has unbound variables";
  tirx::CheckWellFormed(function);
  CheckLayoutMapScopes(function);
  function = tl::RenewDef(std::move(function));

  auto substitute = ffi::make_object<tl::LayoutMapMutator>();
  for (size_t i = 0; i < arguments.size(); ++i) {
    // Restrict capture evaluation to pure references/expressions: substituting
    // a stateful call at several uses would duplicate its execution.
    if (auto primitive = arguments[i].as<PrimExpr>()) {
      TVM_FFI_CHECK(SideEffect(*primitive) <= CallEffectKind::kPure, ValueError)
          << "A tile-op capture must not duplicate a stateful computation";
    } else {
      TVM_FFI_CHECK(arguments[i].as<Var>().has_value(), ValueError)
          << "A non-primitive tile-op capture must be an existing variable";
    }
    substitute->VarRemapSet(function->params[i], arguments[i]);
  }
  for (size_t i = 0; i < arguments.size(); ++i) {
    Type expected = substitute->Mutate(function->params[i]->ty)
                        .ValueOrUnchanged(function->params[i]->ty)
                        .as_or_throw<Type>();
    TVM_FFI_CHECK(ffi::StructuralEqual()(expected, arguments[i]->ty),
                  ValueError)
        << "Tile-op capture type mismatch at parameter " << i << ": expected "
        << expected << ", got " << arguments[i]->ty;
  }
  const SeqStmt original = HelperBody(function->body);
  SeqStmt body =
      SeqStmt(substitute->Mutate(original).ValueOrUnchanged(original));

  // Only static metadata is stored in attrs. Captured expressions have already
  // been remapped into the body, including tensor shape and layout uses.
  using ffi::Any;
  using ffi::Array;
  Array<Any> schema{
      "map", Array<Any>{Array<Any>{"str", "node"}, Array<Any>{"none"}},
      Array<Any>{Array<Any>{"str", "key"},
                 Array<Any>{"str", tl::attr::kLexicalAllocScope}},
      Array<Any>{Array<Any>{"str", "value"}, Array<Any>{"int", int64_t{1}}},
      Array<Any>{Array<Any>{"str", "name"}, Array<Any>{"str", name}}};
  DictAttrs attributes(ffi::Map<ffi::String, Any>{
      {"schema_version", int64_t{1}}, {"annotation_schema", schema}});
  return RegionStmt(Op::Get("tl.attr"), {tvm::Tuple(ffi::Array<Expr>{})}, {},
                    attributes, std::move(body), {}, function->loc);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.InlineTileOpFunction",
                                   InlineTileOpFunction);
}
} // namespace tvm::tl
