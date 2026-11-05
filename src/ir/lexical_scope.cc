/*!
 * \file tl/ir/lexical_scope.cc
 * \brief TileLang-owned lexical regions over the shared RegionStmt contract.
 */

#include "lexical_scope.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {
namespace {

void ValidateScopeAttrs(const DictAttrs &attrs) {
  for (const auto &[key, value] : attrs->dict) {
    TVM_FFI_CHECK(key == "explicit_alloc_scope", ValueError)
        << "Unknown tl.lexical_scope attribute: " << key;
    TVM_FFI_CHECK(value.as<bool>().has_value(), TypeError)
        << "tl.lexical_scope explicit_alloc_scope must be a boolean";
  }
}

ffi::Array<Var> ScopeBodyParams(const CallObj *call) {
  TVM_FFI_CHECK(call->args.empty() && call->ty_args.empty(), ValueError)
      << "tl.lexical_scope has no operands or type arguments";
  ValidateScopeAttrs(call->attrs.value().as_or_throw<DictAttrs>());
  return {};
}

void ValidateScope(const RegionStmtObj *region) {
  TVM_FFI_CHECK(region->args.empty() && region->body_params.empty() &&
                    region->result_vars.empty(),
                ValueError)
      << "tl.lexical_scope has no operands, body parameters, or results";
  ValidateScopeAttrs(region->attrs);
}

} // namespace

const Op &lexical_scope_op() {
  static const Op op = Op::Get("tl.lexical_scope");
  return op;
}

RegionStmt MakeLexicalScope(SeqStmt body, Location loc,
                            bool explicit_alloc_scope) {
  ffi::Map<ffi::String, ffi::Any> attrs;
  if (explicit_alloc_scope)
    attrs.Set("explicit_alloc_scope", true);
  return RegionStmt(lexical_scope_op(), {}, {}, DictAttrs(attrs),
                    std::move(body), {}, std::move(loc));
}

bool IsLexicalScope(const RegionStmtObj *region) {
  return region && region->op.same_as(lexical_scope_op());
}

bool IsExplicitAllocationScope(const RegionStmtObj *region) {
  return IsLexicalScope(region) &&
         region->attrs.GetAttr<bool>("explicit_alloc_scope").value_or(false);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.lexical_scope",
        "Execute a body once while preserving its lexical definition scope.")
      .signature(sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&ScopeBodyParams>(op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateScope>(op_attr::kRegionValidate);
}

} // namespace tvm::tl
