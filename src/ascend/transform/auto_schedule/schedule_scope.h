#ifndef TILELANG_ASCEND_TRANSFORM_SCHEDULE_SCOPE_H_
#define TILELANG_ASCEND_TRANSFORM_SCHEDULE_SCOPE_H_

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "ascend/transform/attr.h"
#include "compat/expr.h"
#include "op/operator.h"
#include "transform/common/attr.h"

namespace tvm::tl::ascend {

constexpr const char *kScheduleYield = "tl.schedule_yield";
constexpr const char *kScheduleCondition = "tl.schedule_condition";
constexpr const char *kScheduleUnitStage = "stage";
constexpr int kUnscheduledStage = -1;

inline ffi::String ScheduleMarkerKey(const RegionStmtNode *region) {
  for (const char *key :
       {attr::kScheduleUnit, attr::kAscendTask, attr::kAscendPerCoreTask,
        attr::kAscendStage, kScheduleCondition, "tilelang_assume",
        attr::kAssumeRequiresRuntimeCheck}) {
    if (region->attrs->dict.count(key))
      return key;
  }
  return "";
}

inline ffi::Any ScheduleMarkerPayload(const RegionStmtNode *region) {
  auto key = ScheduleMarkerKey(region);
  TVM_FFI_CHECK(!key.empty(), ValueError)
      << "Expected an Ascend scheduling marker";
  return region->attrs->dict.at(key);
}

inline bool IsScheduleUnit(const RegionStmtNode *region) {
  return region->attrs->dict.count(attr::kScheduleUnit);
}

inline bool IsScheduleGuardAttribute(const ffi::String &key) {
  return key == "tilelang_assume" || key == attr::kAssumeRequiresRuntimeCheck;
}

inline Stmt SingleScheduleStatement(Stmt statement) {
  while (const auto *sequence = statement.as<SeqStmtNode>()) {
    if (sequence->seq.size() != 1)
      break;
    statement = sequence->seq[0];
  }
  return statement;
}

class ScheduleVarRemapper : public tirx::StmtExprMutator {
public:
  explicit ScheduleVarRemapper(ffi::Map<Var, Var> replacements) {
    for (const auto &[variable, replacement] : replacements)
      VarRemapSet(variable, replacement);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    Expr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    auto mapped = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
                    return Mutate(node->var, mode);
                  }).as_or_throw<ffi::UnchangedOr<Var>>();
    Var variable = std::move(mapped).ValueOrUnchanged(node->var);
    if (!ffi::StructuralEqual()(variable->ty, value->ty))
      variable = variable.CopyWithType(value->ty);
    VarRemapSet(node->var, variable);
    return Bind(variable, value, node->loc);
  }
};

template <typename Value>
inline Value RemapScheduleVariables(const Value &value,
                                    const ffi::Map<Var, Var> &replacements) {
  if (replacements.empty())
    return value;
  auto remapper = ffi::make_object<ScheduleVarRemapper>(replacements);
  return remapper->Mutate(value)
      .ValueOrUnchanged(value)
      .template as_or_throw<Value>();
}

inline ffi::Array<Var> ScheduleDefinitions(const Stmt &statement) {
  if (const auto *binding = statement.as<BindNode>())
    return {binding->var};
  if (const auto *region = statement.as<RegionStmtNode>())
    return region->result_vars;
  ffi::Array<Var> definitions;
  if (const auto *sequence = statement.as<SeqStmtNode>()) {
    for (const Stmt &child : sequence->seq) {
      auto variables = ScheduleDefinitions(child);
      definitions.insert(definitions.end(), variables.begin(), variables.end());
    }
  }
  return definitions;
}

inline RegionStmt MakeScheduleMarker(const ffi::String &key,
                                     const ffi::Any &payload, Stmt body,
                                     Location location = UnknownLoc(),
                                     bool export_definitions = true) {
  BlockAnnotations annotations{{key, payload}};
  ffi::Array<Var> results;
  if (export_definitions)
    results = ScheduleDefinitions(body);
  if (!results.empty()) {
    ffi::Map<Var, Var> replacements;
    ffi::Array<Expr> values;
    for (const Var &variable : results) {
      auto remapper = ffi::make_object<ScheduleVarRemapper>(replacements);
      auto type = remapper->Mutate(variable->ty)
                      .ValueOrUnchanged(variable->ty)
                      .as_or_throw<Type>();
      Var local(variable->name + "_task", type, variable->loc);
      replacements.Set(variable, local);
      values.push_back(variable);
    }
    auto remapper = ffi::make_object<ScheduleVarRemapper>(replacements);
    body = remapper->Mutate(body).ValueOrUnchanged(body);
    values = remapper->Mutate(values)
                 .ValueOrUnchanged(values)
                 .as_or_throw<ffi::Array<Expr>>();
    annotations.Set(kScheduleYield, values);
  }
  return RegionStmt(Op::Get("tl.ascend.schedule_scope"), {}, {},
                    DictAttrs(annotations), SeqStmt(body), results, location);
}

inline Stmt UnwrapScheduleMarker(const RegionStmtNode *region) {
  if (region->result_vars.empty())
    return region->body;
  auto values = region->attrs.GetAttr<ffi::Array<Expr>>(kScheduleYield);
  TVM_FFI_CHECK(values && values->size() == region->result_vars.size(),
                ValueError)
      << "Malformed scheduled result binding";
  ffi::Map<Var, Var> replacements;
  for (size_t index = 0; index < values->size(); ++index) {
    auto variable = (*values)[index].as<Var>();
    TVM_FFI_CHECK(variable.has_value(), ValueError)
        << "Scheduled results must yield lexical definitions";
    replacements.Set(variable.value(), region->result_vars[index]);
  }
  return RemapScheduleVariables(region->body, replacements);
}

inline RegionStmt ReplaceScheduleBody(const RegionStmtNode *region, Stmt body) {
  auto key = ScheduleMarkerKey(region);
  auto result =
      MakeScheduleMarker(key, ScheduleMarkerPayload(region), std::move(body),
                         region->loc, !region->result_vars.empty());
  auto annotations = region->attrs->dict;
  annotations.erase(kScheduleYield);
  for (const auto &[name, value] : result->attrs->dict)
    annotations.Set(name, value);
  result.CopyOnWrite()->attrs = DictAttrs(annotations);
  return result;
}

inline RegionStmt WithSchedulePayload(const RegionStmtNode *region,
                                      const ffi::Any &payload) {
  auto result = ffi::GetRef<RegionStmt>(region);
  auto attributes = region->attrs;
  attributes.CopyOnWrite()->dict.Set(ScheduleMarkerKey(region), payload);
  result.CopyOnWrite()->attrs = attributes;
  return result;
}

inline ffi::Optional<Bind> GetFlatTaskBind(const Stmt &statement) {
  Stmt single = SingleScheduleStatement(statement);
  if (auto binding = single.as<Bind>())
    return binding;
  if (const auto *region = single.as<RegionStmtNode>()) {
    auto key = ScheduleMarkerKey(region);
    if (key == attr::kAscendTask || key == attr::kScheduleUnit)
      return GetFlatTaskBind(UnwrapScheduleMarker(region));
  }
  return std::nullopt;
}

inline Stmt ReplaceFlatTaskBind(const Stmt &statement,
                                const Bind &replacement) {
  Stmt single = SingleScheduleStatement(statement);
  if (single.as<BindNode>())
    return replacement;
  const auto *region = single.as<RegionStmtNode>();
  TVM_FFI_CHECK(region && (ScheduleMarkerKey(region) == attr::kAscendTask ||
                           IsScheduleUnit(region)),
                ValueError)
      << "Expected a flat Task/schedule-unit wrapper around Bind";
  return ReplaceScheduleBody(
      region, ReplaceFlatTaskBind(UnwrapScheduleMarker(region), replacement));
}

inline PrimExpr ScheduleLoopTripCount(const ForNode *loop) {
  PrimExpr zero = MakeConst(loop->var.ty(), 0);
  PrimExpr one = MakeConst(loop->var.ty(), 1);
  PrimExpr forward =
      ceildiv(max(loop->end - loop->begin, zero), max(loop->step, one));
  PrimExpr backward =
      ceildiv(max(loop->begin - loop->end, zero), max(-loop->step, one));
  return Select(loop->step > zero, forward,
                Select(loop->step < zero, backward, zero));
}

inline Range ScheduleLoopRange(const ForNode *loop,
                               sym::Analyzer analyzer = sym::Analyzer()) {
  PrimExpr count = analyzer->Simplify(ScheduleLoopTripCount(loop));
  PrimExpr zero = MakeConst(loop->var.ty(), 0);
  PrimExpr one = MakeConst(loop->var.ty(), 1);
  PrimExpr last = analyzer->Simplify(loop->begin + (count - one) * loop->step);
  PrimExpr lower = analyzer->Simplify(min(loop->begin, last));
  PrimExpr extent = analyzer->Simplify(
      Select(count > zero, max(loop->begin, last) - lower + one, zero));
  return Range::FromMinExtent(lower, extent);
}

using ScheduleUnitMetadata = ffi::Map<ffi::String, ffi::Any>;

inline std::optional<int64_t> GetScheduleUnitInt(const RegionStmtNode *region,
                                                 const char *key) {
  if (!IsScheduleUnit(region))
    return std::nullopt;
  auto metadata =
      ScheduleMarkerPayload(region).as_or_throw<ScheduleUnitMetadata>();
  auto value = metadata.Get(key);
  if (!value)
    return std::nullopt;
  auto integer = GetIntAnnotation(value.value());
  TVM_FFI_CHECK(integer.has_value(), ValueError)
      << "Expected integer schedule metadata: " << key;
  return integer;
}

inline int GetScheduleUnitStage(const RegionStmtNode *region) {
  auto stage = GetScheduleUnitInt(region, kScheduleUnitStage);
  TVM_FFI_CHECK(stage && *stage >= -1 && *stage <= INT32_MAX, ValueError)
      << "Schedule stage must be a constant integer in [-1, INT32_MAX]";
  return static_cast<int>(*stage);
}

} // namespace tvm::tl::ascend

#endif
