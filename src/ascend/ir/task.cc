/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "task.h"
#include "ir/stmt_functor.h"
#include <limits>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl::ascend {
namespace {

inline TileOpArguments DecodeScheduledBindingArguments(const CallObj *call) {
  TVM_FFI_CHECK(call->attrs.has_value(), ValueError)
      << "Scheduled binding metadata requires stage and task";
  auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
  if (attrs->dict.count("operand_count") ||
      attrs->dict.count("annotation_schema") ||
      attrs->dict.count("schema_version"))
    return DecodeTileOpArguments(call);
  // MakeAnnotatedCall stores static stage/task values in native attrs. Task
  // fields owned by other passes may still contain expressions; those use the
  // existing explicit metadata envelope and must not become hidden attr uses.
  TVM_FFI_CHECK(IsStaticAnnotationValue(attrs->dict), ValueError)
      << "Dynamic scheduled binding metadata requires explicit operands";
  auto annotations = attrs->dict;
  annotations.erase("tl.source_annotation_keys");
  return {call->args, annotations};
}

using Annotations = ScheduleUnitMetadata;

void CheckRegionBindings(const RegionStmtObj *region) {
  TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                ValueError)
      << "Ascend task/schedule regions have no body parameters or results";
}

Annotations ReadTaskFields(const ffi::Array<Expr> &args,
                           const DictAttrs &attrs) {
  for (const auto &[key, value] : attrs->dict) {
    TVM_FFI_CHECK(key == "annotations" || key == "annotation_schema",
                  ValueError)
        << "Unknown Ascend task attribute: " << key;
    TVM_FFI_CHECK(IsStaticAnnotationValue(value), TypeError)
        << "Ascend task attrs must be static; put expression-dependent "
           "metadata in explicit operands";
  }
  auto constants = attrs.GetAttr<Annotations>("annotations");
  TVM_FFI_CHECK(constants.has_value(), ValueError)
      << "Ascend task requires an annotations map";
  Annotations result = *constants;
  auto schema = attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema");
  TVM_FFI_CHECK_EQ(args.size(), schema ? 1U : 0U, ValueError)
      << "Ascend task accepts one metadata tvm::Tuple exactly when a schema "
         "exists";
  if (schema) {
    auto dynamic = DecodeAnnotations(args[0].as_or_throw<tvm::Tuple>(), *schema)
                       .cast<Annotations>();
    for (const auto &[key, value] : dynamic) {
      TVM_FFI_CHECK(!result.count(key), ValueError)
          << "Duplicate static and dynamic Ascend task annotation: " << key;
      result.Set(key, value);
    }
  }
  ParseTaskMetadata(result);
  return result;
}

struct EncodedTask {
  ffi::Array<Expr> args;
  DictAttrs attrs;
};

EncodedTask EncodeTask(const Annotations &metadata) {
  ParseTaskMetadata(metadata);
  Annotations constants, dynamic;
  for (const auto &[key, value] : metadata) {
    if (IsStaticAnnotationValue(value))
      constants.Set(key, value);
    else
      dynamic.Set(key, value);
  }
  Annotations fields{{"annotations", constants}};
  ffi::Array<Expr> args;
  if (!dynamic.empty()) {
    auto encoded = EncodeAnnotations(dynamic);
    args.push_back(encoded.payload);
    fields.Set("annotation_schema", encoded.schema);
  }
  return {std::move(args), DictAttrs(std::move(fields))};
}

int ReadStageFields(const ffi::Array<Expr> &args, const DictAttrs &attrs,
                    bool source_scope) {
  TVM_FFI_CHECK(args.empty(), ValueError)
      << "An Ascend stage is static metadata, not an operand";
  // A native region may also carry metadata owned by other passes, including
  // statement templates. Only its body contributes executable statements.
  TVM_FFI_CHECK(attrs->dict.count(kScheduleUnitStage), ValueError)
      << "Ascend stage/schedule unit requires the stage attribute";
  if (source_scope) {
    auto field = attrs->dict.at(kScheduleUnitStage);
    const auto integer = task_metadata_detail::RequireConstantTaskMetadataInt(
        field, kScheduleUnitStage);
    int64_t stage = integer->value.as<int64_t>().value();
    TVM_FFI_CHECK(stage >= 0 && stage <= std::numeric_limits<int>::max(),
                  ValueError)
        << "T.Stage requires a non-negative constant int32 value";
    return static_cast<int>(stage);
  }
  return ReadScheduleStage(attrs->dict);
}

ffi::Array<Var> TaskBodyParams(const CallObj *call) {
  TVM_FFI_CHECK(call->ty_args.empty(), ValueError)
      << "Ascend task/schedule regions have no type arguments";
  auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
  if (call->op.same_as(task_op()) || call->op.same_as(per_core_task_op()))
    ReadTaskFields(call->args, attrs);
  else
    ReadStageFields(call->args, attrs, call->op.same_as(stage_op()));
  return {};
}

void ValidateTaskRegion(const RegionStmtObj *region) {
  CheckRegionBindings(region);
  if (IsSemanticTask(region))
    ReadTaskFields(region->args, region->attrs);
  else
    ReadStageFields(region->args, region->attrs, IsStageScope(region));
}

RegionStmt BuildTask(const Op &op, const Stmt &body,
                     const Annotations &metadata) {
  auto encoded = EncodeTask(metadata);
  return RegionStmt(op, encoded.args, {}, encoded.attrs, SeqStmt(body), {},
                    body->loc);
}

RegionStmt BuildStage(const Op &op, const Stmt &body, int stage) {
  DictAttrs attrs(Annotations{{kScheduleUnitStage, IntImm::Int64(stage)}});
  ReadStageFields({}, attrs, op.same_as(stage_op()));
  return RegionStmt(op, {}, {}, attrs, SeqStmt(body), {}, body->loc);
}

ScheduledBinding DecodeScheduledBinding(const CallObj *call) {
  TVM_FFI_CHECK(IsScheduledBind(call), ValueError)
      << "Expected tl.ascend.scheduled_bind";
  auto decoded = DecodeScheduledBindingArguments(call);
  TVM_FFI_CHECK_EQ(decoded.operands.size(), 1U, ValueError)
      << "Scheduled binding needs exactly one value operand";
  TVM_FFI_CHECK(call->ty_args.empty(), ValueError)
      << "Scheduled binding infers its type from the value operand";
  TVM_FFI_CHECK_EQ(decoded.annotations.size(), 2U, ValueError)
      << "Scheduled binding metadata requires exactly stage and task";
  auto task = decoded.annotations.at("task").cast<ScheduleUnitMetadata>();
  ParseTaskMetadata(task);
  const Expr &value = decoded.operands[0];
  TVM_FFI_CHECK(!value->ty.as<MissingTypeObj>(), TypeError)
      << "Scheduled binding requires a concrete value type";
  auto primitive = value->ty.as<PrimType>();
  TVM_FFI_CHECK(!IsVoidType(value->ty) &&
                    !(primitive && primitive.value().IsVoid()),
                TypeError)
      << "Scheduled binding requires a value, not void";
  TVM_FFI_CHECK(!IsScheduledBind(value.as<CallObj>()), ValueError)
      << "Scheduled binding cannot wrap another scheduled binding";
  return {value, task, ReadScheduleStage(decoded.annotations)};
}

Type InferScheduledBinding(const CallObj *call) {
  return DecodeScheduledBinding(call).value->ty;
}

void ValidateScheduledBinding(const CallObj *call) {
  ReadScheduledBinding(call);
}

Bind BuildScheduledBinding(const Bind &bind, Expr value, int stage,
                           const ScheduleUnitMetadata &task) {
  ScheduleUnitMetadata metadata{
      {kScheduleUnitStage, IntImm(PrimType::Int(64), stage)}, {"task", task}};
  ReadScheduleStage(metadata);
  ParseTaskMetadata(task);
  // Use the expression's location on the call and the statement's location on
  // Bind; a wrapper must not rewrite diagnostic/source provenance.
  Call annotated = MakeAnnotatedCall(value->ty, scheduled_bind_op(), {value},
                                     metadata, value->loc);
  ReadScheduledBinding(annotated.get());
  return Bind(bind->var, annotated, bind->loc);
}

} // namespace

ScheduledBinding ReadScheduledBinding(const CallObj *call) {
  auto binding = DecodeScheduledBinding(call);
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, binding.value->ty), TypeError)
      << "Scheduled binding result type must equal its value type";
  return binding;
}

Bind MakeScheduledBinding(const Bind &bind, int stage,
                          const ScheduleUnitMetadata &task) {
  TVM_FFI_CHECK(!IsScheduledBind(bind.get()), ValueError)
      << "Cannot materialize a binding twice";
  return BuildScheduledBinding(bind, bind->value, stage, task);
}

Bind WithScheduledBindingMetadata(const Bind &bind, int stage,
                                  const ScheduleUnitMetadata &task) {
  auto decoded = ReadScheduledBinding(bind->value.as<CallObj>());
  return BuildScheduledBinding(bind, decoded.value, stage, task);
}

int ReadScheduleStage(const ffi::Any &metadata) {
  auto map = metadata.cast<ScheduleUnitMetadata>();
  auto field = map.Get(kScheduleUnitStage);
  TVM_FFI_CHECK(field.has_value(), ValueError)
      << "Scheduled node is missing integer metadata `stage`";
  const auto integer = task_metadata_detail::RequireConstantTaskMetadataInt(
      field.value(), kScheduleUnitStage);
  int64_t stage = integer->value.as<int64_t>().value();
  TVM_FFI_CHECK(stage >= kUnscheduledStage &&
                    stage <= std::numeric_limits<int>::max(),
                ValueError)
      << "Scheduled stage must be -1 or a non-negative int32 value";
  return static_cast<int>(stage);
}

const Op &task_op() {
  static const Op &op = Op::Get("tl.ascend.task");
  return op;
}
const Op &per_core_task_op() {
  static const Op &op = Op::Get("tl.ascend.per_core_task");
  return op;
}
const Op &schedule_unit_op() {
  static const Op &op = Op::Get("tl.ascend.schedule_unit");
  return op;
}
const Op &stage_op() {
  static const Op &op = Op::Get("tl.ascend.stage");
  return op;
}
const Op &scheduled_bind_op() {
  static const Op &op = Op::Get("tl.ascend.scheduled_bind");
  return op;
}

bool IsTask(const RegionStmtObj *region) {
  return region && region->op.same_as(task_op());
}
bool IsPerCoreTask(const RegionStmtObj *region) {
  return region && region->op.same_as(per_core_task_op());
}
bool IsSemanticTask(const RegionStmtObj *region) {
  return IsTask(region) || IsPerCoreTask(region);
}
bool IsScheduleUnit(const RegionStmtObj *region) {
  return region && region->op.same_as(schedule_unit_op());
}
bool IsStageScope(const RegionStmtObj *region) {
  return region && region->op.same_as(stage_op());
}
Annotations TaskAnnotations(const RegionStmtObj *region) {
  TVM_FFI_CHECK(IsSemanticTask(region), ValueError)
      << "Expected an Ascend Task or PerCoreTask region";
  CheckRegionBindings(region);
  return ReadTaskFields(region->args, region->attrs);
}
RegionStmt MakeTask(const Stmt &body, const Annotations &metadata) {
  return BuildTask(task_op(), body, metadata);
}
RegionStmt MakePerCoreTask(const Stmt &body, const Annotations &metadata) {
  return BuildTask(per_core_task_op(), body, metadata);
}
RegionStmt WithTaskAnnotations(RegionStmt task, const ffi::Any &metadata) {
  TaskAnnotations(task.get());
  auto encoded = EncodeTask(metadata.cast<Annotations>());
  auto *writer = task.CopyOnWrite();
  writer->args = std::move(encoded.args);
  writer->attrs = std::move(encoded.attrs);
  return task;
}
int GetScheduleUnitStage(const RegionStmtObj *region) {
  TVM_FFI_CHECK(IsScheduleUnit(region), ValueError)
      << "Expected an Ascend schedule unit";
  CheckRegionBindings(region);
  return ReadStageFields(region->args, region->attrs, false);
}
int GetScheduleUnitStage(const Stmt &stmt) {
  if (const auto *bind = stmt.as<BindObj>(); IsScheduledBind(bind))
    return ReadScheduledBinding(bind->value.as<CallObj>()).stage;
  return GetScheduleUnitStage(stmt.as<RegionStmtObj>());
}
int GetStageScopeStage(const RegionStmtObj *region) {
  TVM_FFI_CHECK(IsStageScope(region), ValueError) << "Expected T.Stage";
  CheckRegionBindings(region);
  return ReadStageFields(region->args, region->attrs, true);
}
RegionStmt MakeScheduleUnit(const Stmt &body, int stage) {
  return BuildStage(schedule_unit_op(), body, stage);
}
RegionStmt WithScheduleUnitStage(RegionStmt unit, int stage) {
  GetScheduleUnitStage(unit.get());
  auto metadata = unit->attrs->dict;
  metadata.Set(kScheduleUnitStage, IntImm::Int64(stage));
  ReadScheduleStage(metadata);
  unit.CopyOnWrite()->attrs = DictAttrs(metadata);
  return unit;
}
RegionStmt MakeStageScope(const Stmt &body, int stage) {
  return BuildStage(stage_op(), body, stage);
}

std::optional<int> GetTaskSubcore(const Stmt &stmt) {
  Stmt single = SingleStatement(stmt);
  ffi::Map<ffi::String, ffi::Any> metadata;
  if (const auto *bind = single.as<BindObj>()) {
    const auto *call = bind->value.as<CallObj>();
    if (!IsScheduledBind(call))
      return std::nullopt;
    metadata = task_metadata_detail::CopyTaskMetadata(
        DecodeScheduledBindingArguments(call).annotations.at("task"));
  } else {
    const auto *marker = single.as<RegionStmtObj>();
    if (!IsSemanticTask(marker))
      return std::nullopt;
    metadata = TaskAnnotations(marker);
  }
  if (auto value = metadata.Get("subcore")) {
    int64_t sid =
        task_metadata_detail::RequireConstantTaskMetadataInt(*value, "subcore")
            ->value.as<int64_t>()
            .value();
    TVM_FFI_CHECK(sid == 0 || sid == 1, ValueError)
        << "T.Task subcore must be 0 or 1";
    return static_cast<int>(sid);
  }
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char *name : {"tl.ascend.task", "tl.ascend.per_core_task",
                           "tl.ascend.schedule_unit", "tl.ascend.stage"})
    OpDef(name, "An Ascend task or static schedule scope.")
        .signature(sig::var_args<tvm::Tuple>("metadata"),
                   sig::call_attrs<DictAttrsObj>())
        .set_native_fn_attr<&TaskBodyParams>(op_attr::kRegionGetBodyParams)
        .set_native_fn_attr<&ValidateTaskRegion>(op_attr::kRegionValidate);
  ffi::reflection::GlobalDef().def(
      "tl.EncodeAscendTask", [](const Annotations &metadata) {
        auto encoded = EncodeTask(metadata);
        return ffi::Array<ffi::Any>{encoded.args, encoded.attrs};
      });
  OpDef("tl.ascend.scheduled_bind",
        "One native Bind value evaluated by an Ascend scheduled task. "
        "Must be consumed by LowerScheduledTIR before code generation.")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateScheduledBinding>())
      .signature(sig::var_args<Expr>(
                     "operands",
                     "One value and an optional explicit metadata tvm::Tuple."),
                 sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&InferScheduledBinding>(tvm::op_attr::kInferType)
      // Even a pure value defines one scheduled task. Generic simplifiers
      // must not eliminate, duplicate, or move it across task boundaries.
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

} // namespace tvm::tl::ascend
