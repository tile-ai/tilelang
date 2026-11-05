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

#include "native_task_ir.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl::ascend {
namespace {

NativeScheduledBinding DecodeScheduledBinding(const CallObj *call) {
  TVM_FFI_CHECK(IsScheduledBind(call), ValueError)
      << "Expected tl.ascend.scheduled_bind";
  auto decoded = DecodeTileOpArguments(call);
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
  Call annotated = MakeAnnotatedCall(value->ty, Op::Get(kScheduledBind),
                                     {value}, metadata, value->loc);
  ReadScheduledBinding(annotated.get());
  return Bind(bind->var, annotated, bind->loc);
}

} // namespace

NativeScheduledBinding ReadScheduledBinding(const CallObj *call) {
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

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef(kScheduledBind,
        "One native Bind value evaluated by an Ascend scheduled task. "
        "Must be consumed by LowerScheduledTIR before code generation.")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateScheduledBinding>())
      .signature(sig::arg<Expr>("value"), sig::arg<tvm::Tuple>("metadata"),
                 sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&InferScheduledBinding>(tvm::op_attr::kInferType)
      // Even a pure value defines one scheduled task. Generic simplifiers
      // must not eliminate, duplicate, or move it across task boundaries.
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

} // namespace tvm::tl::ascend
