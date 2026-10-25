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

/*!
 * \file lower_scheduled_tir.cc
 * \brief Lower scheduled TIR into final Ascend core TIR.
 */

#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/multi_buffer.h"
#include "./auto_schedule/scheduled_tir.h"
#include "buffer_alias.h"
#include "compat/visit.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm::tl {

using namespace tirx;
using namespace ascend;

namespace {

using VariableSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

VariableSet Definitions(const ffi::ObjectRef &value) {
  VariableSet variables;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      value, [&](const Var &variable, TVMFFIDefRegionKind kind) {
        if (kind != kTVMFFIDefRegionKindNone)
          variables.insert(variable);
        return ffi::WalkResult::Advance();
      });
  return variables;
}

class TaskMarkerStripper : public NativeMutator {
public:
  Stmt VisitStmt_(const RegionStmtNode *node) final {
    auto key = ScheduleMarkerKey(node);
    if (key == attr::kAscendTask || key == attr::kAscendPerCoreTask ||
        key == attr::kScheduleUnit)
      return VisitStmt(UnwrapScheduleMarker(node));
    if (key == kScheduleCondition)
      return If(ScheduleMarkerPayload(node).as_or_throw<PrimExpr>(),
                VisitStmt(UnwrapScheduleMarker(node)), std::nullopt, node->loc);
    return NativeMutator::VisitStmt_(node);
  }
};

class ValueSubstituter : public NativeMutator {
public:
  explicit ValueSubstituter(const ffi::Map<Var, Expr> &values) {
    for (const auto &[variable, value] : values)
      VarRemapSet(variable, value);
  }

  Stmt VisitStmt_(const BindNode *node) final {
    Expr value = VisitExpr(node->value);
    auto mapped = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
                    return Mutate(node->var);
                  }).as_or_throw<ffi::UnchangedOr<Var>>();
    Var variable = std::move(mapped).ValueOrUnchanged(node->var);
    if (!ffi::StructuralEqual()(variable->ty, value->ty))
      variable = variable.CopyWithType(value->ty);
    VarRemapSet(node->var, variable);
    return Bind(variable, value, node->loc);
  }
};

class GuardedValueWriter : public NativeMutator {
public:
  explicit GuardedValueWriter(ffi::Map<Var, TensorVar> cells)
      : cells_(std::move(cells)) {}

  Stmt VisitStmt_(const BindNode *node) final {
    auto cell = cells_.Get(node->var);
    if (!cell)
      return NativeMutator::VisitStmt_(node);
    Expr value = VisitExpr(node->value);
    ffi::Array<Stmt> statements;
    if (auto type = node->var->ty.as<TensorType>()) {
      const auto *call = value.as<CallNode>();
      TVM_FFI_CHECK(!call || !call->op.same_as(tirx::alloc_tensor_op()),
                    ValueError)
          << "A dynamic tensor allocation cannot escape a guarded scheduled "
             "scope";
      TensorVar temporary(node->var->name + "_guarded", type.value());
      statements.push_back(Bind(temporary, value, node->loc));
      value = temporary.data();
    }
    if (value->ty.as<PtrTypeNode>())
      value = Call(PrimType::UInt(64), tirx::reinterpret_op(), {value}, {},
                   {PrimType::UInt(64)}, node->loc);
    statements.push_back(TensorStore(cell.value(), {IntImm::Int32(0)},
                                     value.as_or_throw<PrimExpr>(), node->loc));
    return SeqStmt(statements);
  }

private:
  ffi::Map<Var, TensorVar> cells_;
};

ffi::Map<ffi::String, ffi::Any>
FilterLoopAnnotations(ffi::Map<ffi::String, ffi::Any> annotations) {
  for (const char *key :
       {"num_stages", "enable_offset", "conflict_hint", kMultiBufferEligible,
        kMultiBufferCounterMap, kStorageEpochGuardMap})
    annotations.erase(key);
  return annotations;
}

DictAttrs FilterStorageMetadata(const DictAttrs &attributes,
                                const VariableSet &defined) {
  auto output = attributes->dict;
  for (const char *key :
       {attr::kLayoutMap, kBufferAliasMap, kAivStorageMap, kBufferVersionsMap,
        kManualMultiBuffer, kBufferVersionMode, kL0SFBindings}) {
    auto value = output.Get(key);
    if (!value)
      continue;
    ffi::Map<Var, ffi::Any> filtered;
    for (const auto &[variable, element] :
         value->as_or_throw<ffi::Map<Var, ffi::Any>>()) {
      if (!defined.count(variable))
        continue;
      if (std::string(key) == kBufferAliasMap) {
        ffi::Array<Var> compatible;
        for (const Var &other : element.as_or_throw<ffi::Array<Var>>())
          if (defined.count(other))
            compatible.push_back(other);
        filtered.Set(variable, compatible);
      } else {
        filtered.Set(variable, element);
      }
    }
    if (filtered.empty())
      output.erase(key);
    else
      output.Set(key, filtered);
  }
  output.erase(kSplitSubcores);
  return DictAttrs(output);
}

struct LoweredNode {
  ffi::Array<Stmt> prefix;
  Stmt body{SeqStmt(ffi::Array<Stmt>{})};
  ffi::Array<Stmt> suffix;
};

class CoreLowerer {
public:
  CoreLowerer(CoreMask core, PrimVar sid,
              const std::unordered_set<const TaskNode *> &hoisted)
      : core_(core), sid_(std::move(sid)), hoisted_(hoisted) {}

  bool HasCore(const std::vector<std::shared_ptr<IRStructure>> &nodes,
               bool primary = false) const {
    for (const auto &node : nodes) {
      if (node->IsTask()) {
        auto *task = static_cast<TaskNode *>(node.get());
        if (!hoisted_.count(task) && (task->GetCoreMask() & core_) &&
            (!primary || IsCoreLocalMask(task->GetCoreMask())))
          return true;
      } else if (HasCore(static_cast<ControlNode *>(node.get())->children,
                         primary)) {
        return true;
      }
    }
    return false;
  }

  Stmt Lower(const std::vector<std::shared_ptr<IRStructure>> &nodes) {
    RegisterReaders(nodes);
    Stmt body = LowerList(nodes, nullptr);
    auto substituter = ffi::make_object<ValueSubstituter>(values_);
    return substituter->VisitStmt(body);
  }

private:
  void RegisterReaders(const std::vector<std::shared_ptr<IRStructure>> &nodes) {
    for (const auto &node : nodes) {
      const TaskNode *task;
      if (node->IsTask()) {
        task = static_cast<const TaskNode *>(node.get());
      } else {
        const auto *control = static_cast<const ControlNode *>(node.get());
        task = control->task.get();
        RegisterReaders(control->children);
      }
      for (const Var &variable : task->GetReadVars())
        readers_[variable].insert(task);
      for (const TensorRegion &region : task->GetReadRegions())
        readers_[region->source.as_or_throw<Var>()].insert(task);
      for (const TensorRegion &region : task->GetWriteRegions())
        readers_[region->source.as_or_throw<Var>()].insert(task);
    }
  }

  bool EscapesTask(const Var &variable, const TaskNode *task) const {
    auto readers = readers_.find(variable);
    if (readers == readers_.end())
      return false;
    for (const TaskNode *reader : readers->second)
      if (reader != task)
        return true;
    return false;
  }

  bool Selected(const IRStructure *node) const {
    if (node->IsTask()) {
      const auto *task = static_cast<const TaskNode *>(node);
      return !hoisted_.count(task) && (task->GetCoreMask() & core_) != 0;
    }
    return HasCore(static_cast<const ControlNode *>(node)->children);
  }

  Stmt WrapGuards(const IRStructure *node, Stmt body) const {
    for (auto iterator = node->GetGuards().rbegin();
         iterator != node->GetGuards().rend(); ++iterator) {
      if ((*iterator)->IsCondition()) {
        auto *condition = static_cast<const ConditionGuard *>(iterator->get());
        body = If(condition->condition, body);
      } else {
        auto *attribute = static_cast<const AttributeGuard *>(iterator->get());
        body = RegionStmt(Op::Get("tl.scope"), {}, {},
                          DictAttrs({{attribute->key, attribute->node}}),
                          SeqStmt(body), {}, attribute->loc);
      }
    }
    return body;
  }

  LoweredNode LowerTask(const TaskNode *task, bool staged) {
    auto stripper = ffi::make_object<TaskMarkerStripper>();
    LoweredNode result;
    result.body = stripper->VisitStmt(task->stmt);
    CoreMask selected = task->GetCoreMask() & core_;
    bool subcore = core_ == kCoreVector && IsVectorCore(selected);
    bool guarded = staged || task->HasConditions() || subcore;
    if (guarded) {
      ffi::Map<Var, TensorVar> cells;
      for (const Var &variable : ScheduleDefinitions(result.body)) {
        if (!EscapesTask(variable, task))
          continue;
        Type type = variable->ty;
        auto primitive = type.as<PrimType>();
        bool pointer = type.as<PtrTypeNode>() || type.as<TensorTypeNode>();
        TVM_FFI_CHECK(primitive || pointer, TypeError)
            << "Unsupported guarded scheduled value type: " << type;
        PrimType dtype = primitive.value_or(PrimType::UInt(64));
        TensorVar cell(variable->name + "_value",
                       TensorType({IntImm::Int32(1)}, dtype,
                                  runtime::MemoryScope("local.var")));
        cells.Set(variable, cell);
        result.prefix.push_back(
            Bind(cell, Call(cell.ty(), tirx::alloc_tensor_op(), {}, {},
                            {cell.ty()})));
        result.prefix.push_back(
            TensorStore(cell, {IntImm::Int32(0)}, MakeConst(dtype, 0)));
        Expr value = TensorLoad(cell, {IntImm::Int32(0)});
        if (pointer) {
          Type pointer_type = type;
          if (auto tensor = type.as<TensorType>())
            pointer_type =
                PtrType(tensor.value()->dtype, tensor.value()->scope);
          value = Call(pointer_type, tirx::reinterpret_op(), {value}, {},
                       {pointer_type});
        }
        if (type.as<TensorTypeNode>()) {
          result.suffix.push_back(
              Bind(variable,
                   Call(type, tirx::decl_tensor_op(), {value}, {}, {type})));
        } else {
          values_.Set(variable, value);
        }
      }
      auto writer = ffi::make_object<GuardedValueWriter>(cells);
      result.body = writer->VisitStmt(result.body);
    }
    result.body = WrapGuards(task, result.body);
    if (subcore)
      result.body = If(sid_ == VectorSid(selected), result.body);
    return result;
  }

  Stmt LowerList(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                 const ForNode *loop) {
    int min_stage = std::numeric_limits<int>::max();
    int max_stage = std::numeric_limits<int>::min();
    for (const auto &node : nodes) {
      if (!Selected(node.get()))
        continue;
      min_stage = std::min(min_stage, node->GetStage());
      max_stage = std::max(max_stage, node->GetStage());
    }
    bool staged = loop && min_stage != max_stage;
    ffi::Array<Stmt> statements;
    for (const auto &node : nodes) {
      if (!Selected(node.get()))
        continue;
      LoweredNode lowered;
      if (node->IsTask()) {
        lowered = LowerTask(static_cast<TaskNode *>(node.get()), staged);
      } else {
        const auto *control = static_cast<const ControlNode *>(node.get());
        lowered.body = WrapGuards(
            control, LowerList(control->children, control->control.get()));
      }
      ffi::Map<PrimVar, PrimExpr> replacement;
      PrimExpr valid = IntImm::Bool(true);
      if (staged) {
        PrimExpr index =
            loop->var - loop->step * (node->GetStage() - min_stage);
        replacement.Set(loop->var, index);
        valid = (loop->step > 0 && index >= loop->begin && index < loop->end) ||
                (loop->step < 0 && index <= loop->begin && index > loop->end);
      }
      for (const Stmt &prefix : lowered.prefix)
        statements.push_back(
            replacement.empty() ? prefix : Substitute(prefix, replacement));
      Stmt body = replacement.empty() ? lowered.body
                                      : Substitute(lowered.body, replacement);
      if (staged)
        body = If(valid, body);
      statements.push_back(body);
      for (const Stmt &suffix : lowered.suffix)
        statements.push_back(
            replacement.empty() ? suffix : Substitute(suffix, replacement));
    }
    Stmt body = SeqStmt(statements);
    if (!loop)
      return body;
    PrimExpr end = loop->end;
    if (staged)
      end = end + loop->step * (max_stage - min_stage);
    return For(loop->var, loop->begin, end, loop->step, loop->kind, body,
               FilterLoopAnnotations(loop->annotations), loop->loc);
  }

  CoreMask core_;
  PrimVar sid_;
  const std::unordered_set<const TaskNode *> &hoisted_;
  ffi::Map<Var, Expr> values_;
  std::unordered_map<Var, std::unordered_set<const TaskNode *>, ObjectPtrHash,
                     ObjectPtrEqual>
      readers_;
};

class MixedKernelSidRelocator : public NativeMutator {
public:
  Stmt VisitStmt_(const RegionStmtNode *node) final {
    bool launch = node->op.same_as(tirx::launch_thread_op()) ||
                  node->op.same_as(Op::Get("tl.launch_grid"));
    if (launch && node->args.size() == 2 &&
        node->args[0].as_or_throw<StringImm>()->value == "cthread" &&
        inside_vector_ == 0) {
      launches_.push_back(ffi::GetRef<RegionStmt>(node));
      Stmt body = VisitStmt(node->body);
      launches_.pop_back();
      return body;
    }
    bool vector = KernelRegionName(node) == "VECTOR";
    if (vector)
      ++inside_vector_;
    auto region = NativeMutator::VisitStmt_(node).as_or_throw<RegionStmt>();
    if (vector)
      --inside_vector_;
    if (!vector || launches_.empty())
      return region;
    const auto &outer_launch = launches_.back();
    Var variable = outer_launch->body_params[0].CopyWithName(
        outer_launch->body_params[0]->name);
    Stmt body = RemapScheduleVariables(
        region->body, {{outer_launch->body_params[0], variable}});
    region.CopyOnWrite()->body =
        SeqStmt(RegionStmt(tirx::launch_thread_op(), outer_launch->args,
                           {variable}, DictAttrs(), SeqStmt(body)));
    return region;
  }

private:
  int inside_vector_{0};
  std::vector<RegionStmt> launches_;
};

RegionStmt LowerKernel(const TilelangKernelContext &context) {
  auto scheduled = DecodeScheduledTIR(context.root, context.outer_ctx);
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(scheduled.tree, tasks);
  VariableSet local_definitions;
  for (const TaskNode *task : tasks) {
    auto definitions = Definitions(task->stmt);
    local_definitions.insert(definitions.begin(), definitions.end());
    for (const Var &variable : task->GetWriteVars())
      local_definitions.insert(variable);
  }
  tl::PostOrderVisit(context.root->body, [&](const ObjectRef &value) {
    if (const auto *loop = value.as<ForNode>())
      local_definitions.insert(loop->var);
  });
  if (context.outer_sid)
    local_definitions.insert(context.outer_sid.value());
  std::unordered_set<const TaskNode *> hoisted;
  ffi::Array<Stmt> prefix;
  VariableSet root_definitions;
  for (const TaskNode *task : tasks) {
    auto binding = GetFlatTaskBind(task->stmt);
    if (!binding)
      continue;
    const auto *call = binding.value()->value.as<CallNode>();
    if (!call || !call->op.same_as(tirx::alloc_tensor_op()))
      continue;
    bool independent = true;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        binding.value()->var->ty, [&](const Var &variable) {
          independent &= !local_definitions.count(variable);
          return ffi::WalkResult::Advance();
        });
    if (!independent)
      continue;
    hoisted.insert(task);
    prefix.push_back(binding.value());
    root_definitions.insert(binding.value()->var);
  }
  PrimVar sid = context.outer_sid.value_or(PrimVar("sid", PrimType::Int(32)));
  int vector_count = scheduled.metadata.num_aiv_subcores.value_or(2);
  CoreLowerer vector(VectorCoreMask(vector_count), sid, hoisted);
  CoreLowerer cube(kCoreCube, sid, hoisted);
  bool has_vector = vector.HasCore(scheduled.tree, true);
  bool has_cube = cube.HasCore(scheduled.tree, true);
  if (!has_vector && !has_cube)
    has_vector = vector.HasCore(scheduled.tree);
  auto attributes = context.root->attrs;
  auto emit = [&](CoreLowerer &lowerer, const char *name) {
    Stmt body = lowerer.Lower(scheduled.tree);
    auto definitions = Definitions(body);
    auto attrs = FilterStorageMetadata(attributes, definitions)->dict;
    attrs.Set("name_hint", ffi::String(name));
    attrs.erase(kVectorCount);
    bool is_vector = std::string(name) == "VECTOR";
    bool needs_sid = false;
    for (const Var &variable : tvm::UndefinedVars(body))
      needs_sid |= variable.same_as(sid);
    if (is_vector && (has_cube || needs_sid || context.outer_sid)) {
      attrs.Set(kVectorCount, vector_count);
      if (!context.outer_sid)
        body = RegionStmt(tirx::launch_thread_op(),
                          {StringImm("cthread"), IntImm::Int32(vector_count)},
                          {sid}, DictAttrs(), SeqStmt(body));
    }
    auto region = RegionStmt(Op::Get("tl.scope"), {}, {}, DictAttrs(attrs),
                             SeqStmt(body));
    auto renewed = tirx::RenewDef(tirx::Function({}, SeqStmt(region)));
    prefix.push_back(renewed->body.value());
  };
  if (has_cube)
    emit(cube, "CUBE");
  if (has_vector)
    emit(vector, "VECTOR");
  auto root = context.root;
  auto *rewritten = root.CopyOnWrite();
  rewritten->body = SeqStmt(prefix);
  rewritten->attrs = FilterStorageMetadata(attributes, root_definitions);
  return root;
}

} // namespace

tvm::transform::Pass LowerScheduledTIR() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    function = RewriteTilelangKernels(std::move(function), "LowerScheduledTIR",
                                      LowerKernel);
    auto relocator = ffi::make_object<MixedKernelSidRelocator>();
    function.CopyOnWrite()->body =
        SeqStmt(relocator->VisitStmt(function->body.value()));
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerScheduledTIR");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerScheduledTIR",
                                   LowerScheduledTIR);
}

} // namespace tvm::tl
