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

#include "scheduled_tir.h"
#include "task_analysis.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/analysis.h>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl::ascend {
using namespace tirx;
using namespace prim;
using ffi::GetRef;

namespace {

bool IsKernelMetadataKey(const ffi::String &key) {
  return key == kBufferVersionsMap || key == kManualMultiBuffer ||
         key == kBufferVersionMode || key == kUnlimitMemoryScopes ||
         key == kBufferAliasMap || key == kVectorCount ||
         key == kRootConflictHints;
}

bool IsStorageMapMetadataKey(const ffi::String &key) {
  return key == kBufferVersionsMap || key == kManualMultiBuffer ||
         key == kBufferVersionMode;
}

using StorageMetadataEntries = ffi::Map<Var, ffi::Any>;

int64_t ReadMetadataInteger(const ffi::Any &value, const char *description) {
  if (auto integer = value.try_cast<int64_t>())
    return *integer;
  auto expr = value.cast<PrimExpr>();
  auto integer = TryConstInt(expr);
  TVM_FFI_CHECK(integer && expr.ty().IsScalar() &&
                    !expr.ty().MatchesCode(kDLBool),
                ValueError)
      << description << " must be a constant scalar integer";
  return *integer;
}

void ValidateStorageKey(const Var &key) {
  TVM_FFI_CHECK(key->ty.as<TensorTypeObj>() || key->ty.as<PtrTypeObj>(),
                ValueError)
      << "Scheduled storage metadata requires a tensor or pointer Var: " << key;
}

BufferVersionMap ReadVersions(const ffi::Any &value) {
  BufferVersionMap result;
  for (const auto &[key_any, count_any] :
       value.cast<ffi::Map<ffi::Any, ffi::Any>>()) {
    Var key = key_any.cast<Var>();
    ValidateStorageKey(key);
    int64_t count = ReadMetadataInteger(count_any, "Storage version count");
    TVM_FFI_CHECK(count > 0 && count <= std::numeric_limits<int>::max(),
                  ValueError)
        << "Storage version count must be a positive int32 value";
    result.Set(key, static_cast<int>(count));
  }
  return result;
}

template <typename Value>
void MergeStorageMetadata(ffi::Map<Var, Value> *output,
                          const ffi::Map<Var, Value> &entries,
                          const char *key) {
  for (const auto &[storage, value] : entries) {
    auto previous = output->Get(storage);
    TVM_FFI_CHECK(!previous || *previous == value, ValueError)
        << "Conflicting " << key << " for storage " << storage;
    output->Set(storage, value);
  }
}

void ValidateMetadata(const ScheduledTIRMetadata &metadata) {
  for (const auto *map :
       {&metadata.buffer_versions, &metadata.manual_buffer_versions})
    for (const auto &[storage, count] : *map) {
      ValidateStorageKey(storage);
      TVM_FFI_CHECK_GT(count, 0, ValueError);
    }
  for (const auto &[storage, mode] : metadata.buffer_version_modes) {
    ValidateStorageKey(storage);
    TVM_FFI_CHECK(mode == "auto" || mode == "iteration" || mode == "counter",
                  ValueError)
        << "Buffer version mode must be auto, iteration or counter";
  }
  if (metadata.has_buffer_aliases) {
    for (const auto &[storage, aliases] : metadata.buffer_aliases) {
      ValidateStorageKey(storage);
      for (const Var &alias : aliases)
        ValidateStorageKey(alias);
    }
    ValidateBufferAliasMap(metadata.buffer_aliases);
  }
  if (metadata.num_aiv_subcores)
    TVM_FFI_CHECK(*metadata.num_aiv_subcores == 1 ||
                      *metadata.num_aiv_subcores == 2,
                  ValueError)
        << "Mixed-kernel vector_count must be 1 or 2";
}

void ReadMetadata(const ScheduleUnitMetadata &fields,
                  ScheduledTIRMetadata *out) {
  if (auto value = fields.Get(kBufferVersionsMap))
    MergeStorageMetadata(&out->buffer_versions, ReadVersions(*value),
                         kBufferVersionsMap);
  if (auto value = fields.Get(kManualMultiBuffer))
    MergeStorageMetadata(&out->manual_buffer_versions, ReadVersions(*value),
                         kManualMultiBuffer);
  if (auto value = fields.Get(kBufferVersionMode))
    MergeStorageMetadata(&out->buffer_version_modes,
                         value->cast<BufferVersionModeMap>(),
                         kBufferVersionMode);
  if (auto value = fields.Get(kUnlimitMemoryScopes))
    out->unlimit_memory_scopes = value->cast<ffi::Array<ffi::String>>();
  if (auto value = fields.Get(kBufferAliasMap)) {
    out->buffer_aliases = value->cast<BufferAliasMap>();
    out->has_buffer_aliases = true;
  }
  if (auto value = fields.Get(kVectorCount)) {
    int64_t count = ReadMetadataInteger(*value, "Mixed-kernel vector_count");
    TVM_FFI_CHECK(count == 1 || count == 2, ValueError);
    out->num_aiv_subcores = static_cast<int>(count);
  }
  if (auto value = fields.Get(kRootConflictHints))
    for (const auto &entry : value->cast<ffi::Array<ffi::Any>>())
      out->root_conflict_hints.push_back(entry);
  ValidateMetadata(*out);
}

ScheduleUnitMetadata WriteMetadata(const ScheduledTIRMetadata &metadata) {
  ValidateMetadata(metadata);
  ScheduleUnitMetadata fields;
  if (!metadata.buffer_versions.empty())
    fields.Set(kBufferVersionsMap, metadata.buffer_versions);
  if (!metadata.manual_buffer_versions.empty())
    fields.Set(kManualMultiBuffer, metadata.manual_buffer_versions);
  if (!metadata.buffer_version_modes.empty())
    fields.Set(kBufferVersionMode, metadata.buffer_version_modes);
  if (!metadata.unlimit_memory_scopes.empty())
    fields.Set(kUnlimitMemoryScopes, metadata.unlimit_memory_scopes);
  if (metadata.has_buffer_aliases)
    fields.Set(kBufferAliasMap, metadata.buffer_aliases);
  if (metadata.num_aiv_subcores)
    fields.Set(kVectorCount, IntImm::Int64(*metadata.num_aiv_subcores));
  return fields;
}

NativeTaskAttribute RequireSemanticTaskMarker(const Stmt &stmt) {
  auto task = ReadTaskAttribute(stmt);
  TVM_FFI_CHECK(task && IsSemanticTaskKey(task->key), ValueError)
      << "Scheduled leaf must contain exactly one native T.Task/T.PerCoreTask";
  return *task;
}

// New storage entries may refer to definitions inside a loop or branch. Keep
// each operand-backed entry immediately after its own native definition. Only
// parameters and other outer definitions may remain for the kernel tail.
class StorageMetadataEmitter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit StorageMetadataEmitter(ScheduleUnitMetadata metadata)
      : remaining(std::move(metadata)) {}

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    Array<Stmt> statements;
    for (const Stmt &statement : op->seq) {
      Stmt rewritten = Mutate(statement, mode).ValueOrUnchanged(statement);
      statements.push_back(rewritten);
      const auto *bind = rewritten.as<BindObj>();
      if (!bind)
        continue;
      ScheduleUnitMetadata local;
      auto original = remaining;
      for (const auto &[key, value] : original) {
        if (!IsStorageMapMetadataKey(key))
          continue;
        auto entries = value.cast<StorageMetadataEntries>();
        if (auto entry = entries.Get(bind->var)) {
          local.Set(key, StorageMetadataEntries{{bind->var, *entry}});
          entries.erase(bind->var);
          if (entries.empty())
            remaining.erase(key);
          else
            remaining.Set(key, entries);
        }
      }
      if (!local.empty())
        statements.push_back(MakeNativeAnnotation(local, bind->loc));
    }
    return SeqStmt(statements, op->loc);
  }

  ScheduleUnitMetadata remaining;
};

class ScheduledTIREncoder {
public:
  ScheduledTIREncoder(ScheduleUnitMetadata metadata, bool preserve_root_hints)
      : metadata_(std::move(metadata)),
        preserve_root_hints_(preserve_root_hints) {}

  ScheduleUnitMetadata RewriteFields(ScheduleUnitMetadata fields) {
    auto original = fields;
    for (const auto &[key, original_value] : original) {
      if (!IsKernelMetadataKey(key))
        continue;
      seen_.insert(key);
      if (key == kRootConflictHints && preserve_root_hints_)
        continue;
      if (IsStorageMapMetadataKey(key)) {
        // Each carrier can name only the storage definitions visible there.
        // Update its own entries; new keys follow their storage definitions.
        StorageMetadataEntries rewritten;
        if (auto value = metadata_.Get(key)) {
          auto current = value->cast<StorageMetadataEntries>();
          for (const auto &[storage, unused] :
               original_value.cast<StorageMetadataEntries>()) {
            if (auto update = current.Get(storage)) {
              rewritten.Set(storage, *update);
              emitted_storage_[key].insert(storage.get());
            }
          }
        }
        if (rewritten.empty())
          fields.erase(key);
        else
          fields.Set(key, rewritten);
        continue;
      }
      if (auto value = metadata_.Get(key))
        fields.Set(key, *value);
      else
        fields.erase(key);
    }
    return fields;
  }

  SeqStmt Encode(const IRStructure::ChildList &nodes, Location loc) {
    auto body = EncodeList(nodes, loc);
    ScheduleUnitMetadata missing;
    for (const auto &[key, value] : metadata_) {
      if (IsStorageMapMetadataKey(key)) {
        StorageMetadataEntries entries;
        for (const auto &[storage, entry] :
             value.cast<StorageMetadataEntries>())
          if (!emitted_storage_[key].count(storage.get()))
            entries.Set(storage, entry);
        if (!entries.empty())
          missing.Set(key, entries);
      } else if (!seen_.count(key)) {
        missing.Set(key, value);
      }
    }
    if (!missing.empty()) {
      auto emitter = ffi::make_object<StorageMetadataEmitter>(missing);
      body = emitter->Mutate(body).ValueOrUnchanged(body);
      if (!emitter->remaining.empty()) {
        auto statements = body->seq;
        statements.push_back(MakeNativeAnnotation(emitter->remaining, loc));
        body = SeqStmt(statements, loc);
      }
    }
    return body;
  }

private:
  SeqStmt EncodeList(const IRStructure::ChildList &nodes, Location loc) {
    Array<Stmt> statements;
    for (const auto &node : nodes)
      statements.push_back(EncodeNode(node.get()));
    return SeqStmt(statements, loc);
  }

  ScheduleUnitMetadata TaskMetadataFor(const TaskNode *task,
                                       const ffi::Any &original) {
    TaskMetadata update;
    if (task->GetCoreMask() != kCoreUnassigned)
      update.core_mask = task->GetCoreMask();
    if (task->GetII() > 0) {
      update.ii = task->GetII();
      update.latency = task->GetLatency();
    }
    return MergeTaskMetadata(update, original).cast<ScheduleUnitMetadata>();
  }

  Stmt EncodeNode(const IRStructure *node) {
    if (node->IsTask()) {
      const auto *task = static_cast<const TaskNode *>(node);
      if (const auto *bind = task->stmt.as<BindObj>(); IsScheduledBind(bind)) {
        TVM_FFI_CHECK(!task->HasGuards(), ValueError)
            << "Native scheduled Bind must retain its lexical scope";
        auto original = ReadScheduledBinding(bind->value.as<CallObj>());
        return WithScheduledBindingMetadata(
            GetRef<Bind>(bind), task->GetStage(),
            TaskMetadataFor(task, original.task));
      }
      auto semantic = RequireSemanticTaskMarker(task->stmt);
      Stmt body = semantic.WithNode(TaskMetadataFor(task, semantic.node));
      body = WrapScheduledGuards(body, task->GetGuards());
      return MakeScheduleUnit(body, task->GetStage());
    }
    if (node->IsControl()) {
      const auto *control = static_cast<const ControlNode *>(node);
      For loop = control->control;
      loop.CopyOnWrite()->body = EncodeList(control->children, loop->body->loc);
      return MakeScheduleUnit(WrapScheduledGuards(loop, control->GetGuards()),
                              control->GetStage());
    }
    const auto *scope = static_cast<const ScopeNode *>(node);
    if (const auto *branch = scope->shell.as<IfObj>()) {
      If result = GetRef<If>(branch);
      auto *writer = result.CopyOnWrite();
      writer->then_case = EncodeList(scope->children, branch->then_case->loc);
      if (scope->has_else)
        writer->else_case =
            EncodeList(scope->else_children, branch->else_case.value()->loc);
      return result;
    }
    if (auto attribute = ReadTaskAttribute(scope->shell))
      return attribute->WithBody(
          EncodeList(scope->children, attribute->region->body->loc));
    TVM_FFI_CHECK(IsTaskMetadataStatement(scope->shell), ValueError)
        << "Unknown lexical schedule node";
    const auto *call = scope->shell.as<EvaluateObj>()->value.as<CallObj>();
    auto original = ReadNativeAnnotations(call);
    auto fields = RewriteFields(original);
    if (ffi::StructuralEqual()(fields, original))
      return scope->shell;
    return MakeNativeAnnotation(fields, scope->shell->loc);
  }

  ScheduleUnitMetadata metadata_;
  std::unordered_set<ffi::String> seen_;
  std::unordered_map<ffi::String, std::unordered_set<const VarObj *>>
      emitted_storage_;
  bool preserve_root_hints_;
};

class ScheduledTIRDecoder : public TaskAwareConstrVisitor {
public:
  using TaskAwareConstrVisitor::Visit;
  using TaskAwareConstrVisitor::Visit_;
  ScheduledTIRDecoder(const ConstrSet &outer, ScheduledExtraInfoFactory factory,
                      LayoutMap layouts,
                      std::shared_ptr<NativeStorageContext> storage)
      : factory_(factory), layouts_(std::move(layouts)),
        storage_(std::move(storage)) {
    constr_stack_ = outer.constrs_;
  }

  IRStructure::ChildList DecodeList(const SeqStmt &body,
                                    IRStructure *parent = nullptr) {
    IRStructure::ChildList previous;
    previous.swap(result_);
    auto old_layouts = layouts_;
    Visit(body);
    IRStructure::ChildList result;
    result.swap(result_);
    result_.swap(previous);
    layouts_ = std::move(old_layouts);
    for (size_t i = 0; i < result.size(); ++i) {
      result[i]->SetParent(parent);
      result[i]->SetIndex(i);
    }
    return result;
  }

private:
  template <class T> std::shared_ptr<T> MakeNode() const {
    auto node = std::make_shared<T>(factory_ ? factory_() : nullptr);
    node->SetStorageContext(storage_);
    return node;
  }

  void SetAccesses(TaskNode *task, const Stmt &body) {
    task->stmt = body;
    task->outer_ctx = GetConstrSet();
    ApplyTaskAccesses(
        AnalyzeTaskAccesses(body, layouts_, *storage_, task->outer_ctx), task);
  }

  PrimExpr CaptureCondition(const PrimExpr &condition) {
    if (SideEffect(condition) <= CallEffectKind::kPure)
      return condition;
    // This binding is proof-only. It preserves one evaluation for both arms
    // and gives RenameFrom a definition site for each repeated iteration.
    // The emitted scope keeps its original condition in ScopeNode::shell.
    PrimVar snapshot("schedule_condition", condition.ty());
    constr_stack_.emplace_back(snapshot, condition);
    return snapshot;
  }

  std::shared_ptr<TaskNode> DecodeTask(Stmt body, int stage,
                                       const ScheduleUnitMetadata &fields,
                                       bool per_core = false) {
    auto task = MakeNode<TaskNode>();
    SetAccesses(task.get(), body);
    task->SetStage(stage);
    auto metadata = ParseTaskMetadata(fields);
    if (metadata.core_mask)
      task->SetCoreMask(*metadata.core_mask);
    if (metadata.latency)
      task->SetLatency(*metadata.latency);
    if (metadata.ii)
      task->SetII(*metadata.ii);
    ApplyTaskResourceUsage(AnalyzeTaskResourceUsage(body), task.get());
    if (per_core)
      task->MarkPerCoreTask();
    else
      task->SetSpecialWriteMask(AnalyzeSpecialRegisterWrites(body));
    return task;
  }

  Result Visit_(const BindObj *op) final {
    TVM_FFI_CHECK(IsScheduledBind(op), ValueError)
        << "DecodeScheduledTIR found an unmarked Bind";
    auto binding = ReadScheduledBinding(op->value.as<CallObj>());
    result_.push_back(
        DecodeTask(GetRef<Bind>(op), binding.stage, binding.task));
    return std::nullopt;
  }

  Result Visit_(const IfObj *op) final {
    auto scope = MakeNode<ScopeNode>();
    scope->shell = GetRef<If>(op);
    scope->outer_ctx = GetConstrSet();
    scope->has_else = op->else_case.has_value();
    scope->task = MakeNode<TaskNode>();
    SetAccesses(scope->task.get(), Evaluate(op->condition, op->loc));
    scope->task->SetParent(scope.get());
    // A branch condition is one captured value. Mutable reads cannot be
    // replayed in later proofs after a store changes the condition's memory.
    auto context = Scope();
    PrimExpr condition = CaptureCondition(op->condition);
    scope->condition_snapshot = condition;
    {
      auto guard = MakeGuard(condition);
      scope->children = DecodeList(op->then_case, scope.get());
    }
    if (op->else_case) {
      auto guard = MakeGuard(Not(condition));
      scope->else_children = DecodeList(*op->else_case, scope.get());
    }
    result_.push_back(scope);
    return std::nullopt;
  }

  Result Visit_(const EvaluateObj *op) final {
    TVM_FFI_CHECK(IsTaskMetadataStatement(GetRef<Stmt>(op)), ValueError)
        << "DecodeScheduledTIR found an unmarked Evaluate";
    auto fields = ReadNativeAnnotations(op->value.as<CallObj>());
    if (auto layout = fields.Get("layout_map"))
      for (const auto &[tensor, value] : layout->cast<LayoutMap>())
        layouts_.Set(tensor, value);
    auto scope = MakeNode<ScopeNode>();
    scope->shell = GetRef<Stmt>(op);
    scope->outer_ctx = GetConstrSet();
    result_.push_back(scope);
    return std::nullopt;
  }

  Result Visit_(const RegionStmtObj *op) final {
    auto attribute = ReadTaskAttribute(op);
    TVM_FFI_CHECK(attribute.has_value(), ValueError)
        << "Unexpected native region in scheduled TIR: " << op->op->name;
    if (attribute->key == attr::kScheduleUnit) {
      int stage = GetScheduleUnitStage(op);
      Stmt body = SingleStatement(op->body);
      if (const auto *loop = body.as<ForObj>()) {
        TVM_FFI_CHECK(loop->kind == ForKind::kDefault ||
                          loop->kind == ForKind::kUnrolled,
                      ValueError)
            << "Scheduled control must be serial or unrolled";
        auto control = MakeNode<ControlNode>();
        control->control = GetRef<For>(loop);
        control->SetStage(stage);
        control->task = MakeNode<TaskNode>();
        SetAccesses(control->task.get(),
                    SeqStmt({Evaluate(loop->begin), Evaluate(loop->end),
                             Evaluate(loop->step)}));
        control->task->SetParent(control.get());
        auto context = Scope();
        ConstrSet body_context = GetConstrSet();
        AppendSerialLoopFacts(&body_context, control->control);
        constr_stack_ = std::move(body_context.constrs_);
        control->children = DecodeList(loop->body, control.get());
        control->InitializeLoopBodyContext();
        result_.push_back(control);
      } else {
        auto task = RequireSemanticTaskMarker(body);
        result_.push_back(DecodeTask(
            body, stage, task_metadata_detail::CopyTaskMetadata(task.node),
            task.key == attr::kAscendPerCoreTask));
      }
      return std::nullopt;
    }
    TVM_FFI_CHECK(IsScheduleGuardAttribute(attribute->key) ||
                      attribute->key == "tl.loop.annotations",
                  ValueError)
        << "Unexpected scheduled lexical attribute " << attribute->key;
    auto scope = MakeNode<ScopeNode>();
    scope->shell = attribute->region;
    scope->outer_ctx = GetConstrSet();
    scope->task = MakeNode<TaskNode>();
    Array<Stmt> reads;
    if (auto value = attribute->node.try_cast<PrimExpr>())
      reads.push_back(Evaluate(*value));
    if (auto value = attribute->value.try_cast<PrimExpr>())
      reads.push_back(Evaluate(*value));
    SetAccesses(scope->task.get(), SeqStmt(reads));
    scope->task->SetParent(scope.get());
    auto context = Scope();
    if (attribute->key == "tilelang_assume") {
      PrimExpr condition = CaptureCondition(attribute->node.cast<PrimExpr>());
      scope->condition_snapshot = condition;
      constr_stack_.emplace_back(condition, true);
    }
    scope->children = DecodeList(op->body, scope.get());
    result_.push_back(scope);
    return std::nullopt;
  }

  Result Visit_(const ForObj *) final {
    TVM_FFI_THROW(ValueError) << "Unmarked scheduled For";
    return std::nullopt;
  }
  Result Visit_(const TensorStoreObj *) final {
    TVM_FFI_THROW(ValueError) << "Unmarked scheduled TensorStore";
    return std::nullopt;
  }
  Result Visit_(const WhileObj *) final {
    TVM_FFI_THROW(ValueError) << "Unmarked scheduled While";
    return std::nullopt;
  }
  Result Visit_(const AssertStmtObj *) final {
    TVM_FFI_THROW(ValueError) << "Unmarked scheduled AssertStmt";
    return std::nullopt;
  }

  IRStructure::ChildList result_;
  ScheduledExtraInfoFactory factory_;
  LayoutMap layouts_;
  std::shared_ptr<NativeStorageContext> storage_;
};

} // namespace

std::optional<CoreMask> GetScheduledCoreMask(const Stmt &statement) {
  Stmt unit = SingleStatement(statement);
  if (const auto *bind = unit.as<BindObj>(); IsScheduledBind(bind))
    return ParseTaskMetadata(
               ReadScheduledBinding(bind->value.as<CallObj>()).task)
        .core_mask;
  auto marker = ReadTaskAttribute(unit);
  TVM_FFI_CHECK(marker && marker->key == attr::kScheduleUnit, ValueError)
      << "Expected a native schedule unit";
  Stmt body = SingleStatement(marker->region->body);
  if (body.as<ForObj>())
    return std::nullopt;
  return ParseTaskMetadata(RequireSemanticTaskMarker(body).node).core_mask;
}

CoreMask RequireScheduledCoreMask(const Stmt &statement) {
  auto mask = GetScheduledCoreMask(statement);
  TVM_FFI_CHECK(mask.has_value(), ValueError)
      << "Scheduled unit has no core_mask";
  return *mask;
}

RegionStmt EncodeScheduledTIR(ScheduledTIR scheduled_tir) {
  auto root = ReadTaskAttribute(scheduled_tir.metadata.kernel_root);
  TVM_FFI_CHECK(root && root->key == "tl.sblock", ValueError);
  auto shell = root->value.cast<ScheduleUnitMetadata>();
  bool preserve_root_hints = ffi::StructuralEqual()(
      scheduled_tir.metadata.root_conflict_hints,
      scheduled_tir.metadata.original_root_conflict_hints);
  TVM_FFI_CHECK(preserve_root_hints ||
                    scheduled_tir.metadata.root_conflict_hints.empty(),
                ValueError)
      << "Root conflict hints must retain their original operand carriers or "
         "be explicitly consumed";
  ScheduledTIREncoder encoder(WriteMetadata(scheduled_tir.metadata),
                              preserve_root_hints);
  if (auto original = shell.Get("annotations"))
    shell.Set("annotations",
              encoder.RewriteFields(original->cast<ScheduleUnitMetadata>()));
  auto body = encoder.Encode(scheduled_tir.tree, root->region->body->loc);
  return MakeNativeAttribute(root->node, root->key, shell, body,
                             root->region->loc);
}

ScheduledTIR
DecodeScheduledTIR(const RegionStmt &root, const ConstrSet &outer,
                   ScheduledExtraInfoFactory factory,
                   const ScheduleUnitMetadata &kernel_annotations,
                   NativeParameterAliasContract parameter_alias_contract) {
  auto attribute = ReadTaskAttribute(root);
  TVM_FFI_CHECK(attribute && attribute->key == "tl.sblock" &&
                    NativeBlockName(root.get()).value() == "tilelang_root",
                ValueError)
      << "Scheduled codec requires a native tilelang_root region";
  ScheduledTIR scheduled;
  scheduled.metadata.kernel_root = root;
  scheduled.metadata.outer_ctx = outer;
  ReadMetadata(kernel_annotations, &scheduled.metadata);
  LayoutMap layouts;
  auto shell = attribute->value.cast<ScheduleUnitMetadata>();
  if (auto annotations = shell.Get("annotations")) {
    auto fields = annotations->cast<ScheduleUnitMetadata>();
    ReadMetadata(fields, &scheduled.metadata);
    if (auto layout = fields.Get("layout_map"))
      layouts = layout->cast<LayoutMap>();
  }
  for (const Stmt &stmt : root->body->seq) {
    if (!IsTaskMetadataStatement(stmt))
      continue;
    auto fields =
        ReadNativeAnnotations(stmt.as<EvaluateObj>()->value.as<CallObj>());
    ReadMetadata(fields, &scheduled.metadata);
  }
  // Storage identities defined in nested sequences have carriers there. The
  // analysis map is kernel-wide, but its serialized operands stay lexical.
  VisitNativePostOrder(root->body, [&](const ObjectRef &node) {
    const auto *evaluate = node.as<EvaluateObj>();
    if (!evaluate || !IsTaskMetadataStatement(GetRef<Stmt>(evaluate)))
      return;
    auto fields = ReadNativeAnnotations(evaluate->value.as<CallObj>());
    ScheduleUnitMetadata storage;
    for (const auto &[key, value] : fields)
      if (IsStorageMapMetadataKey(key))
        storage.Set(key, value);
    if (!storage.empty())
      ReadMetadata(storage, &scheduled.metadata);
  });
  scheduled.metadata.original_root_conflict_hints =
      scheduled.metadata.root_conflict_hints;
  auto storage = std::make_shared<NativeStorageContext>();
  storage->SetParameterAliasContract(std::move(parameter_alias_contract));
  storage->Collect(root->body);
  auto decoder =
      ffi::make_object<ScheduledTIRDecoder>(outer, factory, layouts, storage);
  scheduled.tree = decoder->DecodeList(root->body);
  return scheduled;
}

} // namespace tvm::tl::ascend
