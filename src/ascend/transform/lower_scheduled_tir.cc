#include "ir/loop_math.h"
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

/*! \file lower_scheduled_tir.cc
 *  \brief Lower native scheduled tasks into executable Ascend core streams.
 */

#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/scheduled_tir.h"
#include "buffer_alias.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ascend;

namespace {

using VarSet = std::unordered_set<const VarObj *>;

void Append(ffi::Array<Stmt> *out, const Stmt &statement) {
  if (const auto *sequence = statement.as<SeqStmtObj>()) {
    for (const auto &child : sequence->seq)
      Append(out, child);
  } else {
    out->push_back(statement);
  }
}

void CollectDefinitions(const Stmt &statement, VarSet *out) {
  VisitNativePostOrder(statement, [&](const ffi::ObjectRef &object) {
    if (const auto *bind = object.as<BindObj>())
      out->insert(bind->var.get());
    if (const auto *loop = object.as<ForObj>())
      out->insert(loop->var.get());
    if (const auto *region = object.as<RegionStmtObj>()) {
      for (const auto &var : region->body_params)
        out->insert(var.get());
      for (const auto &var : region->result_vars)
        out->insert(var.get());
    }
  });
}

bool ConsumedSchedulingKey(const ffi::String &key) {
  return key == "num_stages" || key == "enable_offset" ||
         key == "tl_pipeline_order" || key == "tl_pipeline_stage" ||
         key == kMultiBufferEligible || key == kBufferVersionsMap ||
         key == kBufferVersionMode || key == "conflict_hint" ||
         key == kRootConflictHints || key == "tl.scoped_conflict_hints" ||
         key == "tl.conflict_scope_id" || key == kSplitSubcores;
}

For PromoteDynamicLoopAnnotations(For loop) {
  ScheduleUnitMetadata constant, dynamic;
  for (const auto &[key, value] : loop->annotations) {
    auto encoded = EncodeAnnotations(value);
    if (UndefinedVars(encoded.payload).empty())
      constant.Set(key, value);
    else
      dynamic.Set(key, value);
  }
  if (dynamic.empty())
    return loop;
  // Native structural definition renewal intentionally skips the constant
  // For annotation map. Retain actual IR references as operands inside the
  // loop's definition region, where every generic visitor can follow them.
  auto body = loop->body;
  auto *node = loop.CopyOnWrite();
  node->annotations = std::move(constant);
  node->body = SeqStmt({MakeNativeAttribute(nullptr, "tl.loop.annotations",
                                            dynamic, body, loop->loc)},
                       body->loc);
  return loop;
}

// Filter storage maps before renewing definitions. Each core retains only
// references whose actual definitions survive extraction. Empty alias maps
// remain present: their presence disables unproved physical storage reuse.
class MetadataCleaner : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  MetadataCleaner(VarSet all, VarSet kept)
      : all_(std::move(all)), kept_(std::move(kept)) {}

  ScheduleUnitMetadata Fields(const ScheduleUnitMetadata &fields) const {
    ScheduleUnitMetadata result;
    for (const auto &[key, value] : fields) {
      if (ConsumedSchedulingKey(key))
        continue;
      if (key == "annotations") {
        result.Set(key, Fields(value.cast<ScheduleUnitMetadata>()));
      } else if (auto retained = Prune(value)) {
        result.Set(key, *retained);
      }
    }
    if (auto aliases = result.Get(kBufferAliasMap))
      ValidateBufferAliasMap(aliases->cast<BufferAliasMap>());
    return result;
  }

private:
  bool UsesRemoved(const ffi::Any &value) const {
    bool removed = false;
    auto encoded = EncodeAnnotations(value);
    VisitNativePostOrder(Evaluate(encoded.payload),
                         [&](const ffi::ObjectRef &object) {
                           if (const auto *var = object.as<VarObj>())
                             removed |= all_.count(var) && !kept_.count(var);
                         });
    return removed;
  }

  std::optional<ffi::Any> Prune(const ffi::Any &value) const {
    if (const auto *map = value.as<ffi::MapObj>()) {
      ffi::Map<ffi::Any, ffi::Any> result;
      for (const auto &[key, item] : *map) {
        if (UsesRemoved(key))
          continue;
        if (auto retained = Prune(item))
          result.Set(key, *retained);
      }
      return result;
    }
    if (const auto *array = value.as<ffi::ArrayObj>()) {
      ffi::Array<ffi::Any> result;
      for (const auto &item : *array)
        if (auto retained = Prune(item))
          result.push_back(*retained);
      return result;
    }
    return UsesRemoved(value) ? std::nullopt : std::optional<ffi::Any>(value);
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    if (IsScheduledBind(op)) {
      Expr value = ReadScheduledBinding(op).value;
      return Mutate(value, mode).ValueOrUnchanged(value);
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj *op,
                                 InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (call && call->op.same_as(Op::Get("tl.annotation"))) {
      auto fields = Fields(ReadNativeAnnotations(call));
      return fields.empty() ? Stmt(SeqStmt({}, op->loc))
                            : MakeNativeAnnotation(fields, op->loc);
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    auto attribute = ReadTaskAttribute(op);
    if (!attribute)
      return StmtExprMutator::Mutate_(op, mode);
    SeqStmt body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    if (IsSemanticTaskKey(attribute->key) ||
        attribute->key == attr::kScheduleUnit ||
        ConsumedSchedulingKey(attribute->key))
      return body;
    ffi::Any node = attribute->node, value = attribute->value;
    if (attribute->key == "tl.loop.annotations" ||
        attribute->key == "tl.kernel" || attribute->key == "tl.sblock") {
      auto fields = Fields(value.cast<ScheduleUnitMetadata>());
      if (attribute->key == "tl.loop.annotations" && fields.empty())
        return body;
      value = fields;
    } else if (attribute->key == kBufferAliasMap ||
               attribute->key == kAivStorageMap) {
      node = Prune(node).value();
      if (attribute->key == kBufferAliasMap)
        ValidateBufferAliasMap(node.cast<BufferAliasMap>());
    }
    return MakeNativeAttribute(node, attribute->key, value, body, op->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    For loop = StmtExprMutator::Mutate_(op, mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                   .as_or_throw<For>();
    loop.CopyOnWrite()->annotations = Fields(loop->annotations);
    return PromoteDynamicLoopAnnotations(loop);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    ffi::Array<Stmt> statements;
    for (const auto &statement : op->seq)
      Append(&statements, Mutate(statement, mode).ValueOrUnchanged(statement));
    return SeqStmt(statements, op->loc);
  }

  VarSet all_, kept_;
};

class MetadataRemapper : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  ScheduleUnitMetadata Fields(const ScheduleUnitMetadata &fields) {
    auto encoded = EncodeAnnotations(fields);
    Expr payload = encoded.payload;
    payload = Mutate(payload).ValueOrUnchanged(payload);
    return DecodeAnnotations(payload.as_or_throw<tvm::Tuple>(), encoded.schema)
        .cast<ScheduleUnitMetadata>();
  }

private:
  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    For loop = StmtExprMutator::Mutate_(op, mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                   .as_or_throw<For>();
    // Native For mutation leaves its static map untouched. Older producers
    // can put IR references there, so apply the same stage substitutions as
    // the operand-bearing metadata carriers before renewing definitions.
    loop.CopyOnWrite()->annotations = Fields(loop->annotations);
    return PromoteDynamicLoopAnnotations(loop);
  }
};

struct StageBounds {
  int first{std::numeric_limits<int>::max()};
  int last{std::numeric_limits<int>::min()};
  bool empty() const { return first > last; }
  void Add(int stage) {
    TVM_FFI_CHECK_GE(stage, 0, ValueError)
        << "LowerScheduledTIR requires resolved nonnegative stages";
    first = std::min(first, stage);
    last = std::max(last, stage);
  }
  void Add(StageBounds bounds) {
    if (!bounds.empty()) {
      Add(bounds.first);
      Add(bounds.last);
    }
  }
};

bool IsTransparentMetadata(const IRStructure *node) {
  if (!node->IsScope())
    return false;
  auto attribute =
      ReadTaskAttribute(static_cast<const ScopeNode *>(node)->shell);
  return attribute && attribute->key == "tl.loop.annotations";
}

// Bounds are proof-only. No expression simplified under a populated context
// is emitted: mutable-read snapshots belong to the analyzer, not native IR.
std::optional<std::pair<int64_t, int64_t>>
IntegerBounds(const PrimExpr &expression, const sym::Analyzer &analyzer) {
  if (auto value = TryConstInt(analyzer->Simplify(expression)))
    return std::pair(*value, *value);
  PrimType type = expression.ty();
  if (!type.IsScalar() || !type.MatchesCode(kDLInt, kDLUInt) ||
      (type.code() == kDLUInt && type.bits() >= 64))
    return std::nullopt;
  auto bound = analyzer->const_int_bound(expression);
  ffi::BigInt minimum = type.code() == kDLUInt
                            ? ffi::BigInt(0)
                            : -(ffi::BigInt(1) << (type.bits() - 1));
  ffi::BigInt maximum =
      (ffi::BigInt(1) << (type.bits() - (type.code() == kDLInt))) - 1;
  minimum = std::max(minimum, ffi::BigInt(bound->min_value));
  maximum = std::min(maximum, ffi::BigInt(bound->max_value));
  auto lo = minimum.as<int64_t>(), hi = maximum.as<int64_t>();
  if (!lo || !hi || *lo > *hi)
    return std::nullopt;
  return std::pair(*lo, *hi);
}

struct PipelineClock {
  PrimVar original;
  PrimVar physical;
  PrimExpr begin;
  PrimExpr step;
  PrimExpr trips;
  int first_stage;

  PrimExpr Logical(int stage) const {
    return physical - IntImm::Int64(int64_t(stage) - first_stage);
  }
  PrimExpr Guard(int stage) const {
    PrimExpr logical = Logical(stage);
    return logical >= IntImm::Int64(0) && logical < trips;
  }
  PrimExpr Coordinate(int stage) const {
    // Clamping keeps pure scalar definitions and native storage declarations
    // valid during fill/drain. Observable reads and effects remain guarded.
    PrimExpr ordinal =
        Max(IntImm::Int64(0), Min(trips - IntImm::Int64(1), Logical(stage)));
    return cast(begin + ordinal * step, original.ty());
  }
};

// SS names one static pool shared by the whole MIX group. Keep each native
// allocation owner outside the separate core streams; definition renewal can
// then freshen private state and views without splitting that physical owner.
struct SharedSSStorage {
  ffi::Array<Stmt> allocations;
  VarSet variables;

  explicit SharedSSStorage(const Stmt &body) { Collect(body); }

private:
  void Collect(const Stmt &statement) {
    // Metadata may contain native Stmt templates. Only declarations reached
    // through executable statement bodies own storage in this kernel.
    if (const auto *sequence = statement.as<SeqStmtObj>()) {
      for (const auto &child : sequence->seq)
        Collect(child);
    } else if (const auto *loop = statement.as<ForObj>()) {
      Collect(loop->body);
    } else if (const auto *loop = statement.as<WhileObj>()) {
      Collect(loop->body);
    } else if (const auto *branch = statement.as<IfObj>()) {
      Collect(branch->then_case);
      if (branch->else_case)
        Collect(*branch->else_case);
    } else if (const auto *region = statement.as<RegionStmtObj>()) {
      Collect(region->body);
    } else if (const auto *bind = statement.as<BindObj>()) {
      auto tensor = bind->var.as<TensorVar>();
      if (!tensor || (tensor->scope().name() != "shared.ss" &&
                      tensor->scope().name() != "shared.ss.dyn"))
        return;
      Expr value = bind->value;
      if (IsScheduledBind(value.as<CallObj>()))
        value = ReadScheduledBinding(value.as<CallObj>()).value;
      const auto *call = value.as<CallObj>();
      if (!call || !call->op.same_as(alloc_tensor_op()))
        return;
      Stmt allocation = Bind(bind->var, value, bind->loc);
      TVM_FFI_CHECK(UndefinedVars(allocation).empty(), ValueError)
          << "Group-shared SS allocations require static, self-contained "
             "native types";
      TVM_FFI_CHECK(variables.insert(bind->var.get()).second, ValueError)
          << "SS storage must have one native allocation definition";
      allocations.push_back(allocation);
    }
  }
};

class SharedSSDefinitionRetirer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit SharedSSDefinitionRetirer(VarSet variables)
      : variables_(std::move(variables)) {}

private:
  ffi::UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                                    InplaceMode mode) final {
    if (value.as<Expr>())
      return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (variables_.count(op->var.get()))
      return SeqStmt({}, op->loc);
    return ffi::Unchanged();
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    SeqStmt body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    if (body.same_as(op->body))
      return ffi::Unchanged();
    auto copy = ffi::make_object<RegionStmtObj>(*op);
    copy->body = body;
    return RegionStmt(copy);
  }

  VarSet variables_;
};

class CoreBodyBuilder {
public:
  CoreBodyBuilder(const ScheduledTIR &scheduled, CoreMask target, PrimVar sid,
                  int vector_count, const VarSet &shared_ss)
      : scheduled_(scheduled), target_(target), sid_(std::move(sid)),
        vector_count_(vector_count),
        remapper_(ffi::make_object<MetadataRemapper>()) {
    if (!shared_ss.empty())
      ss_retirer_ = ffi::make_object<SharedSSDefinitionRetirer>(shared_ss);
    VarSet all, kept;
    CollectDefinitions(scheduled.metadata.kernel_root->body, &all);
    for (const auto &node : scheduled.tree)
      CollectSelectedDefinitions(node.get(), &kept);
    kept.insert(shared_ss.begin(), shared_ss.end());
    cleaner_ =
        ffi::make_object<MetadataCleaner>(std::move(all), std::move(kept));
  }

  SeqStmt Build() {
    SeqStmt body = LowerList(scheduled_.tree, nullptr, kCoreUnassigned);
    // Renew a whole native definition region in one operation: types,
    // metadata operands, aliases and uses follow the corresponding fresh Var.
    return RenewDef(tirx::Function({}, body))->body;
  }

private:
  CoreMask SelectedMask(const IRStructure *node) const {
    if (node->IsTask()) {
      auto mask = static_cast<const TaskNode *>(node)->GetCoreMask();
      TVM_FFI_CHECK_NE(mask, kCoreUnassigned, ValueError)
          << "LowerScheduledTIR requires assigned task cores";
      return mask & target_;
    }
    CoreMask result = kCoreUnassigned;
    for (const auto *list : node->ChildLists())
      for (const auto &child : *list)
        result |= SelectedMask(child.get());
    return result;
  }

  void CollectSelectedDefinitions(const IRStructure *node, VarSet *kept) {
    if (SelectedMask(node) == kCoreUnassigned)
      return;
    if (node->IsTask()) {
      CollectDefinitions(static_cast<const TaskNode *>(node)->stmt, kept);
      return;
    }
    if (node->IsControl())
      kept->insert(static_cast<const ControlNode *>(node)->control->var.get());
    for (const auto *list : node->ChildLists())
      for (const auto &child : *list)
        CollectSelectedDefinitions(child.get(), kept);
  }

  StageBounds Stages(const IRStructure::ChildList &nodes) const {
    StageBounds result;
    for (const auto &node : nodes)
      result.Add(Stages(node.get()));
    return result;
  }

  StageBounds Stages(const IRStructure *node) const {
    StageBounds result;
    if (SelectedMask(node) == kCoreUnassigned)
      return result;
    if (!node->IsScope()) {
      result.Add(node->GetStage());
      return result;
    }
    for (const auto *list : node->ChildLists())
      result.Add(Stages(*list));
    TVM_FFI_CHECK(IsTransparentMetadata(node) || result.empty() ||
                      result.first == result.last,
                  ValueError)
        << "A lexical condition or assumption must stay in one pipeline stage";
    return result;
  }

  template <typename T> T Remap(const T &value) {
    return remapper_->Mutate(value).ValueOrUnchanged(value);
  }

  Stmt GuardStatement(Stmt statement, const PrimExpr &guard) {
    if (IsOne(guard))
      return statement;
    if (const auto *bind = SingleStatement(statement).as<BindObj>()) {
      Expr value = bind->value;
      if (auto primitive = value.as<PrimExpr>()) {
        if (SideEffect(value) > CallEffectKind::kPure)
          value = if_then_else(guard, *primitive, MakeConst(primitive->ty(), 0),
                               value->loc);
      } else if (value->ty.as<PtrTypeObj>()) {
        PrimExpr bits = Call(PrimType::UInt(64), reinterpret_op(), {value})
                            .as_or_throw<PrimExpr>();
        bits = if_then_else(guard, bits, IntImm(PrimType::UInt(64), 0));
        value = Call(value->ty, reinterpret_op(), {bits});
      } else {
        const auto *call = value.as<CallObj>();
        bool declaration = call && (call->op.same_as(alloc_tensor_op()) ||
                                    call->op.same_as(decl_tensor_op()));
        TVM_FFI_CHECK(value.as<TensorVar>().has_value() || declaration,
                      ValueError)
            << "A guarded native tensor definition must be an allocation or "
               "a declared view";
        if (call && call->op.same_as(decl_tensor_op())) {
          auto pointer = call->args[0];
          if (SideEffect(pointer) > CallEffectKind::kPure) {
            PrimExpr bits =
                Call(PrimType::UInt(64), reinterpret_op(), {pointer})
                    .as_or_throw<PrimExpr>();
            bits = if_then_else(guard, bits, IntImm(PrimType::UInt(64), 0));
            Call rewritten = ffi::GetRef<Call>(call);
            rewritten.CopyOnWrite()->args.Set(
                0, Call(pointer->ty, reinterpret_op(), {bits}));
            value = rewritten;
          }
        }
      }
      return Bind(bind->var, value, bind->loc);
    }
    return If(guard, NativeTaskAttribute::AsSequence(statement), std::nullopt,
              statement->loc);
  }

  Stmt ApplyNodeGuards(const IRStructure *node, Stmt body) {
    for (auto it = node->GetGuards().rbegin(); it != node->GetGuards().rend();
         ++it) {
      if ((*it)->IsCondition()) {
        auto *condition = static_cast<const ConditionGuard *>(it->get());
        body = GuardStatement(body, Remap(condition->condition));
      } else {
        const auto *attribute = static_cast<const AttributeGuard *>(it->get());
        Stmt header =
            MakeNativeAttribute(attribute->node, attribute->key,
                                attribute->value, SeqStmt({}), attribute->loc);
        header = Remap(cleaner_->Mutate(header).ValueOrUnchanged(header));
        if (auto retained = ReadTaskAttribute(header))
          body = retained->WithBody(body);
      }
    }
    return body;
  }

  SeqStmt LowerList(const IRStructure::ChildList &nodes,
                    const PipelineClock *clock, CoreMask enclosed) {
    ffi::Array<Stmt> statements;
    for (const auto &node : nodes) {
      if (IsTransparentMetadata(node.get())) {
        if (clock)
          remapper_->VarRemapSet(clock->original,
                                 clock->Coordinate(clock->first_stage));
        Append(&statements,
               LowerScope(static_cast<const ScopeNode *>(node.get()), clock,
                          enclosed));
        continue;
      }
      StageBounds stages = Stages(node.get());
      if (stages.empty()) {
        if (node->IsScope()) {
          const auto *scope = static_cast<const ScopeNode *>(node.get());
          if (IsTaskMetadataStatement(scope->shell)) {
            if (clock)
              remapper_->VarRemapSet(clock->original,
                                     clock->Coordinate(clock->first_stage));
            Stmt metadata =
                cleaner_->Mutate(scope->shell).ValueOrUnchanged(scope->shell);
            Append(&statements, Remap(metadata));
          }
        }
        continue;
      }
      PrimExpr guard = IntImm::Bool(true);
      if (clock) {
        TVM_FFI_CHECK_EQ(stages.first, stages.last, ValueError);
        remapper_->VarRemapSet(clock->original,
                               clock->Coordinate(stages.first));
        guard = clock->Guard(stages.first);
      }
      CoreMask selected = SelectedMask(node.get());
      CoreMask inner_enclosed = enclosed;
      if (target_ != kCoreCube && vector_count_ > 1 && IsVectorCore(selected) &&
          selected != enclosed) {
        PrimExpr subcore = sid_ == IntImm(sid_.ty(), VectorSid(selected));
        guard = IsOne(guard) ? subcore : guard && subcore;
        inner_enclosed = selected;
      }
      Stmt body =
          node->IsTask() ? LowerTask(static_cast<const TaskNode *>(node.get()))
          : node->IsControl()
              ? LowerControl(static_cast<const ControlNode *>(node.get()),
                             inner_enclosed)
              : LowerScope(static_cast<const ScopeNode *>(node.get()), nullptr,
                           inner_enclosed);
      body = ApplyNodeGuards(node.get(), std::move(body));
      Append(&statements, GuardStatement(std::move(body), guard));
    }
    return SeqStmt(statements);
  }

  Stmt LowerTask(const TaskNode *task) {
    Stmt statement = cleaner_->Mutate(task->stmt).ValueOrUnchanged(task->stmt);
    if (ss_retirer_)
      statement = ss_retirer_->Mutate(statement).ValueOrUnchanged(statement);
    return Remap(statement);
  }

  Stmt LowerScope(const ScopeNode *scope, const PipelineClock *clock,
                  CoreMask enclosed) {
    if (const auto *branch = scope->shell.as<IfObj>()) {
      PrimExpr condition = Remap(branch->condition);
      SeqStmt then_case = LowerList(scope->children, nullptr, enclosed);
      ffi::Optional<SeqStmt> else_case;
      if (scope->has_else)
        else_case = LowerList(scope->else_children, nullptr, enclosed);
      return If(condition, then_case, else_case, branch->loc);
    }
    auto attribute = ReadTaskAttribute(scope->shell);
    TVM_FFI_CHECK(attribute.has_value(), ValueError)
        << "Unknown scheduled lexical scope";
    Stmt header = attribute->WithBody(SeqStmt({}));
    header = cleaner_->Mutate(header).ValueOrUnchanged(header);
    header = Remap(header);
    SeqStmt body = LowerList(scope->children, clock, enclosed);
    if (auto retained = ReadTaskAttribute(header))
      return retained->WithBody(body);
    return body;
  }

  Stmt LowerControl(const ControlNode *control, CoreMask enclosed) {
    const For &source = control->control;
    PrimExpr begin = Remap(source->begin), end = Remap(source->end);
    PrimExpr step = Remap(source->step);
    auto annotations = cleaner_->Fields(source->annotations);
    StageBounds stages = Stages(control->children);
    if (stages.empty() || stages.first == stages.last) {
      annotations = remapper_->Fields(annotations);
      SeqStmt body = LowerList(control->children, nullptr, enclosed);
      return PromoteDynamicLoopAnnotations(For(source->var, begin, end, step,
                                               source->kind, body, annotations,
                                               source->loc));
    }

    auto analyzer = CreateTileLangAnalyzer();
    auto context = control->task->outer_ctx.Populate(analyzer);
    TVM_FFI_CHECK(!control->BodyHasLoopBreak() &&
                      CanUseLoopBounds(source->begin, source->end, source->step,
                                       analyzer.get()),
                  ValueError)
        << "Pipeline stage expansion requires a positive, non-wrapping For "
           "without early exit";
    auto begin_bound = IntegerBounds(source->begin, analyzer);
    auto end_bound = IntegerBounds(source->end, analyzer);
    auto step_bound = IntegerBounds(source->step, analyzer);
    TVM_FFI_CHECK(begin_bound && end_bound && step_bound &&
                      step_bound->first > 0,
                  ValueError)
        << "Pipeline iteration arithmetic must have representable int64 bounds";
    ffi::BigInt distance_bound = std::max(
        ffi::BigInt(0), ffi::BigInt(end_bound->second) - begin_bound->first);
    ffi::BigInt trip_bound = distance_bound / step_bound->first +
                             (distance_bound % step_bound->first != 0 ? 1 : 0);
    int64_t span = int64_t(stages.last) - stages.first;
    TVM_FFI_CHECK(distance_bound.as<int64_t>() &&
                      (trip_bound + span).as<int64_t>(),
                  ValueError)
        << "Pipeline fill/drain would overflow the int64 iteration clock";

    PrimExpr begin64 = cast(begin, PrimType::Int(64));
    PrimExpr end64 = cast(end, PrimType::Int(64));
    PrimExpr step64 = cast(step, PrimType::Int(64));
    PrimExpr distance = Max(end64 - begin64, IntImm::Int64(0));
    PrimExpr trips =
        FloorDiv(distance, step64) +
        cast(FloorMod(distance, step64) != IntImm::Int64(0), PrimType::Int(64));
    PrimVar physical(source->var->name + "_pipeline", PrimType::Int(64));
    PipelineClock clock{source->var, physical, begin64,
                        step64,      trips,    stages.first};
    auto previous = remapper_->VarRemapGet(source->var);
    remapper_->VarRemapSet(source->var, clock.Coordinate(clock.first_stage));
    annotations = remapper_->Fields(annotations);
    SeqStmt body = LowerList(control->children, &clock, enclosed);
    remapper_->VarRemapSet(
        source->var, previous.as<ExprObj>() ? previous : ffi::Any(source->var));
    Stmt expanded = PromoteDynamicLoopAnnotations(
        For(physical, IntImm::Int64(0), trips + IntImm::Int64(span),
            IntImm::Int64(1), source->kind, body, annotations, source->loc));
    // A zero-trip source cannot execute speculative definitions in fill/drain.
    return If(begin < end, NativeTaskAttribute::AsSequence(expanded),
              std::nullopt, source->loc);
  }

  const ScheduledTIR &scheduled_;
  CoreMask target_;
  PrimVar sid_;
  int vector_count_;
  ffi::ObjectPtr<MetadataRemapper> remapper_;
  ffi::ObjectPtr<MetadataCleaner> cleaner_;
  ffi::ObjectPtr<SharedSSDefinitionRetirer> ss_retirer_;
};

bool HasCoreStream(const ScheduledTIR &scheduled, CoreMask target) {
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(scheduled.tree, tasks);
  return std::any_of(tasks.begin(), tasks.end(), [&](const TaskNode *task) {
    return IsCoreLocalMask(task->GetCoreMask()) &&
           (task->GetCoreMask() & target) != kCoreUnassigned;
  });
}

RegionStmt CoreBlock(ffi::String name, SeqStmt body,
                     ScheduleUnitMetadata annotations = {}) {
  return MakeNativeAttribute(nullptr, "tl.sblock",
                             ScheduleUnitMetadata{{"name", name},
                                                  {"no_realize", true},
                                                  {"annotations", annotations}},
                             body, body->loc);
}

// cthread names an AIV. An externally declared launch must enclose only the
// VECTOR body; retaining it around CUBE would duplicate the cube stream.
class SidRelocator : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

private:
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    bool is_cthread = op->op.same_as(launch_thread_op()) &&
                      op->args[0].as_or_throw<StringImm>()->value == "cthread";
    if (is_cthread && !inside_vector_) {
      launches_.push_back(ffi::GetRef<RegionStmt>(op));
      SeqStmt body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
      launches_.pop_back();
      return body;
    }
    bool vector = NativeBlockName(op).value_or("") == "VECTOR";
    bool previous = inside_vector_;
    inside_vector_ |= vector;
    RegionStmt region = StmtExprMutator::Mutate_(op, mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                            .as_or_throw<RegionStmt>();
    inside_vector_ = previous;
    if (vector && !launches_.empty()) {
      SeqStmt body = region->body;
      for (auto it = launches_.rbegin(); it != launches_.rend(); ++it) {
        RegionStmt launch = *it;
        launch.CopyOnWrite()->body = body;
        body = SeqStmt({launch}, body->loc);
      }
      // One outer launch may have enclosed several kernels. Their new launch
      // definitions must be fresh independently, including every sid use.
      region.CopyOnWrite()->body = RenewDef(tirx::Function({}, body))->body;
    }
    return region;
  }

  bool inside_vector_{false};
  std::vector<RegionStmt> launches_;
};

// Consume only tl.kernel carriers read by the shared kernel collector. A
// nested launch starts its own metadata chain, and an already lowered root
// ends the chain before the generated VECTOR execution annotation.
class KernelVectorCountConsumer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

private:
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (kernel_rewriter_detail::IsRoot(op)) {
      reached_root_ = true;
      return ffi::Unchanged();
    }
    bool old_chain = in_kernel_chain_;
    bool old_root = reached_root_;
    bool anchor = kernel_rewriter_detail::IsKernelAnchor(op);
    if (anchor)
      in_kernel_chain_ = true;
    reached_root_ = false;
    RegionStmt region = StmtExprMutator::Mutate_(op, mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                            .as_or_throw<RegionStmt>();
    bool owns_root = reached_root_;
    auto attribute = ReadTaskAttribute(region);
    if (in_kernel_chain_ && owns_root && attribute &&
        attribute->key == "tl.kernel") {
      auto fields = attribute->value.cast<ScheduleUnitMetadata>();
      fields.erase(kVectorCount);
      region = MakeNativeAttribute(attribute->node, attribute->key, fields,
                                   region->body, region->loc);
    }
    in_kernel_chain_ = old_chain;
    reached_root_ = old_root || (!anchor && owns_root);
    return region;
  }

  bool in_kernel_chain_{false};
  bool reached_root_{false};
};

RegionStmt LowerKernel(const TilelangKernelContext &context) {
  ScheduledTIR scheduled = DecodeScheduledTIR(
      context.root, context.outer_ctx, nullptr, context.kernel_annotations,
      context.parameter_alias_contract);
  int vector_count = scheduled.metadata.num_aiv_subcores.value_or(2);
  // DecodeScheduledTIR reads metadata statements directly in the root body.
  // Nested scopes have separate owners and must retain their own fields.
  for (const auto &node : scheduled.tree) {
    if (!node->IsScope())
      continue;
    auto *scope = static_cast<ScopeNode *>(node.get());
    if (!IsTaskMetadataStatement(scope->shell))
      continue;
    auto fields = ReadNativeAnnotations(
        scope->shell.as<EvaluateObj>()->value.as<CallObj>());
    if (fields.count(kVectorCount)) {
      fields.erase(kVectorCount);
      scope->shell = MakeNativeAnnotation(fields, scope->shell->loc);
    }
  }
  CoreMask vectors = VectorCoreMask(vector_count);
  bool cube = HasCoreStream(scheduled, kCoreCube);
  bool vector = HasCoreStream(scheduled, vectors);
  if (!cube && !vector)
    vector = true;
  PrimVar sid = context.outer_sid.value_or(PrimVar("sid", PrimType::Int(32)));
  SharedSSStorage shared_ss(context.root->body);
  ffi::Array<Stmt> streams = shared_ss.allocations;
  if (cube)
    streams.push_back(
        CoreBlock("CUBE", CoreBodyBuilder(scheduled, kCoreCube, sid,
                                          vector_count, shared_ss.variables)
                              .Build()));
  if (vector) {
    SeqStmt body = CoreBodyBuilder(scheduled, vectors, sid, vector_count,
                                   shared_ss.variables)
                       .Build();
    bool uses_sid =
        UsesAnyVar(body, [&](const VarObj *var) { return var == sid.get(); });
    if (!context.outer_sid && (cube || uses_sid)) {
      auto launch =
          RegionStmt(launch_thread_op(),
                     {StringImm("cthread"), IntImm::Int32(vector_count)}, {sid},
                     DictAttrs(), body);
      body = SeqStmt({launch}, body->loc);
    }
    ScheduleUnitMetadata annotations;
    if (cube || uses_sid || context.outer_sid)
      annotations.Set(kVectorCount, IntImm::Int32(vector_count));
    streams.push_back(CoreBlock("VECTOR", body, annotations));
  }
  auto root = ReadTaskAttribute(context.root).value();
  auto cleaner = ffi::make_object<MetadataCleaner>(VarSet(), VarSet());
  auto fields = cleaner->Fields(root.value.cast<ScheduleUnitMetadata>());
  if (auto annotations = fields.Get("annotations")) {
    auto values = annotations->cast<ScheduleUnitMetadata>();
    values.erase(kVectorCount);
    fields.Set("annotations", values);
  }
  return MakeNativeAttribute(root.node, root.key, fields, SeqStmt(streams),
                             context.root->loc);
}

} // namespace

tvm::transform::Pass LowerScheduledTIR() {
  auto transform = [](tirx::Function function, const IRModule &,
                      const tvm::transform::PassContext &) {
    function = RewriteTilelangKernels(std::move(function), "LowerScheduledTIR",
                                      LowerKernel);
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      auto consumer = ffi::make_object<KernelVectorCountConsumer>();
      SeqStmt consumed =
          consumer->Mutate(function->body).ValueOrUnchanged(function->body);
      auto relocator = ffi::make_object<SidRelocator>();
      SeqStmt body = relocator->Mutate(consumed).ValueOrUnchanged(consumed);
      function.CopyOnWrite()->body = std::move(body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(transform, 0,
                                             "tl.LowerScheduledTIR");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerScheduledTIR",
                                   LowerScheduledTIR);
}
} // namespace tvm::tl
