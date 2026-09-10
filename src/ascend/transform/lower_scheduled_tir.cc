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
#include "./auto_schedule/scheduled_tir.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "op/builtin.h"
#include "support/check.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;
using ffi::Array;
using ffi::GetRef;
using ffi::Map;
using ffi::ObjectPtrEqual;
using ffi::ObjectPtrHash;
using ffi::Optional;
using ffi::String;

namespace {

Stmt MakeSequence(const Array<Stmt> &statements) {
  if (statements.empty())
    return Stmt();
  if (statements.size() == 1)
    return statements[0];
  return SeqStmt::Flatten(statements);
}

Array<Stmt> GetSequence(const Stmt &stmt) {
  if (const auto *sequence = stmt.as<SeqStmtNode>())
    return sequence->seq;
  return {stmt};
}

class LetVarSubstituter : public StmtExprMutator {
public:
  explicit LetVarSubstituter(const Map<Var, PrimExpr> &vmap) : vmap_(vmap) {}

private:
  PrimExpr VisitExpr_(const VarNode *op) final {
    auto it = vmap_.find(GetRef<Var>(op));
    return it != vmap_.end() ? (*it).second : StmtExprMutator::VisitExpr_(op);
  }

  Buffer VisitBufferUse(const Buffer &buffer) final {
    auto it = buffer_remap_.find(buffer);
    if (it != buffer_remap_.end())
      return (*it).second;
    auto var_it = vmap_.find(buffer->data);
    if (var_it == vmap_.end())
      return buffer;
    const auto *data = (*var_it).second.as<VarNode>();
    if (data == nullptr)
      return buffer;
    Buffer new_buffer = buffer;
    new_buffer.CopyOnWrite()->data = GetRef<Var>(data);
    buffer_remap_[buffer] = new_buffer;
    return new_buffer;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    ffi::Any node = op->node;
    bool node_changed = false;
    if (const auto *expr_node = op->node.as<PrimExprNode>()) {
      PrimExpr old_node = GetRef<PrimExpr>(expr_node);
      PrimExpr new_node = VisitExpr(old_node);
      node_changed = !new_node.same_as(old_node);
      node = std::move(new_node);
    }
    PrimExpr value = VisitExpr(op->value);
    Stmt body = VisitStmt(op->body);
    if (!node_changed && value.same_as(op->value) && body.same_as(op->body))
      return GetRef<Stmt>(op);
    return AttrStmt(node, op->attr_key, value, body, op->span);
  }

  const Map<Var, PrimExpr> &vmap_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
};

Stmt SubstituteLetVars(Stmt stmt, const Map<Var, PrimExpr> &vmap) {
  if (vmap.empty())
    return stmt;
  return LetVarSubstituter(vmap)(std::move(stmt));
}

CoreMask GetUnitCoreMask(const AttrStmtNode *op) {
  return RequireScheduledCoreMask(op);
}

int GetMarkerStage(const AttrStmtNode *op) { return GetScheduleUnitStage(op); }

Stmt StripTaskMarker(const Stmt &stmt) {
  if (const auto *attribute = stmt.as<AttrStmtNode>()) {
    if (attribute->attr_key == attr::kAscendTask)
      return attribute->body;
  }
  return stmt;
}

Optional<Bind> FindTaskBind(const Stmt &task_body) {
  Stmt current = task_body;
  while (true) {
    if (const auto *bind = current.as<BindNode>())
      return GetRef<Bind>(bind);
    if (const auto *conditional = current.as<IfThenElseNode>()) {
      if (conditional->else_case.has_value())
        return std::nullopt;
      current = conditional->then_case;
      continue;
    }
    if (const auto *attribute = current.as<AttrStmtNode>()) {
      current = attribute->body;
      continue;
    }
    return std::nullopt;
  }
}

Stmt ReplaceTaskBind(const Stmt &task_body, const Stmt &replacement) {
  if (task_body.as<BindNode>())
    return replacement;
  if (const auto *conditional = task_body.as<IfThenElseNode>()) {
    ICHECK(!conditional->else_case.has_value());
    return IfThenElse(conditional->condition,
                      ReplaceTaskBind(conditional->then_case, replacement),
                      std::nullopt, conditional->span);
  }
  const auto *attribute = task_body.as<AttrStmtNode>();
  ICHECK(attribute != nullptr)
      << "Expected a native guard chain around Bind, got " << task_body;
  return AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                  ReplaceTaskBind(attribute->body, replacement),
                  attribute->span);
}

bool IsGuardedTaskBind(const AttrStmtNode *marker) {
  Stmt task_body = StripTaskMarker(marker->body);
  return FindTaskBind(task_body).defined() &&
         task_body.as<BindNode>() == nullptr;
}

class CoreExtractor : public StmtMutator {
public:
  static Optional<Stmt> Extract(const Stmt &stmt, CoreMask target_core,
                                bool broadcast_establishes_stream = false) {
    ICHECK(IsConcreteCore(target_core));
    CoreExtractor extractor(target_core, broadcast_establishes_stream);
    Stmt result = extractor.VisitStmt(stmt);
    if (!extractor.found_core_)
      return std::nullopt;
    ICHECK(result.defined()) << "CoreExtractor removed detected core "
                             << static_cast<int>(target_core);
    return result;
  }

private:
  CoreExtractor(CoreMask target_core, bool broadcast_establishes_stream)
      : target_core_(target_core),
        broadcast_establishes_stream_(broadcast_establishes_stream) {}

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> statements;
    for (const Stmt &stmt : op->seq) {
      Stmt rewritten = VisitStmt(stmt);
      if (rewritten.defined())
        statements.push_back(std::move(rewritten));
    }
    return MakeSequence(statements);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    Map<Var, PrimExpr> outer_remap = var_remap_;
    PrimExpr min = Substitute(op->min, var_remap_);
    PrimExpr extent = Substitute(op->extent, var_remap_);
    Optional<PrimExpr> step = op->step;
    if (step.has_value())
      step = Substitute(step.value(), var_remap_);
    Var loop_var = op->loop_var.copy_with_suffix("");
    var_remap_.Set(op->loop_var, loop_var);
    Stmt body = VisitStmt(op->body);
    var_remap_ = std::move(outer_remap);
    if (!body.defined())
      return Stmt();
    return For(loop_var, min, extent, op->kind, std::move(body),
               op->thread_binding, op->annotations, step, op->span);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = Substitute(op->condition, var_remap_);
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case.has_value()) {
      Stmt rewritten_else = VisitStmt(op->else_case.value());
      if (rewritten_else.defined())
        else_case = std::move(rewritten_else);
    }
    if (!then_case.defined()) {
      if (!else_case.has_value())
        return Stmt();
      return IfThenElse(Not(condition), else_case.value(), std::nullopt,
                        op->span);
    }
    return IfThenElse(condition, std::move(then_case), std::move(else_case),
                      op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (IsScheduleUnit(op)) {
      if (GetScheduledCoreMask(op).has_value())
        return ExtractCoreOwnedUnit(op);
      Stmt body = VisitStmt(op->body);
      if (!body.defined())
        return Stmt();
      return AttrStmt(op->node, op->attr_key, op->value, std::move(body),
                      op->span);
    }
    Stmt body = VisitStmt(op->body);
    if (!body.defined())
      return Stmt();
    return SubstituteLetVars(
        AttrStmt(op->node, op->attr_key, op->value, std::move(body), op->span),
        var_remap_);
  }

  Stmt ExtractCoreOwnedUnit(const AttrStmtNode *op) {
    CoreMask task_core_mask = GetUnitCoreMask(op);
    // Broadcast tasks normally follow an existing concrete stream. The caller
    // may allow them to establish the historical Vector fallback when the
    // complete kernel has no concrete task.
    found_core_ |=
        task_core_mask == target_core_ ||
        (broadcast_establishes_stream_ && task_core_mask == kCoreBroadcast);
    if (!HasCore(task_core_mask, target_core_))
      return Stmt();

    Stmt body = SubstituteLetVars(op->body, var_remap_);
    Optional<Bind> bind = FindTaskBind(body);
    if (task_core_mask == kCoreBroadcast && bind.defined()) {
      Var source_var = bind.value()->var;
      Var output_var = source_var.copy_with_suffix("");
      body = ReplaceTaskBind(
          body, Bind(output_var, bind.value()->value, bind.value()->span));
      var_remap_.Set(source_var, output_var);
    }
    return AttrStmt(op->node, op->attr_key, op->value, std::move(body),
                    op->span);
  }

  CoreMask target_core_;
  bool broadcast_establishes_stream_{false};
  bool found_core_{false};
  Map<Var, PrimExpr> var_remap_;
};

class ConditionalLetCollector : public StmtVisitor {
public:
  static Map<Var, Buffer> Collect(const Stmt &stmt) {
    ConditionalLetCollector collector;
    collector(stmt);
    return collector.buffers_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (!IsScheduleUnit(op) || !GetScheduledCoreMask(op).has_value()) {
      StmtVisitor::VisitStmt_(op);
      return;
    }
    Optional<Bind> bind = FindTaskBind(op->body);
    if (!bind.defined() || !IsGuardedTaskBind(op) ||
        bind.value()->value.dtype().is_handle())
      return;
    Var var = bind.value()->var;
    if (!buffers_.count(var)) {
      buffers_.Set(var, decl_buffer({IntImm(DataType::Int(32), 1)},
                                    bind.value()->value.dtype(), var->name_hint,
                                    "local.var"));
    }
  }

  Map<Var, Buffer> buffers_;
};

int GetUnitStage(const Stmt &stmt) {
  Stmt current = stmt;
  while (true) {
    if (const auto *attribute = current.as<AttrStmtNode>()) {
      if (IsScheduleUnit(attribute))
        return GetMarkerStage(attribute);
      current = attribute->body;
      continue;
    }
    if (const auto *conditional = current.as<IfThenElseNode>()) {
      ICHECK(!conditional->else_case.has_value())
          << "A scheduled control guard may not contain an else branch";
      current = conditional->then_case;
      continue;
    }
    LOG(FATAL) << "Expected a `tl.schedule_unit` AttrStmt, got " << current;
  }
}

bool IsScheduledLetUnit(const Stmt &stmt) {
  const auto *attribute = stmt.as<AttrStmtNode>();
  return attribute != nullptr && IsScheduleUnit(attribute) &&
         GetScheduledCoreMask(attribute).has_value() &&
         FindTaskBind(attribute->body).defined();
}

Map<String, ffi::Any>
FilterSchedulingAnnotations(const Map<String, ffi::Any> &annotations) {
  Map<String, ffi::Any> result = annotations;
  result.erase("num_stages");
  result.erase(kMultiBufferEligible);
  result.erase("conflict_hint");
  return result;
}

class PipelineLowerer : public StmtMutator {
public:
  static Stmt Lower(const Stmt &stmt) {
    Map<Var, Buffer> conditional_lets = ConditionalLetCollector::Collect(stmt);
    Map<Var, PrimExpr> substitutions;
    for (const auto &[var, buffer] : conditional_lets) {
      substitutions.Set(var,
                        BufferLoad(buffer, {IntImm(DataType::Int(32), 0)}));
    }
    Stmt rewritten = SubstituteLetVars(stmt, substitutions);
    return PipelineLowerer(std::move(conditional_lets))(rewritten);
  }

private:
  explicit PipelineLowerer(Map<Var, Buffer> conditional_lets)
      : conditional_lets_(std::move(conditional_lets)) {}

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> statements;
    for (const Stmt &stmt : op->seq) {
      Stmt lowered = VisitStmt(stmt);
      if (lowered.defined())
        statements.push_back(std::move(lowered));
    }
    if (statements.empty())
      return Evaluate(0);
    return MakeSequence(statements);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (!IsScheduleUnit(op))
      return StmtMutator::VisitStmt_(op);
    if (!GetScheduledCoreMask(op).has_value())
      return VisitStmt(op->body);
    return LowerCoreOwnedUnit(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    Array<Stmt> children = GetSequence(op->body);
    ICHECK(!children.empty()) << "Scheduled control may not have an empty body";
    int min_stage = std::numeric_limits<int>::max();
    int max_stage = std::numeric_limits<int>::min();
    for (const Stmt &child : children) {
      int stage = GetUnitStage(child);
      min_stage = std::min(min_stage, stage);
      max_stage = std::max(max_stage, stage);
    }

    PrimExpr loop_step =
        op->step.has_value() ? op->step.value() : IntImm(DataType::Int(32), 1);
    Array<Stmt> body;
    for (const Stmt &child : children) {
      int stage = GetUnitStage(child);
      bool is_let = IsScheduledLetUnit(child);
      Stmt lowered = VisitStmt(child);
      ICHECK(lowered.defined());
      if (min_stage == max_stage) {
        body.push_back(std::move(lowered));
        continue;
      }

      Map<Var, PrimExpr> substitution;
      substitution.Set(op->loop_var,
                       op->loop_var -
                           loop_step * (static_cast<int>(stage) - min_stage));
      if (is_let) {
        Map<Var, PrimExpr> clamped_substitution;
        clamped_substitution.Set(
            op->loop_var,
            Max(op->min,
                Min(op->min + op->extent - loop_step,
                    op->loop_var -
                        loop_step * (static_cast<int>(stage) - min_stage))));
        body.push_back(Substitute(std::move(lowered), clamped_substitution));
        continue;
      }

      PrimExpr condition =
          And(op->loop_var < op->min + op->extent, op->loop_var >= op->min);
      if (stage == min_stage)
        condition = op->loop_var < op->min + op->extent;
      if (stage == max_stage)
        condition = op->loop_var >= op->min;
      body.push_back(
          Substitute(IfThenElse(condition, std::move(lowered)), substitution));
    }

    PrimExpr extent = op->extent;
    if (min_stage != max_stage)
      extent = extent + loop_step * (max_stage - min_stage);
    return For(op->loop_var, op->min, extent, op->kind, MakeSequence(body),
               op->thread_binding, FilterSchedulingAnnotations(op->annotations),
               op->step, op->span);
  }

  Stmt LowerCoreOwnedUnit(const AttrStmtNode *op) {
    Optional<Bind> bind = FindTaskBind(op->body);
    if (!bind.defined())
      return op->body;

    Var var = bind.value()->var;
    if (IsGuardedTaskBind(op) && !bind.value()->value.dtype().is_handle()) {
      auto buffer = conditional_lets_.Get(var);
      ICHECK(buffer.has_value())
          << "Missing local.var buffer for guarded scalar Bind " << var;
      Map<String, ffi::Any> annotations;
      annotations.Set(attr::kLocalVarInit, make_zero(buffer.value()->dtype));
      Stmt allocation = AllocBuffer(buffer.value(), annotations);
      Stmt store = ReplaceTaskBind(
          op->body, BufferStore(buffer.value(), bind.value()->value,
                                {IntImm(DataType::Int(32), 0)}));
      return SeqStmt::Flatten(Array<Stmt>{allocation, std::move(store)});
    }

    // Guarded handle bindings must remain unconditional so that their pointer
    // is in scope for later buffer uses. Unguarded bindings are already native.
    return Bind(var, bind.value()->value, bind.value()->span);
  }

  Map<Var, Buffer> conditional_lets_;
};

class TaskMarkerStripper : public StmtMutator {
private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kAscendTask ||
        op->attr_key == attr::kAscendPerCoreTask)
      return VisitStmt(op->body);
    return StmtMutator::VisitStmt_(op);
  }
};

Optional<Stmt> LowerCore(const Stmt &scheduled_body, CoreMask target_core,
                         bool broadcast_establishes_stream = false) {
  Optional<Stmt> extracted = CoreExtractor::Extract(
      scheduled_body, target_core, broadcast_establishes_stream);
  if (!extracted.has_value())
    return std::nullopt;
  Stmt lowered = PipelineLowerer::Lower(extracted.value());
  return TaskMarkerStripper()(std::move(lowered));
}

// Re-write empty AttrStmt markers to nest them properly. A flat Bind remains
// visible to subsequent statements in the same SeqStmt scope.
class AttrStmtNester : public StmtMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> statements;
    for (const Stmt &statement : op->seq)
      statements.push_back(VisitStmt(statement));

    Array<Stmt> flattened;
    for (const Stmt &statement : statements) {
      if (const auto *sequence = statement.as<SeqStmtNode>()) {
        for (const Stmt &inner : sequence->seq)
          flattened.push_back(inner);
      } else {
        flattened.push_back(statement);
      }
    }
    statements = std::move(flattened);

    for (int index = static_cast<int>(statements.size()) - 2; index >= 0;
         --index) {
      if (const auto *attribute = statements[index].as<AttrStmtNode>()) {
        if (IsEmptyBody(attribute->body)) {
          Stmt body = CollectRemaining(statements, index + 1);
          statements =
              TruncateAndReplace(statements, index,
                                 AttrStmt(attribute->node, attribute->attr_key,
                                          attribute->value, std::move(body)));
        }
      }
    }

    if (statements.empty())
      return Evaluate(0);
    return MakeSequence(statements);
  }

private:
  static bool IsEmptyBody(const Stmt &stmt) {
    if (const auto *evaluate = stmt.as<EvaluateNode>()) {
      if (const auto *integer = evaluate->value.as<IntImmNode>())
        return integer->value == 0;
    }
    return false;
  }

  static Stmt CollectRemaining(const Array<Stmt> &statements, int start) {
    if (start >= static_cast<int>(statements.size()))
      return Evaluate(0);
    Array<Stmt> remaining;
    for (int index = start; index < static_cast<int>(statements.size());
         ++index) {
      remaining.push_back(statements[index]);
    }
    return MakeSequence(remaining);
  }

  static Array<Stmt> TruncateAndReplace(const Array<Stmt> &statements,
                                        int index, Stmt replacement) {
    Array<Stmt> result;
    for (int current = 0; current < index; ++current)
      result.push_back(statements[current]);
    result.push_back(std::move(replacement));
    return result;
  }
};

// Recursively flatten all nested SeqStmt nodes before AttrStmtNester absorbs
// empty AttrStmt markers, preventing absorption across nested boundaries.
class SeqStmtFlattener : public StmtMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> visited;
    for (const Stmt &statement : op->seq)
      visited.push_back(VisitStmt(statement));

    Array<Stmt> flattened;
    std::function<void(const Stmt &)> flatten = [&](const Stmt &statement) {
      if (const auto *sequence = statement.as<SeqStmtNode>()) {
        for (const Stmt &inner : sequence->seq)
          flatten(inner);
      } else {
        flattened.push_back(statement);
      }
    };
    for (const Stmt &statement : visited)
      flatten(statement);
    if (flattened.empty())
      return Evaluate(0);
    return MakeSequence(flattened);
  }
};

Stmt ReNestAttrStmts(const Stmt &stmt) {
  Stmt flattened = SeqStmtFlattener()(stmt);
  return AttrStmtNester()(flattened);
}

class MixedKernelSidRelocator : public StmtMutator {
public:
  static Stmt Relocate(const Stmt &stmt) {
    MixedKernelSidRelocator relocator;
    return relocator(stmt);
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    const auto *iter_var = op->node.as<IterVarNode>();
    bool is_outer_cthread =
        op->attr_key == tirx::attr::thread_extent && iter_var != nullptr &&
        iter_var->thread_tag == "cthread" && inside_vector_ == 0;
    if (!is_outer_cthread)
      return StmtMutator::VisitStmt_(op);

    IterVar cthread = GetRef<IterVar>(iter_var);
    cthread_stack_.push_back(cthread);
    Stmt body = VisitStmt(op->body);
    cthread_stack_.pop_back();
    return body;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    bool is_vector = op->name_hint == "VECTOR";
    if (is_vector)
      ++inside_vector_;
    SBlock block = Downcast<SBlock>(StmtMutator::VisitStmt_(op));
    if (is_vector)
      --inside_vector_;
    if (!is_vector || cthread_stack_.empty())
      return block;

    IterVar outer_cthread = cthread_stack_.back();
    IterVar inner_cthread(outer_cthread->dom, outer_cthread->var,
                          outer_cthread->iter_type, outer_cthread->thread_tag);
    Stmt block_body = block->body;
    block.CopyOnWrite()->body =
        AttrStmt(inner_cthread, tirx::attr::thread_extent,
                 outer_cthread->dom->extent, std::move(block_body));
    return block;
  }

  int inside_vector_{0};
  std::vector<IterVar> cthread_stack_;
};

Stmt LowerScheduledKernel(const Stmt &scheduled_body,
                          Optional<Var> external_sid, int vector_count) {
  Optional<Stmt> vector_body = LowerCore(scheduled_body, kCoreVector);
  Optional<Stmt> cube_body = LowerCore(scheduled_body, kCoreCube);
  if (!vector_body.has_value() && !cube_body.has_value()) {
    // Match AssignCore's historical fallback for an all-Broadcast scalar
    // kernel. Broadcast tasks normally follow an existing concrete stream;
    // when no stream exists, establish a Vector stream instead of dropping
    // the complete kernel.
    vector_body = LowerCore(scheduled_body, kCoreVector,
                            /*broadcast_establishes_stream=*/true);
  }
  if (vector_body.has_value() && cube_body.has_value()) {
    Array<Stmt> blocks;
    blocks.push_back(SBlock({}, {}, {}, "CUBE", cube_body.value(), std::nullopt,
                            {}, {}, {}));
    Map<String, ffi::Any> vector_annotations;
    vector_annotations.Set("vector_count", Integer(vector_count));
    if (external_sid.has_value()) {
      blocks.push_back(SBlock({}, {}, {}, "VECTOR", vector_body.value(),
                              std::nullopt, {}, {}, vector_annotations));
    } else {
      Var sid("sid", DataType::Int(32));
      IterVar sid_thread(Range(0, vector_count), sid, IterVarType::kThreadIndex,
                         "cthread");
      Stmt threaded_vector_body =
          AttrStmt(sid_thread, tirx::attr::thread_extent, Integer(vector_count),
                   vector_body.value());
      blocks.push_back(SBlock({}, {}, {}, "VECTOR", threaded_vector_body,
                              std::nullopt, {}, {}, vector_annotations));
    }
    return SeqStmt(blocks);
  }
  if (vector_body.has_value()) {
    return SBlock({}, {}, {}, "VECTOR", vector_body.value(), std::nullopt, {},
                  {}, {});
  }
  if (cube_body.has_value()) {
    return SBlock({}, {}, {}, "CUBE", cube_body.value(), std::nullopt, {}, {},
                  {});
  }
  return Evaluate(0);
}

} // namespace

tvm::transform::Pass LowerScheduledTIR() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    func = RewriteTilelangKernels(
        std::move(func), "LowerScheduledTIR",
        [](const TilelangKernelContext &context) {
          SBlock root = context.root;
          ScheduledTIR scheduled_tir =
              DecodeScheduledTIR(root, context.outer_ctx);
          int vector_count =
              scheduled_tir.metadata.num_aiv_subcores.value_or(2);
          root.CopyOnWrite()->body =
              LowerScheduledKernel(root->body, context.outer_sid, vector_count);
          return root;
        });
    Stmt body = ReNestAttrStmts(func->body);
    func.CopyOnWrite()->body = MixedKernelSidRelocator::Relocate(body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerScheduledTIR", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerScheduledTIR", LowerScheduledTIR);
}

} // namespace tl
} // namespace tvm
