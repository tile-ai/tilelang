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
 * \file warpgroup_partition.cc
 * \brief Warpgroup partition and IRStructure-to-Stmt conversion for TileLang
 * AutoSchedule
 */

#include "./warpgroup_partition.h"

#include "../auto_schedule.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../op/builtin.h"
#include "../../target/utils.h"
#include "../common/attr.h"
#include "../common/collector.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

bool IsLetDeclTask(const TaskNode *task) {
  return task->stmts.size() == 1 && task->stmts[0].as<LetStmtNode>() != nullptr;
}

// Helper: check if an IRStructure node is a LetDecl task (or a ScheduleUnit
// wrapping one)
bool IsLetDeclNode(const IRStructure *node) {
  if (!node)
    return false;
  if (node->IsTask()) {
    return IsLetDeclTask(static_cast<const TaskNode *>(node));
  }
  if (node->IsScheduleUnit()) {
    auto unit = static_cast<const ScheduleUnit *>(node);
    return unit->child && unit->child->IsTask() &&
           IsLetDeclTask(static_cast<const TaskNode *>(unit->child.get()));
  }
  return false;
}

// Helper: check if an IRStructure subtree contains any LetDecl tasks
bool ContainsLetDecl(const IRStructure *node) {
  if (!node)
    return false;
  if (IsLetDeclNode(node))
    return true;
  if (node->IsSequence()) {
    auto seq = static_cast<const SequenceNode *>(node);
    for (const auto &child : seq->children) {
      if (ContainsLetDecl(child.get()))
        return true;
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<const ControlNode *>(node);
    return ContainsLetDecl(ctrl->child.get());
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(node);
    return ContainsLetDecl(wrapper->child.get());
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<const ScheduleUnit *>(node);
    return ContainsLetDecl(unit->child.get());
  }
  return false;
}

// Helper function to clone IRStructure with warpgroup filter.
std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id,
                                    Map<Var, PrimExpr> &var_remap) {
  if (!node)
    return nullptr;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);

    // LetDecl tasks are always included in every warp group clone.
    // Create a fresh variable copy so the two warp groups use different names.
    if (IsLetDeclTask(task)) {
      const auto *let = task->stmts[0].as<LetStmtNode>();
      auto new_var = let->var.copy_with_suffix("");
      // Substitute previously renamed variables in the value expression.
      PrimExpr new_value =
          var_remap.empty() ? let->value : Substitute(let->value, var_remap);
      var_remap.Set(let->var, new_var);
      auto new_task = std::make_shared<TaskNode>();
      new_task->stmts.push_back(LetStmt(new_var, new_value, Evaluate(0)));
      return new_task;
    }

    // Non-LetDecl tasks: only include if warp group matches
    if (!node->containWarpgroupId(warpgroup_id))
      return nullptr;
    auto cloned = task->Clone();
    // Substitute renamed LetDecl variables in task statements
    if (!var_remap.empty()) {
      auto ct = static_cast<TaskNode *>(cloned.get());
      for (size_t i = 0; i < ct->stmts.size(); ++i) {
        ct->stmts[i] = Substitute(ct->stmts[i], var_remap);
      }
    }
    return cloned;
  } else if (node->IsSequence()) {
    // A SequenceNode is included if it contains the target warp group
    // OR if it contains LetDecl tasks (which are always needed).
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto seq = static_cast<SequenceNode *>(node);
    auto new_seq = std::make_shared<SequenceNode>();
    for (const auto &child : seq->children) {
      auto new_child = CloneIRStructureWithWarpgroupFilter(
          child.get(), warpgroup_id, var_remap);
      if (new_child) {
        new_seq->children.push_back(std::move(new_child));
      }
    }
    if (new_seq->children.empty())
      return nullptr;
    return new_seq;
  } else if (node->IsControl()) {
    // A ControlNode is included if it contains the target warp group
    // OR if it contains LetDecl tasks.
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto ctrl = static_cast<ControlNode *>(node);
    auto new_ctrl = std::make_shared<ControlNode>();
    // Apply var_remap to the For statement's min/extent/step so that renamed
    // LetDecl variables are correctly referenced in the loop bounds.
    if (!var_remap.empty()) {
      For new_for = ctrl->control;
      new_for.CopyOnWrite()->min = Substitute(ctrl->control->min, var_remap);
      new_for.CopyOnWrite()->extent =
          Substitute(ctrl->control->extent, var_remap);
      if (ctrl->control->step.has_value()) {
        new_for.CopyOnWrite()->step =
            Substitute(ctrl->control->step.value(), var_remap);
      }
      new_ctrl->control = new_for;
    } else {
      new_ctrl->control = ctrl->control;
    }
    // Clone the task and apply var_remap so each warpgroup gets its own copy
    // with correctly renamed LetDecl variables.
    if (ctrl->task) {
      auto cloned_task =
          std::static_pointer_cast<TaskNode>(ctrl->task->Clone());
      if (!var_remap.empty()) {
        for (size_t i = 0; i < cloned_task->stmts.size(); ++i) {
          cloned_task->stmts[i] = Substitute(cloned_task->stmts[i], var_remap);
        }
      }
      new_ctrl->task = std::move(cloned_task);
    }
    new_ctrl->SetPromote(ctrl->hasPromote());
    new_ctrl->child = CloneIRStructureWithWarpgroupFilter(
        ctrl->child.get(), warpgroup_id, var_remap);
    return new_ctrl;
  } else if (node->IsWrapper()) {
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto wrapper = static_cast<WrapperNode *>(node);
    auto new_wrapper = std::make_shared<WrapperNode>();
    // Apply var_remap to the wrapper statement so that renamed LetDecl
    // variables are correctly substituted in LetStmt values / AttrStmt values.
    new_wrapper->wrapper = var_remap.empty()
                               ? wrapper->wrapper
                               : Substitute(wrapper->wrapper, var_remap);
    new_wrapper->child = CloneIRStructureWithWarpgroupFilter(
        wrapper->child.get(), warpgroup_id, var_remap);
    return new_wrapper;
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    bool child_is_let_decl = IsLetDeclNode(unit->child.get());

    // Include the ScheduleUnit if the child is a LetDecl or the warp group
    // matches.
    if (!child_is_let_decl && !node->containWarpgroupId(warpgroup_id))
      return nullptr;

    auto new_unit = std::make_shared<ScheduleUnit>();
    new_unit->stage = unit->stage;
    new_unit->child = CloneIRStructureWithWarpgroupFilter(
        unit->child.get(), warpgroup_id, var_remap);

    if (!child_is_let_decl) {
      // Copy before/after for the target warp group
      new_unit->before[warpgroup_id] = unit->before[warpgroup_id];
      new_unit->after[warpgroup_id] = unit->after[warpgroup_id];
      // Substitute renamed LetDecl variables in before/after stmts
      if (!var_remap.empty()) {
        for (auto &s : new_unit->before[warpgroup_id]) {
          s = Substitute(s, var_remap);
        }
        for (auto &s : new_unit->after[warpgroup_id]) {
          s = Substitute(s, var_remap);
        }
      }
    }
    return new_unit;
  }
  LOG(FATAL);
}

// Entry point overload — creates a fresh var_remap per call
std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id) {
  Map<Var, PrimExpr> var_remap;
  return CloneIRStructureWithWarpgroupFilter(node, warpgroup_id, var_remap);
}

std::shared_ptr<IRStructure>
RemoveUnusedLetDecls(std::shared_ptr<IRStructure> root) {
  if (!root)
    return nullptr;

  // Phase 1: Collect LetDecl definitions and variable references from
  // non-LetDecl nodes (task stmts and ScheduleUnit before/after).
  struct LetDeclEntry {
    const VarNode *var;
    PrimExpr value;
  };
  std::vector<LetDeclEntry> let_decls;
  std::unordered_set<const VarNode *> referenced_vars;

  std::function<void(const IRStructure *)> collect =
      [&](const IRStructure *node) {
        if (!node)
          return;
        if (node->IsTask()) {
          auto task = static_cast<const TaskNode *>(node);
          if (IsLetDeclTask(task)) {
            const auto *let = task->stmts[0].as<LetStmtNode>();
            let_decls.push_back({let->var.get(), let->value});
          } else {
            VarRefCollector collector;
            for (const auto &stmt : task->stmts) {
              collector(stmt);
            }
            referenced_vars.insert(collector.vars.begin(),
                                   collector.vars.end());
          }
        } else if (node->IsSequence()) {
          for (const auto &child :
               static_cast<const SequenceNode *>(node)->children) {
            collect(child.get());
          }
        } else if (node->IsControl()) {
          auto ctrl = static_cast<const ControlNode *>(node);
          collect(ctrl->task.get());
          collect(ctrl->child.get());
          // Also collect variable references from the For loop bounds
          // (min, extent, step) so their LetDecls are not removed.
          VarRefCollector for_collector;
          for_collector(ctrl->control->min);
          for_collector(ctrl->control->extent);
          if (ctrl->control->step.has_value()) {
            for_collector(ctrl->control->step.value());
          }
          referenced_vars.insert(for_collector.vars.begin(),
                                 for_collector.vars.end());
        } else if (node->IsWrapper()) {
          auto wrapper = static_cast<const WrapperNode *>(node);
          collect(wrapper->task.get());
          collect(wrapper->child.get());
          // Also collect variable references from the wrapper statement
          // (LetStmt value / AttrStmt value) so their LetDecls are not removed.
          VarRefCollector wrapper_collector;
          wrapper_collector(wrapper->wrapper);
          referenced_vars.insert(wrapper_collector.vars.begin(),
                                 wrapper_collector.vars.end());
        } else if (node->IsScheduleUnit()) {
          auto unit = static_cast<const ScheduleUnit *>(node);
          collect(unit->child.get());
          VarRefCollector collector;
          for (const auto &stmts : unit->before) {
            for (const auto &s : stmts)
              collector(s);
          }
          for (const auto &stmts : unit->after) {
            for (const auto &s : stmts)
              collector(s);
          }
          referenced_vars.insert(collector.vars.begin(), collector.vars.end());
        }
      };
  collect(root.get());

  // Phase 2: Transitive closure — if a LetDecl var is referenced,
  // all vars in its value expression are transitively referenced too.
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &entry : let_decls) {
      if (referenced_vars.count(entry.var)) {
        VarRefCollector collector;
        collector(entry.value);
        for (const auto *v : collector.vars) {
          if (!referenced_vars.count(v)) {
            referenced_vars.insert(v);
            changed = true;
          }
        }
      }
    }
  }

  // Phase 3: Filter the tree — remove LetDecl tasks for unused vars.
  std::function<std::shared_ptr<IRStructure>(
      const std::shared_ptr<IRStructure> &)>
      filter_tree = [&](const std::shared_ptr<IRStructure> &node)
      -> std::shared_ptr<IRStructure> {
    if (!node)
      return nullptr;
    if (node->IsTask()) {
      if (IsLetDeclTask(static_cast<const TaskNode *>(node.get()))) {
        const auto *let = static_cast<const TaskNode *>(node.get())
                              ->stmts[0]
                              .as<LetStmtNode>();
        if (!referenced_vars.count(let->var.get())) {
          return nullptr; // Remove unused LetDecl
        }
      }
      return node;
    } else if (node->IsSequence()) {
      auto seq = static_cast<const SequenceNode *>(node.get());
      auto new_seq = std::make_shared<SequenceNode>();
      for (const auto &child : seq->children) {
        auto filtered = filter_tree(child);
        if (filtered)
          new_seq->children.push_back(std::move(filtered));
      }
      if (new_seq->children.empty())
        return nullptr;
      return new_seq;
    } else if (node->IsControl()) {
      auto ctrl = static_cast<const ControlNode *>(node.get());
      auto new_ctrl = std::make_shared<ControlNode>();
      new_ctrl->control = ctrl->control;
      new_ctrl->task = ctrl->task;
      new_ctrl->SetPromote(ctrl->hasPromote());
      new_ctrl->child = filter_tree(ctrl->child);
      if (!new_ctrl->child)
        return nullptr;
      return new_ctrl;
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<const WrapperNode *>(node.get());
      auto new_wrapper = std::make_shared<WrapperNode>();
      new_wrapper->wrapper = wrapper->wrapper;
      new_wrapper->task = wrapper->task;
      new_wrapper->child = filter_tree(wrapper->child);
      return new_wrapper;
    } else if (node->IsScheduleUnit()) {
      auto unit = static_cast<const ScheduleUnit *>(node.get());
      auto new_unit = std::make_shared<ScheduleUnit>();
      new_unit->stage = unit->stage;
      new_unit->before = unit->before;
      new_unit->after = unit->after;
      new_unit->child = filter_tree(unit->child);
      if (!new_unit->child)
        return nullptr;
      return new_unit;
    }
    return node;
  };

  return filter_tree(root);
}

class SimtCopyDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    SimtCopyDetector detector;
    detector.VisitStmt(stmt);
    return detector.has_simt_copy_;
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    auto scope =
        runtime::StorageScope::Create(GetPtrStorageScope(op->buffer->data));
    if (scope.to_string() != "global") {
      has_simt_copy_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool has_simt_copy_{false};
};

Stmt ConvertIRStructureToStmt(IRStructure *root, const bool outer_enable_epi) {
  std::function<Stmt(IRStructure *)> irstructure_to_stmt;
  irstructure_to_stmt = [&irstructure_to_stmt,
                         outer_enable_epi](IRStructure *structure) -> Stmt {
    if (!structure) {
      return Evaluate(0);
    }

    if (structure->IsTask()) {
      auto task = static_cast<TaskNode *>(structure);
      if (task->stmts.empty()) {
        return Evaluate(0);
      } else if (task->stmts.size() == 1) {
        return task->stmts[0];
      } else {
        return SeqStmt(task->stmts);
      }
    } else if (structure->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(structure);
      std::vector<Stmt> stmts;
      for (const auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        for (auto &before : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        Stmt child_stmt = irstructure_to_stmt(unit->child.get());
        stmts.push_back(child_stmt);
        for (auto &after : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
      }
      auto flattened = SeqStmt::Flatten(stmts);
      return flattened;
    } else if (structure->IsControl()) {
      auto ctrl = static_cast<ControlNode *>(structure);
      Var loop_var = ctrl->control->loop_var;
      PrimExpr loop_start = ctrl->control->min;
      PrimExpr loop_extent = ctrl->control->extent;
      PrimExpr loop_step = ctrl->control->step.has_value()
                               ? ctrl->control->step.value()
                               : IntImm(DataType::Int(32), 1);
      int min_stages = 100, max_stages = -1;
      if (ctrl->child->IsSequence()) {
        auto seq = static_cast<SequenceNode *>(ctrl->child.get());
        for (auto &child : seq->children) {
          auto unit = static_cast<ScheduleUnit *>(child.get());
          min_stages = std::min(min_stages, unit->stage);
          max_stages = std::max(max_stages, unit->stage);
        }
      }
      if (!ctrl->hasPromote() || !ctrl->child->IsSequence() ||
          min_stages == max_stages) {
        std::vector<Stmt> stmts;
        if (ctrl->child->IsScheduleUnit()) {
          auto unit = static_cast<ScheduleUnit *>(ctrl->child.get());
          for (auto &before : unit->before) {
            for (auto &stmt : before) {
              stmts.push_back(stmt);
            }
          }
          stmts.push_back(irstructure_to_stmt(unit->child.get()));
          for (auto &after : unit->after) {
            for (auto &stmt : after) {
              stmts.push_back(stmt);
            }
          }
        } else if (ctrl->child->IsSequence()) {
          auto seq = static_cast<SequenceNode *>(ctrl->child.get());
          for (auto &child : seq->children) {
            ICHECK(child->IsScheduleUnit());
            auto unit = static_cast<ScheduleUnit *>(child.get());
            for (auto &before : unit->before) {
              for (auto &stmt : before) {
                stmts.push_back(stmt);
              }
            }
            stmts.push_back(irstructure_to_stmt(unit->child.get()));
            for (auto &after : unit->after) {
              for (auto &stmt : after) {
                stmts.push_back(stmt);
              }
            }
          }
        } else {
          LOG(FATAL);
        }
        Stmt body = SeqStmt::Flatten(stmts);
        // Filter out "num_stages" annotation
        Map<String, Any> filtered_annotations = ctrl->control->annotations;
        filtered_annotations.erase("num_stages");
        return For(loop_var, loop_start, loop_extent, ctrl->control->kind, body,
                   ctrl->control->thread_binding, filtered_annotations);
      }
      auto seq = static_cast<SequenceNode *>(ctrl->child.get());
      Stmt body = Evaluate(0);
      std::vector<std::vector<Stmt>> unit_stages;
      unit_stages.resize(max_stages - min_stages + 1);
      for (auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        std::vector<Stmt> stmts;
        for (auto &before : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        stmts.push_back(irstructure_to_stmt(unit->child.get()));
        for (auto &after : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
        unit_stages[unit->stage - min_stages].push_back(
            SeqStmt::Flatten(stmts));
      }
      // Check if any task in this control node contains loop_break
      // If any task contains loop_break, disable prologue
      std::function<bool(IRStructure *)> check_contains_loop_break;
      check_contains_loop_break =
          [&check_contains_loop_break](IRStructure *structure) -> bool {
        if (!structure)
          return false;

        if (structure->IsTask()) {
          auto task = static_cast<TaskNode *>(structure);
          return task->ContainsLoopBreak();
        } else if (structure->IsSequence()) {
          auto seq = static_cast<SequenceNode *>(structure);
          for (const auto &child : seq->children) {
            auto unit = static_cast<ScheduleUnit *>(child.get());
            if (check_contains_loop_break(unit->child.get())) {
              return true;
            }
          }
          return false;
        } else if (structure->IsScheduleUnit()) {
          auto unit = static_cast<ScheduleUnit *>(structure);
          return check_contains_loop_break(unit->child.get());
        } else if (structure->IsControl()) {
          auto ctrl = static_cast<ControlNode *>(structure);
          return check_contains_loop_break(ctrl->child.get());
        } else if (structure->IsWrapper()) {
          auto wrapper = static_cast<WrapperNode *>(structure);
          return check_contains_loop_break(wrapper->child.get());
        }
        return false;
      };

      // Set enable_pro to true only if:
      // 1. No task contains loop_break
      // 2. Loop boundaries (min and extent) are constants
      bool enable_pro = !check_contains_loop_break(ctrl->child.get());

      // Check if loop boundaries are constants
      bool loop_min_is_const = tir::is_const_int(loop_start);
      bool loop_extent_is_const = tir::is_const_int(loop_extent);

      if (!loop_min_is_const || !loop_extent_is_const) {
        enable_pro = false;
      }

      bool enable_epi = outer_enable_epi && enable_pro;
      std::vector<Stmt> steady;

      for (auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        std::vector<Stmt> stmts;
        for (auto &before : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        stmts.push_back(irstructure_to_stmt(unit->child.get()));
        for (auto &after : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
        Map<Var, PrimExpr> substitution, substitution_cond;
        substitution.Set(loop_var,
                         loop_var - loop_step * (max_stages - unit->stage));
        substitution_cond.Set(
            loop_var,
            Max(loop_start,
                Min(loop_start + loop_extent - loop_step,
                    loop_var - loop_step * (max_stages - unit->stage))));
        if (IsLetDeclNode(unit->child.get())) {
          Stmt stmt = SeqStmt::Flatten(stmts);
          steady.push_back(Substitute(stmt, substitution_cond));
        } else {
          PrimExpr condition =
              And(loop_var < loop_start + loop_extent, loop_var >= loop_start);
          if (unit->stage == min_stages) {
            condition = loop_var >= loop_start;
          }
          if (unit->stage == max_stages) {
            condition = loop_var < loop_start + loop_extent;
          }
          Stmt stmt = IfThenElse(condition, SeqStmt::Flatten(stmts));
          steady.push_back(Substitute(stmt, substitution));
        }
      }
      Stmt new_body = SeqStmt::Flatten(steady);
      auto new_var = loop_var.copy_with_suffix("");
      // Filter out "num_stages" annotation
      Map<String, Any> filtered_annotations = ctrl->control->annotations;
      filtered_annotations.erase("num_stages");
      Map<Var, PrimExpr> substitution;
      substitution.Set(loop_var, new_var);
      For for_op =
          For(new_var, loop_start,
              ctrl->control->extent + loop_step * (max_stages - min_stages),
              ctrl->control->kind, Substitute(new_body, substitution),
              ctrl->control->thread_binding, filtered_annotations);

      Stmt prologue = Evaluate(0);
      if (enable_pro) {
        Map<Var, PrimExpr> sub;
        For new_for = for_op;
        auto pro = loop_var.copy_with_suffix("_prologue");
        sub.Set(new_var, pro);
        new_for.CopyOnWrite()->loop_var = pro;
        new_for.CopyOnWrite()->kind = ForKind::kUnrolled;
        new_for.CopyOnWrite()->extent =
            min(max_stages - min_stages, for_op.get()->extent);
        for_op.CopyOnWrite()->min += loop_step * (max_stages - min_stages);
        for_op.CopyOnWrite()->extent =
            max(0, for_op.get()->extent - (max_stages - min_stages));
        prologue = Substitute(new_for, sub);
      }
      Stmt epilogue = Evaluate(0);
      if (enable_epi) {
        Map<Var, PrimExpr> sub;
        For new_for = for_op;
        auto epi = loop_var.copy_with_suffix("_epilogue");
        sub.Set(new_var, epi);
        new_for.CopyOnWrite()->loop_var = epi;
        new_for.CopyOnWrite()->kind = ForKind::kUnrolled;
        new_for.CopyOnWrite()->min =
            for_op.get()->min +
            loop_step * (for_op.get()->extent - (max_stages - min_stages));
        new_for.CopyOnWrite()->extent =
            min(max_stages - min_stages, for_op.get()->extent);
        for_op.CopyOnWrite()->extent =
            max(0, for_op.get()->extent - (max_stages - min_stages));
        epilogue = Substitute(new_for, sub);
      }
      return SeqStmt({prologue, for_op, epilogue});
    } else if (structure->IsWrapper()) {
      auto wrapper = static_cast<const WrapperNode *>(structure);
      Stmt body = Evaluate(0);
      if (wrapper->child) {
        body = irstructure_to_stmt(wrapper->child.get());
      }
      if (const auto *let = wrapper->wrapper.as<LetStmtNode>()) {
        return LetStmt(let->var, let->value, body);
      } else if (const auto *attr = wrapper->wrapper.as<AttrStmtNode>()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value, body);
      } else {
        LOG(FATAL);
      }
    }

    LOG(FATAL)
        << "Failed to convert IRStructure to Stmt, returning empty statement";
    return Evaluate(0);
  };

  return irstructure_to_stmt(root);
}

// Apply warpgroup partition to entire IRStructure (top-level IfThenElse)
Stmt ApplyWarpgroupPartitionToIRStructure(
    IRStructure *root, IterVar thread_var, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, const bool outer_enable_epi,
    PrimExpr thread_count[2], bool producer_consumer,
    const WarpSpecializeConfig &config, Buffer neutral_sync_shared_barrier) {
  if (!root)
    return Evaluate(0);

  if (root->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(root);
    Stmt body = Evaluate(0);
    if (wrapper->child) {
      body = ApplyWarpgroupPartitionToIRStructure(
          wrapper->child.get(), thread_var, barrier_buffers, barrier_map,
          outer_enable_epi, thread_count, producer_consumer, config,
          neutral_sync_shared_barrier);
    }
    if (const auto *let = wrapper->wrapper.as<LetStmtNode>()) {
      return LetStmt(let->var, let->value, body);
    } else if (const auto *attr = wrapper->wrapper.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    } else {
      LOG(FATAL);
    }
  }

  // Check if there are tasks with mixed warpgroup ids
  std::vector<TaskNodeWithContext> all_tasks;
  CollectAllTaskNodesWithContext(root, all_tasks);

  bool has_warpgroup0 = false;
  bool has_warpgroup1 = false;
  bool has_warpgroup_neutral = false;
  for (auto &task : all_tasks) {
    int wg_id = task.task->GetWarpgroupId();
    if (wg_id == 0)
      has_warpgroup0 = true;
    else if (wg_id == 1)
      has_warpgroup1 = true;
    else if (wg_id == -1)
      has_warpgroup_neutral = true;
  }

  // Convert IRStructure to Stmt for IfThenElse
  std::function<Stmt(IRStructure *)> irstructure_to_stmt;
  irstructure_to_stmt = [&irstructure_to_stmt,
                         outer_enable_epi](IRStructure *structure) -> Stmt {
    if (!structure) {
      return Evaluate(0);
    }

    if (structure->IsTask()) {
      auto task = static_cast<TaskNode *>(structure);
      if (task->stmts.empty()) {
        return Evaluate(0);
      } else if (task->stmts.size() == 1) {
        return task->stmts[0];
      } else {
        return SeqStmt(task->stmts);
      }
    } else if (structure->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(structure);
      std::vector<Stmt> stmts;
      for (const auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        for (auto &before : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        Stmt child_stmt = irstructure_to_stmt(unit->child.get());
        stmts.push_back(child_stmt);
        for (auto &after : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
      }
      auto flattened = SeqStmt::Flatten(stmts);
      return flattened;
    } else if (structure->IsControl()) {
      auto ctrl = static_cast<ControlNode *>(structure);
      Var loop_var = ctrl->control->loop_var;
      PrimExpr loop_start = ctrl->control->min;
      PrimExpr loop_extent = ctrl->control->extent;
      PrimExpr loop_step = ctrl->control->step.has_value()
                               ? ctrl->control->step.value()
                               : IntImm(DataType::Int(32), 1);
      int min_stages = 100, max_stages = -1;
      if (ctrl->child->IsSequence()) {
        auto seq = static_cast<SequenceNode *>(ctrl->child.get());
        for (auto &child : seq->children) {
          auto unit = static_cast<ScheduleUnit *>(child.get());
          min_stages = std::min(min_stages, unit->stage);
          max_stages = std::max(max_stages, unit->stage);
        }
      }
      if (!ctrl->hasPromote() || !ctrl->child->IsSequence() ||
          min_stages == max_stages) {
        std::vector<Stmt> stmts;
        if (ctrl->child->IsScheduleUnit()) {
          auto unit = static_cast<ScheduleUnit *>(ctrl->child.get());
          for (auto &before : unit->before) {
            for (auto &stmt : before) {
              stmts.push_back(stmt);
            }
          }
          stmts.push_back(irstructure_to_stmt(unit->child.get()));
          for (auto &after : unit->after) {
            for (auto &stmt : after) {
              stmts.push_back(stmt);
            }
          }
        } else if (ctrl->child->IsSequence()) {
          auto seq = static_cast<SequenceNode *>(ctrl->child.get());
          for (auto &child : seq->children) {
            ICHECK(child->IsScheduleUnit());
            auto unit = static_cast<ScheduleUnit *>(child.get());
            for (auto &before : unit->before) {
              for (auto &stmt : before) {
                stmts.push_back(stmt);
              }
            }
            stmts.push_back(irstructure_to_stmt(unit->child.get()));
            for (auto &after : unit->after) {
              for (auto &stmt : after) {
                stmts.push_back(stmt);
              }
            }
          }
        } else {
          LOG(FATAL);
        }
        Stmt body = SeqStmt::Flatten(stmts);
        // Filter out "num_stages" annotation
        Map<String, Any> filtered_annotations = ctrl->control->annotations;
        filtered_annotations.erase("num_stages");
        return For(loop_var, loop_start, loop_extent, ctrl->control->kind, body,
                   ctrl->control->thread_binding, filtered_annotations);
      }
      auto seq = static_cast<SequenceNode *>(ctrl->child.get());
      Stmt body = Evaluate(0);
      std::vector<std::vector<Stmt>> unit_stages;
      unit_stages.resize(max_stages - min_stages + 1);
      for (auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        std::vector<Stmt> stmts;
        for (auto &before : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        stmts.push_back(irstructure_to_stmt(unit->child.get()));
        for (auto &after : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
        unit_stages[unit->stage - min_stages].push_back(
            SeqStmt::Flatten(stmts));
      }
      // Check if any task in this control node contains loop_break
      // If any task contains loop_break, disable prologue
      std::function<bool(IRStructure *)> check_contains_loop_break;
      check_contains_loop_break =
          [&check_contains_loop_break](IRStructure *structure) -> bool {
        if (!structure)
          return false;

        if (structure->IsTask()) {
          auto task = static_cast<TaskNode *>(structure);
          return task->ContainsLoopBreak();
        } else if (structure->IsSequence()) {
          auto seq = static_cast<SequenceNode *>(structure);
          for (const auto &child : seq->children) {
            auto unit = static_cast<ScheduleUnit *>(child.get());
            if (check_contains_loop_break(unit->child.get())) {
              return true;
            }
          }
          return false;
        } else if (structure->IsScheduleUnit()) {
          auto unit = static_cast<ScheduleUnit *>(structure);
          return check_contains_loop_break(unit->child.get());
        } else if (structure->IsControl()) {
          auto ctrl = static_cast<ControlNode *>(structure);
          return check_contains_loop_break(ctrl->child.get());
        } else if (structure->IsWrapper()) {
          auto wrapper = static_cast<WrapperNode *>(structure);
          return check_contains_loop_break(wrapper->child.get());
        }
        return false;
      };

      // Set enable_pro to true only if:
      // 1. No task contains loop_break
      // 2. Loop boundaries (min and extent) are constants
      bool enable_pro = !check_contains_loop_break(ctrl->child.get());

      // Check if loop boundaries are constants
      bool loop_min_is_const = tir::is_const_int(loop_start);
      bool loop_extent_is_const = tir::is_const_int(loop_extent);

      if (!loop_min_is_const || !loop_extent_is_const) {
        enable_pro = false;
      }

      bool enable_epi = outer_enable_epi && enable_pro;
      std::vector<Stmt> steady;

      for (auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        std::vector<Stmt> stmts;
        for (auto &before : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        stmts.push_back(irstructure_to_stmt(unit->child.get()));
        for (auto &after : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
        Map<Var, PrimExpr> substitution, substitution_cond;
        substitution.Set(loop_var,
                         loop_var - loop_step * (max_stages - unit->stage));
        substitution_cond.Set(
            loop_var,
            Max(loop_start,
                Min(loop_start + loop_extent - loop_step,
                    loop_var - loop_step * (max_stages - unit->stage))));
        if (IsLetDeclNode(unit->child.get())) {
          Stmt stmt = SeqStmt::Flatten(stmts);
          steady.push_back(Substitute(stmt, substitution_cond));
        } else {
          PrimExpr condition =
              And(loop_var < loop_start + loop_extent, loop_var >= loop_start);
          if (unit->stage == min_stages) {
            condition = loop_var >= loop_start;
          }
          if (unit->stage == max_stages) {
            condition = loop_var < loop_start + loop_extent;
          }
          Stmt stmt = IfThenElse(condition, SeqStmt::Flatten(stmts));
          steady.push_back(Substitute(stmt, substitution));
        }
      }
      Stmt new_body = SeqStmt::Flatten(steady);
      auto new_var = loop_var.copy_with_suffix("");
      // Filter out "num_stages" annotation
      Map<String, Any> filtered_annotations = ctrl->control->annotations;
      filtered_annotations.erase("num_stages");
      Map<Var, PrimExpr> substitution;
      substitution.Set(loop_var, new_var);
      For for_op =
          For(new_var, loop_start,
              ctrl->control->extent + loop_step * (max_stages - min_stages),
              ctrl->control->kind, Substitute(new_body, substitution),
              ctrl->control->thread_binding, filtered_annotations);

      Stmt prologue = Evaluate(0);
      if (enable_pro) {
        Map<Var, PrimExpr> sub;
        For new_for = for_op;
        auto pro = loop_var.copy_with_suffix("_prologue");
        sub.Set(new_var, pro);
        new_for.CopyOnWrite()->loop_var = pro;
        new_for.CopyOnWrite()->kind = ForKind::kUnrolled;
        new_for.CopyOnWrite()->extent =
            min(max_stages - min_stages, for_op.get()->extent);
        for_op.CopyOnWrite()->min += loop_step * (max_stages - min_stages);
        for_op.CopyOnWrite()->extent =
            max(0, for_op.get()->extent - (max_stages - min_stages));
        prologue = Substitute(new_for, sub);
      }
      Stmt epilogue = Evaluate(0);
      if (enable_epi) {
        Map<Var, PrimExpr> sub;
        For new_for = for_op;
        auto epi = loop_var.copy_with_suffix("_epilogue");
        sub.Set(new_var, epi);
        new_for.CopyOnWrite()->loop_var = epi;
        new_for.CopyOnWrite()->kind = ForKind::kUnrolled;
        new_for.CopyOnWrite()->min =
            for_op.get()->min +
            loop_step * (for_op.get()->extent - (max_stages - min_stages));
        new_for.CopyOnWrite()->extent =
            min(max_stages - min_stages, for_op.get()->extent);
        for_op.CopyOnWrite()->extent =
            max(0, for_op.get()->extent - (max_stages - min_stages));
        epilogue = Substitute(new_for, sub);
      }
      return SeqStmt({prologue, for_op, epilogue});
    } else if (structure->IsWrapper()) {
      auto wrapper = static_cast<const WrapperNode *>(structure);
      Stmt body = Evaluate(0);
      if (wrapper->child) {
        body = irstructure_to_stmt(wrapper->child.get());
      }
      if (const auto *let = wrapper->wrapper.as<LetStmtNode>()) {
        return LetStmt(let->var, let->value, body);
      } else if (const auto *attr = wrapper->wrapper.as<AttrStmtNode>()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value, body);
      } else {
        LOG(FATAL);
      }
    }

    LOG(FATAL)
        << "Failed to convert IRStructure to Stmt, returning empty statement";
    return Evaluate(0);
  };

  // If all tasks belong to the same warpgroup, no partition needed
  if (!(has_warpgroup0 && has_warpgroup1)) {
    return irstructure_to_stmt(root);
  }

  // Helper function to clone IRStructure filtering tasks with warpgroup_id ==
  // -1 (neutral tasks)
  std::function<std::shared_ptr<IRStructure>(IRStructure *)>
      clone_neutral_filter;
  clone_neutral_filter =
      [&clone_neutral_filter](
          IRStructure *node) -> std::shared_ptr<IRStructure> {
    if (!node)
      return nullptr;

    if (node->IsTask()) {
      auto task = static_cast<TaskNode *>(node);
      if (task->GetWarpgroupId() == -1) {
        return task->Clone();
      } else {
        auto new_task = std::make_shared<TaskNode>();
        // Empty statements
        return new_task;
      }
    } else if (node->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(node);
      auto new_seq = std::make_shared<SequenceNode>();
      for (const auto &child : seq->children) {
        if (child) {
          auto node = static_cast<ScheduleUnit *>(child.get());
          auto new_node = clone_neutral_filter(node->child.get());
          if (new_node) {
            auto new_unit = std::make_shared<ScheduleUnit>();
            new_unit->child = std::move(new_node);
            new_seq->children.push_back(std::move(new_unit));
          }
        }
      }
      return new_seq;
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<WrapperNode *>(node);
      auto new_wrapper = std::make_shared<WrapperNode>();
      new_wrapper->child = clone_neutral_filter(wrapper->child.get());
      if (new_wrapper->child) {
        return new_wrapper;
      }
      return nullptr;
    } else if (node->IsControl()) {
      return nullptr;
    }
    LOG(FATAL);
  };

  auto has_actual_statements = [](IRStructure *node) -> bool {
    std::vector<TaskNodeWithContext> tasks;
    CollectAllTaskNodesWithContext(node, tasks);
    for (auto &task : tasks) {
      if (!task.task->stmts.empty()) {
        return true;
      }
    }
    return false;
  };

  std::function<std::shared_ptr<IRStructure>(
      IRStructure *, const std::function<bool(int)> &, int)>
      clone_neutral_filter_with_top_level;
  clone_neutral_filter_with_top_level =
      [&clone_neutral_filter_with_top_level, &clone_neutral_filter](
          IRStructure *node, const std::function<bool(int)> &include_top_level,
          int top_level_index) -> std::shared_ptr<IRStructure> {
    if (!node)
      return nullptr;

    if (node->IsTask()) {
      if (include_top_level(top_level_index)) {
        return clone_neutral_filter(node);
      } else {
        auto new_task = std::make_shared<TaskNode>();
        // Empty statements
        return new_task;
      }
    } else if (node->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(node);
      auto new_seq = std::make_shared<SequenceNode>();
      int child_index = 0;
      for (const auto &child : seq->children) {
        if (child) {
          auto schedule_unit = static_cast<ScheduleUnit *>(child.get());
          int next_top_level_index =
              top_level_index == -1 ? child_index : top_level_index;
          auto new_node = clone_neutral_filter_with_top_level(
              schedule_unit->child.get(), include_top_level,
              next_top_level_index);
          if (new_node) {
            auto new_unit = std::make_shared<ScheduleUnit>();
            new_unit->child = std::move(new_node);
            new_seq->children.push_back(std::move(new_unit));
          }
        }
        child_index++;
      }
      return new_seq;
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<WrapperNode *>(node);
      auto new_wrapper = std::make_shared<WrapperNode>();
      new_wrapper->child = clone_neutral_filter_with_top_level(
          wrapper->child.get(), include_top_level, top_level_index);
      if (new_wrapper->child) {
        return new_wrapper;
      }
      return nullptr;
    } else if (node->IsControl()) {
      return nullptr;
    }
    LOG(FATAL);
  };

  int last_warpgroup_task_top_level_index = -1;
  if (root->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(root);
    for (size_t i = 0; i < seq->children.size(); ++i) {
      const auto &child = seq->children[i];
      if (!child) {
        continue;
      }
      auto unit = static_cast<ScheduleUnit *>(child.get());
      std::vector<TaskNodeWithContext> child_tasks;
      CollectAllTaskNodesWithContext(unit->child.get(), child_tasks);
      for (const auto &task : child_tasks) {
        if (task.task->GetWarpgroupId() >= 0) {
          last_warpgroup_task_top_level_index = static_cast<int>(i);
        }
      }
    }
  }

  auto is_epi_top_level_index =
      [last_warpgroup_task_top_level_index](int top_level_index) {
        return last_warpgroup_task_top_level_index >= 0 &&
               top_level_index > last_warpgroup_task_top_level_index;
      };
  auto is_pro_top_level_index = [is_epi_top_level_index](int top_level_index) {
    return !is_epi_top_level_index(top_level_index);
  };

  auto wg_pro_neutral_structure = has_warpgroup_neutral
                                      ? clone_neutral_filter_with_top_level(
                                            root, is_pro_top_level_index, -1)
                                      : nullptr;
  auto wg_epi_neutral_structure = has_warpgroup_neutral
                                      ? clone_neutral_filter_with_top_level(
                                            root, is_epi_top_level_index, -1)
                                      : nullptr;

  auto wg0_structure =
      RemoveUnusedLetDecls(CloneIRStructureWithWarpgroupFilter(root, 0));
  auto wg1_structure =
      RemoveUnusedLetDecls(CloneIRStructureWithWarpgroupFilter(root, 1));

  bool wg_pro_neutral_has_stmts =
      wg_pro_neutral_structure
          ? has_actual_statements(wg_pro_neutral_structure.get())
          : false;
  bool wg_epi_neutral_has_stmts =
      wg_epi_neutral_structure
          ? has_actual_statements(wg_epi_neutral_structure.get())
          : false;
  bool wg0_has_stmts = has_actual_statements(wg0_structure.get());
  bool wg1_has_stmts = has_actual_statements(wg1_structure.get());

  PrimExpr condition = thread_var->var < thread_count[0];
  PrimExpr wg1_condition =
      thread_var->var < (thread_count[0] + thread_count[1]);

  Stmt pro_neutral_body =
      wg_pro_neutral_has_stmts
          ? irstructure_to_stmt(wg_pro_neutral_structure.get())
          : Evaluate(0);
  Stmt epi_neutral_body =
      wg_epi_neutral_has_stmts
          ? irstructure_to_stmt(wg_epi_neutral_structure.get())
          : Evaluate(0);

  // --- Segment the wg0/wg1 structures by ControlNode (for-loop) boundaries ---
  // This produces multiple IfThenElse blocks separated by liveness boundary
  // markers, so that the merge-shared-memory pass can reuse buffers across
  // segments whose lifetimes do not overlap.

  // Helper: segment a top-level SequenceNode's children into groups separated
  // by ControlNode boundaries. Each ControlNode becomes its own segment;
  // consecutive non-ControlNode children are grouped together.
  auto SegmentSequenceChildren = [](IRStructure *structure)
      -> std::vector<std::vector<std::shared_ptr<IRStructure>>> {
    std::vector<std::vector<std::shared_ptr<IRStructure>>> segments;
    if (!structure || !structure->IsSequence()) {
      return segments;
    }
    auto seq = static_cast<SequenceNode *>(structure);

    std::vector<std::shared_ptr<IRStructure>> current;
    for (auto &child : seq->children) {
      auto unit = static_cast<ScheduleUnit *>(child.get());
      if (unit->child && unit->child->IsControl()) {
        if (!current.empty()) {
          segments.push_back(std::move(current));
          current = {};
        }
        segments.push_back({child});
      } else {
        current.push_back(child);
      }
    }
    if (!current.empty()) {
      segments.push_back(std::move(current));
    }

    return segments;
  };

  // Helper: wrap a list of ScheduleUnit children back into a temporary
  // SequenceNode and convert to Stmt.
  auto SegmentToStmt =
      [&irstructure_to_stmt](
          const std::vector<std::shared_ptr<IRStructure>> &children) -> Stmt {
    if (children.empty())
      return Evaluate(0);
    // Even for a single child we go through the SequenceNode path so that
    // ScheduleUnit before/after stmts are emitted correctly.
    auto tmp_seq = std::make_shared<SequenceNode>();
    tmp_seq->children = children;
    return irstructure_to_stmt(tmp_seq.get());
  };

  // Helper: build a single IfThenElse (with wg1 nesting) from a pair of Stmts.
  auto MakeWarpgroupIf = [&condition, &wg1_condition](Stmt wg0_stmt,
                                                      Stmt wg1_stmt) -> Stmt {
    bool wg0_valid = !IsEvaluateZero(wg0_stmt);
    bool wg1_valid = !IsEvaluateZero(wg1_stmt);
    if (wg0_valid && wg1_valid) {
      return IfThenElse(condition, wg0_stmt,
                        IfThenElse(wg1_condition, wg1_stmt, Evaluate(0)));
    } else if (wg0_valid) {
      return IfThenElse(condition, wg0_stmt);
    } else if (wg1_valid) {
      return IfThenElse(wg1_condition, wg1_stmt);
    }
    return Evaluate(0);
  };

  // Helper: collect LetDecl {Var, PrimExpr} pairs from a segment's children.
  // Returns them in order of appearance, which is the order they must be
  // nested.
  auto CollectLetDeclInfo =
      [](const std::vector<std::shared_ptr<IRStructure>> &children)
      -> std::vector<std::pair<Var, PrimExpr>> {
    std::vector<std::pair<Var, PrimExpr>> result;
    for (auto &child : children) {
      auto unit = static_cast<ScheduleUnit *>(child.get());
      IRStructure *inner = unit->child.get();
      // Handle ScheduleUnit wrapping a TaskNode
      if (inner && inner->IsTask()) {
        auto task = static_cast<TaskNode *>(inner);
        if (IsLetDeclTask(task)) {
          const auto *let = task->stmts[0].as<LetStmtNode>();
          result.push_back({let->var, let->value});
        }
      }
    }
    return result;
  };

  // Helper: given a Stmt and accumulated LetDecl pairs from previous segments,
  // create fresh variables with copy_with_suffix, substitute all references
  // in the Stmt, and wrap with LetStmt bindings.  Variables that are not
  // referenced in the body (or in kept variables' value expressions) are
  // pruned to avoid dead declarations.
  auto WrapWithRenamedLetDecls =
      [](Stmt body,
         const std::vector<std::pair<Var, PrimExpr>> &accumulated_lets)
      -> Stmt {
    if (accumulated_lets.empty())
      return body;

    // Build substitution map: old_var -> new_var
    Map<Var, PrimExpr> subst_map;
    // Create fresh vars and accumulate them (in order)
    std::vector<std::pair<Var, PrimExpr>> new_lets;
    for (auto &[old_var, old_value] : accumulated_lets) {
      auto new_var = old_var.copy_with_suffix("");
      subst_map.Set(old_var, new_var);
      PrimExpr new_value = Substitute(old_value, subst_map);
      new_lets.push_back({new_var, new_value});
    }

    // Substitute all references in the body
    body = Substitute(body, subst_map);

    // Determine which variables are actually used.  Walk from innermost to
    // outermost: a variable is "needed" if it appears in the body or in any
    // already-needed variable's value expression.
    std::vector<bool> needed(new_lets.size(), false);
    // Start with variables used directly in the body.
    for (size_t i = 0; i < new_lets.size(); ++i) {
      const Var &v = new_lets[i].first;
      if (UsesVar(body,
                  [&v](const VarNode *node) { return node == v.get(); })) {
        needed[i] = true;
      }
    }
    // Propagate: if variable j is needed and its value uses variable i,
    // then i is also needed.  Iterate until fixpoint.
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t j = 0; j < new_lets.size(); ++j) {
        if (!needed[j])
          continue;
        for (size_t i = 0; i < j; ++i) {
          if (needed[i])
            continue;
          const Var &vi = new_lets[i].first;
          if (UsesVar(new_lets[j].second, [&vi](const VarNode *node) {
                return node == vi.get();
              })) {
            needed[i] = true;
            changed = true;
          }
        }
      }
    }

    // Wrap only needed LetStmt bindings (innermost first)
    for (int i = static_cast<int>(new_lets.size()) - 1; i >= 0; --i) {
      if (needed[i]) {
        body = LetStmt(new_lets[i].first, new_lets[i].second, body);
      }
    }
    return body;
  };

  Stmt if_then_else;
  if (wg0_has_stmts && wg1_has_stmts) {
    auto wg0_segments = SegmentSequenceChildren(wg0_structure.get());
    auto wg1_segments = SegmentSequenceChildren(wg1_structure.get());

    // Only apply segmented splitting when both sides have matching segment
    // counts (they originate from the same root, split at ControlNode
    // boundaries, so they should match). Otherwise fall back to the
    // single-IfThenElse path.
    if (!wg0_segments.empty() && !wg1_segments.empty() &&
        wg0_segments.size() == wg1_segments.size() && wg0_segments.size() > 1) {
      std::vector<Stmt> segmented_stmts;
      bool has_simt_copy = false;
      // Check for SIMT copy in any wg1 segment (needed for set_max_nreg
      // decision).
      {
        Stmt full_wg1 = irstructure_to_stmt(wg1_structure.get());
        has_simt_copy = SimtCopyDetector::Detect(full_wg1);
      }

      // Accumulate LetDecl info from previous segments for variable renaming.
      std::vector<std::pair<Var, PrimExpr>> wg0_accumulated_lets;
      std::vector<std::pair<Var, PrimExpr>> wg1_accumulated_lets;

      for (size_t si = 0; si < wg0_segments.size(); ++si) {
        // Insert liveness boundary between segments.
        segmented_stmts.push_back(
            AttrStmt(Integer(0), attr::kAutoScheduleSharedMemoryBoundary, 0,
                     Evaluate(0)));

        // Collect LetDecl info from current segment before converting to Stmt.
        auto wg0_lets = CollectLetDeclInfo(wg0_segments[si]);
        auto wg1_lets = CollectLetDeclInfo(wg1_segments[si]);

        Stmt wg0_seg_stmt = SegmentToStmt(wg0_segments[si]);
        Stmt wg1_seg_stmt = SegmentToStmt(wg1_segments[si]);

        // For segments after the first, wrap with renamed LetDecl bindings
        // from all previous segments so that variables remain in scope.
        if (si > 0) {
          wg0_seg_stmt =
              WrapWithRenamedLetDecls(wg0_seg_stmt, wg0_accumulated_lets);
          wg1_seg_stmt =
              WrapWithRenamedLetDecls(wg1_seg_stmt, wg1_accumulated_lets);
        }

        // Accumulate this segment's LetDecls for future segments.
        wg0_accumulated_lets.insert(wg0_accumulated_lets.end(),
                                    wg0_lets.begin(), wg0_lets.end());
        wg1_accumulated_lets.insert(wg1_accumulated_lets.end(),
                                    wg1_lets.begin(), wg1_lets.end());

        // Prepend set_max_nreg only to the first segment.
        if (si == 0 && !has_simt_copy && config.enable_set_max_nreg) {
          wg0_seg_stmt =
              SeqStmt({Evaluate(Call(DataType::Handle(), tl::set_max_nreg(),
                                     {config.consumer_max_nreg, 1})),
                       wg0_seg_stmt});
          wg1_seg_stmt =
              SeqStmt({Evaluate(Call(DataType::Handle(), tl::set_max_nreg(),
                                     {config.producer_max_nreg, 0})),
                       wg1_seg_stmt});
        }

        segmented_stmts.push_back(MakeWarpgroupIf(wg0_seg_stmt, wg1_seg_stmt));
      }
      if_then_else = SeqStmt::Flatten(segmented_stmts);
    } else {
      // Fallback: single IfThenElse (original logic).
      Stmt then_body = irstructure_to_stmt(wg0_structure.get());
      Stmt else_body = irstructure_to_stmt(wg1_structure.get());
      bool has_simt_copy = SimtCopyDetector::Detect(else_body);
      if (has_simt_copy || !config.enable_set_max_nreg) {
        if_then_else =
            IfThenElse(condition, then_body,
                       IfThenElse(wg1_condition, else_body, Evaluate(0)));
      } else {
        std::vector<Stmt> then_body_with_nreg{
            Evaluate(Call(DataType::Handle(), tl::set_max_nreg(),
                          {config.consumer_max_nreg, 1})),
            then_body};
        std::vector<Stmt> else_body_with_nreg{
            Evaluate(Call(DataType::Handle(), tl::set_max_nreg(),
                          {config.producer_max_nreg, 0})),
            else_body};
        if_then_else =
            IfThenElse(condition, SeqStmt(then_body_with_nreg),
                       IfThenElse(wg1_condition, SeqStmt(else_body_with_nreg),
                                  Evaluate(0)));
      }
    }
  } else if (wg0_has_stmts) {
    // Only warpgroup 0 has statements, execute unconditionally
    if_then_else = irstructure_to_stmt(wg0_structure.get());
  } else if (wg1_has_stmts) {
    // Only warpgroup 1 has statements, execute unconditionally
    if_then_else = irstructure_to_stmt(wg1_structure.get());
  } else {
    // Neither warpgroup 0 nor 1 has statements
    if_then_else = Evaluate(0);
  }

  PrimExpr barrier_count = config.enable_thread_extend
                               ? thread_count[0] + thread_count[1]
                               : thread_var->dom->extent;

  Stmt pro_and_warpgroup_stmt;
  if (wg_pro_neutral_has_stmts) {
    if (!IsEvaluateZero(if_then_else) && !IsEvaluateZero(pro_neutral_body)) {
      // Both have statements: insert barriers for neutral-to-warpgroup
      // synchronization
      pro_and_warpgroup_stmt = InsertBarriersForNeutralSync(
          pro_neutral_body, if_then_else, barrier_buffers, barrier_map,
          barrier_count, neutral_sync_shared_barrier);
    } else if (!IsEvaluateZero(if_then_else) ||
               !IsEvaluateZero(pro_neutral_body)) {
      // Only one has actual statements
      std::vector<Stmt> stmts;
      if (!IsEvaluateZero(pro_neutral_body)) {
        stmts.push_back(pro_neutral_body);
      }
      if (!IsEvaluateZero(if_then_else)) {
        stmts.push_back(if_then_else);
      }
      if (stmts.size() == 1) {
        pro_and_warpgroup_stmt = stmts[0];
      } else {
        pro_and_warpgroup_stmt = SeqStmt(stmts);
      }
    } else {
      // Both are empty
      pro_and_warpgroup_stmt = Evaluate(0);
    }
  } else {
    pro_and_warpgroup_stmt = if_then_else;
  }

  bool need_shared_barrier_for_epi = false;
  bool need_tmem_barrier_for_epi = false;
  if (wg_epi_neutral_structure) {
    for (const auto *warpgroup_structure :
         {wg0_structure.get(), wg1_structure.get()}) {
      need_shared_barrier_for_epi =
          need_shared_barrier_for_epi ||
          HasSharedWriteReadDependency(warpgroup_structure,
                                       wg_epi_neutral_structure.get());
      need_tmem_barrier_for_epi =
          need_tmem_barrier_for_epi ||
          HasTmemWriteReadDependency(warpgroup_structure,
                                     wg_epi_neutral_structure.get());
    }
  }

  Stmt combined_stmt;
  if (!IsEvaluateZero(pro_and_warpgroup_stmt) &&
      !IsEvaluateZero(epi_neutral_body)) {
    // Both have statements: insert barriers for warpgroup-to-epi_neutral
    // synchronization
    combined_stmt = InsertBarriersForNeutralSyncWithDependency(
        pro_and_warpgroup_stmt, epi_neutral_body, barrier_buffers, barrier_map,
        barrier_count, need_shared_barrier_for_epi, need_tmem_barrier_for_epi,
        Buffer(), thread_var->var, 0, thread_count[0]);
  } else if (!IsEvaluateZero(epi_neutral_body)) {
    combined_stmt = epi_neutral_body;
  } else {
    combined_stmt = pro_and_warpgroup_stmt;
  }

  return combined_stmt;
}

} // namespace tl
} // namespace tvm
