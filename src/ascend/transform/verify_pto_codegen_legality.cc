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
 * \file verify_pto_codegen_legality.cc
 * \brief Verify PTO requirements before code generation without modifying IR.
 *
 * This pass consumes the final device TIR after LowerDeviceKernelLaunch and
 * the final layout and UB rewrites. It currently checks destination alignment
 * for default/continuous explicit-mask VMI stores of b8/b16/b32 data.
 * Ordinary masked stores require provable 32-byte alignment. Stores selected
 * for 1PT_B8/B16/B32 writeback require natural element alignment (1/2/4 bytes):
 * the value must have 1/2/4/8 lanes and the mask must be a recognized
 * full-active seed with the same lane count. Bool data is excluded from this
 * check. The checker resolves byte addresses using the Analyzer facts at each
 * access. Shapes, layouts, masks and addresses are never modified. Functions
 * without a PTO target pass through unchanged.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "arith/ir_visitor_with_analyzer.h"
#include "tirx/transform/ir_utils.h"

#include "../../op/builtin.h"
#include "../target_utils.h"

namespace tvm {
namespace tl {
namespace transform {

using namespace tirx;

namespace {

constexpr int kOrdinaryMaskedStoreAlignment = 32;

enum class StoreMode {
  kOrdinaryMasked,
  kPointStores,
};

enum class Outcome {
  kAligned,
  kMisaligned,
  kUnknown,
};

struct ByteAddress {
  enum class Kind {
    kUbRoot,
    kIntegerBase,
    kUnknown,
  };
  Kind kind = Kind::kUnknown;
  const VarNode *root = nullptr;
  PrimExpr base_bytes;
  PrimExpr offset_bytes;
  std::string error;
};

struct StoreMatch {
  bool checked = false;
  StoreMode mode = StoreMode::kOrdinaryMasked;
  const char *mode_name = "";
  int required_alignment = kOrdinaryMaskedStoreAlignment;
  int elem_bytes = 0;
  DataType value_dtype;
  // Element type of the pointer that reaches the store after descriptor,
  // address_of, typed Var/make_tensor and pure alias resolution.  It is the
  // unit of the instruction offset; it is never defaulted to float32.
  std::optional<DataType> pointer_dtype;
  bool has_descriptor = false;
  DataType descriptor_dtype = DataType::Float(32);
  PrimExpr descriptor_offset;
  PrimExpr instruction_offset;
};

struct SeedInfo {
  bool recognized = false;
  bool full_active = false;
  int64_t mask_lanes = 0;
};

struct Proof {
  Outcome outcome = Outcome::kUnknown;
  PrimExpr byte_offset;
  PrimExpr remainder;
  std::string error;
};

} // namespace

/*! Checker for explicit-mask continuous VMI stores.  Traversal is scoped:
 * loop/branch/attribute bodies push and pop the pointer/pure-definition
 * maps, and the Analyzer provides the existing range/condition facts. */
class PTOMaskedStoreAlignmentChecker final
    : public arith::IRVisitorWithAnalyzer {
public:
  explicit PTOMaskedStoreAlignmentChecker(std::string function_name)
      : function_name_(std::move(function_name)) {}

  using Base = arith::IRVisitorWithAnalyzer;
  using Base::VisitExpr_;
  using Base::VisitStmt_;

  void Run(const PrimFunc &func) { VisitStmt(func->body); }

  // UB arena roots: PTO codegen materializes the merged dyn-shared
  // allocation at UB byte address 0, and MergeUBAllocations aligns the arena
  // to 32 bytes.  Only allocations known to feed that arena are roots; a raw
  // handle is never assumed aligned.  Roots are recorded in lexical order
  // during the single statement walk, so expression DAGs are never re-walked
  // by a whole-body pre-scan.
  void VisitStmt_(const AllocBufferNode *op) final {
    const Var &data = op->buffer->data;
    const std::string scope = GetPtrStorageScope(data);
    if (scope == "shared.dyn" || scope == "shared") {
      ub_roots_.insert(data.get());
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    constraint_scope_.WithNewScope([&]() {
      analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      constraint_scope_.Current().Emplace(&analyzer_, op->extent > 0);
      if (op->step.defined()) {
        const auto *step = op->step.as<IntImmNode>();
        if (step != nullptr && step->value > 1) {
          // The base visitor binds [min, min+extent) but drops the loop
          // lattice.  Record the real iteration congruence so a positive
          // constant step such as i = 0, 8, 16, ... can be proven.
          constraint_scope_.Current().Emplace(
              &analyzer_,
              floormod(op->loop_var - op->min, op->step.value()) == 0);
        }
      }
      AliasScope saved = SaveAliases();
      StmtExprVisitor::VisitStmt_(op);
      RestoreAliases(std::move(saved));
    });
  }

  void VisitStmt_(const SBlockNode *op) final {
    constraint_scope_.WithNewScope([&]() {
      for (const auto &iter_var : op->iter_vars) {
        analyzer_.Bind(iter_var->var, iter_var->dom);
      }
      AliasScope saved = SaveAliases();
      StmtExprVisitor::VisitStmt_(op);
      RestoreAliases(std::move(saved));
    });
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    constraint_scope_.WithNewScope([&]() {
      this->VisitExpr(op->condition);
      PrimExpr real_condition = ExtractRealCondition(op->condition);
      constraint_scope_.WithNewScope([&]() {
        constraint_scope_.Current().Emplace(&analyzer_, real_condition);
        AliasScope saved = SaveAliases();
        this->VisitStmt(op->then_case);
        RestoreAliases(std::move(saved));
      });
      if (op->else_case) {
        constraint_scope_.WithNewScope([&]() {
          constraint_scope_.Current().Emplace(
              &analyzer_, analyzer_.rewrite_simplify(Not(real_condition)));
          AliasScope saved = SaveAliases();
          this->VisitStmt(op->else_case.value());
          RestoreAliases(std::move(saved));
        });
      }
    });
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    constraint_scope_.WithNewScope([&]() {
      if (op->attr_key == tirx::attr::thread_extent ||
          op->attr_key == s_tir::attr::virtual_thread) {
        IterVar iv = Downcast<IterVar>(op->node);
        TVM_FFI_ICHECK_NE(iv->thread_tag.length(), 0U);
        analyzer_.Bind(iv->var, Range::FromMinExtent(
                                    IntImm(op->value->dtype, 0), op->value));
      } else if (op->attr_key == tirx::attr::tilelang_assume) {
        auto condition = Downcast<PrimExpr>(op->node);
        constraint_scope_.Current().Emplace(&analyzer_, condition,
                                            /*is_assume=*/true);
      }
      AliasScope saved = SaveAliases();
      StmtExprVisitor::VisitStmt_(op);
      RestoreAliases(std::move(saved));
    });
  }

  void VisitStmt_(const BindNode *op) final {
    std::unordered_set<const VarNode *> visiting{op->var.get()};
    ExprMemo memo;
    ScalarFacts facts = AnalyzeScalar(op->value, &visiting, &memo);
    if (facts.needs_generic_visit) {
      this->VisitExpr(op->value);
    }
    RegisterDefinition(op->var.get(), op->value, facts);
  }

  void VisitExpr_(const LetNode *op) final {
    std::unordered_set<const VarNode *> visiting{op->var.get()};
    ExprMemo value_memo;
    ScalarFacts value_facts = AnalyzeScalar(op->value, &visiting, &value_memo);
    if (value_facts.needs_generic_visit) {
      this->VisitExpr(op->value);
    }
    AliasScope saved = SaveAliases();
    RegisterDefinition(op->var.get(), op->value, value_facts);
    std::unordered_set<const VarNode *> body_visiting{op->var.get()};
    ExprMemo body_memo;
    ScalarFacts body_facts =
        AnalyzeScalar(op->body, &body_visiting, &body_memo);
    if (body_facts.needs_generic_visit) {
      this->VisitExpr(op->body);
    }
    RestoreAliases(std::move(saved));
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::vmi_vstore())) {
      CheckStore(op);
    }
    Base::VisitExpr_(op);
  }

private:
  struct AliasScope {
    std::unordered_map<const VarNode *, PrimExpr> pointer_defs;
    std::unordered_map<const VarNode *, PrimExpr> value_defs;
  };

  AliasScope SaveAliases() const {
    return AliasScope{pointer_defs_, value_defs_};
  }

  void RestoreAliases(AliasScope saved) {
    pointer_defs_ = std::move(saved.pointer_defs);
    value_defs_ = std::move(saved.value_defs);
  }

  struct ScalarFacts {
    bool origin_safe = false;
    bool needs_generic_visit = false;
    bool has_shared_node = false;
  };
  using ExprMemo = std::unordered_map<const ffi::Object *, ScalarFacts>;
  using CanonMemo = std::unordered_map<const ffi::Object *, PrimExpr>;

  // Bottom-up, memoized facts for one expression / right-hand side.
  //
  // origin_safe: the expression is a scalar (lanes == 1, non-handle) tree of
  // immediates, free variables, casts and pure scalar arithmetic, and every
  // referenced bind definition is origin-safe as well (a lane-1 VMI alias, a
  // memory read, a Call or a cycle is unsafe).  Plain SSA chains of any length
  // are classified through the per-Var cache one shallow definition at a time.
  //
  // needs_generic_visit: the generic statement/expression walk is required
  // because the expression may contain a checked store (a void Call), a Let
  // definition, branch constraints, Reduce axes or an unrecognized node.  A
  // pure elementwise arithmetic expression cannot contain any of these, so
  // skipping the walk is safe for nested checked-store discovery.
  //
  // has_shared_node: a non-leaf expression node is referenced more than once
  // within this right-hand side (a DAG).  This only selects the canonicalized
  // Analyzer binding; the definition and its equality are always kept.
  ScalarFacts AnalyzeScalar(const PrimExpr &value,
                            std::unordered_set<const VarNode *> *visiting,
                            ExprMemo *memo, int depth = 0) {
    // Recursion/stack-safety guard only, not a complexity or acceptance
    // bound: repeated work is already bounded by the per-RHS memo and the
    // per-Var classification cache.  An expression nested deeper than this is
    // treated conservatively (no Analyzer equality and a generic visit), so a
    // dependent store is diagnosed as unprovable instead of risking the
    // compiler stack.
    constexpr int kMaxExpressionRecursionGuard = 4096;
    if (depth > kMaxExpressionRecursionGuard) {
      return ScalarFacts{false, true, false};
    }
    const DataType dtype = value.dtype();
    if (value.as<IntImmNode>() || value.as<FloatImmNode>() ||
        value.as<StringImmNode>()) {
      return ScalarFacts{true, false, false};
    }
    if (const auto *var = value.as<VarNode>()) {
      if (visiting->count(var) != 0) {
        // Self/cyclic reference: not a lawful SSA definition.
        return ScalarFacts{false, false, false};
      }
      auto safe_it = var_origin_safe_.find(var);
      if (safe_it != var_origin_safe_.end()) {
        return ScalarFacts{safe_it->second, false, false};
      }
      auto def_it = value_defs_.find(var);
      if (def_it == value_defs_.end()) {
        // Function/loop/block parameter without a recorded definition.
        return ScalarFacts{true, false, false};
      }
      visiting->insert(var);
      // Analyze the definition with its own memo: sharing inside a separate
      // right-hand side must not mark the current expression as shared.
      ExprMemo def_memo;
      ScalarFacts def_facts =
          AnalyzeScalar(def_it->second, visiting, &def_memo, depth + 1);
      visiting->erase(var);
      var_origin_safe_[var] = def_facts.origin_safe;
      return ScalarFacts{def_facts.origin_safe, false, false};
    }
    auto memo_it = memo->find(value.get());
    if (memo_it != memo->end()) {
      return ScalarFacts{memo_it->second.origin_safe,
                         memo_it->second.needs_generic_visit, true};
    }
    auto child = [&](const PrimExpr &expr) {
      return AnalyzeScalar(expr, visiting, memo, depth + 1);
    };
    const bool scalar_root = !dtype.is_handle() && dtype.lanes() == 1;
    auto binary = [&](const PrimExpr &a, const PrimExpr &b) {
      ScalarFacts fa = child(a);
      ScalarFacts fb = child(b);
      ScalarFacts out;
      out.origin_safe = scalar_root && fa.origin_safe && fb.origin_safe;
      out.needs_generic_visit =
          fa.needs_generic_visit || fb.needs_generic_visit;
      out.has_shared_node = fa.has_shared_node || fb.has_shared_node;
      return out;
    };
    auto ternary = [&](const PrimExpr &a, const PrimExpr &b,
                       const PrimExpr &c) {
      ScalarFacts fa = child(a);
      ScalarFacts fb = child(b);
      ScalarFacts fc = child(c);
      ScalarFacts out;
      out.origin_safe =
          scalar_root && fa.origin_safe && fb.origin_safe && fc.origin_safe;
      out.needs_generic_visit = fa.needs_generic_visit ||
                                fb.needs_generic_visit ||
                                fc.needs_generic_visit;
      out.has_shared_node =
          fa.has_shared_node || fb.has_shared_node || fc.has_shared_node;
      return out;
    };
    ScalarFacts facts;
    if (const auto *node = value.as<CastNode>()) {
      ScalarFacts c = child(node->value);
      facts.origin_safe = scalar_root && c.origin_safe;
      facts.needs_generic_visit = c.needs_generic_visit;
      facts.has_shared_node = c.has_shared_node;
    } else if (const auto *node = value.as<AddNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<SubNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<MulNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<DivNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<FloorDivNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<ModNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<FloorModNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<MinNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<MaxNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<SelectNode>()) {
      facts = ternary(node->condition, node->true_value, node->false_value);
    } else if (const auto *node = value.as<LENode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<LTNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<GTNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<GENode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<EQNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<NENode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<AndNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<OrNode>()) {
      facts = binary(node->a, node->b);
    } else if (const auto *node = value.as<NotNode>()) {
      ScalarFacts c = child(node->a);
      facts.origin_safe = scalar_root && c.origin_safe;
      facts.needs_generic_visit = c.needs_generic_visit;
      facts.has_shared_node = c.has_shared_node;
    } else if (const auto *node = value.as<BufferLoadNode>()) {
      bool needs = false;
      bool shared = false;
      for (const auto &index : node->indices) {
        ScalarFacts f = child(index);
        needs = needs || f.needs_generic_visit;
        shared = shared || f.has_shared_node;
      }
      if (node->predicate.defined()) {
        ScalarFacts f = child(node->predicate.value());
        needs = needs || f.needs_generic_visit;
        shared = shared || f.has_shared_node;
      }
      facts = ScalarFacts{false, needs, shared};
    } else if (const auto *node = value.as<RampNode>()) {
      ScalarFacts f = binary(node->base, node->stride);
      facts = ScalarFacts{false, f.needs_generic_visit, f.has_shared_node};
    } else if (const auto *node = value.as<BroadcastNode>()) {
      ScalarFacts f = child(node->value);
      facts = ScalarFacts{false, f.needs_generic_visit, f.has_shared_node};
    } else if (const auto *node = value.as<ShuffleNode>()) {
      bool needs = false;
      bool shared = false;
      for (const auto &vector : node->vectors) {
        ScalarFacts f = child(vector);
        needs = needs || f.needs_generic_visit;
        shared = shared || f.has_shared_node;
      }
      for (const auto &index : node->indices) {
        ScalarFacts f = child(index);
        needs = needs || f.needs_generic_visit;
        shared = shared || f.has_shared_node;
      }
      facts = ScalarFacts{false, needs, shared};
    } else {
      // Calls (including every VMI producer and vstore), Let, Reduce and
      // every other opaque node keep the generic walk and are not scalar
      // equalities.
      facts = ScalarFacts{false, true, false};
    }
    (*memo)[value.get()] = facts;
    return facts;
  }

  // Rebuild an origin-safe expression with each distinct non-leaf node bound
  // to a fresh shallow symbol.  The Analyzer's scalar binding walks its
  // argument recursively, so handing it a shared DAG (e+k; e=e+e; ...) is
  // exponential even though the classification above is memoized.  With one
  // shallow binding per distinct node the substitution graph stays linear and
  // the equality -- including its modular/alignment consequences -- is fully
  // preserved.
  PrimExpr CanonicalizeForAnalyzer(const PrimExpr &value, CanonMemo *memo) {
    if (value.as<IntImmNode>() || value.as<FloatImmNode>() ||
        value.as<StringImmNode>() || value.as<VarNode>()) {
      return value;
    }
    auto memo_it = memo->find(value.get());
    if (memo_it != memo->end()) {
      return memo_it->second;
    }
    auto canon = [&](const PrimExpr &expr) {
      return CanonicalizeForAnalyzer(expr, memo);
    };
    PrimExpr rebuilt;
    if (const auto *node = value.as<CastNode>()) {
      rebuilt =
          Cast(node->dtype, canon(node->value), node->annotations, node->span);
    } else if (const auto *node = value.as<AddNode>()) {
      rebuilt = Add(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<SubNode>()) {
      rebuilt = Sub(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<MulNode>()) {
      rebuilt = Mul(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<DivNode>()) {
      rebuilt = Div(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<FloorDivNode>()) {
      rebuilt = FloorDiv(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<ModNode>()) {
      rebuilt = Mod(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<FloorModNode>()) {
      rebuilt = FloorMod(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<MinNode>()) {
      rebuilt = Min(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<MaxNode>()) {
      rebuilt = Max(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<SelectNode>()) {
      rebuilt = Select(canon(node->condition), canon(node->true_value),
                       canon(node->false_value), node->span);
    } else if (const auto *node = value.as<LENode>()) {
      rebuilt = LE(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<LTNode>()) {
      rebuilt = LT(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<GTNode>()) {
      rebuilt = GT(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<GENode>()) {
      rebuilt = GE(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<EQNode>()) {
      rebuilt = EQ(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<NENode>()) {
      rebuilt = NE(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<AndNode>()) {
      rebuilt = And(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<OrNode>()) {
      rebuilt = Or(canon(node->a), canon(node->b), node->span);
    } else if (const auto *node = value.as<NotNode>()) {
      rebuilt = Not(canon(node->a), node->span);
    } else {
      // Unreachable for a classified origin-safe expression; keep the value
      // rather than inventing a different semantic.
      (*memo)[value.get()] = value;
      return value;
    }
    Var fresh("__pto_align_dag_" + std::to_string(next_dag_symbol_++),
              rebuilt.dtype());
    analyzer_.Bind(fresh, rebuilt);
    (*memo)[value.get()] = fresh;
    return fresh;
  }

  void RegisterDefinition(const VarNode *var, const PrimExpr &value,
                          const ScalarFacts &facts) {
    if (value.dtype().is_handle()) {
      pointer_defs_[var] = value;
      return;
    }
    // Scalar/index definitions and pointer aliases are recorded separately:
    // value_defs_ backs the origin-safe scalar check and the mask-seed alias
    // walk, while pointer_defs_ backs address resolution.  Seed matching only
    // accepts recognized create_mask/pset calls and compile-time active
    // counts, so a memory-read definition never classifies as a seed.
    value_defs_[var] = value;
    const bool safe = value.dtype().is_scalar() && !value.dtype().is_handle() &&
                      facts.origin_safe;
    var_origin_safe_[var] = safe;
    if (!safe) {
      return;
    }
    if (facts.has_shared_node) {
      // Preserve the equality with a compact substitution graph instead of
      // dropping it (which would lose alignment proofs) or expanding the DAG.
      CanonMemo canon_memo;
      PrimExpr canonical = CanonicalizeForAnalyzer(value, &canon_memo);
      analyzer_.Bind(ffi::GetRef<Var>(var), canonical);
    } else {
      analyzer_.Bind(ffi::GetRef<Var>(var), value);
    }
  }

  // Follow immutable SSA aliases iteratively: the length of a plain alias
  // chain is not a recursion-depth budget.  A visited set detects cycles.
  std::optional<int64_t> ResolveMaskActiveCount(const PrimExpr &expr) const {
    PrimExpr current = expr;
    std::unordered_set<const VarNode *> seen;
    while (const auto *var = current.as<VarNode>()) {
      if (!seen.insert(var).second) {
        return std::nullopt;
      }
      auto it = value_defs_.find(var);
      if (it == value_defs_.end()) {
        return std::nullopt;
      }
      current = it->second;
    }
    if (const auto *imm = current.as<IntImmNode>()) {
      return imm->value;
    }
    return std::nullopt;
  }

  PrimExpr ResolvePureAlias(const PrimExpr &expr) const {
    PrimExpr current = expr;
    std::unordered_set<const VarNode *> seen;
    while (const auto *var = current.as<VarNode>()) {
      if (!seen.insert(var).second) {
        return current; // cycle: not a recognized seed
      }
      auto it = value_defs_.find(var);
      if (it == value_defs_.end()) {
        return current;
      }
      current = it->second;
    }
    return current;
  }

  // Mask seeds recognized by the PTO backend for 1PT_B8/B16/B32 writeback:
  //
  // * A default/continuous vstore with 1/2/4/8 data lanes can use point
  //   writeback when its create_mask(active, size) seed has the same lane
  //   count and a compile-time active count at least as large as size.
  //   This selects full-active pset_b32 PAT_ALL and one point store per
  //   logical element. The value need not come from a reduction.
  // * A pset PAT_ALL mask is also a recognized seed. TileLang does not expose
  //   a pset operator in its Python API; this branch handles direct TIR calls.
  // * create_mask(0), partial masks and dynamic active counts do not match.
  //   These remain ordinary masked stores and require 32-byte alignment.
  //   Mismatched value/mask lanes are rejected by the backend, and arbitrary
  //   all-true expressions are not recognized seeds.
  SeedInfo ClassifyMaskSeed(const PrimExpr &mask) const {
    SeedInfo info;
    const DataType mask_dtype = mask.dtype();
    if (!mask_dtype.is_bool() || mask_dtype.lanes() <= 0) {
      return info;
    }
    info.mask_lanes = mask_dtype.lanes();
    PrimExpr resolved = ResolvePureAlias(mask);
    const auto *call = resolved.as<CallNode>();
    if (call == nullptr) {
      return info;
    }
    auto opt_op = call->op.as<Op>();
    if (!opt_op.has_value()) {
      return info;
    }
    const std::string &name = opt_op.value()->name;
    if (name == "tl.vmi.create_mask") {
      info.recognized = true;
      if (!call->args.empty()) {
        std::optional<int64_t> active = ResolveMaskActiveCount(call->args[0]);
        info.full_active = active.has_value() && *active >= info.mask_lanes;
      }
      return info;
    }
    if (name == "tl.vmi.pset" && !call->args.empty()) {
      if (const auto *dist = call->args[0].as<StringImmNode>()) {
        if (dist->value == "PAT_ALL") {
          info.recognized = true;
          info.full_active = true;
        }
      }
    }
    return info;
  }

  std::optional<DataType> DecodeAccessDType(const PrimExpr &annotation) const {
    const auto *call = annotation.as<CallNode>();
    if (call == nullptr) {
      return std::nullopt;
    }
    std::string n;
    if (!call->args.empty()) {
      if (const auto *name = call->args[0].as<StringImmNode>()) {
        n = name->value;
      }
    }
    if (n.empty()) {
      // The final device IR carries a `tirx.type_annotation` call whose
      // result dtype is the annotated element type (the call has no
      // positional arguments at this stage).  Use that dtype so descriptor
      // and instruction offsets are scaled exactly as codegen does.
      auto opt_op = call->op.as<Op>();
      if (!opt_op.has_value() ||
          opt_op.value()->name != "tirx.type_annotation") {
        return std::nullopt;
      }
      DataType annotated = call->dtype;
      if (!annotated.is_scalar() || annotated.is_handle()) {
        return std::nullopt;
      }
      return annotated;
    }
    if (n == "float" || n == "float32")
      return DataType::Float(32);
    if (n == "half" || n == "float16")
      return DataType::Float(16);
    if (n == "bfloat16" || n == "bfloat16_t")
      return DataType::BFloat(16);
    if (n == "float8_e4m3" || n == "float8_e4m3fn") {
      return DataType::Float8E4M3FN();
    }
    if (n == "float8_e5m2")
      return DataType::Float8E5M2();
    if (n == "int8")
      return DataType::Int(8);
    if (n == "uint8")
      return DataType::UInt(8);
    if (n == "int16")
      return DataType::Int(16);
    if (n == "uint16")
      return DataType::UInt(16);
    if (n == "int32")
      return DataType::Int(32);
    if (n == "uint32")
      return DataType::UInt(32);
    if (n == "int64")
      return DataType::Int(64);
    if (n == "uint64")
      return DataType::UInt(64);
    return std::nullopt;
  }

  std::optional<DataType>
  PointerAnnotationElementType(const VarNode *var) const {
    const auto *ptr = var->type_annotation.as<PointerTypeNode>();
    if (ptr == nullptr) {
      return std::nullopt;
    }
    const auto *prim = ptr->element_type.as<PrimTypeNode>();
    if (prim == nullptr) {
      return std::nullopt;
    }
    DataType dtype = prim->dtype;
    if (dtype.is_void() || dtype.is_handle() || dtype.lanes() != 1) {
      return std::nullopt;
    }
    return dtype;
  }

  // Element type of the pointer that the store actually receives, matching
  // codegen exactly (codegen_pto.cc Bind handling):
  //
  // * A typed Var bound to ``reinterpret(handle, int_addr)`` (T.make_tensor)
  //   emits ``v = pto.castptr(addr, ptr<T>)``: this binding's own annotation
  //   is a real castptr and establishes the type.
  // * Every other binding emits ``v = <expr>`` with no cast.  A typed
  //   annotation on a plain alias therefore does NOT re-type the pointer; the
  //   defining expression's type wins.
  // * ``tvm_access_ptr``/``address_of`` print through GetPointerExpr and cast
  //   to their own descriptor/buffer element type, overriding any outer
  //   annotation.
  // * A pointer type re-view (``reinterpret(handle, pointer)``) prints the
  //   inner expression unchanged.
  //
  // Plain immutable SSA aliases are followed iteratively (their length is not
  // a recursion budget); only pointer expression nesting consumes the
  // stack-safety guard.  Only the Var directly bound to the terminal
  // expression can establish a type; an untyped/unknown pointer is not
  // guessed as float32.
  std::optional<DataType>
  ResolvePointerElementType(const PrimExpr &ptr,
                            std::unordered_set<const VarNode *> *visiting,
                            int nesting = 0) const {
    // Stack-safety guard only; not an alias-length or acceptance limit.
    constexpr int kMaxPointerExpressionNesting = 4096;
    if (nesting > kMaxPointerExpressionNesting) {
      return std::nullopt;
    }
    const VarNode *defining_var = nullptr;
    PrimExpr current = ptr;
    while (const auto *var = current.as<VarNode>()) {
      if (!visiting->insert(var).second) {
        return std::nullopt; // alias cycle
      }
      auto def_it = pointer_defs_.find(var);
      if (def_it == pointer_defs_.end()) {
        // Terminal Var without a binding (parameter/root): its own annotation
        // is the type of the pointer it denotes.
        return PointerAnnotationElementType(var);
      }
      defining_var = var;
      current = def_it->second;
    }
    if (const auto *call = current.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr())) {
        if (!call->args.empty()) {
          return DecodeAccessDType(call->args[0]);
        }
        return std::nullopt;
      }
      if (call->op.same_as(builtin::address_of())) {
        if (call->args.size() == 1U) {
          if (const auto *load = call->args[0].as<BufferLoadNode>()) {
            DataType dtype = load->dtype.element_of();
            if (!dtype.is_void() && !dtype.is_handle()) {
              return dtype;
            }
          }
        }
        return std::nullopt;
      }
      if (call->op.same_as(tirx::builtin::reinterpret()) &&
          call->args.size() == 1U) {
        if (call->args[0].dtype().is_handle()) {
          // Pointer type re-view: the inner pointer's type is the emitted
          // type; an outer Var annotation does not emit a cast.
          return ResolvePointerElementType(call->args[0], visiting,
                                           nesting + 1);
        }
        // Integer-address make_tensor binding: the castptr uses the
        // annotation of the Var directly bound to this reinterpret.
        if (defining_var != nullptr) {
          return PointerAnnotationElementType(defining_var);
        }
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  StoreMatch MatchStore(const CallNode *op) const {
    StoreMatch match;
    // Default continuous masked form only: no mode-changing annotations.
    for (const auto &[key_ref, value] : op->annotations) {
      const std::string key = key_ref;
      if (key == "dist_mode") {
        const auto *mode = value.as<StringImmNode>();
        if (mode == nullptr || (mode->value != "continuous")) {
          return match;
        }
        continue;
      }
      if (key == "group" || key == "stride" || key == "block_stride" ||
          key == "repeat_stride" || key == "pmode") {
        return match;
      }
    }
    // Masked ordinary form is (value, ptr, instruction_offset, mask).  The
    // dintlv no-mask form also has four positional arguments but carries
    // dist_mode="dintlv" and was rejected above.
    if (op->args.size() != 4U) {
      return match;
    }
    const DataType value_dtype = op->args[0].dtype();
    const DataType elem = value_dtype.element_of();
    // Boolean (mask-register) values are not part of the b8/b16/b32 data
    // storage contract; their backend admission stays unchanged.
    if (elem.is_bool()) {
      return match;
    }
    const int bits = elem.bits();
    if (bits != 8 && bits != 16 && bits != 32) {
      return match;
    }
    const DataType mask_dtype = op->args[3].dtype();
    if (!mask_dtype.is_bool()) {
      return match;
    }
    match.checked = true;
    match.value_dtype = value_dtype;
    match.elem_bytes = bits / 8;
    // Record the immediate descriptor only for diagnostics and for the
    // inherited byte offset (ResolveAddress consumes it).  The instruction
    // offset unit is resolved from the *resulting* pointer expression and is
    // never defaulted to float32.
    if (const auto *ptr_call = op->args[1].as<CallNode>()) {
      if (ptr_call->op.same_as(builtin::tvm_access_ptr()) &&
          ptr_call->args.size() == 5U) {
        match.has_descriptor = true;
        match.descriptor_offset = ptr_call->args[2];
        if (auto dtype = DecodeAccessDType(ptr_call->args[0])) {
          match.descriptor_dtype = dtype.value();
        }
      }
    }
    match.instruction_offset = op->args[2];
    const int64_t lanes = value_dtype.lanes();
    const bool point_candidate =
        lanes == 1 || lanes == 2 || lanes == 4 || lanes == 8;
    const SeedInfo seed = ClassifyMaskSeed(op->args[3]);
    if (point_candidate && seed.recognized && seed.full_active &&
        seed.mask_lanes == lanes) {
      match.mode = StoreMode::kPointStores;
      match.mode_name = "point";
      match.required_alignment = bits / 8;
    } else {
      match.mode = StoreMode::kOrdinaryMasked;
      match.mode_name = "ordinary masked";
      match.required_alignment = kOrdinaryMaskedStoreAlignment;
    }
    return match;
  }

  ByteAddress UnknownAddress(const std::string &why) const {
    ByteAddress addr;
    addr.kind = ByteAddress::Kind::kUnknown;
    addr.error = why;
    return addr;
  }

  ByteAddress ResolveAddress(const PrimExpr &ptr,
                             std::unordered_set<const VarNode *> *visiting,
                             int nesting = 0) {
    // Stack-safety guard only; plain alias length is walked iteratively and
    // does not consume this budget.
    constexpr int kMaxPointerExpressionNesting = 4096;
    if (nesting > kMaxPointerExpressionNesting) {
      return UnknownAddress(
          "pointer expression nesting exceeds the stack-safety guard");
    }
    PrimExpr current = ptr;
    while (const auto *var = current.as<VarNode>()) {
      if (!visiting->insert(var).second) {
        return UnknownAddress("pointer alias cycle at `" + var->name_hint +
                              "`");
      }
      auto def_it = pointer_defs_.find(var);
      if (def_it == pointer_defs_.end()) {
        break;
      }
      current = def_it->second;
    }
    if (const auto *var = current.as<VarNode>()) {
      if (ub_roots_.count(var) != 0) {
        ByteAddress addr;
        addr.kind = ByteAddress::Kind::kUbRoot;
        addr.root = var;
        addr.base_bytes = IntImm(DataType::Int(64), 0);
        addr.offset_bytes = IntImm(DataType::Int(64), 0);
        return addr;
      }
      return UnknownAddress("pointer origin unavailable for `" +
                            var->name_hint + "`");
    }
    if (const auto *call = current.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr())) {
        TVM_FFI_ICHECK_EQ(call->args.size(), 5U)
            << "tvm_access_ptr expects 5 arguments";
        auto dtype = DecodeAccessDType(call->args[0]);
        if (!dtype.has_value()) {
          return UnknownAddress("access pointer element type is unavailable");
        }
        ByteAddress base = ResolveAddress(call->args[1], visiting, nesting + 1);
        if (base.kind == ByteAddress::Kind::kUnknown) {
          return base;
        }
        base.offset_bytes = analyzer_.Simplify(
            base.offset_bytes +
            Cast(DataType::Int(64),
                 call->args[2] *
                     IntImm(DataType::Int(64), dtype.value().bits() / 8)));
        return base;
      }
      if (call->op.same_as(builtin::address_of())) {
        if (call->args.size() != 1U) {
          return UnknownAddress("address_of expects one argument");
        }
        const auto *load = call->args[0].as<BufferLoadNode>();
        if (load == nullptr) {
          return UnknownAddress("address_of expects a BufferLoad");
        }
        if (load->indices.size() != 1U) {
          return UnknownAddress(
              "address_of with a non-flat index is unsupported");
        }
        ByteAddress base =
            ResolveAddress(load->buffer->data, visiting, nesting + 1);
        if (base.kind == ByteAddress::Kind::kUnknown) {
          return base;
        }
        base.offset_bytes = analyzer_.Simplify(
            base.offset_bytes +
            Cast(DataType::Int(64),
                 load->indices[0] *
                     IntImm(DataType::Int(64), load->dtype.bits() / 8)));
        return base;
      }
      if (call->op.same_as(tirx::builtin::reinterpret()) &&
          call->args.size() == 1U && current.dtype().is_handle()) {
        if (call->args[0].dtype().is_handle()) {
          // Pointer type re-view: the byte address is unchanged.
          return ResolveAddress(call->args[0], visiting, nesting + 1);
        }
        if (call->args[0].dtype().is_scalar()) {
          ByteAddress addr;
          addr.kind = ByteAddress::Kind::kIntegerBase;
          addr.base_bytes = Cast(DataType::Int(64), call->args[0]);
          addr.offset_bytes = IntImm(DataType::Int(64), 0);
          return addr;
        }
      }
    }
    return UnknownAddress("unrecognized pointer expression");
  }

  Proof Prove(const ByteAddress &address, const PrimExpr &extra_bytes,
              int required_alignment) {
    Proof proof;
    if (required_alignment <= 1) {
      proof.outcome = Outcome::kAligned;
      return proof;
    }
    if (address.kind == ByteAddress::Kind::kUnknown) {
      proof.outcome = Outcome::kUnknown;
      proof.error = address.error;
      return proof;
    }
    PrimExpr base = address.base_bytes.defined() ? address.base_bytes
                                                 : IntImm(DataType::Int(64), 0);
    PrimExpr total = analyzer_.Simplify(Cast(DataType::Int(64), base) +
                                        address.offset_bytes + extra_bytes);
    proof.byte_offset = total;
    PrimExpr remainder = analyzer_.Simplify(
        floormod(total, IntImm(DataType::Int(64), required_alignment)));
    proof.remainder = remainder;
    if (analyzer_.CanProve(remainder == 0)) {
      proof.outcome = Outcome::kAligned;
    } else if (analyzer_.CanProve(remainder != 0)) {
      proof.outcome = Outcome::kMisaligned;
    } else {
      proof.outcome = Outcome::kUnknown;
      proof.error = "cannot prove alignment for every execution";
    }
    return proof;
  }

  void ReportFailure(const StoreMatch &match, size_t store_index,
                     const ByteAddress &address, const Proof &proof) const {
    std::ostringstream oss;
    oss << "[VerifyPTOCodegenLegality] kernel=" << function_name_ << ", store=#"
        << store_index << ": ";
    if (proof.outcome == Outcome::kMisaligned) {
      oss << "destination is misaligned: byte remainder=" << proof.remainder
          << ", required alignment=" << match.required_alignment;
    } else {
      oss << "cannot prove destination alignment: "
          << (proof.error.empty() ? "alignment is not provable" : proof.error)
          << ", required alignment=" << match.required_alignment;
    }
    oss << "\n  mode=" << match.mode_name
        << " vstore, dtype=" << match.value_dtype
        << ", element bytes=" << match.elem_bytes;
    if (match.has_descriptor) {
      oss << "\n  descriptor offset=" << match.descriptor_offset << " (unit "
          << match.descriptor_dtype << ")";
    } else {
      oss << "\n  descriptor offset=<none>";
    }
    oss << ", instruction offset=" << match.instruction_offset << " (unit ";
    if (match.pointer_dtype.has_value()) {
      oss << match.pointer_dtype.value();
    } else {
      oss << "<unresolved>";
    }
    oss << ")";
    if (proof.byte_offset.defined()) {
      oss << "\n  normalized byte offset=" << proof.byte_offset;
    }
    if (address.root != nullptr) {
      oss << "\n  root=" << address.root->name_hint;
    } else if (address.kind == ByteAddress::Kind::kIntegerBase) {
      oss << "\n  base byte address=" << address.base_bytes;
    } else if (!address.error.empty()) {
      oss << "\n  address resolution: " << address.error;
    }
    if (match.mode == StoreMode::kPointStores) {
      oss << "\n  point-writeback explicit-mask vstore requires a destination "
             "aligned to the stored element width ("
          << match.required_alignment
          << " bytes); check the starting byte address and offsets";
    } else {
      oss << "\n  ordinary explicit-mask vstore requires a provably 32-byte "
             "aligned destination; consider a physical row stride whose byte "
             "size is a multiple of 32 and check the starting "
             "column/elem_offset";
    }
    LOG(FATAL) << oss.str();
  }

  void CheckStore(const CallNode *op) {
    StoreMatch match = MatchStore(op);
    if (!match.checked) {
      return;
    }
    const size_t store_index = next_store_index_++;
    std::unordered_set<const VarNode *> visiting;
    ByteAddress address = ResolveAddress(op->args[1], &visiting);
    std::unordered_set<const VarNode *> dtype_visiting;
    match.pointer_dtype =
        ResolvePointerElementType(op->args[1], &dtype_visiting);
    Proof proof;
    if (match.required_alignment > 1) {
      std::string dtype_error;
      if (!match.pointer_dtype.has_value()) {
        dtype_error = "resulting pointer element type is unavailable";
      } else {
        const DataType ptr_dtype = match.pointer_dtype.value();
        const int ptr_bits = ptr_dtype.bits();
        if (ptr_bits < 8 || ptr_bits % 8 != 0) {
          dtype_error =
              "resulting pointer element type is not byte-addressable";
        } else {
          const PrimExpr instruction_bytes =
              Cast(DataType::Int(64),
                   op->args[2] * IntImm(DataType::Int(64), ptr_bits / 8));
          proof = Prove(address, instruction_bytes, match.required_alignment);
        }
      }
      if (!dtype_error.empty()) {
        proof.outcome = Outcome::kUnknown;
        proof.error = dtype_error;
        proof.byte_offset = PrimExpr();
      }
    } else {
      // A one-byte natural requirement holds for every byte address.
      proof.outcome = Outcome::kAligned;
    }
    if (proof.outcome == Outcome::kAligned) {
      return;
    }
    ReportFailure(match, store_index, address, proof);
  }

  std::string function_name_;
  size_t next_store_index_{0};
  std::unordered_set<const VarNode *> ub_roots_;
  std::unordered_map<const VarNode *, PrimExpr> pointer_defs_;
  std::unordered_map<const VarNode *, PrimExpr> value_defs_;
  // Bind/Let definitions are immutable SSA values with unique VarNode
  // identity, and every Var a definition references is already defined in an
  // enclosing scope.  A definition's classification therefore cannot change
  // while that Var can still be referenced; once its lexical scope exits the
  // VarNode cannot appear again.  The cache is intentionally not part of
  // AliasScope, and a Var used before its own definition is classified as a
  // free parameter (non-SSA IR is not claimed as supported).
  std::unordered_map<const VarNode *, bool> var_origin_safe_;
  size_t next_dag_symbol_{0};
};

tvm::transform::Pass VerifyPTOCodegenLegality() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    std::string name = "?";
    if (auto symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      name = symbol.value();
    }
    PTOMaskedStoreAlignmentChecker checker(name);
    checker.Run(f);
    return f;
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.VerifyPTOCodegenLegality", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VerifyPTOCodegenLegality",
                        VerifyPTOCodegenLegality);
}

} // namespace transform
} // namespace tl
} // namespace tvm
