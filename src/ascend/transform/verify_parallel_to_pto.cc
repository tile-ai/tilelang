// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file verify_parallel_to_pto.cc
 * \brief Read-only input-contract verification for the PTO Parallel
 * vectorize path. Runs on the PTO target only.
 *
 * The pass checks every SIMD_VF block that contains a T.Parallel unit:
 * lanes validity, loop structure/ranges, lane fragment presence and
 * invertibility, per-access patterns (continuous / lane-uniform), the
 * value-expression whitelist, element dtype constraints and region-level
 * rules (no GM access, no mixed hand-written VMI, no threadIdx binding).
 * References are checked against a lexical definition environment:
 * params and buffer data pointers at entry; outer loop vars,
 * thread_extent bindings, flat binds and buffer allocations registered
 * at their statement position and restored on scope exit, so a unit
 * sees exactly what is defined where it appears. The first error
 * terminates compilation with a `[VerifyParallelToPTO]` diagnostic.
 *
 * Pure hand-written VMI regions (no T.Parallel) and SIMT_VF blocks are
 * skipped entirely; a region without T.Parallel does not need the lanes
 * annotation.
 */

#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

#include "../../transform/common/attr.h"
#include "parallel_to_pto_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
namespace pto_analysis = ::tvm::tl::pto;

namespace {

// Collects every handle-valued Bind (a data-pointer alias) of a statement
// tree, so a converted region can reject uses of those aliases before any
// scope check consumes the set.
class VerifyParallelToPTOImpl : public StmtExprVisitor {
public:
  void Verify(const PrimFunc &f) {
    // The definition environment starts at function entry with
    // the params and the buffer data pointers. Every other definition is
    // registered at its lexical position during the traversal below and
    // removed when its scope exits.
    for (const auto &p : f->params) {
      defs_.insert(p.get());
    }
    for (const auto &kv : f->buffer_map) {
      defs_.insert(kv.second->data.get());
    }
    CollectVolatileAllocs(f->body);
    pto::CollectHandleBindTargets(f->body, &handle_bind_targets_);
    VisitStmt(f->body);
  }

private:
  // Function-level pre-scan for AllocBuffer nodes carrying the
  // tirx.volatile annotation. Presence of the annotation marks the
  // allocation volatile (PTO codegen semantics:
  // `annotations.count(tirx::attr::kVolatile)`); a False value is NOT
  // interpreted as opt-out. The buffer *data Var* identity is recorded,
  // so any other Buffer view over the same data is covered as well.
  // Lexical visibility of the var stays with the existing definition
  // checks — this set only answers "is this data volatile".
  void CollectVolatileAllocs(const Stmt &body) {
    PostOrderVisit(body, [&](const ObjectRef &n) {
      if (const auto *alloc = n.as<AllocBufferNode>()) {
        if (alloc->annotations.count(tirx::attr::kVolatile)) {
          volatile_allocs_.insert(alloc->buffer->data.get());
        }
      }
    });
  }

private:
  [[noreturn]] void Fail(const std::string &category,
                         const std::string &message) {
    std::ostringstream oss;
    oss << "[VerifyParallelToPTO] " << category << ": " << message;
    // Diagnostics carry the SIMD_VF/Parallel position. The prefix format
    // above is unchanged; the position is appended so existing
    // category/detail matching keeps working.
    if (!unit_context_.empty()) {
      oss << " [at " << unit_context_ << "]";
    }
    LOG(FATAL) << oss.str();
  }

  // Region-level context — failures between/around units report
  // at least the region (and the lane count once it is known).
  void SetRegionContext() {
    std::ostringstream oss;
    oss << "region #" << current_region_;
    if (current_lanes_ > 0) {
      oss << " (lanes " << current_lanes_ << ")";
    }
    unit_context_ = oss.str();
  }

  // ------------------------------------------------------------------
  // Unified unit discovery — same rules as the vectorize pass:
  // sequential units are allowed; each unit is a 1D Parallel or a
  // two-layer adjacent Parallel nest (2D). A third parallel
  // layer, a Parallel under IfThenElse/While, and other unsupported
  // structures are rejected explicitly.
  // ------------------------------------------------------------------
  int64_t ReadLanes(const SBlockNode *block) {
    auto anno = block->annotations.Get("tl.simdvf_lanes");
    if (!anno.has_value()) {
      Fail("lanes", "SIMD_VF block with T.Parallel must carry the "
                    "tl.simdvf_lanes annotation (written by "
                    "T.SimdVF(lanes=...))");
    }
    auto imm = anno.value().try_cast<IntImm>();
    if (!imm.has_value() || imm.value()->dtype != DataType::Int(64)) {
      Fail("lanes", "tl.simdvf_lanes must be an Int64 constant");
    }
    int64_t lanes = imm.value()->value;
    if (lanes != 64 && lanes != 128 && lanes != 256) {
      std::ostringstream oss;
      oss << "T.SimdVF lanes must be one of {64, 128, 256}, got " << lanes;
      Fail("lanes", oss.str());
    }
    return lanes;
  }

  // ------------------------------------------------------------------
  // Region-level checks: everything inside a converting SIMD_VF region
  // must obey the region rules, not just the parallel body.
  // ------------------------------------------------------------------
  class RegionVisitor : public StmtExprVisitor {
  public:
    RegionVisitor(VerifyParallelToPTOImpl *parent) : parent_(parent) {}

    void VisitStmt_(const BufferStoreNode *op) final {
      CheckScope(op->buffer);
      // The base visitor skips BufferStore.predicate
      // (stmt_functor.cc:85); accesses inside it must reach the same
      // admission check (the field sits outside the converting Parallel
      // unit, so the unit-level predicate rule cannot cover it).
      if (op->predicate.defined()) {
        VisitExpr(op->predicate.value());
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      CheckScope(op->buffer);
      // Same gap for BufferLoad.predicate (expr_functor.cc:36);
      // indices stay with the base visitor.
      if (op->predicate.defined()) {
        VisitExpr(op->predicate.value());
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const CallNode *op) final {
      if (const auto *op_node = op->op.as<OpNode>()) {
        std::string name = op_node->name;
        if (name.rfind("tl.vmi.", 0) == 0) {
          parent_->Fail(
              "mixing",
              "hand-written tl.vmi.* calls cannot be mixed with T.Parallel "
              "inside a converting SIMD_VF block");
        }
        if (name == "tl.tileop.reduce") {
          parent_->Fail("expression", "cross-lane reduce is not supported");
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const AttrStmtNode *op) final {
      if (op->attr_key == tirx::attr::thread_extent) {
        // Name the binding that triggered the rejection: a region can carry
        // several thread_extent scopes and the message must say which one.
        // The prefix says "thread bindings" — this branch also rejects
        // blockIdx bindings, not only threadIdx.
        std::ostringstream oss;
        oss << "thread bindings are not allowed inside a converting "
               "SIMD_VF block (found `thread_extent` on binding `";
        if (const auto *iv = op->node.as<IterVarNode>()) {
          oss << iv->var->name_hint;
        } else {
          oss << op->node;
        }
        oss << "`)";
        parent_->Fail("region", oss.str());
      }
      // The legacy volatile_scope marker declares volatile access
      // semantics that are not currently supported; reject it under the
      // same rule instead of letting it pass as ordinary metadata. Scope
      // note: only markers *inside a converting region* are covered here
      // — a volatile_scope wrapping a whole SIMD_VF block from outside
      // never reaches this visitor and stays accepted.
      if (op->attr_key == tl::attr::volatile_scope) {
        parent_->Fail("memory access",
                      "volatile_scope attribute inside a converting region "
                      "marks accesses volatile; volatile accesses are not "
                      "yet supported");
      }
      // The base visitor only descends into value/body. A
      // PrimExpr held by `node` (metadata) can itself contain buffer
      // accesses (a bare BufferLoad or a composite Add/Cast over one, a
      // view over the same volatile data Var); run the same admission
      // walk over the full expression so those loads reach CheckScope.
      // Non-PrimExpr markers keep the existing contract (the unit-level
      // metadata check rejects containers).
      if (auto ne = op->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          VisitExpr(GetRef<PrimExpr>(pe));
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const ForNode *op) final {
      // A For-form thread binding (the public T.thread_binding API
      // constructs exactly this shape) must obey the same region-wide ban
      // as the thread_extent AttrStmt form above — the shared unit scan
      // only rejects wrappers *around* a Parallel, not sibling bindings.
      // This visitor only runs on regions that contain a Parallel, so
      // hand-written VMI regions without one are unaffected. Vectorized
      // sibling loops are intentionally NOT rejected here (out of scope).
      if (op->kind == ForKind::kThreadBinding) {
        std::ostringstream oss;
        oss << "thread bindings are not allowed inside a converting "
               "SIMD_VF block (found ThreadBinding loop on `"
            << op->loop_var->name_hint << "`";
        if (op->thread_binding.defined()) {
          oss << " bound to " << op->thread_binding.value()->thread_tag;
        }
        oss << ")";
        parent_->Fail("region", oss.str());
      }
      StmtExprVisitor::VisitStmt_(op);
    }

  private:
    void CheckScope(const Buffer &buffer) {
      // A buffer declared over a pointer a handle Bind produced (Bind(q,
      // A.data); DeclBuffer(B, data=q)) is a view whose storage the lane
      // analysis cannot follow: q may be an external pointer, a reinterpret
      // cast or a laundered copy, and the origin says nothing about where the
      // view really lives. Reject the access itself instead of letting the
      // ordering analysis guess -- a conservative ordering cannot repair an
      // access that should never have been admitted (a volatile read or a
      // cast the ISA cannot perform, for instance).
      if (parent_->handle_bind_targets_.count(buffer->data.get()) != 0) {
        std::ostringstream oss;
        oss << "buffer `" << buffer->name
            << "` is declared over a data pointer bound by a handle Bind, "
               "which is not supported inside a converting region; access the "
               "buffer the pointer came from instead";
        parent_->Fail("memory access", oss.str());
      }
      // Volatile accesses are outside the supported contract
      // (an unused volatile read must not be DCE-deleted, and a live
      // uniform volatile read must not change its read count through the
      // sload+broadcast lowering). The data-Var identity is matched, so
      // any Buffer view over a volatile allocation is covered.
      if (parent_->volatile_allocs_.count(buffer->data.get()) != 0) {
        std::ostringstream oss;
        oss << "buffer `" << buffer->name
            << "` is allocated with the tirx.volatile annotation; volatile "
               "accesses are not yet supported";
        parent_->Fail("memory access", oss.str());
      }
      const std::string &scope = buffer.scope();
      if (scope == "global") {
        std::ostringstream oss;
        oss << "SIMD_VF block must not access GM directly (buffer `"
            << buffer->name << "` has scope global); stage through UB first";
        parent_->Fail("memory access", oss.str());
      }
      // The remaining scopes only support UB accesses. local /
      // shared.l1 (L1) and other scopes are rejected.
      if (scope != "shared" && scope != "shared.dyn") {
        std::ostringstream oss;
        oss << "buffer `" << buffer->name << "` has scope `" << scope
            << "`; only UB (shared/shared.dyn) accesses are supported";
        parent_->Fail("memory access", oss.str());
      }
      if (!pto_analysis::IsSupportedElementDType(buffer->dtype)) {
        std::ostringstream oss;
        oss << "buffer `" << buffer->name << "` element dtype " << buffer->dtype
            << " is not supported by the unified element "
               "addressing (no sub-byte packed types, no "
               "pre-vectorized elements, no FP64, 8/16/32-bit widths only)";
        parent_->Fail("dtype", oss.str());
      }
    }
    VerifyParallelToPTOImpl *parent_;
  };

  // ------------------------------------------------------------------
  // Unit-level checks
  // ------------------------------------------------------------------
  class UnitVisitor : public StmtExprVisitor {
  public:
    UnitVisitor(VerifyParallelToPTOImpl *parent, const Var &loop_var,
                const pto_analysis::PtoInverseMapping &mapping,
                const std::set<const VarNode *> &external_defs)
        : parent_(parent), loop_var_(loop_var), mapping_(mapping),
          external_defs_(external_defs) {
      // In a 2D unit `loop_var` is the vectorized coordinate; the
      // other one is kept as a serial loop and classifies as uniform.
      if (mapping.is_2d) {
        serial_var_ =
            mapping.select_inner ? mapping.outer_var : mapping.inner_var;
      }
    }

    void VisitStmt_(const AttrStmtNode *op) final {
      // Analyze the *whole* metadata expression with the shared
      // lane-use analysis — composite expressions (Add/Cast over loads)
      // classify correctly, bind identity is preserved, and unknown forms
      // produce a diagnostic instead of a crash. Containers are still
      // rejected.
      auto check_meta_expr = [&](const PrimExpr &meta) {
        // The unified reference check runs *first*, so a
        // ghost anywhere in the metadata expression (including load
        // indices, through any admitted wrapper node) reports the
        // definition diagnostic before address analysis can mis-route
        // it. The base class descends into every child expression, so
        // the earlier hand-written node table's gap (Div/Mod/Min/Max/
        // comparisons treated as leaves, loop-var references slipping
        // through them) cannot recur. The metadata position additionally
        // rejects direct loop-var references and vector-classified
        // binds: the metadata path only remaps scalar var references.
        RefCheck checker(this, /*metadata=*/true);
        checker(meta);
        // Metadata-mode classification (Sub/FloorMod/compare/
        // StringImm propagate instead of being value-whitelist rejects).
        pto_analysis::LaneUseContext ctx = MakeLaneUseContext();
        std::string why;
        pto_analysis::LaneUse use =
            pto_analysis::AnalyzeLaneUseMeta(meta, &ctx, &why);
        if (use == pto_analysis::LaneUse::kVarying) {
          parent_->Fail("expression",
                        "AttrStmt metadata depends on lane-varying values; "
                        "metadata cannot yet be given lane semantics");
        }
        if (use == pto_analysis::LaneUse::kUnknown) {
          std::ostringstream oss;
          oss << "AttrStmt metadata cannot be analyzed: " << why;
          parent_->Fail("expression", oss.str());
        }
      };
      // op->node may be an Any holding a PrimExpr; containers and other
      // object kinds are rejected.
      if (auto ne = op->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          check_meta_expr(GetRef<PrimExpr>(pe));
        } else {
          parent_->Fail("expression",
                        "AttrStmt metadata must be a PrimExpr (containers "
                        "cannot be given lane semantics)");
        }
      }
      // Non-object metadata (int/str atoms) carries no Var references.
      check_meta_expr(op->value);
      // An AttrStmt body is a nested statement scope — a Bind
      // defined inside it must not leak to later siblings, while outer
      // binds stay visible inside.
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Integer> saved_classified = classified_;
      StmtExprVisitor::VisitStmt_(op);
      bind_env_ = std::move(saved_env);
      classified_ = std::move(saved_classified);
    }
    // The StmtVisitor base class already dispatches several node
    // kinds (AllocBuffer/DeclBuffer/...) so VisitStmtDefault_ alone is
    // not a whitelist. Reject those kinds explicitly.

    void VisitStmt_(const ForNode *op) final {
      // Loop-shell admission: a T.serial loop inside T.Parallel is admitted
      // as a preserved shell (Vectorize copies the loop through and converts
      // only its body). The induction variable is lane-uniform by
      // construction, so it is registered as a uniform definition while the
      // body runs through the same unit-level checks. Neither the shell nor
      // its body depends on the trip count being known at compile time: the
      // bound is copied through verbatim, so a runtime extent is admitted as
      // well. A constant extent must still be positive; a runtime extent is
      // the program's contract (the same assumption any `for` carries).
      // Non-serial kinds stay rejected at this stage.
      if (op->kind != ForKind::kSerial) {
        parent_->Fail("loop structure",
                      "only T.serial loops inside T.Parallel are supported "
                      "at this stage (pipelined and while loops are "
                      "rejected)");
      }
      // A zero trip count is a legal empty shell, the same semantics a plain
      // `for` has; only a negative constant is rejected.
      const auto *extent_imm = op->extent.as<IntImmNode>();
      if (extent_imm != nullptr && extent_imm->value < 0) {
        parent_->Fail("loop structure",
                      "T.serial inside T.Parallel must not have a negative "
                      "extent");
      }
      if (op->step.defined()) {
        // A non-unit step only changes which values the induction variable
        // takes; the shell stays lane-uniform, so Vectorize copies the step
        // through. Only a constant positive step is admitted (a runtime step
        // would need its own analysis).
        const auto *step_imm = op->step.value().as<IntImmNode>();
        if (step_imm == nullptr || step_imm->value <= 0) {
          parent_->Fail("loop structure",
                        "T.serial inside T.Parallel must have a positive "
                        "constant step at this stage");
        }
      }
      // The shell is copied through verbatim, so a bound that references a
      // lane-varying or unit-local variable would dangle after conversion.
      {
        bool uses_local_var = false;
        const auto probe = [&uses_local_var, this](const PrimExpr &expr) {
          PostOrderVisit(expr, [&uses_local_var, this](const ObjectRef &n) {
            if (const auto *v = n.as<VarNode>()) {
              // The lane variable disappears with the conversion, and a var
              // bound inside the unit is re-bound by Vectorize; either way a
              // copied bound would dangle.
              if (v == loop_var_.get() || bind_env_.count(GetRef<Var>(v))) {
                uses_local_var = true;
              }
            }
            // A bound read from a buffer (such as `T.serial(n[0])`) is out of
            // scope: the shell is copied through verbatim, so the read would be
            // re-evaluated against a buffer the conversion may have rewritten.
            if (n->IsInstance<BufferLoadNode>()) {
              uses_local_var = true;
            }
          });
        };
        probe(op->min);
        probe(op->extent);
        if (op->step.defined()) {
          probe(op->step.value());
        }
        if (uses_local_var) {
          parent_->Fail("loop structure",
                        "T.serial shell bounds must not reference the "
                        "parallel lane variable or a variable bound inside "
                        "the unit (the shell is copied through after those "
                        "variables are gone or re-bound)");
        }
      }
      // Every other expression position runs the reference check; the shell
      // bounds must too. Without it a variable that is never defined anywhere
      // is copied through verbatim into the output (the shell is rebuilt from
      // these very expressions).
      CheckDefinedVars(op->min);
      CheckDefinedVars(op->extent);
      if (op->step.defined()) {
        CheckDefinedVars(op->step.value());
      }
      // A bound that mentions no lane variable is not automatically supported:
      // the shell is copied into the vectorized loop header, so an opaque call
      // would run once per chunk instead of once per source iteration and its
      // result and side effects are unproven. Classify the bounds with the
      // analysis the metadata positions use: supported uniform scalars (kernel
      // parameters, outer serial variables, the 2D kept coordinate) stay
      // admitted, anything varying or unknown is rejected here.
      {
        pto_analysis::LaneUseContext ctx = MakeLaneUseContext();
        const auto admit_bound = [&](const PrimExpr &bound, const char *which) {
          std::string why;
          pto_analysis::LaneUse use =
              pto_analysis::AnalyzeLaneUseMeta(bound, &ctx, &why);
          if (use == pto_analysis::LaneUse::kUniform) {
            return;
          }
          std::ostringstream oss;
          oss << "T.serial shell " << which
              << " must be a lane-uniform scalar expression";
          if (!why.empty()) {
            oss << " (" << why << ")";
          }
          parent_->Fail("loop structure", oss.str());
        };
        admit_bound(op->min, "lower bound");
        admit_bound(op->extent, "extent");
      }
      defined_loop_vars_.insert(op->loop_var.get());
      // The range is a fact, not an assumption: a predicate like `_t < extent`
      // is provably true here, and the same fact is bound in Legalize before it
      // normalises a predicate away.
      analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent),
                     /*allow_override=*/true);
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Integer> saved_classified = classified_;
      classified_.Set(op->loop_var, Integer(0));
      StmtExprVisitor::VisitStmt_(op);
      bind_env_ = std::move(saved_env);
      classified_ = std::move(saved_classified);
      defined_loop_vars_.erase(op->loop_var.get());
    }
    void VisitStmt_(const IfThenElseNode *op) final {
      // The restricted lane-varying conditional forms are admitted.
      // LegalizeParallelToPTO rewrites them into the Select form
      // before VectorizeParallelToPTO runs, so the admitted shape is exactly
      // what Legalize can rewrite: buffer stores, no-ops, a SeqStmt of those,
      // or a nested conditional whose destinations are lane-varying
      // (lane-uniform stores stay rejected by the store check).
      CheckComputeValue(op->condition, "if condition");
      CheckConditionalBranch(op->then_case, "then");
      if (op->else_case.defined()) {
        CheckConditionalBranch(op->else_case.value(), "else");
      }
    }
    void VisitStmt_(const WhileNode *) final {
      parent_->Fail("loop structure",
                    "while loops inside T.Parallel are not yet supported");
    }
    void VisitStmt_(const SBlockNode *) final {
      parent_->Fail("loop structure",
                    "nested blocks inside T.Parallel are not yet "
                    "supported");
    }
    void VisitStmt_(const SBlockRealizeNode *) final {
      parent_->Fail("loop structure",
                    "nested blocks inside T.Parallel are not yet "
                    "supported");
    }
    void VisitStmt_(const AllocBufferNode *) final {
      parent_->Fail("loop structure",
                    "buffer allocation inside T.Parallel is not yet "
                    "supported");
    }
    void VisitStmt_(const DeclBufferNode *) final {
      parent_->Fail("loop structure",
                    "buffer declaration inside T.Parallel is not yet "
                    "supported");
    }
    void VisitStmtDefault_(const ffi::Object *op) final {
      // Backstop for kinds the base does not dispatch.
      std::ostringstream oss;
      oss << "statement `" << op->GetTypeKey()
          << "` inside T.Parallel is not yet supported";
      parent_->Fail("loop structure", oss.str());
    }
    void VisitStmt_(const AssertStmtNode *) final {
      parent_->Fail("loop structure",
                    "assert statements inside T.Parallel are not yet "
                    "supported");
    }
    void VisitStmt_(const BindNode *op) final {
      // Self/forward reference is rejected before registering.
      bool self_ref = false;
      PostOrderVisit(op->value, [&](const ObjectRef &n) {
        if (auto *v = n.as<VarNode>()) {
          if (v == op->var.get()) {
            self_ref = true;
          }
        }
      });
      if (self_ref) {
        std::ostringstream oss;
        oss << "bind `" << op->var->name_hint
            << "` references itself in its definition";
        parent_->Fail("expression", oss.str());
      }
      // Bind right-hand sides run the shared
      // compute-value admission (whitelist, cast plans, op types,
      // references, lane classification) — the same entry as Evaluate
      // and store values.
      pto_analysis::LaneUse use = CheckComputeValue(
          op->value, "bind `" + op->var->name_hint + "` value");
      // Register only after the analysis succeeded.
      bind_env_.Set(op->var, op->value);
      classified_.Set(op->var,
                      Integer(use == pto_analysis::LaneUse::kUniform ? 0 : 1));
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const EvaluateNode *op) final {
      // An Evaluate value is a compute value — same admission as
      // Bind RHS and store values (it previously ran no lane
      // classification, so a StringImm reached Vectorize's internal
      // error).
      CheckComputeValue(op->value, "evaluate value");
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) final {
      // The store value runs the shared compute-value
      // admission (whitelist, cast plans, op types, references, lane
      // classification).
      CheckComputeValue(op->value, "store value");
      // Store indices get the same unified reference check — the
      // address continuity proof can cancel a common ghost offset and
      // must not double as a definition check.
      CheckDefinedVars(op->indices);
      VerifyAccess(op->buffer, op->indices, /*is_write=*/true, op->predicate);
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      // Load indices are checked against the same definition
      // environment as values and store indices.
      CheckDefinedVars(op->indices);
      VerifyAccess(op->buffer, op->indices, /*is_write=*/false, op->predicate);
      StmtExprVisitor::VisitExpr_(op);
    }

  private:
    // One construction point for the lane-use context. The
    // metadata, operation-type and bind-definition entries share the same
    // definition environment, classification cache and guards.
    pto_analysis::LaneUseContext MakeLaneUseContext() {
      pto_analysis::LaneUseContext ctx;
      ctx.loop_var = loop_var_;
      ctx.serial_var = serial_var_;
      ctx.mapping = mapping_;
      ctx.analyzer = &analyzer_;
      ctx.bind_env = bind_env_;
      ctx.external_defs = external_defs_;
      // An admitted serial induction variable is lane-uniform and lexically
      // defined inside its shell, so the analysis must not treat it as an
      // unknown variable.
      for (const VarNode *v : defined_loop_vars_) {
        ctx.external_defs.insert(v);
      }
      ctx.classified = classified_;
      return ctx;
    }

    // Statements allowed inside a conditional branch under T.Parallel: the
    // branch must stay rewritable as Select arm(s) — buffer stores, no-ops,
    // SeqStmt of those, or a nested conditional (which recurses through the
    // same checks).
    void CheckConditionalBranch(const Stmt &stmt, const char *which) {
      if (const auto *seq = stmt.as<SeqStmtNode>()) {
        for (const Stmt &sub : seq->seq) {
          CheckConditionalBranch(sub, which);
        }
        return;
      }
      if (const auto *nested = stmt.as<IfThenElseNode>()) {
        VisitStmt_(nested);
        return;
      }
      if (const auto *store = stmt.as<BufferStoreNode>()) {
        // The access check is the whole admission here: a destination that is
        // continuous under the layout (start + lane over the full chunk) is
        // per-lane and can be paired; anything else -- interleaved,
        // lane-uniform or data-dependent -- is rejected by it with a precise
        // diagnostic.
        VisitStmt_(store);
        return;
      }
      if (pto_analysis::IsNoOpStmt(stmt)) {
        return;
      }
      std::ostringstream oss;
      oss << "If " << which
          << "-branch under T.Parallel must be buffer stores, nested "
             "conditionals or no-ops (the Select-rewritable shape); other "
             "statements are not supported at this stage";
      parent_->Fail("loop structure", oss.str());
    }

    // The unified reference check. One StmtExprVisitor
    // covers every expression position — values, load/store indices and
    // metadata node/value — relying on the base class's complete descent
    // (BufferLoad indices and every wrapper node the admission tables
    // accept are all visited). Every Var must be the loop var, a bind
    // defined earlier in this unit, or a definition in the lexical
    // snapshot; the metadata position additionally rejects direct
    // loop-var references and vector-classified binds because the
    // metadata path only remaps scalar var references.
    class RefCheck : public StmtExprVisitor {
    public:
      RefCheck(UnitVisitor *parent, bool metadata)
          : parent_(parent), metadata_(metadata) {}
      void VisitExpr_(const VarNode *v) final {
        if (parent_->serial_var_.defined() && v == parent_->serial_var_.get()) {
          // The kept coordinate becomes the new outer serial loop
          // variable. It is a uniform scalar and is not rewritten by the
          // metadata remap, so a direct metadata reference stays legal.
          return;
        }
        if (v == parent_->loop_var_.get()) {
          if (metadata_) {
            parent_->parent_->Fail(
                "expression",
                "AttrStmt metadata directly references the loop variable "
                "(even if algebraically uniform); the metadata path cannot "
                "rewrite it");
          }
          return;
        }
        if (parent_->bind_env_.count(GetRef<Var>(v))) {
          if (metadata_ && parent_->classified_.count(GetRef<Var>(v))) {
            Integer cls = parent_->classified_[GetRef<Var>(v)];
            if (cls->value != 0) {
              std::ostringstream oss;
              oss << "AttrStmt metadata references bind `" << v->name_hint
                  << "` whose value converts to a vector; metadata must "
                     "stay scalar";
              parent_->parent_->Fail("expression", oss.str());
            }
          }
          return;
        }
        if (parent_->external_defs_.count(v) != 0) {
          return;
        }
        if (parent_->defined_loop_vars_.count(v) != 0) {
          // Induction variable of an admitted serial shell; it is
          // lane-uniform and in lexical scope inside the shell body.
          return;
        }
        std::ostringstream oss;
        oss << "variable `" << v->name_hint
            << "` is used but never defined in this kernel";
        parent_->parent_->Fail("expression", oss.str());
      }

    private:
      UnitVisitor *parent_;
      bool metadata_;
    };

    void CheckDefinedVars(const PrimExpr &expr) {
      RefCheck checker(this, /*metadata=*/false);
      checker(expr);
    }
    void CheckDefinedVars(const Array<PrimExpr> &indices) {
      RefCheck checker(this, /*metadata=*/false);
      for (const auto &idx : indices) {
        checker(idx);
      }
    }

    // One admission entry for every compute-value position (Bind
    // RHS, Evaluate, store value): whitelist, cast plans, operation
    // types, reference check and lane classification. Before this only
    // the Bind path ran the lane classification, so a value the
    // whitelist lets through but that is not a legal scalar computation
    // (e.g. an external vector Var) reached Vectorize through Evaluate or
    // a store. `what` is the diagnostic prefix for the position.
    pto_analysis::LaneUse CheckComputeValue(const PrimExpr &value,
                                            const std::string &what) {
      if (auto reason = pto_analysis::FindUnsupportedExprNode(value)) {
        parent_->Fail("expression", what + ": " + reason.value());
      }
      CheckValueCasts(value);
      CheckBinaryOpTypes(value);
      CheckDefinedVars(value);
      pto_analysis::LaneUseContext ctx = MakeLaneUseContext();
      std::string why;
      pto_analysis::LaneUse use =
          pto_analysis::AnalyzeLaneUse(value, &ctx, &why);
      if (use == pto_analysis::LaneUse::kUnknown) {
        std::ostringstream oss;
        oss << what << " cannot be classified: " << why;
        parent_->Fail("expression", oss.str());
      }
      return use;
    }

    /*! The two-operand value nodes, one shared recursion shape: both
     * operands run through the same check, and the arithmetic kinds
     * additionally apply the VMI operation matrix. `vmi_op` carries the
     * kind's VMI op for Add/Sub/Mul/Div (vadd/vsub/vmul/vdiv) and is
     * nullptr for the relational / boolean kinds that Legalize and
     * Vectorize express as vcmp+vsel / vand|vor|vnot (recursion only).
     * The unary Not and the three-operand Select are handled separately
     * by the caller. */
    struct BinaryOperands {
      const char *vmi_op = nullptr;
      DataType dtype;
      PrimExpr a;
      PrimExpr b;
    };

    static std::optional<BinaryOperands> GetBinaryOperands(const PrimExpr &e) {
      if (const auto *n = e.as<AddNode>()) {
        return BinaryOperands{"vadd", n->dtype, n->a, n->b};
      }
      if (const auto *n = e.as<SubNode>()) {
        return BinaryOperands{"vsub", n->dtype, n->a, n->b};
      }
      if (const auto *n = e.as<MulNode>()) {
        return BinaryOperands{"vmul", n->dtype, n->a, n->b};
      }
      if (const auto *n = e.as<DivNode>()) {
        return BinaryOperands{"vdiv", n->dtype, n->a, n->b};
      }
      PrimExpr rel_a;
      PrimExpr rel_b;
      if (pto::MatchRelational(e, &rel_a, &rel_b) != nullptr) {
        return BinaryOperands{nullptr, e.dtype(), rel_a, rel_b};
      }
      if (const auto *n = e.as<AndNode>()) {
        return BinaryOperands{nullptr, n->dtype, n->a, n->b};
      }
      if (const auto *n = e.as<OrNode>()) {
        return BinaryOperands{nullptr, n->dtype, n->a, n->b};
      }
      return std::nullopt;
    }

    // Per-operation type rules driven by the operands'
    // lane classification. Uniform+Uniform stays scalar (only the base
    // dtype and PTO scalar-support are required); any Varying operand
    // makes the node a vector operation and the VMI op matrix applies.
    // Inner sub-expressions are classified individually, so an int8
    // scalar Mul inside a Varying Add is legal while an int8 vector Mul
    // stays rejected.
    void CheckBinaryOpTypes(const PrimExpr &expr) {
      std::function<void(const PrimExpr &)> check =
          [&](const PrimExpr &e) {
            // Ordered comparison of bool operands has no lane-mask lowering;
            // only EQ/NE is expressed in the mask domain.
            {
              PrimExpr rel_a;
              PrimExpr rel_b;
              const pto::RelationalKind *rel =
                  pto::MatchRelational(e, &rel_a, &rel_b);
              if (rel != nullptr && std::string(rel->cmp) != "eq" &&
                  std::string(rel->cmp) != "ne" && rel_a.dtype().is_bool() &&
                  rel_b.dtype().is_bool()) {
                LOG(FATAL) << "[VerifyParallelToPTO] type: ordered comparison "
                              "of bool operands has no lane-mask lowering";
              }
            }
            if (auto bin = GetBinaryOperands(e)) {
              check(bin->a);
              check(bin->b);
              if (bin->vmi_op != nullptr) {
                CheckOneOp(bin->vmi_op, bin->dtype, bin->a, bin->b);
              }
              return;
            }
            if (const auto *cast = e.as<CastNode>()) {
              check(cast->value);
              return;
            }
            // Select, like the relational / boolean nodes of
            // GetBinaryOperands, is expressed by Legalize and Vectorize as
            // vcmp+vsel. Operands recurse through the same check, so lane
            // classification and dtype rules still apply to whatever feeds
            // them.
            if (const auto *sel = e.as<SelectNode>()) {
              if (sel->true_value.dtype().is_bool() ||
                  sel->false_value.dtype().is_bool()) {
                LOG(FATAL) << "[VerifyParallelToPTO] type: a bool-valued "
                              "Select has no vreg lowering";
              }
              check(sel->condition);
              check(sel->true_value);
              check(sel->false_value);
              return;
            }
            if (const auto *notn = e.as<NotNode>()) {
              check(notn->a);
              return;
            }
            // Min/Max are rewritten to Select by Legalize, but their operands
            // must still pass the same type rules here (e.g. the
            // integer-bitwise ban).
            if (const auto *minn = e.as<MinNode>()) {
              // Legalize rewrites Min/Max to Select, which would introduce a
              // bool-ordered comparison and a bool-valued Select after Verify.
              if (minn->dtype.is_bool()) {
                LOG(FATAL) << "[VerifyParallelToPTO] type: Min/Max of bool "
                              "operands has no lane-mask lowering";
              }
              check(minn->a);
              check(minn->b);
              return;
            }
            if (const auto *maxn = e.as<MaxNode>()) {
              // Same gate as Min: Legalize rewrites Max to Select, which would
              // introduce a bool-ordered compare and a bool-valued Select.
              if (maxn->dtype.is_bool()) {
                LOG(FATAL) << "[VerifyParallelToPTO] type: Min/Max of bool "
                              "operands has no lane-mask lowering";
              }
              check(maxn->a);
              check(maxn->b);
              return;
            }
            if (const auto *call = e.as<CallNode>()) {
              if (auto call_op = call->op.as<Op>()) {
                const std::string &name = call_op.value()->name;
                // Boolean mask-combine calls (the shared
                // tirx.bitwise_* / tir.And|Or|Not admission table) and the
                // tl/tir.infinity() constant.
                // A mask combine only has a lowering for bool (lane-mask)
                // values.
                if (pto_analysis::MaskLogicVmiOpName(name) != nullptr) {
                  // An integer bitwise call is rejected with a proper
                  // diagnostic; CheckBinaryOpTypes is static, so report via
                  // LOG(FATAL) (the same channel Fail uses).
                  if (!call->dtype.is_bool()) {
                    LOG(FATAL)
                        << "[VerifyParallelToPTO] type: bitwise mask combine `"
                        << name
                        << "` requires bool (lane-mask) operands; an integer "
                           "bitwise call has no lane-mask lowering";
                  }
                  for (const auto &arg : call->args) {
                    check(arg);
                  }
                  return;
                }
                if (name == "tl.infinity" || name == "tir.infinity") {
                  return;
                }
                // The tirx.fabs/exp/log/sqrt unary math table; the original
                // tirx op name stays the CheckOneUnaryOp key.
                if (pto_analysis::VectorUnaryVmiOpName(name) != nullptr) {
                  check(call->args[0]);
                  CheckOneUnaryOp(name.c_str(), call->dtype, call->args[0]);
                  return;
                }
              }
            }
            if (const auto *load = e.as<BufferLoadNode>()) {
              // Index arithmetic is address analysis, not value arithmetic.
              (void)load;
              return;
            }
            // Vars/constants carry no operation to check.
          };
      check(expr);
    }

    void CheckOneUnaryOp(const char *op_name, DataType dtype,
                         const PrimExpr &arg) {
      pto_analysis::LaneUseContext ctx = MakeLaneUseContext();
      std::string why;
      pto_analysis::LaneUse a = pto_analysis::AnalyzeLaneUse(arg, &ctx, &why);
      if (a == pto_analysis::LaneUse::kUnknown) {
        return;
      }
      const bool vector_op = a == pto_analysis::LaneUse::kVarying;
      if (pto_analysis::IsSupportedFloat8(dtype)) {
        std::ostringstream oss;
        oss << op_name << " on "
            << (vector_op ? "lane-varying " : "lane-uniform ") << dtype
            << " is not supported (FP8 arithmetic is out of scope; convert "
               "to float32 explicitly first)";
        parent_->Fail("dtype", oss.str());
      }
      if (vector_op) {
        if (!pto_analysis::IsSupportedVectorUnaryOp(op_name, dtype)) {
          std::ostringstream oss;
          oss << op_name << " on " << dtype
              << " elements is not supported by the VMI operation matrix";
          parent_->Fail("dtype", oss.str());
        }
        return;
      }
      if (pto_analysis::IsFloatFamily(dtype) &&
          pto_analysis::ContainsRuntimeValue(arg)) {
        std::ostringstream oss;
        oss << op_name << " on " << dtype
            << " is a runtime lane-uniform float computation; it stays "
               "scalar inside SIMD_VF (no scalar float ALU by design). "
               "Compute it outside the SimdVF region, or wait for the "
               "gather/scatter milestone for vector write-back";
        parent_->Fail("dtype", oss.str());
      }
    }

    void CheckOneOp(const char *op_name, DataType dtype, const PrimExpr &lhs,
                    const PrimExpr &rhs) {
      pto_analysis::LaneUseContext ctx = MakeLaneUseContext();
      std::string why;
      pto_analysis::LaneUse a = pto_analysis::AnalyzeLaneUse(lhs, &ctx, &why);
      pto_analysis::LaneUse b = pto_analysis::AnalyzeLaneUse(rhs, &ctx, &why);
      if (a == pto_analysis::LaneUse::kUnknown ||
          b == pto_analysis::LaneUse::kUnknown) {
        // The operand's access analysis failed (stride/pattern/etc.); the
        // dedicated BufferLoad visitor reports that with the correct
        // category later in this same unit walk. Do not double-report it
        // as a dtype error here.
        return;
      }
      bool vector_op = (a == pto_analysis::LaneUse::kVarying ||
                        b == pto_analysis::LaneUse::kVarying);
      // FP8 arithmetic is rejected in both forms, and deliberately before the
      // generic matrix: the conversion matrix admits FP8 operands, so "not in
      // the VMI op matrix" alone would be an opaque reason. No implicit
      // FP8 -> FP32 promotion is inserted either.
      if (pto_analysis::IsSupportedFloat8(dtype)) {
        std::ostringstream oss;
        oss << op_name << " on "
            << (vector_op ? "lane-varying " : "lane-uniform ") << dtype
            << " is not supported (FP8 arithmetic is not supported; convert "
               "to float32 explicitly first)";
        parent_->Fail("dtype", oss.str());
      }
      if (vector_op) {
        // The node emits a VMI op: apply the operation matrix.
        if (!pto_analysis::IsSupportedVectorBinaryOp(op_name, dtype)) {
          std::ostringstream oss;
          oss << op_name << " on " << dtype
              << " elements is not supported by the VMI operation matrix";
          parent_->Fail("dtype", oss.str());
        }
        return;
      }
      // Lane-uniform stays scalar by design, so a runtime-valued float
      // computation here runs on the scalar float ALU inside SIMD_VF.
      // Compile-time constants fold and integer scalars have no such
      // premise, so only float-family runtime values are rejected.
      if (pto_analysis::IsFloatFamily(dtype) &&
          (pto_analysis::ContainsRuntimeValue(lhs) ||
           pto_analysis::ContainsRuntimeValue(rhs))) {
        std::ostringstream oss;
        oss << op_name << " on " << dtype
            << " is a runtime lane-uniform float computation; it stays "
               "scalar inside SIMD_VF (no scalar float ALU by design). "
               "Compute it outside the SimdVF region, or wait for the "
               "gather/scatter milestone for vector write-back";
        parent_->Fail("dtype", oss.str());
      }
      // Pure scalar op: base dtype legality only (int8 scalar mul is
      // fine; int24/FP64 still rejected by the base dtype rule).
      if (!pto_analysis::IsSupportedElementDType(dtype)) {
        std::ostringstream oss;
        oss << "scalar " << op_name << " on " << dtype
            << " is not supported by the base element dtype rule";
        parent_->Fail("dtype", oss.str());
      }
    }

  public:
    // Unified value-Cast check: called for Bind, Evaluate
    // and store values alike. A Cast is only accepted when PlanValueCast
    // proves a VMI-equivalent conversion (attributes included); Cast
    // nodes inside BufferLoad *indices* are index arithmetic handled by
    // address analysis and are not checked here.
    void CheckValueCasts(const PrimExpr &expr) {
      class CastCheck : public StmtExprVisitor {
      public:
        CastCheck(UnitVisitor *parent) : parent_(parent) {}
        void VisitExpr_(const CastNode *op) final {
          // Classify the *input of this Cast* with the existing lane-use
          // analysis, then hand the explicit kind to the shared planner.
          pto_analysis::LaneUseContext ctx = parent_->MakeLaneUseContext();
          std::string lane_why;
          pto_analysis::LaneUse use =
              pto_analysis::AnalyzeLaneUse(op->value, &ctx, &lane_why);
          if (use == pto_analysis::LaneUse::kUnknown) {
            std::ostringstream oss;
            oss << "Cast " << op->value.dtype() << " -> " << op->dtype
                << " cannot classify its input: " << lane_why;
            parent_->parent_->Fail("dtype", oss.str());
            return;
          }
          pto_analysis::CastContext context;
          if (use == pto_analysis::LaneUse::kUniform) {
            context.value_kind = pto_analysis::CastValueKind::kScalar;
            context.logical_lanes = 1;
          } else {
            context.value_kind = pto_analysis::CastValueKind::kVectorVmi;
            context.logical_lanes = parent_->mapping_.lanes;
          }
          std::string why;
          auto plan = pto_analysis::PlanValueCast(op, context, &why);
          if (!plan.has_value()) {
            std::ostringstream oss;
            oss << "Cast " << op->value.dtype() << " -> " << op->dtype
                << " has no supported conversion";
            if (!why.empty()) {
              oss << ": " << why;
            }
            parent_->parent_->Fail("dtype", oss.str());
          }
          // Nested value casts each plan independently; annotations are not
          // computation values and are validated by the planner.
          VisitExpr(op->value);
        }
        void VisitExpr_(const BufferLoadNode *) final {
          // Index casts belong to address analysis; do not descend.
        }

      private:
        UnitVisitor *parent_;
      };
      CastCheck checker(this);
      checker(expr);
    }

    void VerifyAccess(const Buffer &buffer, const Array<PrimExpr> &indices,
                      bool is_write, const Optional<PrimExpr> &predicate) {
      // The predicate *value* is admitted by AdmitPredicates, which runs once
      // per converting region and owns the lexical facts. Only the reference
      // check is repeated here, next to the access it belongs to.
      if (predicate.defined()) {
        CheckDefinedVars(predicate.value());
      }
      try {
        auto access = pto_analysis::AnalyzeBufferAccess(
            buffer, indices, is_write, loop_var_, mapping_, &analyzer_,
            bind_env_);
        if (access.pattern != pto_analysis::AccessPattern::kContinuous &&
            access.pattern != pto_analysis::AccessPattern::kLaneUniform) {
          std::ostringstream oss;
          oss << "buffer `" << buffer->name << "` access pattern "
              << pto_analysis::AccessPatternToString(access.pattern)
              << " is not supported";
          parent_->Fail("memory access", oss.str());
        }
        if (access.pattern == pto_analysis::AccessPattern::kLaneUniform &&
            is_write) {
          // Lane-uniform is only a load pattern.
          std::ostringstream oss;
          oss << "buffer `" << buffer->name
              << "`: lane-uniform stores are not yet supported";
          parent_->Fail("memory access", oss.str());
        }
      } catch (const pto_analysis::PtoAnalysisError &err) {
        parent_->Fail("memory access", err.what());
      }
    }

    VerifyParallelToPTOImpl *parent_;
    const Var &loop_var_;
    /*! 2D only — the non-vectorized coordinate (uniform). */
    Var serial_var_;
    const pto_analysis::PtoInverseMapping &mapping_;
    arith::Analyzer analyzer_;
    Map<Var, PrimExpr> bind_env_;
    // Classification recorded at each Bind definition
    // (0 = uniform, 1 = varying) for this unit.
    Map<Var, Integer> classified_;
    // The lexical definition snapshot at the unit's position —
    // params, buffer data pointers, and every definition (outer loop
    // vars, thread_extent bindings, binds, allocations) that is in scope
    // where the unit appears.
    std::set<const VarNode *> external_defs_;
    // Induction variables of admitted serial loop shells (constant-bound
    // T.serial inside T.Parallel). In scope only while the shell's body
    // is being visited; classified as lane-uniform.
    std::set<const VarNode *> defined_loop_vars_;
  };

  // ------------------------------------------------------------------
  // Scope tracking: every definition — outer serial loop vars,
  // thread_extent bindings, flat binds, buffer allocations/declarations
  // — enters the environment at its lexical position and leaves it when
  // the enclosing scope exits. A unit is verified against a snapshot of
  // this environment at the unit's position, so a sibling scope's
  // definitions can no longer leak in and legitimately-defined outer
  // scalars (blockIdx, outer binds) are no longer rejected.
  // ------------------------------------------------------------------
  void VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kParallel) {
      std::set<const VarNode *> saved = defs_;
      defs_.insert(op->loop_var.get());
      // An enclosing loop range is a fact a predicate inside a nested
      // converting region may rest on (`j < 8`), so it travels with the
      // traversal and is handed to the region's admission walk.
      outer_ranges_.push_back(
          {op->loop_var, Range::FromMinExtent(op->min, op->extent)});
      StmtExprVisitor::VisitStmt_(op);
      outer_ranges_.pop_back();
      defs_ = std::move(saved);
      return;
    }
    // A parallel For outside a converting region is not this pass's
    // concern; its iteration var has no ordered single-assignment
    // definition to track here.
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        // blockIdx/cthread bindings (e.g. `with T.Kernel(2) as bx`) are
        // legitimately-defined outer scalars for the attr body. The
        // no-threadIdx-inside-the-converting-region rule is
        // unaffected: it is enforced by the RegionVisitor inside the
        // region, not here.
        std::set<const VarNode *> saved = defs_;
        defs_.insert(iv->var.get());
        StmtExprVisitor::VisitStmt_(op);
        defs_ = std::move(saved);
        return;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode *op) final {
    // Flat sequence semantics: the bound var is visible to all
    // subsequent statements of the enclosing scope; the scope owners
    // above restore the environment on exit.
    // A handle-valued Bind (Bind(q, A.data)) produces a pointer the lane
    // analysis cannot follow; the function-level pre-pass records its target
    // so any view declared over it is rejected by CheckScope. Plain PTO
    // functions without a converting region keep their behaviour.
    defs_.insert(op->var.get());
    // A Bind that stays valid around the converting region is a shared fact for
    // its predicates (the same safety rule the region walk applies: a snapshot
    // over memory must stay opaque).
    if (!op->value.dtype().is_handle() &&
        pto_analysis::IsSafelyReevaluableValue(op->value)) {
      outer_binds_.Set(op->var, op->value);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    // Flat statement (no body): the allocation is visible to the rest of
    // the enclosing scope.
    defs_.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    defs_.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    // Branch bodies are independent scopes.
    StmtExprVisitor::VisitExpr(op->condition);
    {
      std::set<const VarNode *> saved = defs_;
      VisitStmt(op->then_case);
      defs_ = std::move(saved);
    }
    if (op->else_case.defined()) {
      std::set<const VarNode *> saved = defs_;
      VisitStmt(op->else_case.value());
      defs_ = std::move(saved);
    }
  }

  void VisitStmt_(const WhileNode *op) final {
    StmtExprVisitor::VisitExpr(op->condition);
    std::set<const VarNode *> saved = defs_;
    StmtExprVisitor::VisitStmt(op->body);
    defs_ = std::move(saved);
  }

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF") {
      std::set<const VarNode *> saved = defs_;
      for (const auto &iv : op->iter_vars) {
        defs_.insert(iv->var.get());
      }
      for (const auto &buf : op->alloc_buffers) {
        defs_.insert(buf->data.get());
      }
      StmtExprVisitor::VisitStmt_(op);
      defs_ = std::move(saved);
      return;
    }
    HandleSimdVF(op);
  }

  // ------------------------------------------------------------------
  // SIMD_VF region entry (converting regions only)
  // ------------------------------------------------------------------
  void HandleSimdVF(const SBlockNode *op) {
    // Block-scoped definitions (allocations, iter vars) are in scope for
    // the whole region (supersedes the earlier simd_vf_allocs_ side
    // table).
    std::set<const VarNode *> saved = defs_;
    for (const auto &iv : op->iter_vars) {
      defs_.insert(iv->var.get());
    }
    for (const auto &buf : op->alloc_buffers) {
      defs_.insert(buf->data.get());
    }

    // Shared region scan: same implementation as Vectorize.
    const Stmt &region = op->body;
    pto_analysis::PtoRegionScan scan = pto_analysis::ScanPtoRegion(region);
    if (!scan.has_any_parallel) {
      // Pure hand-written VMI region (possibly with control flow): skipped
      // entirely; no lanes annotation required.
      defs_ = std::move(saved);
      return;
    }
    // Number the converting region for diagnostics. The lane count is
    // reset to unknown first: a failure before this region's own lanes
    // are read must not inherit the previous region's value.
    current_region_ = region_seq_++;
    current_lanes_ = 0;
    SetRegionContext();
    if (scan.has_unsupported_wrapper) {
      Fail("loop structure", scan.issue);
    }
    const std::vector<For> &units = scan.units;

    // Only now is the lanes annotation required.
    int64_t lanes = ReadLanes(op);
    current_lanes_ = lanes;
    SetRegionContext(); // refresh the region context with the lane count

    // Region-level checks across the whole converting region.
    RegionVisitor region_visitor(this);
    region_visitor(region);

    // Access-predicate admission, with the lexical facts a predicate may rest
    // on (enclosing loop ranges, scalar Bind values). A hand-written VMI region
    // without a unit keeps its old behaviour.
    if (!units.empty()) {
      AdmitPredicates(region);
    }

    // Per-unit checks in order (multiple sequential units OK) against the
    // lexical definition snapshot at each unit's position.
    size_t next_unit = 0;
    WalkRegion(region, units, lanes, &next_unit);
    ICHECK(next_unit == units.size())
        << "[VerifyParallelToPTO] internal error: region walk and unit "
           "scan disagree on the unit count";
    defs_ = std::move(saved);
  }

  // ------------------------------------------------------------------
  // Per-unit checks (structure, ranges, layout)
  // ------------------------------------------------------------------
  void VerifyUnit(const For &unit, int64_t lanes, size_t unit_index) {
    // Full per-unit context for diagnostics.
    {
      std::ostringstream oss;
      oss << "region #" << current_region_ << ", unit #" << unit_index
          << " (loop var `" << unit->loop_var->name_hint << "`, extent "
          << unit->extent << ", lanes " << lanes << ")";
      unit_context_ = oss.str();
    }
    if (!is_zero(unit->min)) {
      std::ostringstream oss;
      oss << "loop min must be 0, got " << unit->min;
      Fail("loop range", oss.str());
    }
    // The lane layout assumes unit stride; a non-unit step
    // changes the set of iteration points and has no supported
    // lowering.
    if (unit->step.defined() &&
        !is_one(analyzer_.Simplify(unit->step.value()))) {
      std::ostringstream oss;
      oss << "T.Parallel step must be 1, got " << unit->step;
      Fail("loop range", oss.str());
    }
    const int64_t *extent = as_const_int(unit->extent);
    if (extent == nullptr || *extent <= 0) {
      std::ostringstream oss;
      oss << "loop extent must be a positive compile-time constant, got "
          << unit->extent;
      Fail("loop range", oss.str());
    }
    if (!unit->annotations.count(attr::kPtoParallelLoopLayout)) {
      Fail("layout", "pto_parallel_loop_layout is missing on the "
                     "outermost T.Parallel");
    }

    // Identify the full loop nest and check every member loop
    // *before* the inversion helper (which relies on constant extents) so
    // an illegal inner loop gets its loop-range diagnostic instead of a
    // raw assertion.
    const ForNode *inner_probe = nullptr;
    if (const auto *body_for = unit->body.as<ForNode>()) {
      if (body_for->kind == ForKind::kParallel) {
        inner_probe = body_for;
      }
    }
    if (inner_probe != nullptr) {
      if (!is_zero(inner_probe->min)) {
        std::ostringstream oss;
        oss << "inner loop min must be 0, got " << inner_probe->min;
        Fail("loop range", oss.str());
      }
      if (inner_probe->step.defined() &&
          !is_one(analyzer_.Simplify(inner_probe->step.value()))) {
        std::ostringstream oss;
        oss << "inner T.Parallel step must be 1, got " << inner_probe->step;
        Fail("loop range", oss.str());
      }
      if (as_const_int(inner_probe->extent) == nullptr) {
        std::ostringstream oss;
        oss << "inner loop extent must be a compile-time constant, got "
            << inner_probe->extent;
        Fail("loop range", oss.str());
      } else if (*as_const_int(inner_probe->extent) <= 0) {
        std::ostringstream oss;
        oss << "inner loop extent must be a positive compile-time constant, "
               "got "
            << inner_probe->extent;
        Fail("loop range", oss.str());
      }
      if (inner_probe->loop_var.dtype() != DataType::Int(32)) {
        std::ostringstream oss;
        oss << "inner loop variable must be int32 (the dtype "
               "LayoutInference produces), got "
            << inner_probe->loop_var.dtype();
        Fail("loop range", oss.str());
      }
    }

    pto_analysis::PtoInverseMapping mapping = [&] {
      try {
        return pto_analysis::InvertPtoLaneLayout(unit);
      } catch (const pto_analysis::PtoAnalysisError &err) {
        Fail("layout", err.what());
        return pto_analysis::PtoInverseMapping{};
      }
    }();
    if (mapping.lanes != lanes) {
      std::ostringstream oss;
      oss << "layout lanes (" << mapping.lanes
          << ") disagree with tl.simdvf_lanes (" << lanes << ")";
      Fail("lanes", oss.str());
    }

    // A 2D unit is the outermost For of a two-layer adjacent
    // Parallel nest; the unit visitor runs on the innermost compute body
    // (the layout annotation only sits on the outer For). The
    // inner loop's range contract was already enforced by the
    // pre-inversion block above (min/step/extent/dtype), so no per-loop
    // check is repeated here.
    const ForNode *inner = nullptr;
    if (const auto *body_for = unit->body.as<ForNode>()) {
      if (body_for->kind == ForKind::kParallel) {
        inner = body_for;
      }
    }

    // The vectorized coordinate is the selected dimension of a 2D mapping
    // and the single loop var in 1D (the other 2D coordinate stays a
    // serial loop var and classifies as uniform).
    const Var vector_var =
        mapping.is_2d
            ? (mapping.select_inner ? mapping.inner_var : mapping.outer_var)
            : unit->loop_var;
    // The definition snapshot at the unit's lexical position.
    const Stmt &body_to_verify = inner != nullptr ? inner->body : unit->body;
    UnitVisitor unit_visitor(this, vector_var, mapping, defs_);
    unit_visitor(body_to_verify);
  }

  // Region-level statements (between/around the Parallel units)
  // previously only *registered* definitions without scanning their
  // expressions, so an undefined var in a region-level Bind RHS (or
  // AttrStmt metadata, or serial-For bounds) leaked into the output
  // unchecked. Check that every Var is defined in the current lexical
  // environment; call sites run this *before* registering new
  // definitions so lexical order is preserved (a self reference is
  // undefined too).
  void CheckRegionExprDefined(const PrimExpr &expr, const char *what) {
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *v = node.as<VarNode>()) {
        if (defs_.count(v) == 0) {
          std::ostringstream oss;
          oss << what << " references variable `" << v->name_hint
              << "` which is not defined in the lexical scope of this "
                 "SIMD_VF region";
          Fail("expression", oss.str());
        }
      }
    });
  }

  // Walk the converting region in statement order, mirroring the wrapper
  // kinds DiscoverUnits descends (For/SeqStmt/AttrStmt/SBlock/
  // SBlockRealize). Definitions before a unit (flat binds, serial loop
  // vars, allocations) are registered as encountered so each unit is
  // verified against the environment at its position. Parallel Fors are
  // the discovered units themselves; IfThenElse/While cannot contain
  // units (the scan rejected that), and unit bodies belong to the
  // UnitVisitor.
  // Predicate admission for one converting region.
  //
  // A predicate has no lowering: the vectorizer writes whole chunks, so
  // honouring one would need a store mask or a destination read-back, and
  // dropping it would turn a conditional access into an unconditional one. Only
  // a predicate that provably holds everywhere is therefore admitted, and
  // Legalize's region-wide normaliser then clears the field before folding.
  //
  // The walk carries the lexical facts the predicate may rest on -- the
  // enclosing loop ranges and the scalar `Bind` values -- because a predicate
  // like `j < 8` over an outer serial index is provable only with them. Both
  // passes track the same facts, which is what keeps their verdicts aligned.
  void AdmitPredicates(const Stmt &body) {
    class Admission : public StmtExprVisitor {
    public:
      Admission(VerifyParallelToPTOImpl *parent, std::set<const VarNode *> defs,
                const Map<Var, PrimExpr> &outer_env,
                const std::vector<std::pair<Var, Range>> &outer_ranges)
          : parent_(parent), defs_(std::move(defs)), env_(outer_env) {
        for (const auto &entry : outer_ranges) {
          analyzer_.Bind(entry.first, entry.second, /*allow_override=*/true);
        }
      }

      // Every scope owner saves and restores BOTH halves of the fact context.
      // Restoring only `defs_` would leave a branch-local Bind value visible
      // after its scope ends, so a later predicate could be proven from a value
      // that is not in scope.
      void EnterScope(std::set<const VarNode *> *saved_defs,
                      Map<Var, PrimExpr> *saved_env) {
        *saved_defs = defs_;
        *saved_env = env_;
      }
      void LeaveScope(std::set<const VarNode *> *saved_defs,
                      Map<Var, PrimExpr> *saved_env) {
        defs_ = std::move(*saved_defs);
        env_ = std::move(*saved_env);
      }

      void VisitStmt_(const ForNode *op) final {
        analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent),
                       /*allow_override=*/true);
        std::set<const VarNode *> saved_defs;
        Map<Var, PrimExpr> saved_env;
        EnterScope(&saved_defs, &saved_env);
        defs_.insert(op->loop_var.get());
        StmtExprVisitor::VisitStmt_(op);
        LeaveScope(&saved_defs, &saved_env);
      }
      void VisitStmt_(const IfThenElseNode *op) final {
        VisitExpr(op->condition);
        {
          std::set<const VarNode *> saved_defs;
          Map<Var, PrimExpr> saved_env;
          EnterScope(&saved_defs, &saved_env);
          VisitStmt(op->then_case);
          LeaveScope(&saved_defs, &saved_env);
        }
        if (op->else_case.defined()) {
          std::set<const VarNode *> saved_defs;
          Map<Var, PrimExpr> saved_env;
          EnterScope(&saved_defs, &saved_env);
          VisitStmt(op->else_case.value());
          LeaveScope(&saved_defs, &saved_env);
        }
      }
      void VisitStmt_(const WhileNode *op) final {
        VisitExpr(op->condition);
        std::set<const VarNode *> saved_defs;
        Map<Var, PrimExpr> saved_env;
        EnterScope(&saved_defs, &saved_env);
        StmtExprVisitor::VisitStmt_(op);
        LeaveScope(&saved_defs, &saved_env);
      }
      void VisitStmt_(const AttrStmtNode *op) final {
        std::set<const VarNode *> saved_defs;
        Map<Var, PrimExpr> saved_env;
        EnterScope(&saved_defs, &saved_env);
        StmtExprVisitor::VisitStmt_(op);
        LeaveScope(&saved_defs, &saved_env);
      }
      void VisitStmt_(const SBlockNode *op) final {
        VisitScopedBlock(op->iter_vars, op->alloc_buffers, op->body);
      }
      void VisitStmt_(const SBlockRealizeNode *op) final {
        VisitScopedBlock(op->block->iter_vars, op->block->alloc_buffers,
                         op->block->body);
      }
      void VisitStmt_(const BindNode *op) final {
        StmtExprVisitor::VisitStmt_(op);
        defs_.insert(op->var.get());
        // A Bind is an SSA snapshot. Only a value that cannot observe state may
        // be re-expanded in a predicate; `old = a[i]` must stay opaque, or
        // `old == a[i]` after a store would be proven from a re-read.
        if (!op->value.dtype().is_handle() &&
            pto_analysis::IsSafelyReevaluableValue(op->value)) {
          env_.Set(op->var, op->value);
        }
      }
      void VisitStmt_(const AllocBufferNode *op) final {
        defs_.insert(op->buffer->data.get());
        StmtExprVisitor::VisitStmt_(op);
      }
      void VisitStmt_(const DeclBufferNode *op) final {
        defs_.insert(op->buffer->data.get());
        StmtExprVisitor::VisitStmt_(op);
      }
      // The predicate field of a load/store is not on the base visitor's
      // descent path, so both access kinds admit it explicitly. This is the one
      // place a predicate is judged: references first (against the lexical
      // environment), then whether it provably holds. Doing both here keeps a
      // region-level load from slipping through the reference check that the
      // store path already had.
      void VisitStmt_(const BufferStoreNode *op) final {
        Admit(op->predicate, op->buffer->name);
        StmtExprVisitor::VisitStmt_(op);
      }
      void VisitExpr_(const BufferLoadNode *op) final {
        Admit(op->predicate, op->buffer->name);
        StmtExprVisitor::VisitExpr_(op);
      }

    private:
      void VisitScopedBlock(const Array<IterVar> &iter_vars,
                            const Array<Buffer> &alloc_buffers,
                            const Stmt &body) {
        std::set<const VarNode *> saved_defs;
        Map<Var, PrimExpr> saved_env;
        EnterScope(&saved_defs, &saved_env);
        for (const IterVar &iv : iter_vars) {
          defs_.insert(iv->var.get());
        }
        for (const Buffer &buffer : alloc_buffers) {
          defs_.insert(buffer->data.get());
        }
        StmtExprVisitor::VisitStmt(body);
        LeaveScope(&saved_defs, &saved_env);
      }

      void Admit(const Optional<PrimExpr> &predicate, const String &name) {
        if (!predicate.defined()) {
          return;
        }
        // Runs before any simplification: folding `True | ghost` to True would
        // hide the undefined reference.
        PostOrderVisit(predicate.value(), [&](const ObjectRef &node) {
          if (const auto *var = node.as<VarNode>()) {
            if (defs_.count(var) == 0) {
              std::ostringstream oss;
              oss << "buffer `" << name
                  << "` carries an access predicate that references variable `"
                  << var->name_hint
                  << "`, which is not defined in the lexical scope of this "
                     "SIMD_VF region";
              parent_->Fail("expression", oss.str());
            }
          }
        });
        // The predicate is an expression position like any other, and it is
        // dropped after admission. An unadmitted node that the analyzer folds
        // away (`True | call_extern(...)`, `True | (p % 2 == 0)`) would
        // therefore never be seen by a later pass -- an external call could
        // even carry a side effect that the removal silently discards. Run the
        // same whitelist the compute-value positions run, on the original tree
        // and before the truth proof.
        if (auto reason =
                pto_analysis::FindUnsupportedExprNode(predicate.value())) {
          parent_->Fail("expression", "access predicate on buffer `" +
                                          std::string(name) +
                                          "`: " + reason.value());
        }
        if (!pto_analysis::IsProvablyTruePredicate(predicate.value(),
                                                   &analyzer_, env_)) {
          std::ostringstream oss;
          oss << "buffer `" << name
              << "` carries an access predicate that is not provably true; a "
                 "predicated access has no lowering, so only a "
                 "compile-time-true predicate is accepted (and then ignored)";
          parent_->Fail("memory access", oss.str());
        }
      }

      VerifyParallelToPTOImpl *parent_;
      arith::Analyzer analyzer_;
      Map<Var, PrimExpr> env_;
      std::set<const VarNode *> defs_;
    };
    // Seed with the definition snapshot at the region's lexical position
    // (parameters, enclosing block variables, allocations before the region)
    // and the facts that stay valid across it: enclosing loop ranges and the
    // scalar Binds defined outside this region.
    Admission admission(this, defs_, outer_binds_, outer_ranges_);
    admission(body);
  }

  void WalkRegion(const Stmt &stmt, const std::vector<For> &units,
                  int64_t lanes, size_t *next_unit) {
    if (const auto *for_node = stmt.as<ForNode>()) {
      if (for_node->kind == ForKind::kParallel &&
          !pto_analysis::UsesExplicitSIMD(for_node->body)) {
        ICHECK(*next_unit < units.size() && units[*next_unit].get() == for_node)
            << "[VerifyParallelToPTO] internal error: region walk and "
               "unit scan disagree on unit order";
        VerifyUnit(units[*next_unit], lanes, *next_unit);
        ++*next_unit;
        return;
      }
      // The serial loop bounds are expression positions too.
      // The step is one as well (the base statement visitors do
      // not descend into For::step).
      SetRegionContext();
      CheckRegionExprDefined(for_node->min, "serial loop min");
      CheckRegionExprDefined(for_node->extent, "serial loop extent");
      if (for_node->step.defined()) {
        CheckRegionExprDefined(for_node->step.value(), "serial loop step");
      }
      std::set<const VarNode *> saved = defs_;
      defs_.insert(for_node->loop_var.get());
      WalkRegion(for_node->body, units, lanes, next_unit);
      defs_ = std::move(saved);
      return;
    }
    if (const auto *assertion = stmt.as<AssertStmtNode>()) {
      // A region-level assertion is an expression position: its condition
      // must reference defined values. The node has no body (only the
      // condition, error kind and message parts), so nothing else is
      // walked; an assertion *inside* a unit stays rejected by the unit
      // statement visitor.
      SetRegionContext();
      CheckRegionExprDefined(assertion->condition,
                             "region-level assert condition");
      return;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        WalkRegion(s, units, lanes, next_unit);
      }
      return;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      // Region-level metadata (node/value) is scanned for
      // undefined references before descending.
      SetRegionContext();
      if (auto ne = attr->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          CheckRegionExprDefined(GetRef<PrimExpr>(pe),
                                 "region-level AttrStmt metadata");
        }
      }
      CheckRegionExprDefined(attr->value, "region-level AttrStmt metadata");
      // The body is a nested statement scope.
      {
        std::set<const VarNode *> saved = defs_;
        WalkRegion(attr->body, units, lanes, next_unit);
        defs_ = std::move(saved);
      }
      return;
    }
    if (const auto *block = stmt.as<SBlockNode>()) {
      std::set<const VarNode *> saved = defs_;
      for (const auto &iv : block->iter_vars) {
        defs_.insert(iv->var.get());
      }
      for (const auto &buf : block->alloc_buffers) {
        defs_.insert(buf->data.get());
      }
      WalkRegion(block->body, units, lanes, next_unit);
      defs_ = std::move(saved);
      return;
    }
    if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
      std::set<const VarNode *> saved = defs_;
      for (const auto &iv : realize->block->iter_vars) {
        defs_.insert(iv->var.get());
      }
      for (const auto &buf : realize->block->alloc_buffers) {
        defs_.insert(buf->data.get());
      }
      WalkRegion(realize->block->body, units, lanes, next_unit);
      defs_ = std::move(saved);
      return;
    }
    if (const auto *bind = stmt.as<BindNode>()) {
      // The region-level Bind RHS is checked against the
      // pre-registration environment (lexical order), then the
      // definition itself is registered.
      SetRegionContext();
      CheckRegionExprDefined(bind->value, "region-level bind");
      defs_.insert(bind->var.get());
      return;
    }
    if (const auto *alloc = stmt.as<AllocBufferNode>()) {
      defs_.insert(alloc->buffer->data.get());
      return;
    }
    if (const auto *decl = stmt.as<DeclBufferNode>()) {
      defs_.insert(decl->buffer->data.get());
      return;
    }
    // Statements that contribute no definitions are still
    // expression positions. A store, an Evaluate, a branch or a while
    // outside the Parallel units previously ended the walk silently, so
    // an undefined variable in any of their expression fields passed
    // Verify and stayed in the output.
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      SetRegionContext();
      CheckRegionExprDefined(store->value, "region-level store value");
      for (const auto &idx : store->indices) {
        CheckRegionExprDefined(idx, "region-level store index");
      }
      // The predicate (references and provable truth) is admitted by
      // AdmitPredicates for every load and store alike.
      return;
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      SetRegionContext();
      CheckRegionExprDefined(eval->value, "region-level evaluate value");
      return;
    }
    if (const auto *ite = stmt.as<IfThenElseNode>()) {
      // A Parallel under control flow is rejected by ScanPtoRegion, so
      // the branches carry plain statements only; branch-local
      // definitions must not leak to later siblings.
      SetRegionContext();
      CheckRegionExprDefined(ite->condition, "region-level branch condition");
      {
        std::set<const VarNode *> saved = defs_;
        WalkRegion(ite->then_case, units, lanes, next_unit);
        defs_ = std::move(saved);
      }
      if (ite->else_case.defined()) {
        std::set<const VarNode *> saved = defs_;
        WalkRegion(ite->else_case.value(), units, lanes, next_unit);
        defs_ = std::move(saved);
      }
      return;
    }
    if (const auto *while_node = stmt.as<WhileNode>()) {
      SetRegionContext();
      CheckRegionExprDefined(while_node->condition,
                             "region-level while condition");
      std::set<const VarNode *> saved = defs_;
      WalkRegion(while_node->body, units, lanes, next_unit);
      defs_ = std::move(saved);
      return;
    }
  }

  arith::Analyzer analyzer_;
  // The lexical definition environment. Supersedes the earlier
  // function-wide external-def union, which missed scoped definitions
  // (blockIdx, outer/VF-internal binds) and leaked sibling-scope loop
  // vars into every unit.
  std::set<const VarNode *> defs_;
  // Target Vars of the handle-valued Binds seen anywhere in the function
  // (collected function-wide by CollectHandleBindTargets). A buffer declared
  // over one of these pointers is a view the lane analysis cannot follow, so
  // every access to it is rejected (see CheckScope).
  std::set<const VarNode *> handle_bind_targets_;
  // Facts that stay valid around a converting region and must reach its
  // predicate admission: scalar Binds defined outside it and the enclosing loop
  // ranges. Restricted to safely re-evaluable Binds, like the region walk.
  Map<Var, PrimExpr> outer_binds_;
  std::vector<std::pair<Var, Range>> outer_ranges_;
  // Data Vars of AllocBuffers annotated tirx.volatile (collected
  // function-wide by CollectVolatileAllocs; identity-matched at every
  // access inside a converting region).
  std::set<const VarNode *> volatile_allocs_;
  // Position of the converting region/unit being checked, for
  // diagnostics.
  int region_seq_ = 0;
  int current_region_ = -1;
  std::string unit_context_;
  // Lane count of the converting region under check (0 when unknown); part of
  // the region/unit diagnostic context.
  int64_t current_lanes_ = 0;
};

} // namespace

namespace transform {

tvm::transform::Pass VerifyParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    VerifyParallelToPTOImpl verifier;
    verifier.Verify(f);
    return f;
  };
  return tvm::tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                                  "tl.VerifyParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VerifyParallelToPTO",
                        VerifyParallelToPTO);
}

} // namespace transform

} // namespace tl

} // namespace tvm
