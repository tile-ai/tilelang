/*!
 * \file layout_reducer.cc
 *
 * Compute layout for local.reducer buffers and lower them to local.fragment.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../layout/layout.h"
#include "../op/fill.h"
#include "../op/finalize_reducer.h"
#include "../op/region.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "layout_reducer.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using arith::IRMutatorWithAnalyzer;

/**
 * @brief Construct a ReducerInfoNode from textual op and replication
 * descriptors.
 *
 * Maps op_str to a ReducerOpType ("sum" → SUM, "max" → MAX, "min" → MIN) and
 * rep_str to a ReducerRepType ("all" → ALL, "none" → NONE).
 *
 * @param op_str String identifying the reducer operation.
 * @param rep_str String identifying the replication behavior.
 * @throws RuntimeError if op_str or rep_str is not one of the supported values
 * (triggers ICHECK).
 */
ReducerInfoNode::ReducerInfoNode(const String &op_str, const String &rep_str) {
  if (op_str == "sum")
    op = ReducerOpType::SUM;
  else if (op_str == "max")
    op = ReducerOpType::MAX;
  else if (op_str == "min")
    op = ReducerOpType::MIN;
  else
    ICHECK(false) << "Unrecognized reducer_info op: " << op_str;

  if (rep_str == "all")
    rep = ReducerRepType::ALL;
  else if (rep_str == "none")
    rep = ReducerRepType::NONE;
  else
    ICHECK(false) << "Unrecognized reducer_info rep: " << rep_str;
}

class ReducerLayoutAnnotator : public IRMutatorWithAnalyzer {
public:
private:
  /**
   * @brief Visit an attribute statement and capture the IterVar for
   * threadIdx.x.
   *
   * If the attribute key is `tir::attr::thread_extent` and the node is an
   * `IterVar` whose `thread_tag` equals `"threadIdx.x"`, this sets the
   * mutator's `thread_var_` to that IterVar (after asserting the iterator's
   * extent is an `IntImm`). The previous `thread_var_` is preserved and
   * restored after delegating to the base visitor. Delegates all traversal work
   * to `IRMutatorWithAnalyzer::VisitStmt_`.
   *
   * Side effects:
   * - Temporarily updates the member `thread_var_` during traversal of the
   * child statement so subsequent visitors can read the thread index IterVar.
   *
   * @return The possibly mutated statement returned by the base visitor.
   */
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    auto prev_thread_var = thread_var_;
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    auto result = IRMutatorWithAnalyzer::VisitStmt_(op);
    thread_var_ = prev_thread_var;
    return result;
  }

  /**
   * @brief Visits a TIR Block node to collect reducer metadata and apply
   * discovered buffer layouts.
   *
   * This method:
   * - Extracts reducer information from the block's `attr::kReducerInfo`
   * annotation and populates the internal reducer_info_map_.
   * - Registers allocated buffers by mapping each buffer's data Var to its
   * Buffer in var_to_buffer_.
   * - Recursively visits and rewrites the block body via the base mutator.
   * - Merges any layouts accumulated in new_layout_map_ into the block's
   * `attr::kLayoutMap` annotation (creating or extending the annotation), then
   * clears new_layout_map_ for subsequent blocks.
   *
   * Side effects:
   * - Updates reducer_info_map_, var_to_buffer_, and may set the block-level
   * `kLayoutMap` annotation.
   * - Clears new_layout_map_ after merging.
   *
   * @param op The Block node being visited.
   * @return Stmt The potentially modified Block statement (as a Stmt).
   */
  Stmt VisitStmt_(const BlockNode *op) final {
    // Record annotations
    if (op->annotations.count(attr::kReducerInfo)) {
      auto map = op->annotations.Get(attr::kReducerInfo)
                     ->as<Map<Var, Map<String, String>>>();
      ICHECK(map) << "reducer_replication map is not defined";
      for (auto &&[var, rep] : map.value()) {
        reducer_info_map_.Set(
            var, ReducerInfo{rep.Get("op").value(), rep.Get("rep").value()});
      }
    }
    for (auto &&buffer : op->alloc_buffers) {
      var_to_buffer_.Set(buffer->data, buffer);
    }
    auto result = IRMutatorWithAnalyzer::VisitStmt_(op).as<Block>().value();
    // After iterating over the body, set all layout_map to block
    auto p_result = result.CopyOnWrite();
    auto layout_map = p_result->annotations.Get(attr::kLayoutMap)
                          ->as<Map<Var, Layout>>()
                          .value_or(Map<Var, Layout>());
    for (auto &&[k, v] : new_layout_map_)
      layout_map.Set(k, v);
    if (!layout_map.empty())
      p_result->annotations.Set(attr::kLayoutMap, layout_map);
    new_layout_map_.clear();
    return result;
  }

  /**
   * @brief Visit and possibly annotate a For node for reducer layout lowering.
   *
   * Visits a For node via the base mutator and, if the traversal is currently
   * inside a reduction region (tracked by inside_reducer_range_) and this is
   * the outermost loop of that region, annotates the loop with reducer
   * information and derives per-buffer layout fragments for each reducer
   * buffer.
   *
   * When annotating:
   * - Sets the block-level `attr::kReducerInfo` annotation to the current
   *   inside_reducer_range_ map on the loop.
   * - For each reducer buffer, reads the bound of `thread_var_` (requires the
   *   analyzer to have a const-int bound for it) and creates a Fragment:
   *   - If the reducer's replication type is ALL, creates a replication
   * fragment across the thread extent.
   *   - If the replication type is NONE, builds a flattened index expression
   *     across buffer indices, reduces it modulo the thread extent, adds the
   *     thread minimum offset, and uses that as the fragment index.
   * - Records the constructed Fragments into new_layout_map_ keyed by the
   *   buffer's data Var.
   *
   * Side effects:
   * - May set `attr::kReducerInfo` on the For node's annotations.
   * - Updates `new_layout_map_`.
   * - Reads and relies on `thread_var_`, `analyzer_->const_int_bound`, and
   *   `var_to_buffer_`.
   *
   * Preconditions and checks:
   * - `thread_var_` must be defined and have a constant-int bound when
   * annotating.
   * - Each reducer Var in inside_reducer_range_ must map to an allocated Buffer
   * in var_to_buffer_ (ICHECK enforced).
   *
   * @param op The original For node being visited.
   * @return The (possibly) transformed For statement.
   */
  Stmt VisitStmt_(const ForNode *op) final {
    // only annotate the outermost loop
    bool should_annotate = false;
    if (!inside_reducer_range_.empty() && !already_annotated_ &&
        op->kind == ForKind::kParallel) {
      should_annotate = true;
      already_annotated_ = true;
    }

    auto opt_result = IRMutatorWithAnalyzer::VisitStmt_(op).as<For>();
    ICHECK(opt_result);
    auto result = opt_result.value();

    if (should_annotate) {
      // we are leaving the current loop nest. later ones may annotate again
      already_annotated_ = false;

      auto p_result = result.CopyOnWrite();
      p_result->annotations.Set(attr::kReducerInfo, inside_reducer_range_);

      // Iterate over local.reducer.* buffers, append to reducer_op_map_, set
      // layout by adding layout_map annotations, and convert scope to
      // local.fragment
      for (auto &&[reducer_var, info] : inside_reducer_range_) {
        // analyze thread index bound, need to be inside WS section
        ICHECK(thread_var_.defined());
        ICHECK(analyzer_->const_int_bound.IsBound(thread_var_->var));
        auto const_int_bound = analyzer_->const_int_bound(thread_var_);
        int thread_min = const_int_bound->min_value;
        int thread_extent =
            const_int_bound->max_value - const_int_bound->min_value + 1;

        auto opt_buffer = var_to_buffer_.Get(reducer_var);
        ICHECK(opt_buffer);
        const auto &buffer = opt_buffer.value();
        Fragment f;
        if (info->rep == ReducerRepType::ALL) {
          f = Fragment::FullyReplicated(buffer->shape, thread_extent);
        } else if (info->rep == ReducerRepType::NONE) {
          PrimExpr flatten_idx = InputPlaceholder(0);
          for (int i = 1; i < buffer->shape.size(); ++i)
            flatten_idx = flatten_idx * buffer->shape[i] + InputPlaceholder(i);
          f = Fragment(buffer->shape, {},
                       indexmod(flatten_idx, thread_extent) + thread_min, 1,
                       std::nullopt);
        }
        new_layout_map_.Set(buffer->data, f);
      }
    }
    return result;
  }

  /**
   * @brief Handle BufferStore statements during IR mutation.
   *
   * This override is the visit hook for BufferStoreNode. Currently it delegates
   * to the base IRMutatorWithAnalyzer implementation. Intended as the place to
   * perform reducer-specific viability checks for stores (e.g., validating
   * operations against reducer metadata); such checks are TODO and are not yet
   * implemented.
   *
   * @return Stmt The (possibly transformed) statement returned by the base
   * mutator.
   */
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    //! TODO: check store viable according to info->op
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  /**
   * @brief Processes Call expressions to track reducer ranges and finalize
   * reducer operations.
   *
   * Visits call nodes, detects T.fill calls that target reducer buffers and
   * records their reducer metadata in inside_reducer_range_ until the matching
   * T.finalize_reducer is seen. When a FinalizeReducerOp call is encountered,
   * this method appends the reducer operation enum value to the call arguments
   * and removes the corresponding entry from inside_reducer_range_.
   *
   * Side effects:
   * - Inserts and removes entries in inside_reducer_range_.
   * - Mutates the FinalizeReducerOp call by pushing the reducer op enum as an
   * extra argument.
   *
   * Failure modes:
   * - ICHECK fails if a T.fill targets a reducer already recorded in
   * inside_reducer_range_ (i.e., a prior T.fill without an intervening
   * T.finalize_reducer).
   * - ICHECK fails if T.finalize_reducer has no matching T.fill (no entry in
   * inside_reducer_range_).
   *
   * @param op_ The CallNode being visited.
   * @return PrimExpr The (possibly modified) call expression.
   */
  PrimExpr VisitExpr_(const CallNode *op_) final {
    auto op_ref = IRMutatorWithAnalyzer::VisitExpr_(op_).as<Call>().value();
    auto op = op_ref.CopyOnWrite();
    if (op->op.same_as(Fill::Get())) {
      ICHECK(!op->args.empty());
      if (auto arg0_call = op->args[0].as<Call>()) {
        // tl.region(...) — extract buffer var from its first arg
        if (arg0_call.value()->op.same_as(RegionOp::Get())) {
          ICHECK(!arg0_call.value()->args.empty());
          if (auto bl = arg0_call.value()->args[0].as<BufferLoadNode>()) {
            Var var = bl->buffer->data;
            if (reducer_info_map_.count(var)) {
              ICHECK(inside_reducer_range_.count(var) == 0)
                  << "T.fill on reducer must be enclosed with a "
                     "T.finalize_reducer before next.";
              inside_reducer_range_.Set(var,
                                        reducer_info_map_.Get(var).value());
            }
          }
        }
        // builtin.tvm_access_ptr(...) — existing path (legacy)
        if (arg0_call.value()->op.same_as(builtin::tvm_access_ptr())) {
          ICHECK(arg0_call.value()->args.size() > 1);
          if (auto var = arg0_call.value()->args[1].as<Var>();
              var && reducer_info_map_.count(var.value())) {
            ICHECK(inside_reducer_range_.count(var.value()) == 0)
                << "T.fill on reducer must be enclosed with a "
                   "T.finalize_reducer "
                   "before next.";
            inside_reducer_range_.Set(
                var.value(), reducer_info_map_.Get(var.value()).value());
          }
        }
      } else if (auto bl = op->args[0].as<BufferLoadNode>()) {
        Var var = bl->buffer->data;
        if (reducer_info_map_.count(var)) {
          ICHECK(inside_reducer_range_.count(var) == 0)
              << "T.fill on reducer must be enclosed with a T.finalize_reducer "
                 "before next.";
          inside_reducer_range_.Set(var, reducer_info_map_.Get(var).value());
        }
      }
    } else if (op->op.same_as(FinalizeReducerOp::Get())) {
      ICHECK(op->args.size() == 1);
      Var var;
      if (auto bl = op->args[0].as<BufferLoadNode>()) {
        var = bl->buffer->data;
      } else if (auto reg_call = op->args[0].as<Call>()) {
        if (reg_call.value()->op.same_as(RegionOp::Get())) {
          if (auto bl2 = reg_call.value()->args[0].as<BufferLoadNode>()) {
            var = bl2->buffer->data;
          } else {
            LOG(FATAL) << "tl.region expects BufferLoad as first arg";
          }
        } else {
          var = GetVarFromAccessPtr(op->args[0]);
        }
      } else {
        var = GetVarFromAccessPtr(op->args[0]);
      }
      ICHECK(inside_reducer_range_.count(var) == 1)
          << "T.finalize_reducer must have a pairing T.fill ahead of it, "
             "enclosing a reduction range.";
      op->args.push_back((int)inside_reducer_range_.Get(var).value()->op);
      inside_reducer_range_.erase(var);
    }
    return op_ref;
  }

  /**
   * @brief Construct a ReducerLayoutAnnotator with an arithmetic analyzer.
   *
   * Initializes the annotator's base IRMutatorWithAnalyzer with the provided
   * arith::Analyzer, which the mutator uses to query symbolic bounds and
   * simplify integer expressions during layout inference.
   *
   * @param analyzer Pointer to an arith::Analyzer used for symbolic analysis.
   */
  ReducerLayoutAnnotator(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  IterVar thread_var_;
  Map<Var, ReducerInfo> reducer_info_map_;
  Map<Var, ReducerInfo> inside_reducer_range_;
  bool already_annotated_ = false;
  Map<Var, Buffer> var_to_buffer_;
  Map<Var, Layout> new_layout_map_;

public:
  /**
   * @brief Apply reducer layout substitution to a PrimFunc.
   *
   * Runs the ReducerLayoutAnnotator over the function body to collect reducer
   * metadata, insert layout mappings for reducer buffers, and lower
   * local.reducer usage to local.fragment-compatible forms. Returns a new
   * PrimFunc whose body is the transformed IR.
   *
   * @param f The PrimFunc to transform; passed by value and returned with an
   * updated body.
   * @return PrimFunc The transformed PrimFunc with reducer layouts and related
   * rewrites applied.
   */
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    ReducerLayoutAnnotator substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }
};

/**
 * @brief Create a TVM transform pass that lowers local.reducer buffers to
 * local.fragment layouts.
 *
 * This pass runs ReducerLayoutAnnotator::Substitute on a PrimFunc to collect
 * reducer metadata, compute per-buffer layout fragments for reducer buffers,
 * and annotate blocks with the resulting layout map. It is exposed as a
 * PrimFunc-level pass named "tl.LayoutReducer".
 *
 * @return tvm::transform::Pass A prim-function pass that applies the
 * layout-reduction substitution.
 */
tvm::transform::Pass LayoutReducer() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return ReducerLayoutAnnotator::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LayoutReducer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LayoutReducer", LayoutReducer);
}

} // namespace tl
} // namespace tvm
