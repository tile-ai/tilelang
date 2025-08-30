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
#include "../op/elem.h"
#include "../op/finalize_reducer.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "layout_reducer.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using arith::IRMutatorWithAnalyzer;

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
    if (layout_map.size())
      p_result->annotations.Set(attr::kLayoutMap, layout_map);
    new_layout_map_.clear();
    return result;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // only annotate the outermost loop
    bool should_annotate = false;
    if (inside_reducer_range_.size() > 0 && !already_annotated_) {
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
        auto dtype = thread_var_->var.dtype();
        int thread_min = const_int_bound->min_value;
        int thread_extent =
            const_int_bound->max_value - const_int_bound->min_value + 1;

        auto opt_buffer = var_to_buffer_.Get(reducer_var);
        ICHECK(opt_buffer);
        auto buffer = opt_buffer.value();
        Fragment f;
        if (info->rep == ReducerRepType::ALL) {
          f = Fragment(buffer->shape, {}, ReplicationPlaceholder(),
                       thread_extent, std::nullopt);
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

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    //! TODO: check store viable according to info->op
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op_) final {
    auto op_ref = IRMutatorWithAnalyzer::VisitExpr_(op_).as<Call>().value();
    auto op = op_ref.CopyOnWrite();
    if (op->op.same_as(Fill::Get())) {
      ICHECK(op->args.size() > 0);
      if (auto arg0_call = op->args[0].as<Call>();
          arg0_call &&
          arg0_call.value()->op.same_as(builtin::tvm_access_ptr())) {
        ICHECK(arg0_call.value()->args.size() > 1);
        if (auto var = arg0_call.value()->args[1].as<Var>();
            var && reducer_info_map_.count(var.value())) {
          ICHECK(inside_reducer_range_.count(var.value()) == 0)
              << "T.fill on reducer must be enclosed with a T.finalize_reducer "
                 "before next.";
          inside_reducer_range_.Set(var.value(),
                                    reducer_info_map_.Get(var.value()).value());
        }
      }
    } else if (op->op.same_as(FinalizeReducerOp::Get())) {
      ICHECK(op->args.size() == 1);
      auto var = GetVarFromAccessPtr(op->args[0]);
      ICHECK(inside_reducer_range_.count(var) == 1)
          << "T.finalize_reducer must have a pairing T.fill ahead of it, "
             "enclosing a reduction range.";
      op->args.push_back((int)inside_reducer_range_.Get(var).value()->op);
      inside_reducer_range_.erase(var);
    }
    return op_ref;
  }

  ReducerLayoutAnnotator(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  IterVar thread_var_;
  Map<Var, ReducerInfo> reducer_info_map_;
  Map<Var, ReducerInfo> inside_reducer_range_;
  bool already_annotated_ = false;
  Map<Var, Buffer> var_to_buffer_;
  Map<Var, Layout> new_layout_map_;

public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    ReducerLayoutAnnotator substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }
};

tvm::transform::Pass LayoutReducer() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return ReducerLayoutAnnotator::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LayoutReducer", {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LayoutReducer", LayoutReducer);
});

} // namespace tl
} // namespace tvm
