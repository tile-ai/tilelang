/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file tma_barrier_rewriter.cc
 * \brief Rewrite TMA barriers for cuda GPU (sm90+)
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "../op/builtin.h"
#include "./common/attr.h"
#include "./common/collector.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using arith::IRMutatorWithAnalyzer;
using arith::IRVisitorWithAnalyzer;

class TmaTraitsCollector : public StmtExprVisitor {
public:
  TmaTraitsCollector() { Initialize(); }

  void Initialize() {
    bulk_copy_bytes = 0;
    loop_extents = 1;
  }

  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  PrimExpr BulkCopyBytes() { return bulk_copy_bytes; }

private:
  void VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col())) {
      auto arg0 = call->args[0].as<Call>();
      if (call->op.same_as(tma_load()) && arg0 &&
          !arg0.value()->op.same_as(create_tma_descriptor())) {
        // 1D TMA load has tvm_access_ptr of shared tensor in its args[0]
        bulk_copy_bytes = call->args[3] * loop_extents;
      } else {
        Call access_ptr = Downcast<Call>(call->args[2]);
        ICHECK(access_ptr->op.same_as(builtin::tvm_access_ptr()));
        int type_bytes = access_ptr->args[0]->dtype.bytes();
        bulk_copy_bytes += access_ptr->args[3] * loop_extents * type_bytes;
      }
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  void VisitStmt_(const ForNode *op) final {
    PrimExpr old_loop_evtents = loop_extents;
    loop_extents *= op->extent;
    StmtExprVisitor::VisitStmt_(op);
    loop_extents = old_loop_evtents;
  }

  PrimExpr bulk_copy_bytes = 0;
  PrimExpr loop_extents = 1;
};

class TmaExpectTxRewriter : public IRMutatorWithAnalyzer {
public:
  static PrimFunc Rewrite(PrimFunc f, arith::Analyzer *analyzer) {
    TmaExpectTxRewriter rewriter(analyzer);
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  bool inside_tma_block_{false};
  bool visited_tma_load_{false};
  IterVar thread_var_ = IterVar(Range::FromMinExtent(0, 1), Var("v_thread"),
                                IterVarType::kDataPar);

  PrimExpr makeGetBarrier(PrimExpr barrier_id) {
    return Call(DataType::Handle(), get_mbarrier(), {std::move(barrier_id)});
  }

  Stmt makeExpectTX(PrimExpr barrier_id, PrimExpr bytes) {
    auto call = Call(DataType::Handle(), mbarrier_expect_tx(),
                     {makeGetBarrier(std::move(barrier_id)), std::move(bytes)});
    return Evaluate(call);
  }

  TmaExpectTxRewriter(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {

    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) {
    // Check if this is the TMA block
    bool flag = false;
    if (op->condition.as<CallNode>()) {
      flag = op->condition.as<CallNode>()->op.same_as(tl_shuffle_elect());
    }
    if (op->condition.as<EQNode>() || flag) {
      Stmt ret = IRMutatorWithAnalyzer::VisitStmt_(op);

      if (visited_tma_load_) {
        auto then_case = op->then_case;
        TmaTraitsCollector collector;
        collector.Collect(then_case);

        Array<Stmt> stmts;
        if (!is_zero(collector.BulkCopyBytes())) {
          auto expect_tx = makeExpectTX(0, collector.BulkCopyBytes());
          stmts.push_back(expect_tx);
        }
        stmts.push_back(then_case);
        if (stmts.size() == 1) {
          return IfThenElse(op->condition, stmts[0], op->else_case);
        } else {
          auto seq_stmt = SeqStmt(stmts);
          return IfThenElse(op->condition, seq_stmt, op->else_case);
        }
      }
      visited_tma_load_ = false;
      return ret;
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) {
    if (op->op.same_as(tma_load()) || op->op.same_as(tma_load_im2col())) {
      auto arg0 = op->args[0].as<Call>();
      bool is_1d_tma_load =
          arg0 && !arg0.value()->op.same_as(create_tma_descriptor()) &&
          op->op.same_as(tma_load());
      visited_tma_load_ = true;
      Array<PrimExpr> new_args = op->args;
      new_args.Set(is_1d_tma_load ? 2 : 1,
                   Call(DataType::Handle(), get_mbarrier(),
                        {IntImm(DataType::Int(32), 0)}));
      return Call(op->dtype, op->op, new_args);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }
};

class TmaBarrierCollector : public IRVisitorWithAnalyzer {
public:
  TmaBarrierCollector(Map<Var, Buffer> buffer_data_to_buffer)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)) {}

  Map<ObjectRef, PrimExpr> tma_op_to_barrier_id() {
    return tma_op_to_barrier_id_;
  }
  Map<PrimExpr, IntImm> barrier_id_to_range() { return barrier_id_to_range_; }

private:
  void UpdateBarrierRange(const PrimExpr &barrier_id, const IntImm &extent) {
    if (barrier_id_to_range_.count(barrier_id)) {
      auto old_extent = barrier_id_to_range_[barrier_id];
      ICHECK_EQ(old_extent->value, extent->value)
          << "barrier_id: " << barrier_id << " has different extent";
      barrier_id_to_range_.Set(barrier_id, extent);
    } else {
      barrier_id_to_range_.Set(barrier_id, extent);
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col())) {
        pending_tma_ops_.push_back(GetRef<Call>(call));
      } else if (call->op.same_as(mbarrier_expect_tx())) {
        pending_tma_ops_.push_back(GetRef<Call>(call));
      } else if (call->op.same_as(builtin::ptx_arrive_barrier())) {
        PrimExpr barrier_id = call->args[0];
        for (const auto &tma_call : pending_tma_ops_) {
          tma_op_to_barrier_id_.Set(tma_call, barrier_id);
        }
        auto const_int_bound = analyzer_.const_int_bound(thread_var_);
        auto extent =
            const_int_bound->max_value - const_int_bound->min_value + 1;
        UpdateBarrierRange(barrier_id, IntImm(DataType::Int(32), extent));
        pending_tma_ops_.clear();
      } else if (call->op.same_as(builtin::ptx_wait_barrier())) {
        PrimExpr barrier_id = call->args[0];
        auto const_int_bound = analyzer_.const_int_bound(thread_var_);
        auto extent =
            const_int_bound->max_value - const_int_bound->min_value + 1;
        UpdateBarrierRange(barrier_id, IntImm(DataType::Int(32), extent));
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv;
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  IterVar thread_var_;
  std::vector<Call> pending_tma_ops_;
  Map<ObjectRef, PrimExpr> tma_op_to_barrier_id_;
  Map<PrimExpr, IntImm> barrier_id_to_range_;
  Map<Var, Buffer> buffer_data_to_buffer_;
};

class TmaSequenceCollector : public IRVisitorWithAnalyzer {
public:
  TmaSequenceCollector(Map<ObjectRef, PrimExpr> tma_op_to_barrier_id)
      : tma_op_to_barrier_id_(std::move(tma_op_to_barrier_id)) {}

  std::vector<bool> GetSequence() {
    std::vector<bool> clear_zero_list(expect_tx_count_, false);
    int zero_idx = -1;
    int zero_count = 0;

    for (auto v : sequence) {
      if (v == 0) {
        zero_count += 1;
        zero_idx += 1;
      } else {
        if (zero_count == 1) {
          clear_zero_list[zero_idx] = expect_[zero_idx] && !has_simt_copy_;
          if (clear_zero_list[zero_idx] == false) {
            int begin = int_sets_[zero_idx].min().as<IntImmNode>()->value;
            int end = int_sets_[zero_idx].max().as<IntImmNode>()->value;
            for (int i = begin; i <= end; ++i) {
              restore_barrier_ids_.push_back(i);
            }
          }
        } else {
          for (int i{zero_idx}; i > zero_idx - zero_count; --i) {
            int begin = int_sets_[i].min().as<IntImmNode>()->value;
            int end = int_sets_[i].max().as<IntImmNode>()->value;
            for (int i = begin; i <= end; ++i) {
              restore_barrier_ids_.push_back(i);
            }
          }
        }
        zero_count = 0;
      }
    }

    return clear_zero_list;
  }

  std::vector<int> GetRestoreBarrierIds() { return restore_barrier_ids_; }

  void VisitStmt_(const ForNode *op) final {
    var_int_set_.Set(op->loop_var,
                     arith::IntSet::FromMinExtent(op->min, op->extent));
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(mbarrier_expect_tx())) {
      PrimExpr e =
          tma_op_to_barrier_id_[GetRef<Call>(op)].as<CallNode>()->args[0];
      auto int_set = arith::EvalSet(e, var_int_set_);
      expect_.push_back(if_depth_ == 1);
      sequence.push_back(0);
      int_sets_.push_back(int_set);
      expect_tx_count_ += 1;
    } else if (op->op.same_as(builtin::ptx_arrive_barrier())) {
      sequence.push_back(1);
    } else if (op->op.same_as(builtin::ptx_cp_async_barrier())) {
      has_simt_copy_ = true;
    }
    IRVisitorWithAnalyzer::VisitExpr_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if_depth_ += 1;

    IRVisitorWithAnalyzer::VisitStmt(op->then_case);

    if (op->else_case) {
      IRVisitorWithAnalyzer::VisitStmt(op->else_case.value());
    }
    if_depth_ -= 1;
  }

  std::vector<int> sequence;
  int expect_tx_count_{0};
  std::vector<bool> expect_;
  bool has_simt_copy_{false};
  std::vector<int> restore_barrier_ids_;
  int if_depth_{0};
  Map<ObjectRef, PrimExpr> tma_op_to_barrier_id_;
  arith::Analyzer *analyzer_{};
  Map<Var, arith::IntSet> var_int_set_;
  std::vector<arith::IntSet> int_sets_;
};

class BarrierCreationRewriter : public StmtExprMutator {
public:
  BarrierCreationRewriter(std::vector<int> restore_barrier_ids,
                          PrimExpr producer_thread_extent)
      : restore_barrier_ids_(std::move(restore_barrier_ids)),
        producer_thread_extent_(std::move(producer_thread_extent)) {}

  PrimExpr VisitExpr_(const CallNode *op) {
    if (op->op.same_as(create_list_of_mbarrier())) {
      std::vector<bool> tmp_(op->args.size(), false);
      Array<PrimExpr> new_args;
      for (auto &id : restore_barrier_ids_) {
        tmp_[id] = true;
      }

      for (size_t i{0}; i < op->args.size(); ++i) {
        if (tmp_[i]) {
          new_args.push_back(producer_thread_extent_);
        } else {
          new_args.push_back(op->args[i]);
        }
      }
      return Call(op->dtype, op->op, new_args);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }
  std::vector<int> restore_barrier_ids_;
  PrimExpr producer_thread_extent_;
};

// we trust mbarrier_wait_parity to be correct
class TmaBarrierRewriter : public IRMutatorWithAnalyzer {
public:
  TmaBarrierRewriter(arith::Analyzer *analyzer,
                     Map<ObjectRef, PrimExpr> tma_op_to_barrier_id,
                     Map<PrimExpr, IntImm> barrier_id_to_range,
                     bool has_create_list_of_mbarrier)
      : IRMutatorWithAnalyzer(analyzer),
        tma_op_to_barrier_id_(std::move(tma_op_to_barrier_id)),
        barrier_id_to_range_(std::move(barrier_id_to_range)),
        has_create_list_of_mbarrier_(has_create_list_of_mbarrier) {}

  static PrimFunc Rewrite(PrimFunc f, arith::Analyzer *analyzer) {
    auto buffer_lca = DetectBufferAccessLCA(f);
    Map<Var, Buffer> buffer_data_to_buffer_;
    for (auto [buffer, _] : buffer_lca)
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    f = TmaExpectTxRewriter::Rewrite(f, analyzer);
    TmaBarrierCollector collector(buffer_data_to_buffer_);
    collector(f->body);
    bool has_create_list_of_mbarrier = false;
    PostOrderVisit(f->body, [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        if (call->op.same_as(create_list_of_mbarrier())) {
          has_create_list_of_mbarrier = true;
        } else if (call->op.same_as(builtin::ptx_init_barrier_thread_count())) {
          has_create_list_of_mbarrier = true;
        }
      }
    });
    TmaBarrierRewriter rewriter(analyzer, collector.tma_op_to_barrier_id(),
                                collector.barrier_id_to_range(),
                                has_create_list_of_mbarrier);
    f.CopyOnWrite()->body = rewriter(f->body);
    auto barrier_creation_rewriter = BarrierCreationRewriter(
        rewriter.restore_barrier_ids_, rewriter.producer_thread_extent_);
    f.CopyOnWrite()->body = barrier_creation_rewriter(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const BlockNode *op) {
    auto block = GetRef<Block>(op);
    if (!has_create_list_of_mbarrier_ && !barrier_id_to_range_.empty() &&
        op->name_hint == MainBlockName) {
      ICHECK(false) << "Please declare create_list_of_mbarrier.";
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) {
    if (first_if) {
      if (op->condition.as<GENode>()) {
        producer_thread_extent_ =
            thread_var_->dom->extent - op->condition.as<GENode>()->b;
      }
      TmaSequenceCollector collector(tma_op_to_barrier_id_);
      collector(op->then_case);
      clear_expect_list_ = collector.GetSequence();
      restore_barrier_ids_ = collector.GetRestoreBarrierIds();
      first_if = false;

      is_producer_ = true;

      auto then_case = StmtExprMutator::VisitStmt(op->then_case);

      is_producer_ = false;
      Stmt else_case;
      if (op->else_case.defined())
        else_case = StmtExprMutator::VisitStmt(op->else_case.value());
      return IfThenElse(op->condition, then_case, else_case);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "kWarpSpecializationScope") {
      has_warp_specialization_ = true;
      first_if = true;
    } else if (op->attr_key == tir::attr::thread_extent &&
               Downcast<IterVar>(op->node)->thread_tag == "threadIdx.x") {
      thread_var_ = Downcast<IterVar>(op->node);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) {
    if (op->op.same_as(tma_load()) || op->op.same_as(tma_load_im2col())) {
      // check this must be in the tma_op_to_barrier_id_
      ICHECK(tma_op_to_barrier_id_.count(GetRef<Call>(op)))
          << "tma_load must be in the tma_op_to_barrier_id_";
      auto barrier_id = tma_op_to_barrier_id_[GetRef<Call>(op)];
      auto new_args = op->args;
      auto arg0 = op->args[0].as<Call>();
      auto is_1d_tma_load =
          arg0 && !arg0.value()->op.same_as(create_tma_descriptor()) &&
          !arg0.value()->op.same_as(create_tma_im2col_descriptor());
      if (is_1d_tma_load) {
        new_args.Set(2, barrier_id);
      } else {
        new_args.Set(1, barrier_id);
      }
      return Call(op->dtype, op->op, new_args);
    } else if (op->op.same_as(mbarrier_expect_tx())) {
      ICHECK(tma_op_to_barrier_id_.count(GetRef<Call>(op)))
          << "mbarrier_expect_tx must be in the tma_op_to_barrier_id_";
      auto barrier_id = tma_op_to_barrier_id_[GetRef<Call>(op)];
      auto new_args = op->args;
      new_args.Set(0, barrier_id);
      if (!has_warp_specialization_)
        clear_arrive_ = false;
      else
        clear_arrive_ = clear_expect_list_[cur_expect_idx_++];
      if (clear_arrive_) {
        return Call(op->dtype, builtin::ptx_arrive_barrier_expect_tx(),
                    new_args);
      }
      return Call(op->dtype, op->op, new_args);
    } else if (op->op.same_as(builtin::ptx_arrive_barrier())) {
      if (clear_arrive_) {
        clear_arrive_ = false;
        return 0;
      }
      // by default, all threads must wait.
      auto new_args = op->args;
      return Call(op->dtype, op->op, new_args);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }
  Map<ObjectRef, PrimExpr> tma_op_to_barrier_id_;
  Map<PrimExpr, IntImm> barrier_id_to_range_;
  bool has_create_list_of_mbarrier_;
  bool clear_arrive_{false};
  bool first_if{false}, has_warp_specialization_{false}, is_producer_{false};
  IterVar thread_var_;
  int tma_expect_tx_{0}, cur_expect_idx_{0};
  std::vector<bool> clear_expect_list_;
  std::vector<int> restore_barrier_ids_;
  PrimExpr producer_thread_extent_;
};

tvm::transform::Pass InjectTmaBarrier() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Check if function only uses threadIdx.x before proceeding
    if (!ThreadTagChecker::HasOnlyThreadIdxX(f)) {
      LOG(WARNING) << "InjectTmaBarrier will be disabled because the program "
                      "uses thread tags other than threadIdx.x\n"
                   << "If you want to use TMA barrier, please refactor "
                      "your program to use threadIdx.x only";
      // Return original function unchanged if other thread tags are found
      return f;
    }
    arith::Analyzer analyzer;
    return TmaBarrierRewriter::Rewrite(f, &analyzer);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectTmaBarrier", {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectTmaBarrier", InjectTmaBarrier);
});

} // namespace tl
} // namespace tvm
