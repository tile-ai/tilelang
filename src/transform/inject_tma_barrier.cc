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

#include <unordered_map>
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

static inline bool Is1DTmaLoad(const CallNode *op) {
  if (!op->op.same_as(tma_load())) {
    return false;
  }
  auto arg0 = op->args[0].as<Call>();
  return arg0 && !arg0.value()->op.same_as(create_tma_descriptor()) &&
         !arg0.value()->op.same_as(create_tma_im2col_descriptor());
}

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
      bool is_1d_tma_load = Is1DTmaLoad(op);
      visited_tma_load_ = true;
      Array<PrimExpr> new_args = op->args;
      new_args.Set(is_1d_tma_load ? 2 : 1,
                   Call(DataType::Handle(), get_mbarrier(),
                        {IntImm(DataType::Int(32), 0)}));
      return Call(op->dtype, op->op, new_args, op->annotations);
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
  Map<PrimExpr, IntImm> cluster_barrier_cta_ids() {
    return cluster_barrier_cta_ids_;
  }

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
        pending_tma_ops_.push_back(tvm::ffi::GetRef<Call>(call));
      } else if (call->op.same_as(mbarrier_expect_tx())) {
        pending_tma_ops_.push_back(tvm::ffi::GetRef<Call>(call));
      } else if (call->op.same_as(builtin::ptx_arrive_barrier()) ||
                 call->op.same_as(tl::ptx_arrive_cluster_barrier())) {
        PrimExpr barrier_id = call->args[0];
        for (const auto &tma_call : pending_tma_ops_) {
          tma_op_to_barrier_id_.Set(tma_call, barrier_id);
        }
        // Track cluster barriers and their leader cta_id
        if (call->op.same_as(tl::ptx_arrive_cluster_barrier())) {
          if (const auto *imm = call->args[1].as<IntImmNode>()) {
            cluster_barrier_cta_ids_.Set(
                barrier_id, IntImm(DataType::Int(32), imm->value));
          }
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
  Map<PrimExpr, IntImm> cluster_barrier_cta_ids_;
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
          if (clear_zero_list[zero_idx] == false && !is_cluster_[zero_idx]) {
            int begin = int_sets_[zero_idx].min().as<IntImmNode>()->value;
            int end = int_sets_[zero_idx].max().as<IntImmNode>()->value;
            for (int i = begin; i <= end; ++i) {
              restore_barrier_ids_.push_back(i);
            }
          }
        } else {
          for (int i{zero_idx}; i > zero_idx - zero_count; --i) {
            if (!is_cluster_[i]) {
              int begin = int_sets_[i].min().as<IntImmNode>()->value;
              int end = int_sets_[i].max().as<IntImmNode>()->value;
              for (int j = begin; j <= end; ++j) {
                restore_barrier_ids_.push_back(j);
              }
            }
          }
        }
        zero_count = 0;
      }
    }

    return clear_zero_list;
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(mbarrier_expect_tx())) {
      auto call_ref = tvm::ffi::GetRef<Call>(op);
      if (tma_op_to_barrier_id_.count(call_ref)) {
        PrimExpr barrier_id = tma_op_to_barrier_id_[call_ref];
        // Cluster barriers have a BufferLoad as barrier_id (not get_mbarrier).
        // Skip int_set computation for them — they don't need restore_barrier_ids_.
        bool is_cluster = (barrier_id.as<CallNode>() == nullptr);
        arith::IntSet int_set = arith::IntSet::Nothing();
        if (!is_cluster) {
          PrimExpr e = barrier_id.as<CallNode>()->args[0];
          int_set = arith::EvalSet(e, var_int_set_);
        }
        expect_.push_back(if_depth_ == 1);
        sequence.push_back(0);
        int_sets_.push_back(int_set);
        is_cluster_.push_back(is_cluster);
        expect_tx_count_ += 1;
      }
    } else if (op->op.same_as(builtin::ptx_arrive_barrier()) ||
               op->op.same_as(tl::ptx_arrive_cluster_barrier())) {
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
  std::vector<bool> is_cluster_;
  bool has_simt_copy_{false};
  int if_depth_{0};
  Map<ObjectRef, PrimExpr> tma_op_to_barrier_id_;
};

class ArriveThreadCountCollector : public IRVisitorWithAnalyzer {
public:
  ArriveThreadCountCollector() = default;

  const std::unordered_map<int, int> &barrier_thread_counts() const {
    return barrier_thread_counts_;
  }

private:
  PrimExpr NormalizeBarrierExpr(const PrimExpr &barrier_expr) const {
    if (const auto *call = barrier_expr.as<CallNode>()) {
      if (call->op.same_as(get_mbarrier())) {
        ICHECK_EQ(call->args.size(), 1);
        return call->args[0];
      }
    }
    return barrier_expr;
  }

  int GetCurrentThreadCount() {
    if (inside_elect_if_) {
      return 1;
    }
    if (!thread_var_.defined()) {
      return 1;
    }
    auto bound = analyzer_.const_int_bound(thread_var_);
    int64_t extent = bound->max_value - bound->min_value + 1;
    return static_cast<int>(std::max<int64_t>(extent, 1));
  }

  void UpdateBarrierThreadCount(const PrimExpr &barrier_expr,
                                int thread_count) {
    PrimExpr normalized_barrier_expr = NormalizeBarrierExpr(barrier_expr);

    if (const auto *imm = normalized_barrier_expr.as<IntImmNode>()) {
      int id = static_cast<int>(imm->value);
      auto it = barrier_thread_counts_.find(id);
      if (it == barrier_thread_counts_.end()) {
        barrier_thread_counts_[id] = thread_count;
      } else {
        it->second = std::max(it->second, thread_count);
      }
      return;
    }

    auto int_set = arith::EvalSet(normalized_barrier_expr, var_int_set_);
    const auto *min_imm = int_set.min().as<IntImmNode>();
    const auto *max_imm = int_set.max().as<IntImmNode>();
    if (!min_imm || !max_imm) {
      return;
    }
    int begin = static_cast<int>(min_imm->value);
    int end = static_cast<int>(max_imm->value);
    for (int id = begin; id <= end; ++id) {
      auto it = barrier_thread_counts_.find(id);
      if (it == barrier_thread_counts_.end()) {
        barrier_thread_counts_[id] = thread_count;
      } else {
        it->second = std::max(it->second, thread_count);
      }
    }
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv;
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    var_int_set_.Set(op->loop_var,
                     arith::IntSet::FromMinExtent(op->min, op->extent));
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    bool is_elect_if = false;
    if (const auto *call = op->condition.as<CallNode>()) {
      is_elect_if = call->op.same_as(tl_shuffle_elect());
    }
    if (is_elect_if) {
      bool old_inside = inside_elect_if_;
      inside_elect_if_ = true;
      IRVisitorWithAnalyzer::VisitStmt(op->then_case);
      inside_elect_if_ = old_inside;
      if (op->else_case.defined()) {
        IRVisitorWithAnalyzer::VisitStmt(op->else_case.value());
      }
      return;
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::ptx_arrive_barrier()) ||
        op->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
        op->op.same_as(builtin::ptx_cp_async_barrier()) ||
        op->op.same_as(tl::ptx_cp_async_barrier_noinc())) {
      ICHECK_GE(op->args.size(), 1);
      UpdateBarrierThreadCount(op->args[0], GetCurrentThreadCount());
    }
    IRVisitorWithAnalyzer::VisitExpr_(op);
  }

  IterVar thread_var_;
  bool inside_elect_if_{false};
  Map<Var, arith::IntSet> var_int_set_;
  std::unordered_map<int, int> barrier_thread_counts_;
};

class BarrierCreationRewriter : public StmtExprMutator {
public:
  BarrierCreationRewriter(std::unordered_map<int, int> barrier_thread_counts,
                          int ensure_min_count = 0,
                          PrimExpr default_barrier_thread_count = 1)
      : barrier_thread_counts_(std::move(barrier_thread_counts)),
        ensure_min_count_(ensure_min_count),
        default_barrier_thread_count_(std::move(default_barrier_thread_count)) {
  }

  PrimExpr VisitExpr_(const CallNode *op) {
    if (op->op.same_as(create_list_of_mbarrier())) {
      size_t cur_n = op->args.size();
      size_t need_n =
          std::max<size_t>(cur_n, static_cast<size_t>(ensure_min_count_));

      Array<PrimExpr> new_args;
      new_args.reserve(need_n);

      // Preserve existing entries unless we have explicit arrive-domain counts.
      for (size_t i{0}; i < cur_n; ++i) {
        auto it = barrier_thread_counts_.find(static_cast<int>(i));
        if (it != barrier_thread_counts_.end()) {
          new_args.push_back(Integer(it->second));
        } else {
          new_args.push_back(op->args[i]);
        }
      }
      // Append additional barriers if required.
      for (size_t i = cur_n; i < need_n; ++i) {
        auto it = barrier_thread_counts_.find(static_cast<int>(i));
        if (it != barrier_thread_counts_.end()) {
          new_args.push_back(Integer(it->second));
        } else {
          new_args.push_back(default_barrier_thread_count_);
        }
      }
      return Call(op->dtype, op->op, new_args, op->annotations);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

private:
  std::unordered_map<int, int> barrier_thread_counts_;
  int ensure_min_count_{0};
  PrimExpr default_barrier_thread_count_{1};
};

// we trust mbarrier_wait_parity to be correct
class TmaBarrierRewriter : public IRMutatorWithAnalyzer {
public:
  TmaBarrierRewriter(arith::Analyzer *analyzer,
                     Map<ObjectRef, PrimExpr> tma_op_to_barrier_id,
                     Map<PrimExpr, IntImm> barrier_id_to_range,
                     Map<PrimExpr, IntImm> cluster_barrier_cta_ids,
                     bool has_create_list_of_mbarrier,
                     int cluster_size)
      : IRMutatorWithAnalyzer(analyzer),
        tma_op_to_barrier_id_(std::move(tma_op_to_barrier_id)),
        barrier_id_to_range_(std::move(barrier_id_to_range)),
        cluster_barrier_cta_ids_(std::move(cluster_barrier_cta_ids)),
        has_create_list_of_mbarrier_(has_create_list_of_mbarrier),
        cluster_size_(cluster_size) {}

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
    // Compute total cluster size from the "cluster_dims" block annotation
    int cluster_size = 1;
    PostOrderVisit(f->body, [&](const ObjectRef &node) {
      if (const auto *block = node.as<BlockNode>()) {
        if (block->annotations.count("cluster_dims")) {
          if (auto arr =
                  block->annotations.Get("cluster_dims")
                      ->try_cast<Array<Integer>>()) {
            int sz = 1;
            for (auto d : arr.value())
              sz *= static_cast<int>(d->value);
            cluster_size = sz;
          }
        }
      }
    });
    TmaBarrierRewriter rewriter(analyzer, collector.tma_op_to_barrier_id(),
                                collector.barrier_id_to_range(),
                                collector.cluster_barrier_cta_ids(),
                                has_create_list_of_mbarrier, cluster_size);
    f.CopyOnWrite()->body = rewriter(f->body);
    // Compute the minimum number of barriers actually referenced in the body
    // after TMA barrier rewrites (e.g., get_mbarrier(0) inserted for TMA).
    struct GetMbarrierMaxIdxCollector : public StmtExprVisitor {
      int max_idx{-1};
      void VisitExpr_(const CallNode *op) final {
        if (op->op.same_as(get_mbarrier())) {
          if (op->args.size() == 1) {
            if (const auto *imm = op->args[0].as<IntImmNode>()) {
              max_idx = std::max(max_idx, static_cast<int>(imm->value));
            }
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }
    };

    GetMbarrierMaxIdxCollector max_idx_collector;
    max_idx_collector(f->body);
    int ensure_min_count = max_idx_collector.max_idx + 1; // 0-based -> count

    ArriveThreadCountCollector arrive_thread_count_collector;
    arrive_thread_count_collector(f->body);

    // Default appended barriers to leader-only (=1), but prefer explicit
    // arrive-domain counts collected from actual arrive sites.
    auto barrier_creation_rewriter = BarrierCreationRewriter(
        arrive_thread_count_collector.barrier_thread_counts(), ensure_min_count,
        Integer(1));
    f.CopyOnWrite()->body = barrier_creation_rewriter(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const BlockNode *op) {
    auto block = tvm::ffi::GetRef<Block>(op);
    if (!has_create_list_of_mbarrier_ && !barrier_id_to_range_.empty() &&
        op->name_hint == MainBlockName) {
      ICHECK(false) << "Please declare create_list_of_mbarrier.";
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) {
    if (first_if) {
      TmaSequenceCollector collector(tma_op_to_barrier_id_);
      collector(op->then_case);
      clear_expect_list_ = collector.GetSequence();
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
      cur_expect_idx_ = 0;
    } else if (op->attr_key == tir::attr::thread_extent &&
               Downcast<IterVar>(op->node)->thread_tag == "threadIdx.x") {
      thread_var_ = Downcast<IterVar>(op->node);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  // Intercept mbarrier_expect_tx for cluster barriers: multiply bytes by
  // cluster_size and wrap the call in `if (block_rank_in_cluster() == cta_id)`
  // so only the leader CTA issues the full expect_transaction.
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(mbarrier_expect_tx())) {
        auto call_ref = tvm::ffi::GetRef<Call>(call);
        if (tma_op_to_barrier_id_.count(call_ref)) {
          auto barrier_id = tma_op_to_barrier_id_[call_ref];
          if (cluster_barrier_cta_ids_.count(barrier_id)) {
            auto cta_id = cluster_barrier_cta_ids_[barrier_id];
            // Keep cur_expect_idx_ consistent with VisitExpr_ expectations
            if (has_warp_specialization_) {
              clear_arrive_ = clear_expect_list_[cur_expect_idx_++];
            }
            clear_arrive_ = false;
            // Rewrite barrier arg and multiply bytes by cluster_size
            Array<PrimExpr> new_args = call->args;
            new_args.Set(0, barrier_id);
            new_args.Set(1, call->args[1] *
                                IntImm(DataType::Int(32), cluster_size_));
            auto new_call =
                Call(call->dtype, call->op, new_args, call->annotations);
            // Wrap in `if (block_rank_in_cluster() == cta_id)`
            PrimExpr rank =
                Call(DataType::Int(32), tl::block_rank_in_cluster(), {});
            PrimExpr cond = EQ(rank, IntImm(DataType::Int(32), cta_id->value));
            return IfThenElse(cond, Evaluate(new_call), Stmt());
          }
        }
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) {
    if (op->op.same_as(tma_load()) || op->op.same_as(tma_load_im2col())) {
      auto call_ref = tvm::ffi::GetRef<Call>(op);
      if (!tma_op_to_barrier_id_.count(call_ref)) {
        // Promote raw integer barrier id to get_mbarrier(id) so codegen can
        // emit mbarrier[index]. This handles degenerate producer-only kernels
        // where no arrive()/expect mapping is recorded.
        bool is_1d_tma_load = Is1DTmaLoad(op);
        if (is_1d_tma_load && op->args.size() >= 3) {
          if (const auto *imm = op->args[2].as<IntImmNode>()) {
            Array<PrimExpr> new_args = op->args;
            new_args.Set(2, Call(DataType::Handle(), get_mbarrier(),
                                 {IntImm(DataType::Int(32),
                                         static_cast<int>(imm->value))}));
            return Call(op->dtype, op->op, new_args, op->annotations);
          }
        } else if (!is_1d_tma_load && op->args.size() >= 2) {
          if (const auto *imm = op->args[1].as<IntImmNode>()) {
            Array<PrimExpr> new_args = op->args;
            new_args.Set(1, Call(DataType::Handle(), get_mbarrier(),
                                 {IntImm(DataType::Int(32),
                                         static_cast<int>(imm->value))}));
            return Call(op->dtype, op->op, new_args, op->annotations);
          }
        }
        return IRMutatorWithAnalyzer::VisitExpr_(op);
      }
      auto barrier_id = tma_op_to_barrier_id_[call_ref];
      auto new_args = op->args;
      bool is_1d_tma_load = Is1DTmaLoad(op);
      if (is_1d_tma_load) {
        new_args.Set(2, barrier_id);
      } else {
        new_args.Set(1, barrier_id);
      }
      // For cluster barriers, add use_2cta annotation → emits tma_load_2sm
      Map<String, ObjectRef> new_annotations = op->annotations;
      if (cluster_barrier_cta_ids_.count(barrier_id)) {
        new_annotations.Set("use_2cta", Bool(true));
      }
      return Call(op->dtype, op->op, new_args, new_annotations);
    } else if (op->op.same_as(mbarrier_expect_tx())) {
      auto call_ref = tvm::ffi::GetRef<Call>(op);
      if (!tma_op_to_barrier_id_.count(call_ref)) {
        return IRMutatorWithAnalyzer::VisitExpr_(op);
      }
      auto barrier_id = tma_op_to_barrier_id_[call_ref];
      auto new_args = op->args;
      new_args.Set(0, barrier_id);
      if (!has_warp_specialization_)
        clear_arrive_ = false;
      else
        clear_arrive_ = clear_expect_list_[cur_expect_idx_++];
      if (clear_arrive_) {
        return Call(op->dtype, builtin::ptx_arrive_barrier_expect_tx(),
                    new_args, op->annotations);
      }
      return Call(op->dtype, op->op, new_args, op->annotations);
    } else if (op->op.same_as(builtin::ptx_arrive_barrier()) ||
               op->op.same_as(tl::ptx_arrive_cluster_barrier())) {
      if (clear_arrive_) {
        clear_arrive_ = false;
        return 0;
      }
      // by default, all threads must wait.
      auto new_args = op->args;
      return Call(op->dtype, op->op, new_args, op->annotations);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }
  Map<ObjectRef, PrimExpr> tma_op_to_barrier_id_;
  Map<PrimExpr, IntImm> barrier_id_to_range_;
  Map<PrimExpr, IntImm> cluster_barrier_cta_ids_;
  bool has_create_list_of_mbarrier_;
  bool clear_arrive_{false};
  bool first_if{false}, has_warp_specialization_{false}, is_producer_{false};
  IterVar thread_var_;
  int tma_expect_tx_{0}, cur_expect_idx_{0};
  int cluster_size_{1};
  std::vector<bool> clear_expect_list_;
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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectTmaBarrier", InjectTmaBarrier);
}

} // namespace tl
} // namespace tvm
