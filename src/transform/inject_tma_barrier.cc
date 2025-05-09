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
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "./common/collector.h"
#include "./common/attr.h"

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

  void Collect(Stmt stmt) { VisitStmt(stmt); }

  PrimExpr BulkCopyBytes() { return bulk_copy_bytes; }

private:
  void VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col())) {
      Call access_ptr = Downcast<Call>(call->args[2]);
      ICHECK(access_ptr->op.same_as(builtin::tvm_access_ptr()));
      int type_bytes = access_ptr->args[0]->dtype.bytes();
      bulk_copy_bytes += access_ptr->args[3] * loop_extents * type_bytes;
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
    return Call(DataType::Handle(), get_mbarrier(), {barrier_id});
  }

  Stmt makeExpectTX(PrimExpr barrier_id, PrimExpr bytes) {
    auto call = Call(DataType::Handle(), mbarrier_expect_tx(),
                     {makeGetBarrier(barrier_id), bytes});
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
    const EQNode *eq = op->condition.as<EQNode>();
    if (eq != nullptr) {
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
    if (op->op.same_as(tma_load())) {
      visited_tma_load_ = true;
      Array<PrimExpr> new_args = op->args;
      new_args.Set(1, Call(DataType::Handle(), get_mbarrier(),
                           {IntImm(DataType::Int(32), 0)}));
      return Call(op->dtype, op->op, new_args);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }
};

class TmaBarrierCollector : public IRVisitorWithAnalyzer {
public:
  Map<ObjectRef, PrimExpr> tma_op_to_barrier_id() {
    return tma_op_to_barrier_id_;
  }
  Map<PrimExpr, IntImm> barrier_id_to_range() { return barrier_id_to_range_; }

private:
  void UpdateBarrierRange(PrimExpr barrier_id, IntImm extent) {
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
      if (call->op.same_as(tma_load())) {
        pending_tma_ops_.push_back(GetRef<Call>(call));
      } else if (call->op.same_as(mbarrier_expect_tx())) {
        pending_tma_ops_.push_back(GetRef<Call>(call));
      } else if (call->op.same_as(builtin::ptx_arrive_barrier())) {
        PrimExpr barrier_id = call->args[0];
        for (auto tma_call : pending_tma_ops_) {
          tma_op_to_barrier_id_.Set(tma_call, barrier_id);
        }
        auto const_int_bound = analyzer_.const_int_bound(thread_var_);
        auto extent = const_int_bound->max_value - const_int_bound->min_value + 1;
        UpdateBarrierRange(barrier_id, IntImm(DataType::Int(32), extent));
        pending_tma_ops_.clear();
      } else if (call->op.same_as(builtin::ptx_wait_barrier())) {
        PrimExpr barrier_id = call->args[0];
        auto const_int_bound = analyzer_.const_int_bound(thread_var_);
        auto extent = const_int_bound->max_value - const_int_bound->min_value + 1;
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
};
// we trust mbarrier_wait_parity to be correct
class TmaBarrierRewriter : public IRMutatorWithAnalyzer {
public:
  TmaBarrierRewriter(arith::Analyzer *analyzer,
                     Map<ObjectRef, PrimExpr> tma_op_to_barrier_id,
                     Map<PrimExpr, IntImm> barrier_id_to_range,
                     bool has_create_list_of_mbarrier)
      : IRMutatorWithAnalyzer(analyzer),
        tma_op_to_barrier_id_(tma_op_to_barrier_id),
        barrier_id_to_range_(barrier_id_to_range),
        has_create_list_of_mbarrier_(has_create_list_of_mbarrier) {}

  static PrimFunc Rewrite(PrimFunc f, arith::Analyzer *analyzer) {
    f = TmaExpectTxRewriter::Rewrite(f, analyzer);
    TmaBarrierCollector collector;
    collector(f->body);
    bool has_create_list_of_mbarrier = false;
    PostOrderVisit(f->body, [&](const ObjectRef& node) {
      if (const auto* call = node.as<CallNode>()) {
        if (call->op.same_as(create_list_of_mbarrier())) {
          has_create_list_of_mbarrier = true;
        }
      }
    });
    TmaBarrierRewriter rewriter(analyzer, collector.tma_op_to_barrier_id(),
                                collector.barrier_id_to_range(), has_create_list_of_mbarrier);
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:


  Stmt VisitStmt_(const BlockNode *op){
    auto block = GetRef<Block>(op);
    if (!has_create_list_of_mbarrier_ && op->name_hint == MainBlockName) {
      ICHECK(false) << "Please declare create_list_of_mbarrier.";
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) {
    if (op->op.same_as(tma_load())) {
      // check this must be in the tma_op_to_barrier_id_
      ICHECK(tma_op_to_barrier_id_.count(GetRef<Call>(op)))
          << "tma_load must be in the tma_op_to_barrier_id_";
      auto barrier_id = tma_op_to_barrier_id_[GetRef<Call>(op)];
      auto new_args = op->args;
      new_args.Set(1, barrier_id);
      return Call(op->dtype, op->op, new_args);
    } else if (op->op.same_as(mbarrier_expect_tx())) {
      ICHECK(tma_op_to_barrier_id_.count(GetRef<Call>(op)))
          << "mbarrier_expect_tx must be in the tma_op_to_barrier_id_";
      auto barrier_id = tma_op_to_barrier_id_[GetRef<Call>(op)];
      auto new_args = op->args;
      new_args.Set(0, barrier_id);
      return Call(op->dtype, op->op, new_args);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }
  Map<ObjectRef, PrimExpr> tma_op_to_barrier_id_;
  Map<PrimExpr, IntImm> barrier_id_to_range_;
  bool has_create_list_of_mbarrier_;
};

tvm::transform::Pass InjectTmaBarrier() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
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

TVM_REGISTER_GLOBAL("tl.transform.InjectTmaBarrier")
    .set_body_typed(InjectTmaBarrier);

} // namespace tl
} // namespace tvm
