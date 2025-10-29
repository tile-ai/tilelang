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
 * \file thread_storage_sync.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../op/builtin.h"
#include "./common/thread_sync_types.h"
#include "./storage_access.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

class TileLangThreadSyncPlanner : public TileLangStorageAccessVisitor {
public:
  explicit TileLangThreadSyncPlanner(StorageScope sync_scope)
      : sync_scope_(std::move(sync_scope)) {}

  // The syncs inserted before each statement
  std::unordered_set<const Object *> syncs_inserted_;

protected:
  bool Enabled(const VarNode *buf, const StorageScope &scope) const final {
    return in_device_env() && scope == sync_scope_;
  }
  // Plan the sync
  std::vector<AccessEntry> Summarize(std::vector<StmtEntry> seq,
                                     const ForNode *loop) final {
    // Redirect all "shared.dyn" buffer access to the same buffer var
    // so that the accesses can be planned together.
    Var shared_dyn_buf;
    for (StmtEntry &entry : seq) {
      for (AccessEntry &access : entry.access) {
        if (access.scope.rank == StorageRank::kShared &&
            access.scope.tag == ".dyn" && access.buffer.defined()) {
          if (!shared_dyn_buf.defined()) {
            shared_dyn_buf = access.buffer;
          } else {
            access.buffer = shared_dyn_buf;
          }
        }
      }
    }

    // Unsynced reads and writes
    std::vector<AccessEntry> reads;
    std::vector<AccessEntry> writes;
    // if it is a loop, rotate two times to consider effect of loop.
    // simulation based approach to find dependencies
    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry &s = seq[i];
      // check if sync before statement is needed.
      bool sync_before_stmt = (syncs_inserted_.count(s.stmt) != 0);
      // Apply the syncs added already.

      if (sync_before_stmt) {
        reads.clear();
        writes.clear();
      }

      for (const AccessEntry &acc : s.access) {
        if (acc.type == kRead) {
          if (FindConflict(writes, acc, false)) {
            sync_before_stmt = true;
            break;
          }
        } else if (acc.type == kWrite) {
          if (FindConflict(reads, acc, false) ||
              FindConflict(writes, acc, false)) {
            sync_before_stmt = true;
            break;
          }
        } else if (acc.type == kSync) {
          reads.clear();
          writes.clear();
        }
      }
      // If sync is inserted. remove the irrelevant things.
      if (sync_before_stmt) {
        reads.clear();
        writes.clear();
      }
      // Add the read/write of current statement
      for (const AccessEntry &acc : s.access) {
        if (acc.type == kRead) {
          reads.push_back(acc);
        } else if (acc.type == kWrite) {
          writes.push_back(acc);
        } else if (acc.type == kSync) {
          reads.clear();
          writes.clear();
        }
      }

      if (sync_before_stmt) {
        insert_syncs(s.stmt);
      }
    }
    if (loop != nullptr) {
      // Check if the loop body contains any reads in the same sync scope.
      // If there are reads, we conservatively keep the sync within the loop
      // body to preserve per-iteration ordering when needed. If there are no
      // reads (e.g., only writes to shared.dyn), we can safely hoist the sync
      // to before the loop to avoid redundant barriers.
      bool has_read_in_scope = false;
      for (const StmtEntry &s : seq) {
        for (const AccessEntry &acc : s.access) {
          if (acc.type == kRead && acc.scope == sync_scope_) {
            has_read_in_scope = true;
            break;
          }
        }
        if (has_read_in_scope)
          break;
      }
      // If there is a loop-carried dependency, insert a single sync
      // before the loop rather than hoisting a sync into the loop body.
      // This reduces redundant per-iteration synchronizations for cases
      // where each iteration touches disjoint regions (e.g., stmatrix
      // writes to shared.dyn) and only a global ordering before/after the
      // loop is required.
      for (size_t i = 0; i < seq.size(); ++i) {
        const StmtEntry &s = seq[i];
        if (syncs_inserted_.count(s.stmt) != 0)
          break;
        if (reads.empty() && writes.empty())
          break;
        bool need_loop_sync = false;
        for (const AccessEntry &acc : s.access) {
          if (acc.type == kRead) {
            if (FindConflict(writes, acc, true)) {
              need_loop_sync = true;
              break;
            }
          } else if (acc.type == kWrite) {
            if (FindConflict(reads, acc, true) ||
                FindConflict(writes, acc, true)) {
              need_loop_sync = true;
              break;
            }
          } else if (acc.type == kSync) {
            reads.clear();
            writes.clear();
          }
        }
        if (need_loop_sync) {
          if (!has_read_in_scope) {
            // Mark the loop itself to receive a sync before it, instead of
            // inserting inside the loop body. This ensures a single sync is
            // emitted outside the loop and avoids per-iteration overhead.
            insert_syncs(loop);
          } else {
            // Fall back to inserting before the first conflicting statement
            // inside the loop to maintain correctness when reads are present.
            insert_syncs(s.stmt);
          }
          break;
        }
      }
    }
    // return the exposed entries, remove unnecessary ones.
    int sync_count = 0;
    // head are before first sync, tail are after last sync
    std::vector<AccessEntry> head, tail;
    AccessEntry esync;
    esync.threads = this->env_threads();
    esync.thread_range = this->ComputeThreadRange(esync.threads);
    esync.type = kSync;
    esync.scope = sync_scope_;

    for (const StmtEntry &s : seq) {
      if (syncs_inserted_.count(s.stmt)) {
        if (sync_count != 0) {
          tail.clear();
        } else {
          head.push_back(esync);
        }
        ++sync_count;
      }
      for (const AccessEntry &acc : s.access) {
        if (acc.type == kSync) {
          if (sync_count != 0) {
            tail.clear();
          } else {
            head.push_back(esync);
          }
          ++sync_count;
        } else {
          if (sync_count != 0) {
            tail.push_back(acc);
          } else {
            head.push_back(acc);
          }
        }
      }
    }
    head.insert(head.end(), tail.begin(), tail.end());
    if (loop != nullptr) {
      // clear double buffer flag after a loop is finished.
      for (AccessEntry &e : head) {
        e.double_buffer_write = false;
      }
    }
    return head;
  }

private:
  // find conflicting entry in vec.
  bool FindConflict(const std::vector<AccessEntry> &prev,
                    const AccessEntry &curr, bool loop_carry) {
    for (const AccessEntry &x : prev) {
      if (FindConflict(x, curr, loop_carry)) {
        return true;
      }
    }
    return false;
  }

  bool FindConflict(const AccessEntry &prev, const AccessEntry &curr,
                    bool loop_carry) {
    // Special case: ignore conflicts between async-copy writes (e.g., TMA
    // loads into shared memory). Multiple async writes do not require
    // interspersed barriers among themselves. We still respect conflicts with
    // reads to ensure visibility before consumption.
    if (prev.type == kWrite && curr.type == kWrite && prev.is_async_copy &&
        curr.is_async_copy) {
      return false;
    }
    // Access to different buffers does not conflict.
    if (!prev.buffer.same_as(curr.buffer)) {
      return false;
    }

    // Assumes no race between threads
    // Same index value means no conflicts
    // TODO(tqchen) more standard set based testing.
    bool has_same_index = true;
    bool range_is_equal = true;
    bool range_is_overlap = true;

    for (const auto &kv : prev.thread_range) {
      if (!StructuralEqual()(kv.second, curr.thread_range[kv.first])) {
        range_is_equal = false;
        break;
      }
    }

    if (prev.buffer_indices.size() != curr.buffer_indices.size()) {
      // They are not the same indices, should be conflict.
      return true;
    }
    if (prev.is_pointer_access || curr.is_pointer_access) {
      // For accesses created via tvm_access_ptr we may still be able to prove
      // disjointness using their byte ranges.  If both sides expose a touched
      // interval and we can show they don't overlap, skip the conflict.
      if (prev.is_pointer_access && curr.is_pointer_access &&
          PointerAccessIsDisjoint(prev, curr)) {
        return false;
      }
      // Otherwise fall back to the conservative answer: treat them as
      // overlapping.
      return true;
    }

    for (size_t i = 0; i < prev.buffer_indices.size(); i++) {
      auto prev_dtype = prev.dtype;
      auto curr_dtype = curr.dtype;

      const auto &prev_indice = prev.buffer_indices[i];
      const auto &curr_indice = curr.buffer_indices[i];

      if (!ExprDeepEqual()(prev_indice, curr_indice)) {
        PrimExpr prev_indice_bytes =
            analyzer_.Simplify(prev_indice * prev_dtype.bytes());
        PrimExpr curr_indice_bytes =
            analyzer_.Simplify(curr_indice * curr_dtype.bytes());

        has_same_index = false;

        // If both are const, we can check if they are disjoint
        // by checking if the bounds are disjoint
        // [1024, 2048], [2048, 3072] are disjoint
        // [1024, 2048], [1024, 1024] are not disjoint
        auto prev_bound = analyzer_.const_int_bound(prev_indice_bytes);
        auto curr_bound = analyzer_.const_int_bound(curr_indice_bytes);
        if (prev_bound.defined() && curr_bound.defined()) {
          if ((prev_bound->min_value) > (curr_bound->max_value) ||
              (curr_bound->min_value) > (prev_bound->max_value)) {
            range_is_overlap = false;
            break;
          }
        }

        // if we can prove prev_indice < curr_indice or prev_indice >
        // curr_indice, then they are not overlap
        auto prev_indices_dtype = prev_indice.dtype();
        auto curr_indices_dtype = curr_indice.dtype();
        if (prev_indices_dtype.lanes() != curr_indices_dtype.lanes()) {
          // can not support different lanes binary op like <, >, <=, >=
          // skip otherwise it will lead to error
          continue;
        }

        // provably disjoint means no overlap, for example:
        // we can prove that tx - 128 < tx + 128, tx in [0, 128]
        // However, we should apply tx split because
        // tx < tx + 32 when tx in [0, 128] is not disjoint
        // because [0, 128] is not disjoint with [32, 160]
        // so we should split tx into tx0 and tx1.

        struct ThreadVarInfo {
          const char *name_prev;
          const char *name_curr;
          IterVar iv;
        } thread_vars[] = {
            {"tx1", "tx2", tx_},
            {"ty1", "ty2", ty_},
            {"tz1", "tz2", tz_},
        };

        for (const auto &info : thread_vars) {
          Var prev_var(info.name_prev, info.iv->var.dtype());
          Var curr_var(info.name_curr, info.iv->var.dtype());
          analyzer_.Bind(prev_var, info.iv->dom);
          analyzer_.Bind(curr_var, info.iv->dom);
          prev_indice_bytes =
              Substitute(prev_indice_bytes, {{info.iv->var, prev_var}});
          curr_indice_bytes =
              Substitute(curr_indice_bytes, {{info.iv->var, curr_var}});
        }

        bool provably_disjoint =
            analyzer_.CanProve(prev_indice_bytes < curr_indice_bytes,
                               arith::ProofStrength::kSymbolicBound) ||
            analyzer_.CanProve(prev_indice_bytes > curr_indice_bytes,
                               arith::ProofStrength::kSymbolicBound);

        if (provably_disjoint) {
          range_is_overlap = false;
          break;
        }
      }

      if (!has_same_index) {
        break;
      }
    }

    if (has_same_index && range_is_equal) {
      return false;
    }

    // If this is a read into a double buffer that was previously
    // swapped out, then it doesn't conflict.
    if (prev.double_buffer_write && curr.type == kRead && !loop_carry) {
      return false;
    }

    // If nothing else allows sharing the same buffer, then they are
    // in conflict.
    // if range_is_overlap is true, then they are in conflict, we should return
    // true. if range_is_overlap is false, then they are not in conflict, we
    // should return false.
    return range_is_overlap;
  }

  bool PointerAccessIsDisjoint(const AccessEntry &lhs, const AccessEntry &rhs) {
    if (lhs.touched.size() != 1 || rhs.touched.size() != 1) {
      return false;
    }
    PrimExpr lhs_min = analyzer_.Simplify(lhs.touched[0].min());
    PrimExpr lhs_max = analyzer_.Simplify(lhs.touched[0].max());
    PrimExpr rhs_min = analyzer_.Simplify(rhs.touched[0].min());
    PrimExpr rhs_max = analyzer_.Simplify(rhs.touched[0].max());

    if (analyzer_.CanProve(lhs_max < rhs_min,
                           arith::ProofStrength::kSymbolicBound)) {
      return true;
    }
    if (analyzer_.CanProve(rhs_max < lhs_min,
                           arith::ProofStrength::kSymbolicBound)) {
      return true;
    }
    return false;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        tx_ = iv;
      } else if (iv->thread_tag == "threadIdx.y") {
        ty_ = iv;
      } else if (iv->thread_tag == "threadIdx.z") {
        tz_ = iv;
      }
    }
    TileLangStorageAccessVisitor::VisitStmt_(op);
  }

  void insert_syncs(const Object *obj) {
    if (syncs_inserted_.count(obj))
      return;
    syncs_inserted_.insert(obj);
  }

private:
  // Member variables
  IterVar tx_ =
      IterVar(Range::FromMinExtent(0, 1), Var("tx"), IterVarType::kDataPar);
  IterVar ty_ =
      IterVar(Range::FromMinExtent(0, 1), Var("ty"), IterVarType::kDataPar);
  IterVar tz_ =
      IterVar(Range::FromMinExtent(0, 1), Var("tz"), IterVarType::kDataPar);
  // synchronization scope
  StorageScope sync_scope_;
};

// There are cases where necessary syncthreads is not inserted by
// ThreadSyncInserter. For example, syncthreads is needed after async_wait_queue
// in the second loop below, but since ThreadSyncInserter is not aware of the
// asynchronous semantics, it cannot tell that the syncthreads is needed there.
//
// // Pipeline prologue
// for i in range(125):
//    async_commit_queue(0):
//       async_scope:
//          shared[(i + 3) % 4] = ...
// ...
//
// // Pipeline Epilogue
// for i in range(3):
//    async_wait_queue(0, 2 - i):
//       local[...] = shared[(i + 125) % 4]

// This class adds syncthreads after all async_wait_queue. That includes
// syncthreads that can be inserted by ThreadSyncInserter as well, but
// ThreadSyncInserter will not insert duplicate syncthreads if it finds an
// existing one at the synchronization point.
class ThreadSyncAfterWaitQueueInserter : public StmtExprMutator {
public:
  explicit ThreadSyncAfterWaitQueueInserter(StorageScope sync_scope)
      : sync_scope_(std::move(sync_scope)) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tvm::tir::attr::async_wait_queue_scope) {
      auto sync = Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                                {StringImm(sync_scope_.to_string())}));
      auto inner = op->body.as<AttrStmtNode>();
      ICHECK(inner &&
             inner->attr_key == tvm::tir::attr::async_wait_inflight_count);
      auto zero = make_zero(DataType::Int(32));
      auto new_body = SeqStmt({sync, inner->body});
      return AttrStmt(zero, tvm::tir::attr::async_wait_queue_scope, op->value,
                      AttrStmt(zero, tvm::tir::attr::async_wait_inflight_count,
                               inner->value, new_body));
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  StorageScope sync_scope_;
};

class ThreadSyncInserter : public StmtExprMutator {
public:
  ThreadSyncInserter(StorageScope sync_scope,
                     const std::unordered_set<const Object *> &syncs)
      : sync_scope_(std::move(sync_scope)), syncs_(syncs) {}

  Stmt VisitStmt(const Stmt &stmt) final {
    if (syncs_.empty())
      return stmt;
    if (syncs_.count(stmt.get())) {
      Stmt barrier;
      if (sync_scope_.rank == StorageRank::kGlobal) {
        barrier = MakeGlobalBarrier();
      } else {
        barrier = Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                                {StringImm(sync_scope_.to_string())}));
      }
      // Mutate after query, to avoid stmt change.
      auto ret = StmtExprMutator::VisitStmt(stmt);
      ret = SeqStmt({barrier, ret});
      return ret;
    } else {
      return StmtExprMutator::VisitStmt(stmt);
    }
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    if (sync_scope_.rank == StorageRank::kGlobal &&
        GetScope(op->buffer->data).rank == StorageRank::kGlobal) {
      ++rw_stats_[op->buffer->data].read_count;
    }
    return StmtExprMutator::VisitExpr_(op);
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (sync_scope_.rank == StorageRank::kGlobal &&
        GetScope(op->buffer->data).rank == StorageRank::kGlobal) {
      ++rw_stats_[op->buffer->data].write_count;
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      bool temp = true;
      std::swap(temp, in_thread_env_);
      thread_extents_.push_back(op);
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      thread_extents_.pop_back();
      std::swap(temp, in_thread_env_);
      // first thread scope.
      if (!in_thread_env_ && sync_scope_.rank == StorageRank::kGlobal) {
        ret = InitGlobalBarrier(ret.as<AttrStmtNode>());
        num_blocks_ = PrimExpr();
        is_lead_ = PrimExpr();
      }
      return ret;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      PrimExpr expr = StmtExprMutator::VisitExpr_(op);
      op = expr.as<CallNode>();
      ICHECK_EQ(op->args.size(), 5U);
      Var buffer_var(Downcast<Var>(op->args[1]));
      const IntImmNode *flag = op->args[4].as<IntImmNode>();
      if ((flag->value & 1) && sync_scope_.rank == StorageRank::kGlobal &&
          GetScope(buffer_var).rank == StorageRank::kGlobal) {
        ++rw_stats_[buffer_var].read_count;
      }
      if (flag->value & 2 && sync_scope_.rank == StorageRank::kGlobal &&
          GetScope(buffer_var).rank == StorageRank::kGlobal) {
        ++rw_stats_[buffer_var].write_count;
      }
      return expr;
    } else if (op->op.same_as(builtin::address_of())) {
      PrimExpr expr = StmtExprMutator::VisitExpr_(op);
      op = expr.as<CallNode>();
      ICHECK_EQ(op->args.size(), 1U)
          << "address_of should only have one argument (Buffer)";

      if (auto load = op->args[0].as<BufferLoadNode>()) {
        Var buffer_var(Downcast<Var>(load->buffer->data));
        if (sync_scope_.rank == StorageRank::kGlobal &&
            GetScope(buffer_var).rank == StorageRank::kGlobal) {
          ++rw_stats_[buffer_var].read_count;
        }
        if (sync_scope_.rank == StorageRank::kGlobal &&
            GetScope(buffer_var).rank == StorageRank::kGlobal) {
          ++rw_stats_[buffer_var].write_count;
        }
        return expr;
      } else {
        return StmtExprMutator::VisitExpr_(op);
      }
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

private:
  // RW statistics about data
  struct Entry {
    int read_count{0};
    int write_count{0};
  };

  // Get current storage scope.
  StorageScope GetScope(Var buffer_var) const {
    return StorageScope::Create(GetPtrStorageScope(std::move(buffer_var)));
  }

  // private functions.
  Stmt InitGlobalBarrier(const AttrStmtNode *op) {
    ICHECK(op != nullptr);
    Array<PrimExpr> pargs = {
        StringImm(runtime::symbol::tvm_prepare_global_barrier)};
    Stmt prep =
        Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), pargs));
    Stmt body = op->body;
    for (const auto &kv : rw_stats_) {
      const auto &e = kv.second;
      if (e.read_count != 0 && e.write_count != 0) {
        body = AttrStmt(kv.first, tvm::tir::attr::volatile_scope, 1, body);
      }
    }
    rw_stats_.clear();
    Stmt kinit = Evaluate(
        Call(DataType::Int(32), builtin::tvm_global_barrier_kinit(), {}));
    body = SeqStmt({kinit, body});
    body = AttrStmt(op->node, op->attr_key, op->value, body);
    return SeqStmt({prep, body});
  }
  Stmt MakeGlobalBarrier() {
    ICHECK(sync_scope_.rank == StorageRank::kGlobal);
    if (!num_blocks_.defined()) {
      ICHECK(!is_lead_.defined());
      num_work_dim_ = thread_extents_.size();
      for (const AttrStmtNode *attr : thread_extents_) {
        IterVar iv = Downcast<IterVar>(attr->node);
        runtime::ThreadScope s = runtime::ThreadScope::Create(iv->thread_tag);
        if (s.rank == 0) {
          num_blocks_ =
              (num_blocks_.defined() ? attr->value * num_blocks_ : attr->value);
        } else if (s.rank == 1) {
          PrimExpr cond = iv->var == make_zero(iv->var.dtype());
          is_lead_ = is_lead_.defined() ? (is_lead_ && cond) : cond;
        }
      }
    } else {
      ICHECK_EQ(num_work_dim_, thread_extents_.size());
    }
    return Evaluate(
        Call(DataType::Int(32), builtin::tvm_storage_sync(),
             {StringImm(sync_scope_.to_string()), is_lead_, num_blocks_}));
  }
  // data structure.
  StorageScope sync_scope_;
  const std::unordered_set<const Object *> &syncs_;

  // The read write statistics of storage
  std::unordered_map<Var, Entry, ObjectPtrHash, ObjectPtrEqual> rw_stats_;
  // The statistics for global barrier
  bool in_thread_env_{false};
  // memorized results
  std::vector<const AttrStmtNode *> thread_extents_;
  size_t num_work_dim_{0};
  PrimExpr num_blocks_;
  PrimExpr is_lead_;
};

class ThreadPartialSyncRewriter : public IRMutatorWithAnalyzer {
public:
  static Stmt Rewrite(Stmt stmt) {
    arith::Analyzer analyzer;
    ThreadPartialSyncRewriter rewriter(&analyzer);
    return rewriter(std::move(stmt));
  }

private:
  explicit ThreadPartialSyncRewriter(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const CallNode *call = nullptr;
    if (op->value->IsInstance<CallNode>()) {
      call = op->value.as<CallNode>();
      if (call->op.same_as(builtin::tvm_storage_sync())) {
        const auto &args = call->args;
        ICHECK(!args.empty());
        const auto *scope_node = args[0].as<StringImmNode>();
        ICHECK(scope_node != nullptr);
        const std::string &scope = scope_node->value;

        if (args.size() != 1 || (scope != "shared" && scope != "shared.dyn")) {
          return IRMutatorWithAnalyzer::VisitStmt_(op);
        }

        return ProcessSharedSync(call, scope);
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt ProcessSharedSync(const CallNode *op, const std::string &scope) {
    // Get thread bounds
    auto bound_tx = analyzer_->const_int_bound(tx_);
    auto bound_ty = analyzer_->const_int_bound(ty_);
    auto bound_tz = analyzer_->const_int_bound(tz_);

    // Check if all threads are participating (full extent)
    if (IsFullThreadExtent(tx_, bound_tx) &&
        IsFullThreadExtent(ty_, bound_ty) &&
        IsFullThreadExtent(tz_, bound_tz)) {
      return Evaluate(IRMutatorWithAnalyzer::VisitExpr_(op));
    }

    // Calculate thread extents
    auto extent_tx = CalculateThreadExtent(tx_, bound_tx);
    auto extent_ty = CalculateThreadExtent(ty_, bound_ty);
    auto extent_tz = CalculateThreadExtent(tz_, bound_tz);

    // Create or get barrier info
    ThreadBoundKey key{bound_tx->min_value, bound_tx->max_value,
                       bound_ty->min_value, bound_ty->max_value,
                       bound_tz->min_value, bound_tz->max_value};

    auto [barrier_id, thread_count] =
        GetOrCreateBarrier(key, extent_tx, extent_ty, extent_tz);
    if (thread_count % 32 != 0) {
      // TODO(lei): This is a workaround for the case where the thread count is
      // not a multiple of 32. we should enhance the pass to analysis index
      // instead of buffer expression etc.
      return Stmt();
    }

    // Create new sync call with barrier info
    Array<PrimExpr> new_args = {StringImm(scope),
                                IntImm(DataType::Int(32), barrier_id),
                                IntImm(DataType::Int(32), thread_count)};
    return Evaluate(Call(op->dtype, op->op, new_args));
  }

  std::pair<size_t, size_t> GetOrCreateBarrier(const ThreadBoundKey &key,
                                               size_t extent_tx,
                                               size_t extent_ty,
                                               size_t extent_tz) {
    if (barrier_id_map_.count(key)) {
      return {barrier_id_map_[key], thread_count_map_[key]};
    }

    size_t barrier_id =
        barrier_id_map_.size() +
        static_cast<size_t>(ReservedNamedBarriers::kFirstUsedBarrier);
    size_t thread_count = extent_tx * extent_ty * extent_tz;

    barrier_id_map_[key] = barrier_id;
    thread_count_map_[key] = thread_count;

    return {barrier_id, thread_count};
  }

  size_t CalculateThreadExtent(const IterVar &iv,
                               const arith::ConstIntBound &bound) {
    if (!analyzer_->const_int_bound.IsBound(iv->var)) {
      return 1;
    }
    return bound->max_value - bound->min_value + 1;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        tx_ = iv;
      } else if (iv->thread_tag == "threadIdx.y") {
        ty_ = iv;
      } else if (iv->thread_tag == "threadIdx.z") {
        tz_ = iv;
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  bool IsFullThreadExtent(const IterVar &iv,
                          const arith::ConstIntBound &bound) {
    if (!analyzer_->const_int_bound.IsBound(iv->var)) {
      return true;
    }

    if (!iv->dom.defined()) {
      return true;
    }

    const auto *min_node = iv->dom->min.as<IntImmNode>();
    const auto *extent_node = iv->dom->extent.as<IntImmNode>();

    int64_t min = min_node->value;
    int64_t extent = extent_node->value;
    int64_t max = min + extent - 1;

    return min == bound->min_value && max == bound->max_value;
  }

  // Member variables
  IterVar tx_ =
      IterVar(Range::FromMinExtent(0, 1), Var("tx"), IterVarType::kDataPar);
  IterVar ty_ =
      IterVar(Range::FromMinExtent(0, 1), Var("ty"), IterVarType::kDataPar);
  IterVar tz_ =
      IterVar(Range::FromMinExtent(0, 1), Var("tz"), IterVarType::kDataPar);
  std::unordered_map<ThreadBoundKey, size_t> barrier_id_map_;
  std::unordered_map<ThreadBoundKey, size_t> thread_count_map_;
};

PrimFunc TileLangThreadSync(PrimFunc func, const std::string &storage_scope) {
  StorageScope sync_scope = StorageScope::Create(storage_scope);
  auto *n = func.CopyOnWrite();
  auto stmt = n->body;
  if (sync_scope.rank == StorageRank::kShared && sync_scope.tag.empty()) {
    stmt = ThreadSyncAfterWaitQueueInserter(sync_scope)(stmt);
  }
  TileLangThreadSyncPlanner planner(sync_scope);
  for (const auto &[_, buffer] : func->buffer_map) {
    planner.SetBufferDataToBuffer(buffer->data, buffer);
  }
  planner(stmt);

  stmt =
      ThreadSyncInserter(sync_scope, planner.syncs_inserted_)(std::move(stmt));
  n->body = ThreadPartialSyncRewriter::Rewrite(std::move(stmt));
  return func;
}

using namespace tir::transform;

namespace transform {

tvm::transform::Pass ThreadSync(const String &storage_scope) {
  auto pass_func = [storage_scope](PrimFunc f, const IRModule &m,
                                   const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    // Check if thread storage sync is disabled
    bool disable_syncthreads =
        ctx->GetConfig(kDisableThreadStorageSync, Bool(false)).value()->value;
    if (disable_syncthreads) {
      return f;
    }
    return tl::TileLangThreadSync(std::move(f), storage_scope);
    ;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ThreadSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ThreadSync", ThreadSync);
}

} // namespace transform
} // namespace tl
} // namespace tvm
