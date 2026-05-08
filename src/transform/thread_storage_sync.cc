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
#include "../op/builtin.h"
#include "./common/constr_visitor.h"
#include "./common/shared_access_analysis.h"
#include "./common/thread_sync_types.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include <algorithm>
#include <string>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using namespace ffi;
using arith::IRMutatorWithAnalyzer;
using runtime::StorageRank;
using runtime::StorageScope;

// Similar to ThreadSyncAfterWaitQueueInserter, but for explicit cp.async
// synchronization intrinsics (ptx_wait_group).
//
// In TileLang, cp.async copies may be lowered to explicit ptx_cp_async +
// ptx_commit_group, and pipelining can move ptx_wait_group away from the copy
// statement it originally guarded. tvm_storage_sync barriers inserted by
// ThreadSyncPlanner are based on memory conflicts and may end up *before* the
// wait_group, which is incorrect for cp.async because __syncthreads() does not
// wait for outstanding asynchronous copies.
//
// Correct usage requires:
//   ptx_wait_group(N);
//   tvm_storage_sync("shared");   // __syncthreads()
// before any cross-thread consumption of the shared memory written by cp.async.
//
// This rewriter conservatively inserts a shared-memory storage sync
// immediately after every ptx_wait_group statement unless an identical sync
// already follows.
class ThreadSyncAfterWaitGroupInserter : public StmtExprMutator {
public:
  explicit ThreadSyncAfterWaitGroupInserter(StorageScope sync_scope)
      : sync_scope_(std::move(sync_scope)) {}

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> visited;
    visited.reserve(op->seq.size());
    for (const Stmt &stmt : op->seq) {
      visited.push_back(this->VisitStmt(stmt));
    }

    Array<Stmt> rewritten;
    rewritten.reserve(visited.size());
    for (int i = 0, n = static_cast<int>(visited.size()); i < n; ++i) {
      const Stmt &stmt = visited[i];
      rewritten.push_back(stmt);
      if (IsWaitGroupStmt(stmt)) {
        bool next_is_sync = false;
        if (i + 1 < n && IsStorageSyncStmt(visited[i + 1])) {
          next_is_sync = true;
        }
        if (!next_is_sync) {
          rewritten.push_back(MakeStorageSyncStmt());
        }
      }
    }

    if (rewritten.empty()) {
      return Evaluate(0);
    }
    if (rewritten.size() == 1) {
      return rewritten[0];
    }
    return SeqStmt(rewritten);
  }

private:
  StorageScope sync_scope_;

  Stmt MakeStorageSyncStmt() const {
    return Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                         {StringImm(sync_scope_.to_string())}));
  }

  bool IsWaitGroupStmt(const Stmt &stmt) const {
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return IsWaitGroupStmt(let->body);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return IsWaitGroupStmt(attr->body);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return IsWaitGroupStmt(seq->seq[0]);
      }
      return false;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return IsWaitGroupStmt(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return IsWaitGroupStmt(realize->block->body);
      }
      return false;
    }
    if (const auto *iff = stmt.as<IfThenElseNode>()) {
      if (!iff->else_case.defined()) {
        return IsWaitGroupStmt(iff->then_case);
      }
      return false;
    }

    const auto *eval = stmt.as<EvaluateNode>();
    if (!eval) {
      return false;
    }
    const auto *call = eval->value.as<CallNode>();
    if (!call) {
      return false;
    }
    return call->op.same_as(builtin::ptx_wait_group());
  }

  bool IsStorageSyncStmt(const Stmt &stmt) const {
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return IsStorageSyncStmt(let->body);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return IsStorageSyncStmt(attr->body);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return IsStorageSyncStmt(seq->seq[0]);
      }
      return false;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return IsStorageSyncStmt(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return IsStorageSyncStmt(realize->block->body);
      }
      return false;
    }
    if (const auto *iff = stmt.as<IfThenElseNode>()) {
      if (!iff->else_case.defined()) {
        return IsStorageSyncStmt(iff->then_case);
      }
      return false;
    }

    const auto *eval = stmt.as<EvaluateNode>();
    if (!eval) {
      return false;
    }
    const auto *call = eval->value.as<CallNode>();
    if (!call) {
      return false;
    }
    if (!call->op.same_as(builtin::tvm_storage_sync())) {
      return false;
    }
    if (call->args.size() != 1) {
      return false;
    }
    const auto *scope = call->args[0].as<StringImmNode>();
    if (!scope) {
      return false;
    }
    return scope->value == sync_scope_.to_string();
  }
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

  /*!
   * \brief Calculate the number of threads that satisfy current constraints.
   *
   * This method uses Z3's model enumeration (AllSAT) to precisely count
   * how many thread IDs satisfy all current constraints. This is essential
   * for cases like `if (threadIdx.x % 4 == 0)` where const_int_bound only
   * gives us the range [0, 127] but the actual number of satisfying threads
   * is 32 (i.e., 0, 4, 8, ..., 124).
   *
   * Falls back to range-based calculation if Z3 enumeration fails or returns
   * an invalid result.
   */
  size_t CalculateThreadExtent(const IterVar &iv,
                               const arith::ConstIntBound &bound) {
    if (!analyzer_->const_int_bound.IsBound(iv->var)) {
      return 1;
    }
    auto extent = *as_const_int(iv->dom->extent);
    // Always use Z3 enumeration to count satisfying values.
    // This handles constraints like `tx % 4 == 0` that const_int_bound cannot
    // detect. Z3 enumeration will return the exact count of satisfying values.
    int64_t z3_count =
        analyzer_->z3_prover.CountSatisfyingValues(iv->var, extent);
    if (z3_count > 0) {
      return static_cast<size_t>(z3_count);
    }

    // Fallback to range-based calculation if Z3 enumeration failed
    return static_cast<size_t>(bound->max_value - bound->min_value + 1);
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

struct ConditionThreadProperty {
  bool depends_on_runtime{false};
  bool is_block_uniform{true};
  bool requires_hoist{false};

  void Merge(const ConditionThreadProperty &other) {
    depends_on_runtime = depends_on_runtime || other.depends_on_runtime;
    is_block_uniform = is_block_uniform && other.is_block_uniform;
    requires_hoist = requires_hoist || other.requires_hoist;
  }
};

/*!
 * \brief Analyze whether an if-condition is runtime-dependent and/or uniform
 * across all threads in a block.
 *
 * For sync hoisting decisions we care about two independent properties:
 *
 * 1. Does the condition depend on runtime values such as memory loads?
 * 2. Even if it does, is it still block-uniform, i.e. identical for every
 *    thread in the block?
 *
 * Example:
 * - `token_ids[tx] != -1` is runtime-dependent and non-uniform.
 * - `batch_sizes[bx] > 0` is runtime-dependent but block-uniform.
 *
 * Only runtime-dependent, non-uniform conditions need to force sync hoisting.
 * In addition, some non-uniform threadIdx-only conditions still need hoisting
 * when ThreadPartialSyncRewriter cannot handle them.
 */
class ConditionThreadPropertyChecker : public IRMutatorWithAnalyzer {
public:
  explicit ConditionThreadPropertyChecker(
      arith::Analyzer *analyzer, const Array<IterVar> &env_threads,
      const std::unordered_map<const VarNode *, ConditionThreadProperty>
          &let_var_properties,
      int warp_size = 32)
      : IRMutatorWithAnalyzer(analyzer), env_threads_(env_threads),
        let_var_properties_(let_var_properties), warp_size_(warp_size) {}

  /*!
   * \brief Analyze condition properties relevant to thread-sync hoisting.
   */
  ConditionThreadProperty AnalyzeExpr(const PrimExpr &expr) {
    current_ = ConditionThreadProperty();
    this->VisitExpr(expr);
    return current_;
  }

  ConditionThreadProperty AnalyzeCondition(const PrimExpr &expr,
                                           const IterVar &iv) {
    current_ = ConditionThreadProperty();
    this->VisitExpr(expr);
    auto extent_opt = as_const_int(iv->dom->extent);
    ICHECK(extent_opt != nullptr)
        << "AnalyzeCondition: thread extent must be a "
           "constant, but got: "
        << iv->dom->extent;
    int64_t thread_extent = *extent_opt;
    {
      With<arith::ConstraintContext> ctx(analyzer_, expr);
      auto count = analyzer_->z3_prover.CountSatisfyingValues(
          iv->var, thread_extent, /*min_consecutive=*/warp_size_);
      if (count < 0) {
        // ThreadPartialSyncRewriter cannot safely lower this condition.
        current_.requires_hoist = true;
      }
    }
    return current_;
  }

private:
  StorageScope GetScope(Var buffer_var) const {
    return StorageScope::Create(GetPtrStorageScope(std::move(buffer_var)));
  }

  bool IsThreadVar(const VarNode *op) const {
    for (const auto &iv : env_threads_) {
      if (iv->var.get() == op &&
          runtime::ThreadScope::Create(iv->thread_tag).rank == 1) {
        return true;
      }
    }
    return false;
  }

  bool IsThreadLocalScope(const StorageScope &scope) const {
    switch (scope.rank) {
    case StorageRank::kWarp:
    case StorageRank::kLocal:
    case StorageRank::kWMMAMatrixA:
    case StorageRank::kWMMAMatrixB:
    case StorageRank::kWMMAAccumulator:
    case StorageRank::kAMXTMM:
    case StorageRank::kMMAMatrixA:
    case StorageRank::kMMAMatrixB:
    case StorageRank::kMMAMatrixC:
    case StorageRank::kMetalSimdGroup:
      return true;
    case StorageRank::kGlobal:
    case StorageRank::kShared:
    case StorageRank::kTexture:
      return false;
    }
    return false;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    if (IsThreadVar(op)) {
      current_.is_block_uniform = false;
    }
    auto it = let_var_properties_.find(op);
    if (it != let_var_properties_.end()) {
      current_.Merge(it->second);
    }
    return GetRef<Var>(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    current_.depends_on_runtime = true;
    // Do not mark local-scope loads as non-block-uniform solely based on
    // storage scope.  Thread-local buffers (fragments) commonly hold
    // block-uniform data when populated from block-uniform global addresses
    // (e.g., T.copy(BlockMask[blockIdx.y, :], fragment)).  If the load
    // indices actually depend on threadIdx, the recursive visit of indices
    // below (via IRMutatorWithAnalyzer::VisitExpr_) will correctly set
    // is_block_uniform = false through VisitExpr_(VarNode*).
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr()) ||
        op->op.same_as(builtin::address_of())) {
      current_.depends_on_runtime = true;
      // Do not mark local-scope tvm_access_ptr loads as non-block-uniform
      // solely based on storage scope.  Thread-local buffers (fragments)
      // commonly hold block-uniform data when populated from block-uniform
      // global addresses (e.g., a per-thread fragment that every thread
      // fills with the same global value).  If the access indices actually
      // depend on threadIdx, the recursive visit of args below (via
      // IRMutatorWithAnalyzer::VisitExpr_) will correctly mark the
      // condition as non-block-uniform through VisitExpr_(VarNode*).
      //
      // Mirrors the BufferLoadNode handling above(#90299d68); without this,
      // conditions like `T.any_of(local_fragment[:])` get hoisted out of the
      // if-body and break write-before-read sync between shared-memory loads
      // and mma/tma reads.
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

private:
  ConditionThreadProperty current_;
  const Array<IterVar> &env_threads_;
  const std::unordered_map<const VarNode *, ConditionThreadProperty>
      &let_var_properties_;
  int warp_size_;
};

struct TileLangThreadSyncPlanner : public ConstrVisitor {
  explicit TileLangThreadSyncPlanner(StorageScope sync_scope,
                                     int warp_size = 32,
                                     bool debug_summary = false)
      : sync_scope_(std::move(sync_scope)), warp_size_(warp_size),
        debug_summary_(debug_summary) {
    scope_.push_back(std::vector<StmtEntry>());
  }

  using AccessEntry = shared_access_analysis::AccessEntry;
  using AccessType = shared_access_analysis::AccessType;
  static constexpr AccessType kRead = shared_access_analysis::kRead;
  static constexpr AccessType kWrite = shared_access_analysis::kWrite;
  static constexpr AccessType kSync = shared_access_analysis::kSync;
  static constexpr AccessType kAlloc = shared_access_analysis::kAlloc;
  static constexpr AccessType kReadAcquire =
      shared_access_analysis::kReadAcquire;
  using StmtEntry = shared_access_analysis::StmtEntry;
  // access scope
  std::vector<std::vector<StmtEntry>> scope_;
  StorageScope GetScope(Var buffer_var) const {
    return StorageScope::Create(GetPtrStorageScope(std::move(buffer_var)));
  }
  IterVar GetThreadVar(const std::string &tag) const {
    for (const auto &iv : env_threads_) {
      if (iv->thread_tag == tag) {
        return iv;
      }
    }
    LOG(FATAL) << "Thread variable " << tag << " not found";
    return IterVar();
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Var buf = op->buffer->data;
    buffer_data_to_buffer_.Set(tvm::ffi::GetRef<Var>(buf.get()), op->buffer);
    StorageScope scope = GetScope(buf);
    if (Enabled(buf.get(), scope)) {
      ICHECK(allow_append_)
          << tvm::ffi::GetRef<BufferLoad>(op) << " " << scope.to_string();
      AccessEntry e{.cset = {constr_stack_}};
      e.threads = env_threads();
      e.buffer = buf;
      e.buffer_name = op->buffer;
      e.buffer_indices = op->indices;
      e.dtype = op->dtype.element_of();
      for (const auto &index : op->indices) {
        e.touched.push_back(arith::IntSet::Vector(index));
      }
      e.type = kRead;
      e.scope = scope;
      curr_stmt_.access.emplace_back(std::move(e));
    }
    // traverse child
    ConstrVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    allow_append_ = true;
    ICHECK_EQ(curr_stmt_.access.size(), 0U);
    curr_stmt_.stmt = op;

    Var buf = op->buffer->data;
    buffer_data_to_buffer_.Set(tvm::ffi::GetRef<Var>(buf.get()), op->buffer);
    StorageScope scope = GetScope(buf);
    if (Enabled(buf.get(), scope)) {
      AccessEntry e{.cset = {constr_stack_}};
      e.threads = env_threads();
      e.buffer = buf;
      e.buffer_name = op->buffer;
      e.buffer_indices = op->indices;
      e.dtype = op->value.dtype().element_of();
      for (const auto &index : op->indices) {
        e.touched.push_back(arith::IntSet::Vector(index));
      }
      e.type = kWrite;
      e.scope = scope;
      curr_stmt_.access.emplace_back(std::move(e));
    }
    // traverse child
    ConstrVisitor::VisitStmt_(op);
    // push to the scope
    scope_.back().push_back(curr_stmt_);
    // clear access entry.
    curr_stmt_.access.clear();
    allow_append_ = false;
  }
  void VisitStmt_(const EvaluateNode *op) final {
    allow_append_ = true;
    ICHECK_EQ(curr_stmt_.access.size(), 0U);
    curr_stmt_.stmt = op;
    ConstrVisitor::VisitStmt_(op);
    // push to the scope
    if (!curr_stmt_.access.empty()) {
      scope_.back().push_back(curr_stmt_);
      curr_stmt_.access.clear();
    }
    allow_append_ = false;
  }

  void VisitStmt_(const LetStmtNode *op) final {
    allow_append_ = true;
    ICHECK_EQ(curr_stmt_.access.size(), 0U);
    curr_stmt_.stmt = op;
    this->VisitExpr(op->value);
    // push to the scope
    scope_.back().push_back(curr_stmt_);
    // clear access entry.
    curr_stmt_.access.clear();
    allow_append_ = false;
    // traverse body block
    {
      auto let_prop = AnalyzeExprProperty(op->value);
      auto it = let_var_properties_.find(op->var.get());
      bool had_prev = it != let_var_properties_.end();
      ConditionThreadProperty prev_prop;
      if (had_prev) {
        prev_prop = it->second;
      }
      let_var_properties_[op->var.get()] = let_prop;
      auto guard = MakeGuard(op->var, op->value);
      this->VisitStmt(op->body);
      if (had_prev) {
        let_var_properties_[op->var.get()] = prev_prop;
      } else {
        let_var_properties_.erase(op->var.get());
      }
    }
  }
  void VisitStmt_(const BlockNode *op) final {
    auto block = Downcast<Block>(op);
    for (const auto &buffer : block->alloc_buffers) {
      ICHECK(buffer->IsInstance<BufferNode>());
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    ConstrVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == tvm::tir::attr::coproc_scope) {
      IterVar iv = Downcast<IterVar>(op->node);
      env_threads_.push_back(iv);
      ConstrVisitor::VisitStmt_(op);
      env_threads_.pop_back();
    } else if (op->attr_key == tvm::tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      env_threads_.push_back(iv);
      ICHECK_NE(iv->thread_tag.length(), 0U);

      if (!in_device_env_) {
        in_device_env_ = true;
        scope_.push_back(std::vector<StmtEntry>());
        ConstrVisitor::VisitStmt_(op);
        // no need to take the result as the thread barrier automatically syncs.
        Summarize(std::move(scope_.back()), nullptr);
        in_device_env_ = false;
        scope_.pop_back();
      } else {
        ConstrVisitor::VisitStmt_(op);
      }
      env_threads_.pop_back();
    } else {
      ConstrVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const ForNode *op) final {
    scope_.push_back(std::vector<StmtEntry>());
    ConstrVisitor::VisitStmt_(op);
    StmtEntry s;
    s.stmt = op;
    s.access = Summarize(std::move(scope_.back()), op);
    scope_.pop_back();
    if (!s.access.empty()) {
      // relax the touched set to contain all ranges in the loop.
      std::unordered_map<const VarNode *, arith::IntSet> relax_map;
      relax_map[op->loop_var.get()] =
          arith::IntSet::FromRange(Range::FromMinExtent(op->min, op->extent));
      for (AccessEntry &e : s.access) {
        if (e.buffer.defined()) {
          ICHECK(!e.touched.empty());
          Array<arith::IntSet> new_touched;
          for (const auto &touched : e.touched) {
            new_touched.push_back(arith::EvalSet(touched, relax_map));
          }
          e.touched = std::move(new_touched);
        }
      }
    }
    if (!s.access.empty()) {
      scope_.back().emplace_back(std::move(s));
    }
  }

  /**
   * @brief Visit an IfThenElse statement and collect storage access summaries
   * for its branches.
   *
   * Visits the if-then-else node's condition and both branches to summarize
   * buffer reads, writes, and synchronization events under the condition's
   * constraints.
   *
   * IMPORTANT: If syncs are inserted inside an if-statement with a non-uniform
   * condition (i.e., the condition depends on threadIdx), we must hoist the
   * sync to before the if-statement. Otherwise, only some threads will reach
   * the sync point, causing a deadlock.
   */
  void VisitStmt_(const IfThenElseNode *op) final {
    StmtEntry s;
    // Track syncs inserted before visiting the if body
    std::unordered_set<const Object *> syncs_before_then;
    std::unordered_set<const Object *> syncs_before_else;
    for (const auto &sync : syncs_inserted_) {
      syncs_before_then.insert(sync);
    }

    {
      auto guard = MakeGuard(op->condition);
      allow_append_ = true;
      this->VisitExpr(op->condition);

      // Preserve accesses collected from the condition expression so they
      // participate in dependency analysis. Otherwise, a write to shared memory
      // immediately followed by an if-condition reading that memory would not
      // trigger a sync before the if-statement.
      std::vector<AccessEntry> cond_access = std::move(curr_stmt_.access);
      allow_append_ = false;

      scope_.push_back(std::vector<StmtEntry>());
      {
        this->VisitStmt(op->then_case);
      }

      s.stmt = op;
      s.access = Summarize(std::move(scope_.back()), nullptr);
      scope_.pop_back();
      // Merge the condition's access summary into the if-statement's access
      // list so the planner can insert a sync before the if when necessary.
      if (!cond_access.empty()) {
        s.access.insert(s.access.begin(), cond_access.begin(),
                        cond_access.end());
      }
    }

    // Track syncs inserted after visiting then branch
    for (const auto &sync : syncs_inserted_) {
      syncs_before_else.insert(sync);
    }

    if (op->else_case) {
      auto guard = MakeGuard(tir::Not(op->condition));
      scope_.push_back(std::vector<StmtEntry>());
      {
        this->VisitStmt(op->else_case.value());
      }
      auto v = Summarize(std::move(scope_.back()), nullptr);
      scope_.pop_back();
      s.access.insert(s.access.end(), v.begin(), v.end());
    }

    // Check if any syncs were inserted inside the if-then-else
    std::vector<const Object *> syncs_in_then;
    std::vector<const Object *> syncs_in_else;

    for (const auto &sync : syncs_inserted_) {
      if (syncs_before_then.count(sync) == 0 &&
          syncs_before_else.count(sync) != 0) {
        // Sync was inserted during then branch processing
        syncs_in_then.push_back(sync);
      } else if (syncs_before_else.count(sync) == 0) {
        // Sync was inserted during else branch processing
        syncs_in_else.push_back(sync);
      }
    }

    bool has_syncs_inside = !syncs_in_then.empty() || !syncs_in_else.empty();

    if (has_syncs_inside) {
      // Runtime-dependent conditions are only problematic when they can differ
      // across threads in the same block. Block-uniform runtime conditions
      // such as `batch_sizes[blockIdx.z] > 0` are safe to keep in-place.
      //
      // Separately, some threadIdx-only non-uniform conditions still need
      // hoisting when ThreadPartialSyncRewriter cannot lower them safely.
      arith::Analyzer analyzer;
      ConstrSet constr_set = GetConstrSet();
      constr_set.Populate(analyzer);
      ConditionThreadPropertyChecker checker(&analyzer, env_threads_,
                                             let_var_properties_, warp_size_);
      IterVar tx = GetThreadVar("threadIdx.x");
      auto condition_prop = checker.AnalyzeCondition(op->condition, tx);

      if ((condition_prop.depends_on_runtime &&
           !condition_prop.is_block_uniform) ||
          condition_prop.requires_hoist) {
        LOG(WARNING)
            << "[ThreadSync] Hoisting sync from inside if to before if. "
            << "Condition is not safe for in-if sync: " << op->condition;
        for (const auto &sync : syncs_in_then) {
          syncs_inserted_.erase(sync);
        }
        for (const auto &sync : syncs_in_else) {
          syncs_inserted_.erase(sync);
        }

        // Insert sync before the if-statement itself
        insert_syncs(op);
      }
    }

    scope_.back().emplace_back(std::move(s));
  }

  void VisitStmt_(const WhileNode *op) final {
    StmtEntry s;
    {
      auto guard = MakeGuard(op->condition);
      allow_append_ = true;
      this->VisitExpr(op->condition);
      std::vector<AccessEntry> cond_access = std::move(curr_stmt_.access);
      allow_append_ = false;

      scope_.push_back(std::vector<StmtEntry>());
      {
        this->VisitStmt(op->body);
      }
      s.stmt = op;
      s.access = Summarize(std::move(scope_.back()), nullptr);
      scope_.pop_back();
      if (!cond_access.empty()) {
        s.access.insert(s.access.begin(), cond_access.begin(),
                        cond_access.end());
      }
    }
    scope_.back().emplace_back(std::move(s));
  }

  void VisitExpr_(const CallNode *op) final {
    // Mark async TMA load context so that tvm_access_ptr within the call
    // can be tagged accordingly.
    auto is_tma_load = [&]() {
      if (auto opt = op->op.as<Op>()) {
        const Op &call_op = opt.value();
        return call_op.same_as(tl::tma_load()) ||
               call_op.same_as(tl::tma_load_im2col());
      }
      return false;
    }();
    if (is_tma_load) {
      tma_depth_++;
      for (const auto &a : op->args) {
        this->VisitExpr(a);
      }
      tma_depth_--;
      return;
    }

    // Mark async cp.async load context so that tvm_access_ptr within the call
    // can be tagged accordingly. This allows the sync planner to avoid
    // inserting unnecessary barriers between back-to-back cp.async writes.
    auto is_cp_async = [&]() {
      if (auto opt = op->op.as<Op>()) {
        const Op &call_op = opt.value();
        return call_op.same_as(builtin::ptx_cp_async()) ||
               call_op.same_as(tl::ptx_cp_async());
      }
      return false;
    }();
    if (is_cp_async) {
      cp_async_depth_++;
      for (const auto &a : op->args) {
        this->VisitExpr(a);
      }
      cp_async_depth_--;
      return;
    }

    // Mark the pointer argument of atomic ops as atomic so the sync planner
    // doesn't insert barriers between atomics.
    auto is_atomic_op = [&]() {
      if (auto opt = op->op.as<Op>()) {
        const Op &call_op = opt.value();
        return call_op.same_as(tl::atomic_add_elem_op()) ||
               call_op.same_as(tl::atomic_add_ret_elem_op()) ||
               call_op.same_as(tl::atomic_addx2_elem_op()) ||
               call_op.same_as(tl::atomic_addx4_elem_op()) ||
               call_op.same_as(tl::atomic_load_elem_op()) ||
               call_op.same_as(tl::atomic_store_elem_op()) ||
               call_op.same_as(tl::atomic_max_elem_op()) ||
               call_op.same_as(tl::atomic_max_ret_elem_op()) ||
               call_op.same_as(tl::atomic_min_elem_op()) ||
               call_op.same_as(tl::atomic_min_ret_elem_op());
      }
      return false;
    }();
    if (is_atomic_op) {
      if (!op->args.empty()) {
        atomic_dst_ptr_depth_++;
        this->VisitExpr(op->args[0]);
        atomic_dst_ptr_depth_--;
        for (size_t i = 1; i < op->args.size(); ++i) {
          this->VisitExpr(op->args[i]);
        }
      }
      return;
    }
    if (op->op.same_as(builtin::address_of())) {
      ICHECK_EQ(op->args.size(), 1U);
      if (auto load = op->args[0].as<BufferLoadNode>()) {
        Buffer buffer = load->buffer;
        DataType dtype = buffer->dtype;
        const VarNode *buffer_var = buffer->data.as<VarNode>();
        buffer_data_to_buffer_.Set(tvm::ffi::GetRef<Var>(buffer_var), buffer);
        StorageScope scope = GetScope(tvm::ffi::GetRef<Var>(buffer_var));
        Array<Range> buffer_ranges;
        // from indices to buffer indices
        ICHECK(buffer->shape.size() == load->indices.size());
        // Use buffer shape and indices to compute the buffer_ranges for each
        // dimension.
        for (size_t i = 0; i < buffer->shape.size(); ++i) {
          PrimExpr min = load->indices[i];
          PrimExpr extent = make_const(buffer->shape[i].dtype(), 1);
          buffer_ranges.push_back(Range::FromMinExtent(min, extent));
        }
        if (Enabled(buffer_var, scope)) {
          ICHECK(allow_append_);
          AccessEntry e{.cset = {constr_stack_}};
          e.threads = env_threads();
          e.dtype = dtype;
          e.buffer = Downcast<Var>(buffer->data);
          e.buffer_name = buffer;
          e.buffer_ranges = buffer_ranges;
          for (const auto &index : load->indices) {
            e.touched.push_back(arith::IntSet::Vector(index));
          }
          e.is_pointer_access = true;
          e.is_atomic = (atomic_dst_ptr_depth_ > 0);
          e.type = kRead;
          e.scope = scope;
          curr_stmt_.access.emplace_back(e);
        }
        ConstrVisitor::VisitExpr_(load);
      } else {
        ConstrVisitor::VisitExpr_(op);
      }
    } else if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      DataType dtype = op->args[0].dtype();
      const VarNode *buffer_var = op->args[1].as<VarNode>();
      PrimExpr offset = op->args[2];
      PrimExpr extent = op->args[3];
      const IntImmNode *flag = op->args[4].as<IntImmNode>();
      StorageScope scope = GetScope(tvm::ffi::GetRef<Var>(buffer_var));
      // The buffer scope.
      if (Enabled(buffer_var, scope)) {
        ICHECK(allow_append_);
        Array<Range> buffer_ranges;
        if (buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(buffer_var)) ==
            buffer_data_to_buffer_.end()) {
          // cannot find buffer map, use the default buffer
          buffer_ranges = {Range::FromMinExtent(offset, extent)};
        } else {
          Buffer buffer =
              buffer_data_to_buffer_.at(tvm::ffi::GetRef<Var>(buffer_var));
          auto buffer_shape = buffer->shape;
          // convert 1d offset to multi-dimensional index
          auto linear_to_indices = [](PrimExpr offset,
                                      const Array<PrimExpr> &shape) {
            Array<PrimExpr> indices;
            DataType index_dtype = offset.dtype();
            ICHECK(index_dtype.is_int() || index_dtype.is_uint())
                << "Expected integer offset dtype in tvm_access_ptr, but got "
                << index_dtype;
            PrimExpr remaining = std::move(offset);
            for (size_t i = 0; i < shape.size(); ++i) {
              PrimExpr stride = make_const(index_dtype, 1);
              for (size_t j = i + 1; j < shape.size(); ++j) {
                PrimExpr dim = shape[j];
                if (dim.dtype() != index_dtype) {
                  dim = tir::Cast(index_dtype, dim);
                }
                stride = stride * dim;
              }
              PrimExpr idx = FloorDiv(remaining, stride);
              remaining = FloorMod(remaining, stride);
              indices.push_back(idx);
            }
            return indices;
          };
          Array<PrimExpr> start_indices =
              linear_to_indices(offset, buffer_shape);
          Array<PrimExpr> end_indices =
              linear_to_indices(offset + extent, buffer_shape);
          for (size_t i = 0; i < buffer_shape.size(); ++i) {
            buffer_ranges.push_back(Range::FromMinExtent(
                start_indices[i], end_indices[i] - start_indices[i]));
          }
        }
        AccessEntry e{.cset = {constr_stack_}};
        e.threads = env_threads();
        e.dtype = dtype;
        e.buffer = tvm::ffi::GetRef<Var>(buffer_var);
        e.buffer_ranges = buffer_ranges;
        e.is_pointer_access = true;
        e.is_atomic = (atomic_dst_ptr_depth_ > 0);
        e.touched = {
            arith::IntSet::FromRange(Range::FromMinExtent(offset, extent))};
        e.scope = scope;
        if (flag->value & 1) {
          e.type = kRead;
          e.is_async_copy = (tma_depth_ > 0 || cp_async_depth_ > 0);
          curr_stmt_.access.emplace_back(e);
        }
        if (flag->value & 2) {
          e.type = kWrite;
          e.is_async_copy = (tma_depth_ > 0 || cp_async_depth_ > 0);
          curr_stmt_.access.emplace_back(e);
        }
      }
      ConstrVisitor::VisitExpr_(op);
    } else if (op->op.same_as(builtin::tvm_storage_sync())) {
      ICHECK(allow_append_);
      const std::string &s = op->args[0].as<StringImmNode>()->value;
      if (s != "warp" && s != "cluster") {
        StorageScope scope = StorageScope::Create(s);
        AccessEntry e{.cset = {constr_stack_}};
        e.threads = env_threads();
        e.type = kSync;
        e.scope = StorageScope::Create(s);
        curr_stmt_.access.emplace_back(std::move(e));
      }
    } else {
      ConstrVisitor::VisitExpr_(op);
    }
  }

  void SetBufferDataToBuffer(const Var &buffer_var, const Buffer &buffer) {
    buffer_data_to_buffer_.Set(buffer_var, buffer);
  }

  std::vector<AccessEntry> Summarize(std::vector<StmtEntry> seq,
                                     const ForNode *loop) {
    std::vector<StmtEntry> debug_seq;
    shared_access_analysis::SequenceSummaryResult debug_result;
    if (debug_summary_) {
      debug_seq = seq;
      debug_result = shared_access_analysis::SummarizeAccessSequence(
          debug_seq, loop, sync_scope_, this->env_threads(),
          ConstrSet{.constrs_ = constr_stack_}, syncs_inserted_,
          /*coalesce_dynamic_shared_buffers=*/false);
    }

    auto result = shared_access_analysis::SummarizeAccessSequence(
        std::move(seq), loop, sync_scope_, this->env_threads(),
        ConstrSet{.constrs_ = constr_stack_}, syncs_inserted_,
        /*coalesce_dynamic_shared_buffers=*/true);
    if (debug_summary_) {
      LogSummary(debug_seq, debug_result, result, loop);
    }
    for (const Object *stmt : result.sync_before_stmts) {
      insert_syncs(stmt);
    }
    return std::move(result.exposed_accesses);
  }
  // The syncs inserted before each statement
  std::unordered_set<const Object *> syncs_inserted_;
  const Array<IterVar> &env_threads() const { return env_threads_; }

private:
  ConditionThreadProperty AnalyzeExprProperty(const PrimExpr &expr) const {
    arith::Analyzer analyzer;
    ConstrSet constr_set = GetConstrSet();
    constr_set.Populate(analyzer);
    ConditionThreadPropertyChecker checker(&analyzer, env_threads_,
                                           let_var_properties_, warp_size_);
    return checker.AnalyzeExpr(expr);
  }

  bool Enabled(const VarNode *buf, const StorageScope &scope) {
    return in_device_env() && scope == sync_scope_;
  }
  /*! \return whether we are in device environment. */
  bool in_device_env() const { return in_device_env_; }

  // whether access appending is enabled.
  bool allow_append_{false};
  // Whether we are in device environment
  bool in_device_env_{false};
  // Nesting depth of tma_load/tma_load_im2col calls
  int tma_depth_{0};
  // Nesting depth of cp.async calls (ptx_cp_async)
  int cp_async_depth_{0};
  // Whether we're visiting the pointer argument expression of an atomic call
  // (e.g., atomic_add/atomic_max/atomic_load). When > 0, accesses produced by
  // the pointer metadata ops are tagged as atomic.
  int atomic_dst_ptr_depth_{0};
  // the current free stmt entry.
  StmtEntry curr_stmt_;
  // The involving threads
  Array<IterVar> env_threads_;
  // Thread-uniform/runtime properties for let-bound vars visible in the
  // current lexical scope.
  std::unordered_map<const VarNode *, ConditionThreadProperty>
      let_var_properties_;
  // The buffer map
  Map<Var, Buffer> buffer_data_to_buffer_;
  // synchronization scope
  StorageScope sync_scope_;
  // warp size from target
  int warp_size_;
  bool debug_summary_{false};

  void insert_syncs(const Object *obj) {
    if (syncs_inserted_.count(obj))
      return;
    syncs_inserted_.insert(obj);
  }
  void print_access_tentry(const AccessEntry &access,
                           bool print_constr = false) {
    std::ostringstream output;

    output << "Access Entry Information:\n";
    output << "  Buffer: " << access.buffer << "\n";
    output << "  Buffer Name: " << access.buffer_name << "\n";
    output << "  Data Type: " << access.dtype << "\n";

    std::string type_str;
    switch (access.type) {
    case kRead:
      type_str = "Read";
      break;
    case kWrite:
      type_str = "Write";
      break;
    case kSync:
      type_str = "Sync";
      break;
    case kAlloc:
      type_str = "Alloc";
      break;
    case kReadAcquire:
      type_str = "ReadAcquire";
      break;
    default:
      type_str = "Unknown";
      break;
    }
    output << "  Access Type: " << type_str << "\n";

    output << "  Storage Scope: " << access.scope.to_string() << "\n";

    output << "  Threads: [";
    for (size_t i = 0; i < access.threads.size(); ++i) {
      if (i > 0)
        output << ", ";
      output << access.threads[i]->thread_tag;
    }
    output << "]\n";

    if (print_constr) {
      output << "  Constraint: {";
      arith::Analyzer analyzer_;
      access.cset.Populate(analyzer_);
      output << analyzer_.z3_prover.GetSMTLIB2(std::nullopt);
      output << "}\n";
    }

    output << "  Buffer Indices: [";
    for (size_t i = 0; i < access.buffer_indices.size(); ++i) {
      if (i > 0)
        output << ", ";
      output << access.buffer_indices[i];
    }
    output << "]\n";

    if (!access.buffer_ranges.empty()) {
      output << "  Buffer Ranges: [";
      for (size_t i = 0; i < access.buffer_ranges.size(); ++i) {
        if (i > 0)
          output << ", ";
        output << "[" << access.buffer_ranges[i]->min << ", "
               << access.buffer_ranges[i]->extent << "]";
      }
      output << "]\n";
    }

    if (!access.touched.empty()) {
      output << "  Touched Ranges: [";
      for (size_t i = 0; i < access.touched.size(); ++i) {
        if (i > 0)
          output << ", ";
        output << access.touched[i];
      }
      output << "]\n";
    }

    output << "  Flags: ";
    output << "is_pointer_access="
           << (access.is_pointer_access ? "true" : "false");
    output << ", is_async_copy=" << (access.is_async_copy ? "true" : "false");

    LOG(WARNING) << output.str();
  }
  /*!
   * \brief Check if two access entries conflict, considering loop-carried
   * dependencies.
   *
   * For loop-carry analysis, we use symbolic iteration shift: instead of
   * treating loop_carry as a simple flag, we substitute loop_var with
   * loop_var + step in the "next iteration" access indices and check if they
   * overlap with the "current iteration" access indices.
   *
   * This approach can prove that accesses like A[i] and A[i+1] are disjoint
   * (no loop-carry dependency), while correctly detecting dependencies like
   * A[i] and A[i-1] (loop-carry dependency with distance 1).
   *
   * \param prev The access entry from the previous/current iteration
   * \param curr The access entry to check against
   * \param loop The loop node for loop-carry analysis, nullptr for
   * same-iteration
   * \return true if the accesses conflict and need synchronization
   */
  bool FindConflict(const AccessEntry &prev, const AccessEntry &curr,
                    const ForNode *loop) {
    return shared_access_analysis::FindConflict(prev, curr, loop);
  }

  bool FindConflict(const std::vector<AccessEntry> &prev,
                    const AccessEntry &curr, const ForNode *loop) {
    return shared_access_analysis::FindConflict(prev, curr, loop);
  }

  std::string AccessToString(const AccessEntry &access) const {
    std::ostringstream os;
    switch (access.type) {
    case kRead:
      os << "R:";
      break;
    case kWrite:
      os << "W:";
      break;
    case kSync:
      os << "SYNC:";
      break;
    case kAlloc:
      os << "ALLOC:";
      break;
    case kReadAcquire:
      os << "RA:";
      break;
    }
    if (access.type == kSync) {
      os << access.scope.to_string();
      return os.str();
    }
    if (access.buffer_name.defined()) {
      os << access.buffer_name->name;
    } else if (access.buffer.defined()) {
      os << access.buffer->name_hint;
    } else {
      os << "<anon>";
    }
    if (access.is_pointer_access) {
      os << "[ptr]";
    }
    if (access.is_async_copy) {
      os << "[async]";
    }
    if (access.is_atomic) {
      os << "[atomic]";
    }
    return os.str();
  }

  std::string AccessVecToString(const std::vector<AccessEntry> &accesses) const {
    if (accesses.empty()) {
      return "<empty>";
    }
    std::ostringstream os;
    for (size_t i = 0; i < accesses.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << AccessToString(accesses[i]);
    }
    return os.str();
  }

  void LogSummary(const std::vector<StmtEntry> &seq,
                  const shared_access_analysis::SequenceSummaryResult &raw_result,
                  const shared_access_analysis::SequenceSummaryResult &coalesced_result,
                  const ForNode *loop) const {
    bool has_stmt_access = false;
    for (const StmtEntry &entry : seq) {
      if (!entry.access.empty()) {
        has_stmt_access = true;
        break;
      }
    }
    if (!has_stmt_access && raw_result.exposed_accesses.empty() &&
        coalesced_result.exposed_accesses.empty() &&
        coalesced_result.sync_before_stmts.empty()) {
      return;
    }

    std::ostringstream os;
    os << "[ThreadSyncSummary] scope=" << sync_scope_.to_string();
    if (loop != nullptr) {
      os << " loop=" << loop->loop_var->name_hint;
    } else {
      os << " loop=<none>";
    }
    os << " seq_len=" << seq.size() << "\n";
    for (size_t i = 0; i < seq.size(); ++i) {
      os << "  stmt[" << i << "] " << seq[i].stmt->GetTypeKey() << ": "
         << AccessVecToString(seq[i].access) << "\n";
    }
    os << "  raw_exposed: " << AccessVecToString(raw_result.exposed_accesses)
       << "\n";
    os << "  coalesced_exposed: "
       << AccessVecToString(coalesced_result.exposed_accesses) << "\n";
    if (!coalesced_result.sync_before_stmts.empty()) {
      os << "  inserted_syncs:";
      for (const Object *stmt : coalesced_result.sync_before_stmts) {
        os << ' ' << stmt->GetTypeKey();
      }
      os << "\n";
    }
    LOG(INFO) << os.str();
  }
};

PrimFunc TileLangThreadSync(PrimFunc func, const std::string &storage_scope,
                            bool debug_summary = false) {
  StorageScope sync_scope = StorageScope::Create(storage_scope);
  auto *n = func.CopyOnWrite();
  auto stmt = n->body;
  if (sync_scope.rank == StorageRank::kShared && sync_scope.tag.empty()) {
    stmt = ThreadSyncAfterWaitGroupInserter(sync_scope)(stmt);
  }
  // Get warp size from target, defaulting to 32 if not available
  int warp_size = 32;
  if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
    warp_size = target.value()
                    ->GetAttr<Integer>("thread_warp_size", 32)
                    .value()
                    .IntValue();
  }
  TileLangThreadSyncPlanner planner(sync_scope, warp_size, debug_summary);
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
    bool debug_summary = ctx
                             ->GetConfig(::tvm::tl::kDebugThreadStorageSyncSummary,
                                         Bool(false))
                             .value()
                             ->value;
    return tl::TileLangThreadSync(std::move(f), storage_scope, debug_summary);
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
