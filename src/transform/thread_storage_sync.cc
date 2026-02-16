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
#include "./common/thread_sync_types.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
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

/*!
 * \brief Check if an if-condition depends on runtime-variable values.
 *
 * For sync hoisting decisions, we distinguish two types of non-uniform
 * conditions:
 *
 * 1. Conditions that only depend on threadIdx (e.g., `threadIdx.x >= 512`):
 *    - The number of threads entering the if can be determined at compile time
 *    - ThreadPartialSyncRewriter can handle this by computing the thread count
 *    - No need to hoist sync
 *
 * 2. Conditions that depend on runtime values (e.g., `shared_mem[tx] != -1`):
 *    - Cannot determine at compile time how many threads will enter
 *    - Must hoist sync to before the if to avoid potential deadlock
 *
 * This checker identifies case (2) - conditions that depend on runtime values.
 */
class RuntimeDependentConditionChecker : public IRMutatorWithAnalyzer {
public:
  explicit RuntimeDependentConditionChecker(arith::Analyzer *analyzer,
                                            int warp_size = 32)
      : IRMutatorWithAnalyzer(analyzer), warp_size_(warp_size) {}

  /*!
   * \brief Check if expression depends on runtime-variable values.
   * \return true if the expression depends on values that cannot be determined
   *         at compile time (e.g., shared memory loads), false if it only
   *         depends on compile-time known values (constants, threadIdx,
   * blockIdx).
   */
  bool DependsOnRuntimeValue(const PrimExpr &expr, const IterVar &iv) {
    depends_on_runtime_ = false;
    this->VisitExpr(expr);
    auto extent_opt = as_const_int(iv->dom->extent);
    ICHECK(extent_opt != nullptr)
        << "DependsOnRuntimeValue: thread extent must be a "
           "constant, but got: "
        << iv->dom->extent;
    int64_t thread_extent = *extent_opt;
    {
      With<arith::ConstraintContext> ctx(analyzer_, expr);
      auto count = analyzer_->z3_prover.CountSatisfyingValues(
          iv->var, thread_extent, /*min_consecutive=*/warp_size_);
      if (count < 0) {
        // failed to count satisfying values, return true
        depends_on_runtime_ = true;
      }
    }
    return depends_on_runtime_;
  }

private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    // Any buffer load introduces runtime dependency
    // (we don't know the buffer contents at compile time)
    depends_on_runtime_ = true;
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    // Check tvm_access_ptr and address_of - if used in condition, it's reading
    // memory
    if (op->op.same_as(builtin::tvm_access_ptr()) ||
        op->op.same_as(builtin::address_of())) {
      depends_on_runtime_ = true;
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    // Other calls might also introduce runtime dependency
    // but we'll be conservative and check children
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

private:
  bool depends_on_runtime_{false};
  int warp_size_;
};

struct TileLangThreadSyncPlanner : public ConstrVisitor {
  explicit TileLangThreadSyncPlanner(StorageScope sync_scope,
                                     int warp_size = 32)
      : sync_scope_(std::move(sync_scope)), warp_size_(warp_size) {
    scope_.push_back(std::vector<StmtEntry>());
  }

  /*! \brief Storage access type */
  enum AccessType : uint8_t {
    kRead,
    kWrite,
    kSync,
    kAlloc,
    // acquired version of read, only need to handle WAR dep.
    kReadAcquire
  };
  /*! \brief An access entry */
  struct AccessEntry {
    /*! \brief The thread index that access this entry */
    Array<IterVar> threads;
    /*! \brief The buffer variable, if any */
    Array<PrimExpr> buffer_indices;
    ConstrSet cset;
    /*! \brief The buffer ranges for pointer access */
    Array<Range> buffer_ranges;
    Var buffer = NullValue<Var>();
    Buffer buffer_name;
    /*! \brief The access data type */
    DataType dtype;
    /*! \brief The touched access range
     *
     * Has one IntSet for each index in the buffer being accessed.
     */
    Array<arith::IntSet> touched;
    /*! \brief The type of access */
    AccessType type;
    /*! \brief The storage scope */
    StorageScope scope;
    /*! \brief Whether the access is pointer access */
    bool is_pointer_access = false;
    /*! \brief Whether this access originates from an async copy context
     *         (e.g., inside a TMA load) and therefore multiple writes
     *         among themselves should not force barriers between them. */
    bool is_async_copy = false;
    /*! \brief Whether this access is part of an atomic RMW (e.g., atomic_add).
     *
     * If both sides of a dependency are atomic, we should not insert a thread
     * barrier between them. A barrier would artificially serialize atomics
     * across threads (and can even change observable results for atomics with
     * return values).
     */
    bool is_atomic = false;
  };
  /*! \brief Access pattern about a single statement */
  struct StmtEntry {
    /*! \brief The statement */
    const Object *stmt{};
    /*! \brief access patterns in the statement */
    std::vector<AccessEntry> access;
  };
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
      auto guard = MakeGuard(op->var, op->value);
      this->VisitStmt(op->body);
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
      // Check if the condition depends on runtime values (e.g., shared memory
      // loads). If so, we cannot determine at compile time how many threads
      // will enter the if, so we must hoist the sync to before the if to avoid
      // potential deadlock.
      //
      // If the condition only depends on threadIdx (e.g., `threadIdx.x >=
      // 512`), we use Z3 to check if the thread count is a multiple of 32.
      // If not, ThreadPartialSyncRewriter cannot handle it properly, so we
      // must also hoist the sync.
      arith::Analyzer analyzer;
      ConstrSet constr_set = GetConstrSet();
      constr_set.Populate(analyzer);
      RuntimeDependentConditionChecker checker(&analyzer, warp_size_);
      IterVar tx = GetThreadVar("threadIdx.x");
      bool depends_on_runtime =
          checker.DependsOnRuntimeValue(op->condition, tx);

      if (depends_on_runtime) {
        // Condition depends on runtime values - must hoist sync
        // Condition depends on runtime values - must hoist sync
        LOG(WARNING)
            << "[ThreadSync] Hoisting sync from inside if to before if. "
            << "Condition depends on runtime value: " << op->condition;
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
          auto linear_to_indices = [this](PrimExpr offset,
                                          const Array<PrimExpr> &shape) {
            Array<PrimExpr> indices;
            PrimExpr remaining = std::move(offset);
            for (size_t i = 0; i < shape.size(); ++i) {
              PrimExpr stride = make_const(DataType::Int(32), 1);
              for (size_t j = i + 1; j < shape.size(); ++j) {
                stride = stride * shape[j];
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
          e.is_async_copy = (tma_depth_ > 0);
          curr_stmt_.access.emplace_back(e);
        }
        if (flag->value & 2) {
          e.type = kWrite;
          e.is_async_copy = (tma_depth_ > 0);
          curr_stmt_.access.emplace_back(e);
        }
      }
      ConstrVisitor::VisitExpr_(op);
    } else if (op->op.same_as(builtin::tvm_storage_sync())) {
      ICHECK(allow_append_);
      const std::string &s = op->args[0].as<StringImmNode>()->value;
      if (s != "warp") {
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
          // Same-iteration conflict: loop=nullptr
          if (FindConflict(writes, acc, nullptr)) {
            sync_before_stmt = true;
            break;
          }
        } else if (acc.type == kWrite) {
          // Same-iteration conflict: loop=nullptr
          if (FindConflict(reads, acc, nullptr) ||
              FindConflict(writes, acc, nullptr)) {
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
      // Loop-carried dependency analysis using symbolic iteration shift.
      // We compare accesses at iteration i (end of loop, stored in
      // reads/writes) with accesses at iteration i+1 (beginning of next
      // iteration). By substituting loop_var -> loop_var + step in the "next
      // iteration" indices, we can precisely determine if there's a true
      // dependency.
      //
      // Examples:
      // - A[i] write, A[i] read: No loop-carry (same iteration access)
      // - A[i] write, A[i+1] read: After shift, comparing A[i] vs A[i+1],
      // disjoint
      // - A[i] write, A[i-1] read: After shift, comparing A[i] vs A[i],
      // conflict!
      // - A[i%2] write, A[i%2] read: After shift, comparing A[i%2] vs
      // A[(i+1)%2],
      //   which are disjoint for modulo buffering
      for (size_t i = 0; i < seq.size(); ++i) {
        const StmtEntry &s = seq[i];
        if (syncs_inserted_.count(s.stmt) != 0)
          break;
        if (reads.empty() && writes.empty())
          break;
        bool need_loop_sync = false;
        for (const AccessEntry &acc : s.access) {
          if (acc.type == kRead) {
            // Loop-carry conflict: pass loop for iteration shift analysis
            if (FindConflict(writes, acc, loop)) {
              need_loop_sync = true;
              break;
            }
          } else if (acc.type == kWrite) {
            // Loop-carry conflict: pass loop for iteration shift analysis
            if (FindConflict(reads, acc, loop) ||
                FindConflict(writes, acc, loop)) {
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
    AccessEntry esync{.cset = {constr_stack_}};
    esync.threads = this->env_threads();
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
    return head;
  }
  // The syncs inserted before each statement
  std::unordered_set<const Object *> syncs_inserted_;
  const Array<IterVar> &env_threads() const { return env_threads_; }

private:
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
  // Whether we're visiting the pointer argument expression of an atomic call
  // (e.g., atomic_add/atomic_max/atomic_load). When > 0, accesses produced by
  // the pointer metadata ops are tagged as atomic.
  int atomic_dst_ptr_depth_{0};
  // the current free stmt entry.
  StmtEntry curr_stmt_;
  // The involving threads
  Array<IterVar> env_threads_;
  // The buffer map
  Map<Var, Buffer> buffer_data_to_buffer_;
  // synchronization scope
  StorageScope sync_scope_;
  // warp size from target
  int warp_size_;

  void insert_syncs(const Object *obj) {
    if (syncs_inserted_.count(obj))
      return;
    syncs_inserted_.insert(obj);
  }
  bool PointerAccessIsDisjoint(const AccessEntry &lhs, const AccessEntry &rhs) {
    if (lhs.touched.size() != 1 || rhs.touched.size() != 1) {
      return false;
    }
    ConstrSet prev_cset{lhs.cset};
    ConstrSet curr_cset{rhs.cset};
    arith::Analyzer analyzer;

    struct ThreadVarInfo {
      const char *name_prev;
      const char *name_curr;
    } thread_vars[] = {
        {"tx1", "tx2"},
        {"ty1", "ty2"},
        {"tz1", "tz2"},
    };
    PrimExpr lhs_min = analyzer.Simplify(lhs.touched[0].min());
    PrimExpr lhs_max = analyzer.Simplify(lhs.touched[0].max());
    PrimExpr rhs_min = analyzer.Simplify(rhs.touched[0].min());
    PrimExpr rhs_max = analyzer.Simplify(rhs.touched[0].max());
    for (unsigned idx = 0; idx != 3; ++idx) {
      auto &info = thread_vars[idx];
      Var old_prev_var = lhs.threads[lhs.threads.size() + idx - 3]->var;
      Var old_curr_var = rhs.threads[rhs.threads.size() + idx - 3]->var;
      Var prev_var(info.name_prev, old_prev_var.dtype());
      Var curr_var(info.name_curr, old_curr_var.dtype());
      lhs_min = Substitute(lhs_min, {{old_prev_var, prev_var}});
      lhs_max = Substitute(lhs_max, {{old_prev_var, prev_var}});
      prev_cset = prev_cset.Substitute({{old_prev_var, prev_var}});
      rhs_min = Substitute(rhs_min, {{old_curr_var, curr_var}});
      rhs_max = Substitute(rhs_max, {{old_curr_var, curr_var}});
      curr_cset = curr_cset.Substitute({{old_curr_var, curr_var}});
    }
    prev_cset.Populate(analyzer);
    curr_cset.Populate(analyzer);

    if (analyzer.CanProve(lhs_max < rhs_min,
                          arith::ProofStrength::kSymbolicBound)) {
      return true;
    }
    if (analyzer.CanProve(rhs_max < lhs_min,
                          arith::ProofStrength::kSymbolicBound)) {
      return true;
    }
    return false;
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

    // Atomic ops already provide correctness for concurrent access.
    // Inserting a barrier between atomics is unnecessary and can change
    // program behavior (e.g., atomics with return values).
    if (prev.is_atomic && curr.is_atomic) {
      return false;
    }

    if (prev.buffer_indices.size() != curr.buffer_indices.size()) {
      // They are not the same indices, should be conflict.
      return true;
    }

    if (prev.is_pointer_access || curr.is_pointer_access) {
      // For accesses created via tvm_access_ptr we may still be able to prove
      // disjointness using their byte ranges. If both sides expose a touched
      // interval and we can show they don't overlap, skip the conflict.
      if (prev.is_pointer_access && curr.is_pointer_access &&
          PointerAccessIsDisjoint(prev, curr)) {
        return false;
      }
      // Otherwise fall back to the conservative answer: treat them as
      // overlapping.
      return true;
    }

    // Build substitution map for loop-carry analysis
    // For loop-carry, we compare: Iter(i) vs Iter(i+step)
    // prev represents access at iteration i (end of loop body)
    // curr represents access at iteration i+step (beginning of next iteration)
    ffi::Map<Var, PrimExpr> loop_shift_sub;
    if (loop != nullptr) {
      // Get loop step, default to 1 if not specified
      PrimExpr step = make_const(loop->loop_var.dtype(), 1);
      // Substitute loop_var -> loop_var + step for the "next iteration"
      loop_shift_sub.Set(loop->loop_var, loop->loop_var + step);
    }

    // Check if indices are the same (considering loop shift)
    bool has_same_index = true;
    for (size_t i = 0; i < prev.buffer_indices.size(); i++) {
      const auto &prev_indice = prev.buffer_indices[i];
      PrimExpr curr_indice = curr.buffer_indices[i];

      // For loop-carry, shift the curr index to represent next iteration
      if (loop != nullptr) {
        curr_indice = Substitute(curr_indice, loop_shift_sub);
      }

      if (!ExprDeepEqual()(prev_indice, curr_indice)) {
        has_same_index = false;
        break;
      }
    }

    if (has_same_index) {
      // Use Z3 to check if prev and curr constraints are equivalent.
      // If equivalent, the same set of threads execute both accesses, so no
      // sync is needed.
      PrimExpr prev_constr = prev.cset.ToConjunction();
      PrimExpr curr_constr = curr.cset.ToConjunction();

      arith::Analyzer analyzer;
      for (const auto &iv : prev.threads) {
        if (iv->dom.defined()) {
          analyzer.Bind(iv->var, iv->dom);
        }
      }
      // Add loop variable constraint for loop-carry analysis
      if (loop != nullptr) {
        // For loop-carry analysis, we compare iteration i with iteration i+1.
        // Since i+1 must be a valid iteration, i can only range from min to
        // min+extent-2 (i.e., extent-1 valid pairs instead of extent).
        PrimExpr adjusted_extent =
            loop->extent - make_const(loop->extent.dtype(), 1);
        analyzer.Bind(loop->loop_var,
                      Range::FromMinExtent(loop->min, adjusted_extent));
      }

      // Check P => C: ¬P ∨ C
      bool prev_implies_curr = analyzer.z3_prover.CanProve(
          tir::Or(tir::Not(prev_constr), curr_constr));
      // Check C => P: ¬C ∨ P
      bool curr_implies_prev = analyzer.z3_prover.CanProve(
          tir::Or(tir::Not(curr_constr), prev_constr));

      if (prev_implies_curr && curr_implies_prev) {
        // If constraints are equivalent, they are not in conflict
        return false;
      } else {
        // If constraints are not equivalent, they are in conflict
        return true;
      }
    }

    // Indices are different, need to check if they can overlap
    bool range_is_overlap = true;

    for (size_t i = 0; i < prev.buffer_indices.size(); i++) {
      auto prev_dtype = prev.dtype;
      auto curr_dtype = curr.dtype;

      const auto &prev_indice = prev.buffer_indices[i];
      PrimExpr curr_indice = curr.buffer_indices[i];

      // For loop-carry, shift the curr index to represent next iteration
      if (loop != nullptr) {
        curr_indice = Substitute(curr_indice, loop_shift_sub);
      }

      PrimExpr prev_indice_bytes = prev_indice * prev_dtype.bytes();
      PrimExpr curr_indice_bytes = curr_indice * curr_dtype.bytes();

      ConstrSet prev_cset{prev.cset};
      ConstrSet curr_cset{curr.cset};
      arith::Analyzer analyzer;

      // Add loop variable constraint for loop-carry analysis
      if (loop != nullptr) {
        // For loop-carry analysis, we compare iteration i with iteration i+1.
        // Since i+1 must be a valid iteration, i can only range from min to
        // min+extent-2 (i.e., extent-1 valid pairs instead of extent).
        PrimExpr adjusted_extent =
            loop->extent - make_const(loop->extent.dtype(), 1);
        analyzer.Bind(loop->loop_var,
                      Range::FromMinExtent(loop->min, adjusted_extent));
      }

      // For WAW (Write-after-Write) and RAR (Read-after-Read), we should use
      // the same thread variables because:
      // - WAW: doesn't create true data dependency, only need to check if the
      //   same thread overwrites its own data across iterations
      // - RAR: no dependency at all
      // For RAW (Read-after-Write) and WAR (Write-after-Read), we need to use
      // different thread variables to check cross-thread dependencies.
      bool same_access_type = (prev.type == kWrite && curr.type == kWrite) ||
                              (prev.type == kRead && curr.type == kRead);

      PrimExpr thread_condition = Bool(false);
      ffi::Map<Var, PrimExpr> prev_sub, curr_sub;

      const char *thread_names[] = {"tx", "ty", "tz"};
      for (unsigned idx = 0; idx != 3; ++idx) {
        Var old_prev_var = prev.threads[prev.threads.size() + idx - 3]->var;
        Var old_curr_var = curr.threads[curr.threads.size() + idx - 3]->var;

        if (same_access_type) {
          // For WAW/RAR: use a single shared Var object for both prev and curr
          // This allows the analyzer to see they reference the same thread
          Var shared_var(thread_names[idx], old_prev_var.dtype());
          prev_sub.Set(old_prev_var, shared_var);
          curr_sub.Set(old_curr_var, shared_var);
        } else {
          // For RAW/WAR: use different Var objects to model cross-thread access
          Var prev_var(std::string(thread_names[idx]) + "1",
                       old_prev_var.dtype());
          Var curr_var(std::string(thread_names[idx]) + "2",
                       old_curr_var.dtype());
          thread_condition =
              tir::Or(thread_condition, tir::NE(prev_var, curr_var));
          prev_sub.Set(old_prev_var, prev_var);
          curr_sub.Set(old_curr_var, curr_var);
        }
      }
      if (!same_access_type) {
        analyzer.EnterConstraint(thread_condition);
      }
      prev_cset.Substitute(prev_sub).Populate(analyzer);
      curr_cset.Substitute(curr_sub).Populate(analyzer);
      bool provably_disjoint = false;

      prev_indice_bytes =
          analyzer.Simplify(Substitute(prev_indice_bytes, prev_sub));
      curr_indice_bytes =
          analyzer.Simplify(Substitute(curr_indice_bytes, curr_sub));

      // Handle Ramp expressions by creating a new index variable
      if (const RampNode *prev_ramp = prev_indice_bytes.as<RampNode>()) {
        Var prev_idx("prev_idx", DataType::Int(32));
        analyzer.Bind(prev_idx, Range::FromMinExtent(0, prev_ramp->lanes));
        prev_indice_bytes = prev_ramp->base + prev_idx * prev_ramp->stride;
      }

      if (const RampNode *curr_ramp = curr_indice_bytes.as<RampNode>()) {
        Var curr_idx("curr_idx", DataType::Int(32));
        analyzer.Bind(curr_idx, Range::FromMinExtent(0, curr_ramp->lanes));
        curr_indice_bytes = curr_ramp->base + curr_idx * curr_ramp->stride;
      }

      // Now handle the simplified expressions
      if (prev_indice_bytes.dtype().is_scalar() &&
          curr_indice_bytes.dtype().is_scalar()) {
        if (prev_indice_bytes.dtype() != curr_indice_bytes.dtype()) {
          if (prev_indice_bytes.dtype().bits() <
              curr_indice_bytes.dtype().bits()) {
            prev_indice_bytes =
                tir::Cast(curr_indice_bytes.dtype(), prev_indice_bytes);
          } else {
            curr_indice_bytes =
                tir::Cast(prev_indice_bytes.dtype(), curr_indice_bytes);
          }
        }
        ICHECK(prev_indice_bytes.dtype() == curr_indice_bytes.dtype());
        provably_disjoint =
            analyzer.CanProve(tir::NE(prev_indice_bytes, curr_indice_bytes));
      } else {
        try {
          auto prev_min = analyzer.Simplify(
              Substitute(prev.touched[i].min() * prev_dtype.bytes(), prev_sub));
          auto prev_max = analyzer.Simplify(
              Substitute(prev.touched[i].max() * prev_dtype.bytes(), prev_sub));
          auto curr_min = analyzer.Simplify(
              Substitute(curr.touched[i].min() * curr_dtype.bytes(), curr_sub));
          auto curr_max = analyzer.Simplify(
              Substitute(curr.touched[i].max() * curr_dtype.bytes(), curr_sub));
          provably_disjoint = analyzer.CanProve(analyzer.Simplify(
              tir::Or(prev_min > curr_max, curr_min > prev_max)));
        } catch (const std::exception &e) {
          auto prev_bound = analyzer.const_int_bound(prev_indice_bytes);
          auto curr_bound = analyzer.const_int_bound(curr_indice_bytes);
          if (prev_bound.defined() && curr_bound.defined()) {
            if ((prev_bound->min_value) > (curr_bound->max_value) ||
                (curr_bound->min_value) > (prev_bound->max_value)) {
              range_is_overlap = false;
              break;
            }
          }
        }
        // if (!provably_disjoint) {
        //   LOG(WARNING) << analyzer.z3_prover.GetStats();
        //   LOG(WARNING) <<
        //   analyzer.z3_prover.GetSMTLIB2(tir::Not(tir::Or(prev_min >
        //   curr_max, curr_min > prev_max)));
        // }
      }

      if (provably_disjoint) {
        range_is_overlap = false;
        break;
      }
    }

    return range_is_overlap;
  }

  bool FindConflict(const std::vector<AccessEntry> &prev,
                    const AccessEntry &curr, const ForNode *loop) {
    for (const AccessEntry &x : prev) {
      if (FindConflict(x, curr, loop)) {
        return true;
      }
    }
    return false;
  }
};

PrimFunc TileLangThreadSync(PrimFunc func, const std::string &storage_scope) {
  StorageScope sync_scope = StorageScope::Create(storage_scope);
  auto *n = func.CopyOnWrite();
  auto stmt = n->body;
  if (sync_scope.rank == StorageRank::kShared && sync_scope.tag.empty()) {
    stmt = ThreadSyncAfterWaitQueueInserter(sync_scope)(stmt);
  }
  // Get warp size from target, defaulting to 32 if not available
  int warp_size = 32;
  if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
    warp_size = target.value()
                    ->GetAttr<Integer>("thread_warp_size", 32)
                    .value()
                    .IntValue();
  }
  TileLangThreadSyncPlanner planner(sync_scope, warp_size);
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
