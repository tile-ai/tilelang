#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "common/pipeline_utils.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../target/utils.h"
#include "tvm/ir/expr.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Check whether two regions have intersections.
 * \param region1 The first region.
 * \param region2 The second region.
 * \return Whether region1 and region2 have intersections.
 */
bool MayConflict(const Region &region1, const Region &region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); i++) {
    Range dim1 = region1[i];
    Range dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

class TmemLoadCollector : public StmtExprVisitor {
public:
  TmemLoadCollector() {}

  Buffer result;

private:
  void VisitExpr_(const BufferLoadNode *op) {
    Buffer buf = op->buffer;
    if (buf->data->type_annotation.as<PointerTypeNode>()->storage_scope ==
        "shared") {
      // We only care about shared.tmem buffers
      ICHECK(!result.defined())
          << "TmemLoadCollector: More than one shared buffer visited";
      result = buf;
    }
  }
};

/*!
 * \brief Build the dependency chain between async operations and their
 *        corresponding buffers & synchronizations.
 *
 *        Example:
 *        If we encounter the following pattern:
 *
 *        tcgen5mma_gemm_ts(..., mbar, ...)
 *        mbarrier_wait_parity(mbar)
 *
 *        The builder will link the mbarrier to the buffers used in the
 * TCGEN5MMA
 */
class AsyncDependencyChainBuilder : public StmtExprVisitor {
public:
  AsyncDependencyChainBuilder(Map<Var, Buffer> buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

  std::unordered_map<const BufferNode *, Array<BufferRegion>>
      mbar_to_buffer_reads_;

  std::unordered_map<const BufferNode *, Array<BufferRegion>>
      mbar_to_buffer_writes_;

private:
  Map<Var, Buffer> buffer_data_to_buffer_;

  // Shared memory buffers referenced by initialize_tcgen05_descriptor calls,
  // accumulated during traversal and linked to mbar on tcgen05_mma_arrive.
  std::vector<BufferRegion> pending_tcgen05_smem_reads_;

  // C_tmem buffer from ptx_tcgen05_mma_ss, accumulated during traversal.
  Optional<Buffer> pending_tcgen05_c_buf_;

  // Helper to extract a Buffer from a tvm_access_ptr call expression.
  Optional<Buffer> TryGetBufFromAccessPtr(const PrimExpr &expr) {
    auto call = expr.as<CallNode>();
    if (!call || !call->op.same_as(builtin::tvm_access_ptr()))
      return Optional<Buffer>();
    auto var = call->args[1].as<VarNode>();
    if (!var)
      return Optional<Buffer>();
    auto it = buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(var));
    if (it == buffer_data_to_buffer_.end())
      return Optional<Buffer>();
    return (*it).second;
  }

  void VisitExpr_(const CallNode *op) final {
    auto args = op->args;
    if (op->op.same_as(builtin::call_extern())) {
      std::string func_name_with_template = args[0].as<StringImmNode>()->value;
      std::size_t le_pos = func_name_with_template.find_first_of('<');
      std::string func_name = le_pos == std::string::npos
                                  ? func_name_with_template
                                  : func_name_with_template.substr(0, le_pos);
      // TODO(lei): refactor to use identical ops.
      if (func_name == "tl::tcgen5mma_gemm_ts" ||
          func_name == "tl::tcgen5mma_gemm_ss") {
        // TCGEN5MMA (high-level, before LowerTileOp)
        auto get_buf_from_access_ptr_call =
            [&](const PrimExpr &expr) -> Buffer {
          auto call = expr.as<CallNode>();
          ICHECK(call);
          ICHECK(call->op.same_as(builtin::tvm_access_ptr()));
          auto var = call->args[1].as<VarNode>();
          ICHECK(var);
          auto it = buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(var));
          ICHECK(it != buffer_data_to_buffer_.end());
          return (*it).second;
        };
        Buffer a_buf = get_buf_from_access_ptr_call(args[1]);
        Buffer b_buf = get_buf_from_access_ptr_call(args[2]);
        Buffer mbar_buf = get_buf_from_access_ptr_call(args[4]);

        TmemLoadCollector tmem_collector;
        tmem_collector(args[3]);
        ICHECK(tmem_collector.result.defined())
            << "TmemLoadCollector: No tmem buffer load found in the TCGEN5MMA "
               "call";
        Buffer c_buf = tmem_collector.result;

        PrimExpr clear_accum = args[5];
        mbar_to_buffer_reads_[mbar_buf.get()].push_back(
            BufferRegion::FullRegion(a_buf));
        mbar_to_buffer_reads_[mbar_buf.get()].push_back(
            BufferRegion::FullRegion(b_buf));
        mbar_to_buffer_writes_[mbar_buf.get()].push_back(
            BufferRegion::FullRegion(c_buf));
        auto analyzer = std::make_shared<arith::Analyzer>();
        if (!analyzer->CanProveEqual(clear_accum, Bool(true))) {
          mbar_to_buffer_reads_[mbar_buf.get()].push_back(
              BufferRegion::FullRegion(c_buf));
        }
      }
      // TODO (lei) Link wgmma to buffers and tl.wait_wgmma
    } else if (op->op.same_as(initialize_tcgen05_descriptor())) {
      // Lowered form: initialize_tcgen05_descriptor(desc, start_addr, ...)
      // args[1] is a tvm_access_ptr to the shared memory buffer (A or B).
      if (args.size() >= 2) {
        if (auto buf = TryGetBufFromAccessPtr(args[1])) {
          pending_tcgen05_smem_reads_.push_back(
              BufferRegion::FullRegion(buf.value()));
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    } else if (op->op.same_as(ptx_tcgen05_mma_ss()) ||
               op->op.same_as(ptx_tcgen05_mma_ts())) {
      // Lowered form: ptx_tcgen05_mma_ss(kind, desc_a, off_a, desc_b, off_b,
      //                                  c_tmem, c_off, desc_val, scale, ...)
      // args[5] is C_tmem.data (a Var, not tvm_access_ptr).
      if (args.size() > 5) {
        auto var = args[5].as<VarNode>();
        if (var) {
          auto it = buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(var));
          if (it != buffer_data_to_buffer_.end()) {
            pending_tcgen05_c_buf_ = (*it).second;
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    } else if (op->op.same_as(tcgen05_mma_arrive())) {
      // Lowered form: tcgen05_mma_arrive(mbar_access_ptr)
      // Link accumulated shared memory reads to mbar.
      if (!args.empty()) {
        if (auto mbar_buf = TryGetBufFromAccessPtr(args[0])) {
          const BufferNode *mbar_key = mbar_buf.value().get();
          for (const auto &region : pending_tcgen05_smem_reads_) {
            mbar_to_buffer_reads_[mbar_key].push_back(region);
          }
          if (pending_tcgen05_c_buf_.defined()) {
            mbar_to_buffer_writes_[mbar_key].push_back(
                BufferRegion::FullRegion(pending_tcgen05_c_buf_.value()));
          }
        } else if (!pending_tcgen05_smem_reads_.empty() ||
                   pending_tcgen05_c_buf_.defined()) {
          LOG(WARNING) << "tcgen05_mma_arrive: could not resolve mbar buffer "
                       << "from args[0]; discarding pending state";
        }
      } else if (!pending_tcgen05_smem_reads_.empty() ||
                 pending_tcgen05_c_buf_.defined()) {
        LOG(WARNING) << "tcgen05_mma_arrive: empty args; discarding "
                     << "pending state";
      }
      // Always clear pending state after an arrive, whether successful or not,
      // to prevent stale entries from being misattributed to a future arrive.
      pending_tcgen05_smem_reads_.clear();
      pending_tcgen05_c_buf_ = Optional<Buffer>();
      StmtExprVisitor::VisitExpr_(op);
    } else if (op->op.same_as(tir::builtin::if_then_else())) {
      const PrimExpr &then_expr = args[1];
      const PrimExpr &else_expr = args[2];
      this->VisitExpr(then_expr);
      this->VisitExpr(else_expr);
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }
};

/*!
 * \brief Detect if a statement follows the global memory copy pattern:
 *        1. Contains exactly one buffer store operation
 *        2. Source buffer must be in global memory scope
 *        3. Destination buffer must be in local or shared memory scope
 */
class BufferRegionCollector : public StmtExprVisitor {
public:
  BufferRegionCollector(Map<Var, Buffer> buffer_data_to_buffer,
                        const AsyncDependencyChainBuilder &chain_builder,
                        Target target)
      : buffer_data_to_buffer_(buffer_data_to_buffer),
        chain_builder_(chain_builder), target_(target) {}

  Array<BufferRegion> GetReads() const { return reads_; }

  Array<BufferRegion> GetWrites() const { return writes_; }

  bool GetGlobalCopyPattern() const { return is_global_copy_pattern_; }

  bool GetTmaCopyPattern() const { return is_tma_copy_; }

  bool HasNonCopyTileOp() const { return has_non_copy_tile_op_; }

private:
  static bool IsGlobalLikeBuffer(const Buffer &buffer) {
    return IsGlobalBuffer(buffer) ||
           (buffer.defined() && buffer.scope().empty());
  }

  void HandleTileOp(const TileOperator &tile_op) {
    if (tile_op.as<RegionOpNode>()) {
      return;
    }
    if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
      BufferRegionCollector nested(buffer_data_to_buffer_, chain_builder_,
                                   target_);
      nested(parallel->GetRoot());
      reads_.insert(reads_.end(), nested.GetReads().begin(),
                    nested.GetReads().end());
      writes_.insert(writes_.end(), nested.GetWrites().begin(),
                     nested.GetWrites().end());
      is_global_copy_pattern_ =
          is_global_copy_pattern_ || nested.GetGlobalCopyPattern();
      is_tma_copy_ = is_tma_copy_ || nested.GetTmaCopyPattern();
      has_non_copy_tile_op_ =
          has_non_copy_tile_op_ || nested.HasNonCopyTileOp();
      return;
    }
    AccessRegions access = tile_op->GetAccessRegions();
    reads_.insert(reads_.end(), access.reads.begin(), access.reads.end());
    writes_.insert(writes_.end(), access.writes.begin(), access.writes.end());
    // Detect explicit TMA-like producer ops for pipeline planning.
    // Plain T.copy no longer auto-upgrades to TMA in the generic pipeline
    // path; only warp-specialized rewriting may turn it into
    // tl.tileop.tma_copy.
    if (const auto *copy = tile_op.as<CopyNode>()) {
      if (IsGlobalLikeBuffer(copy->src) && IsSharedBuffer(copy->dst)) {
        is_global_copy_pattern_ = true;
      }
    }
    // Conv2D im2col always uses TMA on Hopper.
    if (const auto *im2col = tile_op.as<Conv2DIm2ColOpNode>()) {
      if (IsGlobalLikeBuffer(im2col->src_) && IsSharedBuffer(im2col->dst_)) {
        is_global_copy_pattern_ = true;
        if (TargetIsHopper(target_)) {
          is_tma_copy_ = true;
        }
      }
      return;
    }
    if (!tile_op.as<CopyNode>()) {
      has_non_copy_tile_op_ = true;
    }
  }

  Optional<Buffer> TryGetBufFromAccessPtr(const PrimExpr &expr) const {
    auto call = expr.as<CallNode>();
    if (!call)
      return Optional<Buffer>();
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      if (call->args.size() <= 1)
        return Optional<Buffer>();
      auto *var = call->args[1].as<VarNode>();
      if (!var)
        return Optional<Buffer>();
      auto it = buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(var));
      if (it == buffer_data_to_buffer_.end())
        return Optional<Buffer>();
      return (*it).second;
    }
    if (call->op.same_as(tl::access_ptr())) {
      if (call->args.empty())
        return Optional<Buffer>();
      auto *load = call->args[0].as<BufferLoadNode>();
      if (!load)
        return Optional<Buffer>();
      return load->buffer;
    }
    return Optional<Buffer>();
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Buffer store_buffer = op->buffer;
    Array<PrimExpr> indices = op->indices;
    // convert indices to region
    Array<Range> region;
    for (const auto &index : indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    auto store_region = BufferRegion(store_buffer, region);
    writes_.push_back(store_region);

    is_global_read_ = false;
    this->VisitExpr(op->value);
    if (is_global_read_ && IsSharedBuffer(store_buffer)) {
      is_global_copy_pattern_ = true;
    }
    is_global_read_ = false;
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    auto load_buffer = op->buffer;
    Array<PrimExpr> indices = op->indices;
    // convert indices to region
    Array<Range> region;
    for (const auto &index : indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    auto load_region = BufferRegion(load_buffer, region);
    reads_.push_back(load_region);

    if (IsGlobalLikeBuffer(op->buffer) && !within_condition_expr_) {
      // skip condition expr of if_then_else node
      // shared[i] = T.if_then_else(global[i] < n, register_a[i], register_b[i])
      // is not a global read shared[i] = T.if_then_else(global[i] < n,
      // global_a[i], global_b[i]) is a global read
      is_global_read_ = true;
    }
  }

  void VisitExpr_(const CallNode *op) final {
    auto args = op->args;
    if (auto tile_op = ParseOperator(tvm::ffi::GetRef<Call>(op));
        tile_op.defined()) {
      HandleTileOp(tile_op);
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    if (op->op.same_as(builtin::address_of())) {
      BufferRegion buffer_region;
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        buffer_region = BufferRegion::FullRegion(load->buffer);
      } else if (const auto *var_node = op->args[0].as<VarNode>()) {
        Var data_var = tvm::ffi::GetRef<Var>(var_node);
        auto it = buffer_data_to_buffer_.find(data_var);
        if (it != buffer_data_to_buffer_.end()) {
          buffer_region = BufferRegion::FullRegion((*it).second);
        }
      }
      if (buffer_region.defined()) {
        // because we only care about the buffer itself instead of indices
        reads_.push_back(buffer_region);
      }
    } else if (op->op.same_as(builtin::tvm_access_ptr())) {
      const VarNode *buffer_var = op->args[1].as<VarNode>();
      ICHECK(buffer_var);
      auto it = buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(buffer_var));
      if (it != buffer_data_to_buffer_.end()) {
        const Buffer &buffer = (*it).second;
        const BufferRegion buffer_region = BufferRegion::FullRegion(buffer);
        // because we only care about the buffer itself instead of indices
        reads_.push_back(buffer_region);
      }
    } else if (op->op.same_as(builtin::ptx_cp_async()) ||
               op->op.same_as(tl::ptx_cp_async())) {
      // Explicit cp.async call: args[0] = dst access ptr, args[1] = src access
      // ptr. Treat as a global->shared copy candidate in pipeline planning.
      if (args.size() >= 2) {
        auto dst_buf = TryGetBufFromAccessPtr(args[0]);
        auto src_buf = TryGetBufFromAccessPtr(args[1]);
        if (src_buf.defined()) {
          reads_.push_back(BufferRegion::FullRegion(src_buf.value()));
        }
        if (dst_buf.defined()) {
          writes_.push_back(BufferRegion::FullRegion(dst_buf.value()));
        }
        if (src_buf.defined() && dst_buf.defined() &&
            IsGlobalLikeBuffer(src_buf.value()) &&
            IsSharedBuffer(dst_buf.value())) {
          is_global_copy_pattern_ = true;
        }
      }
      if (args.size() == 4) {
        // Predicated cp.async should not be treated as a proven full
        // overwrite of the destination buffer at pipeline-planning
        // granularity. Model a conservative dependence on the destination so
        // required initialization or other producer-side writes are not moved
        // across the async copy.
        if (auto dst_buf = TryGetBufFromAccessPtr(args[0])) {
          reads_.push_back(BufferRegion::FullRegion(dst_buf.value()));
        }
        // Preserve dependence from a predicated guard expression.
        this->VisitExpr(args[3]);
      }
      return;
    } else if (op->op.same_as(builtin::if_then_else())) {
      within_condition_expr_ = true;
      this->VisitExpr(op->args[0]);
      within_condition_expr_ = false;
      for (auto i = 1; i < op->args.size(); i++) {
        this->VisitExpr(op->args[i]);
      }
    } else if (op->op.same_as(tl::mbarrier_wait_parity())) {
      // The mbarrier argument is a BufferLoad on a shared.barrier scope
      // buffer.  Only track mbarrier→buffer dependencies for user-allocated
      // barriers.
      if (auto *buf_load = args[0].as<BufferLoadNode>()) {
        Buffer mbar_buf = buf_load->buffer;
        auto buffer_reads =
            chain_builder_.mbar_to_buffer_reads_.find(mbar_buf.get());
        auto buffer_writes =
            chain_builder_.mbar_to_buffer_writes_.find(mbar_buf.get());
        if (buffer_reads != chain_builder_.mbar_to_buffer_reads_.end()) {
          reads_.insert(reads_.end(), buffer_reads->second.begin(),
                        buffer_reads->second.end());
        }
        if (buffer_writes != chain_builder_.mbar_to_buffer_writes_.end()) {
          writes_.insert(
              writes_.end(),
              chain_builder_.mbar_to_buffer_writes_.at(mbar_buf.get()).begin(),
              chain_builder_.mbar_to_buffer_writes_.at(mbar_buf.get()).end());
        }
      }
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    within_condition_expr_ = true;
    this->VisitExpr(op->condition);
    within_condition_expr_ = false;
    this->VisitStmt(op->then_case);
    if (op->else_case.defined()) {
      within_condition_expr_ = true;
      this->VisitStmt(op->else_case.value());
      within_condition_expr_ = false;
    }
  }

private:
  AsyncDependencyChainBuilder chain_builder_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
  Array<BufferRegion> reads_;
  Array<BufferRegion> writes_;
  bool is_global_read_ = false;
  bool under_buffer_store_ = false;
  bool is_global_copy_pattern_ = false;
  bool is_tma_copy_ = false;
  bool has_non_copy_tile_op_ = false;
  bool within_condition_expr_ = false;
};

class PipelinePlanner : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &f, bool use_async_copy = true) {
    PipelinePlanner substituter(use_async_copy);
    for (const auto &[_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "Pipeline_Planning: Require the target attribute";
    substituter.target_ = target.value();
    return substituter.VisitStmt(f->body);
  }

private:
  PipelinePlanner() = default;
  PipelinePlanner(bool use_async_copy) : use_async_copy_(use_async_copy) {}

  /*! \brief Information about a pipeline stage
   *
   * \param reads Array of buffer regions read by this stage
   * \param writes Array of buffer regions written by this stage
   * \param original_stmt_index Original position of this stage in the pipeline
   * before reordering \param order Current position of this stage in the
   * pipeline after reordering (-1 if not yet assigned) \param stage Pipeline
   * stage number this operation belongs to (-1 if not yet assigned) \param
   * copy_stage Whether this stage is a memory copy operation \param
   * last_use_stmt_index Index of the last statement (in original order) that
   * uses the results of this stage (-1 if not yet determined). This field is
   * crucial for pipeline optimization:
   * - For copy stages: indicates the index of the last statement that reads
   * from the copied data, helping determine optimal placement of copy
   * operations
   * - Used to ensure copy operations are scheduled before their consumers
   * - A value of -1 means no subsequent statement uses this stage's output
   * - This information enables better pipeline scheduling by minimizing data
   *   dependencies and maximizing parallelism
   */
  struct PipelineStageInfo {
    Array<BufferRegion> reads, writes;
    int original_stmt_index{};
    int order = -1, stage = -1;
    bool copy_stage = false;
    bool tma_copy = false; // true if this copy stage uses TMA (not cp.async)
    bool conditional_execution = false;
    bool producer_for_copy = false;
    // Commit statements have no buffer writes, but they must be scheduled as a
    // part of their cp.async producer group (after the cp.async calls).
    bool cp_async_commit_stage = false;
    int cp_async_call_count = 0;
    int cp_async_commit_count = 0;
    int cp_async_wait_count = 0;
    // Minimal static wait_group(n) value in this stmt.
    // numeric_limits<int>::max() means no static wait value is observed.
    int cp_async_wait_min_inflight = std::numeric_limits<int>::max();
    bool cp_async_wait_has_dynamic = false;
    int cp_async_group = -1;
    int last_use_stmt_index =
        -1; // Initialized to -1, indicating no consumers found yet

  public:
    bool is_first_stage() const {
      return copy_stage || producer_for_copy || cp_async_commit_stage;
    }
    bool is_copy_stage() const { return copy_stage; }
    bool is_tma_copy() const { return tma_copy; }
    bool is_producer_for_copy() const { return producer_for_copy; }
    bool is_cp_async_commit_stage() const { return cp_async_commit_stage; }
    bool has_cp_async_call() const { return cp_async_call_count > 0; }
    bool has_cp_async_commit() const { return cp_async_commit_count > 0; }
    bool has_cp_async_wait() const { return cp_async_wait_count > 0; }
    bool is_last_use_stmt_index_valid() const {
      return last_use_stmt_index != -1;
    }
  };

  struct AsyncIntrinInfo {
    int cp_async_call_count = 0;
    int cp_async_commit_count = 0;
    int cp_async_wait_count = 0;
    int cp_async_wait_min_inflight = std::numeric_limits<int>::max();
    bool cp_async_wait_has_dynamic = false;
  };

  AsyncIntrinInfo AnalyzeAsyncIntrinsics(const Stmt &stmt) {
    AsyncIntrinInfo info;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      const auto *call = node.as<CallNode>();
      if (call == nullptr) {
        return;
      }
      if (call->op.same_as(builtin::ptx_cp_async()) ||
          call->op.same_as(tl::ptx_cp_async())) {
        ++info.cp_async_call_count;
      } else if (call->op.same_as(builtin::ptx_commit_group())) {
        ++info.cp_async_commit_count;
      } else if (call->op.same_as(builtin::ptx_wait_group())) {
        ++info.cp_async_wait_count;
        if (!call->args.empty()) {
          if (const int64_t *imm = as_const_int(call->args[0])) {
            info.cp_async_wait_min_inflight = std::min(
                info.cp_async_wait_min_inflight, static_cast<int>(*imm));
          } else {
            info.cp_async_wait_has_dynamic = true;
          }
        } else {
          info.cp_async_wait_has_dynamic = true;
        }
      }
    });
    return info;
  }

  bool MayBeConditionallyExecuted(const Stmt &stmt) const {
    bool conditional = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (conditional) {
        return;
      }
      if (const auto *if_then_else = node.as<IfThenElseNode>()) {
        conditional = true;
        return;
      }
      if (const auto *realize = node.as<BlockRealizeNode>()) {
        if (!is_one(realize->predicate)) {
          conditional = true;
        }
      }
    });
    return conditional;
  }

  bool IsAsyncProducerCandidate(const PipelineStageInfo &pinfo) const {
    if (pinfo.conditional_execution) {
      return false;
    }
    if (pinfo.is_tma_copy()) {
      return false;
    }
    if (pinfo.has_cp_async_wait()) {
      return false;
    }
    if (pinfo.has_cp_async_commit() && !pinfo.has_cp_async_call()) {
      return false;
    }
    return pinfo.is_copy_stage() || pinfo.has_cp_async_call();
  }

  bool IsPureCopyStmt(const Stmt &stmt) const {
    auto is_global_like_buffer = [](const Buffer &buffer) {
      return IsGlobalBuffer(buffer) ||
             (buffer.defined() && buffer.scope().empty());
    };
    auto is_pure_raw_copy_value = [&](const PrimExpr &expr,
                                      const auto &self) -> bool {
      if (const auto *load = expr.as<BufferLoadNode>()) {
        return is_global_like_buffer(load->buffer);
      }
      if (const auto *cast = expr.as<CastNode>()) {
        return self(cast->value, self);
      }
      return false;
    };

    bool saw_copy = false;
    bool saw_non_copy_tile_op = false;
    bool saw_non_copy_buffer_store = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (saw_non_copy_tile_op || saw_non_copy_buffer_store) {
        return;
      }
      if (const auto *store = node.as<BufferStoreNode>()) {
        saw_copy = true;
        if ((!IsSharedBuffer(store->buffer) &&
             !IsLocalBuffer(store->buffer, /*allow_var=*/true)) ||
            !is_pure_raw_copy_value(store->value, is_pure_raw_copy_value)) {
          saw_non_copy_buffer_store = true;
        }
        return;
      }
      const auto *call = node.as<CallNode>();
      if (call == nullptr) {
        return;
      }
      auto tile_op = ParseOperator(tvm::ffi::GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      if (tile_op.as<RegionOpNode>()) {
        return;
      }
      if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
        if (IsPureCopyStmt(parallel->GetRoot())) {
          saw_copy = true;
        } else {
          saw_non_copy_tile_op = true;
        }
        return;
      }
      if (tile_op.as<CopyNode>() || tile_op.as<Conv2DIm2ColOpNode>()) {
        saw_copy = true;
      } else {
        saw_non_copy_tile_op = true;
      }
    });
    return saw_copy && !saw_non_copy_tile_op && !saw_non_copy_buffer_store;
  }

  Optional<TileOperator> GetSinglePureCopyTileOp(const Stmt &stmt) const {
    Optional<TileOperator> copy_tile_op;
    bool saw_non_copy_tile_op = false;
    bool saw_multiple_copy_ops = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (saw_non_copy_tile_op || saw_multiple_copy_ops) {
        return;
      }
      const auto *call = node.as<CallNode>();
      if (call == nullptr) {
        return;
      }
      auto tile_op = ParseOperator(tvm::ffi::GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      if (tile_op.as<RegionOpNode>()) {
        return;
      }
      if (tile_op.as<CopyNode>() || tile_op.as<Conv2DIm2ColOpNode>()) {
        if (copy_tile_op.defined()) {
          saw_multiple_copy_ops = true;
          copy_tile_op = Optional<TileOperator>();
        } else {
          copy_tile_op = tile_op;
        }
      } else {
        saw_non_copy_tile_op = true;
        copy_tile_op = Optional<TileOperator>();
      }
    });
    if (saw_non_copy_tile_op || saw_multiple_copy_ops) {
      return Optional<TileOperator>();
    }
    return copy_tile_op;
  }

  static bool IsGlobalLikeBuffer(const Buffer &buffer) {
    return IsGlobalBuffer(buffer) ||
           (buffer.defined() && buffer.scope().empty());
  }

  void ClassifyCopyLikeStage(const Stmt &stmt, PipelineStageInfo *pinfo) const {
    ICHECK(pinfo != nullptr);
    if (pinfo->conditional_execution) {
      return;
    }

    // Explicit cp.async producer statements participate in the synthetic
    // stage-0 producer schedule just like ordinary global->shared copies.
    if (pinfo->has_cp_async_call()) {
      pinfo->copy_stage = true;
      return;
    }

    if (pinfo->copy_stage) {
      return;
    }

    auto copy_tile_op = GetSinglePureCopyTileOp(stmt);
    if (!copy_tile_op.defined()) {
      return;
    }

    if (const auto *copy = copy_tile_op.value().as<CopyNode>()) {
      if (!IsGlobalLikeBuffer(copy->src) || !IsSharedBuffer(copy->dst)) {
        return;
      }
      pinfo->copy_stage = true;
      return;
    }

    if (const auto *im2col = copy_tile_op.value().as<Conv2DIm2ColOpNode>()) {
      if (!IsGlobalLikeBuffer(im2col->src_) || !IsSharedBuffer(im2col->dst_)) {
        return;
      }
      pinfo->copy_stage = true;
      pinfo->tma_copy = TargetIsHopper(target_);
    }
  }

  void AnalyzeCopyLastUse(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    for (auto &pinfo : *pipeline_stage_infos) {
      if (!pinfo.is_first_stage()) {
        continue;
      }

      for (int i = pinfo.original_stmt_index + 1;
           i < static_cast<int>(pipeline_stage_infos->size()); ++i) {
        for (const BufferRegion &read : (*pipeline_stage_infos)[i].reads) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == read->buffer &&
                                    MayConflict(r->region, read->region);
                           }) != pinfo.writes.end()) {
            pinfo.last_use_stmt_index = std::max(pinfo.last_use_stmt_index, i);
          }
        }

        if (!pinfo.is_copy_stage() ||
            (pinfo.cp_async_group >= 0 &&
             pinfo.cp_async_group ==
                 (*pipeline_stage_infos)[i].cp_async_group)) {
          continue;
        }

        for (const BufferRegion &write : (*pipeline_stage_infos)[i].writes) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == write->buffer &&
                                    MayConflict(r->region, write->region);
                           }) != pinfo.writes.end()) {
            LOG(FATAL) << "Pipeline planning error: Multiple writes to "
                          "overlapping buffer regions detected. "
                       << "Stage " << pinfo.original_stmt_index << " and stage "
                       << i << " are both writing to buffer '"
                       << write->buffer->name
                       << "' with overlapping regions. This is not supported "
                          "in pipeline planning.";
          }
        }
      }
    }
  }

  bool EmitImplicitAsyncAnnotations(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos,
      Map<String, Any> *annotations) const {
    if (!TargetHasAsyncCopy(target_) || !use_async_copy_) {
      return false;
    }

    std::vector<int> async_group_ids(pipeline_stage_infos.size(), -1);
    std::vector<int> stmt_indices_by_order(pipeline_stage_infos.size());
    std::iota(stmt_indices_by_order.begin(), stmt_indices_by_order.end(), 0);
    std::stable_sort(stmt_indices_by_order.begin(), stmt_indices_by_order.end(),
                     [&](int lhs, int rhs) {
                       if (pipeline_stage_infos[lhs].order !=
                           pipeline_stage_infos[rhs].order) {
                         return pipeline_stage_infos[lhs].order <
                                pipeline_stage_infos[rhs].order;
                       }
                       return lhs < rhs;
                     });

    int next_async_group_id = 0;
    std::map<std::pair<int, int>, int> implicit_group_ids;
    for (int stmt_idx : stmt_indices_by_order) {
      const auto &pinfo = pipeline_stage_infos[stmt_idx];
      if (!IsAsyncProducerCandidate(pinfo)) {
        continue;
      }
      auto key = std::make_pair(pinfo.stage, pinfo.last_use_stmt_index);
      auto [it, inserted] =
          implicit_group_ids.emplace(key, next_async_group_id);
      if (inserted) {
        ++next_async_group_id;
      }
      async_group_ids[stmt_idx] = it->second;
    }

    if (next_async_group_id == 0) {
      return false;
    }

    std::vector<Integer> async_producers;
    std::vector<Integer> async_producer_groups;
    async_producers.reserve(pipeline_stage_infos.size());
    async_producer_groups.reserve(pipeline_stage_infos.size());
    std::unordered_set<int> async_stage_ids;
    for (size_t i = 0; i < pipeline_stage_infos.size(); ++i) {
      bool is_async_producer = async_group_ids[i] != -1;
      async_producers.push_back(Integer(is_async_producer ? 1 : 0));
      async_producer_groups.push_back(Integer(async_group_ids[i]));
      if (is_async_producer) {
        async_stage_ids.insert(pipeline_stage_infos[i].stage);
      }
    }

    annotations->Set(kPipelineAsyncProducers, Array<Integer>(async_producers));
    annotations->Set(kPipelineAsyncProducerGroups,
                     Array<Integer>(async_producer_groups));

    std::vector<int> sorted_async_stage_ids(async_stage_ids.begin(),
                                            async_stage_ids.end());
    std::sort(sorted_async_stage_ids.begin(), sorted_async_stage_ids.end());
    std::vector<Integer> async_stages;
    async_stages.reserve(sorted_async_stage_ids.size());
    for (int stage_id : sorted_async_stage_ids) {
      async_stages.push_back(Integer(stage_id));
    }
    annotations->Set(tir::attr::software_pipeline_async_stages,
                     Array<Integer>(async_stages));
    return true;
  }

  void MaybeAnnotateLegacyAsyncPipelineLoop(const Stmt &pipeline_body_root,
                                            const Array<Stmt> &pipeline_stmts,
                                            const Array<Integer> &order_array,
                                            const Array<Integer> &stage_array,
                                            Map<String, Any> *annotations) {
    if (!TargetHasAsyncCopy(target_) || !use_async_copy_) {
      return;
    }
    ICHECK_EQ(pipeline_stmts.size(), order_array.size());
    ICHECK_EQ(pipeline_stmts.size(), stage_array.size());

    AsyncDependencyChainBuilder chain_builder(buffer_data_to_buffer_);
    chain_builder(pipeline_body_root);

    std::vector<PipelineStageInfo> pipeline_stage_infos;
    pipeline_stage_infos.reserve(pipeline_stmts.size());
    for (size_t i = 0; i < pipeline_stmts.size(); ++i) {
      auto pinfo = MakePipelineStageInfo(pipeline_stmts[i], i, chain_builder);
      ClassifyCopyLikeStage(pipeline_stmts[i], &pinfo);
      pinfo.order = static_cast<int>(order_array[i]->value);
      pinfo.stage = static_cast<int>(stage_array[i]->value);
      if (!pinfo.is_copy_stage() && !pinfo.conditional_execution &&
          pinfo.stage == 0) {
        bool reads_global = false;
        bool writes_shared = false;
        for (const BufferRegion &read : pinfo.reads) {
          if (IsGlobalLikeBuffer(read->buffer)) {
            reads_global = true;
            break;
          }
        }
        for (const BufferRegion &write : pinfo.writes) {
          if (IsSharedBuffer(write->buffer)) {
            writes_shared = true;
            break;
          }
        }
        if (reads_global && writes_shared) {
          pinfo.copy_stage = true;
        }
      }
      pipeline_stage_infos.push_back(std::move(pinfo));
    }

    AnalyzeCopyLastUse(&pipeline_stage_infos);
    EmitImplicitAsyncAnnotations(pipeline_stage_infos, annotations);
  }

  PipelineStageInfo
  MakePipelineStageInfo(Stmt stmt, int idx,
                        AsyncDependencyChainBuilder &chain_builder) {
    Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
                /*body*/ std::move(stmt));
    Array<Array<BufferRegion>> access =
        GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    auto collector =
        BufferRegionCollector(buffer_data_to_buffer_, chain_builder, target_);
    collector(block);
    PipelineStageInfo pinfo;
    pinfo.reads = std::move(collector.GetReads());
    pinfo.writes = std::move(collector.GetWrites());
    pinfo.original_stmt_index = idx;
    pinfo.conditional_execution = MayBeConditionallyExecuted(block->body);
    bool pure_copy_stage =
        collector.GetGlobalCopyPattern() && IsPureCopyStmt(block->body);
    pinfo.copy_stage = pure_copy_stage;
    pinfo.tma_copy = pure_copy_stage && !pinfo.conditional_execution &&
                     collector.GetTmaCopyPattern();
    auto async_info = AnalyzeAsyncIntrinsics(block->body);
    pinfo.cp_async_call_count = async_info.cp_async_call_count;
    pinfo.cp_async_commit_count = async_info.cp_async_commit_count;
    pinfo.cp_async_wait_count = async_info.cp_async_wait_count;
    pinfo.cp_async_wait_min_inflight = async_info.cp_async_wait_min_inflight;
    pinfo.cp_async_wait_has_dynamic = async_info.cp_async_wait_has_dynamic;
    ClassifyCopyLikeStage(block->body, &pinfo);
    return std::move(pinfo);
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    auto order_anno = loop->annotations.Get("tl_pipeline_order");
    auto stage_anno = loop->annotations.Get("tl_pipeline_stage");
    auto num_stages_anno = loop->annotations.Get("num_stages");
    if (order_anno && stage_anno) {
      // Check if order_anno or stage_anno contains -1, which means TMA+WS is
      // enabled
      bool ws_tma_enabled = false;
      auto order_array = Downcast<Array<Integer>>(order_anno.value());
      auto stage_array = Downcast<Array<Integer>>(stage_anno.value());
      for (const auto &val : order_array) {
        if (val->value == -1) {
          ws_tma_enabled = true;
          break;
        }
      }
      if (!ws_tma_enabled) {
        for (const auto &val : stage_array) {
          if (val->value == -1) {
            ws_tma_enabled = true;
            break;
          }
        }
      }

      if (ws_tma_enabled) {
        return StmtExprMutator::VisitStmt_(loop);
      }

      Map<String, Any> annotations;
      for (const auto &[key, value] : loop->annotations) {
        if (key != "tl_pipeline_order") {
          annotations.Set(key, value);
        }
      }
      annotations.Set(tir::attr::software_pipeline_order, order_anno.value());

      for (const auto &[key, value] : loop->annotations) {
        if (key != "tl_pipeline_stage") {
          annotations.Set(key, value);
        }
      }
      annotations.Set(tir::attr::software_pipeline_stage, stage_anno.value());
      if (TargetHasAsyncCopy(target_) && use_async_copy_) {
        // Legacy explicit stage/order annotations do not carry per-statement
        // async producer metadata yet, so keep the previous stage-level
        // behavior as a fallback for these loops.
        annotations.Set(tir::attr::software_pipeline_async_stages,
                        Array<Integer>{0});
      }
      Stmt pipeline_body_root{nullptr};
      const SeqStmtNode *pipeline_body_seq = nullptr;
      if (const auto *realize = loop->body.as<BlockRealizeNode>()) {
        const auto &block = realize->block;
        for (const auto &buffer : block->alloc_buffers) {
          ICHECK(buffer->IsInstance<BufferNode>());
          buffer_data_to_buffer_.Set(buffer->data, buffer);
        }
        pipeline_body_root = block->body;
      } else {
        pipeline_body_root = loop->body;
      }
      {
        Stmt current = pipeline_body_root;
        while (true) {
          if (const auto *seq_stmt = current.as<SeqStmtNode>()) {
            pipeline_body_seq = seq_stmt;
            break;
          }
          if (const auto *if_then_else = current.as<IfThenElseNode>()) {
            ICHECK(!if_then_else->else_case.defined())
                << "Pipeline_Planning: Can't handle the body of the loop "
                   "because the IfThenElse node has an else branch";
            current = if_then_else->then_case;
            continue;
          }
          if (const auto *let_stmt = current.as<LetStmtNode>()) {
            current = let_stmt->body;
            continue;
          }
          LOG(FATAL) << "Pipeline_Planning: Can't handle the body of the loop "
                     << "because it is not a SeqStmt, IfThenElse without else, "
                     << "or LetStmt wrapping them, but got "
                     << current->GetTypeKey();
        }
      }
      ICHECK(pipeline_body_seq != nullptr);
      MaybeAnnotateLegacyAsyncPipelineLoop(pipeline_body_root,
                                           pipeline_body_seq->seq, order_array,
                                           stage_array, &annotations);
      auto for_node = tvm::ffi::GetRef<For>(loop);
      for_node.CopyOnWrite()->annotations = annotations;
      return for_node;
    }

    if (!num_stages_anno)
      return StmtExprMutator::VisitStmt_(loop);
    int num_stages = num_stages_anno->as<IntImmNode>()->value;
    Stmt pipeline_body_root{nullptr};
    if (const auto *realize = loop->body.as<BlockRealizeNode>()) {
      const auto &block = realize->block;
      for (const auto &buffer : block->alloc_buffers) {
        ICHECK(buffer->IsInstance<BufferNode>());
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
      pipeline_body_root = block->body;
    } else {
      pipeline_body_root = loop->body;
    }
    const SeqStmtNode *pipeline_body_seq = nullptr;
    {
      Stmt current = pipeline_body_root;
      while (true) {
        if (const auto *seq_stmt = current.as<SeqStmtNode>()) {
          pipeline_body_seq = seq_stmt;
          break;
        }
        if (const auto *if_then_else = current.as<IfThenElseNode>()) {
          ICHECK(!if_then_else->else_case.defined())
              << "Pipeline_Planning: Can't handle the body of the loop because "
                 "the IfThenElse node has an else branch";
          current = if_then_else->then_case;
          continue;
        }
        if (const auto *let_stmt = current.as<LetStmtNode>()) {
          current = let_stmt->body;
          continue;
        }
        LOG(FATAL) << "Pipeline_Planning: Can't handle the body of the loop "
                   << "because it is not a SeqStmt, IfThenElse without else, "
                   << "or LetStmt wrapping them, but got "
                   << current->GetTypeKey();
      }
    }
    ICHECK(pipeline_body_seq != nullptr);

    CHECK(num_stages >= 1);
    CHECK(loop->kind == ForKind::kSerial);

    // Flatten nested SeqStmts. TMA copy lowering emits
    // SeqStmt({produce, wait}) which creates nested SeqStmts when placed
    // inside the loop body. Flatten them so pipeline planning can assign
    // individual stages to the produce and wait statements.
    Array<Stmt> flat_stmts;
    std::function<void(const Stmt &)> flatten_seq = [&](const Stmt &s) {
      if (auto *seq = s.as<SeqStmtNode>()) {
        for (const auto &sub : seq->seq) {
          flatten_seq(sub);
        }
      } else {
        flat_stmts.push_back(s);
      }
    };
    for (size_t i = 0; i < pipeline_body_seq->size(); i++) {
      flatten_seq(pipeline_body_seq->seq[i]);
    }

    AsyncDependencyChainBuilder chain_builder(buffer_data_to_buffer_);
    chain_builder(pipeline_body_root);

    std::vector<PipelineStageInfo> pipeline_stage_infos;
    for (size_t i = 0; i < flat_stmts.size(); i++) {
      auto pinfo = MakePipelineStageInfo(flat_stmts[i], i, chain_builder);
      pipeline_stage_infos.push_back(std::move(pinfo));
    }

    // Build a formal cp.async synchronization model in original statement
    // order:
    //   group := cp_async* then commit
    // and map wait_group(n) to the committed groups it must wait for.
    struct CPAsyncGroupInfo {
      int group_id = -1;
      int anchor_cp_async_stmt = -1;
      std::vector<int> cp_async_stmt_indices;
      std::vector<int> commit_stmt_indices;
      std::unordered_set<const BufferNode *> written_buffers;
      int last_use_stmt_index = -1;
    };
    struct WaitDependencyInfo {
      int wait_stmt_index = -1;
      // Committed groups that must be completed before this wait can pass.
      std::vector<int> required_group_ids;
    };

    std::vector<CPAsyncGroupInfo> cp_async_groups;
    std::vector<WaitDependencyInfo> wait_dependencies;
    std::vector<int> committed_groups_in_order;

    auto create_new_group = [&]() -> int {
      int group_id = static_cast<int>(cp_async_groups.size());
      CPAsyncGroupInfo group;
      group.group_id = group_id;
      cp_async_groups.push_back(std::move(group));
      return group_id;
    };

    int open_group = -1;
    for (size_t i = 0; i < pipeline_stage_infos.size(); ++i) {
      auto &pinfo = pipeline_stage_infos[i];
      if (pinfo.has_cp_async_call()) {
        if (open_group == -1) {
          open_group = create_new_group();
        }
        pinfo.cp_async_group = open_group;
        auto &group = cp_async_groups[open_group];
        group.cp_async_stmt_indices.push_back(static_cast<int>(i));
        if (group.anchor_cp_async_stmt == -1) {
          group.anchor_cp_async_stmt = static_cast<int>(i);
        }
        for (const auto &write : pinfo.writes) {
          group.written_buffers.insert(write->buffer.get());
        }
      }
      if (pinfo.has_cp_async_commit()) {
        if (open_group == -1) {
          open_group = create_new_group();
        }
        pinfo.cp_async_group = open_group;
        cp_async_groups[open_group].commit_stmt_indices.push_back(
            static_cast<int>(i));
        committed_groups_in_order.push_back(open_group);
        // A commit closes the currently open cp.async group.
        open_group = -1;
      }
      if (pinfo.has_cp_async_wait()) {
        int committed_count =
            static_cast<int>(committed_groups_in_order.size());
        int retain_inflight = pinfo.cp_async_wait_has_dynamic
                                  ? 0
                                  : pinfo.cp_async_wait_min_inflight;
        int required_count =
            pinfo.cp_async_wait_has_dynamic
                ? committed_count
                : std::max(0, committed_count - retain_inflight);

        WaitDependencyInfo wait_dep;
        wait_dep.wait_stmt_index = static_cast<int>(i);
        wait_dep.required_group_ids.assign(committed_groups_in_order.begin(),
                                           committed_groups_in_order.begin() +
                                               required_count);
        wait_dependencies.push_back(std::move(wait_dep));
      }
    }

    const int pipeline_stmt_count =
        static_cast<int>(pipeline_stage_infos.size());
    auto stmt_reads_buffer_set =
        [&](int stmt_idx,
            const std::unordered_set<const BufferNode *> &buffers) -> bool {
      if (buffers.empty() || stmt_idx < 0 || stmt_idx >= pipeline_stmt_count) {
        return false;
      }
      for (const BufferRegion &read : pipeline_stage_infos[stmt_idx].reads) {
        if (buffers.count(read->buffer.get())) {
          return true;
        }
      }
      return false;
    };

    // Record earliest consumers for each cp.async group, and track all
    // cp.async-written buffers for wait remapping.
    std::unordered_set<const BufferNode *> async_written_buffers;
    std::vector<int> cp_async_group_first_consumer(
        cp_async_groups.size(), std::numeric_limits<int>::max());
    for (size_t group_id = 0; group_id < cp_async_groups.size(); ++group_id) {
      const auto &group = cp_async_groups[group_id];
      async_written_buffers.insert(group.written_buffers.begin(),
                                   group.written_buffers.end());
      for (int stmt_idx = 0; stmt_idx < pipeline_stmt_count; ++stmt_idx) {
        if (pipeline_stage_infos[stmt_idx].is_first_stage()) {
          continue;
        }
        if (stmt_reads_buffer_set(stmt_idx, group.written_buffers)) {
          cp_async_group_first_consumer[group_id] = stmt_idx;
          break;
        }
      }
    }

    // Heuristic for wait_group(0): bind each wait to the first unmatched
    // downstream consumer of cp.async-written shared buffers, then derive the
    // required groups from that consumer's read set.
    //
    // This keeps wait-group scheduling buffer-aware even when wait uses a full
    // drain value, enabling patterns like "wait/decode B first, then wait A".
    int last_bound_consumer_stmt = -1;
    for (auto &wait_dep : wait_dependencies) {
      int wait_stmt_idx = wait_dep.wait_stmt_index;
      if (wait_stmt_idx < 0 || wait_stmt_idx >= pipeline_stmt_count) {
        continue;
      }
      const auto &wait_stmt_info = pipeline_stage_infos[wait_stmt_idx];
      if (!wait_stmt_info.has_cp_async_wait() ||
          wait_stmt_info.cp_async_wait_has_dynamic ||
          wait_stmt_info.cp_async_wait_min_inflight != 0) {
        continue;
      }
      if (wait_stmt_info.has_cp_async_call() ||
          wait_stmt_info.has_cp_async_commit()) {
        continue;
      }

      int search_start =
          std::max(wait_stmt_idx + 1, last_bound_consumer_stmt + 1);
      int consumer_stmt_idx = -1;
      for (int stmt_idx = search_start; stmt_idx < pipeline_stmt_count;
           ++stmt_idx) {
        if (pipeline_stage_infos[stmt_idx].is_first_stage()) {
          continue;
        }
        if (stmt_reads_buffer_set(stmt_idx, async_written_buffers)) {
          consumer_stmt_idx = stmt_idx;
          break;
        }
      }
      if (consumer_stmt_idx < 0) {
        continue;
      }

      std::vector<int> required_groups_for_consumer;
      for (size_t group_id = 0; group_id < cp_async_groups.size(); ++group_id) {
        if (stmt_reads_buffer_set(consumer_stmt_idx,
                                  cp_async_groups[group_id].written_buffers)) {
          required_groups_for_consumer.push_back(static_cast<int>(group_id));
        }
      }
      if (required_groups_for_consumer.empty()) {
        continue;
      }

      wait_dep.required_group_ids = std::move(required_groups_for_consumer);
      last_bound_consumer_stmt = consumer_stmt_idx;
    }

    std::vector<int> cp_async_group_schedule_order;
    cp_async_group_schedule_order.reserve(cp_async_groups.size());
    for (size_t group_id = 0; group_id < cp_async_groups.size(); ++group_id) {
      cp_async_group_schedule_order.push_back(static_cast<int>(group_id));
    }

    // For every copy stage, mark all its dependency stages as producer_for_copy
    // Helper struct to manage copy stage dependency reads
    struct CopyStageDependencyReadsManager {
      std::vector<BufferRegion> regions;

      // Add a region if not already present (by structural equality)
      void AddUnique(const BufferRegion &region) {
        for (const BufferRegion &copy_read : regions) {
          if (region->buffer.same_as(copy_read->buffer)) {
            return;
          }
        }
        regions.push_back(region);
      }

      // Check if a region is present (by structural equality)
      bool Contains(const BufferRegion &region) const {
        for (const BufferRegion &copy_read : regions) {
          if (region->buffer.same_as(copy_read->buffer)) {
            return true;
          }
        }
        return false;
      }

      size_t Size() const { return regions.size(); }
    };

    CopyStageDependencyReadsManager copy_stage_dependency_reads_mgr;

    // Step 1. Collect Copy reads
    for (const auto &pinfo : pipeline_stage_infos) {
      if (pinfo.is_copy_stage()) {
        for (const BufferRegion &read : pinfo.reads) {
          copy_stage_dependency_reads_mgr.AddUnique(read);
        }
      }
    }

    // Step 2. find if pinfo write the copy reads, then update the
    // copy_stage_dependency_reads To prevent infinite loops, we set a maximum
    // number of iterations. In theory, the number of possible updates is
    // bounded by the number of pipeline stages, since each stage can only be
    // marked as producer_for_copy once, and each read can only be added once.
    // But for safety, we add a hard limit.
    const size_t max_iterations = (pipeline_stage_infos.size() * 4) + 16;
    size_t iter_count = 0;

    for (auto &pinfo : pipeline_stage_infos) {
      if (!pinfo.is_copy_stage()) {
        continue;
      }
      auto original_copy_stmt_index = pinfo.original_stmt_index;
      bool updated = true;
      while (updated) {
        updated = false;
        for (auto &pinfo_inner : pipeline_stage_infos) {
          if (pinfo_inner.is_copy_stage()) {
            continue;
          }
          if (pinfo_inner.original_stmt_index >= original_copy_stmt_index) {
            break;
          }

          bool should_prepare = false;
          for (const BufferRegion &write : pinfo_inner.writes) {
            if (copy_stage_dependency_reads_mgr.Contains(write)) {
              should_prepare = true;
              break;
            }
          }
          if (should_prepare && !pinfo_inner.is_producer_for_copy()) {
            pinfo_inner.producer_for_copy = true;
            updated = true;
          }
          if (should_prepare) {
            for (const BufferRegion &read : pinfo_inner.reads) {
              size_t before = copy_stage_dependency_reads_mgr.Size();
              copy_stage_dependency_reads_mgr.AddUnique(read);
              if (copy_stage_dependency_reads_mgr.Size() > before) {
                updated = true;
              }
            }
          }
        }
        iter_count++;
        if (iter_count > max_iterations) {
          LOG(FATAL)
              << "Pipeline planning: Exceeded maximum iterations ("
              << max_iterations << ") in copy stage dependency propagation. "
              << "This may indicate a cyclic or pathological dependency graph.";
        }
      }
    }

    // Analysis use-def chain to determine last_use_stmt_index for copy
    // operations This step is critical for pipeline optimization as it
    // identifies the index of the last statement that consumes data produced by
    // copy stages, enabling optimal placement of copy operations in the
    // pipeline schedule.
    AnalyzeCopyLastUse(&pipeline_stage_infos);

    // Treat each explicit `cp_async* ; commit` producer group as a synthetic
    // copy stage for scheduling. All statements in the group share the same
    // last-use anchor, so stage assignment keeps the producer group together
    // instead of scheduling individual cp.async members near different
    // consumers.
    for (auto &group : cp_async_groups) {
      if (group.anchor_cp_async_stmt < 0) {
        continue;
      }
      int group_last_use = -1;
      int group_last_cp_async_stmt = group.anchor_cp_async_stmt;
      for (int cp_async_stmt_idx : group.cp_async_stmt_indices) {
        group_last_cp_async_stmt =
            std::max(group_last_cp_async_stmt, cp_async_stmt_idx);
        group_last_use = std::max(
            group_last_use,
            pipeline_stage_infos[cp_async_stmt_idx].last_use_stmt_index);
      }
      if (group_last_use < 0) {
        // Fallback to the latest cp.async statement when no consumer is found
        // (rare, but keep local ordering correct).
        group_last_use = group_last_cp_async_stmt;
      }
      group.last_use_stmt_index = group_last_use;
      for (int cp_async_stmt_idx : group.cp_async_stmt_indices) {
        pipeline_stage_infos[cp_async_stmt_idx].last_use_stmt_index =
            group_last_use;
      }
      for (int commit_stmt_idx : group.commit_stmt_indices) {
        auto &commit_info = pipeline_stage_infos[commit_stmt_idx];
        commit_info.last_use_stmt_index = group_last_use;
        // Only mark commit-only statements. If commit is already fused with
        // cp.async calls in the same statement, its local ordering is
        // preserved by the statement itself.
        if (commit_info.has_cp_async_commit() &&
            !commit_info.has_cp_async_call()) {
          commit_info.cp_async_commit_stage = true;
        }
      }
    }

    // Order explicit cp.async producer groups by the lifetime of the data they
    // introduce. Groups whose data dies earlier should be scheduled earlier in
    // the synthetic stage-0 producer schedule, which also matches the desired
    // wait rebinding behavior for wait_group(0) consumers.
    std::stable_sort(
        cp_async_group_schedule_order.begin(),
        cp_async_group_schedule_order.end(), [&](int lhs_group, int rhs_group) {
          int lhs_last_use = cp_async_groups[lhs_group].last_use_stmt_index;
          int rhs_last_use = cp_async_groups[rhs_group].last_use_stmt_index;
          if (lhs_last_use != rhs_last_use) {
            return lhs_last_use < rhs_last_use;
          }
          int lhs_first_consumer = cp_async_group_first_consumer[lhs_group];
          int rhs_first_consumer = cp_async_group_first_consumer[rhs_group];
          if (lhs_first_consumer != rhs_first_consumer) {
            return lhs_first_consumer < rhs_first_consumer;
          }
          return cp_async_groups[lhs_group].anchor_cp_async_stmt <
                 cp_async_groups[rhs_group].anchor_cp_async_stmt;
        });

    // Making stages and orders
    int order_idx = 0;
    // Stage 1. Create pipeline stages and assign order
    for (auto &pinfo : pipeline_stage_infos) {
      // Skip elements that must be in first stage:
      // 1. Copy stages (with active last_use_stmt_index) - these need special
      // handling
      //    because they have consumers that depend on their data
      // 2. All Producer stages for copy stages.
      if (pinfo.is_first_stage() && pinfo.is_last_use_stmt_index_valid()) {
        continue;
      }

      // Main logic stage assignment:
      // - Increment order index
      // - Assign to new stage (current num_stages)
      pinfo.order = order_idx++;
      pinfo.stage = num_stages;

      // Schedule copy stages that have this stage as their last consumer
      // This ensures copy operations are placed right before their final
      // consumer for optimal pipeline efficiency
      for (auto &pinfo_1 : pipeline_stage_infos) {
        if ((pinfo_1.is_first_stage() &&
             pinfo_1.last_use_stmt_index == pinfo.original_stmt_index)) {
          pinfo_1.order = order_idx++;
          pinfo_1.stage = 0; // Copy stages are typically assigned to stage 0
        }
      }
    }

    ICHECK(size_t(order_idx) == pipeline_stage_infos.size())
        << "The number of stages should be equal to the number of pipeline "
           "stages. "
        << "Got " << order_idx << " stages and " << pipeline_stage_infos.size()
        << " pipeline stages.";

    // Step 2. if all the copy is at the end of the order, we can move these
    // copy to the beginning of the order and shrink the stage offset by 1.
    int copy_stage_at_end = [&]() {
      int copy_stage_cnt = 0;
      int copy_order_min = pipeline_stage_infos.size();
      int non_copy_order_max = 0;
      for (auto &pinfo : pipeline_stage_infos) {
        if (pinfo.is_first_stage()) {
          copy_stage_cnt++;
          copy_order_min = std::min(copy_order_min, pinfo.order);
        } else {
          non_copy_order_max = std::max(non_copy_order_max, pinfo.order);
        }
      }
      if (copy_order_min > non_copy_order_max)
        return copy_stage_cnt;
      return -1;
    }();
    if (copy_stage_at_end > 0 && num_stages >= 2) {
      for (auto &pinfo : pipeline_stage_infos) { // move copy to the beginning
        pinfo.order =
            (pinfo.order + copy_stage_at_end) % pipeline_stage_infos.size();
        if (!pinfo.is_copy_stage() && !pinfo.is_producer_for_copy() &&
            !pinfo.is_cp_async_commit_stage())
          pinfo.stage--;
      }
    }

    // Enforce stage(commit) == stage(cp_async anchor) per group.
    for (const auto &group : cp_async_groups) {
      if (group.anchor_cp_async_stmt < 0) {
        continue;
      }
      int anchor_stage = pipeline_stage_infos[group.anchor_cp_async_stmt].stage;
      for (int commit_stmt_idx : group.commit_stmt_indices) {
        pipeline_stage_infos[commit_stmt_idx].stage = anchor_stage;
      }
    }

    // Sanity check: within a cp.async group, commit statements must appear
    // after all cp.async calls in the same stage order.
    for (const auto &group : cp_async_groups) {
      if (group.anchor_cp_async_stmt < 0) {
        continue;
      }
      int max_cp_async_order = -1;
      int anchor_stage = pipeline_stage_infos[group.anchor_cp_async_stmt].stage;
      for (int cp_async_stmt_idx : group.cp_async_stmt_indices) {
        if (pipeline_stage_infos[cp_async_stmt_idx].stage == anchor_stage) {
          max_cp_async_order =
              std::max(max_cp_async_order,
                       pipeline_stage_infos[cp_async_stmt_idx].order);
        }
      }
      for (int commit_stmt_idx : group.commit_stmt_indices) {
        if (pipeline_stage_infos[commit_stmt_idx].stage == anchor_stage) {
          // If commit is fused with cp.async calls in the same statement, the
          // statement-local order is preserved and we cannot enforce an
          // inter-statement order relation.
          if (pipeline_stage_infos[commit_stmt_idx].has_cp_async_call()) {
            continue;
          }
          CHECK_GT(pipeline_stage_infos[commit_stmt_idx].order,
                   max_cp_async_order)
              << "Pipeline planning error: cp.async commit is scheduled before "
                 "its cp.async calls. commit_stmt="
              << commit_stmt_idx << ", commit_order="
              << pipeline_stage_infos[commit_stmt_idx].order
              << ", max_cp_async_order=" << max_cp_async_order
              << ", stage=" << anchor_stage;
        }
      }
    }

    // Enforce wait placement based on the formal group dependency model.
    // For static wait_group(n), it depends on committed groups except the
    // latest n groups. For dynamic wait args, we conservatively treat it as
    // wait_group(0), i.e. draining all committed groups.
    auto get_group_stage = [&](int group_id) -> int {
      if (group_id < 0 ||
          group_id >= static_cast<int>(cp_async_groups.size())) {
        return 0;
      }
      const auto &group = cp_async_groups[group_id];
      if (!group.commit_stmt_indices.empty()) {
        return pipeline_stage_infos[group.commit_stmt_indices.back()].stage;
      }
      if (group.anchor_cp_async_stmt >= 0) {
        return pipeline_stage_infos[group.anchor_cp_async_stmt].stage;
      }
      return 0;
    };

    for (const auto &wait_dep : wait_dependencies) {
      if (wait_dep.wait_stmt_index < 0 ||
          wait_dep.wait_stmt_index >=
              static_cast<int>(pipeline_stage_infos.size())) {
        continue;
      }
      const auto &wait_stmt_info =
          pipeline_stage_infos[wait_dep.wait_stmt_index];
      // If wait is fused with cp.async/commit in the same statement, we cannot
      // place it independently at stage granularity. Keep the statement stage
      // unchanged and rely on the statement's explicit local ordering.
      if (wait_stmt_info.has_cp_async_call() ||
          wait_stmt_info.has_cp_async_commit()) {
        continue;
      }
      if (wait_dep.required_group_ids.empty()) {
        continue;
      }

      int required_stage = pipeline_stage_infos[wait_dep.wait_stmt_index].stage;
      std::unordered_set<const BufferNode *> waited_buffers;
      for (int group_id : wait_dep.required_group_ids) {
        required_stage = std::max(required_stage, get_group_stage(group_id));
        if (group_id >= 0 &&
            group_id < static_cast<int>(cp_async_groups.size())) {
          const auto &group = cp_async_groups[group_id];
          waited_buffers.insert(group.written_buffers.begin(),
                                group.written_buffers.end());
        }
      }

      int dependent_consumer_stage = -1;
      if (!waited_buffers.empty()) {
        for (int stmt_idx = wait_dep.wait_stmt_index + 1;
             stmt_idx < static_cast<int>(pipeline_stage_infos.size());
             ++stmt_idx) {
          if (pipeline_stage_infos[stmt_idx].is_first_stage()) {
            continue;
          }
          bool dependent_read = false;
          for (const BufferRegion &read :
               pipeline_stage_infos[stmt_idx].reads) {
            if (waited_buffers.count(read->buffer.get())) {
              dependent_read = true;
              break;
            }
          }
          if (dependent_read) {
            dependent_consumer_stage = pipeline_stage_infos[stmt_idx].stage;
            break;
          }
        }
      }

      if (dependent_consumer_stage >= 0) {
        CHECK_GE(dependent_consumer_stage, required_stage)
            << "Pipeline planning error: wait_group stage cannot be after its "
               "dependent consumer stage. wait_stmt="
            << wait_dep.wait_stmt_index << ", required_stage=" << required_stage
            << ", consumer_stage=" << dependent_consumer_stage;
        pipeline_stage_infos[wait_dep.wait_stmt_index].stage =
            dependent_consumer_stage;
      } else {
        pipeline_stage_infos[wait_dep.wait_stmt_index].stage = required_stage;
      }
    }

    // Enforce cp.async ordering constraints.
    //
    // PipelinePlanning's scheduling heuristic may float pure control intrinsics
    // like `ptx_commit_group` earlier because they have no buffer read/write
    // regions. This can break cp.async group semantics and generate illegal
    // code patterns such as:
    //   cp_async_commit();
    //   cp_async_gs<...>(...);
    //
    // We fix this by building a small control-dependency graph among pipeline
    // body statements and doing a stable topological sort with the existing
    // `order` as the tie-breaker.
    {
      int n = static_cast<int>(pipeline_stage_infos.size());
      std::vector<int> order_rank(n, 0);
      for (int i = 0; i < n; ++i) {
        order_rank[i] = pipeline_stage_infos[i].order;
      }

      std::vector<std::unordered_set<int>> edges(n);
      std::vector<int> indeg(n, 0);

      auto add_edge = [&](int u, int v) {
        if (u < 0 || v < 0 || u >= n || v >= n || u == v) {
          return;
        }
        if (edges[u].insert(v).second) {
          indeg[v] += 1;
        }
      };

      auto group_schedule_key = [&](const CPAsyncGroupInfo &group) {
        int key = std::numeric_limits<int>::max();
        for (int cp_stmt_idx : group.cp_async_stmt_indices) {
          key = std::min(key, pipeline_stage_infos[cp_stmt_idx].order);
        }
        for (int commit_stmt_idx : group.commit_stmt_indices) {
          key = std::min(key, pipeline_stage_infos[commit_stmt_idx].order);
        }
        if (key == std::numeric_limits<int>::max()) {
          key = group.anchor_cp_async_stmt;
        }
        return key;
      };

      // Respect the synthetic producer-group order computed above. The control
      // dependencies below only preserve cp.async group boundaries; they must
      // not re-prioritize groups using a different heuristic.
      std::vector<int> cp_async_group_schedule_order;
      cp_async_group_schedule_order.reserve(cp_async_groups.size());
      for (size_t group_id = 0; group_id < cp_async_groups.size(); ++group_id) {
        cp_async_group_schedule_order.push_back(static_cast<int>(group_id));
      }
      std::stable_sort(
          cp_async_group_schedule_order.begin(),
          cp_async_group_schedule_order.end(),
          [&](int lhs_group, int rhs_group) {
            int lhs_key = group_schedule_key(cp_async_groups[lhs_group]);
            int rhs_key = group_schedule_key(cp_async_groups[rhs_group]);
            if (lhs_key != rhs_key) {
              return lhs_key < rhs_key;
            }
            return cp_async_groups[lhs_group].anchor_cp_async_stmt <
                   cp_async_groups[rhs_group].anchor_cp_async_stmt;
          });

      // (1) cp.async group semantics:
      //   group := cp_async* ; commit
      // and group boundaries must be preserved:
      //   commit(group_i) happens before cp_async(group_{i+1}).
      for (size_t g = 0; g < cp_async_groups.size(); ++g) {
        const auto &group = cp_async_groups[g];
        for (int cp_stmt_idx : group.cp_async_stmt_indices) {
          for (int commit_stmt_idx : group.commit_stmt_indices) {
            // Only enforce intra-iteration order (same stage).
            if (pipeline_stage_infos[cp_stmt_idx].stage ==
                pipeline_stage_infos[commit_stmt_idx].stage) {
              add_edge(cp_stmt_idx, commit_stmt_idx);
            }
          }
        }
      }
      for (size_t i = 0; i + 1 < cp_async_group_schedule_order.size(); ++i) {
        const auto &group = cp_async_groups[cp_async_group_schedule_order[i]];
        if (group.commit_stmt_indices.empty()) {
          continue;
        }
        const auto &next_group =
            cp_async_groups[cp_async_group_schedule_order[i + 1]];
        for (int commit_stmt_idx : group.commit_stmt_indices) {
          for (int next_cp_stmt_idx : next_group.cp_async_stmt_indices) {
            if (pipeline_stage_infos[commit_stmt_idx].stage ==
                pipeline_stage_infos[next_cp_stmt_idx].stage) {
              add_edge(commit_stmt_idx, next_cp_stmt_idx);
            }
          }
        }
      }

      // (2) wait_group ordering:
      //   - wait must stay before dependent same-stage consumers;
      //   - if wait is in the same stage as a required commit, it must be
      //     after that commit;
      //   - when legal, delay wait to be as close as possible to its first
      //     dependent consumer by placing independent same-stage statements
      //     before wait (without crossing async/control-only boundaries).
      for (const auto &wait_dep : wait_dependencies) {
        int wait_stmt_idx = wait_dep.wait_stmt_index;
        if (wait_stmt_idx < 0 || wait_stmt_idx >= n) {
          continue;
        }

        const auto &wait_stmt_info = pipeline_stage_infos[wait_stmt_idx];
        // If wait is fused with cp.async/commit, rely on local statement order.
        if (wait_stmt_info.has_cp_async_call() ||
            wait_stmt_info.has_cp_async_commit()) {
          continue;
        }

        std::unordered_set<const BufferNode *> waited_buffers;
        for (int group_id : wait_dep.required_group_ids) {
          if (group_id < 0 ||
              group_id >= static_cast<int>(cp_async_groups.size())) {
            continue;
          }
          const auto &group = cp_async_groups[group_id];
          waited_buffers.insert(group.written_buffers.begin(),
                                group.written_buffers.end());

          // If wait shares the same stage with a required commit, it must be
          // ordered after that commit for the same original iteration.
          for (int commit_stmt_idx : group.commit_stmt_indices) {
            if (pipeline_stage_infos[commit_stmt_idx].stage ==
                wait_stmt_info.stage) {
              add_edge(commit_stmt_idx, wait_stmt_idx);
            }
          }
        }

        if (waited_buffers.empty()) {
          continue;
        }

        // Ensure wait happens before all same-stage consumers that read any of
        // the waited buffers.
        int first_dependent_consumer_idx = -1;
        for (int consumer_stmt_idx = wait_stmt_idx + 1; consumer_stmt_idx < n;
             ++consumer_stmt_idx) {
          if (pipeline_stage_infos[consumer_stmt_idx].stage !=
              wait_stmt_info.stage) {
            continue;
          }
          bool dependent_read = false;
          for (const BufferRegion &read :
               pipeline_stage_infos[consumer_stmt_idx].reads) {
            if (waited_buffers.count(read->buffer.get())) {
              dependent_read = true;
              break;
            }
          }
          if (dependent_read) {
            if (first_dependent_consumer_idx == -1) {
              first_dependent_consumer_idx = consumer_stmt_idx;
            }
            add_edge(wait_stmt_idx, consumer_stmt_idx);
          }
        }

        // Delay wait within the same stage until right before the first
        // dependent consumer when possible, so independent prep work can run
        // while async copies are still in flight.
        if (first_dependent_consumer_idx != -1) {
          for (int stmt_idx = wait_stmt_idx + 1;
               stmt_idx < first_dependent_consumer_idx; ++stmt_idx) {
            const auto &mid_stmt_info = pipeline_stage_infos[stmt_idx];
            if (mid_stmt_info.stage != wait_stmt_info.stage) {
              continue;
            }
            // Do not move wait across pure control statements (e.g. barriers)
            // because they are not represented in buffer read/write regions.
            if (mid_stmt_info.reads.empty() && mid_stmt_info.writes.empty()) {
              break;
            }
            // Do not move wait across explicit async synchronization points.
            if (mid_stmt_info.has_cp_async_call() ||
                mid_stmt_info.has_cp_async_commit() ||
                mid_stmt_info.has_cp_async_wait()) {
              break;
            }
            bool touches_waited_buffers = false;
            for (const BufferRegion &read : mid_stmt_info.reads) {
              if (waited_buffers.count(read->buffer.get())) {
                touches_waited_buffers = true;
                break;
              }
            }
            if (!touches_waited_buffers) {
              for (const BufferRegion &write : mid_stmt_info.writes) {
                if (waited_buffers.count(write->buffer.get())) {
                  touches_waited_buffers = true;
                  break;
                }
              }
            }
            if (!touches_waited_buffers) {
              add_edge(stmt_idx, wait_stmt_idx);
            }
          }
        }
      }

      // Stable topological sort: pick the smallest existing order each time.
      using Item = std::pair<int, int>; // (order_rank, stmt_idx)
      std::priority_queue<Item, std::vector<Item>, std::greater<Item>> ready;
      for (int i = 0; i < n; ++i) {
        if (indeg[i] == 0) {
          ready.push({order_rank[i], i});
        }
      }

      std::vector<int> topo_order;
      topo_order.reserve(n);
      while (!ready.empty()) {
        auto [rank, u] = ready.top();
        ready.pop();
        topo_order.push_back(u);
        for (int v : edges[u]) {
          indeg[v] -= 1;
          if (indeg[v] == 0) {
            ready.push({order_rank[v], v});
          }
        }
      }

      CHECK_EQ(static_cast<int>(topo_order.size()), n)
          << "Pipeline planning error: cycle detected while enforcing cp.async "
             "ordering constraints.";

      for (int new_order = 0; new_order < n; ++new_order) {
        pipeline_stage_infos[topo_order[new_order]].order = new_order;
      }
    }

    // Finally, make the pipeline annotation
    Map<String, Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages") {
        annotations.Set(key, value);
      }
    }
    // Preserve the original TileLang pipelining depth for downstream scheduling
    // (e.g. cp.async wait_group relaxation/splitting). We intentionally do NOT
    // keep the legacy key "num_stages" here because multiple downstream passes
    // (e.g. internal buffer versioning / warp specialization) treat it as an
    // active pipeline marker and do not support nested pipelines.
    annotations.Set("tl_pipelined_num_stages", Integer(num_stages));

    std::vector<Integer> orders, stages;
    orders.reserve(pipeline_stage_infos.size());
    stages.reserve(pipeline_stage_infos.size());
    for (auto &pinfo : pipeline_stage_infos) {
      orders.push_back(pinfo.order);
      stages.push_back(pinfo.stage);
    }

    annotations.Set(tir::attr::software_pipeline_stage, Array<Integer>(stages));
    annotations.Set(tir::attr::software_pipeline_order, Array<Integer>(orders));

    // Propagate per-statement TMA eligibility so InjectSoftwarePipeline can
    // rewrite TMA copies to use pipeline-level barrier management.
    {
      std::vector<Integer> tma_copies;
      tma_copies.reserve(pipeline_stage_infos.size());
      for (auto &pinfo : pipeline_stage_infos) {
        tma_copies.push_back(Integer(pinfo.is_tma_copy() ? 1 : 0));
      }
      annotations.Set(kPipelineTmaCopies, Array<Integer>(tma_copies));
    }

    if (TargetHasAsyncCopy(target_) && use_async_copy_) {
      std::vector<int> async_group_ids(pipeline_stage_infos.size(), -1);
      int next_async_group_id = 0;

      for (int scheduled_group_id : cp_async_group_schedule_order) {
        const auto &group = cp_async_groups[scheduled_group_id];
        bool emitted_group = false;
        for (int stmt_idx : group.cp_async_stmt_indices) {
          if (!IsAsyncProducerCandidate(pipeline_stage_infos[stmt_idx])) {
            continue;
          }
          async_group_ids[stmt_idx] = next_async_group_id;
          emitted_group = true;
        }
        if (emitted_group) {
          ++next_async_group_id;
        }
      }

      std::vector<int> stmt_indices_by_order(pipeline_stage_infos.size());
      std::iota(stmt_indices_by_order.begin(), stmt_indices_by_order.end(), 0);
      std::stable_sort(stmt_indices_by_order.begin(),
                       stmt_indices_by_order.end(), [&](int lhs, int rhs) {
                         if (pipeline_stage_infos[lhs].order !=
                             pipeline_stage_infos[rhs].order) {
                           return pipeline_stage_infos[lhs].order <
                                  pipeline_stage_infos[rhs].order;
                         }
                         return lhs < rhs;
                       });
      std::map<std::pair<int, int>, int> implicit_group_ids;
      for (int stmt_idx : stmt_indices_by_order) {
        const auto &pinfo = pipeline_stage_infos[stmt_idx];
        if (!IsAsyncProducerCandidate(pinfo) ||
            async_group_ids[stmt_idx] != -1) {
          continue;
        }
        auto key = std::make_pair(pinfo.stage, pinfo.last_use_stmt_index);
        auto [it, inserted] =
            implicit_group_ids.emplace(key, next_async_group_id);
        if (inserted) {
          ++next_async_group_id;
        }
        async_group_ids[stmt_idx] = it->second;
      }

      std::vector<Integer> async_producers;
      std::vector<Integer> async_producer_groups;
      async_producers.reserve(pipeline_stage_infos.size());
      async_producer_groups.reserve(pipeline_stage_infos.size());
      std::unordered_set<int> async_stage_ids;
      for (size_t i = 0; i < pipeline_stage_infos.size(); ++i) {
        bool is_async_producer = async_group_ids[i] != -1;
        async_producers.push_back(Integer(is_async_producer ? 1 : 0));
        async_producer_groups.push_back(Integer(async_group_ids[i]));
        if (is_async_producer) {
          async_stage_ids.insert(pipeline_stage_infos[i].stage);
        }
      }
      annotations.Set(kPipelineAsyncProducers, Array<Integer>(async_producers));
      annotations.Set(kPipelineAsyncProducerGroups,
                      Array<Integer>(async_producer_groups));
      if (!async_stage_ids.empty()) {
        std::vector<int> sorted_async_stage_ids(async_stage_ids.begin(),
                                                async_stage_ids.end());
        std::sort(sorted_async_stage_ids.begin(), sorted_async_stage_ids.end());
        std::vector<Integer> async_stages;
        async_stages.reserve(sorted_async_stage_ids.size());
        for (int stage_id : sorted_async_stage_ids) {
          async_stages.push_back(Integer(stage_id));
        }
        annotations.Set(tir::attr::software_pipeline_async_stages,
                        Array<Integer>(async_stages));
      }
    }

    // Reconstruct the loop body with the flattened SeqStmt so that
    // InjectSoftwarePipeline sees the correct number of pipeline stages.
    Stmt new_body_seq = SeqStmt(flat_stmts);
    // Rebuild any wrapper layers (IfThenElse, LetStmt, BlockRealize)
    // between the loop body and the SeqStmt.
    Stmt new_loop_body;
    if (const auto *realize = loop->body.as<BlockRealizeNode>()) {
      const auto &block = realize->block;
      // Rebuild: body_root → ... → new_body_seq
      // We need to reconstruct the chain from block->body to the SeqStmt.
      Stmt rebuilt_inner =
          RebuildBodyWrapper(block->body, pipeline_body_seq, new_body_seq);
      Block new_block(block->iter_vars, block->reads, block->writes,
                      block->name_hint, rebuilt_inner, block->init,
                      block->alloc_buffers, block->match_buffers,
                      block->annotations);
      new_loop_body =
          BlockRealize(realize->iter_values, realize->predicate, new_block);
    } else {
      new_loop_body =
          RebuildBodyWrapper(loop->body, pipeline_body_seq, new_body_seq);
    }

    return For(loop->loop_var, loop->min, loop->extent, loop->kind,
               new_loop_body, loop->thread_binding, annotations);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return std::move(block);
  }

  /*!
   * \brief Rebuild the chain of wrapper statements (IfThenElse, LetStmt)
   *        between the loop body root and the inner SeqStmt, replacing
   *        the old SeqStmt with the new (flattened) one.
   */
  Stmt RebuildBodyWrapper(const Stmt &current, const SeqStmtNode *old_seq,
                          const Stmt &new_seq) {
    if (current.get() == old_seq) {
      return new_seq;
    }
    if (const auto *if_node = current.as<IfThenElseNode>()) {
      return IfThenElse(
          if_node->condition,
          RebuildBodyWrapper(if_node->then_case, old_seq, new_seq),
          if_node->else_case);
    }
    if (const auto *let_node = current.as<LetStmtNode>()) {
      return LetStmt(let_node->var, let_node->value,
                     RebuildBodyWrapper(let_node->body, old_seq, new_seq));
    }
    LOG(FATAL) << "RebuildBodyWrapper: unexpected node type "
               << current->GetTypeKey();
    return current;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
  bool use_async_copy_{};
};

tvm::transform::Pass PipelinePlanning() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool use_async_copy =
        ctx->GetConfig<Bool>("tir.use_async_copy", Bool(true)).value();
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = PipelinePlanner::Substitute(f, use_async_copy);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PipelinePlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PipelinePlanning", PipelinePlanning);
}

} // namespace tl
} // namespace tvm
