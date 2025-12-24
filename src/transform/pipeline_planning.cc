#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include <unordered_map>
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
        // TCGEN5MMA
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
                        const AsyncDependencyChainBuilder &chain_builder)
      : buffer_data_to_buffer_(buffer_data_to_buffer),
        chain_builder_(chain_builder) {}

  Array<BufferRegion> GetReads() const { return reads_; }

  Array<BufferRegion> GetWrites() const { return writes_; }

  bool GetGlobalCopyPattern() const { return is_global_copy_pattern_; }

private:
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
    if (is_global_read_ && (store_buffer.scope() == "shared" ||
                            store_buffer.scope() == "shared.dyn")) {
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

    if (op->buffer.scope() == "global" && !within_condition_expr_) {
      // skip condition expr of if_then_else node
      // shared[i] = T.if_then_else(global[i] < n, register_a[i], register_b[i])
      // is not a global read shared[i] = T.if_then_else(global[i] < n,
      // global_a[i], global_b[i]) is a global read
      is_global_read_ = true;
    }
  }

  void VisitExpr_(const CallNode *op) final {
    auto args = op->args;
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
    } else if (op->op.same_as(builtin::if_then_else())) {
      within_condition_expr_ = true;
      this->VisitExpr(op->args[0]);
      within_condition_expr_ = false;
      for (auto i = 1; i < op->args.size(); i++) {
        this->VisitExpr(op->args[i]);
      }
    } else if (op->op.same_as(tl::mbarrier_wait_parity())) {
      ICHECK(args[0].as<BufferLoadNode>());
      Buffer mbar_buf = args[0].as<BufferLoadNode>()->buffer;
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
  Array<BufferRegion> reads_;
  Array<BufferRegion> writes_;
  bool is_global_read_ = false;
  bool under_buffer_store_ = false;
  bool is_global_copy_pattern_ = false;
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
    bool producer_for_copy = false;
    int last_use_stmt_index =
        -1; // Initialized to -1, indicating no consumers found yet

  public:
    bool is_first_stage() const { return copy_stage || producer_for_copy; }
    bool is_copy_stage() const { return copy_stage; }
    bool is_producer_for_copy() const { return producer_for_copy; }
    bool is_last_use_stmt_index_valid() const {
      return last_use_stmt_index != -1;
    }
  };

  PipelineStageInfo
  MakePipelineStageInfo(Stmt stmt, int idx,
                        AsyncDependencyChainBuilder &chain_builder) {
    Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
                /*body*/ std::move(stmt));
    Array<Array<BufferRegion>> access =
        GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    auto collector =
        BufferRegionCollector(buffer_data_to_buffer_, chain_builder);
    collector(block);
    PipelineStageInfo pinfo;
    pinfo.reads = std::move(collector.GetReads());
    pinfo.writes = std::move(collector.GetWrites());
    pinfo.original_stmt_index = idx;
    pinfo.copy_stage = collector.GetGlobalCopyPattern();
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
      if (TargetHasAsyncCopy(target_) && use_async_copy_)
        annotations.Set(tir::attr::software_pipeline_async_stages,
                        Array<Integer>{0});
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

    AsyncDependencyChainBuilder chain_builder(buffer_data_to_buffer_);
    chain_builder(pipeline_body_root);

    std::vector<PipelineStageInfo> pipeline_stage_infos;
    for (size_t i = 0; i < pipeline_body_seq->size(); i++) {
      auto pinfo =
          MakePipelineStageInfo(pipeline_body_seq->seq[i], i, chain_builder);
      pipeline_stage_infos.push_back(std::move(pinfo));
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
    for (auto &pinfo : pipeline_stage_infos) {
      // Only analyze copy stages (memory copy operations)
      if (!pinfo.is_first_stage())
        continue;

      // Check all subsequent statements to find the latest consumer
      for (int i = pinfo.original_stmt_index + 1;
           i < static_cast<int>(pipeline_body_seq->size()); i++) {

        // Check if any read operation in statement 'i' uses data written by
        // this copy stage
        for (const BufferRegion &read : pipeline_stage_infos[i].reads) {
          // Look for overlapping buffer regions between this stage's writes and
          // stage 'i's reads
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == read->buffer &&
                                    MayConflict(r->region, read->region);
                           }) != pinfo.writes.end()) {
            // Update last_use_stmt_index to the maximum (latest) statement
            // index that uses this data This ensures we capture the final
            // consumer of the copied data
            pinfo.last_use_stmt_index = std::max(pinfo.last_use_stmt_index, i);
          }
        }
        // Check for write-after-write conflicts (multiple stages writing to
        // same buffer region) This is important for pipeline correctness and
        // affects last_use_stmt_index analysis
        if (pinfo.is_copy_stage()) {
          for (const BufferRegion &write : pipeline_stage_infos[i].writes) {
            if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                             [&](const BufferRegion &r) {
                               return r->buffer == write->buffer &&
                                      MayConflict(r->region, write->region);
                             }) != pinfo.writes.end()) {
              LOG(FATAL) << "Pipeline planning error: Multiple writes to "
                            "overlapping buffer regions detected. "
                         << "Stage " << pinfo.original_stmt_index
                         << " and stage " << i
                         << " are both writing to buffer '"
                         << write->buffer->name
                         << "' with overlapping regions. This is not supported "
                            "in pipeline planning.";
            }
          }
        }
      }
    }

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
        if (!pinfo.is_copy_stage() && !pinfo.is_producer_for_copy())
          pinfo.stage--;
      }
    }

    // Finally, make the pipeline annotation
    Map<String, Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages") {
        annotations.Set(key, value);
      }
    }

    std::vector<Integer> orders, stages;
    orders.reserve(pipeline_stage_infos.size());
    stages.reserve(pipeline_stage_infos.size());
    for (auto &pinfo : pipeline_stage_infos) {
      orders.push_back(pinfo.order);
      stages.push_back(pinfo.stage);
    }

    annotations.Set(tir::attr::software_pipeline_stage, Array<Integer>(stages));
    annotations.Set(tir::attr::software_pipeline_order, Array<Integer>(orders));
    if (TargetHasAsyncCopy(target_) && use_async_copy_)
      annotations.Set(tir::attr::software_pipeline_async_stages,
                      Array<Integer>{0});

    return For(loop->loop_var, loop->min, loop->extent, loop->kind, loop->body,
               loop->thread_binding, annotations);
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
