/*!
 * \file inject_software_pipeline.cc
 * \brief Transform annotated loops into pipelined one that parallelize
 * producers and consumers
 */
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <functional>
#include <unordered_set>
#include <utility>

#include "support/utils.h"
#include "tir/schedule/utils.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
using namespace tir;
using namespace ffi;
namespace software_pipeline {

struct LetWrapper {
  Var var;
  PrimExpr value;
};

/*!
 * \brief Collector to find all buffers used in a statement.
 *
 * This is used to collect buffers that are actually used in the pipeline loop
 * body, so that we can properly multi-version them for software pipelining.
 */
class BufferUsageCollector : public StmtExprVisitor {
public:
  BufferUsageCollector(
      const Map<Var, Buffer> &buffer_data_to_buffer,
      const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
          &allocated_buffers)
      : buffer_data_to_buffer_(buffer_data_to_buffer),
        allocated_buffers_(allocated_buffers) {}

  Array<Buffer> Collect(const Stmt &stmt) {
    this->VisitStmt(stmt);
    Array<Buffer> result;
    for (const auto &buffer : used_buffers_) {
      result.push_back(buffer);
    }
    return result;
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    AddBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    AddBuffer(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    // Handle tvm_access_ptr which also accesses buffers
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      if (op->args.size() > 1) {
        if (const auto *var = op->args[1].as<VarNode>()) {
          auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
          if (it != buffer_data_to_buffer_.end()) {
            AddBuffer((*it).second);
          }
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    // Also collect buffers allocated in nested blocks within the pipeline body
    for (const auto &buffer : op->alloc_buffers) {
      used_buffers_.insert(buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void AddBuffer(const Buffer &buffer) {
    // Only add buffers that are allocated (not function input/output buffers)
    if (allocated_buffers_.count(buffer)) {
      used_buffers_.insert(buffer);
    }
  }

  const Map<Var, Buffer> &buffer_data_to_buffer_;
  const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
      &allocated_buffers_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> used_buffers_;
};

/*!
 * \brief Create a block and infer the access region with the given body.
 *
 * The result is a opaque block that doesn't contain any block iter vars. In
 * case the body is a block realize without predicate, it is unnecessary to
 * create a new block, the block of the block realize will be returned.
 *
 * \param body The body of the block.
 * \param buffer_data_to_buffer The map from buffer data to buffer.
 * \return The result block.
 */
Block MakeBlock(const Stmt &body,
                const Map<Var, Buffer> &buffer_data_to_buffer) {
  if (const BlockRealizeNode *block_realize = body.as<BlockRealizeNode>()) {
    if (is_one(block_realize->predicate)) {
      // no need to create a new block
      return block_realize->block;
    }
  }
  Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
              /*body*/ body);
  Array<Array<BufferRegion>> access =
      GetBlockReadWriteRegion(block, buffer_data_to_buffer);
  BlockNode *n = block.CopyOnWrite();
  n->reads = access[0];
  n->writes = access[1];
  return block;
}

/*! Structure that represents the provided annotation per block or loop. */
struct PipelineAnnotation {
  int stage;
  int order;
  bool async;
  // Index of the statement in the original loop body order (SeqStmt order)
  int original_idx = -1;
};

using PipelineInfo = std::unordered_map<Block, PipelineAnnotation,
                                        ObjectPtrHash, ObjectPtrEqual>;

struct BufferAccessInfo {
  int def = -1; // the defining stage of the buffer
  int use = -1; // the last using stage of the buffer
};

/*!
 * \brief Rewriter for the body of the software pipeline. This pass inserts
 * `floormod` to indices of the remapped buffer to select the version
 * corresponding to the pipeline stage.
 */
class PipelineBodyRewriter : public StmtExprMutator {
public:
  /*!
   * \brief Constructor of PipelineBodyRewriter.
   * \param buffer_data_to_buffer The map from buffer data to buffer.
   * \param buffer_remap The map from original buffer to the buffer with updated
   * shape for multi-versioning in the software pipeline. \param pipeline_loop
   * The original loop to be software pipelined. \param access_all_versions
   * Whether all versions the buffers in the software pipeline are accessed.
   * This will be used to update block access region. In the prologue and
   * epilogue of a two-stage software pipeline, only one version of these
   * buffers are accessed.
   */
  PipelineBodyRewriter(const Map<Var, Buffer> &buffer_data_to_buffer,
                       const Map<Buffer, Buffer> &buffer_remap,
                       For pipeline_loop, bool access_all_versions)
      : buffer_data_to_buffer_(buffer_data_to_buffer),
        buffer_remap_(buffer_remap), pipeline_loop_(std::move(pipeline_loop)),
        access_all_versions_(access_all_versions) {}

private:
  BufferRegion
  RewritePipelineBufferRegion(const BufferRegion &buffer_region) const {
    auto it = buffer_remap_.find(buffer_region->buffer);
    if (it != buffer_remap_.end()) {
      Region new_region = buffer_region->region;
      const Buffer &new_buffer = (*it).second;
      // For pipeline buffers, relax the access region of the first dimension to
      // full extent if access_all_versions == true
      Range accessed_version =
          access_all_versions_
              ? Range::FromMinExtent(0, new_buffer->shape[0])
              : Range::FromMinExtent(
                    floormod((pipeline_loop_->loop_var - pipeline_loop_->min),
                             new_buffer->shape[0]),
                    Integer(1));
      new_region.insert(new_region.begin(), accessed_version);
      return BufferRegion(new_buffer, new_region);
    }
    return buffer_region;
  }

  PrimExpr RewriteBufferAccess(const Call &call,
                               const std::vector<int> &arg_indices) {
    auto product = [](const Array<PrimExpr> &input) {
      return foldl(
          [](PrimExpr a, PrimExpr b, Span span) {
            return mul(std::move(a), std::move(b), std::move(span));
          },
          make_const(DataType::Int(32), 1), input);
    };
    Array<PrimExpr> new_args = call->args;
    for (int i : arg_indices) {
      const Buffer &buffer =
          buffer_data_to_buffer_.at(Downcast<Var>(call->args[i]));
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        const Buffer &new_buffer = (*it).second;
        const PrimExpr &old_index = call->args[i + 1];
        PrimExpr offset;
        if (new_buffer->strides.empty()) {
          offset = product(buffer->shape);
        } else {
          offset = new_buffer->strides[0];
        }
        PrimExpr new_index =
            old_index +
            floormod(pipeline_loop_->loop_var, new_buffer->shape[0]) * offset;
        new_args.Set(i + 1, new_index);
      }
    }
    return Call(call->dtype, call->op, new_args, call->span);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    for (const Buffer &alloc_buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(alloc_buffer->data, alloc_buffer);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    BlockNode *n = block.CopyOnWrite();
    n->reads.MutateByApply([this](const BufferRegion &buffer_region) {
      return RewritePipelineBufferRegion(buffer_region);
    });
    n->writes.MutateByApply([this](const BufferRegion &buffer_region) {
      return RewritePipelineBufferRegion(buffer_region);
    });
    for (const Buffer &alloc_buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(alloc_buffer->data);
    }
    return block;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_remap_.find(store->buffer);
    if (it == buffer_remap_.end()) {
      return store;
    }
    const Buffer &new_buffer = (*it).second;
    auto *n = store.CopyOnWrite();
    n->buffer = new_buffer;
    PrimExpr version = floormod(
        (pipeline_loop_->loop_var - pipeline_loop_->min), new_buffer->shape[0]);
    n->indices.insert(n->indices.begin(), version);
    return store;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = buffer_remap_.find(load->buffer);
    if (it == buffer_remap_.end()) {
      return load;
    }
    const Buffer &new_buffer = (*it).second;
    auto *n = load.CopyOnWrite();
    n->buffer = new_buffer;
    PrimExpr version = floormod(
        (pipeline_loop_->loop_var - pipeline_loop_->min), new_buffer->shape[0]);
    n->indices.insert(n->indices.begin(), version);
    return load;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      return RewriteBufferAccess(call, {1});
    }
    return call;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  For pipeline_loop_;
  bool access_all_versions_;
};

/*!
 * \brief Rewriter for the software pipeline that rewrite a loop into a
 * pipelined one.
 */
class PipelineRewriter : public StmtExprMutator {
public:
  /*!
   * \brief Constructor of PipelineRewriter.
   * \param buffer_data_to_buffer The map from buffer data to buffer.
   * \param pipeline_allocs All buffers that need multi-versioning in the
   * pipeline. This includes buffers allocated in the pipeline block and
   * buffers allocated in outer blocks that are used in the pipeline.
   * \param local_allocs Buffers that are allocated in the pipeline block
   * itself. These buffers will be re-allocated in the rewritten block.
   * Buffers in pipeline_allocs but not in local_allocs are allocated in outer
   * blocks and should not be re-allocated.
   * \param pipeline_loop The original loop to be software pipelined.
   * \param pipeline_info The pipeline annotation information.
   * \param loop_var_let_wrappers Let wrappers that depend on the loop var.
   */
  PipelineRewriter(Map<Var, Buffer> buffer_data_to_buffer,
                   const Array<Buffer> &pipeline_allocs,
                   const Array<Buffer> &local_allocs, const For &pipeline_loop,
                   const PipelineInfo &pipeline_info,
                   const std::vector<LetWrapper> &loop_var_let_wrappers)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        pipeline_allocs_(pipeline_allocs), local_allocs_(local_allocs),
        pipeline_loop_(pipeline_loop), pipeline_info_(pipeline_info),
        loop_var_let_wrappers_(loop_var_let_wrappers) {}

  Stmt BuildPipeline() {
    // Step 1: Analyze accesses to the buffers in the pipeline and compute the
    // number of versions need to maintain for each buffer.
    std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual>
        infos = GetBufferAccessInfo();
    for (const Buffer &buffer : pipeline_allocs_) {
      auto it = infos.find(buffer);
      if (it == infos.end()) {
        // Buffer is not accessed in the pipeline blocks, skip it
        continue;
      }
      int num_versions = ComputeBufferVersions(buffer, it->second);
      if (num_versions > 1) {
        buffer_remap_.Set(buffer, RewriteAllocBuffer(buffer, num_versions));
      }
    }
    ordered_stmts_.resize(pipeline_info_.size());
    for (const auto &[block, anno] : pipeline_info_) {
      ordered_stmts_.Set(anno.order, block);
    }

    for (const Block &block : ordered_stmts_) {
      int stage = pipeline_info_[block].stage;
      if (pipeline_info_[block].async) {
        auto &state = async_states[stage];
        state.producer_head = pipeline_loop_->min - 1;
        for (auto write_region : block->writes) {
          auto buffer = write_region->buffer;
          state.dst_buffers.insert(buffer.get());
          if (buffer_remap_.count(buffer))
            state.dst_buffers.insert(buffer_remap_[buffer].get());
        }
      }
    }
    std::unordered_set<int> consumed;
    for (const Block &block : ordered_stmts_) {
      int stage = pipeline_info_[block].stage;
      if (pipeline_info_[block].async) {
        auto &state = async_states[stage];
        if (state.commit_groups.empty() || consumed.count(stage)) {
          state.commit_groups.push_back({});
        }
        state.commit_groups.back().push_back(pipeline_info_[block].order);
        consumed.erase(stage);
        for (auto write_region : block->writes) {
          auto buffer = buffer_remap_.count(write_region->buffer)
                            ? buffer_remap_[write_region->buffer]
                            : write_region->buffer;
          state.buffer_to_commit_group_[buffer.get()] =
              state.commit_groups.size() - 1;
        }
      }

      for (auto read_region : block->reads) {
        for (const auto &[producer_stage_id, producer_state] : async_states) {
          if (producer_stage_id <= stage &&
              producer_state.writes(read_region->buffer)) {
            consumed.insert(producer_stage_id);
          }
        }
      }
    }

    // Step 2: Emit the pipeline prologue, body and epilogue.
    Stmt prologue =
        EmitImpl(pipeline_loop_->min, pipeline_loop_->min + max_stage_, true,
                 true, false);
    Stmt body = EmitImpl(pipeline_loop_->min + max_stage_,
                         pipeline_loop_->min + pipeline_loop_->extent, false,
                         false, false);

    Stmt epilogue =
        EmitImpl(pipeline_loop_->min + pipeline_loop_->extent,
                 pipeline_loop_->min + pipeline_loop_->extent + max_stage_,
                 true, true, true);
    SeqStmt stmt = SeqStmt({prologue, body, epilogue});

    // Step 3: Make a new block that contains new buffer allocations after
    // pipeline rewriting.
    // Only include buffers that are locally allocated in the pipeline block.
    // Buffers from outer blocks will be handled separately.
    Array<Buffer> alloc_buffers;
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> local_allocs_set(
        local_allocs_.begin(), local_allocs_.end());
    for (const auto &alloc : local_allocs_) {
      alloc_buffers.push_back(buffer_remap_.Get(alloc).value_or(alloc));
      buffer_data_to_buffer_.erase(alloc->data);
    }
    Block block = MakeBlock(stmt, buffer_data_to_buffer_);
    block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
    return BlockRealize({}, Bool(true), block);
  }

  /*!
   * \brief Get the buffer remapping created during pipeline rewriting.
   * This is used to update alloc_buffers in outer blocks.
   */
  const Map<Buffer, Buffer> &GetBufferRemap() const { return buffer_remap_; }

private:
  /*!
   * \brief Analyze accesses to the buffers in the software pipeline.
   *
   * This method check the 'define' and 'use' stage of the buffers in the
   * software pipeline, which can be used to compute the number of versions
   * needed to maintain after rewriting.
   */
  std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual>
  GetBufferAccessInfo() {
    std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual>
        infos;
    for (const auto &pair : pipeline_info_) {
      const Block &block = pair.first;
      int stage = pair.second.stage;
      max_stage_ = std::max(max_stage_, stage);

      for (const BufferRegion &write : block->writes) {
        if (!infos.count(write->buffer)) {
          infos.emplace(write->buffer, BufferAccessInfo{});
        }
        auto &info = infos.at(write->buffer);
        if (info.def == -1) {
          info.def = stage;
        } else {
          info.def = std::min(info.def, stage);
        }
      }

      for (const BufferRegion &read : block->reads) {
        if (!infos.count(read->buffer)) {
          infos.emplace(read->buffer, BufferAccessInfo{});
        }
        auto &info = infos.at(read->buffer);
        info.use = std::max(info.use, stage);
      }
    }
    return infos;
  }

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

  /*!
   * \brief Compute the number of versions need to maintain for buffer accessed
   * in the software pipeline.
   *
   * This method applies liveness analysis to the target buffer to compute the
   * number of versions need to maintain during the software pipeline.
   * Annotation `attr::double_buffer_scope` is handled here which provides a way
   * to override the result of the analysis. Additional double buffering in the
   * software pipeline can be useful to eliminate synchronizations in GPU
   * devices.
   *
   * \param buffer The target buffer
   * \param buffer_info The access information of the target buffer.
   * \return The number of versions required for the target buffer.
   */
  int ComputeBufferVersions(const Buffer &buffer,
                            const BufferAccessInfo &buffer_info) {
    if (buffer_info.def == -1) {
      // Keep the original number of versions as buffers defined outside the
      // software pipeline should not be mutated.
      return 1;
    }

    // `use - def + 1` is a upper bound of the needed versions
    // We optimize a few case where the number of versions can be smaller than
    // the upper bound
    int num_versions = buffer_info.use - buffer_info.def + 1;
    if (num_versions >= 2) {
      // A special case when `use - def + 1 == 2`. Double buffering is only
      // needed in this case when these exists a reader block_i and a writer
      // block_j such that order(block_i) < order(block_j) and stage(block_i) <
      // stage(block_j) and the access regions of block_i and block_j overlap.
      bool need_multi_version = false;
      for (const auto &pair1 : pipeline_info_) {
        const Block &writer_block = pair1.first;
        const auto &writer_info = pair1.second;

        auto it1 = std::find_if(writer_block->writes.begin(),
                                writer_block->writes.end(),
                                [&](const BufferRegion &buffer_region) {
                                  return buffer_region->buffer.same_as(buffer);
                                });
        if (it1 == writer_block->writes.end()) {
          continue;
        }

        for (const auto &pair2 : pipeline_info_) {
          const Block &reader_block = pair2.first;
          const auto &reader_info = pair2.second;
          auto it2 = std::find_if(
              reader_block->reads.begin(), reader_block->reads.end(),
              [&](const BufferRegion &buffer_region) {
                return buffer_region->buffer.same_as(buffer);
              });
          if (it2 == reader_block->reads.end()) {
            continue;
          }
          if (writer_info.order < reader_info.order &&
              writer_info.stage < reader_info.stage &&
              MayConflict((*it1)->region, (*it2)->region)) {
            need_multi_version = true;
            break;
          }
        }
      }
      if (!need_multi_version) {
        num_versions--;
      }
    }
    return num_versions;
  }

  /*!
   * \brief Rewrite buffer allocation to keep multiple versions of original
   * buffer for pipelined accesses. \param buffer The buffer to be resized.
   * \param num_versions The number of versions to keep.
   * \return The resized buffer.
   */
  Buffer RewriteAllocBuffer(const Buffer &buffer, int num_versions) {
    ObjectPtr<BufferNode> new_buffer =
        tvm::ffi::make_object<BufferNode>(*(buffer.get()));
    new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));
    if (!new_buffer->strides.empty()) {
      ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
      PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
      new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
    }
    return Buffer(new_buffer);
  }

  // Per-stage states that need to be tracked across pipeline prologue, body,
  // and epilogue.
  struct AsyncStateGlobal {
    // Buffers that this stage asynchronously writes.
    std::unordered_set<const BufferNode *> dst_buffers;
    // An imaginary index that the latest async operation associated with this
    // stage has written into. Only valid if all associated predicates are true,
    // so that we can count the number of async invocations exactly. When it is
    // valid, it is the "sum of extents of loops that have been executed" - 1,
    // e.g. for epilogue it is prologue extent + body extent - 1. This is only
    // needed to compute wait count for epilogue without async producers.
    PrimExpr producer_head;
    std::vector<std::vector<int>> commit_groups;
    std::unordered_map<const BufferNode *, int> buffer_to_commit_group_;
    bool writes(const Buffer &buf) const {
      return dst_buffers.count(buf.get()) > 0;
    }
  };

  // Per-stage states that are local to each of pipeline prologue, body, and
  // epilogue.
  struct AsyncStateLocal {
    struct PendingWait {
      // The index into a list of blocks, where async_wait_queue should be
      // attached at the beginning.
      int insert_before;
      // in_flight_count would be a more precise name, but the implementation
      // uses wait_count for brevity.
      PrimExpr wait_count{nullptr};

      bool valid() const { return wait_count.defined(); }
    };

    std::vector<PendingWait> pending_waits;

    // A symbolic expression representing the index the latest async operation
    // associated with this stage has written into, at the "current" iteration.
    Optional<PrimExpr> producer_head;
    // the commit block's predicate
    PrimExpr commit_predicate{nullptr};
  };

  /*! Structure holding intermediate information for pipeline loop rewriting. */
  struct RewrittenBlockInfo {
    int stage;
    int order;
    PrimExpr start;
    PrimExpr end;
    PrimExpr predicate;
    Block block;
    PrimExpr access_index;
    bool is_async;
  };

  void PopulateWaitCounts(const std::vector<RewrittenBlockInfo> &new_blocks,
                          std::map<int, AsyncStateLocal> *async_states_local,
                          bool is_epilogue = false) {
    // Precompute which orders are present in this emit, and their access_index
    std::unordered_map<int, PrimExpr> order_to_access_index;
    std::unordered_set<int> present_orders;
    for (const auto &nb : new_blocks) {
      order_to_access_index[nb.order] = nb.access_index;
      present_orders.insert(nb.order);
    }
    for (size_t i = 0; i < new_blocks.size(); ++i) {
      // 1. Find the unique async producer stage
      int producer_stage_idx = -1;
      for (const auto &read_region : new_blocks[i].block->reads) {
        for (const auto &[stage, state] : async_states) {
          if (stage <= new_blocks[i].stage &&
              state.writes(read_region->buffer)) {
            // Currently only a single async stage dependency is supported
            ICHECK(producer_stage_idx == -1 || producer_stage_idx == stage)
                << "A dependency on multiple async stages is not supported";
            producer_stage_idx = stage;
          }
        }
      }
      if (producer_stage_idx == -1) {
        // This block does not depend on any async producer
        continue;
      }
      const auto &state = async_states[producer_stage_idx];

      auto &dep_local_state = (*async_states_local)[producer_stage_idx];

      // 2. Use buffer_to_commit_group_ to find all actually dependent commit
      // groups
      std::unordered_set<int> dependent_groups;
      for (const auto &read_region : new_blocks[i].block->reads) {
        auto it = state.buffer_to_commit_group_.find(read_region->buffer.get());
        if (it != state.buffer_to_commit_group_.end()) {
          dependent_groups.insert(it->second);
        }
      }

      // If there is no dependent commit group, no wait needs to be inserted
      if (dependent_groups.empty()) {
        continue;
      }

      // 3. Compute wait = max_g max(0, t_consumer - committed_before[g])
      PrimExpr t_consumer = new_blocks[i].access_index;
      PrimExpr wait_expr = make_zero(t_consumer.dtype());

      PrimExpr current_head = dep_local_state.producer_head.defined()
                                  ? dep_local_state.producer_head.value()
                                  : state.producer_head;
      int consumer_order = new_blocks[i].order;

      for (int g : dependent_groups) {
        const auto &group = state.commit_groups[g];
        if (group.empty())
          continue;
        int commit_order = group.back();
        bool commit_present = present_orders.count(commit_order) > 0;

        PrimExpr committed_before;
        if (commit_present && commit_order <= consumer_order) {
          // Commit point is in this iteration and earlier than the current
          // consumer; this iteration's head is visible
          auto commit_predicate = dep_local_state.commit_predicate;
          if (analyzer_.CanProve(!commit_predicate,
                                 arith::ProofStrength::kSymbolicBound)) {
            // it means the commit block is not executed in this iteration
            committed_before = new_blocks[i].start - 1;
          } else if (is_epilogue) {
            committed_before = new_blocks[i].start - 1;
          } else {
            committed_before = order_to_access_index.at(commit_order);
          }
        } else {
          // Commit point is later than the current consumer or not in this
          // iteration; only the previous iteration's head is visible
          if (dep_local_state.producer_head.defined()) {
            auto commit_predicate = dep_local_state.commit_predicate;
            if (analyzer_.CanProve(!commit_predicate,
                                   arith::ProofStrength::kSymbolicBound)) {
              committed_before = new_blocks[i].start - 1;
            } else if (is_epilogue) {
              committed_before = new_blocks[i].start - 1;
            } else {
              committed_before = current_head - 1;
            }
          }
        }

        wait_expr = analyzer_.Simplify(committed_before - t_consumer);
      }

      wait_expr = analyzer_.Simplify(wait_expr);
      dep_local_state.pending_waits.push_back({static_cast<int>(i), wait_expr});
    }
  }

  // Given pipelined blocks and async-related information, generate final loop
  // statements with async scopes (if any).
  Array<Stmt> CompletePipelineLoopStatements(
      const std::vector<RewrittenBlockInfo> &blocks,
      const std::map<int, AsyncStateLocal> &async_states_local) const {
    std::vector<RewrittenBlockInfo> new_blocks = blocks;
    for (const auto &[stage_id, state] : async_states_local) {
      for (const auto &pw : state.pending_waits) {
        auto &block = new_blocks[pw.insert_before].block;
        BlockNode *n = block.CopyOnWrite();
        auto zero = make_zero(DataType::Int(32));
        n->body = AttrStmt(zero, tir::attr::async_wait_queue_scope, stage_id,
                           AttrStmt(zero, tir::attr::async_wait_inflight_count,
                                    pw.wait_count, n->body));
      }
    }

    // mark the last async stmt as commit
    std::unordered_set<int> commit_group_indices;
    for (const auto &[stage_id, state] : async_states) {
      for (size_t i = 0; i < state.commit_groups.size(); ++i) {
        commit_group_indices.insert(state.commit_groups[i].back());
      }
    }

    Array<Stmt> stmts;

    for (size_t i = 0; i < new_blocks.size(); i++) {
      Block block = new_blocks[i].block;
      if (commit_group_indices.count(new_blocks[i].order)) {
        auto commit_queue_scope = AttrStmt(make_zero(DataType::Int(32)),
                                           tir::attr::async_commit_queue_scope,
                                           new_blocks[i].stage, block->body);
        block = MakeBlock(commit_queue_scope, buffer_data_to_buffer_);
      }
      stmts.push_back(BlockRealize({}, new_blocks[i].predicate, block));
    }

    return stmts;
  }

  /*!
   * \brief Emit the pipeline loop in the given range.
   * \param start The start of the range
   * \param end The end of the range
   * \param unroll_loop Whether the loop should be unrolled.
   * \return The result loop.
   */
  Stmt EmitImpl(const PrimExpr &start, const PrimExpr &end, bool unroll_loop,
                bool need_bound_check, bool is_epilogue = false) {
    PrimExpr new_loop_var;
    PrimExpr extent = end - start;
    auto make_nop = []() {
      return BlockRealize({}, Bool(true), MakeBlock(Evaluate(0), {}));
    };

    bool is_unit_loop = analyzer_.CanProveEqual(extent, 1);
    if (is_unit_loop) {
      new_loop_var = start; // use constants as the loop var for unit loops
    } else {
      new_loop_var = pipeline_loop_->loop_var.copy_with_suffix("");
      // Bind the iteration domain [start, end) to strengthen analyzer facts.
      analyzer_.Bind(Downcast<Var>(new_loop_var),
                     Range::FromMinExtent(start, end - start));
    }
    // Keep the bound constraints active for all analysis below.
    // Only meaningful when the loop var is symbolic (non-unit loop).
    std::unique_ptr<With<arith::ConstraintContext>> ctx_lb_guard;
    std::unique_ptr<With<arith::ConstraintContext>> ctx_ub_guard;
    if (!is_unit_loop) {
      Var loop_iter = Downcast<Var>(new_loop_var);
      ctx_lb_guard.reset(
          new With<arith::ConstraintContext>(&analyzer_, loop_iter >= start));
      ctx_ub_guard.reset(
          new With<arith::ConstraintContext>(&analyzer_, loop_iter < end));
    }

    std::vector<RewrittenBlockInfo> new_blocks;

    // Async related
    std::map<int, AsyncStateLocal> async_states_local;

    for (const Block &block : ordered_stmts_) {
      int stage = pipeline_info_.at(block).stage;
      int order = pipeline_info_.at(block).order;

      PrimExpr inbound = Bool(true);
      PrimExpr skewed_loop_var = new_loop_var - stage;
      if (need_bound_check)
        inbound = And(
            pipeline_loop_->min <= skewed_loop_var,
            (skewed_loop_var < pipeline_loop_->min + pipeline_loop_->extent));

      Block new_block = Downcast<Block>(
          PipelineBodyRewriter(buffer_data_to_buffer_, buffer_remap_,
                               pipeline_loop_, max_stage_ != 1)(block));

      PrimExpr delta = start - pipeline_loop_->min;
      // This variable corresponds to
      // - "producer_head" if this stage is an async producer
      // - "consumer_head" if this stage reads from asynchronously written
      // buffers.
      PrimExpr normalized_access_index =
          is_unit_loop ? skewed_loop_var : skewed_loop_var + delta;

      normalized_access_index = analyzer_.Simplify(normalized_access_index);

      // Adjust the block predicate and the body according to the final loop
      // bound
      //  [pipeline_loop_->min, extent).
      if (!is_unit_loop) {
        Var loop_iter = Downcast<Var>(new_loop_var);
        inbound = Substitute(inbound, {{loop_iter, loop_iter + delta}});
      }
      new_block = Downcast<Block>(Substitute(
          new_block, {{pipeline_loop_->loop_var, normalized_access_index}}));

      // If there were Let-wrappers outside the original pipeline body that
      // depended on the pipeline loop var, push them into each rewritten
      // block with the correct per-block substitution.
      if (!loop_var_let_wrappers_.empty()) {
        BlockNode *n = new_block.CopyOnWrite();
        Stmt inner = n->body;
        for (const auto &lw : loop_var_let_wrappers_) {
          PrimExpr substituted = Substitute(
              lw.value, {{pipeline_loop_->loop_var, normalized_access_index}});
          inner = LetStmt(lw.var, substituted, inner);
        }
        n->body = inner;
      }

      if (pipeline_info_[block].async) {
        auto &local_state = async_states_local[stage];
        local_state.producer_head = normalized_access_index;
        local_state.commit_predicate = inbound;
        BlockNode *n = new_block.CopyOnWrite();
        n->body = AttrStmt(make_zero(DataType::Int(32)), tir::attr::async_scope,
                           1, n->body);
      }

      new_blocks.push_back({stage, order, start, end, inbound, new_block,
                            normalized_access_index,
                            pipeline_info_[block].async});
    }

    PopulateWaitCounts(new_blocks, &async_states_local, is_epilogue);

    auto stmts = CompletePipelineLoopStatements(new_blocks, async_states_local);

    Stmt new_loop{nullptr};

    if (stmts.empty()) {
      return make_nop();
    }

    if (stmts.size() == 1) {
      new_loop = stmts[0];
    } else {
      new_loop = SeqStmt(stmts);
    }

    if (!is_unit_loop) {
      Map<String, Any> preserved_annotations;
      for (const auto &kv : pipeline_loop_->annotations) {
        const String &key = kv.first;
        if (kv.first != tir::attr::software_pipeline_stage &&
            kv.first != tir::attr::software_pipeline_order &&
            kv.first != tir::attr::software_pipeline_async_stages) {
          preserved_annotations.Set(key, kv.second);
        }
      }
      new_loop = For(Downcast<Var>(new_loop_var), pipeline_loop_->min, extent,
                     unroll_loop ? ForKind::kUnrolled : pipeline_loop_->kind,
                     std::move(new_loop), std::nullopt, preserved_annotations);
    }
    // Update producer heads in the global async states.
    for (const auto &[stage_id, state] : async_states_local) {
      async_states[stage_id].producer_head += extent;
    }

    return BlockRealize({}, Bool(true),
                        MakeBlock(new_loop, buffer_data_to_buffer_));
  }

  arith::Analyzer analyzer_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Array<Buffer> pipeline_allocs_;
  Array<Buffer> local_allocs_;
  For pipeline_loop_;
  PipelineInfo pipeline_info_;
  int max_stage_ = -1;
  Map<Buffer, Buffer> buffer_remap_;
  Array<Block> ordered_stmts_;
  std::map<int, AsyncStateGlobal> async_states;
  std::vector<LetWrapper> loop_var_let_wrappers_;
};

/*!
 * \brief Build the dependency graph among a array of blocks.
 * \param[in] blocks The array of blocks.
 * \param[out] dep_src2dst Optional, a map to store dependency edges from the
 * source to the destination. \param[out] dep_dst2src Optional, a map to store
 * dependency edges from the destination to the source.
 */
void BuildDependencyGraph(const Array<Block> &blocks,
                          std::unordered_map<Block, Array<Block>, ObjectPtrHash,
                                             ObjectPtrEqual> *dep_src2dst,
                          std::unordered_map<Block, Array<Block>, ObjectPtrHash,
                                             ObjectPtrEqual> *dep_dst2src) {
  std::unordered_map<Var, Array<Block>, ObjectPtrHash, ObjectPtrEqual>
      buffer_writers;

  for (const Block &block : blocks) {
    for (const BufferRegion &read : block->reads) {
      auto it = buffer_writers.find(read->buffer->data);
      if (it != buffer_writers.end()) {
        for (const Block &writer : it->second) {
          if (dep_src2dst != nullptr) {
            (*dep_src2dst)[writer].push_back(block);
          }
          if (dep_dst2src != nullptr) {
            (*dep_dst2src)[block].push_back(writer);
          }
        }
      }
    }
    for (const BufferRegion &write : block->writes) {
      buffer_writers[write->buffer->data].push_back(block);
    }
  }
}

class PipelineInjector : private StmtExprMutator {
public:
  static Stmt Inject(const PrimFunc &func) {
    auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    PipelineInjector injector(global_symbol);
    for (const auto &kv : func->buffer_map) {
      const Buffer &buffer = kv.second;
      injector.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return injector(func->body);
  }

private:
  explicit PipelineInjector(Optional<String> global_symbol)
      : global_symbol_(std::move(global_symbol)) {}

  /*!
   * \brief Check the pipeline satisfies the following conditions:
   * 1. No conflicting order: The order of each statement should be unique.
   * 2. Reordering of statements doesn't break buffer access dependencies.
   * Specifically, for dependency (e.g. read-after-write) from statement A to
   * statement B, it requires: case 1: stage(A) < stage(B) case 2: stage(A) ==
   * stage(B) and order(A) < order(B)
   */
  void ValidatePipelineBody(const PipelineInfo &pipeline_info,
                            const Array<Block> &original_order) {
    std::unordered_set<int> used_orders;
    std::unordered_map<int, int> stage_max_order;
    std::unordered_map<int, const Block *> order_to_block;
    std::unordered_map<const Block *, int> block_to_stage;
    for (const Block &block : original_order) {
      const auto &stmt_info = pipeline_info.at(block);
      int order = stmt_info.order;
      CHECK(!used_orders.count(order))
          << "ValueError: Two statements in the software pipeline cannot have "
             "the same order";
      used_orders.insert(order);
    }

    std::unordered_map<Block, Array<Block>, ObjectPtrHash, ObjectPtrEqual>
        dep_src2dst;
    BuildDependencyGraph(original_order, &dep_src2dst, nullptr);

    for (const auto &pair : dep_src2dst) {
      const Block &src = pair.first;
      const auto &src_info = pipeline_info.at(src);
      const Array<Block> &dsts = pair.second;
      for (const Block &dst : dsts) {
        const auto &dst_info = pipeline_info.at(dst);
        CHECK_LE(src_info.stage, dst_info.stage)
            << "ValueError: statement " << dst << " in stage " << dst_info.stage
            << " cannot depends on statement " << src << " in a later stage "
            << src_info.stage;
        if (src_info.stage == dst_info.stage) {
          CHECK_LT(src_info.order, dst_info.order)
              << "ValueError: two statements with buffer "
                 "access dependency in the same stage of the "
                 "software pipeline cannot be reordered";
        }
      }
    }
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!HasPipelineAnnotation(op)) {
      return for_node;
    }
    // Step 2: Find the body and buffer allocations of the pipeline. The body
    // can be direct child of the for-loop. If the for-loop has BlockRealize as
    // its child, the pipeline body will be the child of the block.
    Stmt pipeline_body_root{nullptr};
    bool pipeline_body_from_block = false;
    Array<Buffer> pipeline_allocs;
    Array<Buffer>
        block_local_allocs; // buffers allocated in the pipeline block itself
    if (const auto *realize = for_node->body.as<BlockRealizeNode>()) {
      const auto &block = realize->block;
      for (const auto &buffer : block->alloc_buffers) {
        ICHECK(buffer->IsInstance<BufferNode>());
        buffer_data_to_buffer_.Set(buffer->data, buffer);
        allocated_buffers_.insert(buffer);
      }
      pipeline_body_root = block->body;
      block_local_allocs = block->alloc_buffers;
      pipeline_body_from_block = true;
    } else {
      pipeline_body_root = for_node->body;
    }

    const SeqStmtNode *pipeline_body_seq = nullptr;
    std::vector<std::function<Stmt(Stmt)>> rewrap_fns;
    std::vector<LetWrapper> loop_var_let_wrappers;
    auto append_attr_wrapper = [&rewrap_fns](const AttrStmtNode *attr) {
      Any node = attr->node;
      String attr_key = attr->attr_key;
      PrimExpr value = attr->value;
      Span span = attr->span;
      rewrap_fns.emplace_back(
          [node = std::move(node), attr_key = std::move(attr_key),
           value = std::move(value), span](Stmt body) -> Stmt {
            return AttrStmt(node, attr_key, value, body, span);
          });
    };
    {
      Stmt current = pipeline_body_root;
      while (true) {
        if (const auto *seq_stmt = current.as<SeqStmtNode>()) {
          pipeline_body_seq = seq_stmt;
          break;
        }
        if (const auto *if_then_else = current.as<IfThenElseNode>()) {
          ICHECK(!if_then_else->else_case.defined())
              << "InjectSoftwarePipeline: Can't handle the body of the loop "
                 "because the IfThenElse node has an else branch";
          PrimExpr condition = if_then_else->condition;
          Span span = if_then_else->span;
          rewrap_fns.emplace_back(
              [condition = std::move(condition), span](Stmt body) -> Stmt {
                return IfThenElse(condition, body, Stmt(), span);
              });
          current = if_then_else->then_case;
          continue;
        }
        if (const auto *let_stmt = current.as<LetStmtNode>()) {
          // If this Let value uses the pipeline loop var, record it and push
          // inside each rewritten block later so the loop var can be
          // substituted with the correct per-iteration index. Otherwise, keep
          // it as a normal wrapper.
          bool uses_loop_var = UsesVar(
              let_stmt->value,
              [v = op->loop_var.get()](const VarNode *vn) { return vn == v; });
          if (uses_loop_var) {
            loop_var_let_wrappers.push_back({let_stmt->var, let_stmt->value});
          } else {
            Var var = let_stmt->var;
            PrimExpr value = let_stmt->value;
            Span span = let_stmt->span;
            rewrap_fns.emplace_back([var = std::move(var),
                                     value = std::move(value),
                                     span](Stmt body) -> Stmt {
              return LetStmt(var, value, body, span);
            });
          }
          current = let_stmt->body;
          continue;
        }
        if (const auto *attr = current.as<AttrStmtNode>()) {
          append_attr_wrapper(attr);
          current = attr->body;
          continue;
        }
        LOG(FATAL) << "ValueError: The body of the software pipeline should be "
                   << "SeqStmt, got " << current->GetTypeKey();
      }
    }
    ICHECK(pipeline_body_seq != nullptr);

    // Step 3: Blockize the components of the pipeline. Each child of the
    // pipelined loop will be converted into a block.
    PipelineInfo pipeline_info;
    Array<Block> original_order; // pipeline body blocks in the original order

    auto f_add_child = [&](const Stmt &child) {
      original_order.push_back(MakeBlock(child, buffer_data_to_buffer_));
    };
    for (size_t i = 0; i < pipeline_body_seq->seq.size(); i++) {
      const Stmt &child = pipeline_body_seq->seq[i];
      const auto *nested_block_realize = child.as<BlockRealizeNode>();
      if (nested_block_realize && is_one(nested_block_realize->predicate) &&
          nested_block_realize->block->body->IsInstance<SeqStmtNode>()) {
        const Block &nested_pipeline_block = nested_block_realize->block;
        ICHECK(nested_pipeline_block->match_buffers
                   .empty()); // match_buffer should have been lowered
        for (const auto &buffer : nested_pipeline_block->alloc_buffers) {
          buffer_data_to_buffer_.Set(buffer->data, buffer);
          allocated_buffers_.insert(buffer);
        }
      }
      f_add_child(child);
    }

    // Collect all buffers that are actually used in the pipeline loop body.
    // This includes buffers allocated in outer blocks (like logits_smem) that
    // are used inside the pipeline loop.
    BufferUsageCollector collector(buffer_data_to_buffer_, allocated_buffers_);
    pipeline_allocs = collector.Collect(SeqStmt(pipeline_body_seq->seq));

    // Build a set of local allocs (buffers allocated in the pipeline block
    // itself) for efficient lookup
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> local_allocs_set;
    for (const auto &buffer : block_local_allocs) {
      local_allocs_set.insert(buffer);
    }
    for (size_t i = 0; i < pipeline_body_seq->seq.size(); i++) {
      const Stmt &child = pipeline_body_seq->seq[i];
      const auto *nested_block_realize = child.as<BlockRealizeNode>();
      if (nested_block_realize && is_one(nested_block_realize->predicate) &&
          nested_block_realize->block->body->IsInstance<SeqStmtNode>()) {
        for (const auto &buffer : nested_block_realize->block->alloc_buffers) {
          local_allocs_set.insert(buffer);
        }
      }
    }

    // Check if any external buffer (from outer blocks) is already used in
    // another pipeline. This would cause conflicts in multi-versioning.
    for (const auto &buffer : pipeline_allocs) {
      // Only check external buffers (not locally allocated in this pipeline)
      if (local_allocs_set.count(buffer) == 0) {
        CHECK(buffers_used_in_pipeline_.count(buffer) == 0)
            << "Buffer '" << buffer->name
            << "' is used in multiple software pipeline loops. "
            << "This is not supported because multi-versioning would conflict.";
        buffers_used_in_pipeline_.insert(buffer);
      }
    }

    auto pipeline_stages = Downcast<Array<Integer>>(
        op->annotations.at(tir::attr::software_pipeline_stage));
    auto pipeline_orders = Downcast<Array<Integer>>(
        op->annotations.at(tir::attr::software_pipeline_order));
    CHECK_EQ(pipeline_stages.size(), original_order.size())
        << "PrimFunc " << global_symbol_ << " has original order "
        << original_order.Map(
               [](const auto &block) { return block->name_hint; })
        << ", but pipeline annotation is " << pipeline_stages
        << " with different size";
    CHECK_EQ(pipeline_orders.size(), original_order.size())
        << "PrimFunc " << global_symbol_ << " has original order "
        << original_order.Map(
               [](const auto &block) { return block->name_hint; })
        << ", but pipeline annotation is " << pipeline_orders
        << " with different size";

    std::unordered_set<int> pipeline_async_stages;
    if (auto annot =
            op->annotations.Get(tir::attr::software_pipeline_async_stages)) {
      for (auto s : Downcast<Array<Integer>>(annot.value())) {
        pipeline_async_stages.insert(s->value);
      }
    }

    for (size_t i = 0; i < pipeline_stages.size(); i++) {
      int stage = static_cast<int>(pipeline_stages[i]->value);
      bool is_async =
          pipeline_async_stages.find(stage) != pipeline_async_stages.end();
      PipelineAnnotation stage_order{
          stage,
          /*order=*/static_cast<int>(pipeline_orders[i]->value), is_async,
          /*original_idx=*/static_cast<int>(i)};
      pipeline_info.emplace(original_order[i], stage_order);
    }

    ValidatePipelineBody(pipeline_info, original_order);

    // Step 4: Rewrite the pipeline body.
    // local_allocs contains buffers allocated in the pipeline block itself.
    // pipeline_allocs contains all buffers that need multi-versioning,
    // including buffers from outer blocks.
    Array<Buffer> local_allocs = block_local_allocs;
    // Add nested block allocs to local_allocs
    for (size_t i = 0; i < pipeline_body_seq->seq.size(); i++) {
      const Stmt &child = pipeline_body_seq->seq[i];
      const auto *nested_block_realize = child.as<BlockRealizeNode>();
      if (nested_block_realize && is_one(nested_block_realize->predicate) &&
          nested_block_realize->block->body->IsInstance<SeqStmtNode>()) {
        const Block &nested_pipeline_block = nested_block_realize->block;
        for (const auto &buffer : nested_pipeline_block->alloc_buffers) {
          local_allocs.push_back(buffer);
        }
      }
    }

    PipelineRewriter rewriter(buffer_data_to_buffer_, pipeline_allocs,
                              local_allocs, tvm::ffi::GetRef<For>(op),
                              pipeline_info, loop_var_let_wrappers);
    Stmt pipeline = rewriter.BuildPipeline();

    // Store the buffer remapping for updating outer block alloc_buffers
    for (const auto &kv : rewriter.GetBufferRemap()) {
      pending_buffer_remap_.Set(kv.first, kv.second);
    }
    auto apply_wrappers = [&](Stmt stmt) {
      for (auto it = rewrap_fns.rbegin(); it != rewrap_fns.rend(); ++it) {
        stmt = (*it)(stmt);
      }
      return stmt;
    };
    if (!rewrap_fns.empty()) {
      if (pipeline_body_from_block) {
        BlockRealize pipeline_realize = Downcast<BlockRealize>(pipeline);
        Block pipeline_block = pipeline_realize->block;
        {
          BlockNode *block_node = pipeline_block.CopyOnWrite();
          block_node->body = apply_wrappers(block_node->body);
        }
        pipeline = BlockRealize(pipeline_realize->iter_values,
                                pipeline_realize->predicate, pipeline_block,
                                pipeline_realize->span);
      } else {
        pipeline = apply_wrappers(pipeline);
      }
    }

    if (const auto *realize = op->body.as<BlockRealizeNode>()) {
      const auto &block = realize->block;
      for (const auto &buffer : block->alloc_buffers) {
        buffer_data_to_buffer_.erase(buffer->data);
        allocated_buffers_.erase(buffer);
      }
    }
    return pipeline;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
      allocated_buffers_.insert(buffer);
    }

    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

    // Update alloc_buffers with any pending buffer remaps from pipeline
    // rewriting. This handles buffers allocated in this block but
    // multi-versioned during pipeline rewriting of inner loops.
    Array<Buffer> new_alloc_buffers;
    for (const auto &buffer : block->alloc_buffers) {
      if (auto remapped = pending_buffer_remap_.Get(buffer)) {
        new_alloc_buffers.push_back(remapped.value());
        // Remove from pending after applying
        pending_buffer_remap_.erase(buffer);
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }

    Array<Array<BufferRegion>> access =
        GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    BlockNode *n = block.CopyOnWrite();
    n->reads = access[0];
    n->writes = access[1];
    n->alloc_buffers = std::move(new_alloc_buffers);

    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
      allocated_buffers_.erase(buffer);
    }
    return block;
  }

  bool HasPipelineAnnotation(const ForNode *op) const {
    auto it1 = op->annotations.find(tir::attr::software_pipeline_stage);
    auto it2 = op->annotations.find(tir::attr::software_pipeline_order);
    bool has_stage = it1 != op->annotations.end();
    bool has_order = it2 != op->annotations.end();
    if (has_stage && has_order) {
      return true;
    }
    if (has_stage) {
      LOG(FATAL)
          << "ValueError: Stage of the software pipeline is not defined.";
    }
    if (has_order) {
      LOG(FATAL)
          << "ValueError: Order of the software pipeline is not defined.";
    }
    return false;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocated_buffers_;
  Map<Buffer, Buffer> pending_buffer_remap_;
  // Buffers from outer blocks that have been used in a pipeline loop.
  // Used to detect if the same buffer is used in multiple pipeline loops.
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffers_used_in_pipeline_;
  Optional<String> global_symbol_;
};
} // namespace software_pipeline

/*!
 * \brief Transform annotated loops into pipelined one that parallelize
 * producers and consumers. \return The IR transform pass.
 */
tir::transform::Pass InjectSoftwarePipeline() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *fptr = f.CopyOnWrite();
    fptr->body = software_pipeline::PipelineInjector::Inject(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSoftwarePipeline", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectSoftwarePipeline",
                        InjectSoftwarePipeline);
}

} // namespace tl
} // namespace tvm
