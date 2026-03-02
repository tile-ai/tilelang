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

struct IfWrapper {
  PrimExpr condition;
  Span span;
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
    return Call(call->dtype, call->op, new_args, call->annotations, call->span);
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
   * \param loop_var_if_wrappers If wrappers with conditions that depend on
   * the loop var.
   */
  PipelineRewriter(Map<Var, Buffer> buffer_data_to_buffer,
                   const Array<Buffer> &pipeline_allocs,
                   const Array<Buffer> &local_allocs, const For &pipeline_loop,
                   const PipelineInfo &pipeline_info,
                   const std::vector<LetWrapper> &loop_var_let_wrappers,
                   const std::vector<IfWrapper> &loop_var_if_wrappers)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        pipeline_allocs_(pipeline_allocs), local_allocs_(local_allocs),
        pipeline_loop_(pipeline_loop), pipeline_info_(pipeline_info),
        loop_var_let_wrappers_(loop_var_let_wrappers),
        loop_var_if_wrappers_(loop_var_if_wrappers) {}

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

    // Step 2: Emit the pipeline prologue, body and epilogue.
    Stmt prologue = EmitImpl(pipeline_loop_->min,
                             pipeline_loop_->min + max_stage_, true, true);
    Stmt body =
        EmitImpl(pipeline_loop_->min + max_stage_,
                 pipeline_loop_->min + pipeline_loop_->extent, false, false);

    Stmt epilogue = EmitImpl(
        pipeline_loop_->min + pipeline_loop_->extent,
        pipeline_loop_->min + pipeline_loop_->extent + max_stage_, true, true);
    SeqStmt stmt = SeqStmt({prologue, body, epilogue});

    // Step 3: Make a new block that contains new buffer allocations after
    // pipeline rewriting.
    // Only include buffers that are locally allocated in the pipeline block.
    // Buffers from outer blocks will be handled separately.
    Array<Buffer> alloc_buffers;
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

  /*! Structure holding intermediate information for pipeline loop rewriting. */
  struct RewrittenBlockInfo {
    PrimExpr predicate;
    Block block;
  };

  /*!
   * \brief Emit the pipeline loop in the given range.
   * \param start The start of the range
   * \param end The end of the range
   * \param unroll_loop Whether the loop should be unrolled.
   * \return The result loop.
   */
  Stmt EmitImpl(const PrimExpr &start, const PrimExpr &end, bool unroll_loop,
                bool need_bound_check) {
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

    for (const Block &block : ordered_stmts_) {
      int stage = pipeline_info_.at(block).stage;
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
      // We iterate in reverse order so that earlier definitions scope over
      // later ones. For example, if we have:
      //   id = ids[i]       # depends on loop var
      //   id2 = ids2[id]    # depends on id
      // We want to produce:
      //   LetStmt(id, ids[...],
      //     LetStmt(id2, ids2[id],
      //       body))
      // So that id2's definition can reference id.
      if (!loop_var_let_wrappers_.empty()) {
        BlockNode *n = new_block.CopyOnWrite();
        Stmt inner = n->body;
        for (auto it = loop_var_let_wrappers_.rbegin();
             it != loop_var_let_wrappers_.rend(); ++it) {
          const auto &lw = *it;
          PrimExpr substituted = Substitute(
              lw.value, {{pipeline_loop_->loop_var, normalized_access_index}});
          inner = LetStmt(lw.var, substituted, inner);
        }
        n->body = inner;
      }

      // Similarly, handle If-wrappers whose conditions depend on the
      // pipeline loop var.
      if (!loop_var_if_wrappers_.empty()) {
        BlockNode *n = new_block.CopyOnWrite();
        Stmt inner = n->body;
        for (auto it = loop_var_if_wrappers_.rbegin();
             it != loop_var_if_wrappers_.rend(); ++it) {
          const auto &iw = *it;
          PrimExpr substituted_condition =
              Substitute(iw.condition,
                         {{pipeline_loop_->loop_var, normalized_access_index}});
          inner = IfThenElse(substituted_condition, inner, Stmt(), iw.span);
        }
        n->body = inner;
      }

      new_blocks.push_back({inbound, new_block});
    }

    Array<Stmt> stmts;
    for (const auto &block_info : new_blocks) {
      stmts.push_back(BlockRealize({}, block_info.predicate, block_info.block));
    }

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
  std::vector<LetWrapper> loop_var_let_wrappers_;
  std::vector<IfWrapper> loop_var_if_wrappers_;
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
    std::vector<IfWrapper> loop_var_if_wrappers;
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

          // Check if the condition depends on the loop variable or any
          // transitively dependent variables (similar to LetStmt handling)
          std::unordered_set<const VarNode *> dependent_vars;
          dependent_vars.insert(op->loop_var.get());
          for (const auto &lw : loop_var_let_wrappers) {
            dependent_vars.insert(lw.var.get());
          }
          bool condition_depends_on_loop = UsesVar(
              if_then_else->condition, [&dependent_vars](const VarNode *vn) {
                return dependent_vars.count(vn) > 0;
              });

          if (condition_depends_on_loop) {
            // If condition depends on loop variable, we need to push it inside
            // each pipeline stage with proper substitution
            loop_var_if_wrappers.push_back(
                {if_then_else->condition, if_then_else->span});
          } else {
            // Otherwise, safe to wrap outside the pipeline
            PrimExpr condition = if_then_else->condition;
            Span span = if_then_else->span;
            rewrap_fns.emplace_back(
                [condition = std::move(condition), span](Stmt body) -> Stmt {
                  return IfThenElse(condition, body, Stmt(), span);
                });
          }
          current = if_then_else->then_case;
          continue;
        }
        if (const auto *let_stmt = current.as<LetStmtNode>()) {
          // If this Let value uses the pipeline loop var OR any variable
          // defined by a previously recorded loop-var-dependent LetStmt,
          // record it and push inside each rewritten block later so the
          // loop var can be substituted with the correct per-iteration index.
          // Otherwise, keep it as a normal wrapper.
          // This handles transitive dependencies like:
          //   id = ids[i]      # depends on loop var
          //   id2 = ids2[id]   # depends on id, so transitively on loop var
          std::unordered_set<const VarNode *> dependent_vars;
          dependent_vars.insert(op->loop_var.get());
          for (const auto &lw : loop_var_let_wrappers) {
            dependent_vars.insert(lw.var.get());
          }
          bool depends_on_loop =
              UsesVar(let_stmt->value, [&dependent_vars](const VarNode *vn) {
                return dependent_vars.count(vn) > 0;
              });
          if (depends_on_loop) {
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

    for (size_t i = 0; i < pipeline_stages.size(); i++) {
      int stage = static_cast<int>(pipeline_stages[i]->value);
      PipelineAnnotation stage_order{
          stage, /*order=*/static_cast<int>(pipeline_orders[i]->value)};
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
                              pipeline_info, loop_var_let_wrappers,
                              loop_var_if_wrappers);
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
