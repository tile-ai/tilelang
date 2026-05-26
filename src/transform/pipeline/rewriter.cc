#include "rewriter.h"

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common/pipeline_utils.h"
#include "op/region.h"
#include "tir/schedule/utils.h"

namespace tvm {
namespace tl {
namespace software_pipeline {

using namespace tirx;
using namespace ffi;

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
      auto buffer_var = Downcast<Var>(call->args[i]);
      auto buf_it = buffer_data_to_buffer_.find(buffer_var);
      if (buf_it == buffer_data_to_buffer_.end()) {
        continue;
      }
      const Buffer &buffer = (*buf_it).second;
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
            floormod((pipeline_loop_->loop_var - pipeline_loop_->min),
                     new_buffer->shape[0]) *
                offset;
        new_args.Set(i + 1, new_index);
      }
    }
    return Call(call->dtype, call->op, new_args, call->annotations, call->span);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &alloc_buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(alloc_buffer->data, alloc_buffer);
    }
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    SBlockNode *n = block.CopyOnWrite();
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
    if (call->op.same_as(RegionOp::Get()) && call->args.size() >= 2) {
      if (auto load = call->args[0].as<BufferLoadNode>()) {
        size_t num_extents = call->args.size() - 2;
        if (load->indices.size() == num_extents + 1) {
          Array<PrimExpr> new_args;
          new_args.push_back(call->args[0]);
          new_args.push_back(call->args[1]);
          new_args.push_back(IntImm(DataType::Int(32), 1));
          for (size_t i = 2; i < call->args.size(); ++i) {
            new_args.push_back(call->args[i]);
          }
          return Call(call->dtype, call->op, new_args, call->annotations,
                      call->span);
        }
      }
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
   * \param scalar_binding_blocks Replayable scalar Bind statements from the
   * pipeline body.
   */
  PipelineRewriter(Map<Var, Buffer> buffer_data_to_buffer,
                   const Array<Buffer> &pipeline_allocs,
                   const Array<Buffer> &local_allocs, const For &pipeline_loop,
                   const PipelineInfo &pipeline_info,
                   const Array<SBlock> &scalar_binding_blocks,
                   Optional<Target> target)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        pipeline_allocs_(pipeline_allocs), local_allocs_(local_allocs),
        pipeline_loop_(pipeline_loop), pipeline_info_(pipeline_info),
        scalar_binding_blocks_(scalar_binding_blocks),
        target_(std::move(target)) {}

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
    std::vector<std::pair<int, SBlock>> ordered_blocks;
    for (const auto &[block, anno] : pipeline_info_) {
      ordered_blocks.emplace_back(anno.order, block);
    }
    std::sort(
        ordered_blocks.begin(), ordered_blocks.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (const auto &[_, block] : ordered_blocks) {
      ordered_stmts_.push_back(block);
    }
    CollectScalarBindings();

    // Step 2: Emit the pipeline prologue, body and epilogue.
    Optional<Integer> pipeline_num_stages =
        GetPipelineNumStages(pipeline_loop_.get());
    Stmt prologue = StripPipelineContextAttrs(EmitImpl(
        pipeline_loop_->min, pipeline_loop_->min + max_stage_, true, true));
    Stmt body = StripPipelineContextAttrs(
        EmitImpl(pipeline_loop_->min + max_stage_,
                 pipeline_loop_->min + pipeline_loop_->extent, false, false));
    Stmt epilogue = StripPipelineContextAttrs(EmitImpl(
        pipeline_loop_->min + pipeline_loop_->extent,
        pipeline_loop_->min + pipeline_loop_->extent + max_stage_, true, true));

    Array<Stmt> pipeline_parts;
    for (const Stmt &part : {prologue, body, epilogue}) {
      for (const Stmt &stmt : FlattenTopLevelSeq(part)) {
        pipeline_parts.push_back(stmt);
      }
    }

    Stmt stmt = pipeline_parts.size() == 1 ? pipeline_parts[0]
                                           : SeqStmt(pipeline_parts);
    stmt = AsyncPipelineLoopWaitRelaxer(this)(stmt);
    Array<Stmt> relaxed_pipeline_parts = FlattenTopLevelSeq(stmt);
    relaxed_pipeline_parts =
        RelaxTrailingConsumerWaits(std::move(relaxed_pipeline_parts),
                                   PipelinedRetainGroups(pipeline_num_stages));
    stmt = relaxed_pipeline_parts.size() == 1 ? relaxed_pipeline_parts[0]
                                              : SeqStmt(relaxed_pipeline_parts);

    // Step 3: Make a new block that contains new buffer allocations after
    // pipeline rewriting.
    // Only include buffers that are locally allocated in the pipeline block.
    // Buffers from outer blocks will be handled separately.
    Array<Buffer> alloc_buffers;
    for (const auto &alloc : local_allocs_) {
      alloc_buffers.push_back(buffer_remap_.Get(alloc).value_or(alloc));
      buffer_data_to_buffer_.erase(alloc->data);
    }
    if (pipeline_num_stages) {
      if (pipeline_num_stages.value().IntValue() > 1) {
        stmt = AttrStmt(Integer(0), kPipelineMVBContextNumStages,
                        Downcast<PrimExpr>(pipeline_num_stages.value()), stmt);
      }
      stmt = AttrStmt(Integer(0), kPipelineContextNumStages,
                      Downcast<PrimExpr>(pipeline_num_stages.value()), stmt);
    }
    SBlock block = MakeBlock(stmt, buffer_data_to_buffer_);
    block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
    return SBlockRealize({}, Bool(true), block);
  }

  /*!
   * \brief Get the buffer remapping created during pipeline rewriting.
   * This is used to update alloc_buffers in outer blocks.
   */
  const Map<Buffer, Buffer> &GetBufferRemap() const { return buffer_remap_; }

private:
  struct ScalarBinding {
    Var var;
    PrimExpr value;
    Span span;
  };

  using ScalarBindingMap =
      std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual>;

  void CollectScalarBindings() {
    scalar_bindings_.clear();
    scalar_binding_map_.clear();
    for (const SBlock &block : scalar_binding_blocks_) {
      if (const auto *bind = block->body.as<BindNode>()) {
        if (!scalar_binding_map_.count(bind->var)) {
          scalar_binding_map_.emplace(bind->var, scalar_bindings_.size());
          scalar_bindings_.push_back({bind->var, bind->value, bind->span});
        }
      }
    }
  }

  VarSet FindScalarBindingUses(const Array<Var> &undefined_vars) const {
    VarSet uses;
    for (const Var &var : undefined_vars) {
      if (scalar_binding_map_.count(var)) {
        uses.insert(var);
      }
    }
    return uses;
  }

  VarSet FindScalarBindingUses(const Stmt &stmt) const {
    return FindScalarBindingUses(UndefinedVars(stmt, Array<Var>{}));
  }

  VarSet FindScalarBindingUses(const PrimExpr &expr) const {
    return FindScalarBindingUses(UndefinedVars(expr));
  }

  void AppendScalarBinding(size_t binding_index, VarSet *emitted,
                           VarSet *visiting,
                           std::vector<size_t> *binding_indices) const {
    const ScalarBinding &binding = scalar_bindings_[binding_index];
    if (emitted->count(binding.var)) {
      return;
    }
    ICHECK(!visiting->count(binding.var))
        << "InjectSoftwarePipeline: cyclic scalar Bind dependency involving "
        << binding.var;

    visiting->insert(binding.var);
    VarSet deps = FindScalarBindingUses(binding.value);
    for (const ScalarBinding &candidate : scalar_bindings_) {
      if (!deps.count(candidate.var)) {
        continue;
      }
      auto it = scalar_binding_map_.find(candidate.var);
      ICHECK(it != scalar_binding_map_.end());
      AppendScalarBinding(it->second, emitted, visiting, binding_indices);
    }
    visiting->erase(binding.var);

    emitted->insert(binding.var);
    binding_indices->push_back(binding_index);
  }

  std::vector<size_t> RequiredScalarBindings(const Stmt &stmt) const {
    std::vector<size_t> binding_indices;
    if (scalar_bindings_.empty()) {
      return binding_indices;
    }

    VarSet uses = FindScalarBindingUses(stmt);
    if (uses.empty()) {
      return binding_indices;
    }

    VarSet emitted;
    VarSet visiting;
    for (const ScalarBinding &binding : scalar_bindings_) {
      if (!uses.count(binding.var)) {
        continue;
      }
      auto it = scalar_binding_map_.find(binding.var);
      ICHECK(it != scalar_binding_map_.end());
      AppendScalarBinding(it->second, &emitted, &visiting, &binding_indices);
    }
    return binding_indices;
  }

  Stmt RewriteScalarBindingForAccess(size_t binding_index,
                                     const PrimExpr &access_index) {
    const ScalarBinding &binding = scalar_bindings_[binding_index];
    Stmt bind = Bind(binding.var, binding.value, binding.span);
    bind = PipelineBodyRewriter(buffer_data_to_buffer_, buffer_remap_,
                                pipeline_loop_, max_stage_ != 1)(bind);
    bind = Substitute(bind, {{pipeline_loop_->loop_var, access_index}});
    return bind;
  }

  SBlock ReplayScalarBindings(SBlock block, const PrimExpr &access_index) {
    std::vector<size_t> binding_indices = RequiredScalarBindings(block->body);
    if (binding_indices.empty()) {
      return block;
    }

    Array<Stmt> seq;
    for (size_t binding_index : binding_indices) {
      seq.push_back(RewriteScalarBindingForAccess(binding_index, access_index));
    }
    for (const Stmt &stmt : FlattenTopLevelSeq(block->body)) {
      seq.push_back(stmt);
    }

    SBlockNode *n = block.CopyOnWrite();
    n->body = SeqStmt(seq);
    return MakeBlock(SBlockRealize({}, Bool(true), block),
                     buffer_data_to_buffer_);
  }

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
      const SBlock &block = pair.first;
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
        const SBlock &writer_block = pair1.first;
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
          const SBlock &reader_block = pair2.first;
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
    ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*(buffer.get()));
    new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));
    if (!new_buffer->strides.empty()) {
      ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
      PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
      new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
    }
    return Buffer(new_buffer);
  }

  struct AsyncStateGlobal {
    BufferSet dst_buffers;
    Optional<PrimExpr> producer_head{PrimExpr(-1)};

    bool writes(const Buffer &buffer) const {
      return dst_buffers.count(buffer) > 0;
    }
  };

  struct AsyncStateLocal {
    struct PendingWait {
      int insert_before{-1};
      PrimExpr wait_count{nullptr};

      bool valid() const { return wait_count.defined(); }
    };

    BufferSet seen;
    Optional<PrimExpr> producer_head;
    Optional<PrimExpr> predicate;
    std::vector<std::vector<size_t>> commit_groups;
    std::map<int, PendingWait> pending_waits;
    std::unordered_map<int, int> annotated_group_to_commit_group;
    bool consumed{false};
  };

  struct RewrittenStmtInfo {
    int stage;
    PrimExpr predicate;
    Array<BufferRegion> reads;
    Array<BufferRegion> writes;
    PrimExpr access_index;
    bool is_async;
    Stmt stmt;
  };

  struct FinalStmtInfo {
    int stage;
    PrimExpr access_index;
    PrimExpr predicate;
    Stmt stmt;
  };

  enum class AsyncSyncStmtKind { kOther, kCommit, kWaitStatic, kWaitDynamic };

  struct ClassifiedAsyncSyncStmt {
    AsyncSyncStmtKind kind{AsyncSyncStmtKind::kOther};
    int wait_n{0};
  };

  struct AsyncSyncSummary {
    int commit{0};
    int wait{0};
  };

  enum class HeadAsyncSyncKind {
    kNone,
    kCommit,
    kWaitStatic,
    kWaitDynamic,
    kBlocked,
  };

  struct HeadAsyncSyncInfo {
    HeadAsyncSyncKind kind{HeadAsyncSyncKind::kNone};
    int wait_n{0};

    bool IsBoundary() const {
      return kind == HeadAsyncSyncKind::kCommit ||
             kind == HeadAsyncSyncKind::kWaitDynamic ||
             kind == HeadAsyncSyncKind::kBlocked;
    }
  };

  enum class HeadSeqMode {
    kSingletonOnly,
    kTakeFirstElement,
  };

  struct DeterministicNoWaitCommitEffect {
    bool deterministic{true};
    bool has_wait{false};
    int commit_groups{0};

    static DeterministicNoWaitCommitEffect Unknown() {
      DeterministicNoWaitCommitEffect effect;
      effect.deterministic = false;
      return effect;
    }

    static DeterministicNoWaitCommitEffect Wait() {
      DeterministicNoWaitCommitEffect effect;
      effect.has_wait = true;
      return effect;
    }
  };

  // Analyze a stmt for one specific question used by wait relaxation:
  // can we prove that it contributes a deterministic number of commit groups
  // without crossing a wait boundary? The analyzer exposes the effect as
  // structured state instead of overloading std::optional<int> with both
  // "unknown" and "has wait" meanings.
  class DeterministicNoWaitCommitAnalyzer {
  public:
    explicit DeterministicNoWaitCommitAnalyzer(const PipelineRewriter *rewriter)
        : rewriter_(rewriter) {}

    DeterministicNoWaitCommitEffect Analyze(const Stmt &stmt) const {
      if (stmt.as<BindNode>()) {
        return DeterministicNoWaitCommitEffect{};
      }
      if (const auto *attr = stmt.as<AttrStmtNode>()) {
        return AnalyzeAttr(attr);
      }
      if (const auto *seq = stmt.as<SeqStmtNode>()) {
        DeterministicNoWaitCommitEffect effect;
        for (const Stmt &s : seq->seq) {
          effect = Combine(effect, Analyze(s));
          if (!effect.deterministic) {
            return effect;
          }
        }
        return effect;
      }
      if (const auto *block = stmt.as<SBlockNode>()) {
        return Analyze(block->body);
      }
      if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
        if (!is_one(realize->predicate)) {
          return DeterministicNoWaitCommitEffect::Unknown();
        }
        return Analyze(realize->block->body);
      }
      if (const auto *for_node = stmt.as<ForNode>()) {
        return AnalyzeFor(for_node);
      }
      if (stmt.as<IfThenElseNode>()) {
        return DeterministicNoWaitCommitEffect::Unknown();
      }
      if (rewriter_->ContainsAsyncSyncScopes(stmt)) {
        return DeterministicNoWaitCommitEffect::Unknown();
      }
      return {};
    }

  private:
    DeterministicNoWaitCommitEffect
    AnalyzeAttr(const AttrStmtNode *attr) const {
      if (PipelineRewriter::IsAsyncWaitQueueScope(attr) ||
          PipelineRewriter::IsAsyncWaitInflightCount(attr)) {
        return DeterministicNoWaitCommitEffect::Wait();
      }
      if (PipelineRewriter::IsAsyncCommitQueueScope(attr)) {
        auto effect = Analyze(attr->body);
        if (!effect.deterministic) {
          return effect;
        }
        ++effect.commit_groups;
        return effect;
      }
      return Analyze(attr->body);
    }

    DeterministicNoWaitCommitEffect AnalyzeFor(const ForNode *for_node) const {
      if (for_node->thread_binding.defined()) {
        return DeterministicNoWaitCommitEffect::Unknown();
      }
      const int64_t *extent_imm = as_const_int(for_node->extent);
      if (extent_imm == nullptr || *extent_imm < 0) {
        return DeterministicNoWaitCommitEffect::Unknown();
      }
      auto effect = Analyze(for_node->body);
      if (!effect.deterministic) {
        return effect;
      }
      effect.commit_groups *= static_cast<int>(*extent_imm);
      return effect;
    }

    static DeterministicNoWaitCommitEffect
    Combine(const DeterministicNoWaitCommitEffect &lhs,
            const DeterministicNoWaitCommitEffect &rhs) {
      if (!lhs.deterministic || !rhs.deterministic) {
        return DeterministicNoWaitCommitEffect::Unknown();
      }
      DeterministicNoWaitCommitEffect effect;
      effect.has_wait = lhs.has_wait || rhs.has_wait;
      effect.commit_groups = lhs.commit_groups + rhs.commit_groups;
      return effect;
    }

    const PipelineRewriter *rewriter_;
  };

  Stmt WrapPipelineStageContext(Stmt stmt,
                                const PrimExpr &normalized_access_index,
                                const Optional<Integer> &pipeline_num_stages) {
    if (!(pipeline_num_stages && pipeline_num_stages.value().IntValue() > 1)) {
      return stmt;
    }
    PrimExpr ns =
        IntImm(DataType::Int(32), pipeline_num_stages.value().IntValue());
    PrimExpr stage_expr =
        analyzer_.Simplify(FloorMod(normalized_access_index, ns));
    PrimExpr parity_expr = analyzer_.Simplify(FloorMod(
        FloorDiv(normalized_access_index, ns), IntImm(DataType::Int(32), 2)));
    stmt = AttrStmt(Integer(0), kPipelineMVBParityExpr, parity_expr, stmt);
    stmt = AttrStmt(Integer(0), kPipelineMVBStageExpr, stage_expr, stmt);
    return stmt;
  }

  Optional<PrimExpr>
  ComputePipelineMbarPhaseExpr(const PrimExpr &normalized_access_index,
                               const Optional<Integer> &pipeline_num_stages) {
    if (!pipeline_num_stages) {
      return Optional<PrimExpr>();
    }
    PrimExpr parity_expr;
    if (pipeline_num_stages.value().IntValue() <= 1) {
      parity_expr =
          FloorMod(normalized_access_index, IntImm(DataType::Int(32), 2));
    } else {
      PrimExpr ns =
          IntImm(DataType::Int(32), pipeline_num_stages.value().IntValue());
      parity_expr = FloorMod(FloorDiv(normalized_access_index, ns),
                             IntImm(DataType::Int(32), 2));
    }
    return analyzer_.Simplify(parity_expr);
  }

  static bool IsAsyncCommitQueueScope(const AttrStmtNode *attr) {
    return attr && attr->attr_key == s_tir::attr::async_commit_queue_scope;
  }

  static bool IsAsyncWaitQueueScope(const AttrStmtNode *attr) {
    return attr && attr->attr_key == s_tir::attr::async_wait_queue_scope;
  }

  static bool IsAsyncWaitInflightCount(const AttrStmtNode *attr) {
    return attr && attr->attr_key == s_tir::attr::async_wait_inflight_count;
  }

  static int
  PipelinedRetainGroups(const Optional<Integer> &pipeline_num_stages) {
    int retain = 1;
    if (pipeline_num_stages) {
      retain = std::max(
          0, static_cast<int>(pipeline_num_stages.value().IntValue()) - 1);
    }
    return retain;
  }

  Stmt StripPipelineContextAttrs(Stmt stmt) const {
    while (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key != kPipelineContextNumStages &&
          attr->attr_key != kPipelineMVBContextNumStages) {
        break;
      }
      stmt = attr->body;
    }
    return stmt;
  }

  Array<Stmt> FlattenTopLevelSeq(const Stmt &stmt) const {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      return seq->seq;
    }
    return {stmt};
  }

  std::optional<int>
  TryGetStaticAsyncWaitCount(const AttrStmtNode *attr) const {
    if (!IsAsyncWaitQueueScope(attr)) {
      return std::nullopt;
    }
    const auto *inner = attr->body.as<AttrStmtNode>();
    if (!IsAsyncWaitInflightCount(inner)) {
      return std::nullopt;
    }
    const int64_t *imm = as_const_int(inner->value);
    if (!imm) {
      return std::nullopt;
    }
    return static_cast<int>(*imm);
  }

  Stmt MakeStaticAsyncWaitStmtLike(const AttrStmtNode *attr,
                                   int new_wait_n) const {
    const auto *inner = attr->body.as<AttrStmtNode>();
    if (!IsAsyncWaitInflightCount(inner)) {
      return AttrStmt(attr->node, attr->attr_key, attr->value, attr->body,
                      attr->span);
    }
    PrimExpr new_wait = make_const(inner->value.dtype(), new_wait_n);
    Stmt new_inner = AttrStmt(inner->node, inner->attr_key, new_wait,
                              inner->body, inner->span);
    return AttrStmt(attr->node, attr->attr_key, attr->value, new_inner,
                    attr->span);
  }

  class HeadAsyncSyncAnalyzer
      : public StmtFunctor<HeadAsyncSyncInfo(const Stmt &)> {
  public:
    static HeadAsyncSyncInfo Analyze(const PipelineRewriter *rewriter,
                                     const Stmt &stmt, HeadSeqMode seq_mode) {
      HeadAsyncSyncAnalyzer analyzer(rewriter, seq_mode);
      return analyzer(stmt);
    }

    HeadAsyncSyncAnalyzer(const PipelineRewriter *rewriter,
                          HeadSeqMode seq_mode)
        : rewriter_(rewriter), seq_mode_(seq_mode) {}

    HeadAsyncSyncInfo VisitStmt_(const AttrStmtNode *op) final {
      if (IsAsyncWaitQueueScope(op)) {
        if (auto wait_n = rewriter_->TryGetStaticAsyncWaitCount(op)) {
          return {HeadAsyncSyncKind::kWaitStatic, *wait_n};
        }
        return {HeadAsyncSyncKind::kWaitDynamic, 0};
      }
      if (IsAsyncCommitQueueScope(op)) {
        return {HeadAsyncSyncKind::kCommit, 0};
      }
      if (IsAsyncWaitInflightCount(op)) {
        return {HeadAsyncSyncKind::kBlocked, 0};
      }
      return VisitStmt(op->body);
    }

    HeadAsyncSyncInfo VisitStmt_(const SeqStmtNode *op) final {
      if (op->seq.empty()) {
        return {};
      }
      if (seq_mode_ == HeadSeqMode::kSingletonOnly && op->seq.size() != 1) {
        return {HeadAsyncSyncKind::kBlocked, 0};
      }
      return VisitStmt(op->seq[0]);
    }

    HeadAsyncSyncInfo VisitStmt_(const SBlockNode *op) final {
      return VisitStmt(op->body);
    }

    HeadAsyncSyncInfo VisitStmt_(const SBlockRealizeNode *op) final {
      if (is_one(op->predicate)) {
        return VisitStmt(op->block->body);
      }
      return {HeadAsyncSyncKind::kBlocked, 0};
    }

    HeadAsyncSyncInfo VisitStmtDefault_(const Object *) final { return {}; }

  private:
    const PipelineRewriter *rewriter_;
    HeadSeqMode seq_mode_;
  };

  HeadAsyncSyncInfo AnalyzeHeadAsyncSync(const Stmt &stmt,
                                         HeadSeqMode seq_mode) const {
    return HeadAsyncSyncAnalyzer::Analyze(this, stmt, seq_mode);
  }

  ClassifiedAsyncSyncStmt ClassifySimpleAsyncSyncStmt(const Stmt &stmt) const {
    HeadAsyncSyncInfo info =
        AnalyzeHeadAsyncSync(stmt, HeadSeqMode::kSingletonOnly);
    switch (info.kind) {
    case HeadAsyncSyncKind::kCommit:
      return {AsyncSyncStmtKind::kCommit, 0};
    case HeadAsyncSyncKind::kWaitStatic:
      return {AsyncSyncStmtKind::kWaitStatic, info.wait_n};
    case HeadAsyncSyncKind::kWaitDynamic:
      return {AsyncSyncStmtKind::kWaitDynamic, 0};
    default:
      return {};
    }
  }

  bool ContainsAsyncSyncScopes(const Stmt &stmt) const {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &obj) {
      if (found) {
        return;
      }
      if (const auto *attr = obj.as<AttrStmtNode>()) {
        if (IsAsyncCommitQueueScope(attr) || IsAsyncWaitQueueScope(attr)) {
          found = true;
        }
      }
    });
    return found;
  }

  bool ContainsAsyncCommitScopes(const Stmt &stmt) const {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &obj) {
      if (found) {
        return;
      }
      if (const auto *attr = obj.as<AttrStmtNode>()) {
        if (IsAsyncCommitQueueScope(attr)) {
          found = true;
        }
      }
    });
    return found;
  }

  AsyncSyncSummary SummarizeAsyncSyncScopes(const Stmt &stmt) const {
    AsyncSyncSummary summary;
    PostOrderVisit(stmt, [&](const ObjectRef &obj) {
      if (const auto *attr = obj.as<AttrStmtNode>()) {
        if (IsAsyncCommitQueueScope(attr)) {
          ++summary.commit;
        } else if (IsAsyncWaitQueueScope(attr)) {
          ++summary.wait;
        }
      }
    });
    return summary;
  }

  std::optional<int>
  TryGetDeterministicNoWaitCommitGroups(const Stmt &stmt) const {
    auto effect = DeterministicNoWaitCommitAnalyzer(this).Analyze(stmt);
    if (!effect.deterministic || effect.has_wait) {
      return std::nullopt;
    }
    return effect.commit_groups;
  }

  int GuaranteedNewGroupsBeforeNextWait(const Array<Stmt> &body,
                                        int start_idx) const {
    int guaranteed_groups = 0;
    for (int i = start_idx, n = static_cast<int>(body.size()); i < n; ++i) {
      AsyncSyncSummary summary = SummarizeAsyncSyncScopes(body[i]);
      if (summary.wait > 0) {
        break;
      }
      if (summary.commit == 0) {
        continue;
      }
      if (auto commits = TryGetDeterministicNoWaitCommitGroups(body[i])) {
        guaranteed_groups += *commits;
        continue;
      }
      break;
    }
    return guaranteed_groups;
  }

  class WaitStaticInSimpleWrapperRewriter
      : public StmtFunctor<Optional<Stmt>(const Stmt &)> {
  public:
    static Optional<Stmt> Rewrite(const PipelineRewriter *rewriter,
                                  const Stmt &stmt, int new_wait_n) {
      if (rewriter->ClassifySimpleAsyncSyncStmt(stmt).kind !=
          AsyncSyncStmtKind::kWaitStatic) {
        return Optional<Stmt>();
      }
      WaitStaticInSimpleWrapperRewriter rewrite(rewriter, new_wait_n);
      return rewrite(stmt);
    }

    WaitStaticInSimpleWrapperRewriter(const PipelineRewriter *rewriter,
                                      int new_wait_n)
        : rewriter_(rewriter), new_wait_n_(new_wait_n) {}

    Optional<Stmt> VisitStmt_(const AttrStmtNode *op) final {
      if (IsAsyncWaitQueueScope(op)) {
        return rewriter_->MakeStaticAsyncWaitStmtLike(op, new_wait_n_);
      }
      Optional<Stmt> body = VisitStmt(op->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      return AttrStmt(op->node, op->attr_key, op->value, body.value(),
                      op->span);
    }

    Optional<Stmt> VisitStmt_(const SeqStmtNode *op) final {
      if (op->seq.size() != 1) {
        return Optional<Stmt>();
      }
      Optional<Stmt> inner = VisitStmt(op->seq[0]);
      if (!inner.defined()) {
        return Optional<Stmt>();
      }
      return SeqStmt({inner.value()});
    }

    Optional<Stmt> VisitStmt_(const SBlockNode *op) final {
      Optional<Stmt> body = VisitStmt(op->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      SBlock new_block = GetRef<SBlock>(op);
      new_block.CopyOnWrite()->body = body.value();
      return new_block;
    }

    Optional<Stmt> VisitStmt_(const SBlockRealizeNode *op) final {
      if (!is_one(op->predicate)) {
        return Optional<Stmt>();
      }
      Optional<Stmt> body = VisitStmt(op->block->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      SBlock new_block = op->block;
      new_block.CopyOnWrite()->body = body.value();
      return SBlockRealize(op->iter_values, op->predicate, new_block, op->span);
    }

    Optional<Stmt> VisitStmtDefault_(const Object *) final {
      return Optional<Stmt>();
    }

  private:
    const PipelineRewriter *rewriter_;
    int new_wait_n_;
  };

  Optional<Stmt> RewriteWaitStaticInSimpleWrapper(const Stmt &stmt,
                                                  int new_wait_n) const {
    return WaitStaticInSimpleWrapperRewriter::Rewrite(this, stmt, new_wait_n);
  }

  std::optional<int> TryGetHeadStaticWaitCount(const Stmt &stmt) const {
    HeadAsyncSyncInfo info =
        AnalyzeHeadAsyncSync(stmt, HeadSeqMode::kTakeFirstElement);
    if (info.kind == HeadAsyncSyncKind::kWaitStatic) {
      return info.wait_n;
    }
    return std::nullopt;
  }

  class FirstStaticWaitCounter
      : public StmtFunctor<std::optional<int>(const Stmt &)> {
  public:
    static std::optional<int> Find(const PipelineRewriter *rewriter,
                                   const Stmt &stmt) {
      FirstStaticWaitCounter finder(rewriter);
      return finder(stmt);
    }

    explicit FirstStaticWaitCounter(const PipelineRewriter *rewriter)
        : rewriter_(rewriter) {}

    std::optional<int> VisitStmt_(const AttrStmtNode *op) final {
      HeadAsyncSyncInfo info = rewriter_->AnalyzeHeadAsyncSync(
          GetRef<Stmt>(op), HeadSeqMode::kTakeFirstElement);
      if (info.kind == HeadAsyncSyncKind::kWaitStatic) {
        return info.wait_n;
      }
      if (info.IsBoundary()) {
        return std::nullopt;
      }
      return VisitStmt(op->body);
    }

    std::optional<int> VisitStmt_(const SeqStmtNode *op) final {
      for (const Stmt &elem : op->seq) {
        HeadAsyncSyncInfo info = rewriter_->AnalyzeHeadAsyncSync(
            elem, HeadSeqMode::kTakeFirstElement);
        if (info.kind == HeadAsyncSyncKind::kWaitStatic) {
          return info.wait_n;
        }
        if (info.IsBoundary() || rewriter_->ContainsAsyncSyncScopes(elem)) {
          return std::nullopt;
        }
      }
      return std::nullopt;
    }

    std::optional<int> VisitStmt_(const SBlockNode *op) final {
      return VisitStmt(op->body);
    }

    std::optional<int> VisitStmt_(const SBlockRealizeNode *op) final {
      if (is_one(op->predicate)) {
        return VisitStmt(op->block->body);
      }
      return std::nullopt;
    }

    std::optional<int> VisitStmtDefault_(const Object *) final {
      return std::nullopt;
    }

  private:
    const PipelineRewriter *rewriter_;
  };

  std::optional<int> TryGetFirstStaticWaitCount(const Stmt &stmt) const {
    return FirstStaticWaitCounter::Find(this, stmt);
  }

  class HeadStaticWaitInWrapperRewriter
      : public StmtFunctor<Optional<Stmt>(const Stmt &)> {
  public:
    static Optional<Stmt> Rewrite(const PipelineRewriter *rewriter,
                                  const Stmt &stmt, int new_wait_n) {
      HeadStaticWaitInWrapperRewriter rewrite(rewriter, new_wait_n);
      return rewrite(stmt);
    }

    HeadStaticWaitInWrapperRewriter(const PipelineRewriter *rewriter,
                                    int new_wait_n)
        : rewriter_(rewriter), new_wait_n_(new_wait_n) {}

    Optional<Stmt> VisitStmt_(const AttrStmtNode *op) final {
      if (IsAsyncWaitQueueScope(op)) {
        return rewriter_->MakeStaticAsyncWaitStmtLike(op, new_wait_n_);
      }
      Optional<Stmt> body = VisitStmt(op->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      return AttrStmt(op->node, op->attr_key, op->value, body.value(),
                      op->span);
    }

    Optional<Stmt> VisitStmt_(const SeqStmtNode *op) final {
      if (op->seq.empty()) {
        return Optional<Stmt>();
      }
      Optional<Stmt> first = VisitStmt(op->seq[0]);
      if (!first.defined()) {
        return Optional<Stmt>();
      }
      Array<Stmt> new_seq = op->seq;
      new_seq.Set(0, first.value());
      return SeqStmt(new_seq);
    }

    Optional<Stmt> VisitStmt_(const SBlockNode *op) final {
      Optional<Stmt> body = VisitStmt(op->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      SBlock new_block = GetRef<SBlock>(op);
      new_block.CopyOnWrite()->body = body.value();
      return new_block;
    }

    Optional<Stmt> VisitStmt_(const SBlockRealizeNode *op) final {
      if (!is_one(op->predicate)) {
        return Optional<Stmt>();
      }
      Optional<Stmt> body = VisitStmt(op->block->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      SBlock new_block = op->block;
      new_block.CopyOnWrite()->body = body.value();
      return SBlockRealize(op->iter_values, op->predicate, new_block, op->span);
    }

    Optional<Stmt> VisitStmtDefault_(const Object *) final {
      return Optional<Stmt>();
    }

  private:
    const PipelineRewriter *rewriter_;
    int new_wait_n_;
  };

  Optional<Stmt> RewriteHeadStaticWaitInWrapper(const Stmt &stmt,
                                                int new_wait_n) const {
    return HeadStaticWaitInWrapperRewriter::Rewrite(this, stmt, new_wait_n);
  }

  class FirstStaticWaitInWrapperRewriter
      : public StmtFunctor<Optional<Stmt>(const Stmt &)> {
  public:
    static Optional<Stmt> Rewrite(const PipelineRewriter *rewriter,
                                  const Stmt &stmt, int new_wait_n) {
      FirstStaticWaitInWrapperRewriter rewrite(rewriter, new_wait_n);
      return rewrite(stmt);
    }

    FirstStaticWaitInWrapperRewriter(const PipelineRewriter *rewriter,
                                     int new_wait_n)
        : rewriter_(rewriter), new_wait_n_(new_wait_n) {}

    Optional<Stmt> VisitStmt_(const AttrStmtNode *op) final {
      if (IsAsyncWaitQueueScope(op)) {
        return rewriter_->MakeStaticAsyncWaitStmtLike(op, new_wait_n_);
      }
      if (IsAsyncCommitQueueScope(op) || IsAsyncWaitInflightCount(op)) {
        return Optional<Stmt>();
      }
      Optional<Stmt> body = VisitStmt(op->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      return AttrStmt(op->node, op->attr_key, op->value, body.value(),
                      op->span);
    }

    Optional<Stmt> VisitStmt_(const SeqStmtNode *op) final {
      Array<Stmt> new_seq = op->seq;
      for (int i = 0, n = static_cast<int>(new_seq.size()); i < n; ++i) {
        Optional<Stmt> updated = VisitStmt(new_seq[i]);
        if (updated.defined()) {
          new_seq.Set(i, updated.value());
          return SeqStmt(new_seq);
        }
        if (rewriter_->ContainsAsyncSyncScopes(new_seq[i])) {
          return Optional<Stmt>();
        }
      }
      return Optional<Stmt>();
    }

    Optional<Stmt> VisitStmt_(const SBlockNode *op) final {
      Optional<Stmt> body = VisitStmt(op->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      SBlock new_block = GetRef<SBlock>(op);
      new_block.CopyOnWrite()->body = body.value();
      return new_block;
    }

    Optional<Stmt> VisitStmt_(const SBlockRealizeNode *op) final {
      if (!is_one(op->predicate)) {
        return Optional<Stmt>();
      }
      Optional<Stmt> body = VisitStmt(op->block->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      SBlock new_block = op->block;
      new_block.CopyOnWrite()->body = body.value();
      return SBlockRealize(op->iter_values, op->predicate, new_block, op->span);
    }

    Optional<Stmt> VisitStmtDefault_(const Object *) final {
      return Optional<Stmt>();
    }

  private:
    const PipelineRewriter *rewriter_;
    int new_wait_n_;
  };

  Optional<Stmt> RewriteFirstStaticWaitInWrapper(const Stmt &stmt,
                                                 int new_wait_n) const {
    return FirstStaticWaitInWrapperRewriter::Rewrite(this, stmt, new_wait_n);
  }

  Stmt MaybeRelaxLoopWaits(const For &loop, int pre_outstanding_lb) const {
    int retain = PipelinedRetainGroups(GetPipelineNumStages(loop.get()));
    if (retain <= 0 || !loop.defined()) {
      return loop;
    }
    const auto *seq = loop->body.as<SeqStmtNode>();
    if (!seq || seq->seq.empty()) {
      return loop;
    }

    Array<Stmt> body = seq->seq;
    bool changed = false;
    int outstanding_lb = std::max(0, pre_outstanding_lb);
    int groups_since_wait_lb = 0;
    bool seen_wait_boundary = false;

    for (int i = 0, n = static_cast<int>(body.size()); i < n; ++i) {
      ClassifiedAsyncSyncStmt cls = ClassifySimpleAsyncSyncStmt(body[i]);
      if (cls.kind == AsyncSyncStmtKind::kCommit) {
        ++outstanding_lb;
        ++groups_since_wait_lb;
        continue;
      }
      if (cls.kind == AsyncSyncStmtKind::kWaitDynamic) {
        seen_wait_boundary = true;
        outstanding_lb = 0;
        groups_since_wait_lb = 0;
        continue;
      }
      if (cls.kind == AsyncSyncStmtKind::kWaitStatic) {
        int effective_wait_n = cls.wait_n;
        if (cls.wait_n == 0) {
          int groups_after_wait_lb =
              GuaranteedNewGroupsBeforeNextWait(body, i + 1);
          int per_sync_groups = groups_since_wait_lb;
          bool uses_head_fallback =
              (per_sync_groups == 0 && !seen_wait_boundary);
          if (uses_head_fallback) {
            per_sync_groups = 1;
          }
          int candidate_wait_n =
              std::max(0, std::min(retain * per_sync_groups, 7));
          bool enough_pre_outstanding =
              !uses_head_fallback || outstanding_lb >= (candidate_wait_n + 1);
          if (candidate_wait_n > 0 && enough_pre_outstanding &&
              (!uses_head_fallback || groups_after_wait_lb > 0)) {
            Optional<Stmt> rewritten_wait =
                RewriteWaitStaticInSimpleWrapper(body[i], candidate_wait_n);
            if (rewritten_wait.defined()) {
              body.Set(i, rewritten_wait.value());
              changed = true;
              effective_wait_n = candidate_wait_n;
            }
          }
        }
        seen_wait_boundary = true;
        outstanding_lb = std::min(outstanding_lb, effective_wait_n);
        groups_since_wait_lb = 0;
        continue;
      }

      AsyncSyncSummary summary = SummarizeAsyncSyncScopes(body[i]);
      if (summary.wait == 0) {
        if (auto commits = TryGetDeterministicNoWaitCommitGroups(body[i])) {
          outstanding_lb += *commits;
          groups_since_wait_lb += *commits;
          continue;
        }
      }
      if (summary.wait > 0) {
        seen_wait_boundary = true;
      }
      outstanding_lb = 0;
      groups_since_wait_lb = 0;
    }

    if (!changed) {
      return loop;
    }
    For new_loop = loop;
    new_loop.CopyOnWrite()->body = body.size() == 1 ? body[0] : SeqStmt(body);
    return new_loop;
  }

  class LoopWaitsInSimpleWrapperRelaxer
      : public StmtFunctor<Optional<Stmt>(const Stmt &)> {
  public:
    static Optional<Stmt> Rewrite(const PipelineRewriter *rewriter,
                                  const Stmt &stmt, int pre_outstanding_lb) {
      LoopWaitsInSimpleWrapperRelaxer relaxer(rewriter, pre_outstanding_lb);
      return relaxer(stmt);
    }

    LoopWaitsInSimpleWrapperRelaxer(const PipelineRewriter *rewriter,
                                    int pre_outstanding_lb)
        : rewriter_(rewriter), pre_outstanding_lb_(pre_outstanding_lb) {}

    Optional<Stmt> VisitStmt_(const ForNode *op) final {
      For loop = GetRef<For>(op);
      Stmt relaxed = rewriter_->MaybeRelaxLoopWaits(loop, pre_outstanding_lb_);
      if (relaxed.same_as(loop)) {
        return Optional<Stmt>();
      }
      return relaxed;
    }

    Optional<Stmt> VisitStmt_(const AttrStmtNode *op) final {
      Optional<Stmt> body = VisitStmt(op->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      return AttrStmt(op->node, op->attr_key, op->value, body.value(),
                      op->span);
    }

    Optional<Stmt> VisitStmt_(const SeqStmtNode *op) final {
      if (op->seq.size() != 1) {
        return Optional<Stmt>();
      }
      Optional<Stmt> inner = VisitStmt(op->seq[0]);
      if (!inner.defined()) {
        return Optional<Stmt>();
      }
      return SeqStmt({inner.value()});
    }

    Optional<Stmt> VisitStmt_(const SBlockNode *op) final {
      Optional<Stmt> body = VisitStmt(op->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      SBlock new_block = GetRef<SBlock>(op);
      new_block.CopyOnWrite()->body = body.value();
      return new_block;
    }

    Optional<Stmt> VisitStmt_(const SBlockRealizeNode *op) final {
      if (!is_one(op->predicate)) {
        return Optional<Stmt>();
      }
      Optional<Stmt> body = VisitStmt(op->block->body);
      if (!body.defined()) {
        return Optional<Stmt>();
      }
      SBlock new_block = op->block;
      new_block.CopyOnWrite()->body = body.value();
      return SBlockRealize(op->iter_values, op->predicate, new_block, op->span);
    }

    Optional<Stmt> VisitStmtDefault_(const Object *) final {
      return Optional<Stmt>();
    }

  private:
    const PipelineRewriter *rewriter_;
    int pre_outstanding_lb_;
  };

  Optional<Stmt> RelaxLoopWaitsInSimpleWrapper(const Stmt &stmt,
                                               int pre_outstanding_lb) const {
    return LoopWaitsInSimpleWrapperRelaxer::Rewrite(this, stmt,
                                                    pre_outstanding_lb);
  }

  class AsyncPipelineLoopWaitRelaxer : public StmtExprMutator {
  public:
    explicit AsyncPipelineLoopWaitRelaxer(const PipelineRewriter *rewriter)
        : rewriter_(rewriter) {}

    Stmt VisitStmt_(const SeqStmtNode *op) final {
      Array<Stmt> visited;
      visited.reserve(op->seq.size());
      for (const Stmt &stmt : op->seq) {
        visited.push_back(this->VisitStmt(stmt));
      }

      int outstanding_lb = 0;
      for (int i = 0, n = static_cast<int>(visited.size()); i < n; ++i) {
        Stmt current = visited[i];
        Optional<Stmt> relaxed =
            rewriter_->RelaxLoopWaitsInSimpleWrapper(current, outstanding_lb);
        if (relaxed.defined()) {
          current = relaxed.value();
          visited.Set(i, current);
        }
        ClassifiedAsyncSyncStmt cls =
            rewriter_->ClassifySimpleAsyncSyncStmt(current);
        if (cls.kind == AsyncSyncStmtKind::kCommit) {
          ++outstanding_lb;
          continue;
        }
        if (cls.kind == AsyncSyncStmtKind::kWaitStatic) {
          outstanding_lb = std::min(outstanding_lb, cls.wait_n);
          continue;
        }
        if (cls.kind == AsyncSyncStmtKind::kWaitDynamic) {
          outstanding_lb = 0;
          continue;
        }
        AsyncSyncSummary summary = rewriter_->SummarizeAsyncSyncScopes(current);
        if (summary.wait == 0) {
          if (auto commits =
                  rewriter_->TryGetDeterministicNoWaitCommitGroups(current)) {
            outstanding_lb += *commits;
            continue;
          }
        }
        if (summary.wait > 0) {
          outstanding_lb = 0;
        }
      }

      if (visited.empty()) {
        return Evaluate(0);
      }
      if (visited.size() == 1) {
        return visited[0];
      }
      return SeqStmt(visited);
    }

  private:
    const PipelineRewriter *rewriter_;
  };

  Array<Stmt> RelaxTrailingConsumerWaits(Array<Stmt> seq, int retain) const {
    if (retain <= 0 || seq.size() <= 1) {
      return seq;
    }
    std::vector<int> suffix_wait_indices;
    for (int i = static_cast<int>(seq.size()) - 1; i >= 0; --i) {
      if (ContainsAsyncCommitScopes(seq[i])) {
        break;
      }
      auto first_wait = TryGetFirstStaticWaitCount(seq[i]);
      if (!first_wait.has_value() || *first_wait != 0) {
        break;
      }
      suffix_wait_indices.push_back(i);
    }
    if (suffix_wait_indices.size() <= 1) {
      return seq;
    }
    for (size_t pos = 1; pos < suffix_wait_indices.size(); ++pos) {
      int idx = suffix_wait_indices[pos];
      // Tail consumers drain the final committed groups with no new commits in
      // between. Relax them progressively from the end so the suffix becomes
      // ..., wait<2>, wait<1>, wait<0> instead of rewriting every drain wait to
      // the same retain count.
      int new_wait_n = std::min(retain, static_cast<int>(pos));
      Optional<Stmt> rewritten =
          RewriteFirstStaticWaitInWrapper(seq[idx], new_wait_n);
      if (rewritten.defined()) {
        seq.Set(idx, rewritten.value());
      }
    }
    return seq;
  }

  void PopulateWaitCounts(const std::vector<RewrittenStmtInfo> &new_stmts,
                          arith::Analyzer *ana_normalized,
                          const BufferCommitGroupMap &buffer_to_commit_group,
                          std::map<int, AsyncStateLocal> *async_states_local) {
    for (size_t i = 0; i < new_stmts.size(); ++i) {
      if (new_stmts[i].is_async) {
        for (const BufferRegion &write_region : new_stmts[i].writes) {
          (*async_states_local)[new_stmts[i].stage].seen.insert(
              write_region->buffer);
        }
        continue;
      }

      int producer_stage_idx = -1;
      for (const BufferRegion &read_region : new_stmts[i].reads) {
        for (const auto &kv : async_states_) {
          if (kv.first <= new_stmts[i].stage &&
              kv.second.writes(read_region->buffer)) {
            ICHECK(producer_stage_idx == -1 || producer_stage_idx == kv.first)
                << "A dependency on multiple async stages is not supported";
            producer_stage_idx = kv.first;
          }
        }
      }

      if (producer_stage_idx == -1) {
        continue;
      }

      auto &dep_local_state = (*async_states_local)[producer_stage_idx];
      int num_commit_group = dep_local_state.commit_groups.size();
      std::vector<Optional<PrimExpr>> producer_head_per_commit;
      std::vector<int> dependent_commit_groups;

      if (num_commit_group == 0) {
        ICHECK(!dep_local_state.producer_head);
        dependent_commit_groups.push_back(-1);
        producer_head_per_commit.push_back(
            async_states_[producer_stage_idx].producer_head);
      } else {
        ICHECK(dep_local_state.producer_head);
        std::vector<bool> need_wait_count(num_commit_group, true);
        for (const BufferRegion &read_region : new_stmts[i].reads) {
          if (!async_states_[producer_stage_idx].writes(read_region->buffer)) {
            continue;
          }
          auto commit_group_id = buffer_to_commit_group.at(read_region->buffer);
          if (!need_wait_count[commit_group_id]) {
            continue;
          }
          dependent_commit_groups.push_back(commit_group_id);
          if (!dep_local_state.seen.count(read_region->buffer)) {
            producer_head_per_commit.push_back(
                dep_local_state.producer_head.value() - 1);
          } else {
            producer_head_per_commit.push_back(
                dep_local_state.producer_head.value());
          }
          need_wait_count[commit_group_id] = false;
        }
      }

      PrimExpr wait_count = [&]() {
        PrimExpr sum = PrimExpr(0);
        for (const Optional<PrimExpr> &producer_head :
             producer_head_per_commit) {
          if (producer_head &&
              ana_normalized->CanProve(producer_head.value() >= 0)) {
            sum += analyzer_.Simplify(producer_head.value() -
                                      new_stmts[i].access_index);
          } else {
            return PrimExpr(0);
          }
        }
        return sum;
      }();

      for (int commit_group_id : dependent_commit_groups) {
        auto &pending_wait = dep_local_state.pending_waits[commit_group_id];
        if (!pending_wait.valid()) {
          pending_wait = {static_cast<int>(i), wait_count};
        } else if (analyzer_.CanProve(wait_count < pending_wait.wait_count)) {
          pending_wait = {pending_wait.insert_before, wait_count};
        }
      }
    }
  }

  std::vector<FinalStmtInfo> CompletePipelineLoopStatements(
      const std::vector<RewrittenStmtInfo> &stmts,
      const std::map<int, AsyncStateLocal> &async_states_local,
      arith::Analyzer *ana_normalized) const {
    std::vector<FinalStmtInfo> new_stmts;
    new_stmts.reserve(stmts.size());
    for (const auto &stmt : stmts) {
      new_stmts.push_back(
          {stmt.stage, stmt.access_index, stmt.predicate, stmt.stmt});
    }

    std::vector<int> commit_group_tags(new_stmts.size(), -1);
    std::unordered_map<int, int> commit_group_tag_to_stage;
    int next_commit_group_tag = 0;
    std::map<int, std::map<int, PrimExpr>> waits_before_stmt;
    auto make_wait_stmt = [](int stage_id, PrimExpr wait_count, Stmt body) {
      auto zero = make_zero(DataType::Int(32));
      return AttrStmt(zero, s_tir::attr::async_wait_queue_scope, stage_id,
                      AttrStmt(zero, s_tir::attr::async_wait_inflight_count,
                               wait_count, body));
    };
    auto merge_wait_before_stmt = [&](int insert_before, int stage_id,
                                      PrimExpr wait_count) {
      auto &waits_at_stmt = waits_before_stmt[insert_before];
      auto it = waits_at_stmt.find(stage_id);
      if (it == waits_at_stmt.end()) {
        waits_at_stmt.emplace(stage_id, ana_normalized->Simplify(wait_count));
      } else if (ana_normalized->CanProve(wait_count < it->second)) {
        it->second = ana_normalized->Simplify(wait_count);
      }
    };

    for (const auto &[stage_id, state] : async_states_local) {
      if (!state.commit_groups.empty()) {
        for (const auto &group_stmt_indices : state.commit_groups) {
          int commit_group_tag = next_commit_group_tag++;
          commit_group_tag_to_stage.emplace(commit_group_tag, stage_id);
          for (size_t stmt_idx : group_stmt_indices) {
            ICHECK(stmt_idx < new_stmts.size());
            commit_group_tags[stmt_idx] = commit_group_tag;
          }
        }
      }

      for (const auto &[commit_group_id, pending_wait] : state.pending_waits) {
        if (!pending_wait.valid()) {
          continue;
        }
        PrimExpr wait_count = ana_normalized->Simplify(pending_wait.wait_count);
        if (state.predicate &&
            !ana_normalized->CanProve(state.predicate.value())) {
          PrimExpr predicate =
              ana_normalized->Simplify(state.predicate.value());
          if (is_zero(predicate)) {
            continue;
          }
          merge_wait_before_stmt(pending_wait.insert_before, stage_id,
                                 wait_count);
          continue;
        }

        merge_wait_before_stmt(pending_wait.insert_before, stage_id,
                               wait_count);
      }
    }

    std::vector<FinalStmtInfo> result;
    for (size_t i = 0; i < new_stmts.size();) {
      if (auto it = waits_before_stmt.find(i); it != waits_before_stmt.end()) {
        for (const auto &[stage_id, wait_count] : it->second) {
          Stmt wait_stmt = make_wait_stmt(stage_id, wait_count, Evaluate(0));
          if (auto state_it = async_states_local.find(stage_id);
              state_it != async_states_local.end() &&
              state_it->second.predicate &&
              !ana_normalized->CanProve(state_it->second.predicate.value())) {
            PrimExpr predicate =
                ana_normalized->Simplify(state_it->second.predicate.value());
            if (is_zero(predicate)) {
              continue;
            }
            wait_stmt = IfThenElse(predicate, wait_stmt, Evaluate(0));
          }
          result.push_back({new_stmts[i].stage, new_stmts[i].access_index,
                            new_stmts[i].predicate, wait_stmt});
        }
      }

      if (commit_group_tags[i] == -1) {
        result.push_back(new_stmts[i]);
        ++i;
        continue;
      }

      int commit_group_tag = commit_group_tags[i];
      int stage_id = commit_group_tag_to_stage.at(commit_group_tag);
      Array<Stmt> group_stmts;
      PrimExpr access_index = new_stmts[i].access_index;
      PrimExpr predicate = new_stmts[i].predicate;
      for (; i < new_stmts.size() && commit_group_tags[i] == commit_group_tag;
           ++i) {
        group_stmts.push_back(new_stmts[i].stmt);
      }
      Stmt group_body =
          group_stmts.size() == 1 ? group_stmts[0] : SeqStmt(group_stmts);
      Stmt commit_queue_scope =
          AttrStmt(make_zero(DataType::Int(32)),
                   s_tir::attr::async_commit_queue_scope, stage_id, group_body);
      if (!is_one(predicate) && !ana_normalized->CanProve(predicate)) {
        PrimExpr simplified_predicate = ana_normalized->Simplify(predicate);
        if (!is_zero(simplified_predicate)) {
          commit_queue_scope =
              IfThenElse(simplified_predicate, commit_queue_scope, Evaluate(0));
        }
      }
      result.push_back({stage_id, access_index, predicate, commit_queue_scope});
    }
    return result;
  }

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
    Optional<Integer> pipeline_num_stages =
        GetPipelineNumStages(pipeline_loop_.get());
    auto make_nop = []() {
      return SBlockRealize({}, Bool(true), MakeBlock(Evaluate(0), {}));
    };

    if (unroll_loop) {
      if (const int64_t *extent_imm = as_const_int(extent)) {
        if (*extent_imm > 1) {
          Array<Stmt> expanded;
          expanded.reserve(static_cast<size_t>(*extent_imm));
          for (int64_t iter = 0; iter < *extent_imm; ++iter) {
            PrimExpr unit_start =
                analyzer_.Simplify(start + IntImm(extent.dtype(), iter));
            PrimExpr unit_end =
                analyzer_.Simplify(start + IntImm(extent.dtype(), iter + 1));
            Stmt unit_stmt =
                EmitImpl(unit_start, unit_end, false, need_bound_check);
            expanded.push_back(StripPipelineContextAttrs(unit_stmt));
          }
          Stmt result = expanded.size() == 1 ? expanded[0] : SeqStmt(expanded);
          if (pipeline_num_stages) {
            if (pipeline_num_stages.value().IntValue() > 1) {
              result = AttrStmt(Integer(0), kPipelineMVBContextNumStages,
                                Downcast<PrimExpr>(pipeline_num_stages.value()),
                                result);
            }
            result = AttrStmt(Integer(0), kPipelineContextNumStages,
                              Downcast<PrimExpr>(pipeline_num_stages.value()),
                              result);
          }
          return result;
        }
      }
    }

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

    arith::Analyzer ana_normalized;
    if (!is_unit_loop) {
      ana_normalized.Bind(Downcast<Var>(new_loop_var),
                          Range(pipeline_loop_->min, extent));
    }

    std::vector<RewrittenStmtInfo> new_stmts;
    std::map<int, AsyncStateLocal> async_states_local;
    BufferCommitGroupMap buffer_to_commit_group;

    for (const SBlock &block : ordered_stmts_) {
      const auto &pipeline_anno = pipeline_info_.at(block);
      int stage = pipeline_anno.stage;
      PrimExpr inbound = Bool(true);
      PrimExpr skewed_loop_var = new_loop_var - stage;
      if (need_bound_check)
        inbound = And(
            pipeline_loop_->min <= skewed_loop_var,
            (skewed_loop_var < pipeline_loop_->min + pipeline_loop_->extent));

      SBlock new_block = Downcast<SBlock>(
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
      inbound = ana_normalized.Simplify(inbound);
      if (is_zero(inbound)) {
        continue;
      }
      new_block = Downcast<SBlock>(Substitute(
          new_block, {{pipeline_loop_->loop_var, normalized_access_index}}));
      new_block = ReplayScalarBindings(new_block, normalized_access_index);

      Stmt rewritten_stmt = SBlockRealize({}, inbound, new_block);
      rewritten_stmt = WrapPipelineStageContext(std::move(rewritten_stmt),
                                                normalized_access_index,
                                                pipeline_num_stages);
      Optional<PrimExpr> pipeline_mbar_phase = ComputePipelineMbarPhaseExpr(
          normalized_access_index, pipeline_num_stages);

      bool is_async = pipeline_anno.async;
      if (is_async) {
        auto &local_state = async_states_local[stage];
        int commit_group_id = -1;
        if (pipeline_anno.async_group_id >= 0) {
          auto it = local_state.annotated_group_to_commit_group.find(
              pipeline_anno.async_group_id);
          if (it == local_state.annotated_group_to_commit_group.end()) {
            commit_group_id = local_state.commit_groups.size();
            local_state.commit_groups.push_back({new_stmts.size()});
            local_state.annotated_group_to_commit_group.emplace(
                pipeline_anno.async_group_id, commit_group_id);
          } else {
            commit_group_id = it->second;
            local_state.commit_groups[commit_group_id].push_back(
                new_stmts.size());
          }
        } else if (local_state.commit_groups.empty() || local_state.consumed) {
          commit_group_id = local_state.commit_groups.size();
          local_state.commit_groups.push_back({new_stmts.size()});
        } else {
          commit_group_id = local_state.commit_groups.size() - 1;
          local_state.commit_groups.back().push_back(new_stmts.size());
        }

        for (const BufferRegion &write_region : new_block->writes) {
          async_states_[stage].dst_buffers.insert(write_region->buffer);
          buffer_to_commit_group[write_region->buffer] = commit_group_id;
        }
        async_states_[stage].producer_head = normalized_access_index;
        local_state.producer_head = normalized_access_index;
        if (!local_state.predicate ||
            ana_normalized.CanProve(local_state.predicate.value())) {
          local_state.predicate = inbound;
        } else {
          local_state.predicate =
              ana_normalized.Simplify(local_state.predicate.value() & inbound);
        }
        rewritten_stmt = AnnotateSimtProducer(rewritten_stmt, target_);
        rewritten_stmt = AttrStmt(make_zero(DataType::Int(32)),
                                  s_tir::attr::async_scope, 1, rewritten_stmt);
      }
      if (pipeline_mbar_phase) {
        rewritten_stmt = AnnotateTileOpMbarPhase(rewritten_stmt,
                                                 pipeline_mbar_phase.value());
      }

      new_stmts.push_back({stage, inbound, new_block->reads, new_block->writes,
                           normalized_access_index, is_async, rewritten_stmt});

      for (const BufferRegion &read_region : new_block->reads) {
        for (const auto &kv : async_states_) {
          if (kv.first <= stage && kv.second.writes(read_region->buffer)) {
            async_states_local[kv.first].consumed = true;
          }
        }
      }
    }

    PopulateWaitCounts(new_stmts, &ana_normalized, buffer_to_commit_group,
                       &async_states_local);
    std::vector<FinalStmtInfo> final_stmts = CompletePipelineLoopStatements(
        new_stmts, async_states_local, &ana_normalized);

    Array<Stmt> stmts;
    for (const auto &stmt_info : final_stmts) {
      stmts.push_back(stmt_info.stmt);
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
        if (kv.first != s_tir::attr::software_pipeline_stage &&
            kv.first != s_tir::attr::software_pipeline_order &&
            kv.first != s_tir::attr::software_pipeline_async_stages &&
            kv.first != kPipelineAsyncProducers &&
            kv.first != kPipelineAsyncProducerGroups &&
            kv.first != kPipelineTmaCopies &&
            kv.first != kPipelineReplayableScalarBinds &&
            kv.first != "num_stages") {
          preserved_annotations.Set(key, kv.second);
        }
      }
      if (pipeline_num_stages &&
          preserved_annotations.find("tl_pipelined_num_stages") ==
              preserved_annotations.end()) {
        preserved_annotations.Set("tl_pipelined_num_stages",
                                  pipeline_num_stages.value());
      }
      new_loop = For(Downcast<Var>(new_loop_var), pipeline_loop_->min, extent,
                     unroll_loop ? ForKind::kUnrolled : pipeline_loop_->kind,
                     std::move(new_loop), std::nullopt, preserved_annotations);
    }
    Stmt result = SBlockRealize({}, Bool(true),
                                MakeBlock(new_loop, buffer_data_to_buffer_));
    if (pipeline_num_stages) {
      if (pipeline_num_stages.value().IntValue() > 1) {
        result =
            AttrStmt(Integer(0), kPipelineMVBContextNumStages,
                     Downcast<PrimExpr>(pipeline_num_stages.value()), result);
      }
      result =
          AttrStmt(Integer(0), kPipelineContextNumStages,
                   Downcast<PrimExpr>(pipeline_num_stages.value()), result);
    }
    return result;
  }

  arith::Analyzer analyzer_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Array<Buffer> pipeline_allocs_;
  Array<Buffer> local_allocs_;
  For pipeline_loop_;
  PipelineInfo pipeline_info_;
  Array<SBlock> scalar_binding_blocks_;
  int max_stage_ = -1;
  Map<Buffer, Buffer> buffer_remap_;
  Optional<Target> target_;
  Array<SBlock> ordered_stmts_;
  std::vector<ScalarBinding> scalar_bindings_;
  ScalarBindingMap scalar_binding_map_;
  std::map<int, AsyncStateGlobal> async_states_;
};

PipelineRewriteResult RewritePipeline(
    Map<Var, Buffer> buffer_data_to_buffer,
    const Array<Buffer> &pipeline_allocs, const Array<Buffer> &local_allocs,
    const For &pipeline_loop, const PipelineInfo &pipeline_info,
    const Array<SBlock> &scalar_binding_blocks, Optional<Target> target) {
  PipelineRewriter rewriter(std::move(buffer_data_to_buffer), pipeline_allocs,
                            local_allocs, pipeline_loop, pipeline_info,
                            scalar_binding_blocks, std::move(target));
  PipelineRewriteResult result;
  result.pipeline = rewriter.BuildPipeline();
  result.buffer_remap = rewriter.GetBufferRemap();
  return result;
}

} // namespace software_pipeline
} // namespace tl
} // namespace tvm
