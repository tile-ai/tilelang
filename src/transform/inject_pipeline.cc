/*!
 * \file inject_software_pipeline.cc
 * \brief Transform annotated loops into pipelined one that parallelize
 * producers and consumers
 */
#include <tvm/arith/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <functional>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../layout/layout.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/gemm.h"
#include "../op/operator.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "common/mbarrier.h"
#include "common/pipeline_utils.h"
#include "support/utils.h"
#include "tir/schedule/utils.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
using namespace tir;
using namespace ffi;
namespace software_pipeline {

namespace {

bool ShapesEqual(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs,
                 arith::Analyzer *analyzer) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!analyzer->CanProveEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

Layout ExpandAnnotatedLayoutForMultiVersionedBuffer(const Layout &layout,
                                                    const Buffer &old_buffer,
                                                    const Buffer &new_buffer) {
  if (!layout.defined() ||
      new_buffer->shape.size() <= old_buffer->shape.size()) {
    return Layout();
  }

  arith::Analyzer analyzer;
  if (!ShapesEqual(layout->InputShape(), old_buffer->shape, &analyzer)) {
    return Layout();
  }

  size_t leading_ndim = new_buffer->shape.size() - old_buffer->shape.size();
  Array<PrimExpr> trailing_shape;
  Array<PrimExpr> leading_shape;
  for (size_t i = 0; i < leading_ndim; ++i) {
    leading_shape.push_back(new_buffer->shape[i]);
  }
  for (size_t i = 0; i < old_buffer->shape.size(); ++i) {
    trailing_shape.push_back(new_buffer->shape[leading_ndim + i]);
  }
  if (!ShapesEqual(trailing_shape, old_buffer->shape, &analyzer)) {
    return Layout();
  }

  return layout->Expand(leading_shape);
}

bool UpdateExpandedLayoutMapForRemappedAllocs(
    const std::vector<std::pair<Buffer, Buffer>> &remapped_allocs,
    Map<String, Any> *annotations) {
  if (remapped_allocs.empty() || !annotations->count(attr::kLayoutMap)) {
    return false;
  }

  auto layout_map_ref = annotations->Get(attr::kLayoutMap);
  if (!layout_map_ref.has_value()) {
    return false;
  }
  auto layout_map = layout_map_ref.value().as<Map<Var, Layout>>();
  if (!layout_map.has_value()) {
    return false;
  }

  Map<Var, Layout> updated_layout_map = layout_map.value();
  std::unordered_set<const VarNode *> visited;
  bool changed = false;
  for (const auto &[old_buffer, new_buffer] : remapped_allocs) {
    if (!visited.insert(old_buffer->data.get()).second ||
        !updated_layout_map.count(old_buffer->data)) {
      continue;
    }
    Layout layout = updated_layout_map[old_buffer->data];
    Layout expanded = ExpandAnnotatedLayoutForMultiVersionedBuffer(
        layout, old_buffer, new_buffer);
    if (!expanded.defined()) {
      continue;
    }
    updated_layout_map.Set(old_buffer->data, expanded);
    changed = true;
  }

  if (changed) {
    annotations->Set(attr::kLayoutMap, updated_layout_map);
  }
  return changed;
}

} // namespace

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
    if (auto tile_op = ParseOperator(tvm::ffi::GetRef<Call>(op));
        tile_op.defined()) {
      AccessRegions access = tile_op->GetAccessRegions();
      for (const auto &region : access.reads) {
        AddBuffer(region->buffer);
      }
      for (const auto &region : access.writes) {
        AddBuffer(region->buffer);
      }
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
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

class TileOpAccessCollector : public StmtExprVisitor {
public:
  Array<BufferRegion> GetReads() const { return reads_; }

  Array<BufferRegion> GetWrites() const { return writes_; }

private:
  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(tvm::ffi::GetRef<Call>(op));
        tile_op.defined()) {
      AccessRegions access = tile_op->GetAccessRegions();
      reads_.insert(reads_.end(), access.reads.begin(), access.reads.end());
      writes_.insert(writes_.end(), access.writes.begin(), access.writes.end());
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  Array<BufferRegion> reads_;
  Array<BufferRegion> writes_;
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
  Block block;
  if (const BlockRealizeNode *block_realize = body.as<BlockRealizeNode>()) {
    if (is_one(block_realize->predicate)) {
      block = block_realize->block;
    }
  }
  if (!block.defined()) {
    block = Block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                  /*name_hint=*/"", /*body*/ body);
  }
  Array<Array<BufferRegion>> access =
      GetBlockReadWriteRegion(block, buffer_data_to_buffer);
  TileOpAccessCollector collector;
  collector(block->body);
  Array<BufferRegion> tile_reads = collector.GetReads();
  Array<BufferRegion> tile_writes = collector.GetWrites();
  BlockNode *n = block.CopyOnWrite();
  n->reads = access[0];
  n->reads.insert(n->reads.end(), tile_reads.begin(), tile_reads.end());
  n->writes = access[1];
  n->writes.insert(n->writes.end(), tile_writes.begin(), tile_writes.end());
  return block;
}

/*! Structure that represents the provided annotation per block or loop. */
struct PipelineAnnotation {
  int stage;
  int order;
  bool async{false};
  int async_group_id{-1};
};

using PipelineInfo = std::unordered_map<Block, PipelineAnnotation,
                                        ObjectPtrHash, ObjectPtrEqual>;

struct BufferAccessInfo {
  int def = -1; // the defining stage of the buffer
  int use = -1; // the last using stage of the buffer
};

// Detect whether a stage body already carries explicit async/cp.async
// semantics. InjectSoftwarePipeline only wants to "upgrade" ordinary producer
// stages into pipeline-managed async producers; if the body already contains
// raw cp.async instructions or async queue attrs, re-marking it as async here
// would stack two async protocols on the same stage.
bool ContainsExplicitAsyncIntrinsics(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &obj) {
    if (found) {
      return;
    }
    if (const auto *attr = obj.as<AttrStmtNode>()) {
      if (attr->attr_key == tir::attr::async_scope ||
          attr->attr_key == tir::attr::async_commit_queue_scope ||
          attr->attr_key == tir::attr::async_wait_queue_scope ||
          attr->attr_key == tir::attr::async_wait_inflight_count) {
        found = true;
        return;
      }
    }
    const auto *call = obj.as<CallNode>();
    if (!call) {
      return;
    }
    if (call->op.same_as(builtin::ptx_cp_async()) ||
        call->op.same_as(tl::ptx_cp_async()) ||
        call->op.same_as(builtin::ptx_commit_group()) ||
        call->op.same_as(builtin::ptx_wait_group())) {
      found = true;
    }
  });
  return found;
}

class SimtProducerAnnotator : public StmtExprMutator {
public:
  static Stmt Annotate(const Stmt &stmt,
                       Optional<Target> target = Optional<Target>()) {
    SimtProducerAnnotator annotator(std::move(target));
    return annotator.VisitStmt(stmt);
  }

private:
  explicit SimtProducerAnnotator(Optional<Target> target)
      : target_(std::move(target)) {}

  Stmt VisitStmt_(const ForNode *op) final {
    Stmt body = VisitStmt(op->body);
    auto annotations = op->annotations;
    // Keep the raw buffer-store cp.async path under outer pipeline-managed
    // commit/wait semantics as well.
    annotations.Set(attr::kParallelAsyncWithoutAsyncCommitWait, Bool(true));
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, annotations, op->step, op->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    static const Op &copy_op = Op::Get("tl.tileop.copy");
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call->op.same_as(copy_op) || !CanUsePipelineManagedCPAsyncCopy(call)) {
      return call;
    }
    // Tile-op copies lower through copy.cc, so they need an explicit
    // per-copy marker to suppress their own implicit commit/wait.
    auto annotations = call->annotations;
    annotations.Set(attr::kAsyncCopyNoImplicitCommitWait,
                    IntImm(DataType::Int(32), 1));
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }

  bool CanUsePipelineManagedCPAsyncCopy(const Call &call) const {
    auto tile_op = ParseOperator(call);
    const auto *copy = tile_op.as<CopyNode>();
    if (copy == nullptr) {
      return false;
    }
    if (!target_.defined()) {
      return copy->CheckPipelineManagedCPAsyncCopy();
    }
    return copy->CheckPipelineManagedCPAsyncCopy(target_.value(), &analyzer_);
  }

  Optional<Target> target_;
  mutable arith::Analyzer analyzer_;
};

class TileOpMbarPhaseAnnotator : public StmtExprMutator {
public:
  static Stmt Annotate(const Stmt &stmt, PrimExpr phase_expr) {
    TileOpMbarPhaseAnnotator annotator(std::move(phase_expr));
    return annotator.VisitStmt(stmt);
  }

private:
  explicit TileOpMbarPhaseAnnotator(PrimExpr phase_expr)
      : phase_expr_(std::move(phase_expr)) {}

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!IsMbarPhaseConsumer(call)) {
      return call;
    }
    if (call->annotations.count(attr::kPipelineMbarPhaseExpr)) {
      return call;
    }
    auto annotations = call->annotations;
    annotations.Set(attr::kPipelineMbarPhaseExpr, phase_expr_);
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }

  bool IsMbarPhaseConsumer(const Call &call) const {
    auto tile_op = ParseOperator(call);
    return tile_op.defined() && (tile_op.as<CopyNode>() != nullptr ||
                                 tile_op.as<Conv2DIm2ColOpNode>() != nullptr ||
                                 tile_op.as<GemmNode>() != nullptr);
  }

  PrimExpr phase_expr_;
};

class AsyncCommitWaitAttrLowerer : public StmtExprMutator {
public:
  static Stmt Lower(const Stmt &stmt) {
    AsyncCommitWaitAttrLowerer lowerer;
    return lowerer.VisitStmt(stmt);
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::async_commit_queue_scope) {
      Stmt body = VisitStmt(op->body);
      Stmt commit =
          Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
      if (is_no_op(body)) {
        return commit;
      }
      return SeqStmt({body, commit});
    }
    if (op->attr_key == tir::attr::async_wait_queue_scope) {
      auto wait_attrs = GetAsyncWaitAttributes(op);
      Stmt body = op->body;
      if (const auto *inner = op->body.as<AttrStmtNode>()) {
        if (inner->attr_key == tir::attr::async_wait_inflight_count) {
          body = inner->body;
        }
      }
      body = VisitStmt(body);
      Stmt wait = Evaluate(Call(DataType::Handle(), builtin::ptx_wait_group(),
                                {wait_attrs.second}));
      if (is_no_op(body)) {
        return wait;
      }
      return SeqStmt({wait, body});
    }
    if (op->attr_key == tir::attr::async_wait_inflight_count) {
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
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
   * \param loop_var_let_wrappers Let wrappers that depend on the loop var.
   * \param loop_var_if_wrappers If wrappers with conditions that depend on
   * the loop var.
   */
  PipelineRewriter(Map<Var, Buffer> buffer_data_to_buffer,
                   const Array<Buffer> &pipeline_allocs,
                   const Array<Buffer> &local_allocs, const For &pipeline_loop,
                   const PipelineInfo &pipeline_info, Optional<Target> target,
                   const std::vector<LetWrapper> &loop_var_let_wrappers,
                   const std::vector<IfWrapper> &loop_var_if_wrappers)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        pipeline_allocs_(pipeline_allocs), local_allocs_(local_allocs),
        pipeline_loop_(pipeline_loop), pipeline_info_(pipeline_info),
        target_(std::move(target)),
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
      if (pipeline_num_stages.value()->value > 1) {
        stmt = AttrStmt(Integer(0), kPipelineMVBContextNumStages,
                        Downcast<PrimExpr>(pipeline_num_stages.value()), stmt);
      }
      stmt = AttrStmt(Integer(0), kPipelineContextNumStages,
                      Downcast<PrimExpr>(pipeline_num_stages.value()), stmt);
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

  struct AsyncStateGlobal {
    std::unordered_set<const BufferNode *> dst_buffers;
    Optional<PrimExpr> producer_head{PrimExpr(-1)};

    bool writes(const Buffer &buffer) const {
      return dst_buffers.count(buffer.get()) > 0;
    }
  };

  struct AsyncStateLocal {
    struct PendingWait {
      int insert_before{-1};
      PrimExpr wait_count{nullptr};

      bool valid() const { return wait_count.defined(); }
    };

    std::unordered_set<const BufferNode *> seen;
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
      if (const auto *let = stmt.as<LetStmtNode>()) {
        return Analyze(let->body);
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
      if (const auto *block = stmt.as<BlockNode>()) {
        return Analyze(block->body);
      }
      if (const auto *realize = stmt.as<BlockRealizeNode>()) {
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

  Stmt
  WrapLoopDependentWrappers(Stmt stmt,
                            const PrimExpr &normalized_access_index) const {
    for (auto it = loop_var_if_wrappers_.rbegin();
         it != loop_var_if_wrappers_.rend(); ++it) {
      const auto &iw = *it;
      PrimExpr substituted_condition = Substitute(
          iw.condition, {{pipeline_loop_->loop_var, normalized_access_index}});
      stmt = IfThenElse(substituted_condition, stmt, Stmt(), iw.span);
    }
    for (auto it = loop_var_let_wrappers_.rbegin();
         it != loop_var_let_wrappers_.rend(); ++it) {
      const auto &lw = *it;
      PrimExpr substituted = Substitute(
          lw.value, {{pipeline_loop_->loop_var, normalized_access_index}});
      stmt = LetStmt(lw.var, substituted, stmt);
    }
    return stmt;
  }

  Stmt WrapPipelineStageContext(Stmt stmt,
                                const PrimExpr &normalized_access_index,
                                const Optional<Integer> &pipeline_num_stages) {
    if (!(pipeline_num_stages && pipeline_num_stages.value()->value > 1)) {
      return stmt;
    }
    PrimExpr ns = IntImm(DataType::Int(32), pipeline_num_stages.value()->value);
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
    if (pipeline_num_stages.value()->value <= 1) {
      parity_expr =
          FloorMod(normalized_access_index, IntImm(DataType::Int(32), 2));
    } else {
      PrimExpr ns =
          IntImm(DataType::Int(32), pipeline_num_stages.value()->value);
      parity_expr = FloorMod(FloorDiv(normalized_access_index, ns),
                             IntImm(DataType::Int(32), 2));
    }
    return analyzer_.Simplify(parity_expr);
  }

  static bool IsAsyncCommitQueueScope(const AttrStmtNode *attr) {
    return attr && attr->attr_key == tir::attr::async_commit_queue_scope;
  }

  static bool IsAsyncWaitQueueScope(const AttrStmtNode *attr) {
    return attr && attr->attr_key == tir::attr::async_wait_queue_scope;
  }

  static bool IsAsyncWaitInflightCount(const AttrStmtNode *attr) {
    return attr && attr->attr_key == tir::attr::async_wait_inflight_count;
  }

  static int
  PipelinedRetainGroups(const Optional<Integer> &pipeline_num_stages) {
    int retain = 1;
    if (pipeline_num_stages) {
      retain =
          std::max(0, static_cast<int>(pipeline_num_stages.value()->value) - 1);
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

  HeadAsyncSyncInfo AnalyzeHeadAsyncSync(const Stmt &stmt,
                                         HeadSeqMode seq_mode) const {
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return AnalyzeHeadAsyncSync(let->body, seq_mode);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (IsAsyncWaitQueueScope(attr)) {
        if (auto wait_n = TryGetStaticAsyncWaitCount(attr)) {
          return {HeadAsyncSyncKind::kWaitStatic, *wait_n};
        }
        return {HeadAsyncSyncKind::kWaitDynamic, 0};
      }
      if (IsAsyncCommitQueueScope(attr)) {
        return {HeadAsyncSyncKind::kCommit, 0};
      }
      if (IsAsyncWaitInflightCount(attr)) {
        return {HeadAsyncSyncKind::kBlocked, 0};
      }
      return AnalyzeHeadAsyncSync(attr->body, seq_mode);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return {};
      }
      if (seq_mode == HeadSeqMode::kSingletonOnly && seq->seq.size() != 1) {
        return {HeadAsyncSyncKind::kBlocked, 0};
      }
      return AnalyzeHeadAsyncSync(seq->seq[0], seq_mode);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return AnalyzeHeadAsyncSync(block->body, seq_mode);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return AnalyzeHeadAsyncSync(realize->block->body, seq_mode);
      }
      return {HeadAsyncSyncKind::kBlocked, 0};
    }
    return {};
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

  Stmt RewriteWaitStaticInSimpleWrapper(const Stmt &stmt, int new_wait_n,
                                        bool *changed) const {
    ClassifiedAsyncSyncStmt cls = ClassifySimpleAsyncSyncStmt(stmt);
    if (cls.kind != AsyncSyncStmtKind::kWaitStatic) {
      return stmt;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (IsAsyncWaitQueueScope(attr)) {
        *changed = true;
        return MakeStaticAsyncWaitStmtLike(attr, new_wait_n);
      }
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      Stmt new_body =
          RewriteWaitStaticInSimpleWrapper(let->body, new_wait_n, changed);
      if (*changed) {
        return LetStmt(let->var, let->value, new_body, let->span);
      }
      return stmt;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      Stmt new_body =
          RewriteWaitStaticInSimpleWrapper(attr->body, new_wait_n, changed);
      if (*changed) {
        return AttrStmt(attr->node, attr->attr_key, attr->value, new_body,
                        attr->span);
      }
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        Stmt inner =
            RewriteWaitStaticInSimpleWrapper(seq->seq[0], new_wait_n, changed);
        if (*changed) {
          return SeqStmt({inner});
        }
      }
      return stmt;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      Stmt inner =
          RewriteWaitStaticInSimpleWrapper(block->body, new_wait_n, changed);
      if (*changed) {
        Block new_block = Downcast<Block>(stmt);
        new_block.CopyOnWrite()->body = inner;
        return new_block;
      }
      return stmt;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        Stmt inner = RewriteWaitStaticInSimpleWrapper(realize->block->body,
                                                      new_wait_n, changed);
        if (*changed) {
          Block new_block = realize->block;
          new_block.CopyOnWrite()->body = inner;
          return BlockRealize(realize->iter_values, realize->predicate,
                              new_block, realize->span);
        }
      }
      return stmt;
    }
    return stmt;
  }

  std::optional<int> TryGetHeadStaticWaitCount(const Stmt &stmt) const {
    HeadAsyncSyncInfo info =
        AnalyzeHeadAsyncSync(stmt, HeadSeqMode::kTakeFirstElement);
    if (info.kind == HeadAsyncSyncKind::kWaitStatic) {
      return info.wait_n;
    }
    return std::nullopt;
  }

  std::optional<int> TryGetFirstStaticWaitCount(const Stmt &stmt) const {
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return TryGetFirstStaticWaitCount(let->body);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      HeadAsyncSyncInfo info =
          AnalyzeHeadAsyncSync(stmt, HeadSeqMode::kTakeFirstElement);
      if (info.kind == HeadAsyncSyncKind::kWaitStatic) {
        return info.wait_n;
      }
      if (info.IsBoundary()) {
        return std::nullopt;
      }
      return TryGetFirstStaticWaitCount(attr->body);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &elem : seq->seq) {
        HeadAsyncSyncInfo info =
            AnalyzeHeadAsyncSync(elem, HeadSeqMode::kTakeFirstElement);
        if (info.kind == HeadAsyncSyncKind::kWaitStatic) {
          return info.wait_n;
        }
        if (info.IsBoundary() || ContainsAsyncSyncScopes(elem)) {
          return std::nullopt;
        }
      }
      return std::nullopt;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return TryGetFirstStaticWaitCount(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return TryGetFirstStaticWaitCount(realize->block->body);
      }
    }
    return std::nullopt;
  }

  Stmt RewriteHeadStaticWaitInWrapper(const Stmt &stmt, int new_wait_n,
                                      bool *changed) const {
    if (const auto *let = stmt.as<LetStmtNode>()) {
      Stmt new_body =
          RewriteHeadStaticWaitInWrapper(let->body, new_wait_n, changed);
      if (*changed) {
        return LetStmt(let->var, let->value, new_body, let->span);
      }
      return stmt;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (IsAsyncWaitQueueScope(attr)) {
        *changed = true;
        return MakeStaticAsyncWaitStmtLike(attr, new_wait_n);
      }
      Stmt new_body =
          RewriteHeadStaticWaitInWrapper(attr->body, new_wait_n, changed);
      if (*changed) {
        return AttrStmt(attr->node, attr->attr_key, attr->value, new_body,
                        attr->span);
      }
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return stmt;
      }
      Array<Stmt> new_seq = seq->seq;
      new_seq.Set(
          0, RewriteHeadStaticWaitInWrapper(seq->seq[0], new_wait_n, changed));
      if (*changed) {
        return SeqStmt(new_seq);
      }
      return stmt;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      Stmt new_body =
          RewriteHeadStaticWaitInWrapper(block->body, new_wait_n, changed);
      if (*changed) {
        Block new_block = Downcast<Block>(stmt);
        new_block.CopyOnWrite()->body = new_body;
        return new_block;
      }
      return stmt;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        Stmt new_body = RewriteHeadStaticWaitInWrapper(realize->block->body,
                                                       new_wait_n, changed);
        if (*changed) {
          Block new_block = realize->block;
          new_block.CopyOnWrite()->body = new_body;
          return BlockRealize(realize->iter_values, realize->predicate,
                              new_block, realize->span);
        }
      }
      return stmt;
    }
    return stmt;
  }

  Stmt RewriteFirstStaticWaitInWrapper(const Stmt &stmt, int new_wait_n,
                                       bool *changed) const {
    if (const auto *let = stmt.as<LetStmtNode>()) {
      Stmt new_body =
          RewriteFirstStaticWaitInWrapper(let->body, new_wait_n, changed);
      if (*changed) {
        return LetStmt(let->var, let->value, new_body, let->span);
      }
      return stmt;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (IsAsyncWaitQueueScope(attr)) {
        *changed = true;
        return MakeStaticAsyncWaitStmtLike(attr, new_wait_n);
      }
      if (IsAsyncCommitQueueScope(attr) || IsAsyncWaitInflightCount(attr)) {
        return stmt;
      }
      Stmt new_body =
          RewriteFirstStaticWaitInWrapper(attr->body, new_wait_n, changed);
      if (*changed) {
        return AttrStmt(attr->node, attr->attr_key, attr->value, new_body,
                        attr->span);
      }
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> new_seq = seq->seq;
      for (int i = 0, n = static_cast<int>(new_seq.size()); i < n; ++i) {
        Stmt updated =
            RewriteFirstStaticWaitInWrapper(new_seq[i], new_wait_n, changed);
        if (*changed) {
          new_seq.Set(i, updated);
          return SeqStmt(new_seq);
        }
        if (ContainsAsyncSyncScopes(new_seq[i])) {
          return stmt;
        }
      }
      return stmt;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      Stmt new_body =
          RewriteFirstStaticWaitInWrapper(block->body, new_wait_n, changed);
      if (*changed) {
        Block new_block = Downcast<Block>(stmt);
        new_block.CopyOnWrite()->body = new_body;
        return new_block;
      }
      return stmt;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        Stmt new_body = RewriteFirstStaticWaitInWrapper(realize->block->body,
                                                        new_wait_n, changed);
        if (*changed) {
          Block new_block = realize->block;
          new_block.CopyOnWrite()->body = new_body;
          return BlockRealize(realize->iter_values, realize->predicate,
                              new_block, realize->span);
        }
      }
      return stmt;
    }
    return stmt;
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
            bool changed_wait = false;
            body.Set(i, RewriteWaitStaticInSimpleWrapper(
                            body[i], candidate_wait_n, &changed_wait));
            if (changed_wait) {
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

  Stmt RelaxLoopWaitsInSimpleWrapper(const Stmt &stmt, int pre_outstanding_lb,
                                     bool *changed) const {
    if (const auto *loop = stmt.as<ForNode>()) {
      Stmt relaxed =
          MaybeRelaxLoopWaits(Downcast<For>(stmt), pre_outstanding_lb);
      *changed = !relaxed.same_as(stmt);
      return relaxed;
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      Stmt new_body =
          RelaxLoopWaitsInSimpleWrapper(let->body, pre_outstanding_lb, changed);
      if (*changed) {
        return LetStmt(let->var, let->value, new_body, let->span);
      }
      return stmt;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      Stmt new_body = RelaxLoopWaitsInSimpleWrapper(
          attr->body, pre_outstanding_lb, changed);
      if (*changed) {
        return AttrStmt(attr->node, attr->attr_key, attr->value, new_body,
                        attr->span);
      }
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        Stmt inner = RelaxLoopWaitsInSimpleWrapper(seq->seq[0],
                                                   pre_outstanding_lb, changed);
        if (*changed) {
          return SeqStmt({inner});
        }
      }
      return stmt;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      Stmt new_body = RelaxLoopWaitsInSimpleWrapper(
          block->body, pre_outstanding_lb, changed);
      if (*changed) {
        Block new_block = Downcast<Block>(stmt);
        new_block.CopyOnWrite()->body = new_body;
        return new_block;
      }
      return stmt;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        Stmt new_body = RelaxLoopWaitsInSimpleWrapper(
            realize->block->body, pre_outstanding_lb, changed);
        if (*changed) {
          Block new_block = realize->block;
          new_block.CopyOnWrite()->body = new_body;
          return BlockRealize(realize->iter_values, realize->predicate,
                              new_block, realize->span);
        }
      }
      return stmt;
    }
    return stmt;
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
        bool changed_loop = false;
        current = rewriter_->RelaxLoopWaitsInSimpleWrapper(
            current, outstanding_lb, &changed_loop);
        if (changed_loop) {
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
      bool changed = false;
      int idx = suffix_wait_indices[pos];
      seq.Set(idx, RewriteFirstStaticWaitInWrapper(seq[idx], retain, &changed));
    }
    return seq;
  }

  void PopulateWaitCounts(
      const std::vector<RewrittenStmtInfo> &new_stmts,
      arith::Analyzer *ana_normalized,
      const std::unordered_map<const BufferNode *, int> &buffer_to_commit_group,
      std::map<int, AsyncStateLocal> *async_states_local) {
    for (size_t i = 0; i < new_stmts.size(); ++i) {
      if (new_stmts[i].is_async) {
        for (const BufferRegion &write_region : new_stmts[i].writes) {
          (*async_states_local)[new_stmts[i].stage].seen.insert(
              write_region->buffer.get());
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
          auto commit_group_id =
              buffer_to_commit_group.at(read_region->buffer.get());
          if (!need_wait_count[commit_group_id]) {
            continue;
          }
          dependent_commit_groups.push_back(commit_group_id);
          if (!dep_local_state.seen.count(read_region->buffer.get())) {
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
      return AttrStmt(zero, tir::attr::async_wait_queue_scope, stage_id,
                      AttrStmt(zero, tir::attr::async_wait_inflight_count,
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
                   tir::attr::async_commit_queue_scope, stage_id, group_body);
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
      return BlockRealize({}, Bool(true), MakeBlock(Evaluate(0), {}));
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
            if (pipeline_num_stages.value()->value > 1) {
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
    std::unordered_map<const BufferNode *, int> buffer_to_commit_group;

    for (const Block &block : ordered_stmts_) {
      const auto &pipeline_anno = pipeline_info_.at(block);
      int stage = pipeline_anno.stage;
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
      inbound = ana_normalized.Simplify(inbound);
      if (is_zero(inbound)) {
        continue;
      }
      new_block = Downcast<Block>(Substitute(
          new_block, {{pipeline_loop_->loop_var, normalized_access_index}}));

      Stmt rewritten_stmt = BlockRealize({}, inbound, new_block);
      rewritten_stmt = WrapLoopDependentWrappers(std::move(rewritten_stmt),
                                                 normalized_access_index);
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
          async_states_[stage].dst_buffers.insert(write_region->buffer.get());
          buffer_to_commit_group[write_region->buffer.get()] = commit_group_id;
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
        rewritten_stmt =
            SimtProducerAnnotator::Annotate(rewritten_stmt, target_);
        rewritten_stmt = AttrStmt(make_zero(DataType::Int(32)),
                                  tir::attr::async_scope, 1, rewritten_stmt);
      }
      if (pipeline_mbar_phase) {
        rewritten_stmt = TileOpMbarPhaseAnnotator::Annotate(
            rewritten_stmt, pipeline_mbar_phase.value());
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
        if (kv.first != tir::attr::software_pipeline_stage &&
            kv.first != tir::attr::software_pipeline_order &&
            kv.first != tir::attr::software_pipeline_async_stages &&
            kv.first != kPipelineAsyncProducers &&
            kv.first != kPipelineAsyncProducerGroups &&
            kv.first != kPipelineTmaCopies && kv.first != "num_stages") {
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
    Stmt result = BlockRealize({}, Bool(true),
                               MakeBlock(new_loop, buffer_data_to_buffer_));
    if (pipeline_num_stages) {
      if (pipeline_num_stages.value()->value > 1) {
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
  int max_stage_ = -1;
  Map<Buffer, Buffer> buffer_remap_;
  Optional<Target> target_;
  Array<Block> ordered_stmts_;
  std::vector<LetWrapper> loop_var_let_wrappers_;
  std::vector<IfWrapper> loop_var_if_wrappers_;
  std::map<int, AsyncStateGlobal> async_states_;
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

// ---------------------------------------------------------------------------
// Helpers for pipeline-level TMA barrier management
// ---------------------------------------------------------------------------

/*!
 * \brief Rewrite a block's body, converting tl.tileop.copy calls to
 *        tl.tileop.tma_copy with barrier and emit_arrive annotations.
 */
class CopyToTmaCopyRewriter : public StmtExprMutator {
public:
  CopyToTmaCopyRewriter(const Buffer &barrier_buf, PrimExpr barrier_id,
                        bool emit_arrive = true)
      : barrier_buf_(barrier_buf), barrier_id_(std::move(barrier_id)),
        emit_arrive_(emit_arrive) {}

  PrimExpr VisitExpr_(const CallNode *op) final {
    static const Op &copy_op = Op::Get("tl.tileop.copy");
    static const Op &tma_copy_op = Op::Get("tl.tileop.tma_copy");
    static const Op &im2col_op = Op::Get("tl.tileop.c2d_im2col");
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(copy_op)) {
      auto new_annotations = call->annotations;
      new_annotations.Set("barrier", MakeBarrierRef(barrier_buf_, barrier_id_));
      new_annotations.Set("is_tma_copy", IntImm(DataType::Int(32), 1));
      new_annotations.Set("emit_arrive",
                          IntImm(DataType::Int(32), emit_arrive_ ? 1 : 0));
      return Call(call->dtype, tma_copy_op, call->args, new_annotations,
                  call->span);
    }
    // Annotate c2d_im2col with pipeline barrier so its Lower() uses it
    // instead of allocating a separate internal barrier.
    if (call->op.same_as(im2col_op)) {
      auto new_annotations = call->annotations;
      new_annotations.Set("barrier", MakeBarrierRef(barrier_buf_, barrier_id_));
      new_annotations.Set("emit_arrive",
                          IntImm(DataType::Int(32), emit_arrive_ ? 1 : 0));
      return Call(call->dtype, call->op, call->args, new_annotations,
                  call->span);
    }
    return call;
  }

private:
  Buffer barrier_buf_;
  PrimExpr barrier_id_;
  bool emit_arrive_;
};

// ---------------------------------------------------------------------------
// ExpandPipelineBarriers — multi-version all barrier buffers for pipelining
// ---------------------------------------------------------------------------

/// Collect all shared.barrier Buffer objects referenced in a statement.
class BarrierBufferCollector : public StmtExprVisitor {
public:
  static std::vector<Buffer>
  Collect(const Array<Block> &blocks,
          const Map<Var, Buffer> &buffer_data_to_buffer) {
    BarrierBufferCollector c(buffer_data_to_buffer);
    for (const auto &block : blocks) {
      c(block->body);
    }
    return {c.barriers_.begin(), c.barriers_.end()};
  }

private:
  explicit BarrierBufferCollector(const Map<Var, Buffer> &buf_map)
      : buf_map_(buf_map) {}

  void VisitExpr_(const BufferLoadNode *op) final {
    if (op->buffer.scope() == "shared.barrier" ||
        op->buffer.scope() == "shared.cluster_barrier") {
      if (!seen_.count(op->buffer.get())) {
        seen_.insert(op->buffer.get());
        barriers_.push_back(op->buffer);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->buffer.scope() == "shared.barrier" ||
        op->buffer.scope() == "shared.cluster_barrier") {
      if (!seen_.count(op->buffer.get())) {
        seen_.insert(op->buffer.get());
        barriers_.push_back(op->buffer);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // Also check barrier refs inside Call annotations (e.g., tma_copy barrier).
  void VisitExpr_(const CallNode *op) final {
    for (const auto &[key, val] : op->annotations) {
      if (auto load = val.as<BufferLoadNode>()) {
        if (load->buffer.scope() == "shared.barrier" ||
            load->buffer.scope() == "shared.cluster_barrier") {
          if (!seen_.count(load->buffer.get())) {
            seen_.insert(load->buffer.get());
            barriers_.push_back(load->buffer);
          }
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  const Map<Var, Buffer> &buf_map_;
  std::unordered_set<const BufferNode *> seen_;
  std::vector<Buffer> barriers_;
};

/// Rewrite barrier references: expand indices and rewrite parity.
class BarrierIndexRewriter : public StmtExprMutator {
public:
  BarrierIndexRewriter(
      const std::unordered_map<const BufferNode *, Buffer> &old_to_new,
      const std::unordered_map<const BufferNode *, PrimExpr> &old_shapes,
      PrimExpr stage_expr, PrimExpr parity_cycle, Var loop_var,
      PrimExpr loop_min)
      : old_to_new_(old_to_new), old_shapes_(old_shapes),
        stage_expr_(std::move(stage_expr)),
        parity_cycle_(std::move(parity_cycle)), loop_var_(std::move(loop_var)),
        loop_min_(std::move(loop_min)) {}

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = old_to_new_.find(load->buffer.get());
    if (it != old_to_new_.end()) {
      auto *n = load.CopyOnWrite();
      PrimExpr old_size = old_shapes_.at(load->buffer.get());
      n->buffer = it->second;
      n->indices.Set(0, stage_expr_ * old_size + n->indices[0]);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = old_to_new_.find(store->buffer.get());
    if (it != old_to_new_.end()) {
      auto *n = store.CopyOnWrite();
      PrimExpr old_size = old_shapes_.at(store->buffer.get());
      n->buffer = it->second;
      n->indices.Set(0, stage_expr_ * old_size + n->indices[0]);
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));

    // Rewrite barrier refs inside annotations (e.g., tma_copy "barrier").
    bool anno_changed = false;
    Map<String, ObjectRef> new_annos = call->annotations;
    for (const auto &[key, val] : call->annotations) {
      if (auto load = val.as<BufferLoadNode>()) {
        auto it = old_to_new_.find(load->buffer.get());
        if (it != old_to_new_.end()) {
          PrimExpr old_size = old_shapes_.at(load->buffer.get());
          auto new_load = BufferLoad(
              it->second, {stage_expr_ * old_size + load->indices[0]});
          new_annos.Set(key, new_load);
          anno_changed = true;
        }
      }
    }
    if (anno_changed) {
      call = Call(call->dtype, call->op, call->args, new_annos, call->span);
    }

    // Rewrite mbarrier_wait_parity parity argument.
    if (call->op.same_as(mbarrier_wait_parity()) && call->args.size() >= 2) {
      if (auto load = call->args[0].as<BufferLoadNode>()) {
        // Check if the barrier ref (possibly already rewritten above)
        // targets one of our expanded barriers.
        const BufferNode *target = load->buffer.get();
        bool is_expanded = false;
        for (const auto &[old_buf, new_buf] : old_to_new_) {
          if (new_buf.get() == target) {
            is_expanded = true;
            break;
          }
        }
        if (is_expanded) {
          // Compute initial-phase offset from the user's original parity.
          arith::Analyzer analyzer;
          PrimExpr user_parity = call->args[1];
          PrimExpr user_parity_at_min = analyzer.Simplify(
              tir::Substitute(user_parity, {{loop_var_, loop_min_}}));
          // New parity = (iteration_block + offset) % 2
          PrimExpr offset = IntImm(DataType::Int(32), 0);
          if (const int64_t *imm = as_const_int(user_parity_at_min)) {
            offset = IntImm(DataType::Int(32), *imm % 2);
          }
          PrimExpr new_parity = FloorMod(parity_cycle_ + offset, 2);
          Array<PrimExpr> new_args = call->args;
          new_args.Set(1, new_parity);
          return Call(call->dtype, call->op, new_args, call->annotations,
                      call->span);
        }
      }
    }
    return call;
  }

private:
  const std::unordered_map<const BufferNode *, Buffer> &old_to_new_;
  const std::unordered_map<const BufferNode *, PrimExpr> &old_shapes_;
  PrimExpr stage_expr_;
  PrimExpr parity_cycle_;
  Var loop_var_;
  PrimExpr loop_min_;
};

/// Expand all shared.barrier buffers in the pipeline body from [N] to
/// [N * num_stages], rewrite barrier indices to include stage offset, and
/// rewrite mbarrier_wait_parity parity expressions.
///
/// This is the unified barrier multi-versioning path that replaces the old
/// late barrier-only fixup in OptimizeForTarget.
/// Returns a map of old→new barrier buffers for outer block alloc_buffers
/// update.
Map<Buffer, Buffer> ExpandPipelineBarriers(
    Array<Block> &original_order, PipelineInfo &pipeline_info,
    Map<Var, Buffer> &buffer_data_to_buffer,
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
        &allocated_buffers,
    Array<Buffer> &block_local_allocs, Array<Buffer> &pipeline_allocs,
    Var loop_var, PrimExpr loop_min, int num_stages) {
  if (num_stages <= 1)
    return {};

  // Only expand barriers that have explicit ptx_arrive_barrier calls in the
  // loop body.  This distinguishes pipeline synchronization barriers (where
  // arrive/wait are user-managed and need per-stage slots) from barriers
  // whose arrival is managed internally by tile-ops (e.g., tcgen05 MMA
  // arrive barriers) — those should NOT be pipeline-expanded.
  // ISP-created pipeline_mbar is handled specially: it's always in
  // block_local_allocs and was just created, so include it too.
  std::unordered_set<const BufferNode *> local_barrier_set;
  for (const Buffer &buf : block_local_allocs) {
    if (buf.scope() == "shared.barrier" ||
        buf.scope() == "shared.cluster_barrier")
      local_barrier_set.insert(buf.get());
  }

  // Find barriers that have explicit ptx_arrive_barrier calls.
  class ArriveBarrierDetector : public StmtExprVisitor {
  public:
    std::unordered_set<const BufferNode *> arrived_;
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(builtin::ptx_arrive_barrier()) && !op->args.empty()) {
        if (auto load = op->args[0].as<BufferLoadNode>()) {
          arrived_.insert(load->buffer.get());
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  };
  ArriveBarrierDetector arrive_det;
  for (const auto &block : original_order) {
    arrive_det(block->body);
  }

  std::vector<Buffer> all_referenced =
      BarrierBufferCollector::Collect(original_order, buffer_data_to_buffer);
  std::vector<Buffer> barriers;
  for (const Buffer &buf : all_referenced) {
    // Include if: (a) it's an ISP-created local barrier, OR
    //             (b) it has explicit ptx_arrive_barrier calls.
    if (local_barrier_set.count(buf.get()) ||
        arrive_det.arrived_.count(buf.get())) {
      barriers.push_back(buf);
    }
  }
  if (barriers.empty())
    return {};

  PrimExpr ns = IntImm(DataType::Int(32), num_stages);
  PrimExpr stage_expr = FloorMod(loop_var - loop_min, ns);
  PrimExpr parity_cycle = FloorMod(FloorDiv(loop_var - loop_min, ns), 2);

  auto replace_in_array = [](Array<Buffer> &arr, const Buffer &old_buf,
                             const Buffer &new_buf) {
    for (size_t i = 0; i < arr.size(); ++i) {
      if (arr[i].same_as(old_buf)) {
        arr.Set(i, new_buf);
      }
    }
  };

  // Create expanded buffer for each barrier.
  std::unordered_map<const BufferNode *, Buffer> old_to_new;
  std::unordered_map<const BufferNode *, PrimExpr> old_shapes;
  for (const Buffer &buf : barriers) {
    old_shapes[buf.get()] = buf->shape[0];
    ObjectPtr<BufferNode> new_node =
        tvm::ffi::make_object<BufferNode>(*(buf.get()));
    new_node->shape = {PrimExpr(num_stages) * buf->shape[0]};
    Buffer new_buf(new_node);
    old_to_new[buf.get()] = new_buf;

    // Update all maps and alloc arrays.
    buffer_data_to_buffer.Set(buf->data, new_buf);
    allocated_buffers.erase(buf);
    allocated_buffers.insert(new_buf);
    replace_in_array(block_local_allocs, buf, new_buf);
    replace_in_array(pipeline_allocs, buf, new_buf);
  }

  // Rewrite all blocks.
  BarrierIndexRewriter rewriter(old_to_new, old_shapes, stage_expr,
                                parity_cycle, loop_var, loop_min);
  for (size_t i = 0; i < original_order.size(); ++i) {
    Block old_block = original_order[i];
    Stmt new_body = rewriter(old_block->body);
    if (!new_body.same_as(old_block->body)) {
      // Also rewrite alloc_buffers in the block (barriers may be allocated
      // here).
      Array<Buffer> new_allocs;
      for (const Buffer &ab : old_block->alloc_buffers) {
        auto it = old_to_new.find(ab.get());
        new_allocs.push_back(it != old_to_new.end() ? it->second : ab);
      }
      Block new_block(old_block->iter_vars, old_block->reads, old_block->writes,
                      old_block->name_hint, new_body, old_block->init,
                      new_allocs, old_block->match_buffers,
                      old_block->annotations);
      PipelineAnnotation anno = pipeline_info.at(old_block);
      pipeline_info.erase(old_block);
      pipeline_info.emplace(new_block, anno);
      original_order.Set(i, new_block);
    }
  }

  // Return the old→new mapping for outer block alloc_buffers update.
  Map<Buffer, Buffer> result;
  for (const auto &[old_ptr, new_buf] : old_to_new) {
    for (const Buffer &old_buf : barriers) {
      if (old_buf.get() == old_ptr) {
        result.Set(old_buf, new_buf);
        break;
      }
    }
  }
  return result;
}

/*!
 * \brief Rewrite TMA-eligible copy blocks in the pipeline body for
 *        pipeline-level barrier management.
 *
 * For each TMA copy: convert tl.tileop.copy → tl.tileop.tma_copy with a
 * per-stage barrier slot and emit_arrive=1 so LowerTileOp emits arrive inside
 * the thread-0 guard.
 *
 * For the first consumer stage block: prepend mbarrier_wait_parity with
 * stage-indexed barrier reference and parity expression.
 *
 * \param original_order  In/out: blocks in original pipeline order.
 * \param pipeline_info   In/out: block → PipelineAnnotation mapping.
 * \param tma_copies      Per-statement TMA flag array from PipelinePlanning.
 * \param buffer_data_to_buffer  In/out: buffer var → Buffer mapping.
 * \param allocated_buffers      In/out: set of allocated buffers.
 * \param block_local_allocs     In/out: buffers allocated in the pipeline
 * block.
 * \return The newly created barrier buffer (undefined if no TMA copies).
 */
Buffer RewritePipelineTmaBarriers(
    Array<Block> &original_order, PipelineInfo &pipeline_info,
    const Array<Integer> &tma_copies, Map<Var, Buffer> &buffer_data_to_buffer,
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
        &allocated_buffers,
    Array<Buffer> &block_local_allocs, Var loop_var, PrimExpr loop_min,
    int num_stages) {
  // Count TMA copies
  int num_tma = 0;
  for (const auto &tc : tma_copies) {
    if (tc->value != 0)
      num_tma++;
  }
  if (num_tma == 0)
    return Buffer();

  // Create pipeline barrier buffer with a single slot.  The generic
  // ExpandPipelineBarriers pass (called later) will expand it to
  // num_stages slots along with all other barrier buffers.
  Buffer barrier_buf = CreateMBarrierBuffer("pipeline_mbar", 1);
  buffer_data_to_buffer.Set(barrier_buf->data, barrier_buf);
  allocated_buffers.insert(barrier_buf);
  block_local_allocs.push_back(barrier_buf);

  // Find the index of the last TMA copy for arrive emission.
  int last_tma_idx = -1;
  for (size_t i = 0; i < original_order.size(); i++) {
    if (static_cast<int>(tma_copies[i]->value) != 0)
      last_tma_idx = static_cast<int>(i);
  }

  // Phase 1: Rewrite TMA copy blocks — all share barrier slot 0.
  // ExpandPipelineBarriers (called later) will rewrite indices to be
  // stage-dependent.  Only the last TMA copy emits arrive.
  for (size_t i = 0; i < original_order.size(); i++) {
    if (static_cast<int>(tma_copies[i]->value) == 0)
      continue;

    bool is_last = (static_cast<int>(i) == last_tma_idx);
    Block old_block = original_order[i];
    CopyToTmaCopyRewriter rewriter(barrier_buf,
                                   /*barrier_id=*/IntImm(DataType::Int(32), 0),
                                   /*emit_arrive=*/is_last);
    Stmt new_body = rewriter(old_block->body);

    Block new_block(old_block->iter_vars, old_block->reads, old_block->writes,
                    old_block->name_hint, new_body, old_block->init,
                    old_block->alloc_buffers, old_block->match_buffers,
                    old_block->annotations);

    PipelineAnnotation anno = pipeline_info.at(old_block);
    pipeline_info.erase(old_block);
    pipeline_info.emplace(new_block, anno);
    original_order.Set(i, new_block);
  }

  // Phase 2: Insert waits in consumer blocks (blocks that depend on TMA data).
  // For simplicity, we insert waits before the first block whose stage > 0.
  // This covers the common case where stage 0 = producers, stage 1 = consumer.
  bool waits_inserted = false;
  for (size_t i = 0; i < original_order.size(); i++) {
    if (waits_inserted)
      break;
    Block old_block = original_order[i];
    int stage = pipeline_info.at(old_block).stage;
    if (stage == 0)
      continue; // still in producer stage

    // Wait on barrier slot 0 with single-slot parity.
    // ExpandPipelineBarriers will rewrite index and parity for versioning.
    Array<Stmt> wait_stmts;
    {
      PrimExpr barrier_ref =
          MakeBarrierRef(barrier_buf, IntImm(DataType::Int(32), 0));
      PrimExpr ns = IntImm(DataType::Int(32), num_stages);
      PrimExpr parity = FloorMod(FloorDiv(loop_var - loop_min, ns), 2);
      wait_stmts.push_back(Evaluate(Call(
          DataType::Handle(), mbarrier_wait_parity(), {barrier_ref, parity})));
    }
    wait_stmts.push_back(old_block->body);
    Stmt new_body = SeqStmt(wait_stmts);

    Block new_block(old_block->iter_vars, old_block->reads, old_block->writes,
                    old_block->name_hint, new_body, old_block->init,
                    old_block->alloc_buffers, old_block->match_buffers,
                    old_block->annotations);

    PipelineAnnotation anno = pipeline_info.at(old_block);
    pipeline_info.erase(old_block);
    pipeline_info.emplace(new_block, anno);
    original_order.Set(i, new_block);
    waits_inserted = true;
  }

  return barrier_buf;
}

class PipelineInjector : private StmtExprMutator {
public:
  static Stmt Inject(const PrimFunc &func) {
    auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    PipelineInjector injector(global_symbol, target);
    for (const auto &kv : func->buffer_map) {
      const Buffer &buffer = kv.second;
      injector.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return injector(func->body);
  }

private:
  explicit PipelineInjector(Optional<String> global_symbol,
                            Optional<Target> target)
      : global_symbol_(std::move(global_symbol)), target_(std::move(target)) {}

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

  bool HasOverlappableStages(const PipelineInfo &pipeline_info) const {
    std::optional<int> first_stage;
    for (const auto &pair : pipeline_info) {
      int stage = pair.second.stage;
      if (!first_stage.has_value()) {
        first_stage = stage;
      } else if (stage != first_stage.value()) {
        return true;
      }
    }
    return false;
  }

  Map<String, Any>
  StripPipelineAnnotations(const Map<String, Any> &annotations) const {
    Map<String, Any> preserved_annotations;
    for (const auto &kv : annotations) {
      const String &key = kv.first;
      if (key != tir::attr::software_pipeline_stage &&
          key != tir::attr::software_pipeline_order &&
          key != tir::attr::software_pipeline_async_stages &&
          key != kPipelineAsyncProducers &&
          key != kPipelineAsyncProducerGroups && key != kPipelineTmaCopies &&
          key != "num_stages" && key != "tl_pipelined_num_stages") {
        preserved_annotations.Set(key, kv.second);
      }
    }
    return preserved_annotations;
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

    std::unordered_set<int> pipeline_async_stages;
    if (auto async_annot =
            op->annotations.Get(tir::attr::software_pipeline_async_stages)) {
      for (const Integer &stage :
           Downcast<Array<Integer>>(async_annot.value())) {
        pipeline_async_stages.insert(static_cast<int>(stage->value));
      }
    }
    Optional<Array<Integer>> pipeline_async_producers;
    if (auto async_producers_anno =
            op->annotations.Get(kPipelineAsyncProducers)) {
      auto async_flags = Downcast<Array<Integer>>(async_producers_anno.value());
      CHECK_EQ(async_flags.size(), original_order.size())
          << "PrimFunc " << global_symbol_ << " has original order "
          << original_order.Map(
                 [](const auto &block) { return block->name_hint; })
          << ", but async producer annotation is " << async_flags
          << " with different size";
      pipeline_async_producers = async_flags;
    }
    Optional<Array<Integer>> pipeline_async_producer_groups;
    if (auto async_groups_anno =
            op->annotations.Get(kPipelineAsyncProducerGroups)) {
      auto async_group_ids =
          Downcast<Array<Integer>>(async_groups_anno.value());
      CHECK_EQ(async_group_ids.size(), original_order.size())
          << "PrimFunc " << global_symbol_ << " has original order "
          << original_order.Map(
                 [](const auto &block) { return block->name_hint; })
          << ", but async producer group annotation is " << async_group_ids
          << " with different size";
      pipeline_async_producer_groups = async_group_ids;
    }

    for (size_t i = 0; i < pipeline_stages.size(); i++) {
      int stage = static_cast<int>(pipeline_stages[i]->value);
      bool is_async_candidate =
          pipeline_async_producers
              ? (pipeline_async_producers.value()[i]->value != 0)
              : (pipeline_async_stages.count(stage) > 0);
      // Stages that already spell out async behavior themselves keep that
      // ownership. The pipeline pass only injects async producer semantics for
      // "plain" producer stages that do not already contain cp.async / async
      // queue operations.
      bool is_async = is_async_candidate &&
                      !ContainsExplicitAsyncIntrinsics(original_order[i]->body);
      PipelineAnnotation stage_order{
          stage,
          /*order=*/static_cast<int>(pipeline_orders[i]->value),
          /*async=*/is_async,
          /*async_group_id=*/
          pipeline_async_producer_groups
              ? static_cast<int>(
                    pipeline_async_producer_groups.value()[i]->value)
              : -1};
      pipeline_info.emplace(original_order[i], stage_order);
    }

    ValidatePipelineBody(pipeline_info, original_order);

    if (!HasOverlappableStages(pipeline_info)) {
      if (const auto *realize = op->body.as<BlockRealizeNode>()) {
        const auto &block = realize->block;
        for (const auto &buffer : block->alloc_buffers) {
          buffer_data_to_buffer_.erase(buffer->data);
          allocated_buffers_.erase(buffer);
        }
      }
      return For(for_node->loop_var, for_node->min, for_node->extent,
                 for_node->kind, for_node->body, for_node->thread_binding,
                 StripPipelineAnnotations(for_node->annotations),
                 for_node->step, for_node->span);
    }

    // Step 3.5: Pipeline-level TMA barrier management.
    // When TMA copies are present (without warp specialization), rewrite
    // them to use tl.tileop.tma_copy with shared pipeline barriers and insert
    // mbarrier_wait_parity before the first consumer stage.
    // Creates pipeline_mbar[pipeline_depth] at final size so LowerTileOp
    // uses the provided barrier instead of allocating separate per-copy ones.
    Buffer pipeline_barrier_buf;
    int num_pipeline_tma_copies = 0;
    {
      int max_stage = 0;
      for (const auto &pair : pipeline_info) {
        max_stage = std::max(max_stage, pair.second.stage);
      }
      // Use the actual pipeline depth (number of buffer copies) for barrier
      // sizing, not the SW pipeline stage count (max_stage + 1).
      // Even for pipeline_depth=1 we create a shared barrier so that
      // LowerTileOp uses it instead of allocating separate per-copy barriers.
      Optional<Integer> pipelined_num_stages = GetPipelineNumStages(op);
      int pipeline_depth =
          pipelined_num_stages.defined()
              ? static_cast<int>(pipelined_num_stages.value()->value)
              : max_stage + 1;
      // Clamp to at least 1 so we always allocate at least one barrier slot.
      pipeline_depth = std::max(pipeline_depth, 1);
      if (max_stage > 0) {
        if (auto tma_copies_anno = op->annotations.Get(kPipelineTmaCopies)) {
          auto tma_copies = Downcast<Array<Integer>>(tma_copies_anno.value());
          if (tma_copies.size() == original_order.size()) {
            for (const auto &tc : tma_copies) {
              if (tc->value != 0)
                num_pipeline_tma_copies++;
            }
            if (num_pipeline_tma_copies > 0) {
              pipeline_barrier_buf = RewritePipelineTmaBarriers(
                  original_order, pipeline_info, tma_copies,
                  buffer_data_to_buffer_, allocated_buffers_,
                  block_local_allocs, op->loop_var, op->min, pipeline_depth);
            }
          }
        }
      }
    }

    // Step 4: Rewrite the pipeline body.
    // local_allocs contains buffers allocated in the pipeline block itself.
    // pipeline_allocs contains all buffers that need multi-versioning,
    // including buffers from outer blocks.
    // Step 4.5: Expand all barrier buffers for pipelining.
    // This handles both ISP-created pipeline_mbar AND user-written
    // T.alloc_barrier, so that no late standalone barrier-only fixup is needed.
    // Must run BEFORE local_allocs is copied from block_local_allocs.
    {
      Optional<Integer> pipelined_ns = GetPipelineNumStages(op);
      int barrier_depth = 1;
      if (pipelined_ns.defined()) {
        barrier_depth = static_cast<int>(pipelined_ns.value()->value);
      } else if (op->annotations.count("num_stages")) {
        barrier_depth = static_cast<int>(
            Downcast<Integer>(op->annotations.Get("num_stages").value())
                ->value);
      }
      Map<Buffer, Buffer> barrier_remap = ExpandPipelineBarriers(
          original_order, pipeline_info, buffer_data_to_buffer_,
          allocated_buffers_, block_local_allocs, pipeline_allocs, op->loop_var,
          op->min, barrier_depth);
      // Register expanded barriers for outer block alloc_buffers update.
      for (const auto &[old_buf, new_buf] : barrier_remap) {
        pending_buffer_remap_.Set(old_buf, new_buf);
      }
    }

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
                              pipeline_info, target_, loop_var_let_wrappers,
                              loop_var_if_wrappers);
    Stmt pipeline = rewriter.BuildPipeline();
    subtree_modified_ = true;

    auto unwrap_outer_attrs = [](Stmt stmt) {
      std::vector<AttrStmt> attrs;
      while (const auto *attr = stmt.as<AttrStmtNode>()) {
        attrs.push_back(Downcast<AttrStmt>(stmt));
        stmt = attr->body;
      }
      return std::make_pair(attrs, stmt);
    };
    auto rewrap_outer_attrs = [](Stmt stmt,
                                 const std::vector<AttrStmt> &attrs) {
      for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
        stmt = AttrStmt((*it)->node, (*it)->attr_key, (*it)->value, stmt,
                        (*it)->span);
      }
      return stmt;
    };

    // Update barrier_init annotations for expanded barrier buffers.
    // For pipeline_mbar (ISP-created): add new entry with arrive_count=1 per
    // slot. For user barriers (T.alloc_barrier): replicate existing arrive
    // counts across the expanded slots.
    {
      auto [outer_attrs, inner_stmt] = unwrap_outer_attrs(pipeline);
      BlockRealize br = Downcast<BlockRealize>(inner_stmt);
      Block block = br->block;
      BlockNode *bn = block.CopyOnWrite();

      Map<Var, Array<PrimExpr>> barrier_init_map;
      if (bn->annotations.count("barrier_init")) {
        barrier_init_map = Downcast<Map<Var, Array<PrimExpr>>>(
            bn->annotations.Get("barrier_init").value());
      }
      bool changed = false;

      // Handle ISP-created pipeline barrier (needs new entry).
      if (pipeline_barrier_buf.defined()) {
        int num_slots = Downcast<IntImm>(pipeline_barrier_buf->shape[0])->value;
        // After ExpandPipelineBarriers, pipeline_mbar has been expanded.
        // Look up the expanded buffer via buffer_data_to_buffer_.
        Buffer expanded_buf =
            buffer_data_to_buffer_[pipeline_barrier_buf->data];
        int expanded_slots = Downcast<IntImm>(expanded_buf->shape[0])->value;
        Array<PrimExpr> counts;
        for (int s = 0; s < expanded_slots; ++s) {
          counts.push_back(IntImm(DataType::Int(32), 1));
        }
        barrier_init_map.Set(expanded_buf->data, counts);
        changed = true;
      }

      // Replicate existing barrier_init entries for expanded barriers.
      Map<Var, Array<PrimExpr>> updated_init;
      for (const auto &[var, counts] : barrier_init_map) {
        Buffer buf = buffer_data_to_buffer_[var];
        int buf_size = Downcast<IntImm>(buf->shape[0])->value;
        int orig_size = static_cast<int>(counts.size());
        if (buf_size > orig_size && orig_size > 0 &&
            buf_size % orig_size == 0) {
          // Replicate pattern to match expanded size.
          Array<PrimExpr> new_counts;
          for (int v = 0; v < buf_size; v += orig_size) {
            for (const auto &c : counts) {
              new_counts.push_back(c);
            }
          }
          updated_init.Set(var, new_counts);
          changed = true;
        } else {
          updated_init.Set(var, counts);
        }
      }

      if (changed) {
        bn->annotations.Set("barrier_init", updated_init);
        pipeline = rewrap_outer_attrs(
            BlockRealize(br->iter_values, br->predicate, block, br->span),
            outer_attrs);
      }
    }

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
        auto [outer_attrs, inner_stmt] = unwrap_outer_attrs(pipeline);
        BlockRealize pipeline_realize = Downcast<BlockRealize>(inner_stmt);
        Block pipeline_block = pipeline_realize->block;
        {
          BlockNode *block_node = pipeline_block.CopyOnWrite();
          block_node->body = apply_wrappers(block_node->body);
        }
        pipeline = rewrap_outer_attrs(
            BlockRealize(pipeline_realize->iter_values,
                         pipeline_realize->predicate, pipeline_block,
                         pipeline_realize->span),
            outer_attrs);
      } else {
        pipeline = apply_wrappers(pipeline);
      }
    }

    pipeline = AsyncCommitWaitAttrLowerer::Lower(pipeline);

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

    bool outer_flag = subtree_modified_;
    subtree_modified_ = false;
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    bool children_modified = subtree_modified_;
    // Propagate to parent: if this subtree was modified, parent should know.
    subtree_modified_ = outer_flag || children_modified;

    // Update alloc_buffers with any pending buffer remaps from pipeline
    // rewriting. This handles buffers allocated in this block but
    // multi-versioned during pipeline rewriting of inner loops.
    bool allocs_changed = false;
    bool layout_changed = false;
    Array<Buffer> new_alloc_buffers;
    std::vector<std::pair<Buffer, Buffer>> remapped_allocs;
    for (const auto &buffer : block->alloc_buffers) {
      if (auto remapped = pending_buffer_remap_.Get(buffer)) {
        new_alloc_buffers.push_back(remapped.value());
        remapped_allocs.emplace_back(buffer, remapped.value());
        pending_buffer_remap_.erase(buffer);
        allocs_changed = true;
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }

    if (!remapped_allocs.empty()) {
      auto ann = block->annotations;
      if (UpdateExpandedLayoutMapForRemappedAllocs(remapped_allocs, &ann)) {
        block.CopyOnWrite()->annotations = std::move(ann);
        layout_changed = true;
      }
    }

    // Replicate barrier_init counts for any expanded barrier buffers.
    if (allocs_changed && block->annotations.count("barrier_init")) {
      Map<Var, Array<PrimExpr>> init_map = Downcast<Map<Var, Array<PrimExpr>>>(
          block->annotations.Get("barrier_init").value());
      Map<Var, Array<PrimExpr>> new_init;
      bool init_changed = false;
      for (const auto &[var, counts] : init_map) {
        // Find the buffer for this var — it may have been remapped.
        Buffer buf;
        for (const auto &ab : new_alloc_buffers) {
          if (ab->data.same_as(var)) {
            buf = ab;
            break;
          }
        }
        if (buf.defined()) {
          int buf_size = Downcast<IntImm>(buf->shape[0])->value;
          int orig_size = static_cast<int>(counts.size());
          if (buf_size > orig_size && orig_size > 0 &&
              buf_size % orig_size == 0) {
            Array<PrimExpr> new_counts;
            for (int v = 0; v < buf_size; v += orig_size) {
              for (const auto &c : counts)
                new_counts.push_back(c);
            }
            new_init.Set(var, new_counts);
            init_changed = true;
            continue;
          }
        }
        new_init.Set(var, counts);
      }
      if (init_changed) {
        BlockNode *bn = block.CopyOnWrite();
        bn->annotations.Set("barrier_init", new_init);
        bn->alloc_buffers = new_alloc_buffers;
        allocs_changed = false; // already applied
      }
    }

    bool modified = children_modified || allocs_changed || layout_changed;
    if (modified) {
      // Recalculate reads/writes only when the block was actually
      // modified by pipeline rewriting.  Unconditional recalculation
      // can embed references to block-local buffers (e.g. local.var)
      // into the block's own read/write annotations, which misleads
      // downstream LCA analysis and causes those buffers to be
      // promoted to kernel parameters.
      //
      // After recalculation:
      // 1. Drop BufferRegions whose buffer is allocated in this block.
      // 2. Widen to full-region any BufferRegion whose index
      //    expressions reference a data var of any buffer allocated
      //    in this block or any nested block. This prevents
      //    downstream LCA analysis from seeing those vars at the
      //    outer scope and promoting them to kernel parameters.
      std::unordered_set<const BufferNode *> local_bufs;
      std::unordered_set<const VarNode *> local_data_vars;
      for (const auto &buf : block->alloc_buffers) {
        local_bufs.insert(buf.get());
        local_data_vars.insert(buf->data.get());
      }
      // Also collect data vars from all nested blocks.
      PostOrderVisit(block->body, [&](const ObjectRef &obj) {
        if (auto *inner = obj.as<BlockNode>()) {
          for (const auto &buf : inner->alloc_buffers) {
            local_data_vars.insert(buf->data.get());
          }
        }
      });
      auto region_uses_local_var = [&](const BufferRegion &br) -> bool {
        for (const auto &range : br->region) {
          bool found = false;
          PostOrderVisit(range->min, [&](const ObjectRef &obj) {
            if (found)
              return;
            if (auto *load = obj.as<BufferLoadNode>()) {
              if (local_data_vars.count(load->buffer->data.get())) {
                found = true;
              }
            }
            if (auto *var = obj.as<VarNode>()) {
              if (local_data_vars.count(var)) {
                found = true;
              }
            }
          });
          if (found)
            return true;
          PostOrderVisit(range->extent, [&](const ObjectRef &obj) {
            if (found)
              return;
            if (auto *load = obj.as<BufferLoadNode>()) {
              if (local_data_vars.count(load->buffer->data.get())) {
                found = true;
              }
            }
            if (auto *var = obj.as<VarNode>()) {
              if (local_data_vars.count(var)) {
                found = true;
              }
            }
          });
          if (found)
            return true;
        }
        return false;
      };
      Array<Array<BufferRegion>> access =
          GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
      auto sanitize = [&](const Array<BufferRegion> &regions) {
        Array<BufferRegion> out;
        for (const auto &br : regions) {
          if (local_bufs.count(br->buffer.get())) {
            continue; // drop block-local buffer
          }
          if (region_uses_local_var(br)) {
            out.push_back(BufferRegion::FullRegion(br->buffer));
          } else {
            out.push_back(br);
          }
        }
        return out;
      };
      BlockNode *n = block.CopyOnWrite();
      n->reads = sanitize(access[0]);
      n->writes = sanitize(access[1]);
      n->alloc_buffers = std::move(new_alloc_buffers);
    }

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
  Optional<Target> target_;
  // Buffers from outer blocks that have been used in a pipeline loop.
  // Used to detect if the same buffer is used in multiple pipeline loops.
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffers_used_in_pipeline_;
  Optional<String> global_symbol_;
  // Track whether any pipeline was actually injected in the current
  // subtree.  Used to avoid unnecessary reads/writes recalculation
  // on blocks whose descendants were not modified.
  bool subtree_modified_ = false;
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
