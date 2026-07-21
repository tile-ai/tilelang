/*!
 * \file inject_pipeline.cc
 * \brief Transform annotated loops into pipelined one that parallelize
 * producers and consumers.
 */
#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "span_utils.h"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/common/target_utils.h"
#include "common/bind_utils.h"
#include "common/mbarrier.h"
#include "common/pipeline_utils.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/region.h"
#include "op/utils.h"
#include "support/utils.h"
#include "tir/schedule/utils.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;
using tirx::GetSBlockReadWriteRegion;
namespace software_pipeline {

using namespace tirx;
using namespace ffi;
using tirx::GetSBlockReadWriteRegion;

using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using BufferMap =
    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using BufferShapeMap =
    std::unordered_map<Buffer, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
using BufferCommitGroupMap =
    std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual>;
using BlockDependencyGraph =
    std::unordered_map<SBlock, Array<SBlock>, ObjectPtrHash, ObjectPtrEqual>;

/*! Structure that represents the provided annotation per block or loop. */
struct PipelineAnnotation {
  int stage;
  int order;
  bool async{false};
  int async_group_id{-1};
};

using PipelineInfo = std::unordered_map<SBlock, PipelineAnnotation,
                                        ObjectPtrHash, ObjectPtrEqual>;

struct BufferAccessInfo {
  int def = -1; // the defining stage of the buffer
  int use = -1; // the last using stage of the buffer
};

struct PipelineRewriteResult {
  Stmt pipeline;
  Map<Buffer, Buffer> buffer_remap;
};

namespace {

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return !is_zero(GetRef<IntImm>(int_val));
    }
  }
  return false;
}

bool GetIsTmaCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "is_tma_copy");
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  return GetBoolAnnotation(op, "force_cp_async");
}

bool CheckTargetIndependentAsyncCopyPreconditions(const CopyNode &op) {
  if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
    return false;
  }
  if (op.src->dtype != op.dst->dtype) {
    return false;
  }
  return true;
}

bool CheckPipelineManagedCPAsyncCopy(const CopyNode &op,
                                     Optional<Target> target) {
  if (GetIsTmaCopy(op) || GetIsAsyncCopy(op) ||
      !CheckTargetIndependentAsyncCopyPreconditions(op)) {
    return false;
  }
  return !target.defined() || TargetHasAsyncCopy(target.value());
}

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

class BufferUsageCollector : public StmtExprVisitor {
public:
  BufferUsageCollector(const Map<Var, Buffer> &buffer_data_to_buffer,
                       const BufferSet &allocated_buffers)
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
    if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
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

  void VisitStmt_(const SBlockNode *op) final {
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
  const BufferSet &allocated_buffers_;
  BufferSet used_buffers_;
};

class TileOpAccessCollector : public StmtExprVisitor {
public:
  Array<BufferRegion> GetReads() const { return reads_; }

  Array<BufferRegion> GetWrites() const { return writes_; }

private:
  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
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
    // Keep SIMT copy lowering under the outer pipeline-managed commit/wait
    // semantics as well.
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
    return CheckPipelineManagedCPAsyncCopy(*copy, target_);
  }

  Optional<Target> target_;
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
                                 tile_op.as<Im2ColOpNode>() != nullptr ||
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
    if (op->attr_key == s_tir::attr::async_commit_queue_scope) {
      Stmt body = VisitStmt(op->body);
      Stmt commit =
          Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
      if (is_no_op(body)) {
        return commit;
      }
      return SeqStmt({body, commit});
    }
    if (op->attr_key == s_tir::attr::async_wait_queue_scope) {
      auto wait_attrs = GetAsyncWaitAttributes(op);
      Stmt body = op->body;
      if (const auto *inner = op->body.as<AttrStmtNode>()) {
        if (inner->attr_key == s_tir::attr::async_wait_inflight_count) {
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
    if (op->attr_key == s_tir::attr::async_wait_inflight_count) {
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

} // namespace

bool IsReplayableScalarBindBlock(const SBlock &block,
                                 const BufferSet &pipeline_write_buffers) {
  return tl::IsReplayableScalarBind(block->body, block->reads,
                                    pipeline_write_buffers);
}

BufferSet CollectPipelineWriteBuffers(const Array<SBlock> &blocks) {
  BufferSet write_buffers;
  for (const SBlock &block : blocks) {
    for (const BufferRegion &write : block->writes) {
      write_buffers.insert(write->buffer);
    }
  }
  return write_buffers;
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
  VarSet visited;
  bool changed = false;
  for (const auto &[old_buffer, new_buffer] : remapped_allocs) {
    if (!visited.insert(old_buffer->data).second ||
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

Array<Buffer>
CollectUsedPipelineBuffers(const Stmt &stmt,
                           const Map<Var, Buffer> &buffer_data_to_buffer,
                           const BufferSet &allocated_buffers) {
  BufferUsageCollector collector(buffer_data_to_buffer, allocated_buffers);
  return collector.Collect(stmt);
}

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
SBlock MakeBlock(const Stmt &body,
                 const Map<Var, Buffer> &buffer_data_to_buffer) {
  SBlock block;
  if (const SBlockRealizeNode *block_realize = body.as<SBlockRealizeNode>()) {
    if (is_one(block_realize->predicate)) {
      block = block_realize->block;
    }
  }
  if (!block.defined()) {
    block = SBlock(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                   /*name_hint=*/"", /*body*/ body);
  }
  Array<Array<BufferRegion>> access =
      GetSBlockReadWriteRegion(block, buffer_data_to_buffer);
  TileOpAccessCollector collector;
  collector(block->body);
  Array<BufferRegion> tile_reads = collector.GetReads();
  Array<BufferRegion> tile_writes = collector.GetWrites();
  SBlockNode *n = block.CopyOnWrite();
  n->reads = access[0];
  n->reads.insert(n->reads.end(), tile_reads.begin(), tile_reads.end());
  n->writes = access[1];
  n->writes.insert(n->writes.end(), tile_writes.begin(), tile_writes.end());
  return block;
}

bool ContainsPipelineAsyncControlAttrs(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &obj) {
    if (found) {
      return;
    }
    if (const auto *attr = obj.as<AttrStmtNode>()) {
      if (attr->attr_key == s_tir::attr::async_commit_queue_scope ||
          attr->attr_key == s_tir::attr::async_wait_queue_scope ||
          attr->attr_key == s_tir::attr::async_wait_inflight_count) {
        found = true;
        return;
      }
    }
  });
  return found;
}

Stmt AnnotateSimtProducer(const Stmt &stmt, Optional<Target> target) {
  return SimtProducerAnnotator::Annotate(stmt, std::move(target));
}

Stmt AnnotateTileOpMbarPhase(const Stmt &stmt, PrimExpr phase_expr) {
  return TileOpMbarPhaseAnnotator::Annotate(stmt, std::move(phase_expr));
}

Stmt LowerAsyncCommitWaitAttrs(const Stmt &stmt) {
  return AsyncCommitWaitAttrLowerer::Lower(stmt);
}

/*!
 * \brief Build the dependency graph among a array of blocks.
 * \param[in] blocks The array of blocks.
 * \param[out] dep_src2dst Optional, a map to store dependency edges from the
 * source to the destination. \param[out] dep_dst2src Optional, a map to store
 * dependency edges from the destination to the source.
 */
void BuildDependencyGraph(const Array<SBlock> &blocks,
                          BlockDependencyGraph *dep_src2dst,
                          BlockDependencyGraph *dep_dst2src) {
  std::unordered_map<Var, Array<SBlock>, ObjectPtrHash, ObjectPtrEqual>
      buffer_writers;

  for (const SBlock &block : blocks) {
    for (const BufferRegion &read : block->reads) {
      auto it = buffer_writers.find(read->buffer->data);
      if (it != buffer_writers.end()) {
        for (const SBlock &writer : it->second) {
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
    static const Op &im2col_op = Op::Get("tl.tileop.im2col");
    static const Op &deprecated_c2d_im2col_op = Op::Get("tl.tileop.c2d_im2col");
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
    // Annotate im2col with pipeline barrier so its Lower() uses it
    // instead of allocating a separate internal barrier.
    if (call->op.same_as(im2col_op) ||
        call->op.same_as(deprecated_c2d_im2col_op)) {
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
// ExpandPipelineBarriers - multi-version all barrier buffers for pipelining
// ---------------------------------------------------------------------------

/// Collect all shared.barrier Buffer objects referenced in a statement.
class BarrierBufferCollector : public StmtExprVisitor {
public:
  static std::vector<Buffer> Collect(const Array<SBlock> &blocks) {
    BarrierBufferCollector c;
    for (const auto &block : blocks) {
      c(block->body);
    }
    return {c.barriers_.begin(), c.barriers_.end()};
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (op->buffer.scope() == "shared.barrier" ||
        op->buffer.scope() == "shared.cluster_barrier") {
      if (!seen_.count(op->buffer)) {
        seen_.insert(op->buffer);
        barriers_.push_back(op->buffer);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->buffer.scope() == "shared.barrier" ||
        op->buffer.scope() == "shared.cluster_barrier") {
      if (!seen_.count(op->buffer)) {
        seen_.insert(op->buffer);
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
          if (!seen_.count(load->buffer)) {
            seen_.insert(load->buffer);
            barriers_.push_back(load->buffer);
          }
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  BufferSet seen_;
  std::vector<Buffer> barriers_;
};

/// Rewrite barrier references: expand indices and rewrite parity.
class BarrierIndexRewriter : public StmtExprMutator {
public:
  BarrierIndexRewriter(const BufferMap &old_to_new,
                       const BufferShapeMap &old_shapes, PrimExpr stage_expr,
                       PrimExpr parity_cycle, Var loop_var, PrimExpr loop_min)
      : old_to_new_(old_to_new), old_shapes_(old_shapes),
        stage_expr_(std::move(stage_expr)),
        parity_cycle_(std::move(parity_cycle)), loop_var_(std::move(loop_var)),
        loop_min_(std::move(loop_min)) {}

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = old_to_new_.find(load->buffer);
    if (it != old_to_new_.end()) {
      auto *n = load.CopyOnWrite();
      PrimExpr old_size = old_shapes_.at(load->buffer);
      n->buffer = it->second;
      n->indices.Set(0, stage_expr_ * old_size + n->indices[0]);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = old_to_new_.find(store->buffer);
    if (it != old_to_new_.end()) {
      auto *n = store.CopyOnWrite();
      PrimExpr old_size = old_shapes_.at(store->buffer);
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
        auto it = old_to_new_.find(load->buffer);
        if (it != old_to_new_.end()) {
          PrimExpr old_size = old_shapes_.at(load->buffer);
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
        bool is_expanded = false;
        for (const auto &kv : old_to_new_) {
          if (load->buffer.same_as(kv.second)) {
            is_expanded = true;
            break;
          }
        }
        if (is_expanded) {
          // Compute initial-phase offset from the user's original parity.
          arith::Analyzer analyzer;
          PrimExpr user_parity = call->args[1];
          PrimExpr user_parity_at_min = analyzer.Simplify(
              tirx::Substitute(user_parity, {{loop_var_, loop_min_}}));
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
  const BufferMap &old_to_new_;
  const BufferShapeMap &old_shapes_;
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
/// Returns a map of old-to-new barrier buffers for outer block alloc_buffers
/// update.
Map<Buffer, Buffer> ExpandPipelineBarriers(
    Array<SBlock> &original_order, PipelineInfo &pipeline_info,
    Map<Var, Buffer> &buffer_data_to_buffer, BufferSet &allocated_buffers,
    Array<Buffer> &block_local_allocs, Array<Buffer> &pipeline_allocs,
    Var loop_var, PrimExpr loop_min, int num_stages) {
  if (num_stages <= 1)
    return {};

  // Only expand barriers that have explicit ptx_arrive_barrier calls in the
  // loop body.  This distinguishes pipeline synchronization barriers (where
  // arrive/wait are user-managed and need per-stage slots) from barriers
  // whose arrival is managed internally by tile-ops (e.g., tcgen05 MMA
  // arrive barriers) - those should NOT be pipeline-expanded.
  // ISP-created pipeline_mbar is handled specially: it's always in
  // block_local_allocs and was just created, so include it too.
  BufferSet local_barrier_set;
  for (const Buffer &buf : block_local_allocs) {
    if (buf.scope() == "shared.barrier" ||
        buf.scope() == "shared.cluster_barrier")
      local_barrier_set.insert(buf);
  }

  // Find barriers that have explicit ptx_arrive_barrier calls.
  class ArriveBarrierDetector : public StmtExprVisitor {
  public:
    BufferSet arrived_;
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(builtin::ptx_arrive_barrier()) && !op->args.empty()) {
        if (auto load = op->args[0].as<BufferLoadNode>()) {
          arrived_.insert(load->buffer);
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
      BarrierBufferCollector::Collect(original_order);
  std::vector<Buffer> barriers;
  for (const Buffer &buf : all_referenced) {
    // Include if: (a) it's an ISP-created local barrier, OR
    //             (b) it has explicit ptx_arrive_barrier calls.
    if (local_barrier_set.count(buf) || arrive_det.arrived_.count(buf)) {
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
  BufferMap old_to_new;
  BufferShapeMap old_shapes;
  for (const Buffer &buf : barriers) {
    old_shapes[buf] = buf->shape[0];
    ObjectPtr<BufferNode> new_node = make_object<BufferNode>(*(buf.get()));
    new_node->shape = {PrimExpr(num_stages) * buf->shape[0]};
    Buffer new_buf(new_node);
    old_to_new[buf] = new_buf;

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
    SBlock old_block = original_order[i];
    Stmt new_body = rewriter(old_block->body);
    if (!new_body.same_as(old_block->body)) {
      // Also rewrite alloc_buffers in the block (barriers may be allocated
      // here).
      Array<Buffer> new_allocs;
      for (const Buffer &ab : old_block->alloc_buffers) {
        auto it = old_to_new.find(ab);
        new_allocs.push_back(it != old_to_new.end() ? it->second : ab);
      }
      SBlock new_block(old_block->iter_vars, old_block->reads,
                       old_block->writes, old_block->name_hint, new_body,
                       old_block->init, new_allocs, old_block->match_buffers,
                       old_block->annotations);
      PipelineAnnotation anno = pipeline_info.at(old_block);
      pipeline_info.erase(old_block);
      pipeline_info.emplace(new_block, anno);
      original_order.Set(i, new_block);
    }
  }

  // Return the old-to-new mapping for outer block alloc_buffers update.
  Map<Buffer, Buffer> result;
  for (const auto &[old_buf, new_buf] : old_to_new) {
    result.Set(old_buf, new_buf);
  }
  return result;
}

/*!
 * \brief Rewrite TMA-eligible copy blocks in the pipeline body for
 *        pipeline-level barrier management.
 *
 * For each TMA copy: convert tl.tileop.copy to tl.tileop.tma_copy with a
 * per-stage barrier slot and emit_arrive=1 so LowerTileOp emits arrive inside
 * the thread-0 guard.
 *
 * For the first consumer stage block: prepend mbarrier_wait_parity with
 * stage-indexed barrier reference and parity expression.
 *
 * \param original_order  In/out: blocks in original pipeline order.
 * \param pipeline_info   In/out: block to PipelineAnnotation mapping.
 * \param tma_copies      Per-statement TMA flag array from PipelinePlanning.
 * \param buffer_data_to_buffer  In/out: buffer var to Buffer mapping.
 * \param allocated_buffers      In/out: set of allocated buffers.
 * \param block_local_allocs     In/out: buffers allocated in the pipeline
 * block.
 * \return The newly created barrier buffer (undefined if no TMA copies).
 */
Buffer RewritePipelineTmaBarriers(
    Array<SBlock> &original_order, PipelineInfo &pipeline_info,
    const Array<Integer> &tma_copies, Map<Var, Buffer> &buffer_data_to_buffer,
    BufferSet &allocated_buffers, Array<Buffer> &block_local_allocs,
    Var loop_var, PrimExpr loop_min, int num_stages) {
  if (!std::any_of(tma_copies.begin(), tma_copies.end(),
                   [](const Integer &tc) { return !is_zero(tc); })) {
    return Buffer();
  }

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
    if (!is_zero(tma_copies[i]))
      last_tma_idx = static_cast<int>(i);
  }

  // Phase 1: Rewrite TMA copy blocks - all share barrier slot 0.
  // ExpandPipelineBarriers (called later) will rewrite indices to be
  // stage-dependent.  Only the last TMA copy emits arrive.
  for (size_t i = 0; i < original_order.size(); i++) {
    if (is_zero(tma_copies[i]))
      continue;

    bool is_last = (static_cast<int>(i) == last_tma_idx);
    SBlock old_block = original_order[i];
    CopyToTmaCopyRewriter rewriter(barrier_buf,
                                   /*barrier_id=*/IntImm(DataType::Int(32), 0),
                                   /*emit_arrive=*/is_last);
    Stmt new_body = rewriter(old_block->body);

    SBlock new_block(old_block->iter_vars, old_block->reads, old_block->writes,
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
  bool waits_inserted = false;
  for (size_t i = 0; i < original_order.size(); i++) {
    if (waits_inserted)
      break;
    SBlock old_block = original_order[i];
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

    SBlock new_block(old_block->iter_vars, old_block->reads, old_block->writes,
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
    Stmt prologue = EmitImpl(pipeline_loop_->min,
                             pipeline_loop_->min + max_stage_, true, true);
    Stmt body =
        EmitImpl(pipeline_loop_->min + max_stage_,
                 pipeline_loop_->min + pipeline_loop_->extent, false, false);
    Stmt epilogue = EmitImpl(
        pipeline_loop_->min + pipeline_loop_->extent,
        pipeline_loop_->min + pipeline_loop_->extent + max_stage_, true, true);

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

    // Step 3: Make a new block only when the rewritten pipeline owns local
    // allocations.  When all buffers are already allocated by an outer block
    // (the common TileLang kernel-root case), an extra synthetic scope would
    // only carry a very large inferred read/write region and make the IR hard
    // to inspect.
    if (local_allocs_.empty()) {
      return stmt;
    }

    // Only include buffers that are locally allocated in the pipeline block.
    // Buffers from outer blocks will be handled separately.
    Array<Buffer> alloc_buffers;
    for (const auto &alloc : local_allocs_) {
      alloc_buffers.push_back(buffer_remap_.Get(alloc).value_or(alloc));
      buffer_data_to_buffer_.erase(alloc->data);
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
    BufferCommitGroupMap buffer_to_commit_group;
    int commit_group_count{0};
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
          int candidate_wait_n = std::max(0, retain * per_sync_groups);
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
    std::vector<int> stmt_to_commit_group(new_stmts.size(), -1);
    std::map<int, std::vector<int>> commit_group_last_stmt;
    for (const auto &[stage_id, state] : *async_states_local) {
      auto &last_stmt = commit_group_last_stmt[stage_id];
      last_stmt.assign(state.commit_groups.size(), -1);
      for (size_t group_id = 0; group_id < state.commit_groups.size();
           ++group_id) {
        for (size_t stmt_idx : state.commit_groups[group_id]) {
          ICHECK_LT(stmt_idx, new_stmts.size());
          ICHECK_EQ(stmt_to_commit_group[stmt_idx], -1);
          stmt_to_commit_group[stmt_idx] = static_cast<int>(group_id);
          last_stmt[group_id] =
              std::max(last_stmt[group_id], static_cast<int>(stmt_idx));
        }
      }
    }

    std::map<int, int> last_committed_group;
    auto record_pending_wait = [&](AsyncStateLocal *state, int commit_group_id,
                                   int insert_before, PrimExpr wait_count) {
      auto &pending_wait = state->pending_waits[commit_group_id];
      if (!pending_wait.valid()) {
        pending_wait = {insert_before, wait_count};
      } else if (analyzer_.CanProve(wait_count < pending_wait.wait_count)) {
        pending_wait = {pending_wait.insert_before, wait_count};
      }
    };

    for (size_t i = 0; i < new_stmts.size(); ++i) {
      if (new_stmts[i].is_async) {
        auto &local_state = (*async_states_local)[new_stmts[i].stage];
        for (const BufferRegion &write_region : new_stmts[i].writes) {
          local_state.seen.insert(write_region->buffer);
        }
        int commit_group_id = stmt_to_commit_group[i];
        if (commit_group_id >= 0) {
          const auto &last_stmt = commit_group_last_stmt.at(new_stmts[i].stage);
          ICHECK_LT(commit_group_id, static_cast<int>(last_stmt.size()));
          if (last_stmt[commit_group_id] == static_cast<int>(i)) {
            last_committed_group[new_stmts[i].stage] = commit_group_id;
          }
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

      if (num_commit_group == 0) {
        ICHECK(!dep_local_state.producer_head);
        const auto &global_state = async_states_[producer_stage_idx];
        PrimExpr tail_start =
            analyzer_.Simplify(pipeline_loop_->min + pipeline_loop_->extent -
                               PrimExpr(max_stage_));
        bool is_tail_consumer =
            ana_normalized->CanProve(new_stmts[i].access_index >= tail_start);
        if (is_tail_consumer && global_state.commit_group_count > 0) {
          int latest_group_id = global_state.commit_group_count - 1;
          PrimExpr latest_producer_head = analyzer_.Simplify(
              pipeline_loop_->min + pipeline_loop_->extent - PrimExpr(1));
          std::vector<bool> need_wait_count(global_state.commit_group_count,
                                            true);
          bool handled = false;
          for (const BufferRegion &read_region : new_stmts[i].reads) {
            if (!global_state.writes(read_region->buffer)) {
              continue;
            }
            auto it =
                global_state.buffer_to_commit_group.find(read_region->buffer);
            if (it == global_state.buffer_to_commit_group.end()) {
              handled = false;
              break;
            }
            int commit_group_id = it->second;
            ICHECK_GE(commit_group_id, 0);
            ICHECK_LT(commit_group_id, global_state.commit_group_count);
            if (!need_wait_count[commit_group_id]) {
              continue;
            }
            PrimExpr wait_count = analyzer_.Simplify(
                (latest_producer_head - new_stmts[i].access_index) *
                    global_state.commit_group_count +
                (latest_group_id - commit_group_id));
            if (!ana_normalized->CanProve(wait_count >= 0)) {
              wait_count = PrimExpr(0);
            }
            record_pending_wait(&dep_local_state, commit_group_id,
                                static_cast<int>(i), wait_count);
            need_wait_count[commit_group_id] = false;
            handled = true;
          }
          if (handled) {
            continue;
          }
        }

        PrimExpr wait_count = PrimExpr(0);
        Optional<PrimExpr> producer_head =
            async_states_[producer_stage_idx].producer_head;
        if (producer_head &&
            ana_normalized->CanProve(producer_head.value() >= 0)) {
          wait_count = analyzer_.Simplify(producer_head.value() -
                                          new_stmts[i].access_index);
        }
        record_pending_wait(&dep_local_state, -1, static_cast<int>(i),
                            wait_count);
        continue;
      }

      ICHECK(dep_local_state.producer_head);
      int latest_group_id = -1;
      Optional<PrimExpr> latest_producer_head;
      if (auto it = last_committed_group.find(producer_stage_idx);
          it != last_committed_group.end()) {
        latest_group_id = it->second;
        latest_producer_head = dep_local_state.producer_head.value();
      } else {
        latest_group_id = num_commit_group - 1;
        latest_producer_head = dep_local_state.producer_head.value() - 1;
      }

      std::vector<bool> need_wait_count(num_commit_group, true);
      for (const BufferRegion &read_region : new_stmts[i].reads) {
        if (!async_states_[producer_stage_idx].writes(read_region->buffer)) {
          continue;
        }
        auto commit_group_id = buffer_to_commit_group.at(read_region->buffer);
        ICHECK_GE(commit_group_id, 0);
        ICHECK_LT(commit_group_id, num_commit_group);
        if (!need_wait_count[commit_group_id]) {
          continue;
        }

        PrimExpr wait_count = PrimExpr(0);
        if (latest_producer_head &&
            ana_normalized->CanProve(latest_producer_head.value() >= 0)) {
          wait_count = analyzer_.Simplify(
              (latest_producer_head.value() - new_stmts[i].access_index) *
                  num_commit_group +
              (latest_group_id - commit_group_id));
          if (!ana_normalized->CanProve(wait_count >= 0)) {
            wait_count = PrimExpr(0);
          }
        }

        record_pending_wait(&dep_local_state, commit_group_id,
                            static_cast<int>(i), wait_count);
        need_wait_count[commit_group_id] = false;
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
            expanded.push_back(unit_stmt);
          }
          Stmt result = expanded.size() == 1 ? expanded[0] : SeqStmt(expanded);
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
          auto &global_state = async_states_[stage];
          global_state.dst_buffers.insert(write_region->buffer);
          global_state.buffer_to_commit_group[write_region->buffer] =
              commit_group_id;
          global_state.commit_group_count =
              std::max(global_state.commit_group_count,
                       static_cast<int>(local_state.commit_groups.size()));
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
                            const Array<SBlock> &original_order) {
    std::unordered_set<int> used_orders;
    for (const SBlock &block : original_order) {
      const auto &stmt_info = pipeline_info.at(block);
      int order = stmt_info.order;
      ICHECK(!used_orders.count(order))
          << "ValueError: Two statements in the software pipeline cannot have "
             "the same order"
          << SpanHintSuffix({block->body->span, block->span});
      used_orders.insert(order);
    }

    std::unordered_map<SBlock, Array<SBlock>, ObjectPtrHash, ObjectPtrEqual>
        dep_src2dst;
    BuildDependencyGraph(original_order, &dep_src2dst, nullptr);

    for (const auto &pair : dep_src2dst) {
      const SBlock &src = pair.first;
      const auto &src_info = pipeline_info.at(src);
      const Array<SBlock> &dsts = pair.second;
      for (const SBlock &dst : dsts) {
        const auto &dst_info = pipeline_info.at(dst);
        ICHECK_LE(src_info.stage, dst_info.stage)
            << "ValueError: statement " << dst << " in stage " << dst_info.stage
            << " cannot depends on statement " << src << " in a later stage "
            << src_info.stage
            << SpanHintSuffix({dst->body->span, src->body->span});
        if (src_info.stage == dst_info.stage) {
          ICHECK_LT(src_info.order, dst_info.order)
              << "ValueError: two statements with buffer "
                 "access dependency in the same stage of the "
                 "software pipeline cannot be reordered"
              << SpanHintSuffix({dst->body->span, src->body->span});
        }
      }
    }
  }

  void ValidateScheduledBindDependencies(const PipelineInfo &pipeline_info,
                                         const Array<SBlock> &scheduled_order) {
    std::unordered_map<Var, SBlock, ObjectPtrHash, ObjectPtrEqual>
        bind_producers;
    for (const SBlock &block : scheduled_order) {
      if (const auto *bind = block->body.as<BindNode>()) {
        bind_producers.emplace(bind->var, block);
      }
    }
    if (bind_producers.empty()) {
      return;
    }

    for (const SBlock &consumer : scheduled_order) {
      Array<Var> undefined_vars = UndefinedVars(consumer->body, Array<Var>{});
      for (const Var &var : undefined_vars) {
        auto it = bind_producers.find(var);
        if (it == bind_producers.end() || it->second.same_as(consumer)) {
          continue;
        }

        const PipelineAnnotation &producer_info = pipeline_info.at(it->second);
        const PipelineAnnotation &consumer_info = pipeline_info.at(consumer);
        ICHECK_EQ(producer_info.stage, consumer_info.stage)
            << "ValueError: scheduled scalar Bind '" << var
            << "' is used from a different pipeline stage. Scalar Bind "
               "statements that cannot be replayed must be scheduled in the "
               "same stage as their consumers."
            << SpanHintSuffix(consumer->body->span);
        ICHECK_LT(producer_info.order, consumer_info.order)
            << "ValueError: scheduled scalar Bind '" << var
            << "' must be ordered before every consumer in the same pipeline "
               "stage."
            << SpanHintSuffix(consumer->body->span);
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

  struct PipelineScheduleUnit {
    SBlock block;
    Array<Buffer> nested_local_allocs;
  };

  struct PipelineSchedule {
    Array<SBlock> original_order;
    Array<Buffer> nested_local_allocs;
  };

  PipelineScheduleUnit MakePipelineScheduleUnit(const Stmt &stmt) {
    PipelineScheduleUnit unit;
    if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
      if (is_one(realize->predicate) &&
          realize->block->body->IsInstance<SeqStmtNode>()) {
        const SBlock &nested_block = realize->block;
        ICHECK(nested_block->match_buffers.empty())
            << "match_buffer should have been lowered before "
               "InjectSoftwarePipeline";
        for (const Buffer &buffer : nested_block->alloc_buffers) {
          buffer_data_to_buffer_.Set(buffer->data, buffer);
          allocated_buffers_.insert(buffer);
          unit.nested_local_allocs.push_back(buffer);
        }
      }
    }
    unit.block = MakeBlock(stmt, buffer_data_to_buffer_);
    return unit;
  }

  PipelineSchedule BuildPipelineSchedule(const Array<Stmt> &stmts) {
    PipelineSchedule schedule;
    for (const Stmt &stmt : stmts) {
      PipelineScheduleUnit unit = MakePipelineScheduleUnit(stmt);
      schedule.original_order.push_back(unit.block);
      schedule.nested_local_allocs.insert(schedule.nested_local_allocs.end(),
                                          unit.nested_local_allocs.begin(),
                                          unit.nested_local_allocs.end());
    }
    return schedule;
  }

  Array<Stmt> StripPipelineDeclarationStmts(const Array<Stmt> &pipeline_body,
                                            Array<Buffer> *block_local_allocs,
                                            Array<Buffer> *flat_local_allocs) {
    ICHECK(block_local_allocs != nullptr);
    ICHECK(flat_local_allocs != nullptr);
    Array<Stmt> stage_stmts;
    bool filtered = false;
    for (const Stmt &child : pipeline_body) {
      if (IsPipelineDeclarationStmt(child)) {
        if (const auto *alloc = child.as<AllocBufferNode>()) {
          const Buffer &buffer = alloc->buffer;
          buffer_data_to_buffer_.Set(buffer->data, buffer);
          allocated_buffers_.insert(buffer);
          block_local_allocs->push_back(buffer);
          flat_local_allocs->push_back(buffer);
        } else {
          const auto *decl = child.as<DeclBufferNode>();
          ICHECK(decl != nullptr);
          const Buffer &buffer = decl->buffer;
          buffer_data_to_buffer_.Set(buffer->data, buffer);
        }
        filtered = true;
        continue;
      }
      stage_stmts.push_back(child);
    }
    if (!filtered) {
      return pipeline_body;
    }
    ICHECK(!stage_stmts.empty())
        << "ValueError: The body of the software pipeline has no stages "
           "after removing buffer declarations";
    return stage_stmts;
  }

  Map<String, Any>
  StripPipelineAnnotations(const Map<String, Any> &annotations) const {
    Map<String, Any> preserved_annotations;
    for (const auto &kv : annotations) {
      const String &key = kv.first;
      if (key != s_tir::attr::software_pipeline_stage &&
          key != s_tir::attr::software_pipeline_order &&
          key != s_tir::attr::software_pipeline_async_stages &&
          key != kPipelineAsyncProducers &&
          key != kPipelineAsyncProducerGroups && key != kPipelineTmaCopies &&
          key != kPipelineReplayableScalarBinds && key != "num_stages" &&
          key != "tl_pipelined_num_stages") {
        preserved_annotations.Set(key, kv.second);
      }
    }
    return preserved_annotations;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    struct ScopedAllocation {
      Buffer buffer;
      bool existed;
    };

    Array<Stmt> seq;
    bool changed = false;
    std::vector<std::pair<Var, Optional<Buffer>>> old_bindings;
    std::vector<ScopedAllocation> old_allocated;
    std::vector<std::pair<size_t, size_t>> flat_alloc_indices;

    auto register_buffer = [&](const Buffer &buffer,
                               bool is_allocation) -> std::optional<size_t> {
      old_bindings.emplace_back(buffer->data,
                                buffer_data_to_buffer_.Get(buffer->data));
      buffer_data_to_buffer_.Set(buffer->data, buffer);
      if (is_allocation) {
        old_allocated.push_back({buffer, allocated_buffers_.count(buffer) > 0});
        allocated_buffers_.insert(buffer);
        return old_allocated.size() - 1;
      }
      return std::nullopt;
    };

    auto apply_pending_flat_alloc_remaps = [&]() {
      for (auto &[stmt_index, alloc_state_index] : flat_alloc_indices) {
        const Buffer &old_buffer = old_allocated[alloc_state_index].buffer;
        if (auto remapped = pending_buffer_remap_.Get(old_buffer)) {
          const auto *alloc = seq[stmt_index].as<AllocBufferNode>();
          ICHECK(alloc != nullptr);
          Buffer new_buffer = remapped.value();
          seq.Set(stmt_index,
                  AllocBuffer(new_buffer, alloc->annotations, alloc->span));
          buffer_data_to_buffer_.Set(old_buffer->data, new_buffer);
          if (!old_allocated[alloc_state_index].existed) {
            allocated_buffers_.erase(old_buffer);
            allocated_buffers_.insert(new_buffer);
          }
          pending_layout_remapped_allocs_.emplace_back(old_buffer, new_buffer);
          old_allocated[alloc_state_index].buffer = new_buffer;
          pending_buffer_remap_.erase(old_buffer);
          changed = true;
        }
      }
    };

    for (const Stmt &child : op->seq) {
      Stmt new_child = VisitStmt(child);
      changed = changed || !new_child.same_as(child);
      seq.push_back(new_child);
      apply_pending_flat_alloc_remaps();

      if (const auto *alloc = new_child.as<AllocBufferNode>()) {
        std::optional<size_t> alloc_state_index =
            register_buffer(alloc->buffer, true);
        ICHECK(alloc_state_index.has_value());
        flat_alloc_indices.emplace_back(seq.size() - 1,
                                        alloc_state_index.value());
      } else if (const auto *decl = new_child.as<DeclBufferNode>()) {
        register_buffer(decl->buffer, false);
      }
    }
    apply_pending_flat_alloc_remaps();

    for (auto it = old_allocated.rbegin(); it != old_allocated.rend(); ++it) {
      if (!it->existed) {
        allocated_buffers_.erase(it->buffer);
      }
    }
    for (auto it = old_bindings.rbegin(); it != old_bindings.rend(); ++it) {
      if (it->second.defined()) {
        buffer_data_to_buffer_.Set(it->first, it->second.value());
      } else {
        buffer_data_to_buffer_.erase(it->first);
      }
    }

    if (!changed) {
      return GetRef<Stmt>(op);
    }
    return SeqStmt(seq, op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!HasPipelineAnnotation(op)) {
      return for_node;
    }
    // Step 2: Find the body and buffer allocations of the pipeline.
    Stmt pipeline_body_root = for_node->body;
    Array<Buffer> pipeline_allocs;
    Array<Buffer> block_local_allocs; // flat allocations inside pipeline body
    Array<Buffer> flat_local_allocs;

    Array<Stmt> pipeline_body_stmts = NormalizePipelineBody(pipeline_body_root);

    // PipelinePlanning emits stage/order annotations only for executable
    // pipeline statements. Flat TIRX keeps loop-local AllocBuffer/DeclBuffer as
    // standalone statements in the loop body, so strip them from the stage
    // stream before blockizing and consuming annotations. The declarations are
    // still registered as local allocations so RewritePipeline can
    // multi-version and reattach them.
    pipeline_body_stmts = StripPipelineDeclarationStmts(
        pipeline_body_stmts, &block_local_allocs, &flat_local_allocs);

    PipelineInfo pipeline_info;
    PipelineSchedule schedule = BuildPipelineSchedule(pipeline_body_stmts);
    Array<SBlock> original_order = schedule.original_order;

    // Collect all buffers that are actually used in the pipeline loop body.
    // This includes buffers allocated in outer blocks (like logits_smem) that
    // are used inside the pipeline loop.
    pipeline_allocs =
        CollectUsedPipelineBuffers(MakePipelineBody(pipeline_body_stmts),
                                   buffer_data_to_buffer_, allocated_buffers_);

    Optional<Array<Integer>> replayable_bind_mask;
    if (auto replayable_bind_anno =
            op->annotations.Get(kPipelineReplayableScalarBinds)) {
      auto mask = Downcast<Array<Integer>>(replayable_bind_anno.value());
      if (mask.size() == original_order.size()) {
        bool valid_mask = true;
        for (size_t i = 0; i < original_order.size(); ++i) {
          if (!is_zero(mask[i]) &&
              original_order[i]->body.as<BindNode>() == nullptr) {
            valid_mask = false;
            break;
          }
        }
        if (valid_mask) {
          replayable_bind_mask = std::move(mask);
        }
      }
    }
    BufferSet pipeline_write_buffers =
        CollectPipelineWriteBuffers(original_order);
    Array<SBlock> scalar_binding_blocks;
    Array<SBlock> scheduled_order;
    std::vector<char> is_replayable_bind;
    is_replayable_bind.reserve(original_order.size());
    for (size_t i = 0; i < original_order.size(); ++i) {
      const SBlock &block = original_order[i];
      const bool semantically_replayable =
          IsReplayableScalarBindBlock(block, pipeline_write_buffers);
      bool replayable = semantically_replayable;
      if (replayable_bind_mask.defined()) {
        replayable = !is_zero(replayable_bind_mask.value()[i]);
        if (replayable && !semantically_replayable) {
          const auto *bind = block->body.as<BindNode>();
          ICHECK(bind != nullptr);
          LOG(FATAL) << "PrimFunc " << global_symbol_ << " marks scalar Bind '"
                     << bind->var << "' as replayable via "
                     << kPipelineReplayableScalarBinds
                     << ", but the Bind has an unsupported type, has side "
                        "effects, or reads a buffer written by the pipeline "
                        "and cannot be replayed safely";
        }
      }
      is_replayable_bind.push_back(replayable ? 1 : 0);
      if (replayable) {
        scalar_binding_blocks.push_back(block);
      } else {
        scheduled_order.push_back(block);
      }
    }
    ICHECK(!scheduled_order.empty())
        << "ValueError: The body of the software pipeline has no schedulable "
           "statements after removing replayable scalar Bind statements";

    auto pipeline_stages = Downcast<Array<Integer>>(
        op->annotations.at(s_tir::attr::software_pipeline_stage));
    auto pipeline_orders = Downcast<Array<Integer>>(
        op->annotations.at(s_tir::attr::software_pipeline_order));
    ICHECK_EQ(pipeline_stages.size(), pipeline_orders.size())
        << "PrimFunc " << global_symbol_
        << " has software_pipeline_stage annotation " << pipeline_stages
        << " and software_pipeline_order annotation " << pipeline_orders
        << " with different sizes";

    bool annotations_include_replayable_binds = false;
    if (pipeline_stages.size() == scheduled_order.size()) {
      annotations_include_replayable_binds = false;
    } else if (pipeline_stages.size() == original_order.size()) {
      annotations_include_replayable_binds = true;
    } else {
      LOG(FATAL) << "PrimFunc " << global_symbol_
                 << " has schedulable pipeline order "
                 << scheduled_order.Map(
                        [](const auto &block) { return block->name_hint; })
                 << " and original order "
                 << original_order.Map(
                        [](const auto &block) { return block->name_hint; })
                 << ", but pipeline annotation is " << pipeline_stages
                 << " with different size";
    }

    std::vector<size_t> scheduled_annotation_indices;
    scheduled_annotation_indices.reserve(scheduled_order.size());
    if (annotations_include_replayable_binds) {
      size_t scheduled_index = 0;
      for (size_t i = 0; i < original_order.size(); ++i) {
        if (is_replayable_bind[i]) {
          continue;
        }
        ICHECK(scheduled_index < scheduled_order.size());
        ICHECK(scheduled_order[scheduled_index].same_as(original_order[i]));
        scheduled_annotation_indices.push_back(i);
        ++scheduled_index;
      }
    } else {
      for (size_t i = 0; i < scheduled_order.size(); ++i) {
        scheduled_annotation_indices.push_back(i);
      }
    }

    auto expected_annotation_size = annotations_include_replayable_binds
                                        ? original_order.size()
                                        : scheduled_order.size();

    std::unordered_set<int> pipeline_async_stages;
    if (auto async_annot =
            op->annotations.Get(s_tir::attr::software_pipeline_async_stages)) {
      for (const Integer &stage :
           Downcast<Array<Integer>>(async_annot.value())) {
        pipeline_async_stages.insert(static_cast<int>(stage.IntValue()));
      }
    }
    Optional<Array<Integer>> pipeline_async_producers;
    if (auto async_producers_anno =
            op->annotations.Get(kPipelineAsyncProducers)) {
      auto async_flags = Downcast<Array<Integer>>(async_producers_anno.value());
      ICHECK_EQ(async_flags.size(), expected_annotation_size)
          << "PrimFunc " << global_symbol_ << " has schedulable order "
          << scheduled_order.Map(
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
      ICHECK_EQ(async_group_ids.size(), expected_annotation_size)
          << "PrimFunc " << global_symbol_ << " has schedulable order "
          << scheduled_order.Map(
                 [](const auto &block) { return block->name_hint; })
          << ", but async producer group annotation is " << async_group_ids
          << " with different size";
      pipeline_async_producer_groups = async_group_ids;
    }

    for (size_t i = 0; i < scheduled_order.size(); i++) {
      size_t annotation_index = scheduled_annotation_indices[i];
      int stage =
          static_cast<int>(pipeline_stages[annotation_index].IntValue());
      bool is_async_candidate =
          pipeline_async_producers
              ? !is_zero(pipeline_async_producers.value()[annotation_index])
              : (pipeline_async_stages.count(stage) > 0);
      // Stages that already carry pipeline async control attrs keep that
      // ownership; the injector only annotates plain producer stages.
      bool is_async = is_async_candidate && !ContainsPipelineAsyncControlAttrs(
                                                scheduled_order[i]->body);
      PipelineAnnotation stage_order{
          stage,
          /*order=*/
          static_cast<int>(pipeline_orders[annotation_index].IntValue()),
          /*async=*/is_async,
          /*async_group_id=*/
          pipeline_async_producer_groups
              ? static_cast<int>(
                    pipeline_async_producer_groups.value()[annotation_index]
                        .IntValue())
              : -1};
      pipeline_info.emplace(scheduled_order[i], stage_order);
    }

    if (annotations_include_replayable_binds) {
      for (const SBlock &binding_block : scalar_binding_blocks) {
        const auto *bind = binding_block->body.as<BindNode>();
        ICHECK(bind != nullptr);
        bool seen_consumer = false;
        bool multiple_consumers = false;
        PipelineAnnotation first_consumer;
        for (const SBlock &consumer : scheduled_order) {
          Array<Var> undefined_vars =
              UndefinedVars(consumer->body, Array<Var>{});
          bool uses_binding = false;
          for (const Var &var : undefined_vars) {
            if (var.same_as(bind->var)) {
              uses_binding = true;
              break;
            }
          }
          if (!uses_binding) {
            continue;
          }
          const PipelineAnnotation &anno = pipeline_info.at(consumer);
          if (!seen_consumer) {
            first_consumer = anno;
            seen_consumer = true;
          } else if (first_consumer.stage != anno.stage ||
                     first_consumer.order != anno.order) {
            multiple_consumers = true;
            break;
          }
        }
        if (multiple_consumers) {
          LOG(WARNING)
              << "Scalar Bind '" << bind->var
              << "' is used by multiple pipeline stages; its annotation is "
                 "ignored and the bind is replayed at each use.";
        }
      }
    }

    ValidateScheduledBindDependencies(pipeline_info, scheduled_order);
    ValidatePipelineBody(pipeline_info, scheduled_order);

    if (!HasOverlappableStages(pipeline_info)) {
      for (const auto &buffer : flat_local_allocs) {
        buffer_data_to_buffer_.erase(buffer->data);
        allocated_buffers_.erase(buffer);
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
              ? static_cast<int>(pipelined_num_stages.value().IntValue())
              : max_stage + 1;
      // Clamp to at least 1 so we always allocate at least one barrier slot.
      pipeline_depth = std::max(pipeline_depth, 1);
      if (max_stage > 0) {
        if (auto tma_copies_anno = op->annotations.Get(kPipelineTmaCopies)) {
          auto raw_tma_copies =
              Downcast<Array<Integer>>(tma_copies_anno.value());
          Array<Integer> tma_copies;
          if (raw_tma_copies.size() == scheduled_order.size()) {
            tma_copies = raw_tma_copies;
          } else if (raw_tma_copies.size() == original_order.size()) {
            for (size_t annotation_index : scheduled_annotation_indices) {
              tma_copies.push_back(raw_tma_copies[annotation_index]);
            }
          }
          if (tma_copies.size() == scheduled_order.size()) {
            bool has_tma_copy =
                std::any_of(tma_copies.begin(), tma_copies.end(),
                            [](const Integer &tc) { return !is_zero(tc); });
            if (has_tma_copy) {
              pipeline_barrier_buf = RewritePipelineTmaBarriers(
                  scheduled_order, pipeline_info, tma_copies,
                  buffer_data_to_buffer_, allocated_buffers_,
                  block_local_allocs, for_node->loop_var, for_node->min,
                  pipeline_depth);
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
        barrier_depth = static_cast<int>(pipelined_ns.value().IntValue());
      } else if (op->annotations.count("num_stages")) {
        barrier_depth = static_cast<int>(
            Downcast<Integer>(op->annotations.Get("num_stages").value())
                .IntValue());
      }
      Map<Buffer, Buffer> barrier_remap = ExpandPipelineBarriers(
          scheduled_order, pipeline_info, buffer_data_to_buffer_,
          allocated_buffers_, block_local_allocs, pipeline_allocs,
          for_node->loop_var, for_node->min, barrier_depth);
      // Register expanded barriers for outer block alloc_buffers update.
      for (const auto &[old_buf, new_buf] : barrier_remap) {
        pending_buffer_remap_.Set(old_buf, new_buf);
      }
    }

    Array<Buffer> local_allocs = block_local_allocs;
    local_allocs.insert(local_allocs.end(),
                        schedule.nested_local_allocs.begin(),
                        schedule.nested_local_allocs.end());

    PipelineRewriteResult rewrite_result = RewritePipeline(
        buffer_data_to_buffer_, pipeline_allocs, local_allocs, for_node,
        pipeline_info, scalar_binding_blocks, target_);
    Stmt pipeline = rewrite_result.pipeline;
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
      auto br_opt = inner_stmt.as<SBlockRealizeNode>();
      if (br_opt == nullptr) {
        ICHECK(!pipeline_barrier_buf.defined())
            << "Pipeline barrier initialization requires a pipeline scope "
               "block";
      } else {
        SBlockRealize br = Downcast<SBlockRealize>(inner_stmt);
        SBlock block = br->block;
        SBlockNode *bn = block.CopyOnWrite();

        Map<Var, Array<PrimExpr>> barrier_init_map;
        if (bn->annotations.count("barrier_init")) {
          barrier_init_map = Downcast<Map<Var, Array<PrimExpr>>>(
              bn->annotations.Get("barrier_init").value());
        }
        bool changed = false;

        // Handle ISP-created pipeline barrier (needs new entry).
        if (pipeline_barrier_buf.defined()) {
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
              SBlockRealize(br->iter_values, br->predicate, block, br->span),
              outer_attrs);
        }
      }
    }

    // Store the buffer remapping for updating outer block alloc_buffers
    for (const auto &kv : rewrite_result.buffer_remap) {
      pending_buffer_remap_.Set(kv.first, kv.second);
    }
    pipeline = LowerAsyncCommitWaitAttrs(pipeline);

    return pipeline;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
      allocated_buffers_.insert(buffer);
    }

    bool outer_flag = subtree_modified_;
    size_t layout_remap_mark = pending_layout_remapped_allocs_.size();
    subtree_modified_ = false;
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
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
    if (pending_layout_remapped_allocs_.size() > layout_remap_mark) {
      std::vector<std::pair<Buffer, Buffer>> flat_remapped_allocs(
          pending_layout_remapped_allocs_.begin() + layout_remap_mark,
          pending_layout_remapped_allocs_.end());
      auto ann = block->annotations;
      if (UpdateExpandedLayoutMapForRemappedAllocs(flat_remapped_allocs,
                                                   &ann)) {
        block.CopyOnWrite()->annotations = std::move(ann);
        pending_layout_remapped_allocs_.erase(
            pending_layout_remapped_allocs_.begin() + layout_remap_mark,
            pending_layout_remapped_allocs_.end());
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
        SBlockNode *bn = block.CopyOnWrite();
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
      BufferSet local_bufs;
      VarSet local_data_vars;
      for (const auto &buf : block->alloc_buffers) {
        local_bufs.insert(buf);
        local_data_vars.insert(buf->data);
      }
      // Also collect data vars from all nested blocks.
      PostOrderVisit(block->body, [&](const ObjectRef &obj) {
        if (auto *inner = obj.as<SBlockNode>()) {
          for (const auto &buf : inner->alloc_buffers) {
            local_data_vars.insert(buf->data);
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
              if (local_data_vars.count(load->buffer->data)) {
                found = true;
              }
            }
            if (auto *var = obj.as<VarNode>()) {
              if (local_data_vars.count(GetRef<Var>(var))) {
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
              if (local_data_vars.count(load->buffer->data)) {
                found = true;
              }
            }
            if (auto *var = obj.as<VarNode>()) {
              if (local_data_vars.count(GetRef<Var>(var))) {
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
          GetSBlockReadWriteRegion(block, buffer_data_to_buffer_);
      auto sanitize = [&](const Array<BufferRegion> &regions) {
        Array<BufferRegion> out;
        for (const auto &br : regions) {
          if (local_bufs.count(br->buffer)) {
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
      SBlockNode *n = block.CopyOnWrite();
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
    auto it1 = op->annotations.find(s_tir::attr::software_pipeline_stage);
    auto it2 = op->annotations.find(s_tir::attr::software_pipeline_order);
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
  std::vector<std::pair<Buffer, Buffer>> pending_layout_remapped_allocs_;
  Optional<Target> target_;
  Optional<String> global_symbol_;
  // Track whether any pipeline was actually injected in the current
  // subtree.  Used to avoid unnecessary reads/writes recalculation
  // on blocks whose descendants were not modified.
  bool subtree_modified_ = false;
};

Stmt InjectPipeline(const PrimFunc &func) {
  return PipelineInjector::Inject(func);
}

} // namespace software_pipeline

/*!
 * \brief Transform annotated loops into pipelined one that parallelize
 * producers and consumers. \return The IR transform pass.
 */
tirx::transform::Pass InjectSoftwarePipeline() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *fptr = f.CopyOnWrite();
    fptr->body = software_pipeline::InjectPipeline(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSoftwarePipeline", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.InjectSoftwarePipeline",
                        InjectSoftwarePipeline);
}

} // namespace tl
} // namespace tvm
