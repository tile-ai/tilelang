#include "helpers.h"

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <utility>

#include "../common/bind_utils.h"
#include "backend/common/target_utils.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/region.h"
#include "op/utils.h"
#include "support/utils.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
namespace software_pipeline {

using namespace tirx;
using namespace ffi;
using tirx::GetSBlockReadWriteRegion;

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
      if (attr->attr_key == s_tir::attr::async_scope ||
          attr->attr_key == s_tir::attr::async_commit_queue_scope ||
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

} // namespace software_pipeline
} // namespace tl
} // namespace tvm
