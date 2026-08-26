/*!
 * \file ws_analysis.cc
 * \brief Statement analysis shared by the warp-specialization passes.
 */

#include "./ws_analysis.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include "cuda/op/builtin.h"
#include "cuda/op/copy.h"
#include "cuda/target_utils.h"
#include "layout/cute_layout.h"
#include "op/gemm.h"
#include "op/gemm_sp.h"
#include "op/operator.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace ffi;

namespace {

/// Detect if a statement is a SIMT global-to-shared memory copy.
/// Matches any statement that writes to shared memory and reads from global
/// memory, without reading shared or local buffers (which would indicate
/// consumer-side compute).  This is intentionally broader than "pure direct
/// copy" so that T.Parallel with complex indexing / if_then_else (later
/// lowered to cp.async) is also captured.
class SimtProducerDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    SimtProducerDetector d;
    d(stmt);
    return d.writes_shared_ && d.reads_global_ && !d.reads_shared_local_;
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsSharedBuffer(op->buffer)) {
      writes_shared_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsGlobalBuffer(op->buffer)) {
      reads_global_ = true;
    }
    if (IsSharedBuffer(op->buffer) || IsLocalBuffer(op->buffer, true)) {
      reads_shared_local_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool writes_shared_{false};
  bool reads_global_{false};
  bool reads_shared_local_{false};
};

class EvaluateCallInSimpleWrapperExtractor
    : public StmtFunctor<Optional<Call>(const Stmt &)> {
public:
  Optional<Call> VisitStmt_(const EvaluateNode *op) final {
    return op->value.as<Call>();
  }

  Optional<Call> VisitStmt_(const IfThenElseNode *op) final {
    if (op->else_case.defined()) {
      return Optional<Call>();
    }
    return VisitStmt(op->then_case);
  }

  Optional<Call> VisitStmt_(const AttrStmtNode *op) final {
    return VisitStmt(op->body);
  }

  Optional<Call> VisitStmt_(const SBlockNode *op) final {
    return VisitStmt(op->body);
  }

  Optional<Call> VisitStmt_(const SBlockRealizeNode *op) final {
    return VisitStmt(op->block->body);
  }

  Optional<Call> VisitStmtDefault_(const Object *) final {
    return Optional<Call>();
  }
};

class BufferRemapper : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt stmt, const BufferRemap &buffer_remap) {
    if (buffer_remap.empty()) {
      return stmt;
    }
    BufferRemapper remapper(buffer_remap);
    return remapper.VisitStmt(stmt);
  }

private:
  explicit BufferRemapper(const BufferRemap &buffer_remap)
      : buffer_remap_(buffer_remap) {
    for (const auto &[old_buf, new_buf] : buffer_remap_) {
      var_remap_.emplace(old_buf->data, new_buf->data);
    }
  }

  Buffer RemapBuffer(const Buffer &buffer) const {
    auto it = buffer_remap_.find(buffer);
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    return buffer;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto it = var_remap_.find(ffi::GetRef<Var>(op));
    if (it != var_remap_.end()) {
      return it->second;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    Buffer new_buffer = RemapBuffer(load->buffer);
    if (!new_buffer.same_as(load->buffer)) {
      return BufferLoad(new_buffer, load->indices, load->predicate, load->span);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    Buffer new_buffer = RemapBuffer(store->buffer);
    if (!new_buffer.same_as(store->buffer)) {
      return BufferStore(new_buffer, store->value, store->indices,
                         store->predicate, store->span);
    }
    return store;
  }

  const BufferRemap &buffer_remap_;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_remap_;
};

class ManualWSDetector : public StmtExprVisitor {
public:
  static bool HasManualWS(const Stmt &stmt) {
    ManualWSDetector d;
    d(stmt);
    return d.found_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    // Detect both the T.ws() language-level attr ("warp_specialize") and
    // the compiler-level attr (kWarpSpecializationScope).
    if (op->attr_key == "warp_specialize" ||
        op->attr_key == attr::kWarpSpecializationScope) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool found_{false};
};

} // namespace

Optional<Call> GetEvaluateCallInSimpleWrapper(const Stmt &stmt) {
  EvaluateCallInSimpleWrapperExtractor extractor;
  return extractor(stmt);
}

static bool ContainsPtxCpAsync(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(builtin::ptx_cp_async()) ||
          call->op.same_as(tl::ptx_cp_async())) {
        found = true;
      }
    }
  });
  return found;
}

static bool IsPtxCommitGroup(const Stmt &stmt) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  return call.defined() &&
         call.value()->op.same_as(builtin::ptx_commit_group());
}

static bool IsPtxWaitGroup(const Stmt &stmt) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  return call.defined() && call.value()->op.same_as(builtin::ptx_wait_group());
}

bool IsBarrierOrTmaControlCall(const CallNode *call) {
  return call->op.same_as(mbarrier_wait_parity()) ||
         call->op.same_as(mbarrier_expect_tx()) ||
         call->op.same_as(builtin::ptx_arrive_barrier()) ||
         call->op.same_as(tl::ptx_arrive_cluster_barrier()) ||
         call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
         call->op.same_as(builtin::ptx_cp_async_barrier()) ||
         call->op.same_as(tl::ptx_cp_async_barrier_noinc()) ||
         call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col()) ||
         call->op.same_as(tma_store()) ||
         call->op.same_as(tma_store_arrive()) ||
         call->op.same_as(tma_store_wait()) ||
         call->op.same_as(builtin::tvm_storage_sync()) ||
         call->op.same_as(tl::sync_grid()) ||
         call->op.same_as(tl::syncthreads_count()) ||
         call->op.same_as(tl::syncthreads_and()) ||
         call->op.same_as(tl::syncthreads_or());
}

std::optional<GemmInfo> GetGemmInfo(const TileOperator &op) {
  if (const auto *gemm = op.as<GemmNode>())
    return GemmInfo{gemm->cRegion_->buffer, gemm->wgWait_};
  if (const auto *gemm = op.as<GemmSPNode>())
    return GemmInfo{gemm->cRegion_->buffer, gemm->wg_wait};
  return std::nullopt;
}

TileStmtKind ClassifyCopy(const CopyNode *copy, Target target) {
  if (copy == nullptr) {
    return TileStmtKind::kConsumer;
  }
  cuda::CopyInstSelection result =
      cuda::ClassifyWarpSpecializedCopy(*copy, target);
  if (!result.supported) {
    return TileStmtKind::kConsumer;
  }
  if (cuda::CopyInstIsTMAStore(result.inst)) {
    return TileStmtKind::kTmaStore;
  }
  if (cuda::CopyInstIsTMALoad(result.inst)) {
    return TileStmtKind::kTmaProducer;
  }
  if (cuda::CopyInstIsCPAsync(result.inst)) {
    return TileStmtKind::kCpAsyncProducer;
  }
  return TileStmtKind::kConsumer;
}

TileStmtKind ClassifyStmt(const Stmt &stmt, Target target) {
  // Tile-op Calls: classify directly via CopyNode checks.
  if (auto *eval = stmt.as<EvaluateNode>()) {
    if (auto *call = eval->value.as<CallNode>()) {
      auto tile_op = ParseOperator(GetRef<Call>(call));
      if (tile_op.defined()) {
        if (auto *copy = tile_op.as<CopyNode>()) {
          return ClassifyCopy(copy, target);
        }
        // Im2Col lowers to tma_load_im2col on Hopper — treat as TMA
        // producer so it goes to the producer warp group.
        if (tile_op.as<Im2ColOpNode>()) {
          if (TargetIsHopper(target)) {
            return TileStmtKind::kTmaProducer;
          }
        }
        if (auto gemm = GetGemmInfo(tile_op)) {
          if (IsTmemBuffer(gemm->accumulator)) {
            return TileStmtKind::kTcgen05Mma;
          }
        }
        return TileStmtKind::kConsumer; // non-copy tile-op
      }
    }
  }
  // Explicit cp.async producer-side statements are already low-level builtins.
  if (ContainsPtxCpAsync(stmt) || IsPtxCommitGroup(stmt) ||
      IsPtxWaitGroup(stmt)) {
    return TileStmtKind::kCpAsyncRaw;
  }
  // Non-tile-op: check for SIMT global-to-shared copy.
  if (SimtProducerDetector::Detect(stmt)) {
    return TileStmtKind::kSimtProducer;
  }
  return TileStmtKind::kConsumer;
}

bool IsProducer(TileStmtKind kind) {
  return kind == TileStmtKind::kTmaProducer ||
         kind == TileStmtKind::kCpAsyncProducer ||
         kind == TileStmtKind::kCpAsyncRaw ||
         kind == TileStmtKind::kSimtProducer;
}

bool HasManualWarpSpecialization(const Stmt &stmt) {
  return ManualWSDetector::HasManualWS(stmt);
}

Stmt RemapBuffers(Stmt stmt, const BufferRemap &remap) {
  return BufferRemapper::Rewrite(std::move(stmt), remap);
}

bool IsTmaCompatibleLayout(const Layout &layout, const Buffer &buffer) {
  Optional<cute::ComposedLayout> composed =
      cute::ComposedLayoutFromTileLang(layout);
  if (!composed.defined())
    return false;
  // Recast to byte space (the swizzle atom is defined on byte addresses).
  cute::ComposedLayout composed_bytes =
      composed.value().Recast(buffer->dtype.bits(), /*new_bits=*/8);
  return composed_bytes->swizzle->IsTMACompatible();
}

void CollectAnnotatedLayouts(const SBlock &block, BufferLayoutMap &layouts) {
  auto anno = block->annotations.Get("layout_map");
  if (!anno.has_value())
    return;
  auto gmap = anno->as<Map<ObjectRef, ObjectRef>>();
  if (!gmap.has_value())
    return;
  for (const auto &[key, val] : gmap.value()) {
    Layout layout;
    if (auto l = val.as<Layout>(); l.has_value())
      layout = l.value();
    if (auto buf = key.as<Buffer>(); buf.has_value()) {
      layouts[buf.value()->data] = {buf.value(), layout};
    } else if (auto var = key.as<Var>(); var.has_value()) {
      for (const auto &buf : block->alloc_buffers) {
        if (buf->data.same_as(var.value())) {
          layouts[buf->data] = {buf, layout};
          break;
        }
      }
    }
  }
}

} // namespace tl
} // namespace tvm
