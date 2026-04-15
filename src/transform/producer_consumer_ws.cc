/*!
 * \file producer_consumer_ws.cc
 * \brief Warp-specialized producer/consumer rewriting at the tile-op level.
 *
 * This pass runs **before** LayoutInference and LowerTileOp, operating on
 * high-level tile ops (`tl.tileop.copy`, `tl.tileop.gemm`, etc.).
 * It recognizes pipelined producer/consumer structure directly from tile-op
 * semantics and splits eligible loops into warp-specialized branches with
 * explicit barrier synchronization.
 *
 * The output IR is equivalent to a hand-written warp-specialized kernel:
 *   - TMA-annotated copies become `tl.tileop.tma_copy` with barrier refs
 *   - Barriers (`mbarrier_wait_parity`, `ptx_arrive_barrier`) are inserted
 *   - The loop body is wrapped in `if (threadIdx.x >= consumer_extent)`
 *
 * Limitations (v1):
 *   - Pure TMA pipelines only (no mixed TMA + cp.async)
 *   - No conditionally guarded loop bodies (phase counters)
 *   - Single pipelined loop per block
 *   - No pre-loop TMA prefetch / prologue optimizations
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/fill.h"
#include "../op/gemm.h"
#include "../op/operator.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "common/mbarrier.h"
#include "multi_version_buffer_rewriter.h"
#include "warp_specialized_rewriter.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

// ---------------------------------------------------------------------------
// Utility: flatten SeqStmt recursively
// ---------------------------------------------------------------------------
void FlattenSeqStmt(const Stmt &s, Array<Stmt> *out) {
  if (auto *seq = s.as<SeqStmtNode>()) {
    for (const auto &sub : seq->seq) {
      FlattenSeqStmt(sub, out);
    }
  } else {
    out->push_back(s);
  }
}

/// Annotation key marking that this function was transformed by the tiled WS
/// pass, so downstream passes can skip redundant transformations.
static constexpr const char *kTiledWSApplied = "tl_tiled_ws_applied";

// ---------------------------------------------------------------------------
// PhaseCounter: local counter for correct barrier parity in guarded loops
// ---------------------------------------------------------------------------
struct PhaseCounter {
  Buffer buf;

  static PhaseCounter Create(const std::string &name) {
    return {decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32), name,
                        "local")};
  }

  PrimExpr Load() const {
    return BufferLoad(buf, {IntImm(DataType::Int(32), 0)});
  }

  Stmt Init() const {
    return BufferStore(buf, IntImm(DataType::Int(32), 0),
                       {IntImm(DataType::Int(32), 0)});
  }

  Stmt Increment() const {
    return BufferStore(buf, Load() + 1, {IntImm(DataType::Int(32), 0)});
  }

  Stmt WrapLoopWithAlloc(Stmt loop) const {
    Stmt body = SeqStmt({Init(), std::move(loop)});
    body = DeclBuffer(buf, body);
    return Allocate(buf->data, buf->dtype, buf->shape, const_true(), body);
  }

  PrimExpr StageExpr(int num_stages) const {
    if (num_stages == 1)
      return IntImm(DataType::Int(32), 0);
    return FloorMod(Load(), num_stages);
  }

  PrimExpr ParityExpr(int num_stages) const {
    if (num_stages == 1)
      return FloorMod(Load(), 2);
    return FloorMod(FloorDiv(Load(), num_stages), 2);
  }
};

// ---------------------------------------------------------------------------
// StageExprReplacer: rewrite loop-var-based stage indexing to counter-based
// ---------------------------------------------------------------------------
class StageExprReplacer : public StmtExprMutator {
public:
  static Stmt Replace(const Stmt &stmt, Var loop_var, PrimExpr loop_min,
                      int num_stages, PrimExpr replacement) {
    StageExprReplacer r(std::move(loop_var), std::move(loop_min), num_stages,
                        std::move(replacement));
    return r.VisitStmt(stmt);
  }

private:
  StageExprReplacer(Var loop_var, PrimExpr loop_min, int num_stages,
                    PrimExpr replacement)
      : loop_var_(std::move(loop_var)), loop_min_(std::move(loop_min)),
        num_stages_(num_stages), replacement_(std::move(replacement)) {}

  PrimExpr VisitExpr_(const FloorModNode *op) final {
    if (is_const_int(op->b, num_stages_) && MatchLinearIdx(op->a)) {
      return replacement_;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  bool MatchLinearIdx(const PrimExpr &expr) const {
    if (expr.same_as(loop_var_))
      return true;
    if (const auto *sub = expr.as<SubNode>()) {
      if (sub->a.same_as(loop_var_)) {
        if (is_const_int(sub->b, 0))
          return true;
        if (sub->b.same_as(loop_min_))
          return true;
      }
    }
    return false;
  }

  Var loop_var_;
  PrimExpr loop_min_;
  int num_stages_;
  PrimExpr replacement_;
};

// ---------------------------------------------------------------------------
// Statement classification
// ---------------------------------------------------------------------------

using BufferDataToBufferMap =
    std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using BufferNodeMap = std::unordered_map<const BufferNode *, Buffer>;
using VarExprMap = std::unordered_map<const VarNode *, PrimExpr>;

struct LocalAccessSummary {
  BufferSet read_buffers;
  BufferSet write_buffers;
  VarSet read_vars;
  VarSet def_vars;

  bool HasTrackedDefs() const {
    return !write_buffers.empty() || !def_vars.empty();
  }
};

struct LocalLiveSet {
  BufferSet buffers;
  VarSet vars;

  bool NeedsAnyDef(const LocalAccessSummary &summary) const {
    for (const auto &buf : summary.write_buffers) {
      if (buffers.count(buf)) {
        return true;
      }
    }
    for (const auto &var : summary.def_vars) {
      if (vars.count(var)) {
        return true;
      }
    }
    return false;
  }

  void AddUses(const LocalAccessSummary &summary) {
    buffers.insert(summary.read_buffers.begin(), summary.read_buffers.end());
    vars.insert(summary.read_vars.begin(), summary.read_vars.end());
  }
};

static void MergeLocalAccessSummary(LocalAccessSummary *dst,
                                    const LocalAccessSummary &src) {
  dst->read_buffers.insert(src.read_buffers.begin(), src.read_buffers.end());
  dst->write_buffers.insert(src.write_buffers.begin(), src.write_buffers.end());
  dst->read_vars.insert(src.read_vars.begin(), src.read_vars.end());
  dst->def_vars.insert(src.def_vars.begin(), src.def_vars.end());
}

static Buffer CloneBranchPrivateBuffer(const Buffer &buffer,
                                       const std::string &suffix) {
  Type new_type = buffer->data->type_annotation;
  if (IsFragmentBuffer(buffer)) {
    const auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr_type);
    new_type = PointerType(ptr_type->element_type, "local");
  }
  Var new_var(buffer->data->name_hint + suffix, new_type);
  return Buffer(new_var, buffer->dtype, buffer->shape, buffer->strides,
                buffer->elem_offset, buffer->name + suffix,
                buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

class BufferRemapper : public StmtExprMutator {
public:
  static Stmt Rewrite(const Stmt &stmt, const BufferNodeMap &buffer_remap) {
    if (buffer_remap.empty()) {
      return stmt;
    }
    BufferRemapper remapper(buffer_remap);
    return remapper.VisitStmt(stmt);
  }

private:
  explicit BufferRemapper(const BufferNodeMap &buffer_remap)
      : buffer_remap_(buffer_remap) {
    for (const auto &[old_buf, new_buf] : buffer_remap_) {
      var_remap_.emplace(old_buf->data.get(), new_buf->data);
    }
  }

  Buffer RemapBuffer(const Buffer &buffer) const {
    auto it = buffer_remap_.find(buffer.get());
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    return buffer;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto it = var_remap_.find(op);
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

  const BufferNodeMap &buffer_remap_;
  VarExprMap var_remap_;
};

enum class TileStmtKind {
  kTmaProducer,     // TMA load producer (global->shared)
  kCpAsyncProducer, // Explicit cp.async / commit / wait_group producer stmt
  kSimtProducer, // Non-tile-op SIMT copy: For loop writing shared from global
  kConsumer,     // Compute (gemm, reduce, element-wise, etc.)
  kOther         // Unclassified
};

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

static const CallNode *GetEvaluateCallInSimpleWrapper(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    return eval->value.as<CallNode>();
  }
  if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
    if (!if_stmt->else_case.defined()) {
      return GetEvaluateCallInSimpleWrapper(if_stmt->then_case);
    }
    return nullptr;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return GetEvaluateCallInSimpleWrapper(attr->body);
  }
  if (const auto *let = stmt.as<LetStmtNode>()) {
    return GetEvaluateCallInSimpleWrapper(let->body);
  }
  if (const auto *block = stmt.as<BlockNode>()) {
    return GetEvaluateCallInSimpleWrapper(block->body);
  }
  if (const auto *realize = stmt.as<BlockRealizeNode>()) {
    return GetEvaluateCallInSimpleWrapper(realize->block->body);
  }
  return nullptr;
}

class BufferDataToBufferCollector : public StmtExprVisitor {
public:
  static BufferDataToBufferMap Collect(const Stmt &stmt) {
    BufferDataToBufferCollector collector;
    collector.VisitStmt(stmt);
    return collector.result_;
  }

private:
  void VisitStmt_(const BlockRealizeNode *op) final {
    CollectBuffers(op->block);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    CollectBuffers(GetRef<Block>(op));
    StmtExprVisitor::VisitStmt_(op);
  }

  void CollectBuffers(const Block &block) {
    for (const auto &buffer : block->alloc_buffers) {
      result_.emplace(buffer->data, buffer);
    }
  }

  BufferDataToBufferMap result_;
};

class LocalAccessCollector : public StmtExprVisitor {
public:
  static LocalAccessSummary Collect(const Stmt &stmt,
                                    const BufferDataToBufferMap &buffer_map) {
    LocalAccessCollector collector(buffer_map);
    collector.VisitStmt(stmt);
    return std::move(collector.summary_);
  }

private:
  explicit LocalAccessCollector(const BufferDataToBufferMap &buffer_map)
      : buffer_data_to_buffer_(buffer_map) {}

  static bool IsBranchPrivateBuffer(const Buffer &buffer) {
    return IsFragmentBuffer(buffer) || IsLocalBuffer(buffer, true);
  }

  void VisitStmt_(const LetStmtNode *op) final {
    VisitExpr(op->value);
    summary_.def_vars.insert(op->var);
    bound_vars_.insert(op->var);
    VisitStmt(op->body);
    bound_vars_.erase(op->var);
  }

  void VisitStmt_(const ForNode *op) final {
    VisitExpr(op->min);
    VisitExpr(op->extent);
    bound_vars_.insert(op->loop_var);
    VisitStmt(op->body);
    bound_vars_.erase(op->loop_var);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsBranchPrivateBuffer(op->buffer)) {
      summary_.read_buffers.insert(op->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsBranchPrivateBuffer(op->buffer)) {
      summary_.write_buffers.insert(op->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    if (bound_vars_.count(var) || buffer_data_to_buffer_.count(var)) {
      return;
    }
    summary_.read_vars.insert(var);
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(ffi::GetRef<Call>(op));
        tile_op.defined()) {
      if (const auto *copy = tile_op.as<CopyNode>()) {
        if (IsBranchPrivateBuffer(copy->src)) {
          summary_.read_buffers.insert(copy->src);
        }
        if (IsBranchPrivateBuffer(copy->dst)) {
          summary_.write_buffers.insert(copy->dst);
        }
        for (const auto &range : copy->src_range) {
          VisitExpr(range->min);
          VisitExpr(range->extent);
        }
        for (const auto &range : copy->dst_range) {
          VisitExpr(range->min);
          VisitExpr(range->extent);
        }
        return;
      }
      if (const auto *fill = tile_op.as<FillNode>()) {
        if (IsBranchPrivateBuffer(fill->dst)) {
          summary_.write_buffers.insert(fill->dst);
        }
        VisitExpr(fill->value);
        for (const auto &range : fill->region) {
          VisitExpr(range->min);
          VisitExpr(range->extent);
        }
        return;
      }
    }

    if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3);
      const auto *base_load = op->args[0].as<BufferLoadNode>();
      ICHECK(base_load);
      if (IsBranchPrivateBuffer(base_load->buffer)) {
        int rw_mask = GetConstAccessMask(op->args[2]);
        if (rw_mask & 1) {
          summary_.read_buffers.insert(base_load->buffer);
        }
        if (rw_mask & 2) {
          summary_.write_buffers.insert(base_load->buffer);
        }
      }
      for (const auto &index : base_load->indices) {
        VisitExpr(index);
      }
      VisitExpr(op->args[1]);
      return;
    }

    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5);
      const auto *var = op->args[1].as<VarNode>();
      ICHECK(var);
      auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
      if (it != buffer_data_to_buffer_.end() &&
          IsBranchPrivateBuffer(it->second)) {
        int rw_mask = GetConstAccessMask(op->args[4]);
        if (rw_mask & 1) {
          summary_.read_buffers.insert(it->second);
        }
        if (rw_mask & 2) {
          summary_.write_buffers.insert(it->second);
        }
      }
      VisitExpr(op->args[2]);
      VisitExpr(op->args[3]);
      return;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  int GetConstAccessMask(const PrimExpr &expr) const {
    if (const int64_t *imm = as_const_int(expr)) {
      return static_cast<int>(*imm);
    }
    return 3;
  }

  const BufferDataToBufferMap &buffer_data_to_buffer_;
  LocalAccessSummary summary_;
  VarSet bound_vars_;
};

enum class PreludeStmtPlacement : uint8_t {
  kKeepSharedPrelude,
  kProducerOnly,
  kConsumerOnly,
  kDuplicateToBoth,
};

static PreludeStmtPlacement
ClassifyPreludeStmt(const Stmt &stmt, const BufferDataToBufferMap &buffer_map,
                    const LocalLiveSet &producer_live_seed,
                    const LocalLiveSet &consumer_live_seed) {
  LocalAccessSummary summary = LocalAccessCollector::Collect(stmt, buffer_map);
  if (!summary.HasTrackedDefs()) {
    return PreludeStmtPlacement::kKeepSharedPrelude;
  }

  bool producer_needs = producer_live_seed.NeedsAnyDef(summary);
  bool consumer_needs = consumer_live_seed.NeedsAnyDef(summary);
  if (producer_needs && consumer_needs) {
    return PreludeStmtPlacement::kDuplicateToBoth;
  }
  if (producer_needs) {
    return PreludeStmtPlacement::kProducerOnly;
  }
  if (consumer_needs) {
    return PreludeStmtPlacement::kConsumerOnly;
  }
  return PreludeStmtPlacement::kKeepSharedPrelude;
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
  const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
  return call && call->op.same_as(builtin::ptx_commit_group());
}

static bool IsPtxWaitGroup(const Stmt &stmt) {
  const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
  return call && call->op.same_as(builtin::ptx_wait_group());
}

static bool IsBarrierOrTmaControlCall(const CallNode *call) {
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
         call->op.same_as(builtin::tvm_storage_sync());
}

static bool IsSyncGlobalToSharedCopyLikeStmt(const Stmt &stmt, Target target) {
  const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!call) {
    return false;
  }
  auto tile_op = ParseOperator(ffi::GetRef<Call>(call));
  if (!tile_op.defined()) {
    return false;
  }
  const auto *copy = tile_op.as<CopyNode>();
  if (copy == nullptr || !copy->CheckCPAsyncCopyPreconditions() ||
      copy->GetIsTmaCopy() || copy->GetIsAsyncCopy()) {
    return false;
  }

  arith::Analyzer analyzer;
  return !copy->CheckBulkLoad(target, &analyzer, /*check_last_dim=*/true);
}

static bool IsProducerMovableLoopPrefixStmt(const Stmt &stmt, Target target) {
  if (IsSyncGlobalToSharedCopyLikeStmt(stmt, target)) {
    return true;
  }

  bool has_allowed_work = false;
  bool has_disallowed = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (has_disallowed) {
      return;
    }
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_storage_sync())) {
        const auto *scope = call->args[0].as<StringImmNode>();
        if (!scope ||
            (scope->value != "shared" && scope->value != "shared.dyn")) {
          has_disallowed = true;
          return;
        }
        has_allowed_work = true;
        return;
      }
      if (IsBarrierOrTmaControlCall(call)) {
        has_disallowed = true;
        return;
      }
    }
    if (const auto *ld = node.as<BufferLoadNode>()) {
      if (IsSharedBuffer(ld->buffer) || IsLocalBuffer(ld->buffer, true)) {
        has_disallowed = true;
        return;
      }
      if (IsGlobalBuffer(ld->buffer)) {
        has_allowed_work = true;
      }
    }
    if (const auto *st = node.as<BufferStoreNode>()) {
      if (IsSharedBuffer(st->buffer)) {
        has_allowed_work = true;
        return;
      }
      has_disallowed = true;
    }
  });
  return has_allowed_work && !has_disallowed;
}

/// Classify a tile-op copy as TMA load producer, cp.async producer, or
/// consumer. Replicates the coarse checks from InstructionAnnotation inline so
/// that the tiled WS pass does not depend on a prior annotation pass.
static TileStmtKind ClassifyCopy(const CopyNode *copy, Target target) {
  // Explicit T.tma_copy() is a load-side primitive: only treat valid
  // global->shared TMA loads as producers.  TMA stores consume previously
  // produced shared data and must stay on the consumer side to preserve
  // per-iteration ordering.
  if (copy->GetIsTmaCopy()) {
    arith::Analyzer analyzer;
    if (copy->CheckBulkLoad(target, &analyzer, /*check_last_dim=*/false)) {
      return TileStmtKind::kTmaProducer;
    }
    return TileStmtKind::kConsumer; // target doesn't support TMA
  }
  // Explicit T.async_copy()
  if (copy->GetIsAsyncCopy()) {
    return TileStmtKind::kCpAsyncProducer;
  }
  // Generic T.copy(): check if TMA is possible
  {
    arith::Analyzer analyzer;
    if (!copy->GetDisableTMA() &&
        copy->CheckBulkLoad(target, &analyzer, /*check_last_dim=*/true)) {
      return TileStmtKind::kTmaProducer;
    }
  }
  return TileStmtKind::kConsumer;
}

/// Classify a single statement in the pipeline loop body.
TileStmtKind ClassifyStmt(const Stmt &stmt, Target target) {
  // Tile-op Calls: classify directly via CopyNode checks.
  if (auto *eval = stmt.as<EvaluateNode>()) {
    if (auto *call = eval->value.as<CallNode>()) {
      auto tile_op = ParseOperator(ffi::GetRef<Call>(call));
      if (tile_op.defined()) {
        if (auto *copy = tile_op.as<CopyNode>()) {
          return ClassifyCopy(copy, target);
        }
        // Conv2D im2col lowers to tma_load_im2col on Hopper — treat as TMA
        // producer so it goes to the producer warp group.
        if (tile_op.as<Conv2DIm2ColOpNode>()) {
          if (TargetIsHopper(target)) {
            return TileStmtKind::kTmaProducer;
          }
        }
        return TileStmtKind::kConsumer; // non-copy tile-op
      }
    }
  }
  // Explicit cp.async producer-side statements are already low-level builtins.
  if (ContainsPtxCpAsync(stmt) || IsPtxCommitGroup(stmt) ||
      IsPtxWaitGroup(stmt)) {
    return TileStmtKind::kCpAsyncProducer;
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
         kind == TileStmtKind::kSimtProducer;
}

// ---------------------------------------------------------------------------
// Helpers: create barrier IR nodes
// ---------------------------------------------------------------------------

static Stmt MakeParityWait(const Buffer &barrier_buf, PrimExpr barrier_id,
                           PrimExpr parity) {
  auto ref = MakeBarrierRef(barrier_buf, std::move(barrier_id));
  return Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                       {ref, std::move(parity)}));
}

static Stmt MakeArriveBarrier(const Buffer &barrier_buf, PrimExpr barrier_id) {
  auto ref = MakeBarrierRef(barrier_buf, std::move(barrier_id));
  return Evaluate(
      Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {ref}));
}

// ---------------------------------------------------------------------------
// Convert tl.tileop.copy → tl.tileop.tma_copy with barrier annotation
// ---------------------------------------------------------------------------

/// Rewrite a `tl.tileop.copy` Call into a `tl.tileop.tma_copy` Call with
/// barrier reference.  The args (src/dst regions) are preserved; only the op
/// and annotations change.
static PrimExpr RewriteCopyToTmaCopy(const Call &copy_call,
                                     const Buffer &barrier_buf,
                                     PrimExpr barrier_id) {
  static const Op &tma_copy_op = Op::Get("tl.tileop.tma_copy");
  auto new_annotations = copy_call->annotations;
  new_annotations.Set("barrier", MakeBarrierRef(barrier_buf, barrier_id));
  new_annotations.Set("is_tma_copy", IntImm(DataType::Int(32), 1));
  return Call(copy_call->dtype, tma_copy_op, copy_call->args, new_annotations,
              copy_call->span);
}

/// Annotate SIMT producer statements so the enclosing transform owns cp.async
/// synchronization.
/// - ForNodes get `kParallelAsyncWithoutAsyncCommitWait = true` so
///   InjectPTXAsyncCopy does not emit commit_group + wait_group(0).
/// - Tile-op copy calls get `kAsyncCopyNoImplicitCommitWait` so copy.cc does
///   not emit its own implicit commit/wait either.
/// This allows the WS pass to emit its own commit_group +
/// cp_async_barrier_noinc, tying cp.async completion to the forward mbarrier.
class SimtProducerAnnotator : public StmtExprMutator {
public:
  static Stmt Annotate(const Stmt &stmt, Target target) {
    SimtProducerAnnotator a(std::move(target));
    return a.VisitStmt(stmt);
  }

private:
  explicit SimtProducerAnnotator(Target target) : target_(std::move(target)) {}

  Stmt VisitStmt_(const ForNode *op) final {
    Stmt body = VisitStmt(op->body);
    auto annotations = op->annotations;
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
    return copy->CheckPipelineManagedCPAsyncCopy(target_, &analyzer_);
  }

  Target target_;
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

/// Annotate a tile-op Call (e.g., c2d_im2col) with a barrier reference.
/// The tile-op's Lower() is expected to check for the "barrier" annotation
/// and use it instead of allocating its own mbarrier.
static PrimExpr AnnotateTileOpBarrier(const Call &tile_call,
                                      const Buffer &barrier_buf,
                                      PrimExpr barrier_id) {
  auto new_annotations = tile_call->annotations;
  new_annotations.Set("barrier", MakeBarrierRef(barrier_buf, barrier_id));
  return Call(tile_call->dtype, tile_call->op, tile_call->args, new_annotations,
              tile_call->span);
}

struct BufferDataAccessInfo {
  bool read{false};
  bool write{false};

  bool HasAnyAccess() const { return read || write; }
};

struct PreludeTmaLoadPlan {
  Stmt stmt;
  const StmtNode *stmt_node{nullptr};
  int wait_pos{-1};
};

static BufferDataAccessInfo
AnalyzeBufferDataAccess(const Stmt &stmt, const Var &buffer_data,
                        const BufferDataToBufferMap &buffer_map) {
  class BufferDataAccessDetector : public StmtExprVisitor {
  public:
    BufferDataAccessDetector(const Var &buffer_data,
                             const BufferDataToBufferMap &buffer_map)
        : buffer_data_(buffer_data), buffer_map_(buffer_map) {}

    BufferDataAccessInfo Result() const { return result_; }

  private:
    void VisitExpr_(const BufferLoadNode *op) final {
      if (op->buffer->data.same_as(buffer_data_)) {
        result_.read = true;
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      if (op->buffer->data.same_as(buffer_data_)) {
        result_.write = true;
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(tl::access_ptr())) {
        ICHECK_EQ(op->args.size(), 3);
        const auto *base_load = op->args[0].as<BufferLoadNode>();
        ICHECK(base_load);
        if (base_load->buffer->data.same_as(buffer_data_)) {
          MarkAccess(op->args[2]);
        }
        for (const auto &index : base_load->indices) {
          VisitExpr(index);
        }
        VisitExpr(op->args[1]);
        return;
      }

      if (op->op.same_as(builtin::tvm_access_ptr())) {
        ICHECK_EQ(op->args.size(), 5);
        const auto *var = op->args[1].as<VarNode>();
        ICHECK(var);
        auto it = buffer_map_.find(GetRef<Var>(var));
        if (it != buffer_map_.end() && it->second->data.same_as(buffer_data_)) {
          MarkAccess(op->args[4]);
        }
        VisitExpr(op->args[2]);
        VisitExpr(op->args[3]);
        return;
      }

      StmtExprVisitor::VisitExpr_(op);
    }

    void MarkAccess(const PrimExpr &rw_expr) {
      int rw_mask = 3;
      if (const int64_t *imm = as_const_int(rw_expr)) {
        rw_mask = static_cast<int>(*imm);
      }
      if (rw_mask & 1) {
        result_.read = true;
      }
      if (rw_mask & 2) {
        result_.write = true;
      }
    }

    Var buffer_data_;
    const BufferDataToBufferMap &buffer_map_;
    BufferDataAccessInfo result_;
  };

  BufferDataAccessDetector detector(buffer_data, buffer_map);
  detector(stmt);
  return detector.Result();
}

static bool CollectPreludeStmtsToPipelineLoop(const Stmt &stmt,
                                              const ForNode *pipeline_loop,
                                              Array<Stmt> *prelude_stmts) {
  if (stmt.get() == pipeline_loop) {
    return true;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (int i = 0; i < static_cast<int>(seq->seq.size()); ++i) {
      Array<Stmt> nested_prelude;
      if (CollectPreludeStmtsToPipelineLoop(seq->seq[i], pipeline_loop,
                                            &nested_prelude)) {
        for (int j = 0; j < i; ++j) {
          prelude_stmts->push_back(seq->seq[j]);
        }
        prelude_stmts->insert(prelude_stmts->end(), nested_prelude.begin(),
                              nested_prelude.end());
        return true;
      }
    }
    return false;
  }
  if (const auto *let = stmt.as<LetStmtNode>()) {
    return CollectPreludeStmtsToPipelineLoop(let->body, pipeline_loop,
                                             prelude_stmts);
  }
  if (const auto *realize = stmt.as<BlockRealizeNode>()) {
    return CollectPreludeStmtsToPipelineLoop(realize->block->body,
                                             pipeline_loop, prelude_stmts);
  }
  if (const auto *block = stmt.as<BlockNode>()) {
    return CollectPreludeStmtsToPipelineLoop(block->body, pipeline_loop,
                                             prelude_stmts);
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return CollectPreludeStmtsToPipelineLoop(attr->body, pipeline_loop,
                                             prelude_stmts);
  }
  return false;
}

static Optional<Var> ExtractProducerWriteBufferData(const Stmt &stmt) {
  const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!call) {
    return Optional<Var>();
  }
  auto tile_op = ParseOperator(ffi::GetRef<Call>(call));
  if (!tile_op.defined()) {
    return Optional<Var>();
  }
  if (const auto *copy = tile_op.as<CopyNode>()) {
    if (IsSharedBuffer(copy->dst)) {
      return copy->dst->data;
    }
  }
  if (const auto *im2col = tile_op.as<Conv2DIm2ColOpNode>()) {
    if (IsSharedBuffer(im2col->dst_)) {
      return im2col->dst_->data;
    }
  }
  return Optional<Var>();
}

static Stmt RewritePreludeTmaProducerStmt(const Stmt &stmt,
                                          const Buffer &barrier_buf,
                                          PrimExpr barrier_id) {
  class PreludeTmaProducerRewriter : public StmtExprMutator {
  public:
    PreludeTmaProducerRewriter(Buffer barrier_buf, PrimExpr barrier_id)
        : barrier_buf_(std::move(barrier_buf)),
          barrier_id_(std::move(barrier_id)) {}

    Stmt Rewrite(const Stmt &stmt) { return VisitStmt(stmt); }

  private:
    PrimExpr VisitExpr_(const CallNode *op) final {
      Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
      if (rewritten_) {
        return call;
      }
      auto tile_op = ParseOperator(call);
      if (!tile_op.defined()) {
        return call;
      }
      PrimExpr rewritten_call;
      if (tile_op.as<CopyNode>()) {
        rewritten_call = RewriteCopyToTmaCopy(call, barrier_buf_, barrier_id_);
      } else if (tile_op.as<Conv2DIm2ColOpNode>()) {
        rewritten_call = AnnotateTileOpBarrier(call, barrier_buf_, barrier_id_);
      } else {
        return call;
      }
      Call new_call = Downcast<Call>(rewritten_call);
      auto annotations = new_call->annotations;
      annotations.Set("emit_arrive", IntImm(DataType::Int(32), 1));
      rewritten_ = true;
      return Call(new_call->dtype, new_call->op, new_call->args, annotations,
                  new_call->span);
    }

    Buffer barrier_buf_;
    PrimExpr barrier_id_;
    bool rewritten_{false};
  };

  PreludeTmaProducerRewriter rewriter(barrier_buf, std::move(barrier_id));
  return rewriter.Rewrite(stmt);
}

// ---------------------------------------------------------------------------
// Main rewriter
// ---------------------------------------------------------------------------

class ProducerConsumerWSRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "ProducerConsumerWS: target attribute is required";

    ProducerConsumerWSRewriter T;
    T.target_ = target.value();
    f.CopyOnWrite()->body = T(f->body);

    if (T.ws_transformed_) {
      f = WithAttr(std::move(f), kTiledWSApplied, IntImm(DataType::Int(32), 1));
    }
    return f;
  }

private:
  // --- Track threadIdx.x binding ---
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_iv_ = iv;
        Optional<PrimExpr> old_num_threads = num_threads_;
        num_threads_ = std::nullopt;
        AttrStmt attr = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
        if (num_threads_.defined()) {
          PrimExpr nt = num_threads_.value();
          thread_iv_.CopyOnWrite()->dom = {0, nt};
          attr.CopyOnWrite()->node = thread_iv_;
          attr.CopyOnWrite()->value = nt;
        }
        num_threads_ = old_num_threads;
        thread_iv_ = {};
        return attr;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // --- Find the block containing the pipeline loop ---
  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    if (!thread_iv_.defined())
      return StmtExprMutator::VisitStmt_(op);

    const Block &orig_block = op->block;

    // Find the pipelined loop.
    const ForNode *pipeline_loop = FindPipelineLoop(orig_block->body);
    if (!pipeline_loop)
      return StmtExprMutator::VisitStmt_(op);

    auto num_stages_anno = pipeline_loop->annotations.Get("num_stages");
    if (!num_stages_anno)
      return StmtExprMutator::VisitStmt_(op);
    int num_stages =
        static_cast<int>(Downcast<Integer>(num_stages_anno.value())->value);
    if (num_stages < 1)
      return StmtExprMutator::VisitStmt_(op);

    // Flatten the loop body.
    Array<Stmt> flat_stmts;
    Stmt loop_body = pipeline_loop->body;
    if (auto *realize = loop_body.as<BlockRealizeNode>()) {
      loop_body = realize->block->body;
    }
    // Unwrap LetStmt chain that dominates the whole loop body.
    std::vector<std::pair<Var, PrimExpr>> outer_let_bindings;
    while (const auto *let = loop_body.as<LetStmtNode>()) {
      outer_let_bindings.emplace_back(let->var, let->value);
      loop_body = let->body;
    }
    // Unwrap a single IfThenElse wrapper (no else branch) so that
    // TMA producers inside conditional loop bodies can be classified.
    // Keep LetStmt chains inside the conditional separate so they stay
    // dominated by the original guard after rebuilding WS branches.
    Optional<PrimExpr> loop_body_condition;
    std::vector<std::pair<Var, PrimExpr>> inner_let_bindings;
    if (const auto *if_stmt = loop_body.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined()) {
        // Peel LetStmt chain from inside the conditional body. These
        // bindings must remain inside the guarded region.
        Stmt inner = if_stmt->then_case;
        while (const auto *let = inner.as<LetStmtNode>()) {
          inner_let_bindings.emplace_back(let->var, let->value);
          inner = let->body;
        }
        loop_body_condition = if_stmt->condition;
        loop_body = inner;
      }
    }
    FlattenSeqStmt(loop_body, &flat_stmts);

    // Classify statements into producer (TMA/SIMT copy) and consumer.
    std::vector<TileStmtKind> kinds;
    int num_tma = 0;
    int num_simt = 0;
    for (const Stmt &s : flat_stmts) {
      auto k = ClassifyStmt(s, target_);
      kinds.push_back(k);
      if (k == TileStmtKind::kTmaProducer)
        ++num_tma;
      if (k == TileStmtKind::kSimtProducer)
        ++num_simt;
    }

    // Require at least one TMA producer.
    if (num_tma == 0)
      return StmtExprMutator::VisitStmt_(op);

    // --- Build the WS transformation ---
    return BuildWSBlock(op, orig_block, pipeline_loop, num_stages, flat_stmts,
                        kinds, outer_let_bindings, inner_let_bindings,
                        loop_body_condition);
  }

  Stmt
  BuildWSBlock(const BlockRealizeNode *orig_realize, const Block &orig_block,
               const ForNode *pipeline_loop, int num_stages,
               const Array<Stmt> &flat_stmts,
               const std::vector<TileStmtKind> &kinds,
               const std::vector<std::pair<Var, PrimExpr>> &outer_let_bindings,
               const std::vector<std::pair<Var, PrimExpr>> &inner_let_bindings,
               Optional<PrimExpr> loop_body_condition = Optional<PrimExpr>()) {
    Var loop_var = pipeline_loop->loop_var;
    PrimExpr loop_min = pipeline_loop->min;
    PrimExpr loop_extent = pipeline_loop->extent;
    PrimExpr linear_idx = loop_var - loop_min;

    PrimExpr base_stage_expr = FloorMod(linear_idx, num_stages);
    PrimExpr base_parity_expr = FloorMod(FloorDiv(linear_idx, num_stages), 2);

    // When the loop body is conditionally guarded, use PhaseCounters
    // instead of the loop variable for barrier stage/parity.  This
    // ensures parity stays correct when iterations are skipped.
    bool needs_phase_counter = loop_body_condition.defined();
    Optional<PhaseCounter> producer_phase_counter;
    Optional<PhaseCounter> consumer_phase_counter;
    PrimExpr p_stage_expr = base_stage_expr;
    PrimExpr p_parity_expr = base_parity_expr;
    PrimExpr c_stage_expr = base_stage_expr;
    PrimExpr c_parity_expr = base_parity_expr;
    if (needs_phase_counter) {
      producer_phase_counter = PhaseCounter::Create("producer_phase_cnt");
      consumer_phase_counter = PhaseCounter::Create("consumer_phase_cnt");
      p_stage_expr = producer_phase_counter.value().StageExpr(num_stages);
      p_parity_expr = producer_phase_counter.value().ParityExpr(num_stages);
      c_stage_expr = consumer_phase_counter.value().StageExpr(num_stages);
      c_parity_expr = consumer_phase_counter.value().ParityExpr(num_stages);
    }

    PrimExpr consumer_extent = thread_iv_->dom->extent;
    PrimExpr producer_extent = IntImm(DataType::Int(32), 128);
    common_prelude_rewrites_.clear();

    bool has_simt_producer = false;
    bool has_cp_async_producer = false;
    int num_producer_groups = 0;
    for (auto k : kinds) {
      if (k == TileStmtKind::kTmaProducer)
        ++num_producer_groups;
      if (k == TileStmtKind::kSimtProducer)
        has_simt_producer = true;
      if (k == TileStmtKind::kCpAsyncProducer)
        has_cp_async_producer = true;
    }

    // --- Barrier allocation ---
    // Layout: [fwd_0..fwd_{G*S-1}] [bp_0..bp_{G*S-1}]
    // where G = num_producer_groups (one per TMA copy), S = num_stages.
    // When SIMT producers are present, all producer types share the same
    // barrier group — the last forward arrive covers everything.
    int num_fwd = num_producer_groups * num_stages;
    int num_bp = num_producer_groups * num_stages;

    buffer_data_to_buffer_ =
        BufferDataToBufferCollector::Collect(orig_block->body);
    Array<Stmt> consumer_compute_stmts;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (!IsProducer(kinds[i])) {
        consumer_compute_stmts.push_back(flat_stmts[i]);
      }
    }

    Array<Stmt> prelude_stmts;
    CollectPreludeStmtsToPipelineLoop(orig_block->body, pipeline_loop,
                                      &prelude_stmts);
    std::vector<PreludeTmaLoadPlan> prelude_tma_plans;
    for (const Stmt &stmt : prelude_stmts) {
      if (ClassifyStmt(stmt, target_) != TileStmtKind::kTmaProducer) {
        continue;
      }
      Optional<Var> write_buffer_data = ExtractProducerWriteBufferData(stmt);
      if (!write_buffer_data.defined()) {
        continue;
      }
      int first_read = -1;
      for (size_t ci = 0; ci < consumer_compute_stmts.size(); ++ci) {
        BufferDataAccessInfo access = AnalyzeBufferDataAccess(
            consumer_compute_stmts[ci], write_buffer_data.value(),
            buffer_data_to_buffer_);
        if (access.read) {
          first_read = static_cast<int>(ci);
          break;
        }
      }
      if (first_read < 0) {
        continue;
      }
      prelude_tma_plans.push_back({stmt, stmt.get(), first_read});
    }

    int total_barriers = num_fwd + num_bp + prelude_tma_plans.size();
    Buffer barrier_buf =
        CreateMBarrierBuffer(injected_mbarrier_name_, total_barriers);
    // arrive_counts are computed later (after producer_extent is finalized).

    std::vector<int> wait_insert_pos(num_producer_groups, 0);
    std::vector<int> arrive_insert_pos(
        num_producer_groups, static_cast<int>(consumer_compute_stmts.size()));
    int access_group_idx = 0;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] != TileStmtKind::kTmaProducer) {
        continue;
      }
      Optional<Var> write_buffer_data =
          ExtractProducerWriteBufferData(flat_stmts[i]);
      if (write_buffer_data.defined()) {
        int first_read = -1;
        int last_access = -1;
        for (size_t ci = 0; ci < consumer_compute_stmts.size(); ++ci) {
          BufferDataAccessInfo access = AnalyzeBufferDataAccess(
              consumer_compute_stmts[ci], write_buffer_data.value(),
              buffer_data_to_buffer_);
          if (access.read && first_read < 0) {
            first_read = static_cast<int>(ci);
          }
          if (access.HasAnyAccess()) {
            last_access = static_cast<int>(ci);
          }
        }
        if (first_read >= 0) {
          wait_insert_pos[access_group_idx] = first_read;
          arrive_insert_pos[access_group_idx] = last_access + 1;
        } else if (last_access >= 0) {
          wait_insert_pos[access_group_idx] = 0;
          arrive_insert_pos[access_group_idx] = last_access + 1;
        }
      }
      ++access_group_idx;
    }

    // --- Determine if TMA barriers can be merged ---
    // When all pure-TMA producers wait at the same consumer position and
    // release at the same position, forward and back-pressure barriers can
    // be shared across all TMA copies, reducing from 2*G*S to 2*S barriers.
    bool can_merge_tma_barriers = (num_producer_groups > 1) &&
                                  !has_simt_producer && !has_cp_async_producer;
    if (can_merge_tma_barriers) {
      for (int g = 1; g < num_producer_groups; ++g) {
        if (wait_insert_pos[g] != wait_insert_pos[0] ||
            arrive_insert_pos[g] != arrive_insert_pos[0]) {
          can_merge_tma_barriers = false;
          break;
        }
      }
    }
    if (can_merge_tma_barriers) {
      // Re-compute barrier layout with a single merged group.
      num_fwd = num_stages;
      num_bp = num_stages;
      total_barriers = num_fwd + num_bp + prelude_tma_plans.size();
      barrier_buf =
          CreateMBarrierBuffer(injected_mbarrier_name_, total_barriers);
    }

    std::vector<Array<Stmt>> producer_loop_prefix_stmts(num_producer_groups);
    std::vector<bool> moved_compute_stmts(consumer_compute_stmts.size(), false);
    int compute_cursor = 0;
    for (int ti = 0; ti < num_producer_groups; ++ti) {
      int wait_pos = wait_insert_pos[ti];
      if (wait_pos <= compute_cursor) {
        compute_cursor = std::max(compute_cursor, wait_pos);
        continue;
      }
      bool all_movable = true;
      for (int ci = compute_cursor; ci < wait_pos; ++ci) {
        if (!IsProducerMovableLoopPrefixStmt(consumer_compute_stmts[ci],
                                             target_)) {
          all_movable = false;
          break;
        }
      }
      if (all_movable) {
        for (int ci = compute_cursor; ci < wait_pos; ++ci) {
          producer_loop_prefix_stmts[ti].push_back(consumer_compute_stmts[ci]);
          moved_compute_stmts[ci] = true;
        }
      }
      compute_cursor = wait_pos;
    }

    bool producer_needs_full_thread_extent = false;
    for (size_t i = 0;
         i < flat_stmts.size() && !producer_needs_full_thread_extent; ++i) {
      if (kinds[i] == TileStmtKind::kSimtProducer ||
          IsSyncGlobalToSharedCopyLikeStmt(flat_stmts[i], target_)) {
        producer_needs_full_thread_extent = true;
      }
    }
    if (!producer_needs_full_thread_extent) {
      for (const auto &prefix_stmts : producer_loop_prefix_stmts) {
        for (const auto &stmt : prefix_stmts) {
          if (IsSyncGlobalToSharedCopyLikeStmt(stmt, target_)) {
            producer_needs_full_thread_extent = true;
            break;
          }
        }
        if (producer_needs_full_thread_extent) {
          break;
        }
      }
    }
    if (producer_needs_full_thread_extent) {
      // LowerTileOp will materialize these producer-side sync copies into
      // explicit SIMT global->shared loops. Keep the producer partition at the
      // original thread extent so the lowered thread mapping stays valid.
      producer_extent = consumer_extent;
    }

    // --- Compute arrive_counts (after producer_extent is finalized) ---
    // Forward arrive_count:
    //   - Pure TMA (possibly merged): 1 (leader thread only)
    //   - Mixed TMA with SIMT/cp.async: producer_extent (all producer threads)
    PrimExpr fwd_arrive_count = (can_merge_tma_barriers ||
                                 (!has_simt_producer && !has_cp_async_producer))
                                    ? IntImm(DataType::Int(32), 1)
                                    : producer_extent;
    Array<PrimExpr> arrive_counts;
    for (int i = 0; i < num_fwd; ++i) {
      arrive_counts.push_back(fwd_arrive_count);
    }
    for (int i = 0; i < num_bp; ++i) {
      arrive_counts.push_back(consumer_extent);
    }
    for (size_t i = 0; i < prelude_tma_plans.size(); ++i) {
      arrive_counts.push_back(IntImm(DataType::Int(32), 1));
    }

    std::vector<Array<Stmt>> prelude_waits_before_consumer(
        consumer_compute_stmts.size());
    PrimExpr prelude_wait_guard =
        needs_phase_counter ? EQ(consumer_phase_counter.value().Load(),
                                 IntImm(DataType::Int(32), 0))
                            : EQ(loop_var, loop_min);
    int prelude_barrier_base = num_fwd + num_bp;
    for (size_t i = 0; i < prelude_tma_plans.size(); ++i) {
      PrimExpr barrier_id = IntImm(DataType::Int(32), prelude_barrier_base + i);
      common_prelude_rewrites_.emplace(
          prelude_tma_plans[i].stmt_node,
          RewritePreludeTmaProducerStmt(prelude_tma_plans[i].stmt, barrier_buf,
                                        barrier_id));
      int wait_pos = prelude_tma_plans[i].wait_pos;
      ICHECK_GE(wait_pos, 0);
      ICHECK_LT(wait_pos, static_cast<int>(consumer_compute_stmts.size()));
      prelude_waits_before_consumer[wait_pos].push_back(IfThenElse(
          prelude_wait_guard, MakeParityWait(barrier_buf, barrier_id,
                                             IntImm(DataType::Int(32), 0))));
    }

    // --- Build producer body ---
    // Producer structure (mixed TMA + SIMT/cp.async):
    //   bp_wait → SIMT copies (all threads, async) → TMA copies (leader) →
    //   commit + cp_async_barrier_noinc.
    // SIMT copies are placed after bp_wait but before TMA so cp.async
    // and TMA can overlap.

    // First pass: collect SIMT/cp.async producer stmts separately.
    Array<Stmt> simt_producer_stmts;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] == TileStmtKind::kSimtProducer) {
        // Annotate ForNodes with kParallelAsyncWithoutAsyncCommitWait so
        // InjectPTXAsyncCopy (called from LowerTileOp) does not insert
        // commit+wait — the WS pass will emit its own commit+barrier_noinc.
        simt_producer_stmts.push_back(
            SimtProducerAnnotator::Annotate(flat_stmts[i], target_));
      } else if (kinds[i] == TileStmtKind::kCpAsyncProducer) {
        simt_producer_stmts.push_back(flat_stmts[i]);
      }
    }

    // Second pass: build the producer body with correct ordering.
    Array<Stmt> producer_stmts;
    int tma_idx = 0;
    int last_tma_idx = num_producer_groups - 1;
    bool simt_stmts_emitted = false;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] == TileStmtKind::kTmaProducer) {
        int barrier_group = can_merge_tma_barriers ? 0 : tma_idx;
        int fwd_base = barrier_group * num_stages;
        int bp_base = num_fwd + barrier_group * num_stages;
        PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) + p_stage_expr;
        PrimExpr bp_id = IntImm(DataType::Int(32), bp_base) + p_stage_expr;

        // Back-pressure wait (only once when barriers are merged)
        if (!can_merge_tma_barriers || tma_idx == 0) {
          producer_stmts.push_back(MakeParityWait(
              barrier_buf, bp_id,
              bitwise_xor(p_parity_expr, IntImm(DataType::Int(32), 1))));
        }

        // After the first bp_wait, emit all SIMT/cp.async producers
        // followed immediately by commit_group so the hardware can start
        // the async transfers as early as possible, overlapping with TMA.
        if (!simt_stmts_emitted && !simt_producer_stmts.empty()) {
          for (const auto &s : simt_producer_stmts) {
            producer_stmts.push_back(s);
          }
          // Commit cp.async group right after issuing — the earlier the
          // commit, the more overlap with subsequent TMA loads.
          if (has_simt_producer || has_cp_async_producer) {
            producer_stmts.push_back(Evaluate(
                Call(DataType::Handle(), builtin::ptx_commit_group(), {})));
          }
          simt_stmts_emitted = true;
        }

        for (const auto &stmt : producer_loop_prefix_stmts[tma_idx]) {
          producer_stmts.push_back(stmt);
        }
        // Convert copy → tma_copy with barrier, or annotate non-copy
        // TMA tile-ops (e.g. c2d_im2col) with barrier reference.
        const auto *eval = flat_stmts[i].as<EvaluateNode>();
        ICHECK(eval);
        Call tile_call = Downcast<Call>(eval->value);
        auto tile_op = ParseOperator(tile_call);
        PrimExpr tma_call;
        // For pure TMA, tell LowerTileOp to emit arrive inside the same
        // tl_shuffle_elect block (via emit_arrive annotation), producing
        // arrive_and_expect_tx instead of separate expect_tx + arrive.
        // When merged barriers, only the last TMA copy should arrive.
        bool emit_arrive_on_this =
            !has_simt_producer && !has_cp_async_producer &&
            (!can_merge_tma_barriers || tma_idx == last_tma_idx);

        if (tile_op.defined() && tile_op.as<CopyNode>()) {
          tma_call = RewriteCopyToTmaCopy(tile_call, barrier_buf, fwd_id);
        } else {
          // Non-copy TMA producer (e.g. Conv2DIm2ColOp): annotate with
          // barrier so Lower() uses the WS barrier instead of its own.
          tma_call = AnnotateTileOpBarrier(tile_call, barrier_buf, fwd_id);
        }
        if (emit_arrive_on_this) {
          auto call = Downcast<Call>(tma_call);
          auto annos = call->annotations;
          annos.Set("emit_arrive", IntImm(DataType::Int(32), 1));
          tma_call = Call(call->dtype, call->op, call->args, annos, call->span);
        }
        producer_stmts.push_back(Evaluate(tma_call));
        ++tma_idx;
      }
      // SIMT/cp.async producers are handled above (after first bp_wait).
      // Consumer/Other statements are skipped in producer.
    }
    // Fallback: if there were no TMA producers to anchor the bp_wait,
    // emit SIMT stmts now (shouldn't happen in the mixed path).
    if (!simt_stmts_emitted && !simt_producer_stmts.empty()) {
      for (const auto &s : simt_producer_stmts) {
        producer_stmts.push_back(s);
      }
    }
    // When any producer-side work is not single-threaded pure-TMA, all
    // producer threads arrive on all forward barriers after finishing it.
    // SIMT copies (later lowered to cp.async by InjectPTXAsyncCopy) and
    // explicit cp.async groups use commit_group + cp_async_barrier_noinc
    // so the async copy completion drives the mbarrier arrival, allowing
    // TMA and cp.async to overlap.  Other groups use MakeArriveBarrier.
    if (has_simt_producer || has_cp_async_producer) {
      // Any SIMT producer will become cp.async after LowerTileOp.
      bool group_has_async_copy = has_simt_producer || has_cp_async_producer;
      for (int g = 0; g < num_producer_groups; ++g) {
        int fwd_base = g * num_stages;
        PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) + p_stage_expr;
        if (group_has_async_copy) {
          // Tie cp.async completion to the forward mbarrier.
          // commit_group was already emitted right after the cp.async
          // instructions (before TMA) to maximize overlap.
          producer_stmts.push_back(Evaluate(
              Call(DataType::Handle(), tl::ptx_cp_async_barrier_noinc(),
                   {MakeBarrierRef(barrier_buf, fwd_id)})));
        } else {
          producer_stmts.push_back(MakeArriveBarrier(barrier_buf, fwd_id));
        }
      }
    }
    // Phase counter increment at end of producer guarded iteration
    if (needs_phase_counter) {
      producer_stmts.push_back(producer_phase_counter.value().Increment());
    }

    // --- Build consumer body ---
    // When barriers are merged, iterate over a single effective group.
    int consumer_barrier_groups =
        can_merge_tma_barriers ? 1 : num_producer_groups;
    Array<Stmt> consumer_stmts;
    std::vector<bool> arrive_emitted(consumer_barrier_groups, false);
    for (size_t ci = 0; ci < consumer_compute_stmts.size(); ++ci) {
      for (const auto &stmt : prelude_waits_before_consumer[ci]) {
        consumer_stmts.push_back(stmt);
      }
      for (int g = 0; g < consumer_barrier_groups; ++g) {
        if (wait_insert_pos[g] == static_cast<int>(ci)) {
          int fwd_base = g * num_stages;
          PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) + c_stage_expr;
          consumer_stmts.push_back(
              MakeParityWait(barrier_buf, fwd_id, c_parity_expr));
        }
      }
      if (!moved_compute_stmts[ci]) {
        consumer_stmts.push_back(consumer_compute_stmts[ci]);
      }
      for (int g = 0; g < consumer_barrier_groups; ++g) {
        if (arrive_insert_pos[g] == static_cast<int>(ci + 1)) {
          int bp_base = num_fwd + g * num_stages;
          PrimExpr bp_id = IntImm(DataType::Int(32), bp_base) + c_stage_expr;
          consumer_stmts.push_back(MakeArriveBarrier(barrier_buf, bp_id));
          arrive_emitted[g] = true;
        }
      }
    }
    if (consumer_compute_stmts.empty()) {
      for (int g = 0; g < consumer_barrier_groups; ++g) {
        int fwd_base = g * num_stages;
        PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) + c_stage_expr;
        consumer_stmts.push_back(
            MakeParityWait(barrier_buf, fwd_id, c_parity_expr));
      }
    }
    for (int g = 0; g < consumer_barrier_groups; ++g) {
      if (!arrive_emitted[g] &&
          arrive_insert_pos[g] ==
              static_cast<int>(consumer_compute_stmts.size())) {
        int bp_base = num_fwd + g * num_stages;
        PrimExpr bp_id = IntImm(DataType::Int(32), bp_base) + c_stage_expr;
        consumer_stmts.push_back(MakeArriveBarrier(barrier_buf, bp_id));
      }
    }
    // Phase counter increment at end of consumer guarded iteration
    if (needs_phase_counter) {
      consumer_stmts.push_back(consumer_phase_counter.value().Increment());
    }

    // --- Wrap with let bindings and optional condition ---
    auto wrap_lets =
        [&](Stmt body,
            const std::vector<std::pair<Var, PrimExpr>> &bindings) -> Stmt {
      for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
        body = LetStmt(it->first, it->second, body);
      }
      return body;
    };

    Stmt producer_body = wrap_lets(SeqStmt(producer_stmts), inner_let_bindings);
    Stmt consumer_body = wrap_lets(SeqStmt(consumer_stmts), inner_let_bindings);

    // Wrap in original condition if the loop body was guarded.
    if (loop_body_condition.defined()) {
      producer_body = IfThenElse(loop_body_condition.value(), producer_body);
      consumer_body = IfThenElse(loop_body_condition.value(), consumer_body);
    }

    producer_body = wrap_lets(producer_body, outer_let_bindings);
    consumer_body = wrap_lets(consumer_body, outer_let_bindings);

    // Rewrite shared-buffer stage indices from loop-var-based to
    // counter-based so they stay in sync with barrier parity.
    if (needs_phase_counter) {
      producer_body = StageExprReplacer::Replace(
          producer_body, loop_var, loop_min, num_stages,
          producer_phase_counter.value().StageExpr(num_stages));
      consumer_body = StageExprReplacer::Replace(
          consumer_body, loop_var, loop_min, num_stages,
          consumer_phase_counter.value().StageExpr(num_stages));
    }
    producer_body =
        TileOpMbarPhaseAnnotator::Annotate(producer_body, p_parity_expr);
    consumer_body =
        TileOpMbarPhaseAnnotator::Annotate(consumer_body, c_parity_expr);

    // --- Build loops (strip pipeline annotations) ---
    // WS handles pipeline overlap via barriers, so strip all pipeline-
    // related annotations to prevent PipelinePlanning / InjectSoftware-
    // Pipeline from re-pipelining the already WS-transformed loops.
    Map<String, Any> loop_annos;
    for (const auto &[key, value] : pipeline_loop->annotations) {
      if (key != "num_stages" && key != "tl_pipeline_order" &&
          key != "tl_pipeline_stage" && key != "software_pipeline_order" &&
          key != "software_pipeline_stage") {
        loop_annos.Set(key, value);
      }
    }

    For producer_loop(loop_var, loop_min, loop_extent, ForKind::kSerial,
                      producer_body, Optional<IterVar>(), loop_annos);
    For consumer_loop(loop_var, loop_min, loop_extent, ForKind::kSerial,
                      consumer_body, Optional<IterVar>(), loop_annos);

    // Wrap loops with phase counter allocation when needed.
    Stmt final_producer_loop = producer_loop;
    Stmt final_consumer_loop = consumer_loop;
    if (needs_phase_counter) {
      final_producer_loop =
          producer_phase_counter.value().WrapLoopWithAlloc(producer_loop);
      final_consumer_loop =
          consumer_phase_counter.value().WrapLoopWithAlloc(consumer_loop);
    }

    // --- Rewrite threadIdx.x for producer partition ---
    // Producer: threadIdx.x - consumer_extent (maps to [0, producer_extent))
    Stmt rewritten_producer = PCThreadIdxRewriter::Rewrite(
        final_producer_loop, thread_iv_->var, thread_iv_->var - consumer_extent,
        producer_extent, false);
    // Consumer: threadIdx.x stays, but extent is consumer_extent
    Stmt rewritten_consumer = final_consumer_loop;

    producer_prelude_live_seed_ = {};
    consumer_prelude_live_seed_ = {};
    producer_prelude_live_seed_.AddUses(LocalAccessCollector::Collect(
        rewritten_producer, buffer_data_to_buffer_));
    consumer_prelude_live_seed_.AddUses(LocalAccessCollector::Collect(
        rewritten_consumer, buffer_data_to_buffer_));

    // Move pre-loop branch-private initialization next to the branch that
    // consumes it. Classification is based on downstream producer/consumer
    // uses of the values defined by each prelude statement.
    extracted_producer_init_ = {};
    extracted_consumer_init_ = {};

    Array<IntImm> ws_partition = {Downcast<IntImm>(producer_extent),
                                  Downcast<IntImm>(consumer_extent)};

    // First pass: find and extract consumer-only pre-loop statements
    // by doing a dry replacement that populates extracted_consumer_init_.
    Stmt dummy_producer = rewritten_producer;
    const Stmt &dummy_consumer = rewritten_consumer;
    Stmt dummy_ws = IfThenElse(GE(thread_iv_->var, consumer_extent),
                               dummy_producer, dummy_consumer);
    dummy_ws =
        AttrStmt(ws_partition, attr::kWarpSpecializationScope, 0, dummy_ws);
    ReplaceResult replaced = ReplacePipelineLoopInStmt(
        orig_block->body, pipeline_loop, dummy_ws, consumer_extent);

    // Producer and consumer partitions cannot safely share the same block-level
    // local/fragment buffers after tiled WS is introduced before
    // LayoutInference: a single fragment layout cannot represent both thread
    // ranges. Clone every branch-private buffer touched by the producer so
    // LayoutInference can infer an independent producer-side thread range.
    BufferNodeMap producer_buffer_remap;
    Array<Buffer> producer_private_buffers;
    {
      std::unordered_set<const BufferNode *> block_alloc_buffers;
      for (const auto &buffer : orig_block->alloc_buffers) {
        block_alloc_buffers.insert(buffer.get());
      }
      LocalAccessSummary producer_access = LocalAccessCollector::Collect(
          rewritten_producer, buffer_data_to_buffer_);
      for (const auto &stmt : extracted_producer_init_) {
        MergeLocalAccessSummary(
            &producer_access,
            LocalAccessCollector::Collect(stmt, buffer_data_to_buffer_));
      }
      auto maybe_clone = [&](const Buffer &buffer) {
        if (!buffer.defined() ||
            !(IsFragmentBuffer(buffer) || IsLocalBuffer(buffer, true)) ||
            !block_alloc_buffers.count(buffer.get()) ||
            producer_buffer_remap.count(buffer.get())) {
          return;
        }
        Buffer cloned = CloneBranchPrivateBuffer(buffer, "_producer_ws");
        producer_buffer_remap.emplace(buffer.get(), cloned);
        producer_private_buffers.push_back(cloned);
      };
      for (const auto &buffer : producer_access.read_buffers) {
        maybe_clone(buffer);
      }
      for (const auto &buffer : producer_access.write_buffers) {
        maybe_clone(buffer);
      }
    }
    if (!producer_buffer_remap.empty()) {
      rewritten_producer =
          BufferRemapper::Rewrite(rewritten_producer, producer_buffer_remap);
      Array<Stmt> remapped_producer_init;
      for (const auto &stmt : extracted_producer_init_) {
        remapped_producer_init.push_back(
            BufferRemapper::Rewrite(stmt, producer_buffer_remap));
      }
      extracted_producer_init_ = remapped_producer_init;
    }

    // If branch-local prelude init/copy was extracted, rebuild with it inside
    // the corresponding WS branch so each branch initializes its own local
    // state before entering the pipelined loop.
    if (!extracted_producer_init_.empty() ||
        !extracted_consumer_init_.empty()) {
      Stmt enriched_producer = rewritten_producer;
      if (!extracted_producer_init_.empty()) {
        Array<Stmt> producer_parts;
        for (const auto &s : extracted_producer_init_) {
          producer_parts.push_back(PCThreadIdxRewriter::Rewrite(
              s, thread_iv_->var, thread_iv_->var - consumer_extent,
              producer_extent, false));
        }
        producer_parts.push_back(rewritten_producer);
        enriched_producer = producer_parts.size() == 1
                                ? producer_parts[0]
                                : SeqStmt(producer_parts);
      }
      Array<Stmt> consumer_parts;
      for (const auto &s : extracted_consumer_init_) {
        consumer_parts.push_back(s);
      }
      consumer_parts.push_back(rewritten_consumer);
      Stmt enriched_consumer = consumer_parts.size() == 1
                                   ? consumer_parts[0]
                                   : SeqStmt(consumer_parts);
      Stmt scoped_producer = enriched_producer;
      const Stmt &scoped_consumer = enriched_consumer;
      Stmt ws_body = IfThenElse(GE(thread_iv_->var, consumer_extent),
                                scoped_producer, scoped_consumer);
      ws_body =
          AttrStmt(ws_partition, attr::kWarpSpecializationScope, 0, ws_body);
      // Second pass: replace again with the enriched WS body.
      // extracted_consumer_init_ is already empty (stmts were removed
      // from the prelude in the first pass result).
      // We need to replace in the ALREADY-modified body from pass 1.
      // But ReplacePipelineLoopInStmt finds the pipeline_loop by
      // pointer comparison, which won't match in the modified tree.
      // Instead, just substitute the dummy_ws in the replaced result.
      // Since dummy_ws appears exactly once in replaced.stmt, do a
      // simple statement replacement on the full placeholder stmt.
      class SubstWsBody : public StmtExprMutator {
      public:
        SubstWsBody(const Stmt &old_ws, const Stmt &new_ws)
            : old_(old_ws), new_(new_ws) {}
        Stmt VisitStmt(const Stmt &stmt) final {
          if (stmt.same_as(old_)) {
            return new_;
          }
          return StmtExprMutator::VisitStmt(stmt);
        }
        Stmt old_, new_;
      };
      SubstWsBody subst(dummy_ws, ws_body);
      replaced.stmt = subst(replaced.stmt);
    }
    ICHECK(replaced.found)
        << "ProducerConsumerWS: failed to replace pipeline loop";
    Stmt new_block_body = SinkGuardedConsumerPostlude::Rewrite(
        replaced.stmt, thread_iv_->var, consumer_extent);

    // --- Update block ---
    Block new_block = orig_block;
    auto *block_ptr = new_block.CopyOnWrite();
    block_ptr->body = new_block_body;
    for (const auto &buffer : producer_private_buffers) {
      block_ptr->alloc_buffers.push_back(buffer);
    }

    // Add barrier buffer to alloc_buffers.
    block_ptr->alloc_buffers.push_back(barrier_buf);

    // Add barrier_init annotation.
    Map<Var, Array<PrimExpr>> barrier_init_map;
    barrier_init_map.Set(barrier_buf->data, arrive_counts);
    auto ann = block_ptr->annotations;
    if (ann.count("barrier_init")) {
      auto existing =
          Downcast<Map<Var, Array<PrimExpr>>>(ann.Get("barrier_init").value());
      for (auto [k, v] : existing) {
        barrier_init_map.Set(k, v);
      }
    }
    ann.Set("barrier_init", barrier_init_map);
    block_ptr->annotations = std::move(ann);

    // Update thread extent at the tiled WS level so LayoutInference sees
    // the producer branch as live and can analyze explicit TMA copies.
    num_threads_ = consumer_extent + producer_extent;
    ws_transformed_ = true;

    // Rebuild BlockRealize.
    BlockRealize new_realize = GetRef<BlockRealize>(orig_realize);
    new_realize.CopyOnWrite()->block = new_block;
    return new_realize;
  }

  // --- Find the first For loop with num_stages annotation ---
  const ForNode *FindPipelineLoop(const Stmt &stmt) {
    if (auto *for_node = stmt.as<ForNode>()) {
      if (for_node->annotations.Get("num_stages")) {
        return for_node;
      }
    }
    // Walk through SeqStmt, LetStmt, etc.
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (auto *result = FindPipelineLoop(s)) {
          return result;
        }
      }
    }
    if (auto *let = stmt.as<LetStmtNode>()) {
      return FindPipelineLoop(let->body);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return FindPipelineLoop(realize->block->body);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return FindPipelineLoop(block->body);
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return FindPipelineLoop(attr->body);
    }
    return nullptr;
  }

  struct ReplaceResult {
    Stmt stmt;
    bool found{false};
  };

  class SinkGuardedConsumerPostlude : public StmtExprMutator {
  public:
    static Stmt Rewrite(const Stmt &stmt, Var thread_var,
                        PrimExpr consumer_extent) {
      SinkGuardedConsumerPostlude sinker(std::move(thread_var),
                                         std::move(consumer_extent));
      return sinker.VisitStmt(stmt);
    }

  private:
    SinkGuardedConsumerPostlude(Var thread_var, PrimExpr consumer_extent)
        : thread_var_(std::move(thread_var)),
          consumer_extent_(std::move(consumer_extent)) {}

    static bool SameExpr(const PrimExpr &lhs, const PrimExpr &rhs) {
      return ExprDeepEqual()(lhs, rhs);
    }

    bool IsWSBranchStmt(const Stmt &stmt, IfThenElse *branch) const {
      const auto *if_node = stmt.as<IfThenElseNode>();
      if (!if_node || !if_node->else_case.defined()) {
        return false;
      }
      const auto *ge = if_node->condition.as<GENode>();
      if (!ge) {
        return false;
      }
      const auto *lhs = ge->a.as<VarNode>();
      if (!lhs || lhs != thread_var_.get()) {
        return false;
      }
      if (!SameExpr(ge->b, consumer_extent_)) {
        return false;
      }
      *branch = GetRef<IfThenElse>(if_node);
      return true;
    }

    bool IsWSBranch(const Stmt &stmt, Stmt *container,
                    IfThenElse *branch) const {
      if (IsWSBranchStmt(stmt, branch)) {
        *container = stmt;
        return true;
      }
      const auto *attr_node = stmt.as<AttrStmtNode>();
      if (!attr_node || attr_node->attr_key != attr::kWarpSpecializationScope) {
        return false;
      }
      if (!IsWSBranchStmt(attr_node->body, branch)) {
        return false;
      }
      *container = stmt;
      return true;
    }

    bool IsGuardedConsumerStmt(const Stmt &stmt, Stmt *body) const {
      const auto *if_node = stmt.as<IfThenElseNode>();
      if (!if_node || if_node->else_case.defined()) {
        return false;
      }
      const auto *lt = if_node->condition.as<LTNode>();
      if (!lt) {
        return false;
      }
      const auto *lhs = lt->a.as<VarNode>();
      if (!lhs || lhs != thread_var_.get()) {
        return false;
      }
      if (!SameExpr(lt->b, consumer_extent_)) {
        return false;
      }
      *body = if_node->then_case;
      return true;
    }

    static Stmt AppendToStmt(const Stmt &stmt, const Array<Stmt> &suffix) {
      if (suffix.empty()) {
        return stmt;
      }
      Array<Stmt> seq;
      if (const auto *seq_stmt = stmt.as<SeqStmtNode>()) {
        for (const auto &s : seq_stmt->seq) {
          seq.push_back(s);
        }
      } else {
        seq.push_back(stmt);
      }
      for (const auto &s : suffix) {
        seq.push_back(s);
      }
      return seq.size() == 1 ? seq[0] : SeqStmt(seq);
    }

    Stmt UpdateWSBranchContainer(const Stmt &container,
                                 const IfThenElse &branch,
                                 const Array<Stmt> &consumer_postlude) const {
      auto *branch_ptr = const_cast<IfThenElse &>(branch).CopyOnWrite();
      ICHECK(branch_ptr->else_case.defined());
      branch_ptr->else_case =
          AppendToStmt(branch_ptr->else_case.value(), consumer_postlude);
      if (container.same_as(branch)) {
        return branch;
      }
      AttrStmt attr = Downcast<AttrStmt>(container);
      attr.CopyOnWrite()->body = branch;
      return attr;
    }

    Stmt VisitStmt_(const SeqStmtNode *op) final {
      Array<Stmt> visited;
      for (const auto &stmt : op->seq) {
        visited.push_back(VisitStmt(stmt));
      }

      Array<Stmt> rebuilt;
      for (int i = 0; i < static_cast<int>(visited.size()); ++i) {
        Stmt ws_container;
        IfThenElse ws_branch;
        if (!IsWSBranch(visited[i], &ws_container, &ws_branch)) {
          rebuilt.push_back(visited[i]);
          continue;
        }

        Array<Stmt> consumer_postlude;
        int j = i + 1;
        for (; j < static_cast<int>(visited.size()); ++j) {
          Stmt body;
          if (!IsGuardedConsumerStmt(visited[j], &body)) {
            break;
          }
          consumer_postlude.push_back(body);
        }
        if (consumer_postlude.empty()) {
          rebuilt.push_back(visited[i]);
          continue;
        }

        rebuilt.push_back(UpdateWSBranchContainer(ws_container, ws_branch,
                                                  consumer_postlude));
        i = j - 1;
      }

      return rebuilt.size() == 1 ? rebuilt[0] : SeqStmt(rebuilt);
    }

    Var thread_var_;
    PrimExpr consumer_extent_;
  };

  Stmt GuardConsumerOnly(const Stmt &stmt, PrimExpr consumer_extent) {
    return IfThenElse(LT(thread_iv_->var, consumer_extent), stmt);
  }

  ReplaceResult ReplacePipelineLoopInStmt(const Stmt &stmt,
                                          const ForNode *pipeline_loop,
                                          const Stmt &ws_body,
                                          PrimExpr consumer_extent) {
    if (stmt.get() == pipeline_loop) {
      return {ws_body, true};
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> new_seq;
      bool found = false;
      // First pass: find which child contains the pipeline loop.
      int loop_idx = -1;
      for (int i = 0; i < static_cast<int>(seq->seq.size()); ++i) {
        ReplaceResult probe = ReplacePipelineLoopInStmt(
            seq->seq[i], pipeline_loop, ws_body, consumer_extent);
        if (probe.found) {
          loop_idx = i;
          break;
        }
      }
      if (loop_idx < 0) {
        return {stmt, false};
      }
      // Classify pre-loop statements using branch-private def/use sets.
      // Shared-prelude statements stay in place; branch-private definitions
      // move next to the branch that consumes them, or are duplicated when
      // both producer and consumer need the same definition.
      for (int i = 0; i < loop_idx; ++i) {
        switch (ClassifyPreludeStmt(seq->seq[i], buffer_data_to_buffer_,
                                    producer_prelude_live_seed_,
                                    consumer_prelude_live_seed_)) {
        case PreludeStmtPlacement::kProducerOnly:
          extracted_producer_init_.push_back(seq->seq[i]);
          break;
        case PreludeStmtPlacement::kConsumerOnly:
          extracted_consumer_init_.push_back(seq->seq[i]);
          break;
        case PreludeStmtPlacement::kDuplicateToBoth:
          extracted_producer_init_.push_back(seq->seq[i]);
          extracted_consumer_init_.push_back(seq->seq[i]);
          break;
        case PreludeStmtPlacement::kKeepSharedPrelude:
          if (auto it = common_prelude_rewrites_.find(seq->seq[i].get());
              it != common_prelude_rewrites_.end()) {
            new_seq.push_back(it->second);
          } else {
            new_seq.push_back(seq->seq[i]);
          }
          break;
        }
      }
      // Replace the pipeline loop itself.
      ReplaceResult result = ReplacePipelineLoopInStmt(
          seq->seq[loop_idx], pipeline_loop, ws_body, consumer_extent);
      new_seq.push_back(result.stmt);
      // Guard post-loop siblings.
      for (int i = loop_idx + 1; i < static_cast<int>(seq->seq.size()); ++i) {
        new_seq.push_back(GuardConsumerOnly(seq->seq[i], consumer_extent));
      }
      return {new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq), true};
    }
    if (auto *let = stmt.as<LetStmtNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          let->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      return {LetStmt(let->var, let->value, result.stmt), true};
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          realize->block->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      Block block = realize->block;
      block.CopyOnWrite()->body = result.stmt;
      BlockRealize new_realize = GetRef<BlockRealize>(realize);
      new_realize.CopyOnWrite()->block = block;
      return {new_realize, true};
    }
    if (auto *block = stmt.as<BlockNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          block->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      Block new_block = GetRef<Block>(block);
      new_block.CopyOnWrite()->body = result.stmt;
      return {new_block, true};
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          attr->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      AttrStmt new_attr = GetRef<AttrStmt>(attr);
      new_attr.CopyOnWrite()->body = result.stmt;
      return {new_attr, true};
    }
    return {stmt, false};
  }

  // --- PCThreadIdxRewriter (simplified for tile-op level) ---
  class PCThreadIdxRewriter : public StmtExprMutator {
  public:
    static Stmt Rewrite(Stmt stmt, Var thread_var, PrimExpr replaced,
                        PrimExpr thread_extent, bool do_shuffle) {
      PCThreadIdxRewriter r(std::move(thread_var), std::move(replaced),
                            std::move(thread_extent));
      return r(std::move(stmt));
    }

  private:
    PCThreadIdxRewriter(Var thread_var, PrimExpr replaced,
                        PrimExpr thread_extent)
        : thread_var_(std::move(thread_var)), replaced_(std::move(replaced)),
          thread_extent_(std::move(thread_extent)) {}

    PrimExpr VisitExpr_(const VarNode *var) final {
      if (var == thread_var_.get()) {
        return replaced_;
      }
      return StmtExprMutator::VisitExpr_(var);
    }

    Var thread_var_;
    PrimExpr replaced_;
    PrimExpr thread_extent_;
  };

  // State
  Target target_;
  IterVar thread_iv_;
  Optional<PrimExpr> num_threads_; // total (consumer + producer)
  bool ws_transformed_{false};
  BufferDataToBufferMap buffer_data_to_buffer_;
  std::unordered_map<const StmtNode *, Stmt> common_prelude_rewrites_;
  LocalLiveSet producer_prelude_live_seed_;
  LocalLiveSet consumer_prelude_live_seed_;
  Array<Stmt> extracted_producer_init_;
  Array<Stmt> extracted_consumer_init_;
};

// ---------------------------------------------------------------------------
// Detect if manual WS is already present (skip if so)
// ---------------------------------------------------------------------------

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

/// Quick pre-scan: check if the function contains a pipelined loop (num_stages
/// >= 1) with at least one TMA load producer tile op and no manual layout
/// annotations (which are incompatible with early MVB expansion).
/// Check whether a layout annotation on a shared buffer is compatible with
/// TMA.  TMA supports identity (linear) layouts and the three standard
/// swizzle modes (32B / 64B / 128B).  Any other layout (e.g. padded,
/// Volta-style) cannot be used with TMA.
static bool IsTmaCompatibleLayout(const Layout &layout, const Buffer &buffer) {
  // Recognised swizzle → TMA with swizzle.
  if (DetectSwizzleMode(layout, buffer) != SwizzleMode::kNone) {
    return true;
  }
  // Identity / row-major linear → TMA without swizzle.
  if (StructuralEqual()(layout, makeLinearLayout(buffer->shape))) {
    return true;
  }
  return false;
}

class TiledWSCandidate : public StmtExprVisitor {
public:
  static bool Check(const Stmt &stmt, Target target) {
    TiledWSCandidate c;
    c.target_ = target;
    c(stmt);
    return c.has_pipeline_loop_ && c.has_tma_tile_op_;
  }

private:
  void VisitStmt_(const ForNode *op) final {
    bool old = in_pipeline_;
    if (auto anno = op->annotations.Get("num_stages")) {
      if (auto *imm = anno->as<IntImmNode>()) {
        if (imm->value >= 1) {
          has_pipeline_loop_ = true;
          in_pipeline_ = true;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
    in_pipeline_ = old;
  }

  void VisitExpr_(const CallNode *op) final {
    if (in_pipeline_ && !has_tma_tile_op_) {
      auto tile_op = ParseOperator(ffi::GetRef<Call>(op));
      if (auto *copy = tile_op.as<CopyNode>()) {
        if (ClassifyCopy(copy, target_) == TileStmtKind::kTmaProducer) {
          // If the destination buffer has a layout annotation, verify
          // that the layout is TMA-compatible (swizzle or linear).
          // Copies whose layout is incompatible with TMA cannot become
          // TMA producers.
          if (HasTmaCompatibleLayout(copy->dst)) {
            has_tma_tile_op_ = true;
          }
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    // Collect layout_map entries so we can cross-check TMA copy targets.
    if (op->annotations.count("layout_map")) {
      auto anno = op->annotations.Get("layout_map");
      if (auto gmap = anno->as<Map<ObjectRef, ObjectRef>>(); gmap.has_value()) {
        for (const auto &[key, val] : gmap.value()) {
          Layout layout;
          if (auto l = val.as<Layout>(); l.has_value())
            layout = l.value();
          if (auto buf = key.as<Buffer>(); buf.has_value()) {
            layout_map_[buf.value()->data.get()] = {buf.value(), layout};
          } else if (auto var = key.as<Var>(); var.has_value()) {
            for (const auto &buf : op->alloc_buffers) {
              if (buf->data.same_as(var.value())) {
                layout_map_[buf->data.get()] = {buf, layout};
                break;
              }
            }
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /// A copy destination is TMA-compatible if it has no layout annotation,
  /// or its annotated layout is a recognised swizzle / linear layout.
  bool HasTmaCompatibleLayout(const Buffer &dst) const {
    auto it = layout_map_.find(dst->data.get());
    if (it == layout_map_.end()) {
      return true; // no annotation → identity layout → TMA OK
    }
    const auto &[buf, layout] = it->second;
    if (!layout.defined()) {
      return false; // annotation present but layout not parseable
    }
    return IsTmaCompatibleLayout(layout, buf);
  }

  Target target_;
  bool in_pipeline_{false};
  bool has_pipeline_loop_{false};
  bool has_tma_tile_op_{false};
  // Map from buffer data Var pointer → (Buffer, Layout) for layout_map entries.
  std::unordered_map<const Object *, std::pair<Buffer, Layout>> layout_map_;
};

} // namespace

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

tvm::transform::Pass ProducerConsumerWarpSpecialized() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Skip if disabled.
    if (ctx->GetConfig(kDisableWarpSpecialized, Optional<Bool>())
            .value_or(false)) {
      return f;
    }
    // Skip if the function already has manual WS.
    if (ManualWSDetector::HasManualWS(f->body)) {
      return f;
    }
    // Skip if TMA is not available.
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetHasBulkCopy(target.value())) {
      return f;
    }
    // Only apply MVB + WS if the function is a tiled WS candidate.
    if (!TiledWSCandidate::Check(f->body, target.value())) {
      LOG(WARNING) << "[WS] skipped: no TMA copies in pipeline loop";
      return f;
    }
    LOG(WARNING) << "[WS] candidate found, applying MVB + WS";
    // Expand shared buffers for pipelining before the WS split.
    // Keep the original so we can fall back if the WS rewriter doesn't fire
    // (e.g. non-tile-op consumers in the loop body).
    PrimFunc original_f = f;
    f = ApplyMultiVersionBufferRewriter(std::move(f));
    PrimFunc result = ProducerConsumerWSRewriter::Substitute(std::move(f));
    if (!result->HasNonzeroAttr(kTiledWSApplied)) {
      LOG(WARNING) << "[WS] rewriter did not fire, falling back";
      // The TMA kernel needs warp specialization for correct pipelined
      // execution.  Since the tiled rewriter could not apply WS (e.g.
      // conditional loop body), strip pipeline annotations so that
      // PipelinePlanning / InjectSoftwarePipeline do not generate
      // broken non-WS TMA pipeline code.
      class StripPipelineAnnotation : public tir::StmtExprMutator {
      public:
        tir::Stmt VisitStmt_(const tir::ForNode *op) final {
          auto stmt = tir::StmtExprMutator::VisitStmt_(op);
          const auto *for_node = stmt.as<tir::ForNode>();
          ICHECK(for_node);
          if (for_node->annotations.count("num_stages")) {
            tir::For new_for = Downcast<tir::For>(stmt);
            auto *n = new_for.CopyOnWrite();
            n->annotations.erase("num_stages");
            return std::move(new_for);
          }
          return stmt;
        }
      };
      StripPipelineAnnotation stripper;
      auto stripped = stripper(original_f->body);
      auto *fn = original_f.CopyOnWrite();
      fn->body = stripped;
      return original_f;
    }
    LOG(WARNING) << "[WS] transformation applied successfully";
    return result;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ProducerConsumerWarpSpecialized",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ProducerConsumerWarpSpecialized",
                        ProducerConsumerWarpSpecialized);
  refl::GlobalDef().def("tl.transform.ProducerConsumerWarpSpecializedTiled",
                        ProducerConsumerWarpSpecialized);
}

} // namespace tl
} // namespace tvm
