/*!
 * \file producer_consumer_ws.cc
 * \brief Producer-consumer warp specialization for sm90+ async-copy pipelines.
 *
 * Works on the inline barrier IR emitted by lowering passes such as
 * LowerBulkCopy / LowerPTXAsyncCopy:
 *   SeqStmt({
 *     AttrStmt("tl.tma_copy_write_buffer", buf, 1,
 *       IfThenElse(threadIdx.x == 0,
 *         SeqStmt({arrive_expect_tx(mbar, bytes), tma_load(...)}))),
 *     mbarrier_wait_parity(mbar, parity)
 *   })
 *
 * The pass splits the pipelined loop into:
 *   producer: issues TMA / cp.async
 *   consumer: waits, computes, and releases buffers
 *
 * For pure-TMA loops we rewrite the forward-barrier protocol so the producer
 * releases the barrier after issuing the TMA copy:
 *   expect_transaction -> tma_load -> arrive
 */

#include "../op/utils.h"
#include "common/mbarrier.h"
#include "common/tma_copy_utils.h"
#include "warp_specialized_rewriter.h"

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tir;
using namespace runtime;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

enum class AsyncProducerKind : uint8_t { kTma, kCpAsync };

struct AsyncCopyBlockInfo {
  AsyncProducerKind kind;
  Stmt producer_stmt;              // TMA issue or cp.async enqueue+commit
  Optional<Stmt> wait_stmt;        // Existing forward wait for TMA blocks
  Optional<Var> write_buffer_data; // shared buffer written by producer
};

using BufferDataToBufferMap =
    std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using VarBindingMap =
    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;

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

  void KillDefs(const LocalAccessSummary &summary) {
    for (const auto &buf : summary.write_buffers) {
      buffers.erase(buf);
    }
    for (const auto &var : summary.def_vars) {
      vars.erase(var);
    }
  }

  void AddUses(const LocalAccessSummary &summary) {
    buffers.insert(summary.read_buffers.begin(), summary.read_buffers.end());
    vars.insert(summary.read_vars.begin(), summary.read_vars.end());
  }
};

// ---------------------------------------------------------------------------
// PhaseCounter: mutable int32 counter for guarded-loop phase tracking
// ---------------------------------------------------------------------------

/*!
 * \brief When a pipeline loop body is conditionally guarded (e.g.
 *        `if block_mask[k]: ...`), the loop-variable-based parity
 *        `(k / num_stages) % 2` can desynchronise because skipped iterations
 *        don't touch barriers.  A PhaseCounter is a local int32[1] buffer
 *        that tracks the *actual* number of guarded-body entries so that
 *        parity/stage are always correct.
 */
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

  /*! Wrap a For-loop with Allocate + DeclBuffer + Init(0). */
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

/*!
 * \brief Replace the loop-variable-based stage expression with a
 *        phase-counter-based one inside producer / consumer statements.
 *
 *  When `needs_phase_counter` is true, the barrier IDs already use
 *  `phase_counter->StageExpr(N)` but the shared-memory buffer offsets
 *  still embed `FloorMod(loop_var - loop_min, N)`.  This mutator
 *  rewrites every matching FloorMod to the replacement expression so
 *  that stage indexing stays in sync with barrier indexing when loop
 *  iterations are conditionally skipped.
 */
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

  /*! Match `loop_var`, `loop_var - loop_min`, or `loop_var - 0`. */
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

  static LocalAccessSummary
  CollectExpr(const PrimExpr &expr, const BufferDataToBufferMap &buffer_map) {
    LocalAccessCollector collector(buffer_map);
    collector.VisitExpr(expr);
    return std::move(collector.summary_);
  }

private:
  explicit LocalAccessCollector(const BufferDataToBufferMap &buffer_map)
      : buffer_data_to_buffer_(buffer_map) {}

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
    if (IsLocalBuffer(op->buffer, true)) {
      summary_.read_buffers.insert(op->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsLocalBuffer(op->buffer, true)) {
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
    if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3);
      const auto *base_load = op->args[0].as<BufferLoadNode>();
      ICHECK(base_load);
      if (IsLocalBuffer(base_load->buffer, true)) {
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
          IsLocalBuffer(it->second, true)) {
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
    if (const auto *imm = expr.as<IntImmNode>()) {
      return static_cast<int>(imm->value);
    }
    return 3;
  }

  const BufferDataToBufferMap &buffer_data_to_buffer_;
  LocalAccessSummary summary_;
  VarSet bound_vars_;
};

class ProducerSimtCopyDetector : public StmtExprVisitor {
public:
  static bool HasSimtCopy(const Stmt &stmt,
                          const BufferDataToBufferMap &buffer_map) {
    ProducerSimtCopyDetector detector(buffer_map);
    detector.VisitStmt(stmt);
    return detector.has_global_read_ && detector.has_shared_write_;
  }

private:
  explicit ProducerSimtCopyDetector(const BufferDataToBufferMap &buffer_map)
      : buffer_data_to_buffer_(buffer_map) {}

  void VisitStmt_(const IfThenElseNode *op) final {
    bool old_in_if_cond = in_if_cond_;
    in_if_cond_ = true;
    VisitExpr(op->condition);
    in_if_cond_ = old_in_if_cond;
    VisitStmt(op->then_case);
    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (!in_if_cond_ && !in_async_copy_ && IsGlobalBuffer(op->buffer)) {
      has_global_read_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (!in_if_cond_ && !in_async_copy_ && IsSharedBuffer(op->buffer)) {
      has_shared_write_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    bool old_in_async_copy = in_async_copy_;
    if (op->op.same_as(tma_load()) || op->op.same_as(tma_load_im2col()) ||
        op->op.same_as(tma_store()) || op->op.same_as(tma_store_arrive()) ||
        op->op.same_as(tma_store_wait()) ||
        op->op.same_as(tl::ptx_cp_async()) ||
        op->op.same_as(builtin::ptx_cp_async())) {
      in_async_copy_ = true;
    }

    if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3);
      const auto *base_load = op->args[0].as<BufferLoadNode>();
      ICHECK(base_load);
      MarkAccess(base_load->buffer, GetConstAccessMask(op->args[2]));
      for (const auto &index : base_load->indices) {
        VisitExpr(index);
      }
      VisitExpr(op->args[1]);
      in_async_copy_ = old_in_async_copy;
      return;
    }

    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5);
      const auto *var = op->args[1].as<VarNode>();
      ICHECK(var);
      auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
      if (it != buffer_data_to_buffer_.end()) {
        MarkAccess(it->second, GetConstAccessMask(op->args[4]));
      }
      VisitExpr(op->args[2]);
      VisitExpr(op->args[3]);
      in_async_copy_ = old_in_async_copy;
      return;
    }

    StmtExprVisitor::VisitExpr_(op);
    in_async_copy_ = old_in_async_copy;
  }

  void MarkAccess(const Buffer &buffer, int rw_mask) {
    if (in_if_cond_ || in_async_copy_ || !buffer.defined()) {
      return;
    }
    if ((rw_mask & 1) && IsGlobalBuffer(buffer)) {
      has_global_read_ = true;
    }
    if ((rw_mask & 2) && IsSharedBuffer(buffer)) {
      has_shared_write_ = true;
    }
  }

  int GetConstAccessMask(const PrimExpr &expr) const {
    if (const auto *imm = expr.as<IntImmNode>()) {
      return static_cast<int>(imm->value);
    }
    return 3;
  }

  const BufferDataToBufferMap &buffer_data_to_buffer_;
  bool has_global_read_{false};
  bool has_shared_write_{false};
  bool in_if_cond_{false};
  bool in_async_copy_{false};
};

// ---------------------------------------------------------------------------
// Helpers (reused from warp_specialized_rewriter.cc patterns)
// ---------------------------------------------------------------------------

static PrimExpr makeGetBarrier(const Buffer &barrier_buf, PrimExpr barrier_id) {
  return MakeBarrierRef(barrier_buf, std::move(barrier_id));
}

static Stmt makeArriveBarrier(const Buffer &barrier_buf, PrimExpr barrier_id) {
  Array<PrimExpr> args = {makeGetBarrier(barrier_buf, std::move(barrier_id))};
  return Evaluate(
      Call(DataType::Handle(), builtin::ptx_arrive_barrier(), args));
}

static Stmt makeCpAsyncBarrierNoInc(const Buffer &barrier_buf,
                                    PrimExpr barrier_id) {
  auto call = Call(DataType::Handle(), tl::ptx_cp_async_barrier_noinc(),
                   {makeGetBarrier(barrier_buf, std::move(barrier_id))});
  return Evaluate(call);
}

static Stmt makeParityWait(const Buffer &barrier_buf, PrimExpr barrier_id,
                           PrimExpr parity) {
  auto call = Call(
      DataType::Handle(), mbarrier_wait_parity(),
      {makeGetBarrier(barrier_buf, std::move(barrier_id)), std::move(parity)});
  return Evaluate(call);
}

static bool IsTrivialNoOpStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *imm = eval->value.as<IntImmNode>()) {
      return imm->value == 0;
    }
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq) {
      if (!IsTrivialNoOpStmt(s)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// AsyncCopyBlockExtractor
// ---------------------------------------------------------------------------

/*!
 * \brief Extract async producer blocks from a flattened loop body.
 *
 * Recognized patterns:
 *
 *  Pattern 1: AttrStmt("tl.tma_copy_write_buffer", ...) + mbarrier_wait_parity
 *  Pattern 2: IfThenElse containing tma_load + mbarrier_wait_parity
 *  Pattern 3: one or more cp_async-only stmts + commit_group + wait_group(0)
 *
 * Everything else is classified as a compute statement.
 */
class AsyncCopyBlockExtractor {
public:
  std::vector<AsyncCopyBlockInfo> blocks;
  std::vector<Stmt> compute_stmts;

  void Extract(const Array<Stmt> &flat_stmts) {
    size_t i = 0;
    while (i < flat_stmts.size()) {
      if (i + 1 < flat_stmts.size() &&
          IsMbarrierWaitParity(flat_stmts[i + 1])) {
        Optional<Var> write_buffer_data =
            ExtractTmaCopyWriteBufferData(flat_stmts[i]);
        // Check Pattern 1/2: TMA producer + wait pair, optionally wrapped in a
        // simple guard/Block/Let/Attr shell. Recover the written shared buffer
        // when the tl.tma_copy_write_buffer annotation survives under wrappers.
        if (write_buffer_data.defined() || ContainsTmaLoad(flat_stmts[i])) {
          blocks.push_back({AsyncProducerKind::kTma,
                            StripTmaCopyWriteBufferAttr(flat_stmts[i]),
                            Optional<Stmt>(flat_stmts[i + 1]),
                            write_buffer_data});
          i += 2;
          continue;
        }
      }
      if (ContainsPtxCpAsync(flat_stmts[i])) {
        size_t cp_async_end = i;
        while (cp_async_end + 1 < flat_stmts.size() &&
               ContainsPtxCpAsync(flat_stmts[cp_async_end + 1])) {
          ++cp_async_end;
        }
        if (cp_async_end + 2 < flat_stmts.size() &&
            IsPtxCommitGroup(flat_stmts[cp_async_end + 1]) &&
            IsPtxWaitGroupZero(flat_stmts[cp_async_end + 2])) {
          Array<Stmt> producer_seq;
          producer_seq.reserve(cp_async_end - i + 2);
          for (size_t j = i; j <= cp_async_end; ++j) {
            producer_seq.push_back(flat_stmts[j]);
          }
          producer_seq.push_back(flat_stmts[cp_async_end + 1]);
          Stmt producer_stmt = producer_seq.size() == 1 ? producer_seq[0]
                                                        : SeqStmt(producer_seq);
          blocks.push_back({AsyncProducerKind::kCpAsync, producer_stmt,
                            Optional<Stmt>(),
                            GetCpAsyncDstBufferData(producer_stmt)});
          i = cp_async_end + 3;
          continue;
        }
      }
      compute_stmts.push_back(flat_stmts[i]);
      i++;
    }
  }

private:
  static const CallNode *GetEvaluateCallInSimpleWrapper(const Stmt &stmt) {
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      return eval->value.as<CallNode>();
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
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
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return GetEvaluateCallInSimpleWrapper(seq->seq[0]);
      }
      return nullptr;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return GetEvaluateCallInSimpleWrapper(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return GetEvaluateCallInSimpleWrapper(realize->block->body);
      }
      return nullptr;
    }
    return nullptr;
  }

  static Optional<Var> ExtractTmaCopyWriteBufferData(const Stmt &stmt) {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == "tl.tma_copy_write_buffer") {
        const auto *v = attr->node.as<VarNode>();
        ICHECK(v);
        return GetRef<Var>(v);
      }
      return ExtractTmaCopyWriteBufferData(attr->body);
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        return ExtractTmaCopyWriteBufferData(if_stmt->then_case);
      }
      return Optional<Var>();
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return ExtractTmaCopyWriteBufferData(let->body);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return ExtractTmaCopyWriteBufferData(seq->seq[0]);
      }
      return Optional<Var>();
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return ExtractTmaCopyWriteBufferData(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return ExtractTmaCopyWriteBufferData(realize->block->body);
      }
      return Optional<Var>();
    }
    return Optional<Var>();
  }

  static bool IsMbarrierWaitParity(const Stmt &stmt) {
    const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
    return call && call->op.same_as(mbarrier_wait_parity());
  }

  static bool ContainsTmaLoad(const Stmt &stmt) {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (auto *call = node.as<CallNode>()) {
        if (call->op.same_as(tma_load()) ||
            call->op.same_as(tma_load_im2col())) {
          found = true;
        }
      }
    });
    return found;
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

  static bool IsPtxWaitGroupZero(const Stmt &stmt) {
    const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
    if (!call || !call->op.same_as(builtin::ptx_wait_group())) {
      return false;
    }
    ICHECK_EQ(call->args.size(), 1);
    const auto *imm = call->args[0].as<IntImmNode>();
    ICHECK(imm);
    return imm->value == 0;
  }

  static Optional<Var> AccessPtrBufferVar(const PrimExpr &ptr) {
    const auto *call = ptr.as<CallNode>();
    if (!call) {
      return Optional<Var>();
    }
    if (call->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(call->args.size(), 3);
      const auto *base_load = call->args[0].as<BufferLoadNode>();
      ICHECK(base_load);
      return base_load->buffer->data;
    }
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(call->args.size(), 5);
      const auto *var = call->args[1].as<VarNode>();
      ICHECK(var);
      return GetRef<Var>(var);
    }
    ICHECK(false) << "Expected tl.access_ptr or tvm_access_ptr";
    throw;
  }

  static Optional<Var> GetCpAsyncDstBufferData(const Stmt &stmt) {
    Optional<Var> found = std::nullopt;
    bool multiple = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (multiple) {
        return;
      }
      const auto *call = node.as<CallNode>();
      if (!call) {
        return;
      }
      if (!(call->op.same_as(builtin::ptx_cp_async()) ||
            call->op.same_as(tl::ptx_cp_async()))) {
        return;
      }
      ICHECK(!call->args.empty());
      Optional<Var> current = AccessPtrBufferVar(call->args[0]);
      if (!current.defined()) {
        return;
      }
      if (!found.defined()) {
        found = current;
      } else if (found.value().get() != current.value().get()) {
        multiple = true;
      }
    });
    if (multiple) {
      return Optional<Var>();
    }
    return found;
  }
};

// ---------------------------------------------------------------------------
// ThreadIdxRewriter (from warp_specialized_rewriter.cc)
// ---------------------------------------------------------------------------

class PCThreadIdxRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt stmt, Var thread_var, PrimExpr replaced,
                      PrimExpr thread_extent, bool do_shuffle = false) {
    auto rewriter =
        PCThreadIdxRewriter(std::move(thread_var), std::move(replaced),
                            std::move(thread_extent), do_shuffle);
    return rewriter(std::move(stmt));
  }

private:
  PCThreadIdxRewriter(Var thread_var, PrimExpr replaced, PrimExpr thread_extent,
                      bool do_shuffle)
      : thread_var_(std::move(thread_var)), replaced_(std::move(replaced)),
        thread_extent_(std::move(thread_extent)), do_shuffle_(do_shuffle) {}

  PrimExpr VisitExpr_(const VarNode *var) final {
    if (var == thread_var_.get()) {
      return replaced_;
    }
    return StmtExprMutator::VisitExpr_(var);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    auto f_uses_thread = [=](const tvm::tir::VarNode *v) {
      return v == thread_var_.get();
    };
    maybe_thread_opt_ = false;
    if (!op->else_case.defined() && op->condition.as<EQNode>() &&
        UsesVar(op->condition, f_uses_thread) &&
        !(UsesVar(op->then_case, f_uses_thread))) {
      auto eq_op = Downcast<EQ>(op->condition);
      if (eq_op->a.as<VarNode>() == thread_var_.get() ||
          eq_op->b.as<VarNode>() == thread_var_.get()) {
        maybe_thread_opt_ = true;
      }
      auto then_case = StmtExprMutator::VisitStmt(op->then_case);
      maybe_thread_opt_ = do_shuffle_ && maybe_thread_opt_ && has_tma_op_;
      has_tma_op_ = false;
      if (maybe_thread_opt_) {
        return IfThenElse(
            Call(DataType::Bool(), tl_shuffle_elect(), {thread_extent_}),
            StmtExprMutator::VisitStmt(op->then_case), std::nullopt);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::tma_load()) ||
        op->op.same_as(tl::tma_load_im2col()) ||
        op->op.same_as(tl::tma_store()) ||
        op->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
        op->op.same_as(mbarrier_expect_tx())) {
      has_tma_op_ = true;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Var thread_var_;
  PrimExpr replaced_;
  PrimExpr thread_extent_;
  bool maybe_thread_opt_ = false;
  bool do_shuffle_;
  bool has_tma_op_ = false;
};

// ---------------------------------------------------------------------------
// MbarrierInitRemover: removes barrier_init annotations and shared.barrier
// buffers from blocks outside the transformed block.
// ---------------------------------------------------------------------------

/*!
 * \brief Post-transform cleanup: remove barrier_init annotations and
 *        shared.barrier alloc_buffers that remain outside the transformed
 *        block.
 *        The new init is already emitted inside the block by the rewriter.
 */
class MbarrierInitRemover : public StmtExprMutator {
public:
  static Stmt Remove(Stmt stmt) {
    MbarrierInitRemover remover;
    return remover(std::move(stmt));
  }

private:
  Stmt VisitStmt_(const BlockNode *op) final {
    // Remove barrier_init annotation and shared.barrier buffers from
    // blocks outside the transformed region.
    bool has_barrier_init = op->annotations.count("barrier_init");
    bool has_barrier_bufs = false;
    for (const auto &buf : op->alloc_buffers) {
      if (buf.scope() == "shared.barrier") {
        has_barrier_bufs = true;
        break;
      }
    }

    if (!has_barrier_init && !has_barrier_bufs) {
      return StmtExprMutator::VisitStmt_(op);
    }

    Block block = GetRef<Block>(op);
    auto block_ptr = block.CopyOnWrite();

    if (has_barrier_init) {
      Map<String, Any> new_annos;
      for (const auto &[key, value] : block_ptr->annotations) {
        if (key != "barrier_init") {
          new_annos.Set(key, value);
        }
      }
      block_ptr->annotations = new_annos;
    }

    if (has_barrier_bufs) {
      Array<Buffer> new_alloc_buffers;
      for (const auto &buf : block_ptr->alloc_buffers) {
        if (buf.scope() != "shared.barrier") {
          new_alloc_buffers.push_back(buf);
        }
      }
      block_ptr->alloc_buffers = new_alloc_buffers;
    }

    block_ptr->body = VisitStmt(block_ptr->body);
    return block;
  }

  // Stop recursion at BlockRealize — the new init is inside the block
  // and we don't want to remove it.
  Stmt VisitStmt_(const BlockRealizeNode *op) final { return GetRef<Stmt>(op); }
};

// ---------------------------------------------------------------------------
// ProducerConsumerWSRewriter — main pass
// ---------------------------------------------------------------------------

class ProducerConsumerWSRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    // Check thread tags
    if (!ThreadTagChecker::HasOnlyThreadIdxX(f)) {
      LOG(WARNING) << "ProducerConsumerWS: disabled because program uses "
                      "thread tags other than threadIdx.x";
      return f;
    }

    ProducerConsumerWSRewriter T;
    f.CopyOnWrite()->body = T(f->body);

    // TODO(lei): This should be refactored
    // If WS was applied, remove any barrier_init annotations and
    // shared.barrier buffers that remain OUTSIDE the block (e.g. at
    // function body level from lower_tile_op). The new init is already
    // inside the block.
    if (T.ws_transformed_) {
      f.CopyOnWrite()->body = MbarrierInitRemover::Remove(f->body);
    }

    return f;
  }

private:
  // Locate the threadIdx.x binding
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent &&
        Downcast<IterVar>(op->node)->thread_tag == "threadIdx.x") {
      thread_iv_ = Downcast<IterVar>(op->node);
      Optional<PrimExpr> old_num_threads = num_threads_;
      num_threads_ = std::nullopt;
      AttrStmt attr_stmt = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
      if (num_threads_.defined()) {
        PrimExpr num_threads = num_threads_.value();
        thread_iv_.CopyOnWrite()->dom = {0, num_threads};
        attr_stmt.CopyOnWrite()->node = thread_iv_;
        attr_stmt.CopyOnWrite()->value = num_threads;
      }
      // clean up if we may have multiple threadIdx.x that
      // need to be transformed
      num_threads_ = old_num_threads;
      thread_iv_ = {};
      return attr_stmt;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    if (!thread_iv_.defined())
      return StmtExprMutator::VisitStmt_(op);

    const Block &orig_block = op->block;

    // Find the explicitly pipelined loop for producer/consumer WS.
    const ForNode *pipeline_loop = FindAnnotatedPipelineLoop(orig_block->body);
    if (!pipeline_loop)
      return StmtExprMutator::VisitStmt_(op);

    auto num_stages_anno = pipeline_loop->annotations.Get("num_stages");
    ICHECK(num_stages_anno);
    int num_stages =
        static_cast<int>(Downcast<Integer>(num_stages_anno.value())->value);
    ICHECK_GE(num_stages, 1);

    // Detect cluster barriers and compute cluster size from block annotations.
    is_cluster_barrier_ = false;
    cluster_size_ = 1;
    for (const auto &buf : orig_block->alloc_buffers) {
      if (buf.scope() == "shared.cluster_barrier") {
        is_cluster_barrier_ = true;
        break;
      }
    }
    if (is_cluster_barrier_ && orig_block->annotations.count("cluster_dims")) {
      if (auto arr = orig_block->annotations.Get("cluster_dims")
                         ->try_cast<Array<Integer>>()) {
        int sz = 1;
        for (auto d : arr.value())
          sz *= static_cast<int>(d->value);
        cluster_size_ = sz;
      }
    }

    // Flatten the loop body
    Array<Stmt> flat_stmts;
    Stmt loop_body_root = pipeline_loop->body;
    if (auto *realize = pipeline_loop->body.as<BlockRealizeNode>()) {
      loop_body_root = realize->block->body;
    }
    std::vector<std::pair<Var, PrimExpr>> loop_body_lets;
    while (const auto *let_stmt = loop_body_root.as<LetStmtNode>()) {
      loop_body_lets.emplace_back(let_stmt->var, let_stmt->value);
      loop_body_root = let_stmt->body;
    }
    FlattenSeqStmt(loop_body_root, &flat_stmts);
    auto rewrap_loop_body_lets = [&](Stmt body) {
      for (auto it = loop_body_lets.rbegin(); it != loop_body_lets.rend();
           ++it) {
        body = LetStmt((*it).first, (*it).second, body);
      }
      return body;
    };
    // Extract async producer blocks (TMA and cp.async)
    AsyncCopyBlockExtractor extractor;
    extractor.Extract(flat_stmts);

    if (extractor.blocks.empty()) {
      // No TMA loads found — fall through to standard pipeline
      return StmtExprMutator::VisitStmt_(op);
    }

    // Check if there are existing tl_pipeline_order/tl_pipeline_stage
    // with -1 values (WS+TMA enabled markers) — if so, use those
    auto order_anno = pipeline_loop->annotations.Get("tl_pipeline_order");
    auto stage_anno = pipeline_loop->annotations.Get("tl_pipeline_stage");
    if (order_anno && stage_anno) {
      auto order_array = Downcast<Array<Integer>>(order_anno.value());
      for (const auto &val : order_array) {
        if (val->value == -1) {
          // Already has WS pipeline annotations — skip
          return StmtExprMutator::VisitStmt_(op);
        }
      }
    }

    VarBindingMap saved_loop_guard_bindings = current_loop_guard_bindings_;
    for (const auto &[var, value] : loop_body_lets) {
      current_loop_guard_bindings_[var] = value;
    }

    BufferDataToBufferMap buffer_data_to_buffer =
        BufferDataToBufferCollector::Collect(GetRef<Stmt>(op));

    // ---------------------------------------------------------------
    // Build producer and consumer loop bodies
    // ---------------------------------------------------------------
    PrimExpr consumer_thread_extent = thread_iv_->dom->extent;
    consumer_thread_extent_ =
        consumer_thread_extent; // Store for RebuildBlockBody
    PrimExpr producer_thread_extent = IntImm(DataType::Int(32), 128);
    producer_thread_extent_ = producer_thread_extent;

    // Barrier layout has two modes:
    // 1) Mixed TMA + cp.async:
    //    keep existing TMA forward ids, append cp.async forward ids, then
    //    append back-pressure ids.
    // 2) Pure TMA:
    //    remap to [loop forward][back-pressure][preloop forward] so producer
    //    and consumer follow the same protocol as the hand-written WS kernels.
    int num_existing_tma_fwd_barriers = 0;
    int num_cp_async_groups = 0;
    for (const auto &block : extractor.blocks) {
      if (block.kind == AsyncProducerKind::kTma) {
        ++num_existing_tma_fwd_barriers;
      } else if (block.kind == AsyncProducerKind::kCpAsync) {
        ++num_cp_async_groups;
      }
    }
    std::vector<int> wait_insert_pos(extractor.blocks.size(), 0);
    std::vector<int> arrive_insert_pos(
        extractor.blocks.size(),
        static_cast<int>(extractor.compute_stmts.size()));
    for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
      if (!extractor.blocks[ti].write_buffer_data.defined()) {
        continue;
      }
      const Var &target = extractor.blocks[ti].write_buffer_data.value();
      int first_read = -1;
      int last_access = -1;
      for (size_t ci = 0; ci < extractor.compute_stmts.size(); ++ci) {
        BufferDataAccessInfo access = AnalyzeBufferDataAccess(
            extractor.compute_stmts[ci], target, buffer_data_to_buffer);
        if (access.read && first_read < 0) {
          first_read = static_cast<int>(ci);
        }
        if (access.HasAnyAccess()) {
          last_access = static_cast<int>(ci);
        }
      }
      if (first_read >= 0) {
        wait_insert_pos[ti] = first_read;
        arrive_insert_pos[ti] = last_access + 1;
      } else if (last_access >= 0) {
        // Write-only statements that touch the producer-written shared buffer
        // do not need the producer result, so keep the forward wait at the
        // loop head while still delaying back-pressure release until the last
        // consumer-side access.
        wait_insert_pos[ti] = 0;
        arrive_insert_pos[ti] = last_access + 1;
      }
    }
    int num_existing_loop_fwd_barriers =
        num_existing_tma_fwd_barriers * num_stages;
    int original_num_existing_loop_fwd_barriers =
        num_existing_loop_fwd_barriers;
    int inferred_existing_required =
        InferMinRequiredBarrierCount(orig_block->body);
    int required_preloop_tma_pairs = CountRewrittenPureTmaPreloopForwardPairs(
        orig_block->body, pipeline_loop);
    bool old_use_full_tma_forward_barrier_protocol =
        use_full_tma_forward_barrier_protocol_;
    bool old_remap_pure_tma_barriers = remap_pure_tma_barriers_;
    int old_pure_tma_preloop_fwd_base = pure_tma_preloop_fwd_base_;
    int old_pure_tma_preloop_fwd_count = pure_tma_preloop_fwd_count_;
    int old_pure_tma_preloop_fwd_cursor = pure_tma_preloop_fwd_cursor_;
    use_full_tma_forward_barrier_protocol_ = (num_cp_async_groups == 0);
    remap_pure_tma_barriers_ = use_full_tma_forward_barrier_protocol_;
    std::vector<Stmt> ws_producer_stmts(extractor.blocks.size());
    std::vector<Optional<Stmt>> ws_wait_stmts(extractor.blocks.size(),
                                              std::nullopt);
    std::vector<Optional<PrimExpr>> producer_issue_guards(
        extractor.blocks.size(), std::nullopt);
    std::vector<Optional<Stmt>> producer_issue_guard_sources(
        extractor.blocks.size(), std::nullopt);
    std::vector<Optional<PrimExpr>> protocol_guards(extractor.blocks.size(),
                                                    std::nullopt);
    std::vector<Optional<Stmt>> protocol_guard_sources(extractor.blocks.size(),
                                                       std::nullopt);
    for (size_t i = 0; i < extractor.blocks.size(); ++i) {
      ws_producer_stmts[i] = extractor.blocks[i].producer_stmt;
      ws_wait_stmts[i] = extractor.blocks[i].wait_stmt;
      producer_issue_guards[i] =
          ExtractNonThreadProducerGuard(extractor.blocks[i].producer_stmt);
      if (producer_issue_guards[i].defined()) {
        producer_issue_guard_sources[i] = extractor.blocks[i].producer_stmt;
        protocol_guards[i] = producer_issue_guards[i];
        protocol_guard_sources[i] = extractor.blocks[i].producer_stmt;
        if (arrive_insert_pos[i] > 0 &&
            arrive_insert_pos[i] <=
                static_cast<int>(extractor.compute_stmts.size())) {
          const Stmt &arrive_source =
              extractor.compute_stmts[arrive_insert_pos[i] - 1];
          Optional<PrimExpr> arrive_guard =
              ExtractNonThreadProducerGuard(arrive_source);
          if (arrive_guard.defined()) {
            protocol_guards[i] = arrive_guard;
            protocol_guard_sources[i] = arrive_source;
          }
        }
      }
      // NOTE: Previously, when the guard was a mask-like boolean expression
      // (e.g. BlockMask[by, bx, k]), the producer would strip the guard and
      // issue TMA loads unconditionally.  This causes unnecessary memory
      // traffic for sparse workloads, so we now keep the guard on the
      // producer side and rely on phase-counter-based parity tracking to
      // maintain barrier synchronisation.
    }

    // ---------------------------------------------------------------
    // Detect whether the pipeline loop needs counter-based phase
    // tracking.  This is necessary when the loop body is conditionally
    // guarded (e.g. `if block_mask[k]`) so that skipped iterations do
    // not desynchronise the mbarrier parity.
    // ---------------------------------------------------------------
    bool needs_phase_counter = false;
    Optional<PrimExpr> uniform_phase_guard;
    Optional<Stmt> uniform_phase_guard_source;
    {
      StructuralEqual eq;
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        if (protocol_guards[i].defined()) {
          if (!needs_phase_counter) {
            needs_phase_counter = true;
            uniform_phase_guard = protocol_guards[i];
            uniform_phase_guard_source = protocol_guard_sources[i];
          } else if (!eq(uniform_phase_guard.value(),
                         protocol_guards[i].value())) {
            // Different guards on different blocks – fall back to
            // original loop-variable parity (no counter).
            needs_phase_counter = false;
            break;
          }
        }
      }
      // Only use counter when ALL blocks share the same guard.
      if (needs_phase_counter) {
        for (size_t i = 0; i < extractor.blocks.size(); ++i) {
          if (!protocol_guards[i].defined()) {
            needs_phase_counter = false;
            break;
          }
        }
      }
    }

    std::optional<PhaseCounter> producer_phase_counter;
    std::optional<PhaseCounter> consumer_phase_counter;
    if (needs_phase_counter) {
      producer_phase_counter = PhaseCounter::Create("producer_phase_cnt");
      consumer_phase_counter = PhaseCounter::Create("consumer_phase_cnt");
    }

    StructuralEqual equal;
    auto same_optional_expr = [&](const Optional<PrimExpr> &guard_a,
                                  const Optional<PrimExpr> &guard_b) {
      if (guard_a.defined() != guard_b.defined()) {
        return false;
      }
      return !guard_a.defined() || equal(guard_a.value(), guard_b.value());
    };
    auto same_guard = [&](size_t lhs, size_t rhs) {
      return same_optional_expr(producer_issue_guards[lhs],
                                producer_issue_guards[rhs]) &&
             same_optional_expr(protocol_guards[lhs], protocol_guards[rhs]);
    };
    std::vector<int> block_group(extractor.blocks.size(), 0);
    int num_block_groups = 0;
    if (!extractor.blocks.empty()) {
      int next_group = 0;
      block_group[0] = next_group++;
      bool current_group_has_tma =
          extractor.blocks[0].kind == AsyncProducerKind::kTma;
      for (size_t i = 1; i < extractor.blocks.size(); ++i) {
        bool merge_with_prev =
            wait_insert_pos[i] == wait_insert_pos[i - 1] &&
            arrive_insert_pos[i] == arrive_insert_pos[i - 1] &&
            same_guard(i - 1, i);
        if (merge_with_prev && !remap_pure_tma_barriers_ &&
            current_group_has_tma &&
            extractor.blocks[i].kind == AsyncProducerKind::kTma) {
          // Mixed groups can safely share one TMA barrier with cp.async
          // arrive-on notifications, but keeping multiple TMA producers on the
          // same preserved protocol would over-arrive the barrier.
          merge_with_prev = false;
        }
        block_group[i] = merge_with_prev ? block_group[i - 1] : next_group++;
        if (!merge_with_prev) {
          current_group_has_tma =
              extractor.blocks[i].kind == AsyncProducerKind::kTma;
        } else if (extractor.blocks[i].kind == AsyncProducerKind::kTma) {
          current_group_has_tma = true;
        }
      }
      num_block_groups = next_group;
    }

    std::vector<Array<Stmt>> producer_loop_prefix_stmts(
        extractor.blocks.size());
    std::vector<bool> moved_compute_stmts(extractor.compute_stmts.size(),
                                          false);
    int compute_cursor = 0;
    for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
      bool is_first_in_group =
          ti == 0 || block_group[ti] != block_group[ti - 1];
      if (!is_first_in_group) {
        continue;
      }
      int wait_pos = wait_insert_pos[ti];
      if (wait_pos <= compute_cursor) {
        compute_cursor = std::max(compute_cursor, wait_pos);
        continue;
      }
      bool all_movable = true;
      for (int ci = compute_cursor; ci < wait_pos; ++ci) {
        if (!IsProducerMovableLoopPrefixStmt(extractor.compute_stmts[ci])) {
          all_movable = false;
          break;
        }
      }
      if (all_movable) {
        for (int ci = compute_cursor; ci < wait_pos; ++ci) {
          producer_loop_prefix_stmts[ti].push_back(extractor.compute_stmts[ci]);
          moved_compute_stmts[ci] = true;
        }
      }
      compute_cursor = wait_pos;
    }

    auto stmt_has_lowered_simt_copy = [&](const Stmt &stmt) {
      return ProducerSimtCopyDetector::HasSimtCopy(stmt, buffer_data_to_buffer);
    };
    bool producer_needs_full_thread_extent =
        std::any_of(ws_producer_stmts.begin(), ws_producer_stmts.end(),
                    stmt_has_lowered_simt_copy);
    if (!producer_needs_full_thread_extent) {
      for (const auto &prefix_stmts : producer_loop_prefix_stmts) {
        for (const auto &stmt : prefix_stmts) {
          if (stmt_has_lowered_simt_copy(stmt)) {
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
      // LowerTileOp may already have materialized SIMT global->shared copies.
      // Those copies cannot be safely remapped onto a smaller producer warp
      // partition, so keep the producer extent at the original thread extent.
      producer_thread_extent = consumer_thread_extent;
    }
    producer_thread_extent_ = producer_thread_extent;

    std::vector<bool> group_has_tma(num_block_groups, false);
    std::vector<bool> group_has_cp_async(num_block_groups, false);
    for (size_t i = 0; i < extractor.blocks.size(); ++i) {
      int group = block_group[i];
      if (extractor.blocks[i].kind == AsyncProducerKind::kTma) {
        group_has_tma[group] = true;
      } else if (extractor.blocks[i].kind == AsyncProducerKind::kCpAsync) {
        group_has_cp_async[group] = true;
      }
    }
    int num_tma_groups = 0;
    int num_cp_async_only_groups = 0;
    for (int group = 0; group < num_block_groups; ++group) {
      if (group_has_tma[group]) {
        ++num_tma_groups;
      } else if (group_has_cp_async[group]) {
        ++num_cp_async_only_groups;
      }
    }
    num_existing_tma_fwd_barriers = num_tma_groups;
    num_existing_loop_fwd_barriers = num_existing_tma_fwd_barriers * num_stages;
    int num_new_cp_async_fwd_barriers = num_cp_async_only_groups * num_stages;

    int num_existing_barriers = 0;
    int num_preloop_fwd_barriers = 0;
    if (remap_pure_tma_barriers_) {
      // Pure-TMA WS remaps pre-loop TMA prefixes to a dedicated barrier range.
      // Some kernels reuse loop barrier ids for those prefixes in the original
      // IR, so `inferred_existing_required` alone can undercount how many
      // distinct pre-loop barriers the rewritten form needs.
      num_preloop_fwd_barriers =
          std::max(required_preloop_tma_pairs,
                   std::max(0, inferred_existing_required -
                                   original_num_existing_loop_fwd_barriers));
      num_existing_barriers =
          num_existing_loop_fwd_barriers + num_preloop_fwd_barriers;
    } else {
      // Mixed TMA/cp.async keeps any existing non-pipelined forward barriers
      // at their original ids. `inferred_existing_required` already accounts
      // for those explicit references, so avoid reserving an extra unused slot.
      num_existing_barriers =
          std::max(num_existing_loop_fwd_barriers, inferred_existing_required);
      num_preloop_fwd_barriers =
          num_existing_barriers - num_existing_loop_fwd_barriers;
    }
    int num_total_fwd_barriers = 0;
    int num_bp_barriers = num_block_groups * num_stages;
    int total_barriers = 0;

    std::vector<int> fwd_bases(extractor.blocks.size(), -1);
    std::vector<int> bp_bases(extractor.blocks.size(), -1);
    std::vector<PrimExpr> mixed_fwd_arrive_counts;

    if (remap_pure_tma_barriers_) {
      // Pure-TMA layout:
      //   [0, loop_fwd)                    : loop forward barriers
      //   [loop_fwd, loop_fwd + bp)       : back-pressure barriers
      //   [loop_fwd + bp, total_barriers) : preloop/prologue forward barriers
      int next_loop_fwd_base = 0;
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        if (i == 0 || block_group[i] != block_group[i - 1]) {
          fwd_bases[i] = next_loop_fwd_base;
          next_loop_fwd_base += num_stages;
        } else {
          fwd_bases[i] = fwd_bases[i - 1];
        }
      }
      num_total_fwd_barriers =
          num_existing_loop_fwd_barriers + num_preloop_fwd_barriers;
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        bp_bases[i] =
            num_existing_loop_fwd_barriers + block_group[i] * num_stages;
      }
      pure_tma_preloop_fwd_base_ =
          num_existing_loop_fwd_barriers + num_bp_barriers;
      pure_tma_preloop_fwd_count_ = num_preloop_fwd_barriers;
      pure_tma_preloop_fwd_cursor_ = 0;
      total_barriers = num_total_fwd_barriers + num_bp_barriers;
    } else {
      // Mixed path:
      //   [0, num_existing_barriers) : pre-existing forward barriers
      //   [existing, total_fwd)      : new cp.async forward barriers
      //   [total_fwd, total)         : back-pressure barriers
      num_total_fwd_barriers =
          num_existing_barriers + num_new_cp_async_fwd_barriers;
      int next_existing_tma_fwd_base = 0;
      int next_cp_async_fwd_base = num_existing_barriers;
      std::vector<int> group_fwd_bases(num_block_groups, -1);
      mixed_fwd_arrive_counts.assign(num_total_fwd_barriers,
                                     IntImm(DataType::Int(32), 1));
      for (int group = 0; group < num_block_groups; ++group) {
        if (group_has_tma[group]) {
          group_fwd_bases[group] = next_existing_tma_fwd_base;
          next_existing_tma_fwd_base += num_stages;
        } else {
          ICHECK(group_has_cp_async[group]);
          group_fwd_bases[group] = next_cp_async_fwd_base;
          next_cp_async_fwd_base += num_stages;
        }
        PrimExpr group_arrive_count = IntImm(DataType::Int(32), 1);
        if (group_has_cp_async[group]) {
          group_arrive_count = producer_thread_extent;
        }
        for (int stage = 0; stage < num_stages; ++stage) {
          mixed_fwd_arrive_counts[group_fwd_bases[group] + stage] =
              group_arrive_count;
        }
      }
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        fwd_bases[i] = group_fwd_bases[block_group[i]];
        bp_bases[i] = num_total_fwd_barriers + block_group[i] * num_stages;
      }
      total_barriers = num_total_fwd_barriers + num_bp_barriers;
      pure_tma_preloop_fwd_base_ = -1;
      pure_tma_preloop_fwd_count_ = 0;
      pure_tma_preloop_fwd_cursor_ = 0;
    }

    // Defensive check: ensure back-pressure barriers do not overlap
    // any existing (forward/prologue) barrier ids in the original IR.
    if (num_bp_barriers > 0 && !remap_pure_tma_barriers_) {
      int existing_last = inferred_existing_required - 1;
      int bp_begin = bp_bases.front();
      int bp_last = bp_begin + num_bp_barriers - 1;
      ICHECK(bp_begin > existing_last)
          << "ProducerConsumerWS: barrier id overlap detected. "
          << "existing_last=" << existing_last << ", bp_begin=" << bp_begin
          << ", bp_last=" << bp_last;
    }

    // Create barrier buffer early so loop body builders can use it.
    barrier_buf_ =
        CreateMBarrierBuffer(injected_mbarrier_name_, total_barriers);

    Var loop_var = pipeline_loop->loop_var;
    PrimExpr loop_extent = pipeline_loop->extent;
    PrimExpr loop_min = pipeline_loop->min;

    // Compute stage and parity expressions.
    // When needs_phase_counter is true, the loop body is conditionally
    // guarded and we use a mutable counter instead of the loop variable
    // to derive stage/parity.  Producer and consumer have separate
    // counters because they run on different warp partitions.
    PrimExpr linear_idx = loop_var - loop_min;
    PrimExpr base_stage_expr = FloorMod(linear_idx, num_stages);
    PrimExpr base_parity_expr = FloorMod(FloorDiv(linear_idx, num_stages), 2);

    PrimExpr producer_stage_expr =
        needs_phase_counter ? producer_phase_counter->StageExpr(num_stages)
                            : base_stage_expr;
    PrimExpr producer_parity_expr =
        needs_phase_counter ? producer_phase_counter->ParityExpr(num_stages)
                            : base_parity_expr;
    PrimExpr consumer_stage_expr =
        needs_phase_counter ? consumer_phase_counter->StageExpr(num_stages)
                            : base_stage_expr;
    PrimExpr consumer_parity_expr =
        needs_phase_counter ? consumer_phase_counter->ParityExpr(num_stages)
                            : base_parity_expr;

    // --- Build Producer Body ---
    Array<Stmt> producer_body_stmts;
    for (size_t ti = 0; ti < extractor.blocks.size(); ti++) {
      const auto &tma = extractor.blocks[ti];
      int group = block_group[ti];
      bool is_first_in_group =
          ti == 0 || block_group[ti] != block_group[ti - 1];
      bool is_last_in_group = ti + 1 == extractor.blocks.size() ||
                              block_group[ti] != block_group[ti + 1];
      PrimExpr bp_id =
          IntImm(DataType::Int(32), bp_bases[ti]) + producer_stage_expr;

      // Back-pressure wait: producer cannot reuse the stage buffer until the
      // consumer releases it. xor(parity, 1) bootstraps the first iteration.
      if (is_first_in_group) {
        producer_body_stmts.push_back(WrapStmtWithGuardSource(
            protocol_guard_sources[ti], protocol_guards[ti],
            makeParityWait(barrier_buf_, bp_id,
                           bitwise_xor(producer_parity_expr, 1))));
        for (const auto &stmt : producer_loop_prefix_stmts[ti]) {
          producer_body_stmts.push_back(stmt);
        }
      }

      Stmt producer_stmt = ws_producer_stmts[ti];
      if (tma.kind == AsyncProducerKind::kTma) {
        ICHECK_GE(fwd_bases[ti], 0);
        PrimExpr barrier_id =
            IntImm(DataType::Int(32), fwd_bases[ti]) + producer_stage_expr;
        if (use_full_tma_forward_barrier_protocol_) {
          // Pure-TMA WS uses a full producer-side release protocol so the
          // consumer waits on a barrier owned by the producer branch.
          producer_stmt = RewriteTmaForwardProducerStmt(
              producer_stmt, barrier_id, is_last_in_group);
        } else {
          // Mixed groups keep the original producer-side TMA protocol, but
          // rebind grouped loads onto a shared forward barrier. If the group
          // also contains cp.async, let cp.async.mbarrier.arrive.noinc own the
          // arrival count so the shared forward barrier stays on the producer
          // thread extent instead of adding an extra leader-only arrive.
          producer_stmt = RewriteTmaStmtBarrierIdPreserveProtocol(
              producer_stmt, barrier_id, group_has_cp_async[group]);
        }
        // Keep expect/load under the same elected lane when lowering has
        // emitted them as adjacent identical IfThenElse wrappers.
        producer_stmt = MergeAdjacentEquivalentIfs(producer_stmt);
      }

      // Execute the producer statement.
      producer_body_stmts.push_back(producer_stmt);
      if (is_last_in_group && group_has_cp_async[group]) {
        ICHECK_GE(fwd_bases[ti], 0);
        PrimExpr fwd_id =
            IntImm(DataType::Int(32), fwd_bases[ti]) + producer_stage_expr;
        producer_body_stmts.push_back(WrapStmtWithGuardSource(
            producer_issue_guard_sources[ti], producer_issue_guards[ti],
            makeCpAsyncBarrierNoInc(barrier_buf_, fwd_id)));
      }
      // Phase counter increment – exactly once per guarded iteration,
      // after ALL groups have issued their barrier ops.
      // MergeAdjacentEquivalentIfs will fold this into the same guard.
      if (needs_phase_counter && ti + 1 == extractor.blocks.size()) {
        producer_body_stmts.push_back(WrapStmtWithGuardSource(
            uniform_phase_guard_source, uniform_phase_guard,
            producer_phase_counter->Increment()));
      }
    }
    Stmt producer_loop_body =
        MergeAdjacentEquivalentIfs(SeqStmt(producer_body_stmts));
    producer_loop_body = rewrap_loop_body_lets(producer_loop_body);

    // --- Build Consumer Body ---
    Array<Stmt> consumer_body_stmts;

    // Place forward waits at first use and back-pressure arrives at last use.
    // If we cannot prove the dependency, fall back to wait-at-head /
    // arrive-at-tail.
    std::vector<bool> arrive_emitted(extractor.blocks.size(), false);
    std::vector<Stmt> normalized_waits;
    normalized_waits.reserve(extractor.blocks.size());
    for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
      ICHECK_GE(fwd_bases[ti], 0);
      PrimExpr fwd_id =
          IntImm(DataType::Int(32), fwd_bases[ti]) + consumer_stage_expr;
      if (ws_wait_stmts[ti].defined()) {
        normalized_waits.push_back(RewriteWaitBarrier(
            ws_wait_stmts[ti].value(), fwd_id, consumer_parity_expr));
      } else {
        normalized_waits.push_back(WrapStmtWithGuardSource(
            producer_issue_guard_sources[ti], producer_issue_guards[ti],
            makeParityWait(barrier_buf_, fwd_id, consumer_parity_expr)));
      }
    }
    // Emit waits / compute / arrives according to insertion points.
    for (size_t ci = 0; ci < extractor.compute_stmts.size(); ++ci) {
      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        bool is_first_in_group =
            ti == 0 || block_group[ti] != block_group[ti - 1];
        if (is_first_in_group && wait_insert_pos[ti] == static_cast<int>(ci)) {
          consumer_body_stmts.push_back(normalized_waits[ti]);
        }
      }
      if (!moved_compute_stmts[ci]) {
        consumer_body_stmts.push_back(extractor.compute_stmts[ci]);
      }
      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        bool is_last_in_group = ti + 1 == extractor.blocks.size() ||
                                block_group[ti] != block_group[ti + 1];
        if (is_last_in_group &&
            arrive_insert_pos[ti] == static_cast<int>(ci + 1)) {
          PrimExpr bp_id =
              IntImm(DataType::Int(32), bp_bases[ti]) + consumer_stage_expr;
          consumer_body_stmts.push_back(WrapStmtWithGuardSource(
              protocol_guard_sources[ti], protocol_guards[ti],
              makeArriveBarrier(barrier_buf_, bp_id)));
          arrive_emitted[ti] = true;
        }
      }
    }

    // Handle degenerate loops with no compute statements.
    if (extractor.compute_stmts.empty()) {
      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        bool is_first_in_group =
            ti == 0 || block_group[ti] != block_group[ti - 1];
        if (is_first_in_group) {
          consumer_body_stmts.push_back(normalized_waits[ti]);
        }
      }
    }

    // Emit loop-tail arrives (blocks with unknown deps or tail use).
    for (size_t ti = 0; ti < extractor.blocks.size(); ti++) {
      bool is_last_in_group = ti + 1 == extractor.blocks.size() ||
                              block_group[ti] != block_group[ti + 1];
      if (is_last_in_group && !arrive_emitted[ti] &&
          arrive_insert_pos[ti] ==
              static_cast<int>(extractor.compute_stmts.size())) {
        PrimExpr bp_id =
            IntImm(DataType::Int(32), bp_bases[ti]) + consumer_stage_expr;
        consumer_body_stmts.push_back(WrapStmtWithGuardSource(
            protocol_guard_sources[ti], protocol_guards[ti],
            makeArriveBarrier(barrier_buf_, bp_id)));
      }
    }
    // Phase counter increment for the consumer side.
    if (needs_phase_counter) {
      consumer_body_stmts.push_back(WrapStmtWithGuardSource(
          uniform_phase_guard_source, uniform_phase_guard,
          consumer_phase_counter->Increment()));
    }
    Stmt consumer_loop_body =
        MergeAdjacentEquivalentIfs(SeqStmt(consumer_body_stmts));
    consumer_loop_body = rewrap_loop_body_lets(consumer_loop_body);

    // --- Replace shared-memory stage expressions with phase counters ---
    // When the loop body is conditionally guarded, the barrier IDs already
    // use phase-counter-based stage/parity, but the shared-memory buffer
    // offsets still embed FloorMod(loop_var - loop_min, num_stages).
    // Rewrite them so that buffer staging stays in sync with barriers.
    if (needs_phase_counter) {
      producer_loop_body = StageExprReplacer::Replace(
          producer_loop_body, loop_var, loop_min, num_stages,
          producer_phase_counter->StageExpr(num_stages));
      consumer_loop_body = StageExprReplacer::Replace(
          consumer_loop_body, loop_var, loop_min, num_stages,
          consumer_phase_counter->StageExpr(num_stages));
    }

    // --- Build the loops ---
    // Remove pipeline annotations since WS handles overlap directly
    Map<String, Any> loop_annos;
    for (const auto &[key, value] : pipeline_loop->annotations) {
      if (key != "num_stages" && key != "tl_pipeline_order" &&
          key != "tl_pipeline_stage" && key != "software_pipeline_order" &&
          key != "software_pipeline_stage") {
        loop_annos.Set(key, value);
      }
    }

    Stmt producer_loop =
        For(loop_var, loop_min, loop_extent, ForKind::kSerial,
            producer_loop_body, Optional<IterVar>(), loop_annos);
    Stmt consumer_loop =
        For(loop_var, loop_min, loop_extent, ForKind::kSerial,
            consumer_loop_body, Optional<IterVar>(), loop_annos);

    // Wrap loops with phase counter allocation when needed.
    if (needs_phase_counter) {
      producer_loop = producer_phase_counter->WrapLoopWithAlloc(producer_loop);
      consumer_loop = consumer_phase_counter->WrapLoopWithAlloc(consumer_loop);
    }

    // Rewrite threadIdx.x in producer: threadIdx.x -> threadIdx.x -
    // consumer_threads Also converts `if (threadIdx.x == 0)` to `if
    // (tl_shuffle_elect(extent))`
    producer_loop = PCThreadIdxRewriter::Rewrite(
        producer_loop, thread_iv_->var,
        thread_iv_->var - consumer_thread_extent, producer_thread_extent,
        /*do_shuffle=*/true);
    consumer_loop = PCThreadIdxRewriter::Rewrite(
        consumer_loop, thread_iv_->var, thread_iv_->var, consumer_thread_extent,
        /*do_shuffle=*/true);

    // Wrap in IfThenElse: producer if threadIdx.x >= consumer_threads
    Stmt ws_body = IfThenElse(GE(thread_iv_->var, consumer_thread_extent),
                              producer_loop, consumer_loop);

    // Add warp specialization scope attribute
    Array<IntImm> ws_partition = {Downcast<IntImm>(producer_thread_extent),
                                  Downcast<IntImm>(consumer_thread_extent)};
    ws_body =
        AttrStmt(ws_partition, attr::kWarpSpecializationScope, 0, ws_body);

    // Forward barriers are producer-owned; back-pressure barriers are released
    // by the full consumer partition.
    Array<PrimExpr> barrier_arrive_counts;
    barrier_arrive_counts.reserve(total_barriers);
    if (remap_pure_tma_barriers_) {
      for (int i = 0; i < num_existing_loop_fwd_barriers; ++i) {
        barrier_arrive_counts.push_back(IntImm(DataType::Int(32), 1));
      }
      for (int i = 0; i < num_bp_barriers; ++i) {
        barrier_arrive_counts.push_back(consumer_thread_extent);
      }
      for (int i = 0; i < num_preloop_fwd_barriers; ++i) {
        barrier_arrive_counts.push_back(IntImm(DataType::Int(32), 1));
      }
    } else {
      ICHECK_EQ(mixed_fwd_arrive_counts.size(),
                static_cast<size_t>(num_total_fwd_barriers));
      for (const auto &count : mixed_fwd_arrive_counts) {
        barrier_arrive_counts.push_back(count);
      }
      for (int i = 0; i < num_bp_barriers; i++) {
        barrier_arrive_counts.push_back(consumer_thread_extent);
      }
    }
    // barrier_arrive_counts will be used for the barrier_init annotation.

    LocalLiveSet producer_live_seed =
        SeedLiveSetFromStmt(producer_loop_body, buffer_data_to_buffer);
    LocalLiveSet consumer_live_seed =
        SeedLiveSetFromStmt(consumer_loop_body, buffer_data_to_buffer);
    // Pre-loop liveness assignment must also account for variables used only in
    // the pipeline loop bounds. Otherwise scalar setup that feeds the loop
    // extent/min can be misclassified as common code and hoisted outside the
    // warp-specialized split.
    producer_live_seed.AddUses(
        LocalAccessCollector::CollectExpr(loop_min, buffer_data_to_buffer));
    producer_live_seed.AddUses(
        LocalAccessCollector::CollectExpr(loop_extent, buffer_data_to_buffer));
    consumer_live_seed.AddUses(
        LocalAccessCollector::CollectExpr(loop_min, buffer_data_to_buffer));
    consumer_live_seed.AddUses(
        LocalAccessCollector::CollectExpr(loop_extent, buffer_data_to_buffer));

    // Reconstruct block body: replace the pipeline loop with ws_body
    // and remove old barrier_init annotations / shared.barrier buffers.
    Stmt new_block_body = RebuildBlockBody(
        orig_block->body, pipeline_loop, ws_body, buffer_data_to_buffer,
        producer_live_seed, consumer_live_seed);

    // Update thread extent
    num_threads_ = consumer_thread_extent + producer_thread_extent;
    ws_transformed_ = true;
    use_full_tma_forward_barrier_protocol_ =
        old_use_full_tma_forward_barrier_protocol;
    remap_pure_tma_barriers_ = old_remap_pure_tma_barriers;
    pure_tma_preloop_fwd_base_ = old_pure_tma_preloop_fwd_base;
    pure_tma_preloop_fwd_count_ = old_pure_tma_preloop_fwd_count;
    pure_tma_preloop_fwd_cursor_ = old_pure_tma_preloop_fwd_cursor;
    current_loop_guard_bindings_ = std::move(saved_loop_guard_bindings);

    // Build the new Block and BlockRealize.
    // Add barrier buffer to alloc_buffers and barrier_init annotation.
    Array<Buffer> new_alloc_buffers = orig_block->alloc_buffers;
    // Remove any existing shared.barrier buffers from old approach
    {
      Array<Buffer> filtered;
      for (const auto &buf : new_alloc_buffers) {
        if (buf.scope() != "shared.barrier") {
          filtered.push_back(buf);
        }
      }
      new_alloc_buffers = filtered;
    }
    new_alloc_buffers.push_back(barrier_buf_);

    Map<String, Any> new_annotations = orig_block->annotations;
    // Remove any old barrier_init and build fresh
    Map<Var, Array<PrimExpr>> barrier_init_map;
    barrier_init_map.Set(barrier_buf_->data, barrier_arrive_counts);
    new_annotations.Set("barrier_init", barrier_init_map);

    Block new_block(orig_block->iter_vars, orig_block->reads,
                    orig_block->writes, orig_block->name_hint, new_block_body,
                    orig_block->init, new_alloc_buffers,
                    orig_block->match_buffers, new_annotations);
    return BlockRealize(op->iter_values, op->predicate, new_block);
  }

  // Handle ForNode with thread bindings
  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        op->thread_binding.value()->thread_tag == "threadIdx.x" &&
        !thread_iv_.defined()) {
      thread_iv_ = op->thread_binding.value();
      Optional<PrimExpr> old_num_threads = num_threads_;
      num_threads_ = std::nullopt;
      For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
      if (num_threads_.defined()) {
        PrimExpr num_threads = num_threads_.value();
        auto n = for_node.CopyOnWrite();
        n->extent = num_threads;
        IterVar new_thread_iv = n->thread_binding.value();
        new_thread_iv.CopyOnWrite()->dom =
            Range::FromMinExtent(Integer(0), num_threads);
        n->thread_binding = new_thread_iv;
      }
      num_threads_ = old_num_threads;
      thread_iv_ = {};
      return for_node;
    }

    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (for_node->kind == ForKind::kThreadBinding && thread_iv_.defined()) {
      ICHECK(for_node->thread_binding.defined());
      String thread_tag = for_node->thread_binding.value()->thread_tag;
      if (thread_tag == "threadIdx.x") {
        Var thread_v = Downcast<Var>(for_node->loop_var);
        Stmt new_body = PCThreadIdxRewriter::Rewrite(for_node->body, thread_v,
                                                     thread_iv_->var, 0);
        return new_body;
      }
    }
    return for_node;
  }

  // ---------------------------------------------------------------------------
  // Utility methods
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

  struct BufferDataAccessInfo {
    bool read{false};
    bool write{false};

    bool HasAnyAccess() const { return read || write; }
  };

  BufferDataAccessInfo
  AnalyzeBufferDataAccess(const Stmt &stmt, const Var &buffer_data,
                          const BufferDataToBufferMap &buffer_map) const {
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
          if (it != buffer_map_.end() &&
              it->second->data.same_as(buffer_data_)) {
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
        if (const auto *imm = rw_expr.as<IntImmNode>()) {
          rw_mask = static_cast<int>(imm->value);
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

  const ForNode *FindAnnotatedPipelineLoop(const Stmt &stmt) {
    if (auto *for_node = stmt.as<ForNode>()) {
      if (for_node->annotations.Get("num_stages")) {
        return for_node;
      }
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        if (auto *result = FindAnnotatedPipelineLoop(s)) {
          return result;
        }
      }
      return nullptr;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return FindAnnotatedPipelineLoop(realize->block->body);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return FindAnnotatedPipelineLoop(block->body);
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return FindAnnotatedPipelineLoop(attr->body);
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      return FindAnnotatedPipelineLoop(let_s->body);
    }
    return nullptr;
  }

  // Infer how many mbarriers are already referenced by this block body.
  // This prevents assigning back-pressure barriers that alias existing
  // forward barriers (e.g. prologue TMA copy barriers outside the pipeline).
  int InferMinRequiredBarrierCount(const Stmt &stmt) {
    class GetMbarrierMaxIdxCollector : public StmtExprVisitor {
    public:
      int max_idx{-1};
      bool has_unbounded{false};

    private:
      void VisitStmt_(const ForNode *op) final {
        // Bind loop variable range so expressions like (k + c) can be bounded.
        analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
        StmtExprVisitor::VisitStmt_(op);
      }

      void VisitExpr_(const BufferLoadNode *op) final {
        if (op->buffer.scope() == "shared.barrier" && op->indices.size() == 1) {
          auto bound = analyzer_.const_int_bound(op->indices[0]);
          if (bound->max_value != arith::ConstIntBound::kPosInf &&
              bound->max_value != arith::ConstIntBound::kNegInf) {
            max_idx = std::max(max_idx, static_cast<int>(bound->max_value));
          } else {
            has_unbounded = true;
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      arith::Analyzer analyzer_;
    };

    GetMbarrierMaxIdxCollector collector;
    collector(stmt);
    ICHECK(!collector.has_unbounded)
        << "ProducerConsumerWS: cannot infer finite upper bound for existing "
        << "mbarrier id expressions. Refusing to allocate back-pressure "
        << "barriers to avoid id overlap.";
    return collector.max_idx + 1;
  }

  int CountRewrittenPureTmaPreloopForwardPairs(const Stmt &stmt,
                                               const ForNode *target_loop) {
    if (stmt.as<ForNode>() == target_loop) {
      return 0;
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> pre_loop_stmts;
      bool found_loop = false;
      int nested_count = 0;
      for (const auto &s : seq->seq) {
        if (!found_loop && ContainsLoop(s, target_loop)) {
          nested_count =
              CountRewrittenPureTmaPreloopForwardPairs(s, target_loop);
          found_loop = true;
        } else if (!found_loop) {
          pre_loop_stmts.push_back(s);
        }
      }
      if (!found_loop) {
        return 0;
      }

      size_t movable_begin = pre_loop_stmts.size();
      while (movable_begin > 0 &&
             IsMovableConsumerPrefixStmt(pre_loop_stmts[movable_begin - 1])) {
        --movable_begin;
      }

      int local_count = 0;
      for (size_t i = 0; i + 1 < movable_begin; ++i) {
        if (ContainsTmaLoadStmt(pre_loop_stmts[i]) &&
            IsMbarrierWaitParityStmt(pre_loop_stmts[i + 1])) {
          ++local_count;
        }
      }
      return nested_count + local_count;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return CountRewrittenPureTmaPreloopForwardPairs(attr->body, target_loop);
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      return CountRewrittenPureTmaPreloopForwardPairs(let_s->body, target_loop);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return CountRewrittenPureTmaPreloopForwardPairs(realize->block->body,
                                                      target_loop);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return CountRewrittenPureTmaPreloopForwardPairs(block->body, target_loop);
    }
    return 0;
  }

  // Single source of truth for barrier/TMA control-like calls that should not
  // be moved across producer/consumer partition boundaries.
  bool IsBarrierOrTmaControlCall(const CallNode *call) {
    return call->op.same_as(mbarrier_wait_parity()) ||
           call->op.same_as(mbarrier_expect_tx()) ||
           call->op.same_as(builtin::ptx_arrive_barrier()) ||
           call->op.same_as(tl::ptx_arrive_cluster_barrier()) ||
           call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
           call->op.same_as(builtin::ptx_cp_async_barrier()) ||
           call->op.same_as(tl::ptx_cp_async_barrier_noinc()) ||
           call->op.same_as(tma_load()) ||
           call->op.same_as(tma_load_im2col()) ||
           call->op.same_as(tma_store()) ||
           call->op.same_as(tma_store_arrive()) ||
           call->op.same_as(tma_store_wait()) ||
           call->op.same_as(builtin::tvm_storage_sync());
  }

  bool IsMovableConsumerPrefixStmt(const Stmt &stmt) {
    bool has_disallowed = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (has_disallowed) {
        return;
      }
      if (auto *call = node.as<CallNode>()) {
        if (IsBarrierOrTmaControlCall(call)) {
          has_disallowed = true;
          return;
        }
      }
      if (auto *ld = node.as<BufferLoadNode>()) {
        // Only move pure local init into the consumer prefix. If a stmt reads
        // global or shared memory, the producer may also depend on its result
        // (for example a mask controlling which async copies to issue).
        if (IsSharedBuffer(ld->buffer) || IsGlobalBuffer(ld->buffer)) {
          has_disallowed = true;
          return;
        }
      }
      if (auto *st = node.as<BufferStoreNode>()) {
        if (IsSharedBuffer(st->buffer) || IsGlobalBuffer(st->buffer)) {
          has_disallowed = true;
          return;
        }
      }
    });
    return !has_disallowed;
  }

  bool IsProducerMovableLoopPrefixStmt(const Stmt &stmt) {
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

  Optional<Stmt> TryPrependToConsumerBranch(const Stmt &stmt,
                                            const Stmt &prepend_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryPrependToConsumerBranch(new_seq.back(), prepend_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryPrependToConsumerBranch(attr->body, prepend_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryPrependToConsumerBranch(let_s->body, prepend_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryPrependToConsumerBranch(realize->block->body, prepend_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryPrependToConsumerBranch(block->body, prepend_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined()) {
        return std::nullopt;
      }
      Stmt new_else = SeqStmt({prepend_stmt, if_stmt->else_case.value()});
      return IfThenElse(if_stmt->condition, if_stmt->then_case, new_else);
    }
    return std::nullopt;
  }

  Optional<Stmt> TryPrependToProducerBranch(const Stmt &stmt,
                                            const Stmt &prepend_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryPrependToProducerBranch(new_seq.back(), prepend_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryPrependToProducerBranch(attr->body, prepend_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryPrependToProducerBranch(let_s->body, prepend_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryPrependToProducerBranch(realize->block->body, prepend_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryPrependToProducerBranch(block->body, prepend_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      Stmt new_then = SeqStmt({prepend_stmt, if_stmt->then_case});
      return IfThenElse(if_stmt->condition, new_then, if_stmt->else_case);
    }
    return std::nullopt;
  }

  Optional<Stmt> TryAppendToProducerBranch(const Stmt &stmt,
                                           const Stmt &append_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryAppendToProducerBranch(new_seq.back(), append_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryAppendToProducerBranch(attr->body, append_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryAppendToProducerBranch(let_s->body, append_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryAppendToProducerBranch(realize->block->body, append_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryAppendToProducerBranch(block->body, append_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      auto nested = TryAppendToProducerBranch(if_stmt->then_case, append_stmt);
      if (nested.defined()) {
        return IfThenElse(if_stmt->condition, nested.value(),
                          if_stmt->else_case);
      }
      Stmt new_then = SeqStmt({if_stmt->then_case, append_stmt});
      return IfThenElse(if_stmt->condition, new_then, if_stmt->else_case);
    }
    return std::nullopt;
  }

  Optional<Stmt> TryAppendToConsumerBranch(const Stmt &stmt,
                                           const Stmt &append_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryAppendToConsumerBranch(new_seq.back(), append_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryAppendToConsumerBranch(attr->body, append_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryAppendToConsumerBranch(let_s->body, append_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryAppendToConsumerBranch(realize->block->body, append_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryAppendToConsumerBranch(block->body, append_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined()) {
        return std::nullopt;
      }
      Stmt new_else = SeqStmt({if_stmt->else_case.value(), append_stmt});
      return IfThenElse(if_stmt->condition, if_stmt->then_case, new_else);
    }
    return std::nullopt;
  }

  bool IsMbarrierWaitParityStmt(const Stmt &stmt) {
    return ExtractWaitBarrierId(stmt).defined();
  }

  Optional<PrimExpr> ExtractWaitBarrierId(const Stmt &stmt) {
    auto extract_from_call = [](const CallNode *call) -> Optional<PrimExpr> {
      if (!call || !call->op.same_as(mbarrier_wait_parity()) ||
          call->args.size() != 2) {
        return std::nullopt;
      }
      // Check for BufferLoad on shared.barrier scope buffer
      if (auto *bl = call->args[0].as<BufferLoadNode>()) {
        if (bl->buffer.scope() == "shared.barrier" && bl->indices.size() == 1) {
          return bl->indices[0];
        }
      }
      return std::nullopt;
    };

    if (auto *eval = stmt.as<EvaluateNode>()) {
      return extract_from_call(eval->value.as<CallNode>());
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        return ExtractWaitBarrierId(if_stmt->then_case);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return ExtractWaitBarrierId(attr->body);
    }
    if (auto *let_stmt = stmt.as<LetStmtNode>()) {
      return ExtractWaitBarrierId(let_stmt->body);
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return ExtractWaitBarrierId(seq->seq[0]);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return ExtractWaitBarrierId(block->body);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return ExtractWaitBarrierId(realize->block->body);
      }
    }
    return std::nullopt;
  }

  Stmt NormalizeForwardWaitParity(const Stmt &wait_stmt,
                                  const PrimExpr &normalized_parity) {
    auto barrier_id = ExtractWaitBarrierId(wait_stmt);
    if (!barrier_id.defined()) {
      return wait_stmt;
    }
    return makeParityWait(barrier_buf_, barrier_id.value(), normalized_parity);
  }

  bool ContainsTmaLoadStmt(const Stmt &stmt) {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (auto *call = node.as<CallNode>()) {
        if (call->op.same_as(tma_load()) ||
            call->op.same_as(tma_load_im2col())) {
          found = true;
        }
      }
    });
    return found;
  }

  bool IsThreadOnlyPredicate(const PrimExpr &expr) const {
    bool uses_thread = false;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        if (thread_iv_.defined() && var == thread_iv_->var.get()) {
          uses_thread = true;
        }
      }
    });
    return uses_thread;
  }

  Optional<PrimExpr> ExtractNonThreadProducerGuard(const Stmt &stmt) const {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return ExtractNonThreadProducerGuard(attr->body);
    }
    if (const auto *let_s = stmt.as<LetStmtNode>()) {
      return ExtractNonThreadProducerGuard(let_s->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      return ExtractNonThreadProducerGuard(realize->block->body);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return ExtractNonThreadProducerGuard(block->body);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        auto guard = ExtractNonThreadProducerGuard(s);
        if (guard.defined()) {
          return guard;
        }
      }
      return std::nullopt;
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        if (!IsThreadOnlyPredicate(if_stmt->condition)) {
          return if_stmt->condition;
        }
        return ExtractNonThreadProducerGuard(if_stmt->then_case);
      }
    }
    return std::nullopt;
  }

  PrimExpr ResolveGuardBinding(const PrimExpr &expr,
                               const VarBindingMap &bindings) const {
    if (const auto *var = expr.as<VarNode>()) {
      auto it = bindings.find(GetRef<Var>(var));
      if (it != bindings.end()) {
        return ResolveGuardBinding(it->second, bindings);
      }
    }
    if (const auto *cast = expr.as<CastNode>()) {
      return ResolveGuardBinding(cast->value, bindings);
    }
    return expr;
  }

  bool IsMaskLikeBooleanExpr(const PrimExpr &expr) const {
    PrimExpr resolved = expr;
    while (const auto *cast = resolved.as<CastNode>()) {
      resolved = cast->value;
    }
    auto is_const_bool = [](const PrimExpr &value, bool expected) {
      if (const auto *imm = value.as<IntImmNode>()) {
        return static_cast<bool>(imm->value) == expected;
      }
      return false;
    };
    if (const auto *load = resolved.as<BufferLoadNode>()) {
      return load->buffer->dtype.is_bool();
    }
    if (const auto *select = resolved.as<SelectNode>()) {
      if (is_const_bool(select->false_value, false)) {
        return IsMaskLikeBooleanExpr(select->true_value);
      }
      if (is_const_bool(select->true_value, false)) {
        return IsMaskLikeBooleanExpr(select->false_value);
      }
    }
    if (const auto *call = resolved.as<CallNode>()) {
      if (const auto *op = call->op.as<OpNode>()) {
        if (op->name == "tl.any_of" || op->name == "tl.all_of") {
          return true;
        }
      }
      if (call->op.same_as(builtin::call_extern()) && !call->args.empty()) {
        if (const auto *name = call->args[0].as<StringImmNode>()) {
          if (name->value == "tl::Any" || name->value == "tl::All") {
            return true;
          }
        }
      }
      if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3) {
        if (is_const_bool(call->args[2], false)) {
          return IsMaskLikeBooleanExpr(call->args[1]);
        }
        if (is_const_bool(call->args[1], false)) {
          return IsMaskLikeBooleanExpr(call->args[2]);
        }
      }
    }
    return false;
  }

  bool CanIssueProducerWithoutGuardImpl(const Stmt &stmt,
                                        VarBindingMap *bindings) const {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return CanIssueProducerWithoutGuardImpl(attr->body, bindings);
    }
    if (const auto *let_s = stmt.as<LetStmtNode>()) {
      bindings->emplace(let_s->var, let_s->value);
      bool result = CanIssueProducerWithoutGuardImpl(let_s->body, bindings);
      bindings->erase(let_s->var);
      return result;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      return CanIssueProducerWithoutGuardImpl(realize->block->body, bindings);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return CanIssueProducerWithoutGuardImpl(block->body, bindings);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        if (CanIssueProducerWithoutGuardImpl(s, bindings)) {
          return true;
        }
      }
      return false;
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        if (!IsThreadOnlyPredicate(if_stmt->condition)) {
          if (const auto *var = if_stmt->condition.as<VarNode>()) {
            Var cond_var = GetRef<Var>(var);
            if (!UsesVar(if_stmt->then_case, [cond_var](const VarNode *vn) {
                  return vn == cond_var.get();
                })) {
              return true;
            }
          }
          PrimExpr resolved =
              ResolveGuardBinding(if_stmt->condition, *bindings);
          return IsMaskLikeBooleanExpr(resolved);
        }
        return CanIssueProducerWithoutGuardImpl(if_stmt->then_case, bindings);
      }
    }
    return false;
  }

  bool CanIssueProducerWithoutGuard(const Stmt &stmt) const {
    VarBindingMap bindings = current_loop_guard_bindings_;
    return CanIssueProducerWithoutGuardImpl(stmt, &bindings);
  }

  Stmt StripNonThreadProducerGuard(const Stmt &stmt) const {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value,
                      StripNonThreadProducerGuard(attr->body), attr->span);
    }
    if (const auto *let_s = stmt.as<LetStmtNode>()) {
      return LetStmt(let_s->var, let_s->value,
                     StripNonThreadProducerGuard(let_s->body));
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      const Block &orig = realize->block;
      Block new_block(orig->iter_vars, orig->reads, orig->writes,
                      orig->name_hint, StripNonThreadProducerGuard(orig->body),
                      orig->init, orig->alloc_buffers, orig->match_buffers,
                      orig->annotations);
      return BlockRealize(realize->iter_values, realize->predicate, new_block);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return Block(block->iter_vars, block->reads, block->writes,
                   block->name_hint, StripNonThreadProducerGuard(block->body),
                   block->init, block->alloc_buffers, block->match_buffers,
                   block->annotations);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> stripped;
      stripped.reserve(seq->seq.size());
      for (const auto &s : seq->seq) {
        stripped.push_back(StripNonThreadProducerGuard(s));
      }
      return stripped.size() == 1 ? stripped[0] : SeqStmt(stripped, seq->span);
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        if (!IsThreadOnlyPredicate(if_stmt->condition)) {
          return StripNonThreadProducerGuard(if_stmt->then_case);
        }
        return IfThenElse(if_stmt->condition,
                          StripNonThreadProducerGuard(if_stmt->then_case),
                          std::nullopt, if_stmt->span);
      }
    }
    return stmt;
  }

  Stmt WrapStmtWithOptionalGuard(const Optional<PrimExpr> &guard,
                                 const Stmt &stmt) const {
    if (!guard.defined()) {
      return stmt;
    }
    return IfThenElse(guard.value(), stmt, std::nullopt);
  }

  Optional<Stmt> WrapStmtWithNonThreadGuardLike(const Stmt &source,
                                                const Stmt &stmt) const {
    if (const auto *attr = source.as<AttrStmtNode>()) {
      Optional<Stmt> wrapped = WrapStmtWithNonThreadGuardLike(attr->body, stmt);
      if (!wrapped.defined()) {
        return std::nullopt;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, wrapped.value(),
                      attr->span);
    }
    if (const auto *let_s = source.as<LetStmtNode>()) {
      Optional<Stmt> wrapped =
          WrapStmtWithNonThreadGuardLike(let_s->body, stmt);
      if (!wrapped.defined()) {
        return std::nullopt;
      }
      return LetStmt(let_s->var, let_s->value, wrapped.value());
    }
    if (const auto *realize = source.as<BlockRealizeNode>()) {
      Optional<Stmt> wrapped =
          WrapStmtWithNonThreadGuardLike(realize->block->body, stmt);
      if (!wrapped.defined()) {
        return std::nullopt;
      }
      const Block &orig = realize->block;
      Block new_block(orig->iter_vars, orig->reads, orig->writes,
                      orig->name_hint, wrapped.value(), orig->init,
                      orig->alloc_buffers, orig->match_buffers,
                      orig->annotations);
      return BlockRealize(realize->iter_values, realize->predicate, new_block);
    }
    if (const auto *block = source.as<BlockNode>()) {
      Optional<Stmt> wrapped =
          WrapStmtWithNonThreadGuardLike(block->body, stmt);
      if (!wrapped.defined()) {
        return std::nullopt;
      }
      return Block(block->iter_vars, block->reads, block->writes,
                   block->name_hint, wrapped.value(), block->init,
                   block->alloc_buffers, block->match_buffers,
                   block->annotations);
    }
    if (const auto *seq = source.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return WrapStmtWithNonThreadGuardLike(seq->seq[0], stmt);
      }
      return std::nullopt;
    }
    if (const auto *if_stmt = source.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        if (!IsThreadOnlyPredicate(if_stmt->condition)) {
          return IfThenElse(if_stmt->condition, stmt, std::nullopt,
                            if_stmt->span);
        }
        Optional<Stmt> wrapped =
            WrapStmtWithNonThreadGuardLike(if_stmt->then_case, stmt);
        if (!wrapped.defined()) {
          return std::nullopt;
        }
        return IfThenElse(if_stmt->condition, wrapped.value(), std::nullopt,
                          if_stmt->span);
      }
    }
    return std::nullopt;
  }

  Stmt WrapStmtWithGuardSource(const Optional<Stmt> &guard_source,
                               const Optional<PrimExpr> &guard,
                               const Stmt &stmt) const {
    if (guard_source.defined()) {
      Optional<Stmt> wrapped =
          WrapStmtWithNonThreadGuardLike(guard_source.value(), stmt);
      if (wrapped.defined()) {
        return wrapped.value();
      }
    }
    return WrapStmtWithOptionalGuard(guard, stmt);
  }

  Stmt RewriteWaitBarrier(const Stmt &wait_stmt, const PrimExpr &new_barrier_id,
                          Optional<PrimExpr> new_parity = std::nullopt) {
    class WaitBarrierRewriter : public StmtExprMutator {
    public:
      WaitBarrierRewriter(const Buffer &barrier_buf, PrimExpr barrier_id,
                          Optional<PrimExpr> parity)
          : barrier_buf_(barrier_buf), barrier_id_(std::move(barrier_id)),
            parity_(std::move(parity)) {}

      PrimExpr VisitExpr_(const CallNode *op) final {
        auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
        if (call->op.same_as(mbarrier_wait_parity()) &&
            call->args.size() == 2) {
          PrimExpr parity = parity_.defined() ? parity_.value() : call->args[1];
          return Call(call->dtype, call->op,
                      {makeGetBarrier(barrier_buf_, barrier_id_), parity},
                      call->annotations, call->span);
        }
        return call;
      }

    private:
      Buffer barrier_buf_;
      PrimExpr barrier_id_;
      Optional<PrimExpr> parity_;
    };

    return MergeAdjacentEquivalentIfs(WaitBarrierRewriter(
        barrier_buf_, new_barrier_id, std::move(new_parity))(wait_stmt));
  }

  Stmt RewriteTmaStmtBarrierIdPreserveProtocol(const Stmt &stmt,
                                               const PrimExpr &barrier_id,
                                               bool drop_arrive = false) {
    class TmaBarrierIdRewriter : public StmtExprMutator {
    public:
      TmaBarrierIdRewriter(const Buffer &barrier_buf, PrimExpr barrier_id,
                           bool drop_arrive, bool is_cluster_barrier,
                           int cluster_size)
          : barrier_buf_(barrier_buf), barrier_id_(std::move(barrier_id)),
            drop_arrive_(drop_arrive), is_cluster_barrier_(is_cluster_barrier),
            cluster_size_(cluster_size) {}

      Stmt VisitStmt_(const EvaluateNode *op) final {
        if (!is_cluster_barrier_) {
          return StmtExprMutator::VisitStmt_(op);
        }
        // For cluster barriers, intercept mbarrier_expect_tx: multiply bytes
        // by cluster_size and wrap in if (block_rank_in_cluster() == 0).
        if (const auto *call = op->value.as<CallNode>()) {
          if ((call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
               call->op.same_as(mbarrier_expect_tx())) &&
              call->args.size() == 2) {
            PrimExpr new_bytes =
                call->args[1] * IntImm(DataType::Int(32), cluster_size_);
            auto new_call =
                Call(call->dtype, call->op,
                     {makeGetBarrier(barrier_buf_, barrier_id_), new_bytes},
                     call->annotations, call->span);
            PrimExpr rank =
                Call(DataType::Int(32), tl::block_rank_in_cluster(), {});
            return IfThenElse(EQ(rank, IntImm(DataType::Int(32), 0)),
                              Evaluate(new_call), Stmt());
          }
        }
        return StmtExprMutator::VisitStmt_(op);
      }

      PrimExpr VisitExpr_(const CallNode *op) final {
        auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
        if ((call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
             call->op.same_as(mbarrier_expect_tx())) &&
            call->args.size() == 2) {
          // For non-cluster barriers, just rewrite the barrier arg.
          // Cluster barriers are handled in VisitStmt_ above.
          if (!is_cluster_barrier_) {
            return Call(
                call->dtype, call->op,
                {makeGetBarrier(barrier_buf_, barrier_id_), call->args[1]},
                call->annotations, call->span);
          }
          return call;
        }
        if (call->op.same_as(tma_load()) ||
            call->op.same_as(tma_load_im2col())) {
          bool is_1d_tma_load = false;
          if (const auto *arg0 = call->args[0].as<CallNode>()) {
            is_1d_tma_load = !arg0->op.same_as(create_tma_descriptor()) &&
                             call->op.same_as(tma_load());
          }
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(is_1d_tma_load ? 2 : 1,
                             makeGetBarrier(barrier_buf_, barrier_id_));
          // For cluster barriers, add use_2cta annotation
          if (is_cluster_barrier_) {
            Map<String, ObjectRef> new_annotations = call->annotations;
            new_annotations.Set("use_2cta", Bool(true));
            new_call->annotations = new_annotations;
          }
          return call;
        }
        if ((call->op.same_as(builtin::ptx_arrive_barrier()) ||
             call->op.same_as(tl::ptx_arrive_cluster_barrier())) &&
            !call->args.empty()) {
          if (drop_arrive_) {
            return IntImm(DataType::Int(32), 0);
          }
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(0, makeGetBarrier(barrier_buf_, barrier_id_));
          return call;
        }
        return call;
      }

    private:
      Buffer barrier_buf_;
      PrimExpr barrier_id_;
      bool drop_arrive_;
      bool is_cluster_barrier_;
      int cluster_size_;
    };

    return MergeAdjacentEquivalentIfs(
        TmaBarrierIdRewriter(barrier_buf_, barrier_id, drop_arrive,
                             is_cluster_barrier_, cluster_size_)(stmt));
  }

  Stmt MergeAdjacentEquivalentIfs(const Stmt &stmt) {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value,
                      MergeAdjacentEquivalentIfs(attr->body), attr->span);
    }
    if (const auto *let_stmt = stmt.as<LetStmtNode>()) {
      return LetStmt(let_stmt->var, let_stmt->value,
                     MergeAdjacentEquivalentIfs(let_stmt->body));
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return Block(block->iter_vars, block->reads, block->writes,
                   block->name_hint, MergeAdjacentEquivalentIfs(block->body),
                   block->init, block->alloc_buffers, block->match_buffers,
                   block->annotations);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      const Block &orig = realize->block;
      Block new_block(orig->iter_vars, orig->reads, orig->writes,
                      orig->name_hint, MergeAdjacentEquivalentIfs(orig->body),
                      orig->init, orig->alloc_buffers, orig->match_buffers,
                      orig->annotations);
      return BlockRealize(realize->iter_values, realize->predicate, new_block);
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      Optional<Stmt> else_case = std::nullopt;
      if (if_stmt->else_case.defined()) {
        else_case = MergeAdjacentEquivalentIfs(if_stmt->else_case.value());
      }
      return IfThenElse(if_stmt->condition,
                        MergeAdjacentEquivalentIfs(if_stmt->then_case),
                        else_case, if_stmt->span);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> merged;
      StructuralEqual equal;
      for (size_t i = 0; i < seq->seq.size();) {
        const auto *if0 = seq->seq[i].as<IfThenElseNode>();
        if (if0 && !if0->else_case.defined()) {
          Array<Stmt> then_stmts;
          then_stmts.push_back(if0->then_case);
          size_t j = i + 1;
          while (j < seq->seq.size()) {
            const auto *ifj = seq->seq[j].as<IfThenElseNode>();
            if (!ifj || ifj->else_case.defined() ||
                !equal(if0->condition, ifj->condition)) {
              break;
            }
            then_stmts.push_back(ifj->then_case);
            ++j;
          }
          if (then_stmts.size() == 1) {
            merged.push_back(seq->seq[i]);
          } else {
            Stmt merged_then = MergeAdjacentEquivalentIfs(
                then_stmts.size() == 1 ? then_stmts[0] : SeqStmt(then_stmts));
            merged.push_back(IfThenElse(if0->condition, merged_then,
                                        std::nullopt, if0->span));
          }
          i = j;
          continue;
        }
        merged.push_back(seq->seq[i]);
        ++i;
      }
      return merged.size() == 1 ? merged[0] : SeqStmt(merged, seq->span);
    }
    return stmt;
  }

  Stmt RewriteTmaForwardProducerStmt(const Stmt &stmt,
                                     const PrimExpr &barrier_id,
                                     bool append_arrive) {
    class TmaForwardBarrierStmtRewriter : public StmtExprMutator {
    public:
      TmaForwardBarrierStmtRewriter(const Buffer &barrier_buf,
                                    PrimExpr barrier_id,
                                    bool is_cluster_barrier, int cluster_size)
          : barrier_buf_(barrier_buf), barrier_id_(std::move(barrier_id)),
            is_cluster_barrier_(is_cluster_barrier),
            cluster_size_(cluster_size) {}

      Stmt VisitStmt_(const EvaluateNode *op) final {
        if (!is_cluster_barrier_) {
          return StmtExprMutator::VisitStmt_(op);
        }
        if (const auto *call = op->value.as<CallNode>()) {
          if ((call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
               call->op.same_as(mbarrier_expect_tx())) &&
              call->args.size() == 2) {
            PrimExpr new_bytes =
                call->args[1] * IntImm(DataType::Int(32), cluster_size_);
            auto new_call =
                Call(call->dtype, mbarrier_expect_tx(),
                     {makeGetBarrier(barrier_buf_, barrier_id_), new_bytes},
                     call->annotations, call->span);
            PrimExpr rank =
                Call(DataType::Int(32), tl::block_rank_in_cluster(), {});
            return IfThenElse(EQ(rank, IntImm(DataType::Int(32), 0)),
                              Evaluate(new_call), Stmt());
          }
        }
        return StmtExprMutator::VisitStmt_(op);
      }

      PrimExpr VisitExpr_(const CallNode *op) final {
        auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
        if ((call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
             call->op.same_as(mbarrier_expect_tx())) &&
            call->args.size() == 2) {
          if (!is_cluster_barrier_) {
            return Call(
                call->dtype, mbarrier_expect_tx(),
                {makeGetBarrier(barrier_buf_, barrier_id_), call->args[1]},
                call->annotations, call->span);
          }
          return call;
        }
        if (call->op.same_as(tma_load()) ||
            call->op.same_as(tma_load_im2col())) {
          bool is_1d_tma_load = false;
          if (const auto *arg0 = call->args[0].as<CallNode>()) {
            is_1d_tma_load = !arg0->op.same_as(create_tma_descriptor()) &&
                             call->op.same_as(tma_load());
          }
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(is_1d_tma_load ? 2 : 1,
                             makeGetBarrier(barrier_buf_, barrier_id_));
          if (is_cluster_barrier_) {
            Map<String, ObjectRef> new_annotations = call->annotations;
            new_annotations.Set("use_2cta", Bool(true));
            new_call->annotations = new_annotations;
          }
          return call;
        }
        if ((call->op.same_as(builtin::ptx_arrive_barrier()) ||
             call->op.same_as(tl::ptx_arrive_cluster_barrier())) &&
            !call->args.empty()) {
          return IntImm(DataType::Int(32), 0);
        }
        return call;
      }

    private:
      Buffer barrier_buf_;
      PrimExpr barrier_id_;
      bool is_cluster_barrier_;
      int cluster_size_;
    };

    // Rebind the producer-side barrier id and finish the stage with a normal
    // barrier arrival. Pure-TMA pipelines do not need cp.async.mbarrier.arrive.
    Stmt rewritten = MergeAdjacentEquivalentIfs(TmaForwardBarrierStmtRewriter(
        barrier_buf_, barrier_id, is_cluster_barrier_, cluster_size_)(stmt));
    if (!append_arrive) {
      return rewritten;
    }
    Optional<PrimExpr> guard = ExtractNonThreadProducerGuard(stmt);
    Stmt elect_arrive = IfThenElse(
        Call(DataType::Bool(), tl_shuffle_elect(), {producer_thread_extent_}),
        makeArriveBarrier(barrier_buf_, barrier_id), std::nullopt);
    elect_arrive = WrapStmtWithOptionalGuard(guard, elect_arrive);
    return MergeAdjacentEquivalentIfs(SeqStmt({rewritten, elect_arrive}));
  }

  Stmt RewritePureTmaForwardPairsWithFreshBarriers(const Stmt &stmt) {
    class OutsideLoopPureTmaRewriter : public StmtExprMutator {
    public:
      explicit OutsideLoopPureTmaRewriter(ProducerConsumerWSRewriter *parent)
          : parent_(parent) {}

      Stmt VisitStmt_(const SeqStmtNode *op) final {
        Array<Stmt> new_seq;
        bool changed = false;
        for (size_t i = 0; i < op->seq.size(); ++i) {
          if (i + 1 < op->seq.size() &&
              parent_->ContainsTmaLoadStmt(op->seq[i]) &&
              parent_->IsMbarrierWaitParityStmt(op->seq[i + 1])) {
            ICHECK_GE(parent_->pure_tma_preloop_fwd_base_, 0);
            ICHECK_LT(parent_->pure_tma_preloop_fwd_cursor_,
                      parent_->pure_tma_preloop_fwd_count_);
            PrimExpr barrier_id = IntImm(
                DataType::Int(32), parent_->pure_tma_preloop_fwd_base_ +
                                       parent_->pure_tma_preloop_fwd_cursor_++);
            Stmt producer_stmt = parent_->MergeAdjacentEquivalentIfs(
                parent_->RewriteTmaStmtBarrierIdPreserveProtocol(
                    StripTmaCopyWriteBufferAttr(op->seq[i]), barrier_id));
            Stmt wait_stmt =
                parent_->RewriteWaitBarrier(op->seq[i + 1], barrier_id);
            new_seq.push_back(producer_stmt);
            new_seq.push_back(wait_stmt);
            ++i;
            changed = true;
            continue;
          }
          Stmt visited = StmtExprMutator::VisitStmt(op->seq[i]);
          new_seq.push_back(visited);
          changed = changed || !visited.same_as(op->seq[i]);
        }
        if (!changed) {
          return GetRef<Stmt>(op);
        }
        return new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq);
      }

    private:
      ProducerConsumerWSRewriter *parent_;
    };

    OutsideLoopPureTmaRewriter rewriter(this);
    return rewriter(stmt);
  }

  bool IsSharedDependentConsumerPreStmt(const Stmt &stmt) {
    bool has_shared_access = false;
    bool has_control_ops = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (has_control_ops) {
        return;
      }
      if (auto *call = node.as<CallNode>()) {
        if (IsBarrierOrTmaControlCall(call)) {
          has_control_ops = true;
          return;
        }
      }
      if (auto *ld = node.as<BufferLoadNode>()) {
        if (IsSharedBuffer(ld->buffer)) {
          has_shared_access = true;
        }
      }
      if (auto *st = node.as<BufferStoreNode>()) {
        if (IsSharedBuffer(st->buffer)) {
          has_shared_access = true;
        }
      }
    });
    return has_shared_access && !has_control_ops;
  }

  bool IsBranchLocalPreStmtCandidate(const Stmt &stmt,
                                     const LocalAccessSummary &summary) {
    if (!summary.HasTrackedDefs()) {
      return false;
    }
    bool has_disallowed = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (has_disallowed) {
        return;
      }
      if (const auto *call = node.as<CallNode>()) {
        if (IsBarrierOrTmaControlCall(call)) {
          has_disallowed = true;
          return;
        }
      }
      if (const auto *ld = node.as<BufferLoadNode>()) {
        if (IsSharedBuffer(ld->buffer)) {
          has_disallowed = true;
          return;
        }
      }
      if (const auto *st = node.as<BufferStoreNode>()) {
        if (IsSharedBuffer(st->buffer) || IsGlobalBuffer(st->buffer)) {
          has_disallowed = true;
          return;
        }
      }
    });
    return !has_disallowed;
  }

  LocalLiveSet SeedLiveSetFromStmt(const Stmt &stmt,
                                   const BufferDataToBufferMap &buffer_map) {
    LocalLiveSet live;
    live.AddUses(LocalAccessCollector::Collect(stmt, buffer_map));
    return live;
  }

  /*!
   * \brief Rebuild the block body, replacing the pipeline loop with
   *        ws_body and removing old barrier_init annotations /
   *        shared.barrier buffers.
   *
   *  Statements after the pipeline loop (e.g. epilogue, store) should execute
   *  only on consumer threads. Prefer appending them into the consumer branch
   *  of the warp-specialized if/else to keep a single top-level partition.
   *  If that is not possible, fall back to an explicit consumer-thread guard.
   */
  Stmt RebuildBlockBody(const Stmt &body, const ForNode *target_loop,
                        const Stmt &ws_body,
                        const BufferDataToBufferMap &buffer_data_to_buffer,
                        const LocalLiveSet &producer_live_seed,
                        const LocalLiveSet &consumer_live_seed) {
    // If this IS the target loop, replace it
    if (body.as<ForNode>() == target_loop) {
      return ws_body;
    }

    if (auto *seq = body.as<SeqStmtNode>()) {
      Array<Stmt> new_seq;
      Array<Stmt> pre_loop_stmts;
      Array<Stmt> post_loop_stmts;
      bool found_loop = false;
      Optional<Stmt> rebuilt_loop = std::nullopt;

      for (const auto &s : seq->seq) {
        if (!found_loop && ContainsLoop(s, target_loop)) {
          // Replace the pipeline loop
          rebuilt_loop =
              RebuildBlockBody(s, target_loop, ws_body, buffer_data_to_buffer,
                               producer_live_seed, consumer_live_seed);
          found_loop = true;
        } else if (found_loop) {
          // Collect statements after the pipeline loop
          post_loop_stmts.push_back(s);
        } else {
          // Statements before the pipeline loop.
          pre_loop_stmts.push_back(s);
        }
      }

      // Move a movable suffix of pre-loop statements into consumer branch
      // (e.g. fragment initialization), keeping barriers/syncs outside.
      size_t movable_begin = pre_loop_stmts.size();
      while (movable_begin > 0 &&
             IsMovableConsumerPrefixStmt(pre_loop_stmts[movable_begin - 1])) {
        --movable_begin;
      }

      // Split non-movable pre-loop statements into:
      //   common statements kept outside the WS split
      //   producer-side async issues
      //   consumer-side waits / shared-dependent setup
      //   branch-local prefix code that is assigned by actual downstream use
      //     (producer only / consumer only / duplicated).
      //
      // We drive the branch-local assignment with a backward liveness walk over
      // local buffers / Let vars. This avoids duplicating consumer-only local
      // initialization into the producer branch.
      enum class PrefixRole : uint8_t {
        kUnknown,
        kSkip,
        kCommon,
        kProducer,
        kConsumer,
        kBoth,
        kConsumerShared,
        kSpecialTmaStart,
      };

      Array<Stmt> common_pre_stmts;
      Array<Stmt> producer_prefix_ordered_stmts;
      Array<Stmt> consumer_prefix_early_stmts;
      Array<Stmt> consumer_wait_prefix_stmts;
      Array<Stmt> consumer_shared_prefix_stmts;
      std::vector<PrefixRole> prefix_roles(movable_begin, PrefixRole::kUnknown);
      std::vector<Optional<Stmt>> rewritten_producer_prefix(movable_begin,
                                                            std::nullopt);
      std::vector<Optional<Stmt>> rewritten_consumer_wait(movable_begin,
                                                          std::nullopt);

      auto apply_to_live = [](LocalLiveSet *live,
                              const LocalAccessSummary &summary) {
        live->KillDefs(summary);
        live->AddUses(summary);
      };

      LocalLiveSet producer_live = producer_live_seed;
      LocalLiveSet consumer_live = consumer_live_seed;
      for (size_t j = movable_begin; j < pre_loop_stmts.size(); ++j) {
        consumer_live.AddUses(LocalAccessCollector::Collect(
            pre_loop_stmts[j], buffer_data_to_buffer));
      }
      for (const auto &stmt : post_loop_stmts) {
        consumer_live.AddUses(
            LocalAccessCollector::Collect(stmt, buffer_data_to_buffer));
      }

      for (int i = static_cast<int>(movable_begin) - 1; i >= 0; --i) {
        if (i > 0 && ContainsTmaLoadStmt(pre_loop_stmts[i - 1]) &&
            IsMbarrierWaitParityStmt(pre_loop_stmts[i])) {
          prefix_roles[i] = PrefixRole::kSkip;
          continue;
        }

        if (static_cast<size_t>(i + 1) < movable_begin &&
            ContainsTmaLoadStmt(pre_loop_stmts[i]) &&
            IsMbarrierWaitParityStmt(pre_loop_stmts[i + 1])) {
          Stmt producer_prefix_stmt =
              StripTmaCopyWriteBufferAttr(pre_loop_stmts[i]);
          Stmt consumer_wait_stmt = pre_loop_stmts[i + 1];
          if (remap_pure_tma_barriers_) {
            ICHECK_GE(pure_tma_preloop_fwd_base_, 0);
            ICHECK_LT(pure_tma_preloop_fwd_cursor_,
                      pure_tma_preloop_fwd_count_);
            PrimExpr barrier_id =
                IntImm(DataType::Int(32), pure_tma_preloop_fwd_base_ +
                                              pure_tma_preloop_fwd_cursor_++);
            producer_prefix_stmt =
                RewriteTmaForwardProducerStmt(producer_prefix_stmt, barrier_id,
                                              /*append_arrive=*/true);
            consumer_wait_stmt =
                RewriteWaitBarrier(consumer_wait_stmt, barrier_id);
          } else if (use_full_tma_forward_barrier_protocol_) {
            auto barrier_id = ExtractWaitBarrierId(pre_loop_stmts[i + 1]);
            ICHECK(barrier_id.defined())
                << "ProducerConsumerWS: failed to extract pre-loop TMA "
                   "forward barrier id";
            producer_prefix_stmt = RewriteTmaForwardProducerStmt(
                producer_prefix_stmt, barrier_id.value(),
                /*append_arrive=*/true);
          }
          producer_prefix_stmt =
              MergeAdjacentEquivalentIfs(producer_prefix_stmt);
          rewritten_producer_prefix[i] = producer_prefix_stmt;
          rewritten_consumer_wait[i] = consumer_wait_stmt;
          prefix_roles[i] = PrefixRole::kSpecialTmaStart;
          prefix_roles[i + 1] = PrefixRole::kSkip;
          apply_to_live(&producer_live,
                        LocalAccessCollector::Collect(producer_prefix_stmt,
                                                      buffer_data_to_buffer));
          apply_to_live(&consumer_live,
                        LocalAccessCollector::Collect(consumer_wait_stmt,
                                                      buffer_data_to_buffer));
          continue;
        }

        const Stmt &stmt = pre_loop_stmts[i];
        LocalAccessSummary summary =
            LocalAccessCollector::Collect(stmt, buffer_data_to_buffer);
        if (remap_pure_tma_barriers_ &&
            IsBranchLocalPreStmtCandidate(stmt, summary)) {
          bool producer_needed = producer_live.NeedsAnyDef(summary);
          bool consumer_needed = consumer_live.NeedsAnyDef(summary);
          if (producer_needed && consumer_needed) {
            prefix_roles[i] = PrefixRole::kBoth;
            apply_to_live(&producer_live, summary);
            apply_to_live(&consumer_live, summary);
          } else if (producer_needed) {
            prefix_roles[i] = PrefixRole::kProducer;
            apply_to_live(&producer_live, summary);
          } else if (consumer_needed) {
            prefix_roles[i] = PrefixRole::kConsumer;
            apply_to_live(&consumer_live, summary);
          } else {
            prefix_roles[i] = PrefixRole::kCommon;
            apply_to_live(&producer_live, summary);
            apply_to_live(&consumer_live, summary);
          }
          continue;
        }

        if (IsSharedDependentConsumerPreStmt(stmt)) {
          prefix_roles[i] = PrefixRole::kConsumerShared;
          apply_to_live(&consumer_live, summary);
        } else {
          prefix_roles[i] = PrefixRole::kCommon;
          apply_to_live(&producer_live, summary);
          apply_to_live(&consumer_live, summary);
        }
      }

      for (size_t i = 0; i < movable_begin; ++i) {
        switch (prefix_roles[i]) {
        case PrefixRole::kSkip:
          break;
        case PrefixRole::kCommon:
          common_pre_stmts.push_back(pre_loop_stmts[i]);
          break;
        case PrefixRole::kProducer:
          producer_prefix_ordered_stmts.push_back(pre_loop_stmts[i]);
          break;
        case PrefixRole::kConsumer:
          consumer_prefix_early_stmts.push_back(pre_loop_stmts[i]);
          break;
        case PrefixRole::kBoth:
          producer_prefix_ordered_stmts.push_back(pre_loop_stmts[i]);
          consumer_prefix_early_stmts.push_back(pre_loop_stmts[i]);
          break;
        case PrefixRole::kConsumerShared:
          consumer_shared_prefix_stmts.push_back(pre_loop_stmts[i]);
          break;
        case PrefixRole::kSpecialTmaStart:
          ICHECK(rewritten_producer_prefix[i].defined());
          ICHECK(rewritten_consumer_wait[i].defined());
          producer_prefix_ordered_stmts.push_back(
              rewritten_producer_prefix[i].value());
          consumer_wait_prefix_stmts.push_back(
              rewritten_consumer_wait[i].value());
          break;
        case PrefixRole::kUnknown:
          common_pre_stmts.push_back(pre_loop_stmts[i]);
          break;
        }
      }

      for (const auto &s : common_pre_stmts) {
        new_seq.push_back(s);
      }

      auto MakeOptionalStmt = [](const Array<Stmt> &stmts) -> Optional<Stmt> {
        if (stmts.empty()) {
          return std::nullopt;
        }
        return stmts.size() == 1 ? Optional<Stmt>(stmts[0])
                                 : Optional<Stmt>(SeqStmt(stmts));
      };

      Array<Stmt> consumer_prefix_stmts;
      for (const auto &s : consumer_prefix_early_stmts) {
        consumer_prefix_stmts.push_back(s);
      }
      // Keep pure local init before waits to delay blocking until needed.
      for (size_t j = movable_begin; j < pre_loop_stmts.size(); ++j) {
        consumer_prefix_stmts.push_back(pre_loop_stmts[j]);
      }
      for (const auto &s : consumer_wait_prefix_stmts) {
        consumer_prefix_stmts.push_back(s);
      }
      for (const auto &s : consumer_shared_prefix_stmts) {
        consumer_prefix_stmts.push_back(s);
      }
      Optional<Stmt> consumer_prefix = MakeOptionalStmt(consumer_prefix_stmts);
      Optional<Stmt> producer_prefix =
          MakeOptionalStmt(producer_prefix_ordered_stmts);

      Optional<Stmt> ws_stmt = rebuilt_loop;
      Optional<Stmt> producer_guard = std::nullopt;
      Optional<Stmt> pre_guard = std::nullopt;
      Optional<Stmt> post_guard = std::nullopt;

      // Merge TMA-issue producer prefix into producer branch.
      if (producer_prefix.defined()) {
        ICHECK(thread_iv_.defined());
        Stmt rewritten = PCThreadIdxRewriter::Rewrite(
            producer_prefix.value(), thread_iv_->var,
            thread_iv_->var - consumer_thread_extent_, producer_thread_extent_,
            /*do_shuffle=*/true);
        if (ws_stmt.defined()) {
          auto merged = TryPrependToProducerBranch(ws_stmt.value(), rewritten);
          if (merged.defined()) {
            ws_stmt = merged.value();
          } else {
            producer_guard = IfThenElse(
                GE(thread_iv_->var, consumer_thread_extent_), rewritten);
          }
        } else {
          producer_guard = IfThenElse(
              GE(thread_iv_->var, consumer_thread_extent_), rewritten);
        }
      }

      // Merge movable pre-loop suffix into consumer branch when possible.
      if (consumer_prefix.defined()) {
        if (ws_stmt.defined()) {
          auto merged = TryPrependToConsumerBranch(ws_stmt.value(),
                                                   consumer_prefix.value());
          if (merged.defined()) {
            ws_stmt = merged.value();
          } else {
            ICHECK(thread_iv_.defined());
            pre_guard = IfThenElse(LT(thread_iv_->var, consumer_thread_extent_),
                                   consumer_prefix.value());
          }
        } else {
          ICHECK(thread_iv_.defined());
          pre_guard = IfThenElse(LT(thread_iv_->var, consumer_thread_extent_),
                                 consumer_prefix.value());
        }
      }

      // Keep post-loop statements on consumer threads.
      if (!post_loop_stmts.empty()) {
        Stmt post_body = post_loop_stmts.size() == 1 ? post_loop_stmts[0]
                                                     : SeqStmt(post_loop_stmts);
        if (remap_pure_tma_barriers_) {
          // When the target loop remaps pure-TMA forward barriers to the WS
          // layout, any remaining TMA forward pairs outside that loop need
          // fresh ids as well. Otherwise a rewritten pre-loop pair can alias a
          // later consumer-only TMA loop that still uses its original id.
          post_body = RewritePureTmaForwardPairsWithFreshBarriers(post_body);
        }
        bool merged = false;
        if (ws_stmt.defined()) {
          auto merged_stmt =
              TryAppendToConsumerBranch(ws_stmt.value(), post_body);
          if (merged_stmt.defined()) {
            ws_stmt = merged_stmt.value();
            merged = true;
          }
        }
        if (!merged) {
          ICHECK(thread_iv_.defined());
          post_guard = IfThenElse(LT(thread_iv_->var, consumer_thread_extent_),
                                  post_body);
        }
      }

      if (producer_guard.defined()) {
        new_seq.push_back(producer_guard.value());
      }
      if (pre_guard.defined()) {
        new_seq.push_back(pre_guard.value());
      }
      if (ws_stmt.defined()) {
        new_seq.push_back(ws_stmt.value());
      }
      if (post_guard.defined()) {
        new_seq.push_back(post_guard.value());
      }

      if (new_seq.size() == 1)
        return new_seq[0];
      return SeqStmt(new_seq);
    }

    // Walk through wrapper nodes
    if (auto *attr = body.as<AttrStmtNode>()) {
      if (ContainsLoop(attr->body, target_loop)) {
        Stmt new_body = RebuildBlockBody(
            attr->body, target_loop, ws_body, buffer_data_to_buffer,
            producer_live_seed, consumer_live_seed);
        return AttrStmt(attr->node, attr->attr_key, attr->value, new_body);
      }
    }
    if (auto *let_s = body.as<LetStmtNode>()) {
      if (ContainsLoop(let_s->body, target_loop)) {
        Stmt new_body = RebuildBlockBody(
            let_s->body, target_loop, ws_body, buffer_data_to_buffer,
            producer_live_seed, consumer_live_seed);
        return LetStmt(let_s->var, let_s->value, new_body);
      }
    }

    // Fallback: return unchanged
    return body;
  }

  bool ContainsLoop(const Stmt &stmt, const ForNode *target) {
    if (stmt.as<ForNode>() == target)
      return true;
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        if (ContainsLoop(s, target))
          return true;
      }
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return ContainsLoop(attr->body, target);
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      return ContainsLoop(let_s->body, target);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return ContainsLoop(realize->block->body, target);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return ContainsLoop(block->body, target);
    }
    return false;
  }

  IterVar thread_iv_;
  PrimExpr
      consumer_thread_extent_; // Original thread extent (consumer warp count)
  PrimExpr producer_thread_extent_ = IntImm(DataType::Int(32), 128);
  Buffer barrier_buf_; // shared.barrier scope buffer for mbarriers
  Optional<PrimExpr> num_threads_;
  bool ws_transformed_ = false;
  bool use_full_tma_forward_barrier_protocol_ = false;
  bool remap_pure_tma_barriers_ = false;
  int pure_tma_preloop_fwd_base_ = -1;
  int pure_tma_preloop_fwd_count_ = 0;
  int pure_tma_preloop_fwd_cursor_ = 0;
  VarBindingMap current_loop_guard_bindings_;
  bool is_cluster_barrier_ = false;
  int cluster_size_ = 1;
};

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

using namespace tir::transform;

// Check only for manual warp specialization ("warp_specialize" attr).
// Unlike WarpSpecializedDetector, we do NOT skip when TMA+mbarrier are
// both present, since that is the expected input pattern for this pass.
class ManualWSDetector : public StmtExprVisitor {
public:
  static bool HasManualWS(const Stmt &stmt) {
    ManualWSDetector d;
    d.VisitStmt(stmt);
    return d.has_manual_ws_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "warp_specialize" &&
        op->value.as<IntImmNode>()->value == 1) {
      has_manual_ws_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  bool has_manual_ws_ = false;
};

tvm::transform::Pass ProducerConsumerWarpSpecialized() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool disable_warp_specialized =
        ctx->GetConfig<Bool>(kDisableWarpSpecialized, Bool(false)).value();
    if (disable_warp_specialized)
      return f;

    // Skip if user has manual warp specialization
    if (ManualWSDetector::HasManualWS(f->body))
      return f;

    return ProducerConsumerWSRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ProducerConsumerWarpSpecialized",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ProducerConsumerWarpSpecialized",
                        ProducerConsumerWarpSpecialized);
}

} // namespace tl
} // namespace tvm
