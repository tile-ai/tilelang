/*!
 * \file insert_oob_padding.cc
 * \brief Handle Ascend DMA copy out-of-bounds tails and GM->L1 padding before
 *        AutoSchedule.
 *
 * For GM-facing `tl.tileop.copy` paths that can cross a tensor boundary
 * (GM<->UB, GM->L1, L0C->GM), this pass clamps the out-of-bounds tail by
 * rewriting the copy's region args to semantic in-bounds ranges. It does not
 * insert copy control flow: LowerDMACopy derives both empty-tile and
 * fixed/singleton-axis in-bounds guards from the clamped ranges after
 * AutoSchedule. For a padded GM->L1 copy it appends the
 * padding fill(s) as standalone tl.fill statements.
 * For FP8 L1 data
 * consumed by an MX scale-bearing L1->L0 copy, it automatically emits
 * DeepGEMM-compatible K-tail fills after the paired data loads. AutoSchedule
 * keeps those MTE2 fills ordered before L1->L0 consumes the buffers.
 *
 * Doing this before AutoSchedule (rather than inside copy.cc's LowerTileOp
 * lowering, which runs after AutoSchedule) makes the fill visible to
 * pipe/barrier analysis: it is scheduled as its own MTE2 task (see
 * auto_schedule/task_analysis.h's ResourceAnalyzer) and participates in the
 * dependency before the MTE1 L1->L0 loads that read the padded buffer. copy.cc
 * lowering consumes the already-clamped ranges and inserts the final copy
 * guard.
 * Both stages share the clamp/fill helpers in oob_padding.h.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_map>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/oob_padding.h"
#include "ascend/op/utils.h"
#include "layout/layout.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/gemm_blockscaled.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

namespace {

// Rebuild a tl.region call from a buffer, its clamped per-axis ranges and the
// access mask, matching the builtin's argument layout:
//   region(BufferLoad(buffer, mins), access_mask, extent0, extent1, ...)
PrimExpr MakeRegionCall(const Buffer &buffer, const Array<Range> &ranges,
                        int access_mask) {
  static const Op &region_op = region();
  Array<PrimExpr> mins;
  Array<PrimExpr> args;
  for (const auto &r : ranges)
    mins.push_back(r->min);
  args.push_back(BufferLoad(buffer, mins));
  args.push_back(IntImm(DataType::Int(32), access_mask));
  for (const auto &r : ranges)
    args.push_back(r->extent);
  return Call(DataType::Handle(), region_op, args);
}

// Build the standalone padding fill statements for a padded GM->L1 copy, or an
// empty vector when the copy needs no padding. Mirrors copy.cc's
// DMAPath::kGMToL1 major-layout branch (valid extents from the clamped source,
// dst geometry from the full destination range).
std::vector<Stmt> MakeL1PaddingStmts(const AscendCopyNode *copy,
                                     const LayoutMap &layout_map,
                                     arith::Analyzer *analyzer) {
  std::vector<Stmt> stmts;
  if (!(IsGlobalBuffer(copy->src) && IsL1Buffer(copy->dst)))
    return stmts;
  if (copy->dst_range.size() < 2 || copy->src_range.size() < 2)
    return stmts;

  Optional<Layout> layout = ascend::FindLayoutForBuffer(layout_map, copy->dst);
  if (!layout.defined())
    return stmts;
  AscendFractalLayoutInfo dst_layout;
  if (!TryExtractAscendFractalLayout(layout.value(), copy->dst, &dst_layout))
    return stmts;
  // Only the major-like (data) NZ fractal needs an explicit padding fill; the
  // scale-factor (kSF) layout does not go through this path.
  if (!IsAscendMajorKind(dst_layout.kind))
    return stmts;

  // Clamp the copy's ranges the same way copy.cc lowering will, to recover the
  // valid (in-bounds) source extents that determine the padding tail size.
  ascend::BoundedDMACopyRanges bounded =
      ascend::ClampDMACopyTail(*copy, analyzer);
  // Only pad the tail that an out-of-bounds clamp actually cut off. A copy
  // whose destination L1 buffer is merely over-allocated (larger than the valid
  // source region, but the source itself is fully in bounds) is left untouched:
  // its extra columns/rows are not clamped, so no fill is emitted. When the
  // source *is* clamped (OOB), we pad each partial tail dimension (M/N and the
  // K reduction axis). A row extent that is provably either full or empty does
  // not receive a row fill.
  if (!bounded.clamped)
    return stmts;
  StridedLayout src_layout = NormalizeMTE2DLayout(
      copy->src, copy->src_range, analyzer, "Ascend GM->L1 padding");
  size_t src_inner_axis = static_cast<size_t>(src_layout.modes[0].axis);
  size_t src_row_axis = static_cast<size_t>(src_layout.modes[1].axis);
  PrimExpr valid_src_rows = bounded.src[src_row_axis]->extent;
  PrimExpr valid_src_cols = bounded.src[src_inner_axis]->extent;

  bool needs_transpose = copy->transpose != 0;
  PrimExpr valid_rows = needs_transpose ? valid_src_cols : valid_src_rows;
  PrimExpr valid_cols = needs_transpose ? valid_src_rows : valid_src_cols;

  Optional<Stmt> col_padding =
      ascend::MakeL1ColPadding(*copy, dst_layout, analyzer, valid_cols);
  if (col_padding.defined())
    stmts.push_back(col_padding.value());

  Optional<Stmt> row_padding =
      ascend::MakeL1RowPadding(*copy, dst_layout, analyzer, valid_rows);
  if (row_padding.defined())
    stmts.push_back(row_padding.value());

  return stmts;
}

using VarAxisMap =
    std::unordered_map<Var, LogicalAxis, ObjectPtrHash, ObjectPtrEqual>;

void SetKAxis(VarAxisMap *axes, const Buffer &buffer, LogicalAxis axis) {
  auto [it, inserted] = axes->emplace(buffer->data, axis);
  ICHECK(inserted || it->second == axis)
      << "Conflicting MX K-axis uses for buffer " << buffer->name;
}

// Match a dense or block-scaled GEMM tile op and parse the typed reference.
std::optional<Gemm> ParseGemmLikeCall(const CallNode *op) {
  static const Op &gemm_op = Op::Get("tl.tileop.gemm");
  static const Op &gemm_blockscaled_op = Op::Get("tl.tileop.gemm_blockscaled");
  if (op->op.same_as(gemm_blockscaled_op))
    return GemmBlockScaled(op->args, op->annotations);
  if (op->op.same_as(gemm_op))
    return Gemm(op->args, op->annotations);
  return std::nullopt;
}

bool IsBlockScaledGemm(const Gemm &gemm) {
  return gemm->IsInstance<GemmBlockScaledNode>();
}

// First recover the semantic K axis of each blockscaled GEMM operand. This is
// deliberately independent of the inferred layout tag: a physical [K, MN]
// buffer can carry the same 2-D map tag as a [MN, K] buffer while the GEMM
// transpose flag gives the unambiguous operand meaning.
class MxOperandKAxisCollector : public StmtExprVisitor {
public:
  static VarAxisMap Collect(const Stmt &stmt) {
    MxOperandKAxisCollector collector;
    collector(stmt);
    return std::move(collector.result_);
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (std::optional<Gemm> gemm_opt = ParseGemmLikeCall(op)) {
      const Gemm &gemm = *gemm_opt;
      if (IsBlockScaledGemm(gemm)) {
        SetKAxis(&result_, gemm->a_,
                 gemm->transA_ ? LogicalAxis::kRow : LogicalAxis::kCol);
        SetKAxis(&result_, gemm->b_,
                 gemm->transB_ ? LogicalAxis::kCol : LogicalAxis::kRow);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  VarAxisMap result_;
};

// Then propagate the GEMM K axis backwards through scale-bearing L1->L0
// copies. A user-requested L1->L0 transpose swaps the source axis. Any earlier
// GM->L1 transpose is already reflected in the L1 destination region.
class MxL1KAxisCollector : public StmtExprVisitor {
public:
  static VarAxisMap Collect(const Stmt &stmt) {
    VarAxisMap operand_axes = MxOperandKAxisCollector::Collect(stmt);
    MxL1KAxisCollector collector(operand_axes);
    collector(stmt);
    return std::move(collector.result_);
  }

private:
  explicit MxL1KAxisCollector(const VarAxisMap &operand_axes)
      : operand_axes_(operand_axes) {}

  void VisitExpr_(const CallNode *op) final {
    if (IsAscendCopyCall(op)) {
      AscendCopy copy(op->args, op->annotations);
      auto it = operand_axes_.find(copy->dst->data);
      // Only L1->L0 DATA loads feeding a block-scaled GEMM propagate; the
      // standalone MX scale loads target the shared.l0a.sf/.l0b.sf handle
      // scopes, which the L0 data predicates exclude.
      if (IsL1Buffer(copy->src) &&
          (IsL0ABuffer(copy->dst) || IsL0BBuffer(copy->dst)) &&
          it != operand_axes_.end()) {
        LogicalAxis src_k_axis = it->second;
        if (copy->transpose != 0) {
          src_k_axis = src_k_axis == LogicalAxis::kRow ? LogicalAxis::kCol
                                                       : LogicalAxis::kRow;
        }
        SetKAxis(&result_, copy->src, src_k_axis);
      }
    } else if (std::optional<Gemm> gemm_opt = ParseGemmLikeCall(op)) {
      const Gemm &gemm = *gemm_opt;
      if (IsL1Buffer(gemm->a_)) {
        auto it = operand_axes_.find(gemm->a_->data);
        if (it != operand_axes_.end())
          SetKAxis(&result_, gemm->a_, it->second);
      }
      if (IsL1Buffer(gemm->b_)) {
        auto it = operand_axes_.find(gemm->b_->data);
        if (it != operand_axes_.end())
          SetKAxis(&result_, gemm->b_, it->second);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  const VarAxisMap &operand_axes_;
  VarAxisMap result_;
};

struct MxKTailFill {
  Stmt stmt;
  PrimExpr condition;
};

// Build DeepGEMM's fill_tail_block for an actual-sized FP8 GM->L1 copy. The
// destination allocation retains the full static tile geometry while the copy
// ranges carry actual_mn/actual_k. LayoutInference has already tagged the L1
// buffer as MajorK or MajorMN, so the physical fill can be selected without
// exposing layout details on the Python surface.
std::optional<MxKTailFill> MakeMxKTailFill(const Copy &copy,
                                           LogicalAxis semantic_k_axis,
                                           const LayoutMap &layout_map,
                                           arith::Analyzer *analyzer) {
  if (!IsGlobalBuffer(copy->src) || !IsL1Buffer(copy->dst)) {
    return std::nullopt;
  }
  // FP4 has a different logical-K-to-byte mapping. Keep this automatic path
  // scoped to dense FP8; existing full-region OOB padding still handles FP4.
  // Fill geometry is defined entirely by the destination L1 region and its
  // semantic K axis, so a GM->L1 transpose does not change it.
  if (copy->dst->dtype.lanes() != 1 || !copy->dst->dtype.is_float8() ||
      copy->dst_range.size() != copy->dst->shape.size() ||
      copy->dst_range.size() < 2) {
    return std::nullopt;
  }

  Optional<Layout> layout = ascend::FindLayoutForBuffer(layout_map, copy->dst);
  if (!layout.defined())
    return std::nullopt;
  AscendFractalLayoutInfo info;
  if (!TryExtractAscendFractalLayout(layout.value(), copy->dst, &info) ||
      !IsAscendMajorKind(info.kind)) {
    return std::nullopt;
  }

  size_t ndim = copy->dst_range.size();
  size_t k_axis = semantic_k_axis == LogicalAxis::kRow ? ndim - 2 : ndim - 1;
  size_t mn_axis = semantic_k_axis == LogicalAxis::kRow ? ndim - 1 : ndim - 2;
  PrimExpr zero = make_zero(copy->dst_range[k_axis]->extent.dtype());
  if (!analyzer->CanProveEqual(copy->dst_range[k_axis]->min, zero) ||
      !analyzer->CanProveEqual(copy->dst_range[mn_axis]->min, zero)) {
    return std::nullopt;
  }

  PrimExpr actual_k = analyzer->Simplify(copy->dst_range[k_axis]->extent);
  PrimExpr shape_k = copy->dst->shape[k_axis];
  PrimExpr shape_mn = copy->dst->shape[mn_axis];
  auto C = [&](int64_t value) { return make_const(actual_k.dtype(), value); };
  PrimExpr k64 = analyzer->Simplify(FloorDiv(actual_k + C(63), C(64)) * C(64));
  ICHECK(analyzer->CanProveEqual(FloorMod(shape_k, C(64)), C(0)))
      << "Ascend MX FP8 K-tail fill requires the L1 K allocation extent to "
         "be divisible by 64, got "
      << shape_k << " for buffer " << copy->dst->name << " with shape "
      << copy->dst->shape << " and layout kind " << static_cast<int>(info.kind);

  bool is_k_major = semantic_k_axis == info.c0_axis;

  Array<Range> fill_region = copy->dst_range;
  PrimExpr needs_fill;
  if (is_k_major) {
    PrimExpr nz32 =
        analyzer->Simplify(FloorDiv(actual_k + C(31), C(32)) * C(32));
    needs_fill = analyzer->Simplify(nz32 < k64);
    fill_region.Set(k_axis,
                    Range::FromMinExtent(nz32, analyzer->Simplify(k64 - nz32)));
    fill_region.Set(mn_axis, Range::FromMinExtent(C(0), shape_mn));
  } else {
    ICHECK(analyzer->CanProveEqual(FloorMod(shape_mn, info.c0), C(0)))
        << "Ascend MX FP8 K-tail fill requires a MajorMN extent divisible by "
           "the FP8 C0 width, got MN="
        << shape_mn << " and C0=" << info.c0;
    PrimExpr tail_k = analyzer->Simplify(k64 - actual_k);
    needs_fill = analyzer->Simplify(actual_k < k64);
    fill_region.Set(k_axis, Range::FromMinExtent(actual_k, tail_k));
    fill_region.Set(mn_axis, Range::FromMinExtent(C(0), shape_mn));
  }

  if (is_zero(needs_fill) || analyzer->CanProve(Not(needs_fill)))
    return std::nullopt;

  Stmt fill =
      ascend::MakeL1Fill(copy->dst, fill_region, make_zero(copy->dst->dtype));
  return MxKTailFill{fill, needs_fill};
}

class OOBPaddingInserter : public arith::IRMutatorWithAnalyzer {
public:
  static Stmt Apply(Stmt stmt, arith::Analyzer *analyzer) {
    OOBPaddingInserter inserter(analyzer);
    return inserter(std::move(stmt));
  }

private:
  using Parent = arith::IRMutatorWithAnalyzer;
  explicit OOBPaddingInserter(arith::Analyzer *analyzer) : Parent(analyzer) {}

  bool IsMxL1Data(const Buffer &buffer) const {
    return mx_l1_k_axes_.count(buffer->data) != 0;
  }

  LogicalAxis GetMxKAxis(const Buffer &buffer) const {
    auto it = mx_l1_k_axes_.find(buffer->data);
    ICHECK(it != mx_l1_k_axes_.end())
        << "Missing collected MX K axis for L1 buffer " << buffer->name;
    return it->second;
  }

  Optional<Copy> AsMxGMToL1Copy(const Stmt &stmt) const {
    const auto *evaluate = stmt.as<EvaluateNode>();
    if (!evaluate)
      return std::nullopt;
    const auto *call = evaluate->value.as<CallNode>();
    if (!call || !IsAscendCopyCall(call))
      return std::nullopt;
    AscendCopy copy(call->args, call->annotations);
    if (!IsGlobalBuffer(copy->src) || !IsL1Buffer(copy->dst) ||
        !IsMxL1Data(copy->dst)) {
      return std::nullopt;
    }
    return copy;
  }

  void FlushMxKTailFills(Array<Stmt> *seq, std::vector<MxKTailFill> *pending) {
    if (pending->empty())
      return;

    // These fill predicates are the only control flow intentionally created by
    // this pre-AutoSchedule pass. They guard standalone MTE2 fill tasks, not
    // copies. Each affected L1 buffer has already received its unconditional
    // GM->L1 write in `seq`, so the predicates cannot change its write-first
    // classification. Keep them visible here so AutoSchedule can order each
    // possible fill against the later L1 consumer.
    PrimExpr any_fill = const_false();
    for (const MxKTailFill &fill : *pending) {
      any_fill = analyzer_->Simplify(Or(any_fill, fill.condition));
    }

    Array<Stmt> body;
    for (const MxKTailFill &fill : *pending) {
      Stmt fill_stmt = fill.stmt;
      if (!analyzer_->CanProveEqual(fill.condition, any_fill)) {
        fill_stmt = IfThenElse(fill.condition, fill_stmt);
      }
      body.push_back(fill_stmt);
    }
    Stmt fill_group = body.size() == 1 ? body[0] : Stmt(SeqStmt(body));
    if (!is_one(any_fill) && !analyzer_->CanProve(any_fill)) {
      fill_group = IfThenElse(any_fill, fill_group);
    }
    seq->push_back(fill_group);
    pending->clear();
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    LayoutMap saved = layout_map_;
    VarAxisMap saved_mx_l1_k_axes = mx_l1_k_axes_;
    if (auto layout_map_ref = op->annotations.Get(attr::kLayoutMap)) {
      if (auto map = layout_map_ref.value().as<Map<Buffer, Layout>>()) {
        layout_map_ = map.value();
      }
    }
    VarAxisMap local_mx_l1_k_axes = MxL1KAxisCollector::Collect(op->body);
    for (const auto &[data, axis] : local_mx_l1_k_axes) {
      auto [it, inserted] = mx_l1_k_axes_.emplace(data, axis);
      ICHECK(inserted || it->second == axis)
          << "Conflicting nested MX K-axis uses for L1 buffer data";
    }
    // Parent binds this block's iter vars into the analyzer before recursing.
    Stmt stmt = Parent::VisitStmt_(op);
    layout_map_ = saved;
    mx_l1_k_axes_ = std::move(saved_mx_l1_k_axes);
    return stmt;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    std::vector<MxKTailFill> pending;
    for (const Stmt &original : op->seq) {
      Optional<Copy> mx_copy = AsMxGMToL1Copy(original);
      if (!mx_copy.defined()) {
        FlushMxKTailFills(&seq, &pending);
        seq.push_back(Parent::VisitStmt(original));
        continue;
      }

      bool saved_defer = defer_mx_fill_;
      defer_mx_fill_ = true;
      seq.push_back(Parent::VisitStmt(original));
      defer_mx_fill_ = saved_defer;
      Copy mx_copy_value = mx_copy.value();
      std::optional<MxKTailFill> fill =
          MakeMxKTailFill(mx_copy_value, GetMxKAxis(mx_copy_value->dst),
                          layout_map_, analyzer_);
      if (fill.has_value())
        pending.push_back(std::move(fill.value()));
    }
    FlushMxKTailFills(&seq, &pending);
    return SeqStmt(seq, op->span);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !IsAscendCopyCall(call)) {
      return Parent::VisitStmt_(op);
    }

    AscendCopy copy_obj(call->args, call->annotations);
    const AscendCopyNode *copy = copy_obj.get();

    ascend::DMAPath dma_path = ascend::GetDMAPath(copy->src, copy->dst);
    // Only GM-facing paths need semantic range clamping. All DMA paths receive
    // their empty-tile guard later in LowerDMACopy.
    bool clampable = dma_path == ascend::DMAPath::kGMToUB ||
                     dma_path == ascend::DMAPath::kUBToGM ||
                     dma_path == ascend::DMAPath::kGMToL1 ||
                     dma_path == ascend::DMAPath::kL0CToGM;
    if (!clampable) {
      return Parent::VisitStmt_(op);
    }

    // Clamp the copy's tail to the in-bounds region. The traversal analyzer
    // has the enclosing loop / thread / block-iter ranges bound (via
    // IRMutatorWithAnalyzer), matching copy.cc's lowering-time analyzer.
    ascend::BoundedDMACopyRanges bounded =
        ascend::ClampDMACopyTail(*copy, analyzer_);

    // Rebuild the copy call with clamped ranges when clamping shrank them.
    Stmt copy_stmt = GetRef<Stmt>(op);
    if (bounded.clamped) {
      int src_mask = NormalizeToAccessRegion(call->args[0]).access_mask;
      int dst_mask = NormalizeToAccessRegion(call->args[1]).access_mask;
      Array<PrimExpr> new_args = call->args;
      new_args.Set(0, MakeRegionCall(copy->src, bounded.src, src_mask));
      new_args.Set(1, MakeRegionCall(copy->dst, bounded.dst, dst_mask));
      copy_stmt =
          Evaluate(Call(call->dtype, call->op, new_args, call->annotations));
    }

    // Emit standalone semantic fills for a padded GM->L1 copy. MX K-tail fills
    // are grouped by VisitStmt_(SeqStmtNode*) so paired A/B loads are followed
    // by both fills. AutoSchedule sees their exact tl.fill write regions and
    // orders them before the MTE1 consumers; LowerTileOp later converts them
    // to ascend_fill_l1, which codegen emits as asc_fill_l1. The fallback here
    // handles a copy that is not
    // inside a SeqStmt.
    std::vector<Stmt> fills = MakeL1PaddingStmts(copy, layout_map_, analyzer_);
    std::vector<MxKTailFill> mx_fills;
    if (!defer_mx_fill_ && IsMxL1Data(copy->dst)) {
      std::optional<MxKTailFill> mx_fill = MakeMxKTailFill(
          copy_obj, GetMxKAxis(copy->dst), layout_map_, analyzer_);
      if (mx_fill.has_value())
        mx_fills.push_back(std::move(mx_fill.value()));
    }

    if (fills.empty() && mx_fills.empty()) {
      return copy_stmt;
    }
    Array<Stmt> seq;
    seq.push_back(copy_stmt);
    for (auto &f : fills)
      seq.push_back(f);
    FlushMxKTailFills(&seq, &mx_fills);
    return SeqStmt(seq);
  }

  LayoutMap layout_map_;
  VarAxisMap mx_l1_k_axes_;
  bool defer_mx_fill_{false};
};

} // namespace

tvm::transform::Pass AscendInsertOOBPadding() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const PassContext &ctx) {
    arith::Analyzer analyzer;
    // Dynamic Persistent-loop tail proofs can exceed the default 10k budget.
    analyzer.z3_prover.SetRLimit(50000);
    PrimFuncNode *write_ptr = func.CopyOnWrite();
    write_ptr->body =
        OOBPaddingInserter::Apply(std::move(write_ptr->body), &analyzer);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendInsertOOBPadding", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendInsertOOBPadding",
                        AscendInsertOOBPadding);
}

} // namespace tl
} // namespace tvm
