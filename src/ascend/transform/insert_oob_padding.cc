/*!
 * \file ascend/transform/insert_oob_padding.cc
 * \brief Clamp native DMA regions and expose L1 padding before scheduling.
 *
 * Copies remain unconditional semantic tile operations. Only MX tail fills
 * acquire predicates here; DMA lowering derives copy guards from the final
 * possibly-empty regions after scheduling.
 */
#include "ascend/ir/block.h"
#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/oob_padding.h"
#include "ascend/op/utils.h"
#include "ir/alias_analysis.h"
#include "ir/analyzer.h"
#include "ir/annotations.h"
#include "op/gemm_blockscaled.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>

#include "ir/stmt_functor.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_map>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;
using namespace prim;

// Access masks belong to the tile operation. Replace only its native semantic
// region; tensor origins and physical strides remain on the TensorVar.
Expr RewriteRegion(const Expr &original, const Array<Range> &ranges) {
  auto region = NormalizeToTensorRegion(original);
  return TensorRegion(region->source, ranges, region->ty, original->loc);
}

std::vector<Stmt> MakeL1PaddingStmts(const AscendCopyNode &copy,
                                     const LayoutMap &layout_map,
                                     sym::AnalyzerObj *analyzer) {
  std::vector<Stmt> result;
  if (!(IsGlobalBuffer(copy.src) && IsL1Buffer(copy.dst)) ||
      copy.dst_range.size() < 2 || copy.src_range.size() < 2)
    return result;
  auto layout = ascend::FindLayoutForBuffer(layout_map, copy.dst);
  AscendFractalLayoutInfo info;
  if (!layout || !TryExtractAscendFractalLayout(*layout, copy.dst, &info) ||
      !IsAscendMajorKind(info.kind))
    return result;
  auto bounded = ascend::ClampDMACopyTail(copy, analyzer);
  // Overallocated in-bounds copies do not authorize an extra fill.
  if (!bounded.clamped)
    return result;
  auto source = NormalizeMTE2DLayout(copy.src, copy.src_range, analyzer,
                                     "Ascend GM->L1 padding");
  PrimExpr rows = bounded.src[source.modes[1].axis]->extent;
  PrimExpr cols = bounded.src[source.modes[0].axis]->extent;
  if (copy.transpose != 0)
    std::swap(rows, cols);
  if (auto fill = ascend::MakeL1ColPadding(copy, info, analyzer, cols))
    result.push_back(*fill);
  if (auto fill = ascend::MakeL1RowPadding(copy, info, analyzer, rows))
    result.push_back(*fill);
  return result;
}

using AxisMap =
    std::unordered_map<Expr, LogicalAxis, ObjectPtrHash, ObjectPtrEqual>;

void SetKAxis(AxisMap *axes, const TensorAliasGroups &aliases,
              const TensorVar &tensor, LogicalAxis axis) {
  auto [it, inserted] = axes->emplace(aliases.Identity(tensor), axis);
  TVM_FFI_CHECK(inserted || it->second == axis, ValueError)
      << "Conflicting MX K-axis uses for tensor " << tensor->name;
}

std::optional<Gemm> ParseGemm(const CallObj *op) {
  if (!op->op.same_as(Gemm::Get()) && !op->op.same_as(GemmBlockScaled::Get()))
    return std::nullopt;
  auto decoded = DecodeTileOpArguments(op);
  if (op->op.same_as(GemmBlockScaled::Get()))
    return GemmBlockScaled(decoded.operands, decoded.annotations);
  return Gemm(decoded.operands, decoded.annotations);
}

// Recover K from the GEMM semantics, not from a physical layout tag. Then
// propagate it back through L1->L0 data copies, accounting for transpose.
AxisMap CollectMxL1Axes(const Stmt &body, const TensorAliasGroups &aliases) {
  AxisMap operands, result;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        auto gemm = ParseGemm(call);
        if (gemm && (*gemm)->IsInstance<GemmBlockScaledNode>()) {
          SetKAxis(&operands, aliases, (*gemm)->a_,
                   (*gemm)->transA_ ? LogicalAxis::kRow : LogicalAxis::kCol);
          SetKAxis(&operands, aliases, (*gemm)->b_,
                   (*gemm)->transB_ ? LogicalAxis::kCol : LogicalAxis::kRow);
        }
        return ffi::WalkResult::Advance();
      });
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        if (IsAscendCopyCall(call)) {
          auto decoded = DecodeTileOpArguments(call);
          AscendCopy copy(decoded.operands, decoded.annotations);
          auto found = operands.find(aliases.Identity(copy->dst));
          if (IsL1Buffer(copy->src) &&
              (IsL0ABuffer(copy->dst) || IsL0BBuffer(copy->dst)) &&
              found != operands.end()) {
            LogicalAxis axis = found->second;
            if (copy->transpose != 0)
              axis = axis == LogicalAxis::kRow ? LogicalAxis::kCol
                                               : LogicalAxis::kRow;
            SetKAxis(&result, aliases, copy->src, axis);
          }
        } else if (auto gemm = ParseGemm(call)) {
          for (const TensorVar &operand : {(*gemm)->a_, (*gemm)->b_}) {
            auto found = operands.find(aliases.Identity(operand));
            if (IsL1Buffer(operand) && found != operands.end())
              SetKAxis(&result, aliases, operand, found->second);
          }
        }
        return ffi::WalkResult::Advance();
      });
  return result;
}

struct MxKTailFill {
  Stmt stmt;
  PrimExpr condition;
};

std::optional<MxKTailFill> MakeMxKTailFill(const Copy &copy, LogicalAxis k_axis,
                                           const LayoutMap &layouts,
                                           sym::AnalyzerObj *analyzer) {
  auto dtype = copy->dst.dtype();
  if (!IsGlobalBuffer(copy->src) || !IsL1Buffer(copy->dst) ||
      dtype.lanes() != 1 ||
      !dtype.MatchesCode(kDLFloat8_e3m4, kDLFloat8_e4m3, kDLFloat8_e4m3b11fnuz,
                         kDLFloat8_e4m3fn, kDLFloat8_e4m3fnuz, kDLFloat8_e5m2,
                         kDLFloat8_e5m2fnuz, kDLFloat8_e8m0fnu,
                         kDLFloat4_e2m1fn) ||
      copy->dst_range.size() != copy->dst.shape().size() ||
      copy->dst_range.size() < 2)
    return std::nullopt;
  auto layout = ascend::FindLayoutForBuffer(layouts, copy->dst);
  AscendFractalLayoutInfo info;
  if (!layout || !TryExtractAscendFractalLayout(*layout, copy->dst, &info) ||
      !IsAscendMajorKind(info.kind))
    return std::nullopt;

  size_t rank = copy->dst_range.size();
  size_t k = k_axis == LogicalAxis::kRow ? rank - 2 : rank - 1;
  size_t mn = k_axis == LogicalAxis::kRow ? rank - 1 : rank - 2;
  if (!analyzer->CanProveEqual(copy->dst_range[k]->begin, 0) ||
      !analyzer->CanProveEqual(copy->dst_range[mn]->begin, 0))
    return std::nullopt;
  PrimExpr actual_k = analyzer->Simplify(copy->dst_range[k]->extent);
  PrimExpr shape_k = copy->dst.shape()[k], shape_mn = copy->dst.shape()[mn];
  auto C = [&](int64_t value) { return MakeConst(actual_k.ty(), value); };
  PrimExpr k64 = analyzer->Simplify(FloorDiv(actual_k + C(63), C(64)) * C(64));
  TVM_FFI_CHECK(analyzer->CanProveEqual(FloorMod(shape_k, C(64)), C(0)),
                ValueError)
      << "Ascend MX K-tail fill requires the L1 K allocation extent to "
         "be divisible by 64, got "
      << shape_k << " for tensor " << copy->dst->name;

  Array<Range> fill_region = copy->dst_range;
  PrimExpr needs_fill{UnsafeInit{}};
  if (k_axis == info.c0_axis) {
    // GM->L1 hardware clears a whole C0 (64 logical FP4 values). Fill only
    // the rest of the 64-element scale group that hardware did not write.
    int64_t c0 = 256 / dtype.bits();
    PrimExpr padded_k =
        analyzer->Simplify(FloorDiv(actual_k + C(c0 - 1), C(c0)) * C(c0));
    needs_fill = analyzer->Simplify(padded_k < k64);
    fill_region.Set(
        k, Range::FromMinExtent(padded_k, analyzer->Simplify(k64 - padded_k)));
  } else {
    TVM_FFI_CHECK(analyzer->CanProveEqual(FloorMod(shape_mn, info.c0), C(0)),
                  ValueError)
        << "Ascend MX K-tail fill requires a MajorMN extent divisible by "
           "the operand C0 width, got MN="
        << shape_mn << " and C0=" << info.c0;
    needs_fill = analyzer->Simplify(actual_k < k64);
    fill_region.Set(
        k, Range::FromMinExtent(actual_k, analyzer->Simplify(k64 - actual_k)));
  }
  fill_region.Set(mn, Range::FromMinExtent(C(0), shape_mn));
  if (IsZero(needs_fill) || analyzer->CanProve(Not(needs_fill)))
    return std::nullopt;
  return MxKTailFill{
      ascend::MakeL1Fill(copy->dst, fill_region, MakeConst(dtype, 0)),
      needs_fill};
}

class OOBPaddingInserter : public StmtExprMutatorWithAnalyzer {
public:
  OOBPaddingInserter(sym::AnalyzerObj *analyzer,
                     const TensorAliasGroups &aliases, AxisMap axes)
      : StmtExprMutatorWithAnalyzer(analyzer), aliases_(aliases),
        mx_axes_(std::move(axes)) {}
  using StmtExprMutatorWithAnalyzer::Mutate_;

  tirx::Function Apply(tirx::Function function) {
    MarkTensorParamShapes(function);
    function.CopyOnWrite()->body =
        Mutate(function->body).ValueOrUnchanged(function->body);
    CheckWellFormed(function);
    return function;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    auto saved = layouts_;
    Array<Stmt> result;
    std::vector<MxKTailFill> pending;
    for (const Stmt &stmt : op->seq) {
      auto copy = AsMxGMToL1Copy(stmt);
      if (!copy)
        FlushFills(&result, &pending);
      bool saved_defer = defer_mx_fill_;
      defer_mx_fill_ = copy.has_value();
      result.push_back(Mutate(stmt, mode).ValueOrUnchanged(stmt));
      defer_mx_fill_ = saved_defer;
      if (copy) {
        auto fill = MakeMxKTailFill(*copy, AxisFor((*copy)->dst), layouts_,
                                    GetSyncedAnalyzer());
        if (fill)
          pending.push_back(*fill);
      }
    }
    FlushFills(&result, &pending);
    layouts_ = saved;
    return SeqStmt(result, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    auto saved = layouts_;
    if (ascend::IsAscendBlock(op)) {
      RecordMetadata(ascend::AscendBlockAnnotations(op));
    } else if (op->op->name == "tl.attr") {
      auto fields = ReadAnnotations(op);
      String key = fields.at("key").cast<String>();
      if (key == "tl.kernel")
        RecordMetadata(fields.at("value").cast<Map<String, Any>>());
      else
        RecordMetadata({{key, fields.at("value")}});
    }
    auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    layouts_ = saved;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (IsAnnotationCall(call)) {
      RecordMetadata(ReadAnnotations(call));
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    }
    if (!call || !IsAscendCopyCall(call))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    auto decoded = DecodeTileOpArguments(call);
    AscendCopy copy(decoded.operands, decoded.annotations);
    auto path = ascend::GetDMAPath(copy->src, copy->dst);
    if (path != ascend::DMAPath::kGMToUB && path != ascend::DMAPath::kUBToGM &&
        path != ascend::DMAPath::kGMToL1 && path != ascend::DMAPath::kL0CToGM)
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);

    auto bounded = ascend::ClampDMACopyTail(*copy.get(), GetSyncedAnalyzer());
    Stmt rewritten = GetRef<Stmt>(op);
    if (bounded.clamped) {
      decoded.operands.Set(0, RewriteRegion(decoded.operands[0], bounded.src));
      decoded.operands.Set(1, RewriteRegion(decoded.operands[1], bounded.dst));
      rewritten = Evaluate(
          MakeAnnotatedCall(call->ty, call->op.as_or_throw<Op>(),
                            decoded.operands, decoded.annotations, call->loc),
          op->loc);
    }
    auto fills = MakeL1PaddingStmts(*copy.get(), layouts_, GetSyncedAnalyzer());
    std::vector<MxKTailFill> mx_fills;
    if (!defer_mx_fill_ && IsMxData(copy->dst)) {
      if (auto fill = MakeMxKTailFill(copy, AxisFor(copy->dst), layouts_,
                                      GetSyncedAnalyzer()))
        mx_fills.push_back(*fill);
    }
    if (fills.empty() && mx_fills.empty())
      return rewritten;
    Array<Stmt> result{rewritten};
    for (const Stmt &fill : fills)
      result.push_back(fill);
    FlushFills(&result, &mx_fills);
    return SeqStmt(result, op->loc);
  }

private:
  void RecordMetadata(const Map<String, Any> &metadata) {
    if (auto value = metadata.Get("layout_map")) {
      for (const auto &[tensor, layout] : value.value().cast<LayoutMap>()) {
        if (auto previous = layouts_.Get(tensor))
          TVM_FFI_CHECK((*previous)->IsEqual(layout.get()), ValueError)
              << "Conflicting layouts for " << tensor->name;
        layouts_.Set(tensor, layout);
      }
    }
  }

  bool IsMxData(const TensorVar &tensor) const {
    return mx_axes_.count(aliases_.Identity(tensor));
  }

  LogicalAxis AxisFor(const TensorVar &tensor) const {
    return mx_axes_.at(aliases_.Identity(tensor));
  }

  std::optional<Copy> AsMxGMToL1Copy(const Stmt &stmt) const {
    const auto *evaluate = stmt.as<EvaluateObj>();
    const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
    if (!call || !IsAscendCopyCall(call))
      return std::nullopt;
    auto decoded = DecodeTileOpArguments(call);
    AscendCopy copy(decoded.operands, decoded.annotations);
    if (!IsGlobalBuffer(copy->src) || !IsL1Buffer(copy->dst) ||
        !IsMxData(copy->dst))
      return std::nullopt;
    return copy;
  }

  void FlushFills(Array<Stmt> *result, std::vector<MxKTailFill> *pending) {
    if (pending->empty())
      return;
    PrimExpr any_fill = IntImm(PrimType::Bool(), 0);
    for (const auto &fill : *pending)
      any_fill = GetSyncedAnalyzer()->Simplify(Or(any_fill, fill.condition));
    Array<Stmt> body;
    for (const auto &fill : *pending) {
      Stmt stmt = fill.stmt;
      if (!GetSyncedAnalyzer()->CanProveEqual(fill.condition, any_fill))
        stmt = If(fill.condition, SeqStmt(stmt));
      body.push_back(stmt);
    }
    Stmt group = SeqStmt(body);
    if (!IsOne(any_fill) && !GetSyncedAnalyzer()->CanProve(any_fill))
      group = If(any_fill, SeqStmt(group));
    result->push_back(group);
    pending->clear();
  }

  const TensorAliasGroups &aliases_;
  AxisMap mx_axes_;
  LayoutMap layouts_;
  bool defer_mx_fill_{false};
};
} // namespace

namespace transform {
tvm::transform::Pass AscendInsertOOBPadding() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
          return function;
        auto analyzer = CreateTileLangAnalyzer();
        TensorAliasGroups aliases;
        aliases.Collect(function);
        auto rewriter = ffi::make_object<OOBPaddingInserter>(
            analyzer.get(), aliases, CollectMxL1Axes(function->body, aliases));
        return rewriter->Apply(std::move(function));
      },
      0, "tl.AscendInsertOOBPadding");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendInsertOOBPadding",
                                   AscendInsertOOBPadding);
}
} // namespace transform
} // namespace tvm::tl
