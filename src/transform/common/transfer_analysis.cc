#include "transfer_analysis.h"

#include <cmath>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>
#include <utility>

#include "op/copy.h"
#include "op/operator.h"
#include "op/parallel.h"
#include "op/utils.h"

namespace tvm {
namespace tl {
namespace {

using namespace tirx;

bool IsZeroFill(const PrimExpr &expr) {
  if (const auto *broadcast = expr.as<BroadcastNode>()) {
    return IsZeroFill(broadcast->value);
  }
  if (const auto *value = expr.as<FloatImmNode>()) {
    // Hardware zero fill produces +0, not the distinct bit pattern of -0.
    return value->value == 0.0 && !std::signbit(value->value);
  }
  if (const auto *value = expr.as<IntImmNode>()) {
    return value->value == 0;
  }
  return false;
}

bool IsGlobalLike(const Buffer &buffer) {
  return IsGlobalBuffer(buffer) || buffer.scope().empty();
}

class TransferAnalyzer : public StmtExprVisitor {
public:
  static TransferSummary Analyze(const Stmt &stmt, const Target &target) {
    TransferAnalyzer analyzer(target);
    analyzer(stmt);
    return analyzer.summary_;
  }

private:
  explicit TransferAnalyzer(Target target) : target_(std::move(target)) {}

  void RecordTransfer(const Buffer &src, const Buffer &dst,
                      bool converts_value) {
    summary_.byte_preserving &= !converts_value;
    summary_.has_global_to_shared |= IsGlobalLike(src) && IsSharedBuffer(dst);
    summary_.only_transfers &=
        IsGlobalLike(src) &&
        (IsSharedBuffer(dst) || IsLocalBuffer(dst, /*allow_var=*/true));
  }

  void RecordRead(const Buffer &buffer) {
    summary_.reads_global |= IsGlobalLike(buffer);
    summary_.reads_shared_or_local |=
        IsSharedBuffer(buffer) || IsLocalBuffer(buffer, /*allow_var=*/true);
  }

  void RecordWrite(const Buffer &buffer) {
    summary_.writes_shared |= IsSharedBuffer(buffer);
    summary_.writes_other |= !IsSharedBuffer(buffer);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    RecordWrite(op->buffer);
    auto value = AnalyzeTransferValue(op->value);
    if (value) {
      RecordTransfer(value->source->buffer, op->buffer, value->converts_value);
    } else {
      summary_.only_transfers = false;
    }
    // Include address and predicate dependencies, not just the loaded value.
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    RecordRead(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    summary_.execution_guard = true;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    summary_.execution_guard |= !is_one(op->predicate);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(ffi::GetRef<Call>(op));
        tile_op.defined()) {
      if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
        VisitStmt(parallel->GetRoot());
        return;
      }
      AccessRegions access = tile_op->GetAccessRegions();
      for (const auto &region : access.reads) {
        RecordRead(region->buffer);
      }
      for (const auto &region : access.writes) {
        RecordWrite(region->buffer);
      }
      if (const auto *copy = tile_op.as<CopyNode>()) {
        RecordTransfer(copy->src, copy->dst,
                       copy->src->dtype != copy->dst->dtype);
      } else if (const auto *im2col = tile_op.as<Im2ColOpNode>();
                 im2col && Im2ColUsesTMA(target_)) {
        RecordTransfer(im2col->src_, im2col->dst_, false);
        summary_.tma = true;
      } else {
        summary_.only_transfers = false;
        summary_.opaque_effect = true;
      }
      return;
    }
    // Inspect the call's own effect, not SideEffect(call), which would fold
    // ordinary BufferLoad arguments into it. Unknown read-state calls also
    // have dependencies that this transfer summary cannot resolve.
    static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    auto callee = op->op.as<Op>();
    summary_.opaque_effect |= !callee || !effects.count(callee.value()) ||
                              effects[callee.value()]->value >
                                  static_cast<int>(CallEffectKind::kPure);
    StmtExprVisitor::VisitExpr_(op);
  }

  Target target_;
  TransferSummary summary_;
};

} // namespace

std::optional<TransferValue> AnalyzeTransferValue(const PrimExpr &value) {
  using namespace tirx;
  if (const auto *load = value.as<BufferLoadNode>()) {
    // A masked BufferLoad has a separate contract; do not invent zero fill.
    if (load->predicate.defined()) {
      return std::nullopt;
    }
    return TransferValue{ffi::GetRef<BufferLoad>(load), std::nullopt, false};
  }
  if (const auto *cast = value.as<CastNode>()) {
    auto transfer = AnalyzeTransferValue(cast->value);
    if (transfer) {
      transfer->converts_value |= cast->dtype != cast->value.dtype();
    }
    return transfer;
  }
  const auto *call = value.as<CallNode>();
  if (!call || !call->op.same_as(builtin::if_then_else()) ||
      !IsZeroFill(call->args[2])) {
    return std::nullopt;
  }
  auto transfer = AnalyzeTransferValue(call->args[1]);
  if (transfer) {
    transfer->zero_fill_predicate =
        transfer->zero_fill_predicate.defined()
            ? And(call->args[0], transfer->zero_fill_predicate.value())
            : call->args[0];
  }
  return transfer;
}

TransferSummary AnalyzeTransfers(const tirx::Stmt &stmt, const Target &target) {
  return TransferAnalyzer::Analyze(stmt, target);
}

} // namespace tl
} // namespace tvm
