/*!
 * \file tl/cpu/op/scan.cc
 * \brief Serial CPU lowering for inclusive scan operators.
 */

#include "op/scan.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/runtime/logging.h>

#include "backend/common/target_utils.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace cpu {

enum class ScanKind { kSum, kMax };

bool IsSupportedScanBuffer(const Buffer &buffer) {
  return IsLocalBuffer(buffer, /*allow_var=*/true) || IsSharedBuffer(buffer) ||
         IsGlobalBuffer(buffer);
}

template <typename ScanOpNode>
Stmt LowerSerialScan(const ScanOpNode &op, const LowerArgs &lower_args,
                     arith::Analyzer *analyzer, ScanKind kind,
                     const char *op_name) {
  (void)analyzer;
  ICHECK(IsSupportedScanBuffer(op.src))
      << "CPU " << op_name << " does not support source scope `"
      << op.src.scope() << "`; expected local, shared, or global.";
  ICHECK(IsSupportedScanBuffer(op.dst))
      << "CPU " << op_name << " does not support destination scope `"
      << op.dst.scope() << "`; expected local, shared, or global.";

  const auto &src_region = op.srcRegion_->region;
  const auto &dst_region = op.dstRegion_->region;
  int ndim = static_cast<int>(src_region.size());
  ICHECK_GT(ndim, 0) << "CPU " << op_name << " requires a non-scalar region";
  ICHECK_EQ(ndim, static_cast<int>(dst_region.size()))
      << "CPU " << op_name << " requires matching source/destination ranks";
  ICHECK_GE(op.dim, 0) << "CPU " << op_name << " requires a normalized dim";
  ICHECK_LT(op.dim, ndim) << "CPU " << op_name << " dim " << op.dim
                          << " is out of range for rank " << ndim;

  bool identical_region = op.src.same_as(op.dst);
  for (int i = 0; i < ndim; ++i) {
    ICHECK(StructuralEqual()(src_region[i]->extent, dst_region[i]->extent))
        << "CPU " << op_name
        << " requires matching source/destination region extents; dim " << i
        << " differs: " << src_region[i]->extent << " vs "
        << dst_region[i]->extent;
    identical_region =
        identical_region &&
        StructuralEqual()(src_region[i]->min, dst_region[i]->min) &&
        StructuralEqual()(src_region[i]->extent, dst_region[i]->extent);
  }
  ICHECK(!op.src->data.same_as(op.dst->data) || identical_region)
      << "CPU " << op_name
      << " does not support non-identical source/destination regions or "
         "different physical layouts on the same buffer yet; use distinct "
         "buffers or an identical in-place region.";

  (void)lower_args;
  Buffer src_buffer = op.src;
  Buffer dst_buffer = op.dst;
  DataType dtype = dst_buffer->dtype;
  Buffer accumulator = decl_buffer({1}, dtype, "scan_acc", "local");

  Array<Var> outer_vars;
  outer_vars.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    outer_vars.push_back(
        Var("scan_i" + std::to_string(i), src_region[i]->extent->dtype));
  }

  auto make_indices = [&](const BufferRegion &region,
                          const PrimExpr &scan_index) {
    Array<PrimExpr> indices;
    indices.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      PrimExpr offset = i == op.dim ? scan_index : outer_vars[i];
      indices.push_back(region->region[i]->min + offset);
    }
    return indices;
  };
  auto make_src_value = [&](const BufferRegion &region,
                            const PrimExpr &scan_index) {
    PrimExpr value = BufferLoad(src_buffer, make_indices(region, scan_index));
    return value->dtype == dtype ? value : Cast(dtype, value);
  };

  PrimExpr extent = src_region[op.dim]->extent;
  PrimExpr one = make_const(extent->dtype, 1);
  PrimExpr two = make_const(extent->dtype, 2);
  PrimExpr first = op.reverse ? extent - one : make_zero(extent->dtype);
  Array<PrimExpr> first_dst_indices = make_indices(op.dstRegion_, first);

  Array<Stmt> statements;
  statements.push_back(
      BufferStore(accumulator, make_src_value(op.srcRegion_, first), {0}));
  statements.push_back(
      BufferStore(dst_buffer, BufferLoad(accumulator, {0}), first_dst_indices));

  Var offset("scan_offset", extent->dtype);
  PrimExpr current = op.reverse ? extent - two - offset : offset + one;
  PrimExpr value = make_src_value(op.srcRegion_, current);
  PrimExpr previous = BufferLoad(accumulator, {0});
  PrimExpr updated = kind == ScanKind::kSum ? PrimExpr(previous + value)
                                            : PrimExpr(Max(previous, value));
  Stmt body = SeqStmt({BufferStore(accumulator, updated, {0}),
                       BufferStore(dst_buffer, BufferLoad(accumulator, {0}),
                                   make_indices(op.dstRegion_, current))});
  statements.push_back(For(offset, make_zero(extent->dtype), extent - one,
                           ForKind::kSerial, body, std::nullopt));

  Stmt scan = SeqStmt::Flatten(statements);
  for (int i = ndim - 1; i >= 0; --i) {
    if (i == op.dim) {
      continue;
    }
    scan = For(outer_vars[i], make_zero(src_region[i]->extent->dtype),
               src_region[i]->extent, ForKind::kSerial, scan, std::nullopt);
  }
  scan = IfThenElse(extent > make_zero(extent->dtype), scan);
  return SeqStmt({AllocBuffer(accumulator), scan});
}

struct CumSum {
  static Stmt Lower(const CumSumOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    return LowerSerialScan(op, lower_args, analyzer, ScanKind::kSum, "cumsum");
  }
};

struct CumMax {
  static Stmt Lower(const CumMaxOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    return LowerSerialScan(op, lower_args, analyzer, ScanKind::kMax, "cummax");
  }
};

} // namespace cpu

namespace {

bool MatchCPUScanTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUScan() {
  RegisterCumSumImpl(CumSumImpl{
      "cpu.CumSum",
      MatchCPUScanTarget,
      cpu::CumSum::Lower,
  });
  RegisterCumMaxImpl(CumMaxImpl{
      "cpu.CumMax",
      MatchCPUScanTarget,
      cpu::CumMax::Lower,
  });
  return true;
}

const bool cpu_scan_registered = RegisterCPUScan();

} // namespace

} // namespace tl
} // namespace tvm
