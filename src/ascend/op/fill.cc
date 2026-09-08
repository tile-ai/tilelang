/*!
 * \file tl/ascend/op/fill.cc
 * \brief Ascend implementation for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/oob_padding.h"
#include "backend/common/target_utils.h"
#include "op/builtin.h"
#include "op/utils.h"

#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

namespace {

using namespace tirx;

bool MatchAscendFillTarget(Target target) { return TargetIsAscend(target); }

size_t LogicalAxisIndex(size_t ndim, LogicalAxis axis) {
  ICHECK_GE(ndim, 2U);
  return axis == LogicalAxis::kRow ? ndim - 2 : ndim - 1;
}

PrimExpr MakeL1BasePtr(const FillNode &op) {
  auto I = [](int64_t value) { return make_const(DataType::Int(32), value); };
  size_t ndim = op.region.size();
  Array<PrimExpr> indices;
  for (size_t i = 0; i < ndim; ++i) {
    indices.push_back(i + 2 < ndim ? op.region[i]->min
                                   : make_zero(op.region[i]->min.dtype()));
  }
  PrimExpr matrix_elements = op.dst->shape[ndim - 2] * op.dst->shape[ndim - 1];
  return Call(DataType::Handle(), tl::access_ptr(),
              {BufferLoad(op.dst, indices), matrix_elements, I(kAccessWrite)});
}

Stmt LowerL1Fill(const FillNode &op, const LowerArgs &lower_args,
                 arith::Analyzer *analyzer) {
  ICHECK(IsL1Buffer(op.dst));
  ICHECK_EQ(op.region.size(), op.dst->shape.size());
  ICHECK_GE(op.region.size(), 2U)
      << "Ascend L1 fill requires a rank >= 2 destination, got " << op.dst->name
      << " with shape " << op.dst->shape;

  Optional<Layout> layout =
      ascend::FindLayoutForBuffer(lower_args.layout_map, op.dst);
  ICHECK(layout.defined())
      << "Ascend L1 fill requires a canonical fractal layout on buffer "
      << op.dst->name;
  AscendFractalLayoutInfo info;
  ICHECK(TryExtractAscendFractalLayout(layout.value(), op.dst, &info) &&
         IsAscendMajorKind(info.kind))
      << "Ascend L1 fill requires a MajorK or MajorMN fractal layout on "
         "buffer "
      << op.dst->name;

  for (size_t i = 0; i + 2 < op.region.size(); ++i) {
    ICHECK(analyzer->CanProveEqual(op.region[i]->extent,
                                   make_const(op.region[i]->extent.dtype(), 1)))
        << "Ascend L1 fill supports one matrix at a time; leading axis " << i
        << " of buffer " << op.dst->name << " has extent "
        << op.region[i]->extent;
  }

  const Range &c0_range =
      op.region[LogicalAxisIndex(op.region.size(), info.c0_axis)];
  const Range &row16_range =
      op.region[LogicalAxisIndex(op.region.size(), info.row16_axis)];
  auto is_aligned = [&](const PrimExpr &value, const PrimExpr &alignment) {
    return analyzer->CanProveEqual(
        FloorMod(value, cast(value.dtype(), alignment)),
        make_zero(value.dtype()));
  };
  ICHECK(is_aligned(c0_range->min, info.c0) &&
         is_aligned(c0_range->extent, info.c0))
      << "Ascend L1 fill requires a C0-aligned region on buffer "
      << op.dst->name << ", got " << c0_range << " with C0=" << info.c0;

  PrimExpr fill_value = analyzer->Simplify(op.value);
  ICHECK(analyzer->CanProveEqual(fill_value, make_zero(fill_value.dtype())))
      << "Ascend L1 fill currently supports only zero, got " << op.value;

  int bits = op.dst->dtype.bits();
  ICHECK_EQ(op.dst->dtype.lanes(), 1)
      << "Ascend L1 fill requires a scalar destination dtype, got "
      << op.dst->dtype;
  ICHECK(bits > 0 && bits <= 32 && (bits <= 16 || bits == 32))
      << "Ascend L1 fill supports destination element widths up to 16 bits "
         "or exactly 32 bits, got "
      << op.dst->dtype;

  PrimExpr dst_pitch_blocks = analyzer->Simplify(info.outer1 * info.row_frac);
  PrimExpr c0_start = analyzer->Simplify(FloorDiv(c0_range->min, info.c0));
  PrimExpr repeat_times =
      analyzer->Simplify(FloorDiv(c0_range->extent, info.c0));
  PrimExpr row_start = row16_range->min;
  PrimExpr block_num = row16_range->extent;
  PrimExpr dst_gap = analyzer->Simplify(dst_pitch_blocks - block_num);
  PrimExpr byte_offset =
      analyzer->Simplify((c0_start * dst_pitch_blocks + row_start) *
                         make_const(c0_range->min.dtype(), 32));
  int fill_word_bits = bits <= 16 ? 16 : 32;

  return Evaluate(
      Call(DataType::Void(), ascend_fill_l1(),
           {MakeL1BasePtr(op), byte_offset, make_const(DataType::UInt(32), 0),
            repeat_times, block_num, dst_gap,
            make_const(DataType::Int(32), fill_word_bits)}));
}

Stmt LowerAscendFill(const FillNode &op, const LowerArgs &lower_args,
                     arith::Analyzer *analyzer) {
  if (IsL1Buffer(op.dst)) {
    return LowerL1Fill(op, lower_args, analyzer);
  }
  return backend::Fill::Lower(op, lower_args, analyzer);
}

bool RegisterAscendFill() {
  RegisterFillImpl(FillImpl{
      "ascend.Fill",
      MatchAscendFillTarget,
      LowerAscendFill,
  });
  return true;
}

const bool ascend_fill_registered = RegisterAscendFill();

} // namespace

} // namespace tl
} // namespace tvm
