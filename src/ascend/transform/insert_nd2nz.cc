/*!
 * \file ascend/transform/insert_nd2nz.cc
 * \brief Rewrite layout-driven Ascend ND-to-NZ copies before AutoSchedule.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/builtin.h"
#include "ascend/op/nd2nz.h"
#include "ascend/op/oob_padding.h"
#include "ascend/op/utils.h"
#include "op/copy.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

namespace {

std::string Nd2NzDTypeName(DataType dtype) {
  if (dtype.is_bfloat16())
    return "bfloat16_t";
  if (dtype.is_float16())
    return "half";
  if (dtype.is_float() && dtype.bits() == 32)
    return "float";
  TVM_FFI_THROW(ValueError)
      << "Unsupported nd2nz dtype: " << dtype
      << ". The UB scatter supports only float16/bfloat16/float32; add a "
         "template specialization in src/tl_templates/ascend/nd2nz_copy.h "
         "to extend it.";
  return "";
}

int64_t RequireStaticInt(const PrimExpr &expr, const std::string &what) {
  const int64_t *value = as_const_int(expr);
  if (value == nullptr) {
    TVM_FFI_THROW(ValueError) << what << " must be static, got " << expr;
  }
  return *value;
}

Array<PrimExpr> RegionExtents(const Array<Range> &ranges) {
  Array<PrimExpr> extents;
  extents.reserve(ranges.size());
  for (const Range &range : ranges)
    extents.push_back(range->extent);
  return extents;
}

PrimExpr ExtentProduct(const Array<PrimExpr> &extents) {
  PrimExpr result = IntImm(DataType::Int(32), 1);
  for (const PrimExpr &extent : extents)
    result = result * extent;
  return result;
}

PrimExpr MakeTLAccessPtr(const Buffer &buffer, const Array<Range> &ranges,
                         const Array<PrimExpr> &pointer_extents, int rw_mask) {
  ICHECK_EQ(ranges.size(), buffer->shape.size());
  ICHECK_EQ(pointer_extents.size(), ranges.size());
  Array<PrimExpr> mins;
  mins.reserve(ranges.size());
  for (const Range &range : ranges)
    mins.push_back(range->min);
  return Call(DataType::Handle(), tl::access_ptr(),
              {BufferLoad(buffer, mins), ExtentProduct(pointer_extents),
               IntImm(DataType::Int(32), rw_mask)});
}

PrimExpr MakeTLAccessPtr(const Buffer &buffer, const Array<Range> &ranges,
                         int rw_mask) {
  return MakeTLAccessPtr(buffer, ranges, RegionExtents(ranges), rw_mask);
}

bool IsCanonicalNZLayout(const Layout &layout, const Buffer &buffer) {
  if (!layout.defined() || !buffer.defined() || buffer->shape.size() < 2)
    return false;
  int bits = buffer->dtype.bits();
  if (bits <= 0 || 256 % bits != 0)
    return false;
  if (!IsAscendFractalLayout(layout))
    return false;
  AscendFractalLayoutInfo info;
  return TryExtractAscendFractalLayout(layout, buffer, &info) &&
         info.kind == AscendFractalKind::kMajorK;
}

class Nd2NzInserter : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    arith::Analyzer analyzer;
    analyzer.z3_prover.SetRLimit(50000);
    Nd2NzInserter inserter(&analyzer);
    inserter.MarkBufferMapShapes(func);
    PrimFuncNode *write_ptr = func.CopyOnWrite();
    write_ptr->body = inserter(std::move(write_ptr->body));
    return func;
  }

private:
  using Parent = arith::IRMutatorWithAnalyzer;
  explicit Nd2NzInserter(arith::Analyzer *analyzer) : Parent(analyzer) {}

  Optional<Layout> LayoutFor(const Buffer &buffer) const {
    return ascend::FindLayoutForBuffer(layout_map_, buffer);
  }

  bool IsNZBuffer(const Buffer &buffer) const {
    Optional<Layout> layout = LayoutFor(buffer);
    return layout.defined() &&
           (IsCanonicalNZLayout(layout.value(), buffer) ||
            IsAscendCompactNZLayout(layout.value(), buffer));
  }

  bool HasPhysicalLayout(const Buffer &buffer) const {
    Optional<Layout> layout = LayoutFor(buffer);
    return layout.defined() && !layout.value().as<FragmentNode>();
  }

  bool HasCompactTrailingMatrix(const Buffer &buffer) const {
    if (buffer->strides.empty())
      return true;
    if (buffer->strides.size() != buffer->shape.size() ||
        buffer->shape.size() < 2) {
      return false;
    }
    size_t col_axis = buffer->shape.size() - 1;
    size_t row_axis = col_axis - 1;
    PrimExpr unit_stride = make_const(buffer->strides[col_axis].dtype(), 1);
    return analyzer_->CanProveEqual(buffer->strides[col_axis], unit_stride) &&
           analyzer_->CanProveEqual(buffer->strides[row_axis],
                                    buffer->shape[col_axis]);
  }

  bool HasAlignedScatterBase(const Buffer &buffer,
                             const Array<Range> &ranges) const {
    DataType index_dtype = DataType::Int(64);
    PrimExpr offset = cast(index_dtype, buffer->elem_offset);
    PrimExpr compact_stride = make_const(index_dtype, 1);
    for (size_t axis = buffer->shape.size(); axis-- > 0;) {
      PrimExpr stride = buffer->strides.empty()
                            ? compact_stride
                            : cast(index_dtype, buffer->strides[axis]);
      offset = offset + cast(index_dtype, ranges[axis]->min) * stride;
      compact_stride = compact_stride * cast(index_dtype, buffer->shape[axis]);
    }
    PrimExpr element_bytes =
        make_const(index_dtype, (buffer->dtype.bits() + 7) / 8);
    PrimExpr alignment = make_const(index_dtype, 32);
    return analyzer_->CanProveEqual(FloorMod(offset * element_bytes, alignment),
                                    make_const(index_dtype, 0));
  }

  void ValidateRegionInBounds(const Buffer &buffer, const Array<Range> &ranges,
                              const std::string &role) const {
    ICHECK_EQ(ranges.size(), buffer->shape.size());
    for (size_t axis = 0; axis < ranges.size(); ++axis) {
      const Range &range = ranges[axis];
      bool lower_in_bounds = analyzer_->CanProve(range->min >= 0);
      bool non_negative_extent = analyzer_->CanProve(range->extent >= 0);
      bool upper_in_bounds = analyzer_->CanProve(range->min + range->extent <=
                                                 buffer->shape[axis]);
      if (!lower_in_bounds || !non_negative_extent || !upper_in_bounds) {
        TVM_FFI_THROW(ValueError)
            << "UB ND->NZ " << role
            << " region must be provably in bounds on axis " << axis
            << ": min=" << range->min << ", extent=" << range->extent
            << ", buffer extent=" << buffer->shape[axis] << ".";
      }
    }
  }

  std::pair<std::string, std::string>
  ValidateScatterTemplate(const Buffer &src, const Buffer &dst, int64_t rows,
                          int64_t cols) const {
    std::string src_dtype = Nd2NzDTypeName(src->dtype);
    std::string dst_dtype = Nd2NzDTypeName(dst->dtype);
    if (rows % 16 != 0) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ scatter requires rows divisible by 16, got " << rows
          << ".";
    }
    if (src_dtype == dst_dtype) {
      int element_bytes = (src->dtype.bits() + 7) / 8;
      if (cols * element_bytes % 32 != 0) {
        TVM_FFI_THROW(ValueError)
            << "UB ND->NZ same-dtype scatter requires a 32-byte-aligned row, "
               "got cols="
            << cols << ", dtype=" << src->dtype << ".";
      }
    } else if (src_dtype == "float" && dst_dtype == "bfloat16_t") {
      if (cols % 64 != 0) {
        TVM_FFI_THROW(ValueError)
            << "UB ND->NZ float32-to-bfloat16 scatter requires cols divisible "
               "by 64, got "
            << cols << "; partial-VL conversion is not implemented.";
      }
    } else if (src_dtype == "bfloat16_t" && dst_dtype == "float") {
      if (cols % 128 != 0) {
        TVM_FFI_THROW(ValueError)
            << "UB ND->NZ bfloat16-to-float32 scatter requires cols divisible "
               "by 128, got "
            << cols << "; partial-VL conversion is not implemented.";
      }
    } else {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ scatter does not support dtype conversion "
          << src->dtype << " -> " << dst->dtype
          << "; supported conversions are float32 <-> bfloat16.";
    }
    return {std::move(src_dtype), std::move(dst_dtype)};
  }

  PrimExpr NZPitch(const Buffer &buffer, const Array<Range> &ranges) const {
    auto layout = LayoutFor(buffer);
    if (!layout.defined() || !IsNZBuffer(buffer))
      TVM_FFI_THROW(ValueError)
          << "NZ copy requires a compact NZ or MajorK layout: " << buffer->name;
    ValidateRegionInBounds(buffer, ranges, "NZ");
    for (size_t i = 0; i + 2 < ranges.size(); ++i)
      if (!analyzer_->CanProveEqual(ranges[i]->extent, 1))
        TVM_FFI_THROW(ValueError)
            << "NZ copy leading region extents must all be 1";
    int c0 = AscendC0(buffer->dtype.bits());
    if (!analyzer_->CanProveEqual(floormod(ranges.back()->min, c0), 0) ||
        !analyzer_->CanProveEqual(floormod(ranges.back()->extent, c0), 0) ||
        !analyzer_->CanProveEqual(floormod(buffer->elem_offset, c0), 0))
      TVM_FFI_THROW(ValueError) << "NZ copy requires C0-aligned column starts, "
                                   "widths, and base offsets";
    PrimExpr pitch = buffer->shape[buffer->shape.size() - 2];
    if (IsCanonicalNZLayout(layout.value(), buffer)) {
      const auto physical_shape = layout.value()->OutputShape();
      pitch = analyzer_->Simplify(physical_shape[physical_shape.size() - 3] *
                                  physical_shape[physical_shape.size() - 2]);
    }
    RequireStaticInt(pitch, "NZ storage pitch");
    return pitch;
  }

  PrimExpr NZPointer(const Buffer &buffer, const Array<Range> &ranges,
                     PrimExpr pitch, int mask) const {
    PrimExpr rows = ranges[ranges.size() - 2]->extent;
    PrimExpr cols = ranges.back()->extent;
    int c0 = AscendC0(buffer->dtype.bits());
    Array<PrimExpr> extents(ranges.size(), Integer(1));
    extents.Set(
        extents.size() - 1,
        analyzer_->Simplify(((floordiv(cols, c0) - 1) * pitch + rows) * c0));
    return MakeTLAccessPtr(buffer, ranges, extents, mask);
  }

  Stmt Intrinsic(const Op &op, Array<PrimExpr> args, bool scatter) const {
    // Each bit proves that the corresponding pointer argument's logical
    // rectangle maps to the NZ DMA blocks before LowerTileOp.
    Call call(DataType::Void(), op, args,
              {{"nd2nz_logical_regions", Integer(3)}});
    auto geometry = ParseNd2NzGeometry(call.get(), scatter);
    return geometry.Empty() ? Evaluate(Integer(0)) : Evaluate(call);
  }

  std::optional<Stmt> RewriteUBToUB(const CopyNode &copy) {
    if (!IsSharedBuffer(copy.src) || !IsSharedBuffer(copy.dst))
      return std::nullopt;
    Optional<Layout> dst_layout = LayoutFor(copy.dst);
    if (!dst_layout.defined() ||
        !IsAscendCompactNZLayout(dst_layout.value(), copy.dst)) {
      return std::nullopt;
    }
    if (IsNZBuffer(copy.src))
      return std::nullopt;
    if (HasPhysicalLayout(copy.src)) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ scatter requires an unannotated dense ND source, but "
          << copy.src->name << " has a physical layout.";
    }
    if (!HasCompactTrailingMatrix(copy.src)) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ scatter requires a compact trailing source matrix, but "
          << copy.src->name << " has explicit strides " << copy.src->strides
          << ".";
    }
    if (copy.annotations.Get("dual_dst_ctl")) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ scatter does not support dual_copy; use one T.copy "
             "per AIV partition.";
    }
    if (copy.src_range.size() < 2 || copy.dst_range.size() < 2) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ scatter requires source and destination regions with "
             "rank >= 2.";
    }
    if (copy.src_range.size() != copy.src->shape.size() ||
        copy.dst_range.size() != copy.dst->shape.size()) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ scatter requires each region rank to match its buffer "
             "rank.";
    }
    ValidateRegionInBounds(copy.src, copy.src_range, "source");
    ValidateRegionInBounds(copy.dst, copy.dst_range, "destination");
    if (!HasAlignedScatterBase(copy.src, copy.src_range)) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ scatter requires a 32-byte-aligned source address, "
             "got buffer "
          << copy.src->name << " with ranges " << copy.src_range << ".";
    }
    for (size_t i = 0; i + 2 < copy.src_range.size(); ++i) {
      if (!analyzer_->CanProveEqual(copy.src_range[i]->extent, 1)) {
        TVM_FFI_THROW(ValueError)
            << "UB ND->NZ source leading region extents must all be 1, got "
            << copy.src_range;
      }
    }
    for (size_t i = 0; i + 2 < copy.dst_range.size(); ++i) {
      if (!analyzer_->CanProveEqual(copy.dst_range[i]->extent, 1)) {
        TVM_FFI_THROW(ValueError)
            << "UB ND->NZ destination leading region extents must all be 1, "
               "got "
            << copy.dst_range;
      }
    }

    size_t src_ndim = copy.src_range.size();
    size_t dst_ndim = copy.dst_range.size();
    PrimExpr rows = copy.src_range[src_ndim - 2]->extent;
    PrimExpr cols = copy.src_range[src_ndim - 1]->extent;
    if (!analyzer_->CanProveEqual(copy.dst_range[dst_ndim - 2]->extent, rows) ||
        !analyzer_->CanProveEqual(copy.dst_range[dst_ndim - 1]->extent, cols)) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ source/destination logical regions must match.";
    }
    if (is_zero(analyzer_->Simplify(rows)) ||
        is_zero(analyzer_->Simplify(cols)))
      return Evaluate(Integer(0));
    int64_t rows_i = RequireStaticInt(rows, "UB ND->NZ scatter rows");
    int64_t cols_i = RequireStaticInt(cols, "UB ND->NZ scatter cols");
    auto [src_dtype, dst_dtype] =
        ValidateScatterTemplate(copy.src, copy.dst, rows_i, cols_i);

    if (!analyzer_->CanProveEqual(copy.src_range[src_ndim - 1]->min, 0) ||
        !analyzer_->CanProveEqual(copy.src->shape[src_ndim - 1], cols)) {
      TVM_FFI_THROW(ValueError) << "UB ND->NZ source " << copy.src->name
                                << " must cover a full contiguous row.";
    }
    PrimExpr dst_pitch = NZPitch(copy.dst, copy.dst_range);
    PrimExpr src_ptr = MakeTLAccessPtr(copy.src, copy.src_range, kAccessRead);
    PrimExpr dst_ptr =
        NZPointer(copy.dst, copy.dst_range, dst_pitch, kAccessWrite);
    return Intrinsic(ascend_nd2nz_scatter(),
                     {src_ptr, dst_ptr, rows, cols, dst_pitch,
                      StringImm(src_dtype), StringImm(dst_dtype)},
                     true);
  }

  std::optional<Stmt> RewriteUBToL1(const CopyNode &copy) {
    if (!IsSharedBuffer(copy.src) || !IsL1Buffer(copy.dst) ||
        copy.dst_range.size() < 2 || !LayoutFor(copy.dst).defined()) {
      return std::nullopt;
    }
    if (!IsCanonicalNZLayout(LayoutFor(copy.dst).value(), copy.dst))
      TVM_FFI_THROW(ValueError)
          << "UB-to-L1 NZ post-copy requires a MajorK destination layout";
    if (copy.src_range.size() < 2)
      TVM_FFI_THROW(ValueError)
          << "UB-to-L1 NZ copy requires rank >= 2 regions";
    size_t dst_ndim = copy.dst_range.size();
    size_t src_ndim = copy.src_range.size();
    PrimExpr rows = copy.dst_range[dst_ndim - 2]->extent;
    PrimExpr cols = copy.dst_range[dst_ndim - 1]->extent;
    if (!analyzer_->CanProveEqual(copy.src_range[src_ndim - 2]->extent, rows) ||
        !analyzer_->CanProveEqual(copy.src_range.back()->extent, cols))
      TVM_FFI_THROW(ValueError)
          << "NZ copy source/destination logical regions must match";
    if (is_zero(analyzer_->Simplify(rows)) ||
        is_zero(analyzer_->Simplify(cols)))
      return Evaluate(Integer(0));
    PrimExpr dst_pitch = NZPitch(copy.dst, copy.dst_range);
    PrimExpr dst_ptr =
        NZPointer(copy.dst, copy.dst_range, dst_pitch, kAccessWrite);
    std::string dst_dtype = Nd2NzDTypeName(copy.dst->dtype);
    auto make_post_copy = [&](PrimExpr source_ptr, PrimExpr src_pitch) {
      return Intrinsic(ascend_nd2nz_post_copy(),
                       {dst_ptr, source_ptr, rows, cols, src_pitch, dst_pitch,
                        StringImm(dst_dtype)},
                       false);
    };
    bool same_dtype = copy.src->dtype == copy.dst->dtype;
    if (IsNZBuffer(copy.src)) {
      if (!same_dtype)
        TVM_FFI_THROW(ValueError)
            << "NZ post-copy cannot perform a dtype conversion";
      PrimExpr src_pitch = NZPitch(copy.src, copy.src_range);
      return make_post_copy(
          NZPointer(copy.src, copy.src_range, src_pitch, kAccessRead),
          src_pitch);
    }
    if (HasPhysicalLayout(copy.src) || !HasCompactTrailingMatrix(copy.src))
      TVM_FFI_THROW(ValueError)
          << "ND->NZ requires an unannotated dense ND source";
    ValidateRegionInBounds(copy.src, copy.src_range, "source");
    for (size_t i = 0; i + 2 < src_ndim; ++i)
      if (!analyzer_->CanProveEqual(copy.src_range[i]->extent, 1))
        TVM_FFI_THROW(ValueError)
            << "ND->NZ source leading region extents must all be 1";
    if (!HasAlignedScatterBase(copy.src, copy.src_range) ||
        !analyzer_->CanProveEqual(copy.src_range.back()->min, 0) ||
        !analyzer_->CanProveEqual(copy.src->shape.back(), cols))
      TVM_FFI_THROW(ValueError) << "ND->NZ source must cover full contiguous "
                                   "rows at a 32-byte-aligned address";
    PrimExpr src_ptr = MakeTLAccessPtr(copy.src, copy.src_range, kAccessRead);
    int64_t rows_i = RequireStaticInt(rows, "ND->NZ rows");
    int64_t cols_i = RequireStaticInt(cols, "ND->NZ cols");
    if (cols_i == AscendC0(copy.dst->dtype.bits()) && same_dtype)
      return make_post_copy(src_ptr, rows);
    auto [src_dtype, checked_dst_dtype] =
        ValidateScatterTemplate(copy.src, copy.dst, rows_i, cols_i);
    // A fresh scratch buffer uses one extra row per C0 group. Existing NZ
    // buffers instead retain the pitch of their complete storage layout.
    PrimExpr scratch_pitch = rows + 1;
    PrimExpr scratch_extent = analyzer_->Simplify(scratch_pitch * cols);
    Buffer scratch = decl_buffer({scratch_extent}, copy.dst->dtype,
                                 "nz_scratch", "shared.dyn");
    ICHECK(!pending_allocs_.empty());
    pending_allocs_.back().push_back(scratch);
    Array<Range> scratch_ranges{Range::FromMinExtent(0, scratch_extent)};
    PrimExpr scratch_write =
        MakeTLAccessPtr(scratch, scratch_ranges, kAccessWrite);
    PrimExpr scratch_read =
        MakeTLAccessPtr(scratch, scratch_ranges, kAccessRead);
    Stmt scatter = Intrinsic(
        ascend_nd2nz_scatter(),
        {src_ptr, scratch_write, rows, cols, analyzer_->Simplify(scratch_pitch),
         StringImm(src_dtype), StringImm(checked_dst_dtype)},
        true);
    return SeqStmt({scatter, make_post_copy(scratch_read, analyzer_->Simplify(
                                                              scratch_pitch))});
  }

  void ValidatePostCopyPointer(const PrimExpr &pointer, int64_t span_bytes,
                               int64_t element_bytes, bool source) const {
    const auto *access = pointer.as<CallNode>();
    if (!access || !access->op.same_as(tl::access_ptr()))
      return; // Raw pointer callers own the allocation/alignment contract.
    const auto *load = access->args[0].as<BufferLoadNode>();
    if (!load)
      return;
    const Buffer &buffer = load->buffer;
    if ((source ? !IsSharedBuffer(buffer) : !IsL1Buffer(buffer)) ||
        buffer->dtype.bytes() != element_bytes)
      TVM_FFI_THROW(ValueError)
          << "NZ post-copy requires matching UB/L1 pointer element widths";
    Array<PrimExpr> indices = load->indices;
    Array<PrimExpr> shape = buffer->shape;
    Array<PrimExpr> strides = buffer->strides;
    if (auto layout = LayoutFor(buffer)) {
      indices = layout.value()->Forward(indices);
      shape = layout.value()->OutputShape();
      // LowerTileOp materializes dense physical storage. Logical strides
      // neither describe those axes nor necessarily have the same rank.
      strides = {};
    }
    PrimExpr offset = make_const(DataType::Int(64), 0);
    PrimExpr capacity = Integer(1), stride = Integer(1);
    for (size_t i = shape.size(); i-- > 0;) {
      PrimExpr step = strides.empty() ? stride : strides[i];
      offset = offset + cast(DataType::Int(64), indices[i]) * step;
      capacity = capacity + (cast(DataType::Int(64), shape[i]) - 1) * step;
      stride = stride * cast(DataType::Int(64), shape[i]);
    }
    // Alignment uses the absolute address; bounds are relative to this view.
    PrimExpr address = offset + cast(DataType::Int(64), buffer->elem_offset);
    if (!analyzer_->CanProveEqual(
            floormod(address * Integer(element_bytes), 32), 0))
      TVM_FFI_THROW(ValueError)
          << "NZ post-copy requires a 32-byte-aligned pointer: "
          << buffer->name;
    if (!analyzer_->CanProve(address >= 0 && offset >= 0 &&
                             offset + Integer(span_bytes / element_bytes) <=
                                 capacity))
      TVM_FFI_THROW(ValueError)
          << "NZ post-copy physical span exceeds pointer buffer capacity: "
          << buffer->name;
  }

  bool HasLogicalPostCopyDestination(const PrimExpr &pointer,
                                     const Nd2NzGeometry &geometry) const {
    const auto *access = pointer.as<CallNode>();
    const auto *load = access && access->op.same_as(tl::access_ptr())
                           ? access->args[0].as<BufferLoadNode>()
                           : nullptr;
    if (!load || !IsNZBuffer(load->buffer))
      return false;
    const Buffer &buffer = load->buffer;
    size_t ndim = buffer->shape.size();
    Array<Range> ranges;
    for (size_t i = 0; i < ndim; ++i) {
      PrimExpr extent = Integer(i + 2 == ndim   ? geometry.rows
                                : i + 1 == ndim ? geometry.cols
                                                : 1);
      if (!analyzer_->CanProve(load->indices[i] >= 0 &&
                               load->indices[i] + extent <= buffer->shape[i]))
        return false;
      ranges.push_back(Range::FromMinExtent(load->indices[i], extent));
    }
    int c0 = AscendC0(buffer->dtype.bits());
    if (!analyzer_->CanProveEqual(floormod(load->indices.back(), c0), 0) ||
        !analyzer_->CanProveEqual(floormod(buffer->elem_offset, c0), 0))
      return false;
    return analyzer_->CanProveEqual(NZPitch(buffer, ranges),
                                    Integer(geometry.dst_pitch));
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (inside_simt_vf_)
      return Parent::VisitStmt_(op);
    const auto *call = op->value.as<CallNode>();
    if (call && call->op.same_as(ascend_nd2nz_post_copy())) {
      auto geometry = ParseNd2NzGeometry(call, false);
      if (geometry.Empty())
        return Evaluate(Integer(0));
      ValidatePostCopyPointer(call->args[0],
                              geometry.SpanBytes(geometry.dst_pitch),
                              geometry.dst_bytes, false);
      ValidatePostCopyPointer(call->args[1],
                              geometry.SpanBytes(geometry.src_pitch),
                              geometry.src_bytes, true);
      // Explicit pitches need not match a multidimensional view's layout.
      // Prove the destination independently; an opaque source must not erase
      // the disjoint AIV row slices of a valid NZ destination.
      Call validated = GetRef<Call>(call);
      validated.CopyOnWrite()->annotations.Set(
          "nd2nz_logical_regions",
          Integer(HasLogicalPostCopyDestination(call->args[0], geometry) ? 1
                                                                         : 0));
      return Evaluate(validated);
    }
    if (call == nullptr || !IsAscendCopyCall(call))
      return Parent::VisitStmt_(op);
    AscendCopy copy(call->args, call->annotations);
    if (std::optional<Stmt> rewritten = RewriteUBToUB(*copy.get()))
      return rewritten.value();
    if (std::optional<Stmt> rewritten = RewriteUBToL1(*copy.get()))
      return rewritten.value();
    return Parent::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    LayoutMap saved_layout_map = layout_map_;
    bool saved_inside_simt_vf = inside_simt_vf_;
    if (auto layout_map_ref = op->annotations.Get(attr::kLayoutMap)) {
      if (auto map = layout_map_ref.value().as<Map<Buffer, Layout>>())
        layout_map_ = map.value();
    }
    if (op->name_hint == "SIMT_VF")
      inside_simt_vf_ = true;
    pending_allocs_.emplace_back();
    Stmt rewritten = Parent::VisitStmt_(op);
    std::vector<Buffer> pending = std::move(pending_allocs_.back());
    pending_allocs_.pop_back();
    inside_simt_vf_ = saved_inside_simt_vf;
    layout_map_ = std::move(saved_layout_map);
    if (pending.empty())
      return rewritten;
    SBlock block = Downcast<SBlock>(rewritten);
    SBlockNode *write_ptr = block.CopyOnWrite();
    for (Buffer &buffer : pending)
      write_ptr->alloc_buffers.push_back(std::move(buffer));
    return block;
  }

  LayoutMap layout_map_;
  std::vector<std::vector<Buffer>> pending_allocs_;
  bool inside_simt_vf_{false};
};

} // namespace

tvm::transform::Pass InsertNd2Nz() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const PassContext &ctx) {
    return Nd2NzInserter::Rewrite(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InsertNd2Nz", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InsertNd2Nz", InsertNd2Nz);
}

} // namespace tl
} // namespace tvm
