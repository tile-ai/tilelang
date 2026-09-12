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
  Layout expected = MakeAscendNZLayout(buffer);
  return layout->IsEqual(expected.get());
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

  Array<PrimExpr> ValidatePrepackedNZSource(const CopyNode &copy, PrimExpr rows,
                                            PrimExpr cols,
                                            PrimExpr full_rows) const {
    RequireStaticInt(rows, "NZ post-copy rows");
    RequireStaticInt(cols, "NZ post-copy cols");
    RequireStaticInt(full_rows, "NZ post-copy full_rows");
    ICHECK_EQ(copy.src_range.size(), copy.src->shape.size());
    ICHECK_GE(copy.src_range.size(), 2U);
    for (size_t i = 0; i + 2 < copy.src_range.size(); ++i) {
      if (!analyzer_->CanProveEqual(copy.src_range[i]->extent, 1)) {
        TVM_FFI_THROW(ValueError)
            << "NZ source " << copy.src->name
            << " leading region extents must all be 1, got " << copy.src_range;
      }
    }
    size_t ndim = copy.src_range.size();
    bool mins_are_zero =
        analyzer_->CanProveEqual(copy.src_range[ndim - 2]->min, 0) &&
        analyzer_->CanProveEqual(copy.src_range[ndim - 1]->min, 0);
    bool extents_are_full =
        analyzer_->CanProveEqual(copy.src->shape[ndim - 2],
                                 copy.src_range[ndim - 2]->extent + 1) &&
        analyzer_->CanProveEqual(copy.src->shape[ndim - 1],
                                 copy.src_range[ndim - 1]->extent);
    if (!mins_are_zero || !extents_are_full) {
      TVM_FFI_THROW(ValueError)
          << "NZ source " << copy.src->name
          << " trailing region must cover the full padded matrix: ranges="
          << copy.src_range << ", buffer shape=" << copy.src->shape;
    }
    if (!analyzer_->CanProveEqual(copy.src_range[ndim - 2]->extent, rows) ||
        !analyzer_->CanProveEqual(copy.src_range[ndim - 1]->extent, cols)) {
      TVM_FFI_THROW(ValueError)
          << "NZ source " << copy.src->name
          << " region must match post-copy geometry " << rows << "x" << cols;
    }
    Array<PrimExpr> pointer_extents = RegionExtents(copy.src_range);
    pointer_extents.Set(ndim - 2, copy.src->shape[ndim - 2]);
    return pointer_extents;
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
    int64_t rows_i = RequireStaticInt(rows, "UB ND->NZ scatter rows");
    int64_t cols_i = RequireStaticInt(cols, "UB ND->NZ scatter cols");
    auto [src_dtype, dst_dtype] =
        ValidateScatterTemplate(copy.src, copy.dst, rows_i, cols_i);

    if (!analyzer_->CanProveEqual(copy.src_range[src_ndim - 1]->min, 0) ||
        !analyzer_->CanProveEqual(copy.src->shape[src_ndim - 1], cols)) {
      TVM_FFI_THROW(ValueError) << "UB ND->NZ source " << copy.src->name
                                << " must cover a full contiguous row.";
    }
    if (!analyzer_->CanProveEqual(copy.dst_range[dst_ndim - 2]->min, 0) ||
        !analyzer_->CanProveEqual(copy.dst_range[dst_ndim - 1]->min, 0)) {
      TVM_FFI_THROW(ValueError) << "UB ND->NZ destination " << copy.dst->name
                                << " must start at trailing offset [0, 0].";
    }
    if (!analyzer_->CanProveEqual(copy.dst->shape[dst_ndim - 2], rows + 1) ||
        !analyzer_->CanProveEqual(copy.dst->shape[dst_ndim - 1], cols)) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ compact destination " << copy.dst->name
          << " must reserve exactly one padding row.";
    }

    PrimExpr src_ptr = MakeTLAccessPtr(copy.src, copy.src_range, kAccessRead);
    Array<PrimExpr> dst_extents = RegionExtents(copy.dst_range);
    dst_extents.Set(dst_ndim - 2, copy.dst->shape[dst_ndim - 2]);
    PrimExpr dst_ptr =
        MakeTLAccessPtr(copy.dst, copy.dst_range, dst_extents, kAccessWrite);
    Array<PrimExpr> args{src_ptr,
                         dst_ptr,
                         rows,
                         cols,
                         StringImm(dst_dtype),
                         StringImm(src_dtype)};
    return Evaluate(Call(DataType::Handle(), ascend_nd2nz_scatter(), args));
  }

  std::optional<Stmt> RewriteUBToL1(const CopyNode &copy) {
    if (!IsSharedBuffer(copy.src) || !IsL1Buffer(copy.dst) ||
        copy.dst_range.size() < 2 || !LayoutFor(copy.dst).defined()) {
      return std::nullopt;
    }
    bool src_is_nz = IsNZBuffer(copy.src);

    size_t dst_ndim = copy.dst_range.size();
    PrimExpr rows = copy.dst_range[dst_ndim - 2]->extent;
    PrimExpr cols = copy.dst_range[dst_ndim - 1]->extent;
    PrimExpr full_rows = copy.dst->shape[copy.dst->shape.size() - 2];
    std::string src_dtype = Nd2NzDTypeName(copy.src->dtype);
    std::string dst_dtype = Nd2NzDTypeName(copy.dst->dtype);

    Array<PrimExpr> src_extents =
        src_is_nz ? ValidatePrepackedNZSource(copy, rows, cols, full_rows)
                  : RegionExtents(copy.src_range);
    PrimExpr src_ptr =
        MakeTLAccessPtr(copy.src, copy.src_range, src_extents, kAccessRead);
    PrimExpr dst_ptr = MakeTLAccessPtr(copy.dst, copy.dst_range, kAccessWrite);
    bool same_dtype = copy.src->dtype == copy.dst->dtype;

    auto make_post_copy = [&](PrimExpr source_ptr) {
      Array<PrimExpr> args{dst_ptr, source_ptr, rows,
                           cols,    full_rows,  StringImm(dst_dtype)};
      return Evaluate(Call(DataType::Handle(), ascend_nd2nz_post_copy(), args));
    };

    if (src_is_nz) {
      if (!same_dtype) {
        TVM_FFI_THROW(ValueError)
            << "NZ source " << copy.src->name
            << " cannot be post-copied with a dtype conversion ("
            << copy.src->dtype << " -> " << copy.dst->dtype << ")";
      }
      return make_post_copy(src_ptr);
    }

    const int64_t *cols_i = as_const_int(cols);
    int elem_bytes = (copy.dst->dtype.bits() + 7) / 8;
    if (cols_i != nullptr && *cols_i * elem_bytes == 32 && same_dtype) {
      return make_post_copy(src_ptr);
    }

    PrimExpr scratch_extent = (rows + 1) * cols;
    Buffer scratch = decl_buffer({scratch_extent}, copy.dst->dtype,
                                 "nz_scratch", "shared.dyn");
    ICHECK(!pending_allocs_.empty());
    pending_allocs_.back().push_back(scratch);
    Array<Range> scratch_ranges{
        Range::FromMinExtent(IntImm(DataType::Int(32), 0), scratch_extent)};
    PrimExpr scratch_write =
        MakeTLAccessPtr(scratch, scratch_ranges, kAccessWrite);
    PrimExpr scratch_read =
        MakeTLAccessPtr(scratch, scratch_ranges, kAccessRead);
    Stmt scatter = Evaluate(Call(DataType::Handle(), ascend_nd2nz_scatter(),
                                 {src_ptr, scratch_write, rows, cols,
                                  StringImm(dst_dtype), StringImm(src_dtype)}));
    return SeqStmt({scatter, make_post_copy(scratch_read)});
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (inside_simt_vf_)
      return Parent::VisitStmt_(op);
    const auto *call = op->value.as<CallNode>();
    static const Op &copy_op = Op::Get("tl.tileop.copy");
    if (call == nullptr || !call->op.same_as(copy_op))
      return Parent::VisitStmt_(op);
    Copy copy(call->args, call->annotations);
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
