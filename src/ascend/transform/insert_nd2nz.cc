/*!
 * \file ascend/transform/insert_nd2nz.cc
 * \brief Rewrite layout-driven Ascend ND-to-NZ copies before AutoSchedule.
 */

#include "ascend/transform/source_scopes.h"
#include "ir/annotations.h"
#include "ir/stmt_functor.h"
#include "layout/expr_utils.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>

#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ascend/analysis/storage.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/nd2nz.h"
#include "ascend/op/oob_padding.h"
#include "ascend/op/utils.h"
#include "ir/lexical_scope.h"
#include "ir/tensor_utils.h"
#include "op/copy.h"
#include "op/utils.h"
#include "support/check.h"
#include <tvm/sym/analyzer.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;
using namespace ascend;
using namespace tirx::transform;

namespace {

std::string Nd2NzDTypeName(PrimType dtype) {
  if (dtype.MatchesElementType(kDLBfloat, 16))
    return "bfloat16_t";
  if (dtype.MatchesElementType(kDLFloat, 16))
    return "half";
  if (dtype.MatchesCode(kDLFloat) && dtype.bits() == 32)
    return "float";
  TVM_FFI_THROW(ValueError)
      << "Unsupported nd2nz dtype: " << dtype
      << ". The UB scatter supports only float16/bfloat16/float32; add a "
         "template specialization in src/tl_templates/ascend/nd2nz_copy.h "
         "to extend it.";
  return "";
}

int64_t RequireStaticInt(const PrimExpr &expr, const std::string &what) {
  auto value = TryConstInt(expr);
  if (!value) {
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
  PrimExpr result = IntImm(PrimType::Int(32), 1);
  for (const PrimExpr &extent : extents)
    result = result * extent;
  return result;
}

Expr MakeTLAccessPtr(const TensorVar &buffer, const Array<Range> &ranges,
                     const Array<PrimExpr> &pointer_extents, int rw_mask) {
  ICHECK_EQ(ranges.size(), buffer.shape().size());
  ICHECK_EQ(pointer_extents.size(), ranges.size());
  Array<PrimExpr> starts;
  for (const Range &range : ranges)
    starts.push_back(range->begin);
  return MakeTensorAccessPtr(TensorLoad(buffer, starts),
                             ExtentProduct(pointer_extents), rw_mask);
}

Expr MakeTLAccessPtr(const TensorVar &buffer, const Array<Range> &ranges,
                     int rw_mask) {
  return MakeTLAccessPtr(buffer, ranges, RegionExtents(ranges), rw_mask);
}

bool IsCanonicalNZLayout(const Layout &layout, const TensorVar &buffer) {
  if (!layout.defined() || !buffer.defined() || buffer.shape().size() < 2)
    return false;
  int bits = buffer.dtype().bits();
  if (bits <= 0 || 256 % bits != 0)
    return false;
  if (!IsAscendFractalLayout(layout))
    return false;
  AscendFractalLayoutInfo info;
  return TryExtractAscendFractalLayout(layout, buffer, &info) &&
         info.kind == AscendFractalKind::kMajorK;
}

class Nd2NzInserter : public StmtExprMutatorWithAnalyzer {
public:
  static tirx::Function Rewrite(tirx::Function func) {
    if (func->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return func;
    sym::Analyzer analyzer;
    analyzer->z3_prover.SetRLimit(50000);
    auto inserter = make_object<Nd2NzInserter>(analyzer);
    inserter->MarkTensorParamShapes(func);
    inserter->storage_.Collect(func->body);
    func.CopyOnWrite()->body =
        inserter->Mutate(func->body).ValueOrUnchanged(func->body);
    return func;
  }

public:
  using Parent = StmtExprMutatorWithAnalyzer;
  using Parent::Mutate;
  using Parent::Mutate_;
  explicit Nd2NzInserter(const sym::Analyzer &analyzer) : Parent(analyzer) {}

private:
  auto Strided(const TensorVar &tensor) const {
    auto result = MatchShapeStridePattern(tensor.ty());
    TVM_FFI_CHECK(result.has_value(), ValueError)
        << "ND->NZ requires a shape-aligned strided native tensor view: "
        << tensor;
    return result.value();
  }
  Array<PrimExpr> Strides(const TensorVar &tensor) const {
    return ffi::get<1>(Strided(tensor));
  }
  PrimExpr Origin(const TensorVar &tensor) const {
    Array<PrimExpr> zeros(tensor.shape().size(), IntImm::Int64(0));
    return TensorElementOffset(tensor, zeros);
  }
  PrimExpr PointerByteOffset(const TensorVar &tensor) const {
    auto address = storage_.Address(tensor);
    TVM_FFI_CHECK(address.exact, ValueError)
        << "ND->NZ pointer alignment needs a known native backing address: "
        << tensor;
    return address.byte_offset;
  }
  PrimExpr AbsoluteBaseBytes(const TensorVar &tensor) const {
    return cast(Origin(tensor), PrimType::Int(64)) *
               IntImm::Int64(tensor.dtype().StorageBytes()) +
           cast(PointerByteOffset(tensor), PrimType::Int(64));
  }

  Optional<Layout> LayoutFor(const TensorVar &buffer) const {
    return ascend::FindLayoutForBuffer(layout_map_, buffer);
  }

  bool IsNZBuffer(const TensorVar &buffer) const {
    Optional<Layout> layout = LayoutFor(buffer);
    return layout.has_value() &&
           (IsCanonicalNZLayout(layout.value(), buffer) ||
            IsAscendCompactNZLayout(layout.value(), buffer));
  }

  bool HasPhysicalLayout(const TensorVar &buffer) const {
    Optional<Layout> layout = LayoutFor(buffer);
    return layout.has_value() && !layout.value().as<FragmentNode>();
  }

  bool HasCompactTrailingMatrix(const TensorVar &tensor) const {
    auto shape_stride = MatchShapeStridePattern(tensor.ty());
    if (!shape_stride || tensor.shape().size() < 2)
      return false;
    const auto &strides = ffi::get<1>(*shape_stride);
    size_t col = strides.size() - 1, row = col - 1;
    return GetSyncedAnalyzer()->CanProveEqual(strides[col], 1) &&
           GetSyncedAnalyzer()->CanProveEqual(strides[row],
                                              tensor.shape()[col]);
  }

  bool HasAlignedScatterBase(const TensorVar &tensor,
                             const Array<Range> &ranges) const {
    Array<PrimExpr> starts;
    for (const Range &range : ranges)
      starts.push_back(range->begin);
    PrimExpr byte_address =
        cast(TensorElementOffset(tensor, starts), PrimType::Int(64)) *
            IntImm::Int64(tensor.dtype().StorageBytes()) +
        PointerByteOffset(tensor);
    return GetSyncedAnalyzer()->CanProveEqual(floormod(byte_address, 32), 0);
  }

  void ValidateRegionInBounds(const TensorVar &buffer,
                              const Array<Range> &ranges,
                              const std::string &role) const {
    ICHECK_EQ(ranges.size(), buffer.shape().size());
    for (size_t axis = 0; axis < ranges.size(); ++axis) {
      const Range &range = ranges[axis];
      bool lower_in_bounds = GetSyncedAnalyzer()->CanProve(range->begin >= 0);
      bool non_negative_extent =
          GetSyncedAnalyzer()->CanProve(range->extent >= 0);
      bool upper_in_bounds = GetSyncedAnalyzer()->CanProve(
          range->begin + range->extent <= buffer.shape()[axis]);
      if (!lower_in_bounds || !non_negative_extent || !upper_in_bounds) {
        TVM_FFI_THROW(ValueError)
            << "UB ND->NZ " << role
            << " region must be provably in bounds on axis " << axis
            << ": min=" << range->begin << ", extent=" << range->extent
            << ", buffer extent=" << buffer.shape()[axis] << ".";
      }
    }
  }

  std::pair<std::string, std::string>
  ValidateScatterTemplate(const TensorVar &src, const TensorVar &dst,
                          int64_t rows, int64_t cols) const {
    std::string src_dtype = Nd2NzDTypeName(src.dtype());
    std::string dst_dtype = Nd2NzDTypeName(dst.dtype());
    if (rows % 16 != 0) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ scatter requires rows divisible by 16, got " << rows
          << ".";
    }
    if (src_dtype == dst_dtype) {
      int element_bytes = (src.dtype().bits() + 7) / 8;
      if (cols * element_bytes % 32 != 0) {
        TVM_FFI_THROW(ValueError)
            << "UB ND->NZ same-dtype scatter requires a 32-byte-aligned row, "
               "got cols="
            << cols << ", dtype=" << src.dtype() << ".";
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
          << src.dtype() << " -> " << dst.dtype()
          << "; supported conversions are float32 <-> bfloat16.";
    }
    return {std::move(src_dtype), std::move(dst_dtype)};
  }

  PrimExpr NZPitch(const TensorVar &buffer, const Array<Range> &ranges) const {
    auto layout = LayoutFor(buffer);
    if (!layout.has_value() || !IsNZBuffer(buffer))
      TVM_FFI_THROW(ValueError)
          << "NZ copy requires a compact NZ or MajorK layout: " << buffer->name;
    ValidateRegionInBounds(buffer, ranges, "NZ");
    for (size_t i = 0; i + 2 < ranges.size(); ++i)
      if (!GetSyncedAnalyzer()->CanProveEqual(ranges[i]->extent, 1))
        TVM_FFI_THROW(ValueError)
            << "NZ copy leading region extents must all be 1";
    int c0 = AscendC0(buffer.dtype().bits());
    if (!GetSyncedAnalyzer()->CanProveEqual(floormod(ranges.back()->begin, c0),
                                            0) ||
        !GetSyncedAnalyzer()->CanProveEqual(floormod(ranges.back()->extent, c0),
                                            0) ||
        !GetSyncedAnalyzer()->CanProveEqual(
            floormod(AbsoluteBaseBytes(buffer), 32), 0))
      TVM_FFI_THROW(ValueError) << "NZ copy requires C0-aligned column starts, "
                                   "widths, and base offsets";
    PrimExpr pitch = buffer.shape()[buffer.shape().size() - 2];
    if (IsCanonicalNZLayout(layout.value(), buffer)) {
      const auto physical_shape = layout.value()->OutputShape();
      pitch = GetSyncedAnalyzer()->Simplify(
          physical_shape[physical_shape.size() - 3] *
          physical_shape[physical_shape.size() - 2]);
    }
    RequireStaticInt(pitch, "NZ storage pitch");
    return pitch;
  }

  Expr NZPointer(const TensorVar &buffer, const Array<Range> &ranges,
                 PrimExpr pitch, int mask) const {
    PrimExpr rows = ranges[ranges.size() - 2]->extent;
    PrimExpr cols = ranges.back()->extent;
    int c0 = AscendC0(buffer.dtype().bits());
    Array<PrimExpr> extents(ranges.size(), IntImm::Int32(1));
    extents.Set(extents.size() - 1,
                GetSyncedAnalyzer()->Simplify(
                    ((floordiv(cols, c0) - 1) * pitch + rows) * c0));
    return MakeTLAccessPtr(buffer, ranges, extents, mask);
  }

  Stmt Intrinsic(const Op &op, Array<Expr> args, bool scatter) const {
    // Each bit proves that the corresponding pointer argument's logical
    // rectangle maps to the NZ DMA blocks before LowerTileOp.
    Call call = MakeAnnotatedCall(PrimType::Void(), op, args, {});
    auto geometry = ParseNd2NzGeometry(call.get(), scatter);
    if (geometry.Empty())
      return Evaluate(IntImm::Int32(0));
    int logical_regions = 3;
    if (!scatter) {
      logical_regions =
          (HasLogicalPostCopyRegion(args[0], geometry, false) ? 1 : 0) |
          (HasLogicalPostCopyRegion(args[1], geometry, true) ? 2 : 0);
    }
    return Evaluate(MakeAnnotatedCall(
        PrimType::Void(), op, args,
        {{"nd2nz_logical_regions", IntImm::Int32(logical_regions)}}));
  }

  std::optional<Stmt> RewriteUBToUB(const CopyNode &copy) {
    if (!IsSharedBuffer(copy.src) || !IsSharedBuffer(copy.dst))
      return std::nullopt;
    Optional<Layout> dst_layout = LayoutFor(copy.dst);
    if (!dst_layout.has_value() ||
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
          << copy.src->name << " has explicit strides " << Strides(copy.src)
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
    if (copy.src_range.size() != copy.src.shape().size() ||
        copy.dst_range.size() != copy.dst.shape().size()) {
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
      if (!GetSyncedAnalyzer()->CanProveEqual(copy.src_range[i]->extent, 1)) {
        TVM_FFI_THROW(ValueError)
            << "UB ND->NZ source leading region extents must all be 1, got "
            << copy.src_range;
      }
    }
    for (size_t i = 0; i + 2 < copy.dst_range.size(); ++i) {
      if (!GetSyncedAnalyzer()->CanProveEqual(copy.dst_range[i]->extent, 1)) {
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
    if (!GetSyncedAnalyzer()->CanProveEqual(
            copy.dst_range[dst_ndim - 2]->extent, rows) ||
        !GetSyncedAnalyzer()->CanProveEqual(
            copy.dst_range[dst_ndim - 1]->extent, cols)) {
      TVM_FFI_THROW(ValueError)
          << "UB ND->NZ source/destination logical regions must match.";
    }
    if (IsZero(GetSyncedAnalyzer()->Simplify(rows)) ||
        IsZero(GetSyncedAnalyzer()->Simplify(cols)))
      return Evaluate(IntImm::Int32(0));
    int64_t rows_i = RequireStaticInt(rows, "UB ND->NZ scatter rows");
    int64_t cols_i = RequireStaticInt(cols, "UB ND->NZ scatter cols");
    auto [src_dtype, dst_dtype] =
        ValidateScatterTemplate(copy.src, copy.dst, rows_i, cols_i);

    if (!GetSyncedAnalyzer()->CanProveEqual(copy.src_range[src_ndim - 1]->begin,
                                            0) ||
        !GetSyncedAnalyzer()->CanProveEqual(copy.src.shape()[src_ndim - 1],
                                            cols)) {
      TVM_FFI_THROW(ValueError) << "UB ND->NZ source " << copy.src->name
                                << " must cover a full contiguous row.";
    }
    PrimExpr dst_pitch = NZPitch(copy.dst, copy.dst_range);
    Expr src_ptr = MakeTLAccessPtr(copy.src, copy.src_range, kAccessRead);
    Expr dst_ptr = NZPointer(copy.dst, copy.dst_range, dst_pitch, kAccessWrite);
    return Intrinsic(ascend_nd2nz_scatter(),
                     {src_ptr, dst_ptr, rows, cols, dst_pitch,
                      StringImm(src_dtype), StringImm(dst_dtype)},
                     true);
  }

  std::optional<Stmt> RewriteUBToL1(const CopyNode &copy) {
    if (!IsSharedBuffer(copy.src) || !IsL1Buffer(copy.dst) ||
        copy.dst_range.size() < 2 || !LayoutFor(copy.dst).has_value()) {
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
    if (!GetSyncedAnalyzer()->CanProveEqual(
            copy.src_range[src_ndim - 2]->extent, rows) ||
        !GetSyncedAnalyzer()->CanProveEqual(copy.src_range.back()->extent,
                                            cols))
      TVM_FFI_THROW(ValueError)
          << "NZ copy source/destination logical regions must match";
    if (IsZero(GetSyncedAnalyzer()->Simplify(rows)) ||
        IsZero(GetSyncedAnalyzer()->Simplify(cols)))
      return Evaluate(IntImm::Int32(0));
    PrimExpr dst_pitch = NZPitch(copy.dst, copy.dst_range);
    Expr dst_ptr = NZPointer(copy.dst, copy.dst_range, dst_pitch, kAccessWrite);
    std::string dst_dtype = Nd2NzDTypeName(copy.dst.dtype());
    auto make_post_copy = [&](Expr source_ptr, PrimExpr src_pitch) {
      return Intrinsic(ascend_nd2nz_post_copy(),
                       {dst_ptr, source_ptr, rows, cols, src_pitch, dst_pitch,
                        StringImm(dst_dtype)},
                       false);
    };
    bool same_dtype = copy.src.dtype() == copy.dst.dtype();
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
      if (!GetSyncedAnalyzer()->CanProveEqual(copy.src_range[i]->extent, 1))
        TVM_FFI_THROW(ValueError)
            << "ND->NZ source leading region extents must all be 1";
    if (!HasAlignedScatterBase(copy.src, copy.src_range) ||
        !GetSyncedAnalyzer()->CanProveEqual(copy.src_range.back()->begin, 0) ||
        !GetSyncedAnalyzer()->CanProveEqual(copy.src.shape().back(), cols))
      TVM_FFI_THROW(ValueError) << "ND->NZ source must cover full contiguous "
                                   "rows at a 32-byte-aligned address";
    Expr src_ptr = MakeTLAccessPtr(copy.src, copy.src_range, kAccessRead);
    int64_t rows_i = RequireStaticInt(rows, "ND->NZ rows");
    int64_t cols_i = RequireStaticInt(cols, "ND->NZ cols");
    if (cols_i == AscendC0(copy.dst.dtype().bits()) && same_dtype)
      return make_post_copy(src_ptr, rows);
    auto [src_dtype, checked_dst_dtype] =
        ValidateScatterTemplate(copy.src, copy.dst, rows_i, cols_i);
    // A fresh scratch buffer uses one extra row per C0 group. Existing NZ
    // buffers instead retain the pitch of their complete storage layout.
    PrimExpr scratch_pitch = rows + 1;
    PrimExpr scratch_extent =
        GetSyncedAnalyzer()->Simplify(scratch_pitch * cols);
    TensorVar scratch("nz_scratch",
                      TensorType({scratch_extent}, copy.dst.dtype(),
                                 runtime::MemoryScope("shared.dyn")));
    Bind scratch_definition(
        scratch, Call(scratch.ty(), alloc_tensor_op(), {}, {}, {scratch.ty()}));
    storage_.bindings.Set(scratch, scratch_definition->value);
    // This temporary has the lifetime of the nearest source block, as did
    // its original block allocation. A copy may be nested in a loop or
    // branch, while scheduling metadata referring to this storage belongs
    // to that enclosing block. Keep its definition at that boundary.
    if (!pending_allocs_.empty())
      pending_allocs_.back().push_back(scratch_definition);
    Array<Range> scratch_ranges{Range::FromMinExtent(0, scratch_extent)};
    Expr scratch_write = MakeTLAccessPtr(scratch, scratch_ranges, kAccessWrite);
    Expr scratch_read = MakeTLAccessPtr(scratch, scratch_ranges, kAccessRead);
    Stmt scatter =
        Intrinsic(ascend_nd2nz_scatter(),
                  {src_ptr, scratch_write, rows, cols,
                   GetSyncedAnalyzer()->Simplify(scratch_pitch),
                   StringImm(src_dtype), StringImm(checked_dst_dtype)},
                  true);
    Array<Stmt> body;
    // Native functions without an explicit allocation owner retain their
    // existing local allocation lifetime at the copy position.
    if (pending_allocs_.empty())
      body.push_back(scratch_definition);
    body.push_back(scatter);
    body.push_back(make_post_copy(
        scratch_read, GetSyncedAnalyzer()->Simplify(scratch_pitch)));
    return SeqStmt(body);
  }

  void ValidatePostCopyPointer(const Expr &pointer, int64_t span_bytes,
                               int64_t element_bytes, bool source) const {
    const auto *access = pointer.as<CallObj>();
    if (!access || !access->op.same_as(tl::access_ptr()))
      return; // Raw pointer callers own the allocation/alignment contract.
    const auto *load = access->args[0].as<TensorLoadObj>();
    if (!load)
      return;
    const TensorVar buffer = load->source.as_or_throw<TensorVar>();
    if ((source ? !IsSharedBuffer(buffer) : !IsL1Buffer(buffer)) ||
        buffer.dtype().StorageBytes() != element_bytes)
      TVM_FFI_THROW(ValueError)
          << "NZ post-copy requires matching UB/L1 pointer element widths";
    Array<PrimExpr> indices = load->indices;
    Array<PrimExpr> shape = buffer.shape();
    Array<PrimExpr> strides = Strides(buffer);
    if (auto layout = LayoutFor(buffer)) {
      indices = layout.value()->Forward(indices);
      shape = layout.value()->OutputShape();
      // LowerTileOp materializes dense physical storage. Logical strides
      // neither describe those axes nor necessarily have the same rank.
      strides = {};
    }
    PrimExpr offset = IntImm(PrimType::Int(64), 0);
    PrimExpr capacity = IntImm::Int32(1), stride = IntImm::Int32(1);
    for (size_t i = shape.size(); i-- > 0;) {
      PrimExpr step = strides.empty() ? stride : strides[i];
      offset = offset + cast(indices[i], PrimType::Int(64)) * step;
      capacity = capacity + (cast(shape[i], PrimType::Int(64)) - 1) * step;
      stride = stride * cast(shape[i], PrimType::Int(64));
    }
    // Alignment uses the absolute address; bounds are relative to this view.
    PrimExpr address_bytes =
        (offset + cast(Origin(buffer), PrimType::Int(64))) *
            IntImm::Int64(element_bytes) +
        PointerByteOffset(buffer);
    if (!GetSyncedAnalyzer()->CanProveEqual(floormod(address_bytes, 32), 0))
      TVM_FFI_THROW(ValueError)
          << "NZ post-copy requires a 32-byte-aligned pointer: "
          << buffer->name;
    if (!GetSyncedAnalyzer()->CanProve(
            address_bytes >= 0 && offset >= 0 &&
            offset + IntImm::Int32(span_bytes / element_bytes) <= capacity))
      TVM_FFI_THROW(ValueError)
          << "NZ post-copy physical span exceeds pointer buffer capacity: "
          << buffer->name;
  }

  bool HasLogicalPostCopyRegion(const Expr &pointer,
                                const Nd2NzGeometry &geometry,
                                bool source) const {
    const auto *access = pointer.as<CallObj>();
    const auto *load = access && access->op.same_as(tl::access_ptr())
                           ? access->args[0].as<TensorLoadObj>()
                           : nullptr;
    if (!load)
      return false;
    const TensorVar buffer = load->source.as_or_throw<TensorVar>();
    size_t ndim = buffer.shape().size();
    if (ndim < 2)
      return false;
    Array<Range> ranges;
    for (size_t i = 0; i < ndim; ++i) {
      PrimExpr extent = IntImm::Int32(i + 2 == ndim   ? geometry.rows
                                      : i + 1 == ndim ? geometry.cols
                                                      : 1);
      if (!GetSyncedAnalyzer()->CanProve(load->indices[i] >= 0 &&
                                         load->indices[i] + extent <=
                                             buffer.shape()[i]))
        return false;
      ranges.push_back(Range::FromMinExtent(load->indices[i], extent));
    }
    int c0 = AscendC0(buffer.dtype().bits());
    if (!GetSyncedAnalyzer()->CanProveEqual(floormod(load->indices.back(), c0),
                                            0) ||
        !GetSyncedAnalyzer()->CanProveEqual(
            floormod(AbsoluteBaseBytes(buffer), 32), 0))
      return false;
    int64_t pitch = source ? geometry.src_pitch : geometry.dst_pitch;
    if (IsNZBuffer(buffer))
      return GetSyncedAnalyzer()->CanProveEqual(NZPitch(buffer, ranges),
                                                IntImm::Int32(pitch));
    // A dense ND source with one C0-wide full row has the same physical
    // layout as one NZ column group. No broader ND/NZ equivalence is assumed.
    return source && !HasPhysicalLayout(buffer) &&
           HasCompactTrailingMatrix(buffer) && geometry.cols == c0 &&
           geometry.src_pitch == geometry.rows &&
           GetSyncedAnalyzer()->CanProveEqual(load->indices.back(), 0) &&
           GetSyncedAnalyzer()->CanProveEqual(buffer.shape().back(),
                                              IntImm::Int32(geometry.cols));
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    if (inside_simt_vf_)
      return Parent::Mutate_(op, mode);
    const auto *call = op->value.as<CallObj>();
    if (call && call->op.same_as(ascend_nd2nz_post_copy())) {
      const auto args = DecodeCallOperands(call);
      auto geometry = ParseNd2NzGeometry(call, false);
      if (geometry.Empty())
        return Evaluate(IntImm::Int32(0));
      ValidatePostCopyPointer(args[0], geometry.SpanBytes(geometry.dst_pitch),
                              geometry.dst_bytes, false);
      ValidatePostCopyPointer(args[1], geometry.SpanBytes(geometry.src_pitch),
                              geometry.src_bytes, true);
      // Explicit pitches need not match a multidimensional view's layout.
      // Prove the destination independently; an opaque source must not erase
      // the disjoint AIV row slices of a valid NZ destination.
      Map<String, Any> metadata;
      if (call->attrs &&
          call->attrs.value().as_or_throw<DictAttrs>()->dict.count(
              "annotation_schema"))
        metadata = DecodeTileOpArguments(call).annotations;
      else if (call->attrs) {
        metadata = call->attrs.value().as_or_throw<DictAttrs>()->dict;
        metadata.erase("tl.source_annotation_keys");
      }
      int logical_regions =
          HasLogicalPostCopyRegion(args[0], geometry, false) ? 1 : 0;
      // A source pointer carries a physical span, not a logical region
      // contract. Revalidate the source proof on a generated call so a
      // repeated pass preserves it, without adding that contract to an
      // untagged explicit pointer.
      if (auto previous = metadata.Get("nd2nz_logical_regions")) {
        if (auto bits = previous.value().try_cast<PrimExpr>()) {
          auto mask = TryConstInt(bits.value());
          if (mask && (*mask & 2) &&
              HasLogicalPostCopyRegion(args[1], geometry, true))
            logical_regions |= 2;
        }
      }
      metadata.Set("nd2nz_logical_regions", IntImm::Int32(logical_regions));
      Call validated = MakeAnnotatedCall(call->ty, call->op.as_or_throw<Op>(),
                                         args, metadata, call->loc);
      return Evaluate(validated);
    }
    if (call == nullptr || !IsAscendCopyCall(call))
      return Parent::Mutate_(op, mode);
    auto decoded = DecodeTileOpArguments(call);
    AscendCopy copy(decoded.operands, decoded.annotations);
    if (std::optional<Stmt> rewritten = RewriteUBToUB(*copy.get()))
      return rewritten.value();
    if (std::optional<Stmt> rewritten = RewriteUBToL1(*copy.get()))
      return rewritten.value();
    return Parent::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    LayoutMap saved = layout_map_;
    bool simt = inside_simt_vf_;
    auto attribute = ReadSourceAttribute(op);
    // Kernel normalization may already have removed its ordinary root
    // sblock. The remaining kernel body retains that allocation boundary.
    bool block = IsAscendBlock(op) || IsLexicalScope(op) ||
                 (attribute && attribute->key == "tl.kernel");
    if (block)
      pending_allocs_.emplace_back();
    if (IsAscendBlock(op)) {

      if (op->attrs.GetAttr<ffi::String>("name").value() == "SIMT_VF")
        inside_simt_vf_ = true;
    }
    auto result = Parent::Mutate_(op, mode);
    layout_map_ = std::move(saved);
    inside_simt_vf_ = simt;
    if (block) {
      std::vector<Bind> pending = std::move(pending_allocs_.back());
      pending_allocs_.pop_back();
      if (!pending.empty()) {
        RegionStmt rewritten = std::move(result)
                                   .ValueOrUnchanged(GetRef<RegionStmt>(op))
                                   .as_or_throw<RegionStmt>();
        Array<Stmt> body;
        for (const Bind &binding : pending)
          body.push_back(binding);
        for (const Stmt &stmt : rewritten->body->seq)
          body.push_back(stmt);
        rewritten.CopyOnWrite()->body = SeqStmt(body, rewritten->body->loc);
        return rewritten;
      }
    }
    return result;
  }

  void ReadLayouts(const Map<String, Any> &fields) {
    if (auto value = fields.Get("layout_map"))
      for (const auto &[tensor, layout] : value->cast<LayoutMap>())
        layout_map_.Set(tensor, layout);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode) final {
    LayoutMap saved = layout_map_;
    Array<Stmt> body;
    for (const Stmt &stmt : op->seq) {
      if (const auto *evaluate = stmt.as<EvaluateObj>()) {
        if (const auto *call = evaluate->value.as<CallObj>();
            IsAnnotationCall(call))
          ReadLayouts(ReadAnnotations(call));
      }
      body.push_back(Mutate(stmt).ValueOrUnchanged(stmt));
    }
    layout_map_ = std::move(saved);
    return SeqStmt(std::move(body), op->loc);
  }

  LayoutMap layout_map_;
  StorageProvenance storage_ = MakeAscendStorageProvenance();
  std::vector<std::vector<Bind>> pending_allocs_;
  bool inside_simt_vf_{false};
};

} // namespace

tvm::transform::Pass InsertNd2Nz() {
  auto pass_func = [](tirx::Function func, const IRModule &mod,
                      const PassContext &ctx) {
    return Nd2NzInserter::Rewrite(std::move(func));
  };
  return CreateFunctionPass(pass_func, 0, "tl.InsertNd2Nz");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InsertNd2Nz", InsertNd2Nz);
}

} // namespace tl
} // namespace tvm
