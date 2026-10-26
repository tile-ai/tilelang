#include "compat/analyzer.h"
/*!
 * \file normalize_fractal_storage.cc
 * \brief Normalize Cube storage and aliases before OOB handling and scheduling.
 *
 * This pass does not initialize the added padding. Copy/compute operations
 * must honor valid regions and their own padding requirements; consumers must
 * not assume all padded elements are zero.
 */
#include "op/prim.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>

#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/transform/auto_schedule/schedule_scope.h"
#include "ascend/transform/native_storage.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {
using namespace ffi;
using namespace tirx;
namespace {

BlockAnnotations CallAnnotations(const CallNode *call) {
  if (auto attrs = call->attrs.as<DictAttrs>())
    return attrs.value()->dict;
  return {};
}

class BufferCollector : public ascend::StorageAliasAnalysis {
public:
  LayoutMap layouts;
  std::unordered_set<TensorVar, ObjectPtrHash, ObjectPtrEqual> used_buffers;

private:
  void VisitExpr_(const VarNode *node) final {
    if (!alias_binding_) {
      if (auto tensor = GetRef<Var>(node).as<TensorVar>())
        used_buffers.insert(tensor.value());
    }
  }

  void VisitExpr_(const CallNode *node) final {
    if (node->op.same_as(Op::Get("tl.layout_hint"))) {
      auto tensor = node->args[0].as_or_throw<TensorVar>();
      auto layout = node->attrs.value()
                        .as_or_throw<DictAttrs>()
                        .GetAttr<Layout>("layout")
                        .value();
      if (auto previous = layouts.Get(tensor))
        TVM_FFI_CHECK(previous.value()->IsEqual(layout.get()), ValueError)
            << "Conflicting layouts for " << tensor->name;
      layouts.Set(tensor, layout);
      tensors.push_back(tensor);
      return;
    }
    if (node->op.same_as(tensor_data_ptr_op()))
      return;
    ascend::AnalysisVisitor::VisitExpr_(node);
  }

  void VisitStmt_(const BindNode *node) final {
    bool previous = alias_binding_;
    alias_binding_ = node->value.as<TensorVar>().has_value();
    ascend::StorageAliasAnalysis::VisitStmt_(node);
    alias_binding_ = previous;
  }

  void VisitStmt_(const RegionStmtNode *node) final {
    Visit(node->args);
    ascend::StorageAliasAnalysis::VisitStmt_(node);
    if (auto value = node->attrs->dict.Get(attr::kLayoutMap)) {
      for (const auto &[tensor, layout] : value->cast<LayoutMap>()) {
        if (auto previous = layouts.Get(tensor)) {
          ICHECK(previous.value()->IsEqual(layout.get()))
              << "Conflicting layouts for " << tensor->name;
        }
        layouts.Set(tensor, layout);
        tensors.push_back(tensor);
      }
    }
  }
  bool alias_binding_{false};
};

class FractalStorageNormalizer : public ascend::AnalyzerMutator {
public:
  FractalStorageNormalizer(sym::Analyzer analyzer, BufferCollector *storage)
      : ascend::AnalyzerMutator(analyzer), storage_(storage) {}

  static tirx::Function Rewrite(tirx::Function function) {
    if (!function->body)
      return function;
    auto storage = make_object<BufferCollector>();
    storage->Collect(function);
    for (const auto &[tensor, layout] :
         function->GetAttr<LayoutMap>(attr::kLayoutMap).value_or(LayoutMap{})) {
      if (auto previous = storage->layouts.Get(tensor))
        TVM_FFI_CHECK(previous.value()->IsEqual(layout.get()), ValueError)
            << "Conflicting function layout for " << tensor->name;
      storage->layouts.Set(tensor, layout);
    }
    sym::Analyzer analyzer;
    analyzer->z3_prover.SetRLimit(50000);
    auto rewriter =
        make_object<FractalStorageNormalizer>(analyzer, storage.get());
    rewriter->canonical_layouts_ = storage->layouts;
    for (const auto &[tensor, layout] : storage->layouts) {
      Var root = rewriter->Root(tensor);
      if (!IsAscendFractalLayout(layout) ||
          !(IsL1Buffer(tensor) || IsL0ABuffer(tensor) || IsL0BBuffer(tensor) ||
            IsL0CBuffer(tensor)) ||
          !storage->allocations.count(root))
        continue;
      if (auto anchor = rewriter->anchors_.Get(root)) {
        AscendFractalLayoutInfo previous, current;
        ICHECK(TryExtractAscendFractalLayout(storage->layouts[anchor.value()],
                                             anchor.value(), &previous));
        ICHECK(TryExtractAscendFractalLayout(layout, tensor, &current));
        ICHECK(previous.kind == current.kind &&
               storage->layouts[anchor.value()]->IsEqual(layout.get()) &&
               anchor.value().dtype().bits() == tensor.dtype().bits())
            << "Conflicting canonical Cube views of " << tensor->name;
        if (storage->used_buffers.count(tensor) &&
            !storage->used_buffers.count(anchor.value())) {
          rewriter->anchors_.Set(root, tensor);
        }
      } else {
        rewriter->anchors_.Set(root, tensor);
      }
    }
    for (const TensorVar &tensor : storage->tensors) {
      Var root = rewriter->Root(tensor);
      if (auto anchor = rewriter->anchors_.Get(root)) {
        ICHECK(!storage->used_buffers.count(tensor) ||
               tensor.dtype() == anchor.value().dtype())
            << "Cube alias accesses must use the canonical operand dtype: "
            << tensor->name;
        rewriter->RegisterBuffer_(tensor, anchor.value(),
                                  storage->allocations[root],
                                  storage->layouts[anchor.value()]);
      }
    }
    if (rewriter->buffers_.empty())
      return function;
    auto body = function->body.value();
    function.CopyOnWrite()->body =
        rewriter->Mutate(body).ValueOrUnchanged(body);
    auto attrs = function->attrs;
    function.CopyOnWrite()->attrs = rewriter->Mutate(attrs)
                                        .as_or_throw<UnchangedOr<DictAttrs>>()
                                        .ValueOrUnchanged(attrs);
    if (auto layouts = function->GetAttr<LayoutMap>(attr::kLayoutMap)) {
      LayoutMap canonical;
      for (const auto &[tensor, layout] : *layouts)
        canonical.Set(
            tensor, rewriter->canonical_layouts_.Get(tensor).value_or(layout));
      auto attributes = function->attrs;
      attributes.CopyOnWrite()->dict.Set(attr::kLayoutMap, canonical);
      function.CopyOnWrite()->attrs = attributes;
    }
    return function;
  }

private:
  using Parent = ascend::AnalyzerMutator;

  Var Root(const TensorVar &tensor) const {
    return storage_->Resolve(tensor).value().root;
  }

  sym::Analyzer Analyzer() const { return GetRef<sym::Analyzer>(analyzer_); }

  void RegisterBuffer_(const TensorVar &tensor, const TensorVar &anchor,
                       const TensorVar &owner, const Layout &layout) {
    ICHECK_EQ(tensor.dtype().lanes(), 1);
    ICHECK_EQ(tensor.dtype().bits(), anchor.dtype().bits())
        << "Cube aliases must preserve element width: " << tensor->name;
    auto access = storage_->Resolve(tensor).value();
    auto geometry = tirx::MatchShapeStridePattern(tensor.ty());
    ICHECK(geometry &&
           tl::arith::CanProve(analyzer_, geometry.value().get<2>() == 0) &&
           tl::arith::CanProve(analyzer_, access.bit_offset == 0))
        << "Ascend fractal buffers require a zero base and strided storage";
    auto strides = geometry.value().get<1>();
    PrimExpr stride = IntImm(tensor.ty()->DefaultIndexType(), 1);
    for (size_t dimension = tensor.shape().size(); dimension > 0; --dimension) {
      size_t axis = dimension - 1;
      ICHECK(tl::arith::CanProve(analyzer_, tensor.shape()[axis] == 1) ||
             tl::arith::CanProve(analyzer_, strides[axis] == stride))
          << "Cube aliases require dense logical strides: " << tensor->name;
      stride = stride * tensor.shape()[axis];
    }
    Array<PrimExpr> identity;
    for (size_t axis = 0; axis < anchor.shape().size(); ++axis)
      identity.push_back(InputPlaceholder(axis));
    logical_maps_.Set(
        tensor,
        Layout(anchor.shape(), identity)->Reshape(tensor.shape(), Analyzer()));
    AscendFractalLayoutInfo info;
    ICHECK(TryExtractAscendFractalLayout(layout, anchor, &info));
    auto shape = anchor.shape();
    ICHECK_GE(shape.size(), 2U);
    for (size_t axis = shape.size() - 2; axis < shape.size(); ++axis) {
      bool row = axis == shape.size() - 2;
      PrimExpr alignment = (row == (info.c0_axis == LogicalAxis::kRow))
                               ? info.c0
                               : info.row_frac;
      shape.Set(axis, tl::arith::Simplify(
                          analyzer_,
                          floordiv(shape[axis] + alignment - 1, alignment) *
                              alignment));
    }
    Var root = access.root;
    TensorVar padded = canonical_buffers_.Get(root).value_or(owner);
    if (!canonical_buffers_.count(root)) {
      padded = owner
                   .CopyWithType(TensorType(
                       shape, anchor.dtype(), owner.ty()->scope,
                       owner.ty()->data_alignment, std::nullopt, owner->loc))
                   .as_or_throw<TensorVar>();
    }
    Layout padded_layout;
    switch (info.kind) {
    case AscendFractalKind::kMajorK:
      padded_layout = MakeAscendMajorKLayout(padded);
      break;
    case AscendFractalKind::kMajorMN:
      padded_layout = MakeAscendMajorMNLayout(padded);
      break;
    case AscendFractalKind::kL0C:
      padded_layout = MakeAscendL0CLayout(padded);
      break;
    case AscendFractalKind::kSF:
      padded_layout = MakeAscendSFLayout(padded);
      break;
    }
    buffers_.Set(tensor, padded);
    canonical_buffers_.Set(root, padded);
    canonical_layouts_.Set(padded, padded_layout);
  }

  Array<PrimExpr> RewriteIndices_(const TensorVar &buffer,
                                  const Array<PrimExpr> &indices) {
    auto rewritten = indices.Map([&](PrimExpr index) {
      return VisitExpr(index).as_or_throw<PrimExpr>();
    });
    if (auto layout = logical_maps_.Get(buffer))
      return layout.value()->Forward(rewritten).Map([&](PrimExpr index) {
        return tl::arith::Simplify(analyzer_, index);
      });
    return rewritten;
  }
  TensorRegion RewriteRegion_(const TensorRegion &region) {
    TensorVar buffer = region->source.as_or_throw<TensorVar>();
    Array<Range> ranges = region->region.Map([&](const Range &range) {
      return Range::FromMinExtent(
          VisitExpr(range->begin).as_or_throw<PrimExpr>(),
          VisitExpr(range->extent).as_or_throw<PrimExpr>());
    });
    if (!buffers_.count(buffer))
      return TensorRegion(buffer, ranges, TensorRegionType());
    if (!StructuralEqual()(buffer.shape(), anchors_[Root(buffer)].shape())) {
      // A dense reshape is injective only while trailing region extents do
      // not wrap across the original view's strides. Otherwise equal bounding
      // box volume alone can hide duplicated logical elements.
      for (size_t i = 1; i < ranges.size(); ++i)
        ICHECK(tl::arith::CanProve(analyzer_,
                                   ranges[i]->extent <= buffer.shape()[i]))
            << "Cube region crosses an original reshape axis: " << region;
      Array<Range> mapped = logical_maps_[buffer]->MapRegionBounds(ranges);
      PrimExpr old_size = PrimExpr(1), new_size = PrimExpr(1);
      for (const Range &range : ranges)
        old_size = old_size * range->extent;
      for (const Range &range : mapped)
        new_size = new_size * range->extent;
      // A bounding box with holes cannot replace a tile operation's region.
      ICHECK(tl::arith::CanProveEqual(analyzer_, old_size, new_size))
          << "Reshaped Cube region is not a contiguous rectangular matrix: "
          << region;
      ranges = mapped;
    }
    return TensorRegion(buffers_[buffer], ranges, TensorRegionType());
  }
  Call NormalizeL0Copy_(Call call) {
    AscendCopy copy(call->args, CallAnnotations(call.get()));
    for (const auto &ranges : {copy->src_range, copy->dst_range})
      for (const Range &range : ranges)
        if (tl::arith::CanProve(analyzer_, range->extent <= 0))
          return call;
    TensorRegion source{ffi::UnsafeInit{}};
    Optional<TensorRegion> destination;
    Optional<PrimExpr> pitch;
    if (IsL0SFBuffer(copy->dst)) {
      // SF handles carry the effective shape of their bound data tile and
      // have no independent storage. Only the L1 source is a transfer region.
      source = ascend::GetL1ToL0SFSourceRegion(*copy.get(), canonical_layouts_,
                                               Analyzer());
    } else {
      auto regions = ascend::GetL1ToL0CopyRegions(
          *copy.get(), canonical_layouts_, Analyzer());
      source = regions.source;
      destination = regions.destination;
      pitch = regions.pitch;
    }
    auto *writer = call.CopyOnWrite();
    writer->args.Set(0, source);
    if (destination.has_value()) {
      writer->args.Set(1, destination.value());
      // Preserve the original compact pitch if the destination was larger
      // than the actual transfer. This also makes repeated normalization
      // stable.
      AscendFractalRegionInfo normalized;
      ICHECK(TryExtractAscendFractalRegion(canonical_layouts_[copy->dst],
                                           destination.value()->region,
                                           &normalized));
      if (!tl::arith::CanProveEqual(analyzer_, normalized.outer1->extent,
                                    pitch.value())) {
        auto attrs = CallAnnotations(writer);
        attrs.Set("l0_pitch", pitch.value());
        writer->attrs = DictAttrs(attrs);
      }
    }
    return call;
  }
  Call NormalizeFixpipeCopy_(Call call) {
    constexpr int kFixpipeAlignmentBytes = 32;
    AscendCopy rewritten(call->args, CallAnnotations(call.get()));
    ICHECK_GE(rewritten->src_range.size(), 2U);
    ICHECK_EQ(rewritten->src_range.size(), rewritten->src.shape().size());
    // Lowering skips the DMA if any source or destination extent is
    // nonpositive.
    for (const Array<Range> &ranges :
         {rewritten->src_range, rewritten->dst_range}) {
      for (const Range &range : ranges) {
        if (tl::arith::CanProve(analyzer_, range->extent <= 0))
          return call;
      }
    }
    // Canonical L0C matrices use the trailing axes, as required by lowering.
    const size_t src_inner_axis = rewritten->src_range.size() - 1;

    // Use the same row/inner axes as lowering, including size-one UB views.
    // An unproven explicit stride must never fall back to dense geometry.
    auto dst_layout = TryNormalizeMTE2DLayout(rewritten->dst,
                                              rewritten->dst_range, Analyzer());
    if (!dst_layout) {
      LOG(WARNING) << "Ascend L0C->UB copy: cannot prove a contiguous 2D "
                      "destination layout for "
                   << rewritten->dst->name << "; automatic padding skipped. "
                   << "Shape: " << rewritten->dst.shape()
                   << ", strides: " << rewritten->dst.ty()->layout << ".";
      return call;
    }
    const size_t dst_inner_axis = dst_layout->modes[0].axis;
    PrimExpr dst_row_stride = dst_layout->modes[1].stride;
    const int dst_elem_bits =
        rewritten->dst.dtype().bits() * rewritten->dst.dtype().lanes();
    PrimExpr dst_row_stride_bytes = tl::arith::Simplify(
        analyzer_, AscendMTEBytesFromElements(dst_row_stride, dst_elem_bits));
    PrimExpr stride_remainder =
        floormod(dst_row_stride_bytes, tl::MakeConst(dst_row_stride_bytes.ty(),
                                                     kFixpipeAlignmentBytes));
    // The descriptor's stride must be aligned even for a single-row copy.
    if (!tl::arith::CanProveEqual(analyzer_, stride_remainder, 0)) {
      LOG(WARNING) << "Ascend L0C->UB copy: destination "
                   << rewritten->dst->name << " row stride is "
                   << dst_row_stride_bytes << " bytes; "
                   << (tl::arith::CanProve(analyzer_, stride_remainder != 0)
                           ? "it is not a multiple of "
                           : "cannot prove it is a multiple of ")
                   << kFixpipeAlignmentBytes
                   << " bytes. Pad the UB row stride to a 32-byte boundary.";
    }

    // Preserve the N split point and its 2:1 region ratio.
    int dual_dst_ctl = rewritten->dual_dst_ctl;
    if (dual_dst_ctl == 2)
      return call;

    ICHECK_GT(dst_elem_bits, 0);
    ICHECK_LE(dst_elem_bits, kFixpipeAlignmentBytes * 8);
    ICHECK_EQ((kFixpipeAlignmentBytes * 8) % dst_elem_bits, 0)
        << "Ascend L0C->UB destination dtype must divide a 32-byte fixpipe "
           "write unit, got "
        << rewritten->dst.dtype();
    PrimExpr source_width = rewritten->src_range[src_inner_axis]->extent;
    PrimExpr destination_width = rewritten->dst_range[dst_inner_axis]->extent;
    // Normal copy takes n_size from UB; M-split takes it from L0C.
    PrimExpr transfer_width =
        dual_dst_ctl == 1 ? source_width : destination_width;
    PrimExpr alignment_elements = tl::MakeConst(
        transfer_width.ty(), kFixpipeAlignmentBytes * 8 / dst_elem_bits);
    if (tl::arith::CanProveEqual(
            analyzer_, floormod(transfer_width, alignment_elements), 0))
      return call;

    PrimExpr padded_width = tl::arith::Simplify(
        analyzer_,
        floordiv(transfer_width + alignment_elements - 1, alignment_elements) *
            alignment_elements);
    // Keep a larger source tile without enlarging the normal copy's n_size.
    auto widen_region = [&](const PrimExpr &width) {
      return tl::arith::Simplify(
          analyzer_, max(width, tl::CastValue(padded_width, width.ty())));
    };
    PrimExpr padded_src =
        dual_dst_ctl == 1 ? padded_width : widen_region(source_width);
    PrimExpr padded_dst =
        dual_dst_ctl == 1 ? widen_region(destination_width) : padded_width;
    // A padded stride does not authorize writing beyond the logical shape
    // (in particular, beyond the last allocated row).
    PrimExpr dst_capacity =
        min(rewritten->dst.shape()[dst_inner_axis], dst_row_stride);
    auto fits = [&](const Range &range, const PrimExpr &width,
                    const PrimExpr &capacity) {
      return tl::arith::CanProve(
          analyzer_, range->begin >= 0 && range->begin + width <= capacity,
          sym::ProofStrength::kSymbolicBound);
    };
    // L0C's canonical shape already includes fractal padding and bounds reads.
    if (!fits(rewritten->src_range[src_inner_axis], padded_src,
              rewritten->src.shape()[src_inner_axis]) ||
        !fits(rewritten->dst_range[dst_inner_axis], padded_dst, dst_capacity)) {
      LOG(WARNING)
          << "Ascend L0C->UB copy: cannot prove padding n_size="
          << transfer_width << " to " << padded_width
          << " elements fits source " << rewritten->src->name << " (width "
          << rewritten->src.shape()[src_inner_axis] << ") and destination "
          << rewritten->dst->name << " (row capacity " << dst_capacity
          << "); automatic padding skipped. Ensure the regions "
             "fit the buffer shapes and UB row stride after 32-byte alignment.";
      return call;
    }

    // Select preserves empty copies and remains analyzable by layout bounds.
    auto pad_range = [&](const Range &range, const PrimExpr &width) {
      return Range::FromMinExtent(
          range->begin,
          tl::arith::Simplify(analyzer_,
                              Select(range->extent > 0,
                                     tl::CastValue(width, range->extent.ty()),
                                     range->extent)));
    };
    Array<Range> src_ranges = rewritten->src_range;
    src_ranges.Set(src_inner_axis,
                   pad_range(src_ranges[src_inner_axis], padded_src));
    Array<Range> dst_ranges = rewritten->dst_range;
    dst_ranges.Set(dst_inner_axis,
                   pad_range(dst_ranges[dst_inner_axis], padded_dst));
    auto *writer = call.CopyOnWrite();
    writer->args.Set(
        0, TensorRegion(rewritten->src, src_ranges, TensorRegionType()));
    writer->args.Set(
        1, TensorRegion(rewritten->dst, dst_ranges, TensorRegionType()));
    return call;
  }

  TensorVar RemapTensor(const TensorVar &tensor) const {
    return buffers_.Get(tensor).value_or(tensor);
  }

  Expr VisitExpr_(const VarNode *node) final {
    if (auto tensor = GetRef<Var>(node).as<TensorVar>())
      return RemapTensor(tensor.value());
    return Parent::VisitExpr_(node);
  }

  Stmt VisitStmt_(const BindNode *node) final {
    auto tensor = node->var.as<TensorVar>();
    if (!tensor || !buffers_.count(tensor.value()))
      return Parent::VisitStmt_(node);
    TensorVar rewritten = buffers_[tensor.value()];
    VarRemapSet(node->var, rewritten);
    if (const auto *call = node->value.as<CallNode>();
        call && call->op.same_as(alloc_tensor_op())) {
      return Bind(rewritten,
                  Call(rewritten.ty(), call->op, call->args, call->attrs,
                       {rewritten.ty()}, call->loc),
                  node->loc);
    }
    return SeqStmt({});
  }

  PrimExpr VisitExpr_(const TensorLoadNode *node) final {
    TensorVar tensor = node->source.as_or_throw<TensorVar>();
    return TensorLoad(RemapTensor(tensor),
                      RewriteIndices_(tensor, node->indices), node->loc);
  }

  Stmt VisitStmt_(const TensorStoreNode *node) final {
    TensorVar tensor = node->dest.as_or_throw<TensorVar>();
    return TensorStore(
        RemapTensor(tensor), RewriteIndices_(tensor, node->indices),
        VisitExpr(node->value).as_or_throw<PrimExpr>(), node->loc);
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode *node, InplaceMode) final {
    return RewriteRegion_(GetRef<TensorRegion>(node));
  }

  Expr VisitExpr_(const CallNode *node) final {
    if (node->op.same_as(Op::Get("tl.layout_hint"))) {
      auto tensor = node->args[0].as_or_throw<TensorVar>();
      auto rewritten = RemapTensor(tensor);
      auto attributes = node->attrs.value().as_or_throw<DictAttrs>();
      auto layout = attributes.GetAttr<Layout>("layout").value();
      attributes.CopyOnWrite()->dict.Set(
          "layout", canonical_layouts_.Get(rewritten).value_or(layout));
      return Call(node->ty, node->op, {rewritten}, attributes, node->ty_args,
                  node->loc);
    }
    if (node->op.same_as(region())) {
      TensorRegion region = NormalizeToBufferRegion(GetRef<Call>(node));
      return RewriteRegion_(region);
    }
    if (IsAscendCopyCall(node)) {
      Copy original(node->args, CallAnnotations(node));
      Call call = Parent::VisitExpr_(node).as_or_throw<Call>();
      if (IsL1Buffer(original->src) &&
          (IsL0ABuffer(original->dst) || IsL0BBuffer(original->dst) ||
           IsL0SFBuffer(original->dst))) {
        return NormalizeL0Copy_(call);
      }
      if (IsL0CBuffer(original->src) && IsSharedBuffer(original->dst))
        return NormalizeFixpipeCopy_(call);
      if (!IsGlobalBuffer(original->src) || !IsL1Buffer(original->dst) ||
          !buffers_.count(original->dst)) {
        return call;
      }
      bool source_has_tail = false;
      for (size_t axis = 0; axis < original->src_range.size(); ++axis) {
        source_has_tail |=
            !tl::arith::CanProve(analyzer_,
                                 original->src_range[axis]->begin +
                                         original->src_range[axis]->extent <=
                                     original->src.shape()[axis],
                                 sym::ProofStrength::kSymbolicBound);
      }
      if (source_has_tail) {
        Copy rewritten(call->args, CallAnnotations(call.get()));
        auto ranges = rewritten->dst_range;
        const auto &logical_shape = anchors_[Root(original->dst)].shape();
        for (size_t axis = 0; axis < ranges.size(); ++axis) {
          if (tl::arith::CanProve(analyzer_, ranges[axis]->begin == 0) &&
              tl::arith::CanProve(analyzer_, ranges[axis]->extent ==
                                                 logical_shape[axis])) {
            ranges.Set(axis,
                       Range::FromMinExtent(ranges[axis]->begin,
                                            rewritten->dst.shape()[axis]));
          }
        }
        call.CopyOnWrite()->args.Set(
            1, TensorRegion(rewritten->dst, ranges, TensorRegionType()));
      }
      return call;
    }
    if (node->op.same_as(ptr_byte_offset_op()) ||
        node->op.same_as(tensor_data_ptr_op()) ||
        (node->op.same_as(reinterpret_op()) && node->ty.as<PtrTypeNode>())) {
      if (auto access = storage_->Resolve(GetRef<Expr>(node));
          access && anchors_.count(access->root)) {
        auto anchor = anchors_[access->root];
        auto pointer = node->ty.as_or_throw<PtrType>();
        ICHECK(pointer->element_type == anchor.dtype())
            << "Cube raw pointer must use the canonical operand dtype "
            << anchor.dtype();
        PrimExpr bits = VisitExpr(access->bit_offset).as_or_throw<PrimExpr>();
        int element_bits = ascend::ElementBits(anchor.dtype());
        ICHECK(
            tl::arith::CanProve(analyzer_, floormod(bits, element_bits) == 0))
            << "Cube pointer byte offset must address a whole element";
        PrimExpr remaining =
            tl::arith::Simplify(analyzer_, floordiv(bits, element_bits));
        Array<PrimExpr> indices(anchor.shape().size(),
                                IntImm(PrimType::Int(64), 0));
        for (size_t dimension = anchor.shape().size(); dimension > 0;
             --dimension) {
          size_t axis = dimension - 1;
          PrimExpr extent = ascend::StorageIndex(anchor.shape()[axis]);
          indices.Set(axis,
                      tl::arith::Simplify(
                          analyzer_,
                          axis == 0 ? remaining : floormod(remaining, extent)));
          remaining = floordiv(remaining, extent);
        }
        auto attrs = CallAnnotations(node);
        auto integer_attr = [&](const char *key, int fallback) -> PrimExpr {
          if (auto value = attrs.Get(key)) {
            if (auto expression = value->as<PrimExpr>())
              return expression.value();
            auto integer = GetIntAnnotation(value.value());
            TVM_FFI_CHECK(integer, TypeError)
                << key << " must be an integer expression";
            return IntImm(PrimType::Int(64), integer.value());
          }
          return IntImm(PrimType::Int(64), fallback);
        };
        return Call(pointer, tl::access_ptr(),
                    {TensorLoad(canonical_buffers_[access->root], indices),
                     integer_attr(attr::kAccessExtent, 1),
                     integer_attr(attr::kAccessMask, 3)},
                    node->attrs, {}, node->loc);
      }
    }
    return Parent::VisitExpr_(node);
  }

  Stmt VisitStmt_(const RegionStmtNode *node) final {
    auto key = ascend::ScheduleMarkerKey(node);
    if (!key.empty()) {
      Stmt body = VisitStmt(ascend::UnwrapScheduleMarker(node));
      Any payload = ascend::ScheduleMarkerPayload(node);
      payload = Mutate(payload).ValueOrUnchanged(payload);
      return ascend::MakeScheduleMarker(key, payload, body, node->loc,
                                        !node->result_vars.empty());
    }
    auto region = Parent::VisitStmt_(node).as_or_throw<RegionStmt>();
    if (auto annotation = node->attrs->dict.Get(attr::kLayoutMap)) {
      LayoutMap layouts;
      for (const auto &[tensor, layout] : annotation->cast<LayoutMap>()) {
        auto rewritten = RemapTensor(tensor);
        layouts.Set(rewritten,
                    canonical_layouts_.Get(rewritten).value_or(layout));
      }
      auto attrs = region->attrs->dict;
      attrs.Set(attr::kLayoutMap, layouts);
      region.CopyOnWrite()->attrs = DictAttrs(attrs);
    }
    return region;
  }

  BufferCollector *storage_;
  Map<Var, TensorVar> anchors_;
  Map<Var, TensorVar> canonical_buffers_;
  Map<TensorVar, TensorVar> buffers_;
  LayoutMap canonical_layouts_;
  Map<TensorVar, Layout> logical_maps_;
};
} // namespace
namespace transform {
tvm::transform::Pass NormalizeAscendFractalStorage() {
  auto pass_func = [](tirx::Function func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return FractalStorageNormalizer::Rewrite(std::move(func));
  };
  return tirx::transform::CreateFunctionPass(
      pass_func, 0, "tl.NormalizeAscendFractalStorage");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.NormalizeAscendFractalStorage",
                              NormalizeAscendFractalStorage);
}
} // namespace transform
} // namespace tl
} // namespace tvm
