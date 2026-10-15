/*!
 * \file normalize_fractal_storage.cc
 * \brief Normalize Cube storage and aliases before OOB handling and scheduling.
 *
 * This pass does not initialize the added padding. Copy/compute operations
 * must honor valid regions and their own padding requirements; consumers must
 * not assume all padded elements are zero.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>

#include "arith/ir_mutator_with_analyzer.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {
using namespace ffi;
using namespace tirx;
namespace {
class BufferCollector : public StmtExprVisitor {
public:
  Map<Var, Buffer> owners;
  Map<Buffer, Layout> layouts;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> buffers;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> used_buffers;

private:
  void VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
    buffers.insert(buffer);
    if (alloc_data)
      owners.Set(buffer->data, buffer);
    StmtExprVisitor::VisitBufferDef(buffer, alloc_data);
  }
  void VisitBufferUse(const Buffer &buffer) final {
    buffers.insert(buffer);
    used_buffers.insert(buffer);
  }
  void VisitStmt_(const SBlockNode *op) final {
    if (auto value = op->annotations.Get(attr::kLayoutMap)) {
      for (const auto &[buffer, layout] :
           value.value().cast<Map<Buffer, Layout>>()) {
        if (auto previous = layouts.Get(buffer))
          ICHECK(previous.value()->IsEqual(layout.get()))
              << "Conflicting layouts for " << buffer->name;
        layouts.Set(buffer, layout);
        buffers.insert(buffer);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

// Every alias has the same padded matrix shape and canonical layout. Its old
// reshape is absorbed into indices/regions here, not a special layout or hidden
// pitch. Ordinary capacity accounting and version expansion therefore see the
// complete storage through every alias. GM/UB and compact NZ are untouched.
class FractalStorageNormalizer : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    BufferCollector collector;
    collector(func->body);
    arith::Analyzer analyzer;
    analyzer.z3_prover.SetRLimit(50000);
    FractalStorageNormalizer rewriter(&analyzer);
    rewriter.MarkBufferMapShapes(func);
    for (const auto &[buffer, layout] : collector.layouts) {
      if (!IsAscendFractalLayout(layout) ||
          !(IsL1Buffer(buffer) || IsL0ABuffer(buffer) || IsL0BBuffer(buffer) ||
            IsL0CBuffer(buffer)) ||
          !collector.owners.count(buffer->data))
        continue;
      if (auto anchor = rewriter.anchors_.Get(buffer->data)) {
        AscendFractalLayoutInfo lhs, rhs;
        ICHECK(TryExtractAscendFractalLayout(collector.layouts[anchor.value()],
                                             anchor.value(), &lhs));
        ICHECK(TryExtractAscendFractalLayout(layout, buffer, &rhs));
        ICHECK(lhs.kind == rhs.kind &&
               collector.layouts[anchor.value()]->IsEqual(layout.get()) &&
               anchor.value()->dtype.bits() == buffer->dtype.bits())
            << "Conflicting canonical Cube views of " << buffer->name;
        if (collector.used_buffers.count(buffer) &&
            !collector.used_buffers.count(anchor.value()))
          rewriter.anchors_.Set(buffer->data, buffer);
      } else {
        rewriter.anchors_.Set(buffer->data, buffer);
      }
    }
    for (const Buffer &buffer : collector.buffers) {
      if (auto anchor = rewriter.anchors_.Get(buffer->data)) {
        ICHECK(!collector.used_buffers.count(buffer) ||
               buffer->dtype == anchor.value()->dtype)
            << "Cube alias accesses must use the canonical operand dtype: "
            << buffer->name << " has " << buffer->dtype << ", expected "
            << anchor.value()->dtype;
        rewriter.RegisterBuffer_(buffer, anchor.value(),
                                 collector.owners[buffer->data],
                                 collector.layouts[anchor.value()]);
      }
    }
    if (rewriter.buffers_.empty())
      return func;
    func.CopyOnWrite()->body = rewriter(func->body);
    return func;
  }

private:
  using Parent = arith::IRMutatorWithAnalyzer;
  explicit FractalStorageNormalizer(arith::Analyzer *analyzer)
      : Parent(analyzer) {}

  void RegisterBuffer_(const Buffer &buffer, const Buffer &anchor,
                       const Buffer &owner, const Layout &layout) {
    ICHECK_EQ(buffer->dtype.lanes(), 1);
    ICHECK_EQ(buffer->dtype.bits(), anchor->dtype.bits())
        << "Cube aliases must preserve element width: " << buffer->name
        << " has dtype " << buffer->dtype << ", canonical view " << anchor->name
        << " has dtype " << anchor->dtype;
    ICHECK(buffer->axis_separators.empty() && is_zero(buffer->elem_offset))
        << "Ascend fractal buffers require a zero base and no axis separators";
    if (!buffer->strides.empty()) {
      ICHECK_EQ(buffer->strides.size(), buffer->shape.size());
      PrimExpr stride = Integer(1);
      for (int i = static_cast<int>(buffer->shape.size()) - 1; i >= 0; --i) {
        ICHECK(analyzer_->CanProveEqual(buffer->shape[i], 1) ||
               analyzer_->CanProveEqual(buffer->strides[i], stride))
            << "Cube aliases require dense logical strides: " << buffer->name;
        stride = stride * buffer->shape[i];
      }
    }
    Array<PrimExpr> identity;
    for (size_t i = 0; i < anchor->shape.size(); ++i)
      identity.push_back(InputPlaceholder(i));
    logical_maps_.Set(
        buffer,
        Layout(anchor->shape, identity)->Reshape(buffer->shape, analyzer_));
    AscendFractalLayoutInfo info;
    ICHECK(TryExtractAscendFractalLayout(layout, anchor, &info));
    Array<PrimExpr> shape = anchor->shape;
    for (size_t axis = shape.size() - 2; axis < shape.size(); ++axis) {
      bool is_row = axis == shape.size() - 2;
      PrimExpr align = (is_row == (info.c0_axis == LogicalAxis::kRow))
                           ? info.c0
                           : info.row_frac;
      shape.Set(axis, analyzer_->Simplify(
                          floordiv(shape[axis] + align - 1, align) * align));
    }
    // Coalesce aliases to one Buffer identity and geometry so versioning
    // expands the allocation and every use together. An unused equal-width
    // backing allocation may adopt the canonical operand dtype.
    Buffer padded = canonical_buffers_.Get(buffer->data).value_or(owner);
    if (!StructuralEqual()(padded->shape, shape) || !padded->strides.empty() ||
        padded->dtype != anchor->dtype) {
      auto *writer = padded.CopyOnWrite();
      writer->shape = shape;
      writer->strides = {};
      writer->dtype = anchor->dtype;
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
    buffers_.Set(buffer, padded);
    canonical_buffers_.Set(buffer->data, padded);
    layouts_.Set(buffer, padded_layout);
  }

  Array<PrimExpr> RewriteIndices_(const Buffer &buffer,
                                  const Array<PrimExpr> &indices) {
    auto rewritten =
        indices.Map([&](PrimExpr index) { return VisitExpr(index); });
    if (auto layout = logical_maps_.Get(buffer))
      return layout.value()->Forward(rewritten).Map(
          [&](PrimExpr index) { return analyzer_->Simplify(index); });
    return rewritten;
  }
  BufferRegion RewriteRegion_(const BufferRegion &region) {
    Buffer buffer = region->buffer;
    Array<Range> ranges = region->region.Map([&](const Range &range) {
      return Range::FromMinExtent(VisitExpr(range->min),
                                  VisitExpr(range->extent));
    });
    if (!buffers_.count(buffer))
      return BufferRegion(buffer, ranges);
    if (!StructuralEqual()(buffer->shape, anchors_[buffer->data]->shape)) {
      // A dense reshape is injective only while trailing region extents do
      // not wrap across the original view's strides. Otherwise equal bounding
      // box volume alone can hide duplicated logical elements.
      for (size_t i = 1; i < ranges.size(); ++i)
        ICHECK(analyzer_->CanProve(ranges[i]->extent <= buffer->shape[i]))
            << "Cube region crosses an original reshape axis: " << region;
      Array<Range> mapped = logical_maps_[buffer]->MapRegionBounds(ranges);
      PrimExpr old_size = Integer(1), new_size = Integer(1);
      for (const Range &range : ranges)
        old_size = old_size * range->extent;
      for (const Range &range : mapped)
        new_size = new_size * range->extent;
      // A bounding box with holes cannot replace a tile operation's region.
      ICHECK(analyzer_->CanProveEqual(old_size, new_size))
          << "Reshaped Cube region is not a contiguous rectangular matrix: "
          << region;
      ranges = mapped;
    }
    return BufferRegion(buffers_[buffer], ranges);
  }
  PrimExpr MakeRegion_(const BufferRegion &region, Call original) {
    Array<PrimExpr> mins;
    for (const Range &range : region->region)
      mins.push_back(range->min);
    Array<PrimExpr> args{BufferLoad(region->buffer, mins), original->args[1]};
    for (const Range &range : region->region)
      args.push_back(range->extent);
    original.CopyOnWrite()->args = std::move(args);
    return original;
  }
  Call NormalizeFixpipeCopy_(Call call) {
    constexpr int kFixpipeAlignmentBytes = 32;
    AscendCopy rewritten(call->args, call->annotations);
    ICHECK_GE(rewritten->src_range.size(), 2U);
    ICHECK_EQ(rewritten->src_range.size(), rewritten->src->shape.size());
    // Lowering skips the DMA if any source or destination extent is
    // nonpositive.
    for (const Array<Range> &ranges :
         {rewritten->src_range, rewritten->dst_range}) {
      for (const Range &range : ranges) {
        if (analyzer_->CanProve(range->extent <= 0))
          return call;
      }
    }
    // Canonical L0C matrices use the trailing axes, as required by lowering.
    const size_t src_inner_axis = rewritten->src_range.size() - 1;

    // Use the same row/inner axes as lowering, including size-one UB views.
    // An unproven explicit stride must never fall back to dense geometry.
    auto dst_layout = TryNormalizeMTE2DLayout(rewritten->dst,
                                              rewritten->dst_range, analyzer_);
    if (!dst_layout) {
      LOG(WARNING) << "Ascend L0C->UB copy: cannot prove a contiguous 2D "
                      "destination layout for "
                   << rewritten->dst->name << "; automatic padding skipped. "
                   << "Shape: " << rewritten->dst->shape
                   << ", strides: " << rewritten->dst->strides << ".";
      return call;
    }
    const size_t dst_inner_axis = dst_layout->modes[0].axis;
    PrimExpr dst_row_stride = dst_layout->modes[1].stride;
    const int dst_elem_bits =
        rewritten->dst->dtype.bits() * rewritten->dst->dtype.lanes();
    PrimExpr dst_row_stride_bytes = analyzer_->Simplify(
        AscendMTEBytesFromElements(dst_row_stride, dst_elem_bits));
    PrimExpr stride_remainder =
        floormod(dst_row_stride_bytes, make_const(dst_row_stride_bytes.dtype(),
                                                  kFixpipeAlignmentBytes));
    // The descriptor's stride must be aligned even for a single-row copy.
    if (!analyzer_->CanProveEqual(stride_remainder, 0)) {
      LOG(WARNING) << "Ascend L0C->UB copy: destination "
                   << rewritten->dst->name << " row stride is "
                   << dst_row_stride_bytes << " bytes; "
                   << (analyzer_->CanProve(stride_remainder != 0)
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
        << rewritten->dst->dtype;
    PrimExpr source_width = rewritten->src_range[src_inner_axis]->extent;
    PrimExpr destination_width = rewritten->dst_range[dst_inner_axis]->extent;
    // Normal copy takes n_size from UB; M-split takes it from L0C.
    PrimExpr transfer_width =
        dual_dst_ctl == 1 ? source_width : destination_width;
    PrimExpr alignment_elements = make_const(
        transfer_width.dtype(), kFixpipeAlignmentBytes * 8 / dst_elem_bits);
    if (analyzer_->CanProveEqual(floormod(transfer_width, alignment_elements),
                                 0))
      return call;

    PrimExpr padded_width = analyzer_->Simplify(
        floordiv(transfer_width + alignment_elements - 1, alignment_elements) *
        alignment_elements);
    // Keep a larger source tile without enlarging the normal copy's n_size.
    auto widen_region = [&](const PrimExpr &width) {
      return analyzer_->Simplify(max(width, cast(width.dtype(), padded_width)));
    };
    PrimExpr padded_src =
        dual_dst_ctl == 1 ? padded_width : widen_region(source_width);
    PrimExpr padded_dst =
        dual_dst_ctl == 1 ? widen_region(destination_width) : padded_width;
    // A padded stride does not authorize writing beyond the logical shape
    // (in particular, beyond the last allocated row).
    PrimExpr dst_capacity =
        min(rewritten->dst->shape[dst_inner_axis], dst_row_stride);
    auto fits = [&](const Range &range, const PrimExpr &width,
                    const PrimExpr &capacity) {
      return analyzer_->CanProve(range->min >= 0 &&
                                     range->min + width <= capacity,
                                 arith::ProofStrength::kSymbolicBound);
    };
    // L0C's canonical shape already includes fractal padding and bounds reads.
    if (!fits(rewritten->src_range[src_inner_axis], padded_src,
              rewritten->src->shape[src_inner_axis]) ||
        !fits(rewritten->dst_range[dst_inner_axis], padded_dst, dst_capacity)) {
      LOG(WARNING)
          << "Ascend L0C->UB copy: cannot prove padding n_size="
          << transfer_width << " to " << padded_width
          << " elements fits source " << rewritten->src->name << " (width "
          << rewritten->src->shape[src_inner_axis] << ") and destination "
          << rewritten->dst->name << " (row capacity " << dst_capacity
          << "); automatic padding skipped. Ensure the regions "
             "fit the buffer shapes and UB row stride after 32-byte alignment.";
      return call;
    }

    // Select preserves empty copies and remains analyzable by layout bounds.
    auto pad_range = [&](const Range &range, const PrimExpr &width) {
      return Range::FromMinExtent(
          range->min, analyzer_->Simplify(Select(
                          range->extent > 0, cast(range->extent.dtype(), width),
                          range->extent)));
    };
    Array<Range> src_ranges = rewritten->src_range;
    src_ranges.Set(src_inner_axis,
                   pad_range(src_ranges[src_inner_axis], padded_src));
    Array<Range> dst_ranges = rewritten->dst_range;
    dst_ranges.Set(dst_inner_axis,
                   pad_range(dst_ranges[dst_inner_axis], padded_dst));
    auto *writer = call.CopyOnWrite();
    writer->args.Set(0, MakeRegion_(BufferRegion(rewritten->src, src_ranges),
                                    Downcast<Call>(writer->args[0])));
    writer->args.Set(1, MakeRegion_(BufferRegion(rewritten->dst, dst_ranges),
                                    Downcast<Call>(writer->args[1])));
    return call;
  }
  Buffer VisitBufferDef(const Buffer &buffer, bool) final {
    return VisitBufferUse(buffer);
  }
  Buffer VisitBufferUse(const Buffer &buffer) final {
    return buffers_.Get(buffer).value_or(buffer);
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = GetRef<BufferLoad>(op);
    auto *writer = load.CopyOnWrite();
    writer->indices = RewriteIndices_(op->buffer, op->indices);
    writer->buffer = VisitBufferUse(op->buffer);
    if (op->predicate.defined())
      writer->predicate = VisitExpr(op->predicate.value());
    return load;
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = GetRef<BufferStore>(op);
    auto *writer = store.CopyOnWrite();
    writer->value = VisitExpr(op->value);
    writer->indices = RewriteIndices_(op->buffer, op->indices);
    writer->buffer = VisitBufferUse(op->buffer);
    if (op->predicate.defined())
      writer->predicate = VisitExpr(op->predicate.value());
    return store;
  }
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(region())) {
      BufferRegion buffer_region = NormalizeToBufferRegion(GetRef<Call>(op));
      if (buffers_.count(buffer_region->buffer))
        return MakeRegion_(RewriteRegion_(buffer_region), GetRef<Call>(op));
    }
    if (IsAscendCopyCall(op)) {
      Copy copy(op->args, op->annotations);
      Call call = Downcast<Call>(Parent::VisitExpr_(op));
      if (IsL0CBuffer(copy->src) && IsSharedBuffer(copy->dst))
        return NormalizeFixpipeCopy_(call);
      if (!IsGlobalBuffer(copy->src) || !IsL1Buffer(copy->dst) ||
          !buffers_.count(copy->dst))
        return call;
      bool source_has_tail = false;
      for (size_t i = 0; i < copy->src_range.size(); ++i)
        source_has_tail |= !analyzer_->CanProve(
            copy->src_range[i]->min + copy->src_range[i]->extent <=
                copy->src->shape[i],
            arith::ProofStrength::kSymbolicBound);
      if (source_has_tail) {
        // An OOB fill needs complete destination axes. Enlarge only axes
        // covering the whole original matrix, never a partial slice. Keep the
        // GM region unchanged: extra storage does not authorize an extra read.
        Copy rewritten(call->args, call->annotations);
        Array<Range> ranges = rewritten->dst_range;
        const auto &logical_shape = anchors_[copy->dst->data]->shape;
        for (size_t i = 0; i < ranges.size(); ++i) {
          if (analyzer_->CanProveEqual(ranges[i]->min, 0) &&
              analyzer_->CanProveEqual(ranges[i]->extent, logical_shape[i]))
            ranges.Set(i, Range::FromMinExtent(ranges[i]->min,
                                               rewritten->dst->shape[i]));
        }
        call.CopyOnWrite()->args.Set(
            1, MakeRegion_(BufferRegion(rewritten->dst, ranges),
                           Downcast<Call>(call->args[1])));
      }
      return call;
    }
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      auto data = op->args[1].as<Var>();
      if (data && anchors_.count(data.value())) {
        Buffer anchor = anchors_[data.value()];
        ICHECK(op->args[0].dtype() == anchor->dtype)
            << "Cube raw pointer must use the canonical operand dtype "
            << anchor->dtype << "; take the pointer through that view";
        Buffer target = canonical_buffers_[data.value()];
        PrimExpr remaining = VisitExpr(op->args[2]);
        Array<PrimExpr> indices(anchor->shape.size(), Integer(0));
        for (int i = static_cast<int>(anchor->shape.size()) - 1; i >= 0; --i) {
          indices.Set(i, i == 0 ? remaining
                                : floormod(remaining, anchor->shape[i]));
          remaining = floordiv(remaining, anchor->shape[i]);
        }
        // Preserve the caller's extent; only the logical base address changes.
        return Call(op->dtype, tl::access_ptr(),
                    {BufferLoad(target, indices), VisitExpr(op->args[3]),
                     VisitExpr(op->args[4])},
                    op->annotations, op->span);
      }
    }
    return Parent::VisitExpr_(op);
  }
  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock prepared = GetRef<SBlock>(op);
    // The base mutator changes buffers but not region ranks. Rewrite explicit
    // access metadata before it sees a rank-changing alias.
    prepared.CopyOnWrite()->reads =
        op->reads.Map([&](const BufferRegion &r) { return RewriteRegion_(r); });
    prepared.CopyOnWrite()->writes = op->writes.Map(
        [&](const BufferRegion &r) { return RewriteRegion_(r); });
    for (const auto &match : op->match_buffers)
      ICHECK(!buffers_.count(match->buffer) &&
             !buffers_.count(match->source->buffer))
          << "Cube match_buffer regions are not supported; use a dense "
             "reshape/view";
    SBlock block = Downcast<SBlock>(Parent::VisitStmt_(prepared.get()));
    if (auto value = op->annotations.Get(attr::kLayoutMap)) {
      Map<Buffer, Layout> layouts;
      for (const auto &[buffer, layout] :
           value.value().cast<Map<Buffer, Layout>>())
        layouts.Set(buffers_.Get(buffer).value_or(buffer),
                    layouts_.Get(buffer).value_or(layout));
      block.CopyOnWrite()->annotations.Set(attr::kLayoutMap, layouts);
    }
    return block;
  }
  Map<Var, Buffer> anchors_;
  Map<Var, Buffer> canonical_buffers_;
  Map<Buffer, Buffer> buffers_;
  Map<Buffer, Layout> layouts_;
  Map<Buffer, Layout> logical_maps_;
};
} // namespace
namespace transform {
tirx::transform::Pass NormalizeAscendFractalStorage() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tirx::transform::PassContext &) {
    return FractalStorageNormalizer::Rewrite(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.NormalizeAscendFractalStorage", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.NormalizeAscendFractalStorage",
                              NormalizeAscendFractalStorage);
}
} // namespace transform
} // namespace tl
} // namespace tvm
