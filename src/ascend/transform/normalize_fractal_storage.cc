/*!
 * \file ascend/transform/normalize_fractal_storage.cc
 * \brief Normalize native Cube allocations and aliases before scheduling.
 *
 * Added storage is not initialized here. Semantic copy regions and the padding
 * pass still determine which elements are valid and which tails need fills.
 */
#include "ascend/ir/task.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/transform/source_scopes.h"
#include "ir/alias_analysis.h"
#include "ir/annotations.h"
#include "op/builtin.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/stmt_functor.h>

#include "ir/stmt_functor.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;
using namespace prim;

bool SameDType(PrimType a, PrimType b) {
  return a.MatchesElementType(b.code(), b.bits()) && a.lanes() == b.lanes();
}

bool IsCubeStorage(const TensorVar &tensor) {
  return IsL1Buffer(tensor) || IsL0ABuffer(tensor) || IsL0BBuffer(tensor) ||
         IsL0CBuffer(tensor);
}

class FractalStorageNormalizer : public StmtExprMutatorWithAnalyzer {
public:
  explicit FractalStorageNormalizer(sym::AnalyzerObj *analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer) {}
  using StmtExprMutatorWithAnalyzer::Mutate_;

  tirx::Function Apply(tirx::Function function) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    CheckWellFormed(function);
    MarkTensorParamShapes(function);
    CollectAliases(function);
    Collect(function->body);
    for (const auto &[identity, tensors] : aliases_.groups) {
      auto owner = owners_.Get(identity);
      if (!owner)
        continue;
      Optional<TensorVar> anchor;
      for (const auto &tensor : tensors) {
        auto layout = layouts_.Get(tensor);
        if (!layout || !IsCubeStorage(tensor) ||
            !IsAscendFractalLayout(*layout))
          continue;
        if (anchor) {
          AscendFractalLayoutInfo a, b;
          TVM_FFI_CHECK(TryExtractAscendFractalLayout(layouts_.at(*anchor),
                                                      *anchor, &a) &&
                            TryExtractAscendFractalLayout(*layout, tensor, &b),
                        ValueError);
          TVM_FFI_CHECK(a.kind == b.kind &&
                            layouts_.at(*anchor)->IsEqual(layout->get()) &&
                            anchor.value().dtype().bits() ==
                                tensor.dtype().bits(),
                        ValueError)
              << "Conflicting canonical Cube views of " << tensor->name;
          if (used_.count(tensor) && !used_.count(*anchor))
            anchor = tensor;
        } else {
          anchor = tensor;
        }
      }
      if (!anchor)
        continue;
      anchors_.Set(identity, *anchor);
      RegisterGroup(*owner, *anchor, tensors);
    }
    if (replacements_.empty())
      return function;
    SeqStmt body = Mutate(function->body).ValueOrUnchanged(function->body);
    function.CopyOnWrite()->body = body;
    CheckWellFormed(function);
    return function;
  }

  UnchangedOr<Expr> Mutate_(const VarObj *op, InplaceMode mode) final {
    if (auto tensor = GetRef<Var>(op).as<TensorVar>())
      if (auto replacement = replacements_.Get(*tensor))
        return *replacement;
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (auto tensor = op->var.as<TensorVar>()) {
      if (auto replacement = replacements_.Get(*tensor)) {
        Expr value = op->value;
        std::optional<ascend::ScheduledBinding> scheduled;
        if (ascend::IsScheduledBind(value.as<CallObj>())) {
          scheduled = ascend::ReadScheduledBinding(value.as<CallObj>());
          value = scheduled->value;
        }
        const auto *call = value.as<CallObj>();
        TVM_FFI_CHECK(call, ValueError)
            << "Cube storage needs an explicit native allocation or view";
        if (call->op.same_as(decl_tensor_op()))
          return SeqStmt(Array<Stmt>{}, op->loc);
        TVM_FFI_CHECK(call->op.same_as(alloc_tensor_op()), ValueError)
            << "Cube owner must be a native alloc_tensor binding";
        Array<Expr> args;
        for (const Expr &arg : call->args)
          args.push_back(Mutate(arg, mode).ValueOrUnchanged(arg));
        Bind result(*replacement,
                    Call(replacement->ty(), call->op, args, call->attrs,
                         {replacement->ty()}, call->loc),
                    op->loc);
        return scheduled ? ascend::MakeScheduledBinding(
                               result, scheduled->stage, scheduled->task)
                         : result;
      }
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                InplaceMode mode) final {
    auto tensor = op->source.as_or_throw<TensorVar>();
    if (!replacements_.count(tensor))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    return TensorLoad(replacements_.at(tensor),
                      RewriteIndices(tensor, op->indices, mode), op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op, InplaceMode mode) final {
    auto tensor = op->dest.as_or_throw<TensorVar>();
    if (!replacements_.count(tensor))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    return TensorStore(
        replacements_.at(tensor), RewriteIndices(tensor, op->indices, mode),
        Mutate(op->value, mode).ValueOrUnchanged(op->value), op->loc);
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionObj *op, InplaceMode mode) final {
    auto tensor = op->source.as_or_throw<TensorVar>();
    if (!replacements_.count(tensor))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    Array<Range> ranges;
    for (const Range &range : op->region)
      ranges.push_back(Range::FromMinExtent(
          Mutate(range->begin, mode).ValueOrUnchanged(range->begin),
          Mutate(range->extent, mode).ValueOrUnchanged(range->extent)));
    auto anchor = anchors_.at(aliases_.Identity(tensor));
    if (!StructuralEqual()(tensor.shape(), anchor.shape())) {
      for (size_t i = 1; i < ranges.size(); ++i)
        TVM_FFI_CHECK(GetSyncedAnalyzer()->CanProve(ranges[i]->extent <=
                                                    tensor.shape()[i]),
                      ValueError)
            << "Cube region crosses an original reshape axis: "
            << GetRef<TensorRegion>(op);
      auto mapped = logical_maps_.at(tensor)->MapRegionBounds(ranges);
      PrimExpr old_size = IntImm::Int32(1), new_size = IntImm::Int32(1);
      for (const Range &range : ranges)
        old_size = old_size * range->extent;
      for (const Range &range : mapped)
        new_size = new_size * range->extent;
      TVM_FFI_CHECK(GetSyncedAnalyzer()->CanProveEqual(old_size, new_size),
                    ValueError)
          << "Reshaped Cube region is not a contiguous rectangular matrix: "
          << GetRef<TensorRegion>(op);
      ranges = mapped;
    }
    return TensorRegion(replacements_.at(tensor), ranges, op->ty, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (!IsAnnotationCall(call))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    auto statement =
        MakeAnnotation(RewriteMetadata(ReadAnnotations(call)), op->loc);
    if (auto sequence = statement.as<SeqStmt>())
      return StmtExprMutatorWithAnalyzer::Mutate_(sequence->get(), mode)
          .ValueOrUnchanged(*sequence);
    auto value = statement.as_or_throw<Evaluate>()->value;
    value = StmtExprMutatorWithAnalyzer::Mutate_(
                value.as_or_throw<Call>().get(), mode)
                .ValueOrUnchanged(value);
    return Evaluate(value, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (op->op->name != "tl.attr")
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    auto fields = ReadAnnotations(op);
    String key = fields.at("key").cast<String>();
    if (key == "layout_map")
      fields.Set("value", RewriteMetadata({{"layout_map", fields.at("value")}})
                              .at("layout_map"));
    else if (key == "tl.kernel")
      fields.Set("value",
                 RewriteMetadata(fields.at("value").cast<Map<String, Any>>()));
    auto encoded = EncodeAnnotations(fields);
    auto prepared = GetRef<RegionStmt>(op);
    prepared.CopyOnWrite()->args = {encoded.payload};
    auto attrs = op->attrs->dict;
    attrs.Set("schema_version", 1);
    attrs.Set("annotation_schema", encoded.schema);
    prepared.CopyOnWrite()->attrs = DictAttrs(attrs);
    return StmtExprMutatorWithAnalyzer::Mutate_(prepared.get(), mode);
  }

  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    if (op->op.same_as(tensor_data_ptr_op())) {
      if (auto tensor = op->args[0].as<TensorVar>()) {
        if (auto replacement = replacements_.Get(*tensor))
          return tirx::reinterpret(op->ty, replacement->data(), op->loc);
      }
    }
    if (op->op.same_as(tl::ptr_access())) {
      auto identity = aliases_.Identity(op->args[0]);
      if (auto anchor = anchors_.Get(identity)) {
        auto dtype = op->args[0]
                         ->ty.as_or_throw<PtrType>()
                         ->element_type.as_or_throw<PrimType>();
        TVM_FFI_CHECK(SameDType(dtype, anchor->dtype()), ValueError)
            << "Cube raw pointer must use the canonical operand dtype "
            << anchor->dtype() << "; take the pointer through that view";
        PrimExpr remaining =
            Mutate(op->args[1].as_or_throw<PrimExpr>(), mode)
                .ValueOrUnchanged(op->args[1].as_or_throw<PrimExpr>());
        Array<PrimExpr> indices(anchor->shape().size(), IntImm::Int32(0));
        for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
          indices.Set(i, GetSyncedAnalyzer()->Simplify(
                             i == 0 ? remaining
                                    : FloorMod(remaining, anchor->shape()[i])));
          remaining = FloorDiv(remaining, anchor->shape()[i]);
        }
        TensorVar canonical = replacements_.at(*anchor);
        return Call(canonical.ty()->DataPtrType(), Op::Get("tl.access_ptr"),
                    {TensorLoad(canonical, indices),
                     Mutate(op->args[2], mode).ValueOrUnchanged(op->args[2]),
                     Mutate(op->args[3], mode).ValueOrUnchanged(op->args[3])},
                    op->attrs, {}, op->loc);
      }
    }
    if (!IsAscendCopyCall(op))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    auto decoded = DecodeTileOpArguments(op);
    AscendCopy original(decoded.operands, decoded.annotations);
    Call result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                      .ValueOrUnchanged(GetRef<Expr>(op))
                      .as_or_throw<Call>();
    if (IsL1Buffer(original->src) &&
        (IsL0ABuffer(original->dst) || IsL0BBuffer(original->dst) ||
         IsL0SFBuffer(original->dst)))
      return NormalizeL0Copy(result);
    if (IsL0CBuffer(original->src) && IsSharedBuffer(original->dst))
      return NormalizeFixpipeCopy(result);
    if (!IsGlobalBuffer(original->src) || !IsL1Buffer(original->dst) ||
        !replacements_.count(original->dst))
      return result;
    bool source_has_tail = false;
    for (size_t i = 0; i < original->src_range.size(); ++i)
      source_has_tail |= !GetSyncedAnalyzer()->CanProve(
          original->src_range[i]->begin + original->src_range[i]->extent <=
              original->src.shape()[i],
          sym::ProofStrength::kSymbolicBound);
    if (!source_has_tail)
      return result;
    auto rewritten_args = DecodeTileOpArguments(result.get());
    AscendCopy rewritten(rewritten_args.operands, rewritten_args.annotations);
    Array<Range> ranges = rewritten->dst_range;
    auto logical_shape = anchors_.at(aliases_.Identity(original->dst)).shape();
    for (size_t i = 0; i < ranges.size(); ++i) {
      if (GetSyncedAnalyzer()->CanProveEqual(ranges[i]->begin, 0) &&
          GetSyncedAnalyzer()->CanProveEqual(ranges[i]->extent,
                                             logical_shape[i]))
        ranges.Set(i, Range::FromMinExtent(ranges[i]->begin,
                                           rewritten->dst.shape()[i]));
    }
    rewritten_args.operands.Set(
        1,
        TensorRegion(rewritten->dst, ranges, TensorRegionType(), result->loc));
    return MakeAnnotatedCall(result->ty, result->op.as_or_throw<Op>(),
                             rewritten_args.operands,
                             rewritten_args.annotations, result->loc);
  }

private:
  void CollectAliases(const tirx::Function &function) {
    // A scheduled_bind evaluates its first operand and defines the same
    // native Var. Its scheduling metadata must not hide allocation or view
    // provenance from a pass run on already scheduled input.
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const BindObj *binding) -> ffi::Expected<ffi::WalkResult> {
          Expr value = binding->value;
          if (ascend::IsScheduledBind(value.as<CallObj>()))
            value = ascend::ReadScheduledBinding(value.as<CallObj>()).value;
          aliases_.bindings.Set(binding->var, value);

          return ffi::WalkResult::Advance();
        });
    for (const Var &param : function->params)
      if (auto tensor = param.as<TensorVar>())
        aliases_.Record(*tensor);
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (auto tensor = node.as<TensorVar>())
            aliases_.Record(*tensor);
          return ffi::WalkResult::Advance();
        });
  }

  void Collect(const Stmt &body) {
    auto record = [&](const Map<String, Any> &metadata) {
      if (auto value = metadata.Get("layout_map")) {
        for (const auto &[tensor, layout] : value->cast<LayoutMap>()) {
          if (auto previous = layouts_.Get(tensor))
            TVM_FFI_CHECK((*previous)->IsEqual(layout.get()), ValueError)
                << "Conflicting layouts for " << tensor->name;
          layouts_.Set(tensor, layout);
        }
      }
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        body, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *bind = node.as<BindObj>()) {
            Expr value = bind->value;
            if (ascend::IsScheduledBind(value.as<CallObj>()))
              value = ascend::ReadScheduledBinding(value.as<CallObj>()).value;
            const auto *call = value.as<CallObj>();
            if (call && call->op.same_as(alloc_tensor_op())) {
              if (auto tensor = bind->var.as<TensorVar>())
                owners_.Set(*tensor, *tensor);
            }
          }
          if (const auto *load = node.as<TensorLoadObj>())
            used_.insert(load->source.as_or_throw<TensorVar>());
          if (const auto *store = node.as<TensorStoreObj>())
            used_.insert(store->dest.as_or_throw<TensorVar>());
          if (const auto *region = node.as<TensorRegionObj>())
            used_.insert(region->source.as_or_throw<TensorVar>());
          if (const auto *call = node.as<CallObj>(); IsAnnotationCall(call))
            record(ReadAnnotations(call));
          if (const auto *region = node.as<RegionStmtObj>();
              region && region->op->name == "tl.attr") {
            auto fields = ReadAnnotations(region);
            String key = fields.at("key").cast<String>();
            if (key == "tl.kernel")
              record(fields.at("value").cast<Map<String, Any>>());
            else
              record({{key, fields.at("value")}});
          }
          return ffi::WalkResult::Advance();
        });
    canonical_layouts_ = layouts_;
  }

  void RegisterGroup(const TensorVar &owner, const TensorVar &anchor,
                     const Array<TensorVar> &tensors) {
    AscendFractalLayoutInfo info;
    TVM_FFI_CHECK(
        TryExtractAscendFractalLayout(layouts_.at(anchor), anchor, &info),
        ValueError);
    Array<PrimExpr> shape = anchor.shape();
    TVM_FFI_CHECK(shape.size() >= 2, ValueError);
    for (size_t i = shape.size() - 2; i < shape.size(); ++i) {
      bool row = i == shape.size() - 2;
      PrimExpr align =
          row == (info.c0_axis == LogicalAxis::kRow) ? info.c0 : info.row_frac;
      shape.Set(i, GetSyncedAnalyzer()->Simplify(
                       FloorDiv(shape[i] + align - 1, align) * align));
    }
    auto type = make_object<TensorTypeObj>(*owner.ty().get());
    type->shape = shape;
    type->dtype = anchor.dtype();
    type->layout = std::nullopt;
    auto anchor_pattern = MatchShapeStridePattern(anchor.ty());
    TVM_FFI_CHECK(anchor_pattern.has_value(), ValueError)
        << "Cube storage requires a strided logical tensor";
    PrimExpr origin = anchor_pattern.value().get<2>();
    TVM_FFI_CHECK(GetSyncedAnalyzer()->CanProve(origin >= 0), ValueError)
        << "Cube allocation origin must be nonnegative";
    // Native TensorType carries its physical origin explicitly. Retain that
    // prefix once while rebuilding the row strides for the padded matrix.
    if (!IsZero(origin)) {
      Array<tirx::Iter> iters;
      PrimExpr stride = IntImm(origin.ty(), 1);
      for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        iters.insert(iters.begin(),
                     tirx::Iter(shape[i], stride, tirx::Axis::Get("m")));
        stride = stride * shape[i];
      }
      type->layout =
          tirx::TileLayout(iters, {}, {{tirx::Axis::Get("m"), origin}});
    }
    TensorVar canonical(owner->name, TensorType(type), owner->loc);
    Layout layout;
    switch (info.kind) {
    case AscendFractalKind::kMajorK:
      layout = MakeAscendMajorKLayout(canonical);
      break;
    case AscendFractalKind::kMajorMN:
      layout = MakeAscendMajorMNLayout(canonical);
      break;
    case AscendFractalKind::kL0C:
      layout = MakeAscendL0CLayout(canonical);
      break;
    case AscendFractalKind::kSF:
      layout = MakeAscendSFLayout(canonical);
      break;
    }
    canonical_layouts_.Set(canonical, layout);
    Array<PrimExpr> identity;
    for (size_t i = 0; i < anchor.shape().size(); ++i)
      identity.push_back(InputPlaceholder(i));
    Layout logical(anchor.shape(), identity);
    for (const TensorVar &tensor : tensors) {
      TVM_FFI_CHECK(tensor.dtype().lanes() == 1 &&
                        tensor.dtype().bits() == anchor.dtype().bits(),
                    ValueError)
          << "Cube aliases must preserve element width: " << tensor->name;
      TVM_FFI_CHECK(!used_.count(tensor) ||
                        SameDType(tensor.dtype(), anchor.dtype()),
                    ValueError)
          << "Cube alias accesses must use the canonical operand dtype: "
          << tensor->name;
      auto pattern = MatchShapeStridePattern(tensor.ty());
      TVM_FFI_CHECK(pattern && GetSyncedAnalyzer()->CanProveEqual(
                                   pattern.value().get<2>(), origin),
                    ValueError)
          << "Cube aliases require the same element origin and dense logical "
             "strides";
      auto strides = pattern.value().get<1>();
      PrimExpr stride = IntImm::Int32(1);
      for (int i = static_cast<int>(tensor.shape().size()) - 1; i >= 0; --i) {
        TVM_FFI_CHECK(
            GetSyncedAnalyzer()->CanProveEqual(tensor.shape()[i], 1) ||
                GetSyncedAnalyzer()->CanProveEqual(strides[i], stride),
            ValueError)
            << "Cube aliases require dense logical strides: " << tensor->name;
        stride = stride * tensor.shape()[i];
      }
      logical_maps_.Set(tensor,
                        logical->Reshape(tensor.shape(), GetSyncedAnalyzer()));
      replacements_.Set(tensor, canonical);
    }
  }

  Array<PrimExpr> RewriteIndices(const TensorVar &tensor,
                                 const Array<PrimExpr> &indices,
                                 InplaceMode mode) {
    auto rewritten = indices.Map([&](const PrimExpr &index) {
      return Mutate(index, mode).ValueOrUnchanged(index);
    });
    return logical_maps_.at(tensor)->Forward(rewritten).Map(
        [&](const PrimExpr &index) {
          return GetSyncedAnalyzer()->Simplify(index);
        });
  }

  Map<String, Any> RewriteMetadata(Map<String, Any> metadata) {
    if (auto value = metadata.Get("layout_map")) {
      LayoutMap result;
      for (const auto &[tensor, layout] : value->cast<LayoutMap>()) {
        TensorVar replacement = replacements_.Get(tensor).value_or(tensor);
        result.Set(replacement,
                   canonical_layouts_.Get(replacement).value_or(layout));
      }
      metadata.Set("layout_map", result);
    }
    return metadata;
  }

  Call NormalizeL0Copy(Call call) {
    auto decoded = DecodeTileOpArguments(call.get());
    AscendCopy copy(decoded.operands, decoded.annotations);
    for (const auto &ranges : {copy->src_range, copy->dst_range})
      for (const Range &range : ranges)
        if (GetSyncedAnalyzer()->CanProve(range->extent <= 0))
          return call;
    // A previously normalized dynamic tile already carries its empty-copy
    // Select. Validate it again, but retain provably equal original ranges
    // instead of accumulating equivalent nested guards on repeated passes.
    auto stable_region = [&](const Expr &original, TensorRegion normalized) {
      auto previous = NormalizeToTensorRegion(original);
      auto ranges = normalized->region;
      if (ranges.size() == previous->region.size()) {
        for (size_t i = 0; i < ranges.size(); ++i) {
          PrimExpr begin = ranges[i]->begin, extent = ranges[i]->extent;
          if (GetSyncedAnalyzer()->CanProveEqual(begin,
                                                 previous->region[i]->begin))
            begin = previous->region[i]->begin;
          if (GetSyncedAnalyzer()->CanProveEqual(extent,
                                                 previous->region[i]->extent))
            extent = previous->region[i]->extent;
          ranges.Set(i, Range::FromMinExtent(begin, extent));
        }
      }
      return TensorRegion(normalized->source, ranges, normalized->ty,
                          normalized->loc);
    };
    if (IsL0SFBuffer(copy->dst)) {
      decoded.operands.Set(0, stable_region(decoded.operands[0],
                                            ascend::GetL1ToL0SFSourceRegion(
                                                *copy.get(), canonical_layouts_,
                                                GetSyncedAnalyzer())));
    } else {
      auto regions = ascend::GetL1ToL0CopyRegions(
          *copy.get(), canonical_layouts_, GetSyncedAnalyzer());
      decoded.operands.Set(0,
                           stable_region(decoded.operands[0], regions.source));
      decoded.operands.Set(
          1, stable_region(decoded.operands[1], regions.destination));
      AscendFractalRegionInfo normalized;
      TVM_FFI_CHECK(TryExtractAscendFractalRegion(
                        canonical_layouts_.at(copy->dst),
                        regions.destination->region, &normalized),
                    ValueError);
      if (!GetSyncedAnalyzer()->CanProveEqual(normalized.outer1->extent,
                                              regions.pitch))
        decoded.annotations.Set("l0_pitch", regions.pitch);
    }
    return MakeAnnotatedCall(call->ty, call->op.as_or_throw<Op>(),
                             decoded.operands, decoded.annotations, call->loc);
  }

  Call NormalizeFixpipeCopy(Call call) {
    constexpr int kFixpipeAlignmentBytes = 32;
    auto decoded = DecodeTileOpArguments(call.get());
    AscendCopy rewritten(decoded.operands, decoded.annotations);
    ICHECK_GE(rewritten->src_range.size(), 2U);
    ICHECK_EQ(rewritten->src_range.size(), rewritten->src.shape().size());
    // Lowering skips the DMA if any source or destination extent is
    // nonpositive.
    for (const Array<Range> &ranges :
         {rewritten->src_range, rewritten->dst_range}) {
      for (const Range &range : ranges) {
        if (GetSyncedAnalyzer()->CanProve(range->extent <= 0))
          return call;
      }
    }
    // Canonical L0C matrices use the trailing axes, as required by lowering.
    const size_t src_inner_axis = rewritten->src_range.size() - 1;

    // Use the same row/inner axes as lowering, including size-one UB views.
    // An unproven explicit stride must never fall back to dense geometry.
    auto dst_layout = TryNormalizeMTE2DLayout(
        rewritten->dst, rewritten->dst_range, GetSyncedAnalyzer());
    if (!dst_layout) {
      LOG(WARNING) << "Ascend L0C->UB copy: cannot prove a contiguous 2D "
                      "destination layout for "
                   << rewritten->dst->name << "; automatic padding skipped. "
                   << "Shape: " << rewritten->dst.shape()
                   << ", strides: " << rewritten->dst.ty() << ".";
      return call;
    }
    const size_t dst_inner_axis = dst_layout->modes[0].axis;
    PrimExpr dst_row_stride = dst_layout->modes[1].stride;
    const int dst_elem_bits =
        rewritten->dst.dtype().bits() * rewritten->dst.dtype().lanes();
    PrimExpr dst_row_stride_bytes = GetSyncedAnalyzer()->Simplify(
        AscendMTEBytesFromElements(dst_row_stride, dst_elem_bits));
    PrimExpr stride_remainder =
        FloorMod(dst_row_stride_bytes,
                 MakeConst(dst_row_stride_bytes.ty(), kFixpipeAlignmentBytes));
    // The descriptor's stride must be aligned even for a single-row copy.
    if (!GetSyncedAnalyzer()->CanProveEqual(stride_remainder, 0)) {
      LOG(WARNING) << "Ascend L0C->UB copy: destination "
                   << rewritten->dst->name << " row stride is "
                   << dst_row_stride_bytes << " bytes; "
                   << (GetSyncedAnalyzer()->CanProve(stride_remainder != 0)
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
        << rewritten->dst.ty();
    PrimExpr source_width = rewritten->src_range[src_inner_axis]->extent;
    PrimExpr destination_width = rewritten->dst_range[dst_inner_axis]->extent;
    // Normal copy takes n_size from UB; M-split takes it from L0C.
    PrimExpr transfer_width =
        dual_dst_ctl == 1 ? source_width : destination_width;
    PrimExpr alignment_elements = MakeConst(
        transfer_width.ty(), kFixpipeAlignmentBytes * 8 / dst_elem_bits);
    if (GetSyncedAnalyzer()->CanProveEqual(
            FloorMod(transfer_width, alignment_elements), 0))
      return call;

    PrimExpr padded_width = GetSyncedAnalyzer()->Simplify(
        FloorDiv(transfer_width + alignment_elements - 1, alignment_elements) *
        alignment_elements);
    // Keep a larger source tile without enlarging the normal copy's n_size.
    auto widen_region = [&](const PrimExpr &width) {
      return GetSyncedAnalyzer()->Simplify(
          Max(width, cast(padded_width, width.ty())));
    };
    PrimExpr padded_src =
        dual_dst_ctl == 1 ? padded_width : widen_region(source_width);
    PrimExpr padded_dst =
        dual_dst_ctl == 1 ? widen_region(destination_width) : padded_width;
    // A padded stride does not authorize writing beyond the logical shape
    // (in particular, beyond the last allocated row).
    PrimExpr dst_capacity =
        Min(rewritten->dst.shape()[dst_inner_axis], dst_row_stride);
    auto fits = [&](const Range &range, const PrimExpr &width,
                    const PrimExpr &capacity) {
      return GetSyncedAnalyzer()->CanProve(range->begin >= 0 &&
                                               range->begin + width <= capacity,
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
          range->begin, GetSyncedAnalyzer()->Simplify(Select(
                            range->extent > 0, cast(width, range->extent.ty()),
                            range->extent)));
    };
    Array<Range> src_ranges = rewritten->src_range;
    src_ranges.Set(src_inner_axis,
                   pad_range(src_ranges[src_inner_axis], padded_src));
    Array<Range> dst_ranges = rewritten->dst_range;
    dst_ranges.Set(dst_inner_axis,
                   pad_range(dst_ranges[dst_inner_axis], padded_dst));
    decoded.operands.Set(0, TensorRegion(rewritten->src, src_ranges,
                                         TensorRegionType(), call->loc));
    decoded.operands.Set(1, TensorRegion(rewritten->dst, dst_ranges,
                                         TensorRegionType(), call->loc));
    return MakeAnnotatedCall(call->ty, call->op.as_or_throw<Op>(),
                             decoded.operands, decoded.annotations, call->loc);
  }

  TensorAliasGroups aliases_;
  Map<Expr, TensorVar> owners_, anchors_;
  Map<TensorVar, TensorVar> replacements_;
  LayoutMap layouts_, canonical_layouts_, logical_maps_;
  std::unordered_set<TensorVar, ObjectPtrHash, ObjectPtrEqual> used_;
};
} // namespace

namespace transform {
tvm::transform::Pass NormalizeAscendFractalStorage() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        sym::Analyzer analyzer;
        return ffi::make_object<FractalStorageNormalizer>(analyzer.get())
            ->Apply(std::move(function));
      },
      0, "tl.NormalizeAscendFractalStorage");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.NormalizeAscendFractalStorage",
                                   NormalizeAscendFractalStorage);
}
} // namespace transform
} // namespace tvm::tl
