/*!
 * \file transform/common/pipeline_tensor.h
 * \brief Version native tensor definitions and explicit address operands.
 */
#ifndef TILELANG_TRANSFORM_COMMON_PIPELINE_TENSOR_H_
#define TILELANG_TRANSFORM_COMMON_PIPELINE_TENSOR_H_

#include "ir/stmt_functor.h"
#include "ir/tensor_utils.h"
#include "layout/expr_utils.h"
#include "op/builtin.h"
#include "pipeline_access.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl {

inline tirx::TensorVar ExpandPipelineTensor(const tirx::TensorVar &tensor,
                                            int versions,
                                            bool flatten_leading = false) {
  TVM_FFI_CHECK_GT(versions, 0, ValueError);
  auto type = ffi::make_object<tirx::TensorTypeObj>(*tensor.ty().get());
  PrimExpr count = IntImm(type->DefaultIndexType(), versions);
  ffi::Array<PrimExpr> inner_shape = type->shape;
  ffi::Array<PrimExpr> outer_shape;
  if (flatten_leading) {
    TVM_FFI_CHECK(!type->shape.empty(), ValueError);
    type->shape.Set(0, type->shape[0] * count);
    for (size_t i = 0; i < inner_shape.size(); ++i)
      outer_shape.push_back(i == 0 ? count : IntImm(inner_shape[i].ty(), 1));
  } else {
    type->shape.insert(type->shape.begin(), count);
    inner_shape.insert(inner_shape.begin(), IntImm(count.ty(), 1));
    for (size_t i = 0; i < inner_shape.size(); ++i)
      outer_shape.push_back(i == 0 ? count : IntImm(inner_shape[i].ty(), 1));
  }
  if (type->layout) {
    type->layout = type->layout.value().Tile(
        tirx::TileLayout::DefaultLayout(outer_shape), outer_shape, inner_shape);
  }
  return tirx::TensorVar(tensor->name, tirx::TensorType(type), tensor->loc);
}

inline Bind AllocatePipelineTensor(const tirx::TensorVar &tensor,
                                   const PipelineMemoryAnalysis *memory) {
  ffi::Optional<Attrs> attrs;
  if (auto definition = memory->definitions().Get(tensor))
    if (const auto *call = definition->as<CallObj>())
      attrs = call->attrs;
  return Bind(tensor,
              Call(tensor.ty(), tirx::alloc_tensor_op(), {}, attrs,
                   {tensor.ty()}, tensor->loc),
              tensor->loc);
}

// mode=kVersion indexes a leading version dimension; mode=kBarrier repeats
// slot indices in one flat barrier tensor; mode=kReconcile keeps a pipeline's
// own indices while reconciling sibling capacity, using slot zero outside any
// expanded pipeline. This class never mutates a TensorVar's type in place.
class PipelineTensorRewriter : public tl::LayoutMapMutator {
public:
  using tl::LayoutMapMutator::Mutate;
  using tl::LayoutMapMutator::Mutate_;
  enum class Mode { kVersion, kBarrier, kReconcile };

  PipelineTensorRewriter(
      ffi::Map<tirx::TensorVar, tirx::TensorVar> replacements,
      Mode mode = Mode::kReconcile,
      ffi::Optional<PrimExpr> iteration = std::nullopt,
      ffi::Optional<PrimExpr> parity_cycle = std::nullopt,
      ffi::Optional<PrimVar> original_loop = std::nullopt,
      ffi::Optional<PrimExpr> original_begin = std::nullopt,
      const PipelineMemoryAnalysis *memory = nullptr)
      : replacements_(std::move(replacements)), mode_(mode),
        iteration_(std::move(iteration)),
        parity_cycle_(std::move(parity_cycle)),
        original_loop_(std::move(original_loop)),
        original_begin_(std::move(original_begin)), memory_(memory) {}

  Stmt Rewrite(const Stmt &statement) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        statement, [&](const BindObj *bind) -> ffi::Expected<ffi::WalkResult> {
          local_definitions_.insert(bind->var);
          return ffi::WalkResult::Advance();
        });
    Stmt body = Mutate(statement).ValueOrUnchanged(statement);
    if (external_aliases_.empty())
      return body;
    ffi::Array<Stmt> sequence = external_aliases_;
    sequence.push_back(body);
    return SeqStmt(sequence);
  }

private:
  PrimExpr Version(const tirx::TensorVar &tensor) const {
    if (!iteration_)
      return IntImm(tensor.ty()->DefaultIndexType(), 0);
    // Tensor shape indices and the pipeline induction variable may have
    // different integer types. The native builder promotes them before
    // constructing FloorMod; the node constructor requires identical types.
    return floormod(iteration_.value(), tensor.shape()[0]);
  }

  ffi::Array<PrimExpr> RewriteIndices(const tirx::TensorVar &old_tensor,
                                      const tirx::TensorVar &tensor,
                                      ffi::Array<PrimExpr> indices) {
    for (size_t i = 0; i < indices.size(); ++i)
      indices.Set(i, Mutate(indices[i]).ValueOrUnchanged(indices[i]));
    if (mode_ == Mode::kBarrier) {
      TVM_FFI_CHECK_EQ(indices.size(), 1U, ValueError);
      indices.Set(0, iteration_.value() * old_tensor.shape()[0] + indices[0]);
    } else if (old_tensor.shape().size() < tensor.shape().size()) {
      TVM_FFI_CHECK_EQ(old_tensor.shape().size() + 1, tensor.shape().size(),
                       ValueError);
      indices.insert(indices.begin(), Version(tensor));
    }
    return indices;
  }

  ffi::UnchangedOr<Expr> Mutate_(const VarObj *op, InplaceMode mode) final {
    Var original = ffi::GetRef<Var>(op);
    if (auto tensor = original.as<tirx::TensorVar>())
      if (auto replacement = replacements_.Get(*tensor))
        return *replacement;
    if (memory_ && mode_ == Mode::kVersion &&
        !local_definitions_.count(original)) {
      if (auto cached = alias_remap_.Get(original))
        return *cached;
      auto root = memory_->Root(original);
      auto tensor = root ? root.value().as<tirx::TensorVar>() : std::nullopt;
      auto replacement = tensor ? replacements_.Get(*tensor) : std::nullopt;
      if (replacement) {
        // The alias was already evaluated in its original defining scope.
        // Shift that captured pointer; do not repeat a possibly mutable load
        // used by the original alias's byte-offset or shape expression.
        auto pointer_type = original->ty.as<PtrType>();
        auto tensor_type = original->ty.as<tirx::TensorType>();
        Expr pointer = pointer_type
                           ? Expr(original)
                           : Expr(Call(tensor_type.value()->DataPtrType(),
                                       tirx::tensor_data_ptr_op(), {original}));
        pointer = Call(pointer->ty, tirx::ptr_byte_offset_op(),
                       {pointer, VersionByteOffset(*tensor, *replacement)}, {},
                       {}, op->loc);
        Expr value = pointer;
        if (tensor_type)
          value = Call(*tensor_type, tirx::decl_tensor_op(), {pointer}, {},
                       {*tensor_type}, op->loc);
        Var alias(original->name, original->ty, original->loc);
        external_aliases_.push_back(Bind(alias, value, original->loc));
        alias_remap_.Set(original, alias);
        memory_->RecordGeneratedAlias(alias, value, *replacement);
        return alias;
      }
    }
    return tl::LayoutMapMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                     InplaceMode mode) final {
    auto old_tensor = op->source.as<tirx::TensorVar>();
    auto tensor = old_tensor ? replacements_.Get(*old_tensor) : std::nullopt;
    if (!tensor)
      return tl::LayoutMapMutator::Mutate_(op, mode);
    return TensorLoad(
        *tensor, RewriteIndices(*old_tensor, *tensor, op->indices), op->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op,
                                 InplaceMode mode) final {
    auto old_tensor = op->dest.as<tirx::TensorVar>();
    auto tensor = old_tensor ? replacements_.Get(*old_tensor) : std::nullopt;
    if (!tensor)
      return tl::LayoutMapMutator::Mutate_(op, mode);
    auto value = Mutate(op->value, mode).ValueOrUnchanged(op->value);
    return TensorStore(*tensor,
                       RewriteIndices(*old_tensor, *tensor, op->indices), value,
                       op->loc);
  }

  ffi::UnchangedOr<Expr> Mutate_(const TensorRegionObj *op,
                                 InplaceMode mode) final {
    auto old_tensor = op->source.as<tirx::TensorVar>();
    auto tensor = old_tensor ? replacements_.Get(*old_tensor) : std::nullopt;
    if (!tensor)
      return tl::LayoutMapMutator::Mutate_(op, mode);
    ffi::Array<Range> bounds;
    for (const Range &range : op->region)
      bounds.push_back(Range::FromMinExtent(
          Mutate(range->begin).ValueOrUnchanged(range->begin),
          Mutate(range->extent).ValueOrUnchanged(range->extent), range->loc));
    if (mode_ == Mode::kBarrier) {
      TVM_FFI_CHECK_EQ(bounds.size(), 1U, ValueError);
      PrimExpr offset = iteration_.value() * old_tensor->shape()[0];
      bounds.Set(0, Range::FromMinExtent(bounds[0]->begin + offset,
                                         bounds[0]->extent));
    } else if (old_tensor->shape().size() < tensor->shape().size()) {
      PrimExpr version = Version(*tensor);
      bounds.insert(bounds.begin(),
                    Range::FromMinExtent(version, IntImm(version.ty(), 1)));
    }
    return TensorRegion(*tensor, bounds, TensorRegionType(), op->loc);
  }

  PrimExpr VersionByteOffset(const tirx::TensorVar &old_tensor,
                             const tirx::TensorVar &tensor) {
    if (old_tensor.shape().size() == tensor.shape().size())
      return IntImm(tensor.ty()->DefaultIndexType(), 0);
    ffi::Array<PrimExpr> zero;
    for (const auto &extent : tensor.shape())
      zero.push_back(IntImm(extent.ty(), 0));
    auto selected = zero;
    selected.Set(0, Version(tensor));
    PrimExpr offset = TensorElementOffset(tensor, selected) -
                      TensorElementOffset(tensor, zero);
    PrimExpr bits = offset * IntImm(offset.ty(), tensor.dtype().bits() *
                                                     tensor.dtype().lanes());
    sym::Analyzer analyzer;
    TVM_FFI_CHECK(analyzer->CanProve(floormod(bits, 8) == 0), ValueError)
        << "A versioned raw tensor pointer must start at a byte boundary";
    return analyzer->Simplify(floordiv(bits, 8));
  }

  Expr VersionedPointer(const tirx::TensorVar &old_tensor,
                        const tirx::TensorVar &tensor, Location loc) {
    Expr pointer = Call(tensor.ty()->DataPtrType(), tirx::tensor_data_ptr_op(),
                        {tensor}, {}, {}, loc);
    if (mode_ != Mode::kVersion ||
        old_tensor.shape().size() == tensor.shape().size())
      return pointer;
    return Call(pointer->ty, tirx::ptr_byte_offset_op(),
                {pointer, VersionByteOffset(old_tensor, tensor)}, {}, {}, loc);
  }

  ffi::Map<ffi::String, ffi::Any>
  RewriteAnnotations(ffi::Map<ffi::String, ffi::Any> annotations) {
    if (auto raw = annotations.Get("layout_map")) {
      ffi::Map<Var, Layout> layouts;
      for (const auto &[key, value] :
           raw->cast<ffi::Map<ffi::Any, ffi::Any>>()) {
        auto var = key.cast<Var>();
        auto layout = value.cast<Layout>();
        if (auto old = var.as<tirx::TensorVar>()) {
          if (auto tensor = replacements_.Get(*old)) {
            if (tensor->shape().size() == old->shape().size() + 1)
              layout = layout->Expand({tensor->shape()[0]});
            var = *tensor;
          }
        }
        layouts.Set(var, layout);
      }
      annotations.Set("layout_map", layouts);
    }
    if (auto raw = annotations.Get("barrier_init")) {
      ffi::Map<Var, ffi::Array<PrimExpr>> initializers;
      for (const auto &[key, value] :
           raw->cast<ffi::Map<ffi::Any, ffi::Any>>()) {
        auto var = key.cast<Var>();
        ffi::Array<PrimExpr> counts;
        for (const auto &count : value.cast<ffi::Array<ffi::Any>>()) {
          if (auto expr = count.as<PrimExpr>())
            counts.push_back(*expr);
          else
            counts.push_back(IntImm::Int32(GetIntegerAnnotation(count)));
        }
        if (auto old = var.as<tirx::TensorVar>()) {
          if (auto tensor = replacements_.Get(*old)) {
            var = *tensor;
            auto slots = TryConstInt(tensor->shape()[0]);
            TVM_FFI_CHECK(slots && !counts.empty() &&
                              *slots % counts.size() == 0,
                          ValueError);
            auto pattern = counts;
            while (counts.size() < static_cast<size_t>(*slots))
              for (const auto &count : pattern)
                counts.push_back(count);
          }
        }
        initializers.Set(var, counts);
      }
      annotations.Set("barrier_init", initializers);
    }
    return annotations;
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    if (op->op.same_as(tirx::tensor_data_ptr_op())) {
      if (auto old = op->args[0].as<tirx::TensorVar>())
        if (auto tensor = replacements_.Get(*old))
          return VersionedPointer(*old, *tensor, op->loc);
    }
    if (op->op.same_as(layout_map_op())) {
      auto declaration =
          MakeAnnotation(RewriteAnnotations(ReadAnnotations(op)), op->loc)
              .as_or_throw<Evaluate>();
      auto call = declaration->value.as_or_throw<Call>();
      auto result = MutateLayoutMapCall(call.get(), this, mode).value();
      return std::move(result).ValueOrUnchanged(ffi::Any(call)).cast<Expr>();
    }
    if (const auto *name = op->op.as<OpObj>();
        name && name->name == "tl.annotation") {
      auto encoded = EncodeAnnotations(RewriteAnnotations(ReadAnnotations(op)));
      auto payload = Mutate(encoded.payload).ValueOrUnchanged(encoded.payload);
      return Call(op->ty, op->op, {payload},
                  DictAttrs({{"schema_version", 1},
                             {"annotation_schema", encoded.schema}}),
                  {}, op->loc);
    }
    Call call = tl::LayoutMapMutator::Mutate_(op, mode)
                    .ValueOrUnchanged(ffi::GetRef<Call>(op))
                    .as_or_throw<Call>();
    if (mode_ == Mode::kBarrier && parity_cycle_ && original_loop_ &&
        original_begin_) {
      if (call->op->IsInstance<OpObj>() &&
          call->op.as_or_throw<Op>().same_as(tl::mbarrier_wait_parity()) &&
          call->args.size() >= 2) {
        const auto *load = call->args[0].as<TensorLoadObj>();
        bool expanded = false;
        for (const auto &[old, tensor] : replacements_)
          expanded |= load && load->source.same_as(tensor);
        if (expanded) {
          sym::Analyzer analyzer;
          auto at_min =
              analyzer->Simplify(tirx::SubstituteWithDataTypeLegalization(
                  call->args[1].as_or_throw<PrimExpr>(),
                  [values = ffi::Map<Var, PrimExpr>{{original_loop_.value(),
                                                     original_begin_.value()}}](
                      const Var &var) -> ffi::Optional<PrimExpr> {
                    return values.Get(var);
                  }));
          auto initial = TryConstInt(at_min);
          TVM_FFI_CHECK(initial.has_value(), ValueError)
              << "Pipeline barrier initial parity must be constant at the loop "
                 "origin";
          auto args = call->args;
          args.Set(
              1, floormod(parity_cycle_.value() +
                              IntImm(parity_cycle_.value().ty(), *initial % 2),
                          2));
          return Call(call->ty, call->op, args, call->attrs, call->ty_args,
                      call->loc);
        }
      }
    }
    return call;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (auto old = op->var.as<tirx::TensorVar>()) {
      if (auto tensor = replacements_.Get(*old)) {
        const auto *call = op->value.as<CallObj>();
        TVM_FFI_CHECK(call && call->op.same_as(tirx::alloc_tensor_op()),
                      ValueError)
            << "Only an allocation definition may be resized by the pipeline";
        return Bind(*tensor,
                    Call(tensor->ty(), call->op, call->args, call->attrs,
                         {tensor->ty()}, call->loc),
                    op->loc);
      }
    }
    if (mode_ == Mode::kVersion && (op->var->ty.as<PtrTypeObj>() ||
                                    op->var->ty.as<tirx::TensorTypeObj>())) {
      Expr value = Mutate(op->value, mode).ValueOrUnchanged(op->value);
      Var alias(op->var->name, value->ty, op->var->loc);
      VarRemapSet(op->var, alias);
      if (memory_)
        memory_->RecordGeneratedDefinition(alias, value);
      return Bind(alias, value, op->loc);
    }
    return tl::LayoutMapMutator::Mutate_(op, mode);
  }

  ffi::Map<tirx::TensorVar, tirx::TensorVar> replacements_;
  Mode mode_;
  ffi::Optional<PrimExpr> iteration_, parity_cycle_;
  ffi::Optional<PrimVar> original_loop_;
  ffi::Optional<PrimExpr> original_begin_;
  const PipelineMemoryAnalysis *memory_{nullptr};
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      local_definitions_;
  ffi::Array<Stmt> external_aliases_;
  ffi::Map<Var, Var> alias_remap_;
};

} // namespace tvm::tl
#endif // TILELANG_TRANSFORM_COMMON_PIPELINE_TENSOR_H_
