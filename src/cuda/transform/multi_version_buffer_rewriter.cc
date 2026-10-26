#include "multi_version_buffer_rewriter.h"
#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "transform/common/pipeline_utils.h"

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

tirx::Layout PrependVersionLayout(const tirx::Layout &layout,
                                  const Array<Iter> &versions) {
  if (auto tile = layout.as<TileLayout>()) {
    Array<Iter> dimensions = versions;
    dimensions.insert(dimensions.end(), tile.value()->shard.begin(),
                      tile.value()->shard.end());
    return TileLayout(dimensions, tile.value()->replica, tile.value()->offset);
  }
  auto composed = layout.as_or_throw<ComposeLayout>();
  auto tile = PrependVersionLayout(composed->tile_layout, versions)
                  .as_or_throw<TileLayout>();
  return ComposeLayout(composed->per_element, composed->swizzle_len,
                       composed->atom_len, tile, composed->swizzle_inner);
}

TensorVar ExpandPipelineAlias(const TensorVar &alias, const TensorVar &owner,
                              const TensorVar &expanded) {
  TVM_FFI_CHECK(expanded.shape().size() > owner.shape().size(), ValueError)
      << "A pipeline alias requires a versioned owner";
  int version_rank = expanded.shape().size() - owner.shape().size();
  int owner_bits = IsPacked4BitStorage(owner.dtype())
                       ? owner.dtype().bits() * owner.dtype().lanes()
                       : owner.dtype().StorageBytes() * 8;
  int alias_bits = IsPacked4BitStorage(alias.dtype())
                       ? alias.dtype().bits() * alias.dtype().lanes()
                       : alias.dtype().StorageBytes() * 8;
  Array<PrimExpr> zero(expanded.shape().size(), PrimExpr(0));
  Array<PrimExpr> base = expanded.ty()->ElemOffset(zero);
  TVM_FFI_CHECK_EQ(base.size(), 1U, ValueError)
      << "Versioned aliases require a memory-only layout";
  Array<PrimExpr> shape;
  Array<Iter> versions;
  sym::Analyzer analyzer;
  for (int axis = 0; axis < version_rank; ++axis) {
    Array<PrimExpr> indices = zero;
    indices.Set(axis, PrimExpr(1));
    PrimExpr bits = Cast(PrimType::Int(64),
                         expanded.ty()->ElemOffset(indices)[0] - base[0]) *
                    IntImm::Int64(owner_bits);
    TVM_FFI_CHECK(
        tl::arith::CanProve(analyzer, floormod(bits, alias_bits) == 0),
        ValueError)
        << "Pipeline version offsets must align with alias elements";
    versions.push_back(
        Iter(expanded.shape()[axis],
             tl::arith::Simplify(analyzer, floordiv(bits, alias_bits)),
             Axis::Get("m")));
    shape.push_back(expanded.shape()[axis]);
  }
  shape.insert(shape.end(), alias.shape().begin(), alias.shape().end());
  auto layout = PrependVersionLayout(
      alias.ty()->layout.value_or(TileLayout::DefaultLayout(alias.shape())),
      versions);
  return TensorVar(alias->name + "_pipeline",
                   TensorType(shape, alias.dtype(), alias.scope(),
                              alias.ty()->data_alignment, layout, alias->loc),
                   alias->loc);
}

class BufferVersionRewriter : public tvm::tl::StmtExprMutator {
public:
  BufferVersionRewriter(Map<TensorVar, TensorVar> replacements,
                        Map<TensorVar, Array<PrimExpr>> versions)
      : replacements_(std::move(replacements)), versions_(std::move(versions)) {
    for (const auto &[original, storage] : replacements_) {
      TVM_FFI_CHECK(versions_.count(original), ValueError)
          << "Every versioned tensor needs version coordinates";
      TVM_FFI_CHECK(storage.shape().size() == original.shape().size() +
                                                  versions_[original].size() &&
                        storage.dtype() == original.dtype(),
                    ValueError)
          << "Versioned tensors must prepend version dimensions and preserve "
             "element types";
    }
  }

private:
  Array<PrimExpr> Indices(const TensorVar &tensor,
                          const Array<PrimExpr> &indices, InplaceMode mode) {
    Array<PrimExpr> result = versions_.Get(tensor).value_or(Array<PrimExpr>{});
    for (const PrimExpr &index : indices)
      result.push_back(Mutate(index, mode).ValueOrUnchanged(index));
    return result;
  }

  TensorRegion Region(const TensorVar &tensor, const Array<Range> &ranges,
                      Location location, InplaceMode mode) {
    Array<Range> result;
    if (auto versions = versions_.Get(tensor))
      for (const PrimExpr &version : *versions)
        result.push_back(Range::FromMinExtent(version, PrimExpr(1)));
    for (const Range &range : ranges)
      result.push_back(Range::FromMinExtent(
          Mutate(range->begin, mode).ValueOrUnchanged(range->begin),
          Mutate(range->extent, mode).ValueOrUnchanged(range->extent)));
    return TensorRegion(replacements_.Get(tensor).value_or(tensor), result,
                        TensorRegionType(), location);
  }

  UnchangedOr<Expr> Mutate_(const VarNode *node, InplaceMode mode) final {
    if (auto tensor = GetRef<Var>(node).as<TensorVar>())
      if (auto replacement = replacements_.Get(*tensor))
        return *replacement;
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                InplaceMode mode) final {
    auto tensor = node->source.as<TensorVar>();
    if (tensor && replacements_.count(*tensor))
      return TensorLoad(replacements_[*tensor],
                        Indices(*tensor, node->indices, mode), node->loc);
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                            InplaceMode mode) final {
    auto tensor = node->dest.as<TensorVar>();
    if (tensor && replacements_.count(*tensor))
      return TensorStore(
          replacements_[*tensor], Indices(*tensor, node->indices, mode),
          Mutate(node->value, mode).ValueOrUnchanged(node->value), node->loc);
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode *node,
                            InplaceMode mode) final {
    auto tensor = node->source.as<TensorVar>();
    if (tensor && replacements_.count(*tensor))
      return Region(*tensor, node->region, node->loc, mode);
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node, InplaceMode mode) final {
    if (node->op.same_as(tensor_data_ptr_op())) {
      auto tensor = node->args[0].as<TensorVar>();
      if (tensor && replacements_.count(*tensor)) {
        Array<PrimExpr> indices(tensor.value().shape().size(), PrimExpr(0));
        return Call(std::nullopt, address_of_op(),
                    {TensorLoad(replacements_[*tensor],
                                Indices(*tensor, indices, mode))});
      }
    }
    if (node->op.same_as(masked_load_op()) ||
        node->op.same_as(masked_store_op())) {
      bool store = node->op.same_as(masked_store_op());
      auto tensor = node->args[0].as<TensorVar>();
      if (tensor && replacements_.count(*tensor)) {
        Array<Expr> arguments{replacements_[*tensor]};
        if (store)
          arguments.push_back(
              Mutate(node->args[1], mode).ValueOrUnchanged(node->args[1]));
        for (const PrimExpr &version : versions_[*tensor])
          arguments.push_back(version);
        for (size_t index = store ? 2 : 1; index < node->args.size(); ++index)
          arguments.push_back(Mutate(node->args[index], mode)
                                  .ValueOrUnchanged(node->args[index]));
        return Call(node->ty, node->op, arguments, node->attrs, node->ty_args,
                    node->loc);
      }
    }
    if (node->op.same_as(region()) && node->args[0].as<TensorLoadNode>()) {
      auto load = node->args[0].as_or_throw<TensorLoad>();
      auto tensor = load->source.as<TensorVar>();
      if (tensor && replacements_.count(*tensor)) {
        Array<Expr> arguments{
            Mutate(node->args[0], mode).ValueOrUnchanged(node->args[0]),
            node->args[1]};
        for (size_t index = 0; index < versions_[*tensor].size(); ++index)
          arguments.push_back(PrimExpr(1));
        for (size_t index = 2; index < node->args.size(); ++index)
          arguments.push_back(Mutate(node->args[index], mode)
                                  .ValueOrUnchanged(node->args[index]));
        return Call(node->ty, node->op, arguments, node->attrs, node->ty_args,
                    node->loc);
      }
    }
    Call call = GetRef<Call>(node);
    if (ParseOperator(call).defined()) {
      auto arguments = call->args;
      for (size_t index = 0; index < arguments.size(); ++index)
        if (auto tensor = arguments[index].as<TensorVar>();
            tensor && replacements_.count(*tensor)) {
          Array<Range> ranges;
          for (const PrimExpr &extent : tensor.value().shape())
            ranges.push_back(Range(PrimExpr(0), extent));
          arguments.Set(index, Region(*tensor, ranges, node->loc, mode));
        }
      call.CopyOnWrite()->args = arguments;
    }
    return tvm::tl::StmtExprMutator::Mutate_(call.get(), mode)
        .ValueOrUnchanged(call);
  }

  Map<TensorVar, TensorVar> replacements_;
  Map<TensorVar, Array<PrimExpr>> versions_;
};
} // namespace

Stmt RewriteBufferVersions(
    Stmt statement, ffi::Map<tirx::TensorVar, tirx::TensorVar> replacements,
    ffi::Map<tirx::TensorVar, ffi::Array<PrimExpr>> versions) {
  auto rewriter = ffi::make_object<BufferVersionRewriter>(
      std::move(replacements), std::move(versions));
  return rewriter->Mutate(statement).ValueOrUnchanged(statement);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.cuda.analysis.ExpandPipelineTensor", ExpandPipelineTensor)
      .def("tl.cuda.analysis.ExpandPipelineAlias", ExpandPipelineAlias)
      .def("tl.cuda.analysis.RewriteBufferVersions", RewriteBufferVersions);
}
} // namespace tvm::tl
