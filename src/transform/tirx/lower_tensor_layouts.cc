#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "transform/common/loop_aware_mutator.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "flatten_tensor_utils.h"
#include "tirx_compat/transform.h"

#include "transform/tirx_passes.h"

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {

class LayoutApplier : public LoopAwareMutator {
public:
  using LoopAwareMutator::Mutate;
  using LoopAwareMutator::Mutate_;
  static std::pair<Stmt, ffi::Array<Var>> Flatten(const Stmt &stmt,
                                                  const ffi::Array<Var> &params,
                                                  const Target &target) {
    sym::Analyzer ana;
    auto storage_lower = ffi::make_object<LayoutApplier>(ana, target);
    ffi::Array<Var> new_params;
    new_params.reserve(params.size());
    std::vector<std::pair<TensorVar, TensorVar>> param_flattened_tensors;
    for (const Var &param : params) {
      auto tensor = param.as<TensorVar>();
      if (!tensor) {
        new_params.push_back(param);
        continue;
      }
      storage_lower->tensor_aliases_.Set(tensor.value(), tensor.value());
      TensorVar flattened = storage_lower->GetFlattenedTensor(tensor.value());
      TensorVar source = tensor.value();
      std::unordered_set<const VarNode *> shape_variables;
      for (const PrimExpr &extent : source.shape()) {
        for (const Var &variable : tvm::UndefinedVars(extent))
          shape_variables.insert(variable.get());
      }
      bool preserves_metadata = true;
      for (const Var &variable : tvm::UndefinedVars(source.ty()))
        preserves_metadata &= shape_variables.count(variable.get()) != 0;
      if (source.ty()->layout.has_value() &&
          source.ty()->HasDefaultLayout(ana) && preserves_metadata) {
        auto type = ffi::make_object<TensorTypeNode>(*source.ty().get());
        type->layout = std::nullopt;
        source =
            TensorVar(source->name, TensorType(std::move(type)), source->loc);
      }
      if (!flattened.same_as(source)) {
        param_flattened_tensors.emplace_back(flattened, source);
      }
      new_params.push_back(source);
    }
    auto new_stmt =
        storage_lower->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
    for (const auto &[buf, source] : param_flattened_tensors) {
      new_stmt = SeqStmt({Bind(buf, Call(buf.ty(), decl_tensor_op(),
                                         {source.data()}, {}, {buf.ty()})),
                          std::move(new_stmt)});
    }
    return std::make_pair(new_stmt, new_params);
  }

public:
  explicit LayoutApplier(const sym::Analyzer &analyzer, const Target &)
      : LoopAwareMutator(analyzer) {}

protected:
  UnchangedOr<Stmt> Mutate_(const ForNode *op, InplaceMode inplace_mode) final {
    if (tl::arith::CanProve(analyzer_,
                            (op->step > 0 && op->begin >= op->end) ||
                                (op->step < 0 && op->begin <= op->end))) {
      return SeqStmt({}, op->loc);
    }
    return LoopAwareMutator::Mutate_(op, inplace_mode);
  }

  ffi::Any VisitAny(const ffi::Any &any) {
    if (any == nullptr) {
      return any;
    }
    if (auto tensor = any.as<TensorVar>()) {
      return GetFlattenedTensor(tensor.value());
    } else if (auto prim_expr = any.as<PrimExpr>()) {
      return Mutate(prim_expr.value(), InplaceMode::kDisallow)
          .ValueOrUnchanged(prim_expr.value());
    } else if (auto stmt = any.as<Stmt>()) {
      return Mutate(stmt.value(), InplaceMode::kDisallow)
          .ValueOrUnchanged(stmt.value());
    }
    return any;
  }

  UnchangedOr<Expr> Mutate_(const CallNode *op,
                            InplaceMode inplace_mode) final {
    if (op->op.same_as(tensor_data_ptr_op()) && op->args.size() == 1) {
      if (auto var = op->args[0].as<Var>();
          var.has_value() && var.value()->ty.as<TensorTypeNode>()) {
        auto root_opt = tensor_aliases_.Get(var.value());
        TVM_FFI_ICHECK(root_opt.has_value())
            << "tensor_data_ptr projects " << var.value()->name
            << ", which has no visible definition "
            << "(AllocTensor/DeclTensor/Function parameter) at this point";
        Var root = root_opt.value();
        if (auto mapped = VarRemapGet(root); mapped != nullptr) {
          root = mapped.as_or_throw<Var>();
        }
        return root.as_or_throw<TensorVar>().data();
      }
    }
    return LoopAwareMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *op,
                            InplaceMode inplace_mode) final {
    if (auto source = op->value.as<TensorVar>(); source.has_value()) {
      auto tensor = op->var.as_or_throw<TensorVar>();
      Call declaration(tensor.ty(), decl_tensor_op(), {source.value().data()},
                       {}, {tensor.ty()});
      Bind binding(op->var, declaration, op->loc);
      return MutateDeclTensor(binding.get(), declaration.get(), inplace_mode)
          .ValueOrUnchanged(binding);
    }
    if (const auto *call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(alloc_tensor_op()))
        return MutateAllocTensor(op, call, inplace_mode);
      if (call->op.same_as(decl_tensor_op()))
        return MutateDeclTensor(op, call, inplace_mode);
      if (call->op.same_as(Op::Get("tirx.cuda.decl_tmem"))) {
        auto tensor = op->var.as_or_throw<TensorVar>();

        Call declaration(tensor.ty(), decl_tensor_op(),
                         {Call(tensor.ty()->DataPtrType(), reinterpret_op(),
                               {call->args[0]})},
                         {}, {tensor.ty()});
        Bind binding(op->var, declaration, op->loc);
        return MutateDeclTensor(binding.get(), declaration.get(), inplace_mode)
            .ValueOrUnchanged(binding);
      }
    }
    return LoopAwareMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode *op,
                                      const CallNode *tensor_call,
                                      InplaceMode inplace_mode) {
    tensor_aliases_.Set(op->var, op->var);
    auto mutate = [this](TensorVar buf) {
      return GetFlattenedTensor(buf, true);
    };
    auto tensor = mutate(op->var.as_or_throw<TensorVar>());
    ffi::Array<Expr> args = tensor_call->args;

    if (!args.empty()) {
      args.Set(0, Mutate(args[0], inplace_mode).ValueOrUnchanged(args[0]));
    }
    return Bind(tensor,
                Call(tensor.ty(), tirx::alloc_tensor_op(), args,
                     tensor_call->attrs, {tensor.ty()}, tensor_call->loc),
                op->loc);
  }

  UnchangedOr<Stmt> MutateDeclTensor(const BindNode *op,
                                     const CallNode *tensor_call,
                                     InplaceMode inplace_mode) {
    RegisterTensorAlias(op->var.as_or_throw<TensorVar>(), tensor_call->args[0]);
    auto data_result = Mutate(tensor_call->args[0], inplace_mode);
    bool data_unchanged = data_result.UnchangedOrSameAs(tensor_call->args[0]);
    Expr data = std::move(data_result).ValueOrUnchanged(tensor_call->args[0]);
    auto tensor = GetFlattenedTensor(op->var.as_or_throw<TensorVar>());
    if (tensor.same_as(op->var.as_or_throw<TensorVar>()) && data_unchanged) {
      return ffi::Unchanged();
    }
    return Bind(tensor,
                Call(tensor.ty(), decl_tensor_op(), {std::move(data)},
                     tensor_call->attrs, {tensor.ty()}, tensor_call->loc),
                op->loc);
  }

  TensorVar GetFlattenedTensor(TensorVar buf, bool is_alloc = false) {
    if (auto mapped = VarRemapGet(buf); mapped != nullptr) {
      return mapped.as_or_throw<TensorVar>();
    }
    TensorVar flattened = buf;
    ffi::ObjectPtr<TensorTypeNode> type;
    if (is_alloc) {
      if (auto tile_layout = buf.ty()->layout.as<TileLayoutNode>()) {

        sym::Analyzer ana;
        PrimExpr mem_span = IntImm::Int32(1);
        for (const auto &iter : tile_layout->shard) {
          if (iter->axis->IsMemoryAxis()) {
            mem_span = mem_span + (iter->extent - 1) * iter->stride;
          }
        }
        for (const auto &iter : tile_layout->replica) {
          if (iter->axis->IsMemoryAxis()) {
            mem_span = mem_span + (iter->extent - 1) * iter->stride;
          }
        }
        for (const auto &[axis, off] : tile_layout->offset) {
          if (axis->IsMemoryAxis()) {
            mem_span = mem_span + off;
          }
        }
        flattened = buf;
        type = ffi::make_object<TensorTypeNode>(*flattened.ty().get());
        type->shape = {tl::arith::Simplify(ana, mem_span)};
        type->layout = std::nullopt;
      } else if (auto compose = buf.ty()->layout.as<ComposeLayoutNode>()) {

        PrimExpr mem_span = compose->tile_layout.GetSpan(ffi::String("m"));
        if (compose->tile_layout.IsTrivial()) {
          mem_span = IntImm::Int32(1);
          for (const auto &extent : buf.shape())
            mem_span = mem_span * extent;
        }
        if (compose->swizzle_len != 0) {
          int changed_bits = compose->per_element + compose->swizzle_len;
          if (!compose->swizzle_inner)
            changed_bits += compose->atom_len;
          PrimExpr block = IntImm(mem_span.ty(), int64_t{1} << changed_bits);
          mem_span = floordiv(mem_span + block - 1, block) * block;
        }
        type = ffi::make_object<TensorTypeNode>(*buf.ty().get());
        type->shape = {tl::arith::Simplify(analyzer_, mem_span)};
        type->layout = std::nullopt;
      } else {
        flattened = FlattenedTensor(buf, ffi::GetRef<sym::Analyzer>(analyzer_));
        type = ffi::make_object<TensorTypeNode>(*flattened.ty().get());
      }
    } else {
      flattened = FlattenedTensor(buf, ffi::GetRef<sym::Analyzer>(analyzer_));
      type = ffi::make_object<TensorTypeNode>(*flattened.ty().get());
    }

    for (size_t i = 0; i < type->shape.size(); ++i) {
      type->shape.Set(i, tl::arith::CanonicalSimplify(
                             analyzer_, LoopAwareMutator::Mutate(
                                            ffi::AnyView(type->shape[i]),
                                            InplaceMode::kDisallow)
                                            .ValueOrUnchanged(type->shape[i])
                                            .as_or_throw<PrimExpr>()));
    }
    type->layout = std::nullopt;

    if (!is_alloc) {
      if (auto pattern = MatchShapeStridePattern(buf.ty());
          pattern.has_value()) {
        PrimExpr origin = pattern.value().get<2>();
        origin = tl::arith::CanonicalSimplify(
            analyzer_, LoopAwareMutator::Mutate(ffi::AnyView(origin),
                                                InplaceMode::kDisallow)
                           .ValueOrUnchanged(origin)
                           .as_or_throw<PrimExpr>());
        if (!prim::IsZero(origin)) {
          static const auto memory_axis = Axis::Get("m");
          type->layout = TileLayout({Iter(type->shape[0], 1, memory_axis)}, {},
                                    {{memory_axis, origin}});
        }
      }
    }
    if (tl::StructuralEqual()(buf.ty(), TensorType(type))) {
      return buf;
    }
    flattened =
        TensorVar(flattened->name, TensorType(std::move(type)), flattened->loc);

    VarRemapSet(buf, flattened);
    return flattened;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *op,
                            InplaceMode inplace_mode) final {

    auto value = Mutate(op->value, inplace_mode);
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore store = ffi::GetRef<TensorStore>(op);
    if (!value.UnchangedOrSameAs(op->value) ||
        !indices.UnchangedOrSameAs(op->indices)) {
      auto *n = store.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    return VisitTensorAccess(std::move(store));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *op,
                                InplaceMode inplace_mode) final {
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad load = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      load.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return VisitTensorAccess(std::move(load));
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode *op,
                            InplaceMode inplace_mode) final {
    TensorVar tensor = op->source.as_or_throw<TensorVar>();
    auto regions = Mutate(op->region, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<Range>>>()
                       .ValueOrUnchanged(op->region);
    auto flattened_regions =
        MapTensorRegionBounds(regions, ffi::GetRef<sym::Analyzer>(analyzer_),
                              [&](const ffi::Array<PrimExpr> &indices) {
                                return GetSimplifiedElemOffset(tensor, indices);
                              });
    return TensorRegion(GetFlattenedTensor(tensor),
                        std::move(flattened_regions), op->ty, op->loc);
  }

  ffi::Array<PrimExpr>
  GetSimplifiedElemOffset(const TensorVar &tensor,
                          const ffi::Array<PrimExpr> &indices) {
    if (tensor.ty()->layout.has_value()) {
      if (auto tile = tensor.ty()->layout.value().as<TileLayoutNode>();
          tile && ffi::GetRef<TileLayout>(tile).HasThreadAxis()) {
        LOG(FATAL)
            << "Cannot lower direct TensorLoad/TensorStore on a tensor with "
               "thread-axis "
            << "layout: unable to verify that the coordinate matches the "
               "current thread. "
            << "Use .view() + .local() to decompose thread and memory axes.";
      }
      auto layout = tensor.ty()->layout.value();
      layout =
          LoopAwareMutator::Mutate(ffi::AnyView(layout), InplaceMode::kDisallow)
              .ValueOrUnchanged(layout)
              .as_or_throw<Layout>();
      auto flat = GetFlattenedTensor(tensor);
      auto pattern = MatchShapeStridePattern(flat.ty()).value();
      if (!prim::IsZero(pattern.get<2>())) {

        auto tile = layout.as_or_throw<TileLayout>();
        layout = TileLayout(tile->shard, tile->replica, {});
      }
      auto res = layout.Canonicalize().Apply(indices, tensor.shape());
      if (res.empty() && tensor.shape().empty()) {
        return {IntImm(tensor.ty()->DefaultIndexType(), 0)};
      }
      TVM_FFI_ICHECK_EQ(res.size(), 1) << "Expected a single element offset";
      return {tl::arith::Simplify(analyzer_, (*res.begin()).second)};
    }
    auto flattened_indices = tensor.ty()->ElemOffset(indices);
    TVM_FFI_ICHECK_EQ(flattened_indices.size(), 1)
        << "Expected a single element offset";
    return {tl::arith::Simplify(analyzer_, flattened_indices[0])};
  }

  template <typename Node> Node VisitTensorAccess(Node node) {
    TVM_FFI_ICHECK(node->dest.template as_or_throw<TensorVar>().defined());
    auto flattened_indices = GetSimplifiedElemOffset(
        node->dest.template as_or_throw<TensorVar>(), node->indices);
    TensorVar flattened_tensor =
        GetFlattenedTensor(node->dest.template as_or_throw<TensorVar>());
    auto writer = node.CopyOnWrite();
    writer->dest = flattened_tensor;
    writer->indices = flattened_indices;
    return node;
  }

  TensorLoad VisitTensorAccess(TensorLoad node) {
    TensorVar tensor = node->source.as_or_throw<tvm::tirx::TensorVar>();
    TVM_FFI_ICHECK(tensor.defined());
    return TensorLoad(GetFlattenedTensor(tensor),
                      GetSimplifiedElemOffset(tensor, node->indices),
                      node->loc);
  }

private:
  void RegisterTensorAlias(TensorVar tensor, const Expr &data) {
    Var root = tensor;
    if (const auto *call = data.as<CallNode>();
        call && call->op.same_as(tensor_data_ptr_op()) &&
        call->args.size() == 1) {
      if (auto source = call->args[0].as<Var>();
          source.has_value() && source.value()->ty.as<TensorTypeNode>()) {
        auto source_root = tensor_aliases_.Get(source.value());
        TVM_FFI_ICHECK(source_root.has_value())
            << "Tensor alias source " << source.value()->name
            << " must be registered before its DeclTensor alias";
        root = source_root.value();
      }
    }
    tensor_aliases_.Set(tensor, root);
  }

  ffi::Map<Var, Var> tensor_aliases_;
};

namespace {
Target ResolveTarget(const Function &f) {
  auto target = f->GetAttr(tvm::attr::kTarget);
  if (!target.has_value()) {
    target = Target::Current(false).value();
  }
  return target.value();
}
} // namespace

} // namespace

Pass LowerTensorLayouts() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value())
      return f;
    Target target = ResolveTarget(f);
    auto *n = f.CopyOnWrite();
    auto [body, params] =
        LayoutApplier::Flatten(n->body.value(), n->params, target);
    n->body = std::move(body);
    n->params = std::move(params);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tl.LowerTensorLayouts");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl._tirx_passes.LowerTensorLayouts",
                                   [] { return LowerTensorLayouts(); });
}

} // namespace tvm::tl::tirx_passes
