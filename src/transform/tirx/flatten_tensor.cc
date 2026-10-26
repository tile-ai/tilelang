#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "transform/common/loop_aware_mutator.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "flatten_tensor_utils.h"
#include "tirx_compat/transform.h"

#include "transform/tirx_passes.h"

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {

using namespace tvm::prim;

class TensorFlattener : public LoopAwareMutator {
public:
  using LoopAwareMutator::Mutate;
  using LoopAwareMutator::Mutate_;
  static Function Flatten(Function func) {
    if (!func->body.has_value())
      return func;
    sym::Analyzer ana;
    auto pass = ffi::make_object<TensorFlattener>(ana);
    pass->MarkTensorParamShapes(func);
    for (const Var &param : func->params) {
      if (auto tensor = param.as<TensorVar>()) {
        pass->extern_tensors_.insert(tensor.value());
        pass->Define(tensor.value());
      }
    }
    auto body_result = pass->Mutate(func->body, InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(func->body);
    auto body = std::move(body_result).ValueOrUnchanged(func->body);

    for (size_t i = func->params.size(); i > 0; i--) {
      if (auto old_buf = func->params[i - 1].as<TensorVar>()) {
        if (pass->tensors_used_.count(old_buf.value())) {
          auto new_buf = pass->Lookup(old_buf.value()).flattened;
          if (!old_buf.value().same_as(new_buf)) {
            body = SeqStmt({Bind(new_buf, Call(new_buf.ty(), decl_tensor_op(),
                                               {old_buf.value().data()}, {},
                                               {new_buf.ty()})),
                            std::move(body).value()});
            body_unchanged = false;
          }
        }
      }
    }

    if (!body_unchanged) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

public:
  explicit TensorFlattener(const sym::Analyzer &ana) : LoopAwareMutator(ana) {}

private:
  struct FlatInfo {

    TensorVar fold_view;

    TensorVar flattened;
  };

  const FlatInfo &Define(const TensorVar &buf) {
    if (auto it = flat_map_.find(buf); it != flat_map_.end()) {
      return it->second;
    }

    auto view_type = ffi::make_object<TensorTypeNode>(*buf.ty().get());
    auto mutate_expr = [this](const PrimExpr &expr) {
      return Mutate(expr).ValueOrUnchanged(expr);
    };
    view_type->shape = view_type->shape.Map(mutate_expr);
    if (view_type->layout.has_value()) {
      view_type->layout =
          this->Mutate(ffi::AnyView(view_type->layout), InplaceMode::kDisallow)
              .ValueOrUnchanged(view_type->layout)
              .cast<ffi::Optional<Layout>>();
    }
    TensorVar fold_view =
        TensorVar(buf->name, TensorType(std::move(view_type)), buf->loc);

    auto flat =
        FlattenedTensor(fold_view, ffi::GetRef<sym::Analyzer>(analyzer_), true);
    auto type = ffi::make_object<TensorTypeNode>(*flat.ty().get());
    for (size_t i = 0; i < type->shape.size(); ++i) {
      type->shape.Set(i,
                      tl::arith::CanonicalSimplify(analyzer_, type->shape[i]));
    }
    type->layout = std::nullopt;

    TensorVar flattened =
        (!extern_tensors_.count(buf) &&
         tl::StructuralEqual()(TensorType(type), buf.ty()))
            ? buf
            : TensorVar(buf->name, TensorType(std::move(type)), buf->loc);

    VarRemapSet(buf, flattened);
    auto [it, inserted] =
        flat_map_.emplace(buf, FlatInfo{fold_view, flattened});
    return it->second;
  }

  const FlatInfo &Lookup(const TensorVar &buf) {
    auto it = flat_map_.find(buf);
    TVM_FFI_ICHECK(it != flat_map_.end())
        << "Tensor " << buf->name
        << " is used before its definition (AllocTensor/DeclTensor/Function "
           "param)";
    return it->second;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *op,
                            InplaceMode inplace_mode) final {
    if (const auto *call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(alloc_tensor_op()))
        return MutateAllocTensor(op, call, inplace_mode);
      if (call->op.same_as(decl_tensor_op()))
        return MutateDeclTensor(op, call, inplace_mode);
    }
    return LoopAwareMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode *op,
                                      const CallNode *tensor_call,
                                      InplaceMode inplace_mode) {
    const FlatInfo &info = Define(op->var.as_or_throw<TensorVar>());
    ffi::Array<Expr> args = tensor_call->args;
    if (!args.empty()) {
      args.Set(0, Mutate(args[0], inplace_mode).ValueOrUnchanged(args[0]));
    }
    if (info.flattened.same_as(op->var.as_or_throw<TensorVar>()) &&
        args.same_as(tensor_call->args)) {
      return ffi::Unchanged();
    }

    return Bind(info.flattened,
                Call(info.flattened.ty(), tirx::alloc_tensor_op(), args,
                     tensor_call->attrs, {info.flattened.ty()},
                     tensor_call->loc),
                op->loc);
  }

  UnchangedOr<Stmt> MutateDeclTensor(const BindNode *op,
                                     const CallNode *tensor_call,
                                     InplaceMode inplace_mode) {
    Expr data = tensor_call->args[0];
    bool is_extern_tensor_source = false;
    if (const auto *call = tensor_call->args[0].as<CallNode>();
        call && call->op.same_as(tensor_data_ptr_op()) &&
        call->args.size() == 1) {
      if (const auto *var = call->args[0].as<VarNode>();
          var && var->ty.as<TensorTypeNode>()) {
        is_extern_tensor_source = extern_tensors_.count(
            ffi::GetRef<Var>(var).as_or_throw<TensorVar>());
      }
    }
    if (!is_extern_tensor_source) {
      data = Mutate(tensor_call->args[0], inplace_mode)
                 .ValueOrUnchanged(tensor_call->args[0]);
    }
    const FlatInfo &info = Define(op->var.as_or_throw<TensorVar>());
    if (info.flattened.same_as(op->var.as_or_throw<TensorVar>()) &&
        data.same_as(tensor_call->args[0])) {
      return ffi::Unchanged();
    }
    return Bind(info.flattened,
                Call(info.flattened.ty(), decl_tensor_op(), {std::move(data)},
                     tensor_call->attrs, {info.flattened.ty()},
                     tensor_call->loc),
                op->loc);
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
    return VisitTensorAccess(std::move(store),
                             op->dest.as_or_throw<TensorVar>());
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *op,
                                InplaceMode inplace_mode) final {
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad load = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      load.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return VisitTensorAccess(std::move(load),
                             op->source.as_or_throw<TensorVar>());
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode *op,
                            InplaceMode inplace_mode) final {
    TensorVar original = op->source.as_or_throw<TensorVar>();
    tensors_used_.insert(original);
    const FlatInfo &info = Lookup(original);
    auto regions = Mutate(op->region, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<Range>>>()
                       .ValueOrUnchanged(op->region);
    auto flattened_regions =
        MapTensorRegionBounds(regions, ffi::GetRef<sym::Analyzer>(analyzer_),
                              [&](const ffi::Array<PrimExpr> &indices) {
                                return FoldIndices(info, indices);
                              });
    return TensorRegion(info.flattened, std::move(flattened_regions), op->ty,
                        op->loc);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *op,
                            InplaceMode inplace_mode) final {
    if (op->op.same_as(masked_load_op()) || op->op.same_as(masked_store_op())) {
      bool is_load = op->op.same_as(masked_load_op());
      TensorVar original = op->args[0].as_or_throw<TensorVar>();
      ffi::Array<PrimExpr> indices;
      for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
        indices.push_back(this->Mutate(op->args[i])
                              .ValueOrUnchanged(op->args[i])
                              .as_or_throw<PrimExpr>());
      }
      tensors_used_.insert(original);
      const FlatInfo &info = Lookup(original);
      ffi::Array<Expr> args{info.flattened};
      if (!is_load)
        args.push_back(this->Mutate(op->args[1])
                           .ValueOrUnchanged(op->args[1])
                           .as_or_throw<Expr>());
      for (const PrimExpr &index : FoldIndices(info, indices))
        args.push_back(index);
      args.push_back(this->Mutate(op->args[op->args.size() - 1])
                         .ValueOrUnchanged(op->args[op->args.size() - 1])
                         .as_or_throw<Expr>());
      return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->loc);
    }
    if (op->op.same_as(tensor_data_ptr_op()) && op->args.size() == 1) {
      if (auto var = op->args[0].as<Var>()) {
        if (var.value()->ty.as<TensorTypeNode>()) {
          TensorVar original = var.value().as_or_throw<TensorVar>();
          tensors_used_.insert(original);
          return Lookup(original).flattened.data();
        }
      }
    }
    return LoopAwareMutator::Mutate_(op, inplace_mode);
  }

  ffi::Array<PrimExpr> FoldIndices(const FlatInfo &info,
                                   const ffi::Array<PrimExpr> &indices) {
    auto flattened_indices = info.fold_view.ty()->ElemOffset(indices);
    return this->IterMapSimplifyWithContext(flattened_indices, false);
  }

  template <typename Node>
  Node VisitTensorAccess(Node node, const TensorVar &original_tensor) {
    TVM_FFI_ICHECK(node->dest.template as_or_throw<TensorVar>().defined());
    tensors_used_.insert(original_tensor);
    const FlatInfo &info = Lookup(original_tensor);
    auto flattened_indices = FoldIndices(info, node->indices);

    auto writer = node.CopyOnWrite();
    writer->dest = info.flattened;
    writer->indices = flattened_indices;
    return node;
  }

  TensorLoad VisitTensorAccess(TensorLoad node,
                               const TensorVar &original_tensor) {
    tensors_used_.insert(original_tensor);
    const FlatInfo &info = Lookup(original_tensor);
    return TensorLoad(info.flattened, FoldIndices(info, node->indices),
                      node->loc);
  }

  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      tensors_used_;

  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      extern_tensors_;

  std::unordered_map<Var, FlatInfo, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      flat_map_;
};

Function FlattenTensor(Function f) { return TensorFlattener::Flatten(f); }

} // namespace

Pass FlattenTensor() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    return FlattenTensor(std::move(f));
  };
  return CreateFunctionPass(pass_func, 0, "tl.FlattenTensor");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl._tirx_passes.FlattenTensor",
                                   [] { return FlattenTensor(); });
}

} // namespace tvm::tl::tirx_passes
