/*!
 * \file lower_access_ptr.cc
 * \brief Consume TileLang access contracts at the native address boundary.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../ir/analyzer.h"
#include "../op/builtin.h"

namespace tvm::tl {
namespace {

class AccessPtrLowerer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    if (!op->op.same_as(access_ptr()) && !op->op.same_as(ptr_access())) {
      return StmtExprMutator::Mutate_(op, mode);
    }
    if (op->op.same_as(access_ptr())) {
      TVM_FFI_CHECK_EQ(op->args.size(), 3U, ValueError);
      const auto load = op->args[0].as_or_throw<TensorLoad>();
      Expr source = Mutate(load->source, mode).ValueOrUnchanged(load->source);
      ffi::Array<PrimExpr> indices;
      for (const PrimExpr &original : load->indices) {
        PrimExpr index = Mutate(original, mode).ValueOrUnchanged(original);
        // The pointer names the first lane, while extent already counts all
        // lanes requested by the frontend. It does not evaluate a memory load.
        if (const auto *ramp = index.as<prim::RampObj>())
          index = ramp->base;
        if (const auto *broadcast = index.as<prim::BroadcastObj>())
          index = broadcast->value;
        indices.push_back(index);
      }
      TensorLoad address(source, indices, load->loc);
      // Keep the logical address until LowerTensorLayouts/FlattenTensor. The
      // owning native tensor type accounts for strides, offsets and composed
      // layouts; computing a row-major offset here would lose those contracts.
      return Call(op->ty, tirx::address_of_op(), {address}, {}, {}, op->loc);
    }

    TVM_FFI_CHECK_EQ(op->args.size(), 4U, ValueError);
    Expr pointer = Mutate(op->args[0], mode).ValueOrUnchanged(op->args[0]);
    PrimExpr offset = Mutate(op->args[1], mode)
                          .ValueOrUnchanged(op->args[1])
                          .as_or_throw<PrimExpr>();
    const PtrType pointer_type = pointer->ty.as_or_throw<PtrType>();
    const PrimType element_type =
        pointer_type->element_type.as_or_throw<PrimType>();
    const int bits = element_type.bits() * element_type.lanes();
    TVM_FFI_CHECK_GT(bits, 0, ValueError)
        << "Access pointer needs a sized element type";
    // Sub-byte element addresses select their containing physical byte, as in
    // TileLang's packed storage addressing. Dtype reinterpretation was already
    // made explicit by the producer, before this element offset is applied.
    PrimExpr bytes =
        floordiv(offset * IntImm(offset.ty(), bits), IntImm(offset.ty(), 8));
    bytes = analyzer_->Simplify(bytes);
    return Call(op->ty, tirx::ptr_byte_offset_op(), {pointer, bytes}, {}, {},
                op->loc);
  }

private:
  sym::Analyzer analyzer_ = CreateTileLangAnalyzer();
};

} // namespace

namespace transform {

tvm::transform::Pass LowerAccessPtr() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    auto lowerer = ffi::make_object<AccessPtrLowerer>();
    const auto body = function->body;
    SeqStmt result = lowerer->Mutate(body).ValueOrUnchanged(body);
    if (!result.same_as(body))
      function.CopyOnWrite()->body = result;
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerAccessPtr");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerAccessPtr",
                                   LowerAccessPtr);
}

} // namespace transform
} // namespace tvm::tl
