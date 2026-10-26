#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

#include "op/prim.h"
#include "transform/common/data_type_rewriter.h"
#include "transform/tirx_passes.h"

namespace tvm::tl {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;

class Int32DTypeNarrower : public IndexDataTypeNormalizer {
public:
  using IndexDataTypeNormalizer::Mutate;
  using IndexDataTypeNormalizer::Mutate_;

  static Function RewriteDataType(Function func) {
    CheckTensorParams(func);
    auto narrower = ffi::make_object<Int32DTypeNarrower>(func);
    return narrower->Rewrite(func);
  }

  explicit Int32DTypeNarrower(Function func)
      : IndexDataTypeNormalizer(PrimType::Int(32)), func_(std::move(func)) {}

private:
  static void CheckTensorParams(const Function &func) {
    for (const Var &param : func->params) {
      if (auto tensor = param.as<TensorVar>();
          tensor &&
          tensor.value().dtype().MatchesCode(DLDataTypeCode::kDLInt) &&
          tensor.value().dtype().bits() > 32) {
        TVM_FFI_THROW(InternalError)
            << "The tensor parameter " << tensor.value() << " has dtype "
            << tensor.value().dtype() << ". The function is " << func;
      }
    }
  }

  void CheckAllocatedTensor(const TensorVar &buf) const {

    static const auto local = runtime::MemoryScope::Get("local");
    bool is_local_scalar = buf.scope() == local && buf.IsScalar();
    if (!is_local_scalar && buf.dtype().MatchesCode(DLDataTypeCode::kDLInt) &&
        buf.dtype().bits() > 32) {
      TVM_FFI_THROW(InternalError)
          << "The tensor " << buf << " allocated in the function has dtype "
          << buf.dtype() << ". The function is " << func_;
    }
  }

  bool ShouldClampShiftAmounts() const final { return true; }

  UnchangedOr<PrimExpr> Mutate_(const IntImmNode *op,
                                InplaceMode inplace_mode) final {

    if (op->ty.as_or_throw<PrimType>() == PrimType::Int(64)) {
      TVM_FFI_ICHECK_LE(
          op->value,
          prim::max_value(target_data_type_).as_or_throw<IntImm>()->value);
      return IntImm::Int32(op->value);
    }
    return ffi::Unchanged();
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *op,
                            InplaceMode inplace_mode) final {
    if (const auto *call = op->value.as<CallNode>();
        call && call->op.same_as(tirx::alloc_tensor_op()))
      return MutateAllocTensor(op, inplace_mode);
    return IndexDataTypeNormalizer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode *op,
                                      InplaceMode inplace_mode) {
    auto result = IndexDataTypeNormalizer::Mutate_(op, inplace_mode);
    auto alloc = std::move(result)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<Bind>();
    CheckAllocatedTensor(alloc->var.as_or_throw<TensorVar>());
    return alloc;
  }

  Function func_;
};

Function ForceNarrowIndexToInt32(Function func) {
  return Int32DTypeNarrower::RewriteDataType(func);
}

namespace tirx_passes {

Pass ForceNarrowIndexToInt32() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    return tl::ForceNarrowIndexToInt32(f);
  };
  return CreateFunctionPass(pass_func, 0, "tl.ForceNarrowIndexToInt32");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl._tirx_passes.ForceNarrowIndexToInt32",
                        ForceNarrowIndexToInt32);
}

} // namespace tirx_passes
} // namespace tvm::tl
