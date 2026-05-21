#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tilelang {

using namespace tirx;
using tvm::ffi::Array;

class HexagonIntrinsicLowerer : public StmtExprMutator {
public:
  HexagonIntrinsicLowerer() {}

  Stmt Run(Stmt stmt) { return this->VisitStmt(stmt); }

  Stmt VisitStmt_(const EvaluateNode *op) override {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::call_extern())) {
        if (const StringImmNode *func_name =
                call->args[0].as<StringImmNode>()) {

          // Lower HMX MMA placeholder
          if (func_name->value == "hmx_mma_placeholder") {
            Array<PrimExpr> new_args;
            new_args.push_back(StringImm("HexKL_mma_i8acc32"));
            new_args.push_back(
                call->args[3]); // C_acc (accumulator — first arg to HexKL)
            new_args.push_back(call->args[1]); // A_vtcm
            new_args.push_back(call->args[2]); // B_vtcm
            return Evaluate(
                Call(DataType::Int(32), builtin::call_extern(), new_args));
          }

          // HexagonDmaCopy is not yet available in HexKL v73.
          // to-do: LowerHexagonDMA pass.
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

namespace transform {

tvm::transform::Pass LowerHexagonIntrinsics() {
  auto pass_func = [=](PrimFunc f, IRModule m,
                       tvm::transform::PassContext ctx) {
    auto *n = f.CopyOnWrite();
    n->body = HexagonIntrinsicLowerer().Run(std::move(n->body));
    return f;
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tilelang.transform.LowerHexagonIntrinsics", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tilelang.transform.LowerHexagonIntrinsics",
                        LowerHexagonIntrinsics);
}

} // namespace transform
} // namespace tilelang
} // namespace tvm
