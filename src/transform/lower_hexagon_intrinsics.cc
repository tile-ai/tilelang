#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tilelang {

using namespace tir;
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
  return tvm::tir::transform::CreatePrimFuncPass(
      pass_func, 0, "tilelang.transform.LowerHexagonIntrinsics", {});
}

// Memory scope descriptors
// These are queried by TVM's storage analysis to understand capacity/alignment.
// Fields confirmed from tvm/target/target_info.h:
//   unit_bits     — addressable unit size in bits
//   max_num_bits  — total memory capacity in bits
//   max_simd_bits — widest SIMD operation in bits (HVX = 1024-bit)
//   head_address  — base address PrimExpr (IntImm 0 = no fixed mapping)

static MemoryInfo GetHmxAccMem() {
  auto n = tvm::ffi::make_object<MemoryInfoNode>();
  // HMX accumulator register file: 32×32 int32 = 32768 bits
  n->unit_bits = 32;                // 32-bit int32 elements
  n->max_num_bits = 32LL * 32 * 32; // 32768 bits total
  n->max_simd_bits = 1024;          // HVX vector width
  n->head_address = IntImm(DataType::Int(32), 0);
  return MemoryInfo(n);
}

static MemoryInfo GetVtcmMem() {
  auto n = tvm::ffi::make_object<MemoryInfoNode>();
  // VTCM on Hexagon v73: 8 MB
  n->unit_bits = 8;                        // byte-addressable
  n->max_num_bits = 8LL * 1024 * 1024 * 8; // 8 MB in bits
  n->max_simd_bits = 1024;                 // HVX vector width
  n->head_address = IntImm(DataType::Int(32), 0);
  return MemoryInfo(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tilelang.transform.LowerHexagonIntrinsics", LowerHexagonIntrinsics)
      .def("tvm.info.mem.global.hmx.acc", GetHmxAccMem)
      .def("tvm.info.mem.global.vtcm", GetVtcmMem);
}

} // namespace transform
} // namespace tilelang
} // namespace tvm
