#ifndef TVM_TL_TRANSFORM_COMMON_LAUNCH_PLAN_H_
#define TVM_TL_TRANSFORM_COMMON_LAUNCH_PLAN_H_

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <string>

namespace tvm {
namespace tl {

// Host preparation uses ordinary Bind nodes. SplitHostDevice owns the ABI.
// Keep this after device simplification so preparation cannot be inlined back
// into the device region.
class LaunchPlan {
public:
  // Callers must supply pure, host-evaluable expressions with in-scope
  // operands.
  tirx::Var Prepare(const PrimExpr &value, const std::string &name) {
    for (const tirx::Bind &binding : bindings_) {
      if (ffi::StructuralEqual()(binding->value, value)) {
        return binding->var;
      }
    }
    tirx::Var var(name, value.dtype());
    bindings_.push_back(tirx::Bind(var, value));
    return var;
  }

  // CUDA's TVM scalar packer uses 32/64-bit slots and lacks uint64 slots.
  // Arithmetic keeps its logical type; only the captured ABI value changes.
  PrimExpr PrepareArgument(const PrimExpr &value, const std::string &name) {
    DataType dtype = value.dtype();
    DataType abi_type = dtype;
    if (dtype.is_scalar() && (dtype.is_int() || dtype.is_uint())) {
      if (dtype.bits() < 32) {
        abi_type = dtype.is_int() ? DataType::Int(32) : DataType::UInt(32);
      } else if (dtype == DataType::UInt(64)) {
        abi_type = DataType::Int(64);
      }
    }
    return tvm::cast(dtype, Prepare(tvm::cast(abi_type, value), name));
  }

  tirx::Stmt Materialize(const tirx::Stmt &launch) const {
    ffi::Array<tirx::Stmt> statements;
    for (const tirx::Bind &binding : bindings_) {
      statements.push_back(binding);
    }
    statements.push_back(launch);
    return tirx::SeqStmt::Flatten(statements);
  }

private:
  ffi::Array<tirx::Bind> bindings_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_LAUNCH_PLAN_H_
