/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_HIP_H_
#define TVM_TL_TARGET_CODEGEN_HIP_H_

#include "backend/common/codegen/codegen_ir_utils.h"
#include "support/check.h"
#include <tvm/codegen/codegen.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>

#include <string>
#include <unordered_map>

#include "backend/common/codegen/codegen_c_base.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangHIP final : public CodeGenTileLangCBase {
public:
  using CodeGenTileLangCBase::Dispatch_;
  using CodeGenTileLangCBase::PrintType;
  CodeGenTileLangHIP();
  void SetTarget(Target target) { target_ = std::move(target); }
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const tirx::Function &f, std::ostream &os) final;
  void Dispatch_(const ForObj *op) final;
  void PrintStorageSync(const CallObj *op) final;
  void PrintStorageScope(const runtime::MemoryScope &memory_scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, const PrimType &t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final;             // NOLINT(*)
  void PrintType(const PrimType &t, std::ostream &os) final; // NOLINT(*)
  void PrintVecElemLoad(const std::string &vec, const PrimType &t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, const PrimType &t, int i,
                         const std::string &value) final;
  void BindThreadIndex(const PrimVar &var,
                       const ffi::String &thread_tag) final; // NOLINT(*)
  void PrintVecElemLoadExpr(const PrimType &t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, const PrimType &from,
                         const PrimType &target) final;
  // overload visitor
  void Dispatch_(const prim::RampObj *op, std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const prim::BroadcastObj *op,
                 std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const FloatImmObj *op, std::ostream &os) final;
  void Dispatch_(const CallObj *op, std::ostream &os) final;
  void Dispatch_(const prim::CastObj *op, std::ostream &os) final;
  void Dispatch_(const prim::NotObj *op, std::ostream &os) final;
  void Dispatch_(const prim::SelectObj *op, std::ostream &os) final;
  void Dispatch_(const prim::ShuffleObj *op,
                 std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const BindObj *op) final;
  void Dispatch_(const RegionStmtObj *op) final;
  void Dispatch_(const TensorStoreObj *op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const tirx::Function &f);

protected:
  void EmitPreamble_(std::ostream &os) const final;
  virtual std::string GetBufferRef(const PrimType &t, const VarObj *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<Expr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const TensorLoadObj *op,
                           std::ostream &os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmObj *op, std::ostream &os,
                         CodeGenTileLangHIP *p);

  // whether need hip_cooperative_groups.h
  bool need_cooperative_groups_{false};
  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need mfma.h
  bool need_wmma_h_{false};
  // whether need fp8.h
  bool enable_fp8_{false};
  // whether need hip_fp4.h (gfx950 only)
  bool enable_fp4_{false};
  // Map from FP4 buffer Var to packed buffer variable name (gfx950)
  std::unordered_map<Var, std::string> fp4_packed_buffers_;
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;
  // Target
  ffi::Optional<Target> target_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_HIP_H_
