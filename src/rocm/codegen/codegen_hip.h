#ifndef TVM_TL_TARGET_CODEGEN_HIP_H_
#define TVM_TL_TARGET_CODEGEN_HIP_H_

#include "backend/common/codegen/codegen_c_base.h"
#include "support/check.h"

#include <tvm/codegen/codegen.h>
#include <tvm/target/target.h>

#include <string>
#include <unordered_map>

namespace tvm::codegen {

class CodeGenTileLangHIP final : public CodeGenTileLangCBase {
public:
  CodeGenTileLangHIP();
  using CodeGenC::Dispatch_;
  using CodeGenC::PrintType;
  void SetTarget(Target target) { target_ = std::move(target); }
  void PrintFuncPrefix(std::ostream &stream) final;
  void PrintExtraAttrs(const tirx::Function &function,
                       std::ostream &stream) final;
  void Dispatch_(const ForNode *node) final;
  void PrintStorageSync(const CallNode *node) final;
  void PrintStorageScope(const runtime::MemoryScope &scope,
                         std::ostream &stream) final;
  void PrintVecBinaryOp(const std::string &operation, const PrimType &type,
                        PrimExpr left, PrimExpr right,
                        std::ostream &stream) final;
  void PrintType(const PrimType &type, std::ostream &stream) final;
  void PrintVecElemLoad(const std::string &vector, const PrimType &type,
                        int lane, std::ostream &stream) final;
  void PrintVecElemStore(const std::string &vector, const PrimType &type,
                         int lane, const std::string &value) final;
  void BindThreadIndex(const PrimVar &variable,
                       const ffi::String &thread_tag) final;
  void PrintVecElemLoadExpr(const PrimType &type, int lane,
                            const std::string &value,
                            std::ostream &stream) final;
  std::string CastFromTo(std::string value, const PrimType &source,
                         const PrimType &destination) final;
  void Dispatch_(const prim::RampNode *node, std::ostream &stream) final;
  void Dispatch_(const prim::BroadcastNode *node, std::ostream &stream) final;
  void Dispatch_(const FloatImmNode *node, std::ostream &stream) final;
  void Dispatch_(const CallNode *node, std::ostream &stream) final;
  void Dispatch_(const prim::CastNode *node, std::ostream &stream) final;
  void Dispatch_(const prim::NotNode *node, std::ostream &stream) final;
  void Dispatch_(const prim::SelectNode *node, std::ostream &stream) final;
  void Dispatch_(const prim::ShuffleNode *node, std::ostream &stream) final;
  void Dispatch_(const BindNode *node) final;
  void Dispatch_(const RegionStmtNode *node) final;
  void Dispatch_(const TensorStoreNode *node) final;
  void AddFunction(const GlobalVar &global,
                   const tirx::Function &function) final;

protected:
  void EmitPreamble_(std::ostream &stream) const final;
  std::string GetBufferRef(const PrimType &type, const VarNode *buffer,
                           PrimExpr index) final;
  void PrintCallExtern(Type return_type, ffi::String symbol,
                       const ffi::Array<Expr> &arguments,
                       bool skip_first_argument, std::ostream &stream) final;

private:
  void HandleVolatileLoads(const std::string &value, const TensorLoadNode *node,
                           std::ostream &stream) final;
  bool IsScopePartOfType() const final { return false; }
  friend void PrintConst(const FloatImmNode *node, std::ostream &stream,
                         CodeGenTileLangHIP *codegen);

  bool need_cooperative_groups_{false};
  bool need_math_constants_h_{false};
  bool need_wmma_h_{false};
  bool enable_fp8_{false};
  bool enable_fp4_{false};
  std::unordered_map<Var, std::string> fp4_packed_buffers_;
  int barrier_count_ = -1;
  bool need_mma_h_{false};
  bool need_cast_smem_ptr_to_int_{false};
  const std::string barrier_name_ = "barrier";
  const int barrier_alignment_bytes_ = 16;
  ffi::Optional<Target> target_;
};

} // namespace tvm::codegen

#endif
