/*!
 * \file cpu/codegen/codegen_c.h
 * \brief TileLang CPU C code generation over native TIRx IR.
 */
#ifndef TILELANG_CPU_CODEGEN_CODEGEN_C_H_
#define TILELANG_CPU_CODEGEN_CODEGEN_C_H_

#include <unordered_set>

#include "backend/common/codegen/codegen_c_host.h"

namespace tvm::codegen {

/*!
 * \brief CPU device emitter sharing TileLang's FFI and scalar C implementation.
 *
 * The device-specific behavior is the template preamble, clamp, OpenMP loop
 * chains, stack storage, and portable non-finite floating-point literals.
 */
class CodeGenTileLangC final : public tl::CodeGenCHost {
public:
  using tl::CodeGenCHost::Dispatch;
  using tl::CodeGenCHost::Dispatch_;
  using tl::CodeGenCHost::PrintType;

  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
            std::string target_str,
            const std::unordered_set<std::string> &devices);
  void Dispatch_(const CallObj *op, std::ostream &os) override;
  void InitFuncState(const Function &func) override;
  void Dispatch_(const FloatImmObj *op, std::ostream &os) override;
  void Dispatch_(const prim::BroadcastObj *op, std::ostream &os) override;
  void Dispatch_(const prim::SelectObj *op, std::ostream &os) override;
  void Dispatch_(const prim::NotObj *op, std::ostream &os) override;
  void PrintType(const PrimType &type, std::ostream &os) override;
  void PrintVecBinaryOp(const std::string &op, const PrimType &type,
                        PrimExpr lhs, PrimExpr rhs, std::ostream &os) override;
  void Dispatch_(const ForObj *op) override;
  void PrintStorageScope(const runtime::MemoryScope &scope,
                         std::ostream &os) override;

protected:
  void PrintVecElemLoad(const std::string &vec, const PrimType &type, int lane,
                        std::ostream &os) override;
  void PrintVecElemStore(const std::string &vec, const PrimType &type, int lane,
                         const std::string &value) override;
  void PrintVecElemLoadExpr(const PrimType &type, int lane,
                            const std::string &value,
                            std::ostream &os) override;

private:
  // Identity-based membership keeps a separate parallel loop inside a wrapper
  // independent of the directly nested OpenMP collapse chain.
  std::unordered_set<const ForObj *> omp_chain_members_;
};

} // namespace tvm::codegen

#endif // TILELANG_CPU_CODEGEN_CODEGEN_C_H_
