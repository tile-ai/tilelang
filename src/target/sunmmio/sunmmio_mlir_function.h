#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_FUNCTION_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_FUNCTION_H_

#include "sunmmio_mlir_context.h"
#include "sunmmio_mlir_type.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace tvm {
namespace codegen {

class SunmmioMlirFunction {
public:
  explicit SunmmioMlirFunction(SunmmioMlirContext &ctx);

  void BeginModule();
  void EndModule();

  void BeginFunction(const std::string &name,
                     const std::vector<BuilderArg> &args);
  void EndFunction();
  void EmitReturn();

  void BeginFor(const std::string &iv, const SunMMIOValue &lb,
                const SunMMIOValue &ub, const SunMMIOValue &step,
                const ffi::Map<ffi::String, ffi::Any> &annotations,
                const std::vector<SunMMIOValue> &live_out_values);
  void EndFor();

  void BeginIf(const SunMMIOValue &cond,
               const std::vector<SunMMIOValue> &live_out_values);
  void BeginElse();
  void EndIf();

  void BeginWhile(const std::vector<SunMMIOValue> &live_out_values);
  void BeginWhileBody(const SunMMIOValue &cond);
  void EndWhile();

  void EmitAssert(const SunMMIOValue &cond, const std::string &msg_text);

private:
  SunmmioMlirContext &ctx_;
  SunmmioMlirType type_;
  mlir::func::FuncOp current_func_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_FUNCTION_H_
