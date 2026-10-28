#include "backend/common/op/arg_reduce.h"

namespace tvm {
namespace tl {
namespace {
struct CudaArgReduce : backend::ArgReduceLowerer<CudaArgReduce> {
  static int WarpSize(Target target) { return TargetCudaGetWarpSize(target); }
  static tirx::Stmt Sync(const LowerArgs &args) {
    return tirx::Evaluate(tirx::Call(
        DataType::Int(32), tirx::builtin::tvm_storage_sync(),
        {tirx::StringImm("shared"), Integer(1), args.thread_bounds->extent}));
  }
};
bool Match(Target target) {
  return TargetIsCuda(target) && !TargetIsCuTeDSL(target);
}
const bool registered = [] {
  RegisterArgReduceImpl({"cuda.ArgReduce", Match, CudaArgReduce::Lower});
  return true;
}();
} // namespace
} // namespace tl
} // namespace tvm
