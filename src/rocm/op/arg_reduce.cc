#include "backend/common/op/arg_reduce.h"

namespace tvm {
namespace tl {
namespace {
struct RocmArgReduce : backend::ArgReduceLowerer<RocmArgReduce> {
  static int WarpSize(Target target) { return TargetRocmGetWarpSize(target); }
  static tirx::Stmt Sync(const LowerArgs &) {
    return tirx::Evaluate(tirx::Call(DataType::Int(32),
                                     tirx::builtin::tvm_storage_sync(),
                                     {tirx::StringImm("shared")}));
  }
  static tirx::Stmt Lower(const ArgReduceOpNode &op, const LowerArgs &args,
                          arith::Analyzer *analyzer) {
    CHECK(args.block_thread_extent.defined() &&
              analyzer->CanProveEqual(args.thread_bounds->min, 0) &&
              analyzer->CanProveEqual(args.thread_bounds->extent,
                                      args.block_thread_extent),
          ValueError)
        << "ROCm arg_reduce requires all threads in the block to participate";
    return backend::ArgReduceLowerer<RocmArgReduce>::Lower(op, args, analyzer);
  }
};
const bool registered = [] {
  RegisterArgReduceImpl({"rocm.ArgReduce", TargetIsRocm, RocmArgReduce::Lower});
  return true;
}();
} // namespace
} // namespace tl
} // namespace tvm
