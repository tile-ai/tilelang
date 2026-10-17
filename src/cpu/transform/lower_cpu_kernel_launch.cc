/*!
 * \file lower_cpu_kernel_launch.cc
 * \brief Prepare CPU grid identity and per-launch thread counts.
 */

#include "cpu/op/builtin.h"
#include "support/check.h"
#include "transform/common/attr.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

bool IsGridBinding(const For &loop) {
  if (loop->kind != ForKind::kThreadBinding || !loop->thread_binding) {
    return false;
  }
  std::string tag = loop->thread_binding.value()->thread_tag;
  return tag.rfind("blockIdx.", 0) == 0;
}

// Read only this launch's header, leaving nested launches independent.
class CPULaunchAnnotationExtractor : public StmtMutator {
public:
  Optional<PrimExpr> NumThreads() const { return num_threads_; }

private:
  Stmt VisitStmt_(const ForNode *op) final { return GetRef<Stmt>(op); }

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = GetRef<SBlock>(op);
    if (!IsDeviceMainBlock(op)) {
      return block;
    }
    auto value = op->annotations.Get(attr::kCPUNumThreads);
    if (!value) {
      return block;
    }
    int64_t count;
    if (const auto *imm = value->as<IntImmNode>()) {
      count = imm->value;
    } else {
      count = value->cast<int64_t>();
    }
    ICHECK_GT(count, 0) << attr::kCPUNumThreads << " must be positive";
    num_threads_ = IntImm(DataType::Int(32), count);
    block.CopyOnWrite()->annotations.erase(attr::kCPUNumThreads);
    return block;
  }

  Optional<PrimExpr> num_threads_;
};

class CPUKernelLaunchLowerer : public StmtMutator {
public:
  explicit CPUKernelLaunchLowerer(bool parallel) : parallel_(parallel) {}

private:
  Stmt VisitStmt_(const ForNode *op) final {
    For head = GetRef<For>(op);
    if (!IsGridBinding(head)) {
      return StmtMutator::VisitStmt_(op);
    }
    std::vector<For> loops;
    Stmt body = head;
    while (const auto *node = body.as<ForNode>()) {
      For loop = GetRef<For>(node);
      if (!IsGridBinding(loop)) {
        break;
      }
      loops.push_back(loop);
      body = loop->body;
    }
    CPULaunchAnnotationExtractor extractor;
    body = VisitStmt(extractor(body));
    Optional<PrimExpr> num_threads = extractor.NumThreads();
    if (!parallel_ && !num_threads && body.same_as(loops.back()->body)) {
      return head;
    }
    for (size_t i = loops.size(); i-- > 0;) {
      For loop = loops[i];
      auto *node = loop.CopyOnWrite();
      node->body = body;
      if (parallel_ && !node->annotations.count(attr::kCPUGridDim)) {
        node->annotations.Set(
            attr::kCPUGridDim,
            IntImm(DataType::Int(32), static_cast<int64_t>(i)));
      }
      if (i == 0 && num_threads &&
          !node->annotations.count(attr::kCPUNumThreads)) {
        node->annotations.Set(attr::kCPUNumThreads, num_threads.value());
      }
      body = loop;
    }
    return body;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    Stmt block = StmtMutator::VisitStmt_(op);
    // A zero-dimensional launch has no grid loop to receive the metadata.
    CPULaunchAnnotationExtractor extractor;
    return extractor(block);
  }

  bool parallel_;
};

} // namespace

namespace transform {

tvm::transform::Pass LowerCPUKernelLaunch() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    bool parallel = ctx->GetConfig<Bool>(kCPUParallel, Bool(false)).value();
    CPUKernelLaunchLowerer lowerer(parallel);
    func.CopyOnWrite()->body = lowerer(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.LowerCPUKernelLaunch", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cpu.transform.LowerCPUKernelLaunch",
                        LowerCPUKernelLaunch);
}

} // namespace transform
} // namespace tl
} // namespace tvm
