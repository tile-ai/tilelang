/*!
 * \file materialize_kernel_launch.cc
 * \brief Materialize the target-neutral kernel launch nest emitted by
 *        T.Kernel into a backend-specific form.
 *
 * T.Kernel traces into a nest of For loops with ForKind::kThreadBinding
 * tagged blockIdx.* / threadIdx.*. This pass runs right after BindTarget
 * and rewrites the nest according to `lower_thread_binding`, which each
 * backend pipeline chooses for itself (no target dispatch happens here):
 *  - lower_thread_binding = true (SIMT backends, e.g. CUDA/ROCm/Metal):
 *    each launch loop becomes an AttrStmt thread_extent, reusing the loop
 *    variable so body references stay valid.
 *  - lower_thread_binding = false (backends without SIMT, e.g. CPU):
 *    blockIdx.* loops become serial For loops over the grid extent;
 *    threadIdx.* loops are ignored; they become unit serial loops so the
 *    loop variable stays defined (pinned to 0) while the requested thread
 *    extent (e.g. the default threads=128) is dropped.
 *
 * Only the outermost contiguous launch nest is converted; thread_binding
 * loops deeper inside the kernel body (separated by the tilelang_root
 * block) are left for LowerOpaqueBlock to handle at its usual stage.
 */

#include "common/attr.h"
#include "op/builtin.h"
#include "support/check.h"
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tvm::ffi;

namespace {

bool IsBlockBinding(const ForNode *op) {
  if (op->kind != ForKind::kThreadBinding || !op->thread_binding.defined())
    return false;
  std::string tag = op->thread_binding.value()->thread_tag;
  return tag.rfind("blockIdx.", 0) == 0;
}

bool IsThreadBinding(const ForNode *op) {
  if (op->kind != ForKind::kThreadBinding || !op->thread_binding.defined())
    return false;
  std::string tag = op->thread_binding.value()->thread_tag;
  return tag.rfind("threadIdx.", 0) == 0;
}

bool IsLaunchBinding(const ForNode *op) {
  return IsBlockBinding(op) || IsThreadBinding(op);
}

class KernelLaunchMaterializer : public StmtMutator {
public:
  explicit KernelLaunchMaterializer(bool lower_thread_binding,
                                    bool annotate_grid)
      : lower_thread_binding_(lower_thread_binding),
        annotate_grid_(annotate_grid) {}

  Stmt VisitStmt_(const ForNode *op) final {
    if (IsLaunchBinding(op)) {
      block_dim_ = 0; // annotation values are per-launch grid dim indices
      return ConvertNest(op);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  // Peel the contiguous launch nest rooted at `op` without descending into
  // the kernel body below it.
  Stmt ConvertNest(const ForNode *op) {
    Stmt body;
    if (const ForNode *inner = op->body.as<ForNode>();
        inner && IsLaunchBinding(inner)) {
      body = ConvertNest(inner);
    } else {
      body = op->body;
    }
    if (lower_thread_binding_) {
      ffi::String tag = op->thread_binding.value()->thread_tag;
      IterVar iter_var(Range::FromMinExtent(op->min, op->extent), op->loop_var,
                       IterVarType::kThreadIndex, tag);
      return AttrStmt(std::move(iter_var), tirx::attr::thread_extent,
                      op->extent, std::move(body), op->span);
    }
    // No SIMT: grid dims run as plain serial loops; thread dims are ignored
    // (a unit loop keeps the loop variable defined and pinned to 0).
    bool is_block = IsBlockBinding(op);
    PrimExpr extent =
        is_block ? op->extent : PrimExpr(IntImm(op->extent.dtype(), 1));
    Map<ffi::String, ffi::Any> annotations = op->annotations;
    if (annotate_grid_ && is_block && !annotations.count(attr::kCPUGridDim)) {
      // Annotate every grid dim, including unit-extent ones: the consumer
      // treats the annotated nest as one contiguous chain, and skipping a
      // middle dim would cut the deeper dims off from parallelization (the
      // llvm side skips unit dims itself when picking the loop to mark).
      annotations.Set(attr::kCPUGridDim, IntImm(DataType::Int(32), block_dim_));
    }
    if (is_block) {
      ++block_dim_;
    }
    return For(op->loop_var, op->min, std::move(extent), ForKind::kSerial,
               std::move(body),
               /*thread_binding=*/std::nullopt, std::move(annotations),
               op->step, op->span);
  }

  bool lower_thread_binding_;
  bool annotate_grid_;
  int block_dim_ = 0;
};

} // namespace

tvm::transform::Pass MaterializeKernelLaunch(bool lower_thread_binding) {
  using namespace tirx::transform;
  auto pass_func = [lower_thread_binding](
                       PrimFunc func, const IRModule &mod,
                       const tvm::transform::PassContext &ctx) -> PrimFunc {
    bool annotate_grid = false;
    if (!lower_thread_binding) {
      annotate_grid = ctx->GetConfig<Bool>(kCPUParallel, Bool(false)).value();
    }
    KernelLaunchMaterializer mutator(lower_thread_binding, annotate_grid);
    func.CopyOnWrite()->body = mutator(func->body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MaterializeKernelLaunch", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MaterializeKernelLaunch",
                        MaterializeKernelLaunch);
}

} // namespace tl
} // namespace tvm
