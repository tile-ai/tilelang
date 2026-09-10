/*!
 * \file materialize_kernel_launch.cc
 * \brief Materialize the target-neutral kernel launch nest emitted by
 *        T.Kernel into a backend-specific form.
 *
 * T.Kernel traces into
 *
 *   for bx in thread_binding(blockIdx.x):        # one per grid axis
 *     tx = tl.launch_thread_idx(0)               # x, y, z placeholders
 *     ty = tl.launch_thread_idx(1)
 *     tz = tl.launch_thread_idx(2)
 *     block tilelang_root { annotations: tl.launch_threads?, ... }
 *
 * The grid loops are the program-index space every backend shares. The
 * thread placeholders only reserve Var identities the body may reference;
 * what they mean, and how many threads run, is decided here once the Target
 * is bound. Each backend pipeline chooses the mode for itself (no target
 * dispatch happens in this pass):
 *  - lower_thread_binding = true (SIMT backends, e.g. CUDA/ROCm/Metal):
 *    grid loops become AttrStmt thread_extent scopes; each thread placeholder
 *    is rebound as a threadIdx.* thread_extent over the same Var, with the
 *    extent taken from the `tl.launch_threads` annotation (T.Kernel
 *    threads=...) or, failing that, from `default_threads`.
 *  - lower_thread_binding = false (backends without SIMT, e.g. CPU):
 *    grid loops become serial For loops; thread placeholders are dropped and
 *    `tl.launch_threads` is ignored. A body that references a thread index
 *    has no meaning on such a target and is rejected.
 * Launch annotations listed in `unsupported_annotations` (e.g. `cluster_dims`
 * on a target without thread block clusters) are rejected instead of being
 * silently dropped further down the pipeline.
 *
 * Only the outermost contiguous launch nest is converted; thread_binding
 * loops deeper inside the kernel body (separated by the tilelang_root
 * block) are left for LowerOpaqueBlock to handle at its usual stage.
 */

#include "../op/builtin.h"
#include "common/attr.h"
#include "support/check.h"
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Optional;

namespace {

constexpr int kNumThreadAxes = 3;
constexpr const char *kThreadTags[kNumThreadAxes] = {
    "threadIdx.x", "threadIdx.y", "threadIdx.z"};

bool IsBlockBinding(const ForNode *op) {
  if (op->kind != ForKind::kThreadBinding || !op->thread_binding.defined())
    return false;
  std::string tag = op->thread_binding.value()->thread_tag;
  return tag.rfind("blockIdx.", 0) == 0;
}

// `v = tl.launch_thread_idx(axis)` emitted by T.Kernel.
bool IsLaunchThreadPlaceholder(const Stmt &stmt) {
  const BindNode *bind = stmt.as<BindNode>();
  if (!bind)
    return false;
  const CallNode *call = bind->value.as<CallNode>();
  return call && call->op.same_as(launch_thread_idx());
}

int ThreadAxisOf(const BindNode *bind) {
  const CallNode *call = bind->value.as<CallNode>();
  ICHECK(call && call->args.size() == 1);
  const int64_t *axis = as_const_int(call->args[0]);
  ICHECK(axis && *axis >= 0 && *axis < kNumThreadAxes)
      << "tl.launch_thread_idx expects a constant axis in [0, 3), got "
      << call->args[0];
  return static_cast<int>(*axis);
}

// The tilelang_root block that carries the launch annotations, if `body` is
// the kernel body directly below the launch nest.
const SBlockNode *GetLaunchBlock(const Stmt &body) {
  const SBlockRealizeNode *realize = body.as<SBlockRealizeNode>();
  if (!realize || !IsDeviceMainBlock(realize->block.get()))
    return nullptr;
  return realize->block.get();
}

Optional<Array<PrimExpr>> GetLaunchThreads(const Stmt &body) {
  const SBlockNode *block = GetLaunchBlock(body);
  if (!block)
    return std::nullopt;
  if (auto threads = block->annotations.Get(attr::kLaunchThreads)) {
    if (auto arr = threads.value().try_cast<Array<PrimExpr>>())
      return arr.value();
    LOG(FATAL) << "Expected `" << attr::kLaunchThreads
               << "` to be an Array<PrimExpr>, but got "
               << threads.value().GetTypeKey();
  }
  return std::nullopt;
}

class KernelLaunchMaterializer : public StmtMutator {
public:
  KernelLaunchMaterializer(bool lower_thread_binding,
                           Optional<Array<PrimExpr>> default_threads,
                           Array<ffi::String> unsupported_annotations,
                           ffi::String target_name)
      : lower_thread_binding_(lower_thread_binding),
        default_threads_(std::move(default_threads)),
        unsupported_annotations_(std::move(unsupported_annotations)),
        target_name_(std::move(target_name)) {}

  Stmt VisitStmt_(const ForNode *op) final {
    if (IsBlockBinding(op)) {
      return ConvertNest(GetRef<Stmt>(op));
    }
    return StmtMutator::VisitStmt_(op);
  }

  // A launch without grid axes starts directly at the thread placeholders.
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    if (op->size() > 0 && IsLaunchThreadPlaceholder(op->seq[0])) {
      return ConvertNest(GetRef<Stmt>(op));
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  // Peel the contiguous launch nest rooted at `root` without descending into
  // the kernel body below it, then rebuild it in the backend's form.
  Stmt ConvertNest(const Stmt &root) {
    std::vector<const ForNode *> grid_loops;
    Stmt body = root;
    while (const ForNode *loop = body.as<ForNode>()) {
      if (!IsBlockBinding(loop))
        break;
      grid_loops.push_back(loop);
      body = loop->body;
    }

    std::vector<const BindNode *> thread_binds;
    body = PeelThreadPlaceholders(body, &thread_binds);
    RejectUnsupportedAnnotations(body);

    body = lower_thread_binding_ ? BindThreads(thread_binds, body)
                                 : DropThreads(thread_binds, body);

    for (auto it = grid_loops.rbegin(); it != grid_loops.rend(); ++it) {
      const ForNode *loop = *it;
      if (lower_thread_binding_) {
        ffi::String tag = loop->thread_binding.value()->thread_tag;
        IterVar iter_var(Range::FromMinExtent(loop->min, loop->extent),
                         loop->loop_var, IterVarType::kThreadIndex, tag);
        body = AttrStmt(std::move(iter_var), tirx::attr::thread_extent,
                        loop->extent, std::move(body), loop->span);
      } else {
        body = For(loop->loop_var, loop->min, loop->extent, ForKind::kSerial,
                   std::move(body),
                   /*thread_binding=*/std::nullopt, loop->annotations,
                   loop->step, loop->span);
      }
    }
    return body;
  }

  // Split the leading `v = tl.launch_thread_idx(axis)` binds off `stmt` and
  // return what follows them.
  static Stmt PeelThreadPlaceholders(const Stmt &stmt,
                                     std::vector<const BindNode *> *binds) {
    const SeqStmtNode *seq = stmt.as<SeqStmtNode>();
    if (!seq)
      return stmt;
    size_t i = 0;
    while (i < seq->size() && IsLaunchThreadPlaceholder(seq->seq[i])) {
      binds->push_back(seq->seq[i].as<BindNode>());
      ++i;
    }
    if (i == 0)
      return stmt;
    ICHECK_LT(i, seq->size())
        << "T.Kernel launch has thread placeholders but no body";
    if (i + 1 == seq->size())
      return seq->seq[i];
    return SeqStmt(Array<Stmt>(seq->seq.begin() + i, seq->seq.end()));
  }

  // SIMT: every placeholder becomes a threadIdx.* thread_extent scope over
  // the same Var so body references stay valid.
  Stmt BindThreads(const std::vector<const BindNode *> &thread_binds,
                   Stmt body) {
    if (thread_binds.empty())
      return body;
    Array<PrimExpr> extents = ResolveThreadExtents(body);
    for (auto it = thread_binds.rbegin(); it != thread_binds.rend(); ++it) {
      const BindNode *bind = *it;
      int axis = ThreadAxisOf(bind);
      PrimExpr extent = extents[axis];
      IterVar iter_var(
          Range::FromMinExtent(make_zero(bind->var.dtype()), extent), bind->var,
          IterVarType::kThreadIndex, kThreadTags[axis]);
      body = AttrStmt(std::move(iter_var), tirx::attr::thread_extent, extent,
                      std::move(body), bind->span);
    }
    return body;
  }

  // No SIMT: thread placeholders carry no meaning, so they are removed. A body
  // that reads one would otherwise silently run as a single thread.
  Stmt DropThreads(const std::vector<const BindNode *> &thread_binds,
                   Stmt body) {
    for (const BindNode *bind : thread_binds) {
      const VarNode *var = bind->var.get();
      if (UsesVar(body, [var](const VarNode *v) { return v == var; })) {
        LOG(FATAL) << "T.Kernel body references thread index `"
                   << bind->var->name_hint << "`, but target `" << target_name_
                   << "` has no SIMT threads. Express the computation with "
                      "tile-level operators (T.Parallel, T.copy, ...) instead "
                      "of per-thread indexing.";
      }
    }
    return body;
  }

  void RejectUnsupportedAnnotations(const Stmt &body) const {
    const SBlockNode *block = GetLaunchBlock(body);
    if (!block)
      return;
    for (const ffi::String &key : unsupported_annotations_) {
      if (block->annotations.count(key)) {
        LOG(FATAL) << "T.Kernel launch annotation `" << key
                   << "` is not supported on target `" << target_name_ << "`";
      }
    }
  }

  Array<PrimExpr> ResolveThreadExtents(const Stmt &body) {
    Optional<Array<PrimExpr>> threads = GetLaunchThreads(body);
    if (!threads.defined())
      threads = default_threads_;
    ICHECK(threads.defined())
        << "T.Kernel did not specify threads= and target `" << target_name_
        << "` provides no default thread-block size";
    Array<PrimExpr> extents = threads.value();
    ICHECK_LE(extents.size(), static_cast<size_t>(kNumThreadAxes));
    while (extents.size() < static_cast<size_t>(kNumThreadAxes)) {
      extents.push_back(IntImm(DataType::Int(32), 1));
    }
    return extents;
  }

  bool lower_thread_binding_;
  Optional<Array<PrimExpr>> default_threads_;
  Array<ffi::String> unsupported_annotations_;
  ffi::String target_name_;
};

} // namespace

tvm::transform::Pass
MaterializeKernelLaunch(bool lower_thread_binding,
                        Optional<Array<PrimExpr>> default_threads,
                        Optional<Array<ffi::String>> unsupported_annotations) {
  using namespace tirx::transform;
  Array<ffi::String> unsupported =
      unsupported_annotations.value_or(Array<ffi::String>());
  auto pass_func = [lower_thread_binding, default_threads, unsupported](
                       PrimFunc func, const IRModule &mod,
                       const tvm::transform::PassContext &ctx) -> PrimFunc {
    ffi::String target_name = "<unbound>";
    if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
      target_name = target.value()->kind->name;
    }
    KernelLaunchMaterializer mutator(lower_thread_binding, default_threads,
                                     unsupported, target_name);
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
