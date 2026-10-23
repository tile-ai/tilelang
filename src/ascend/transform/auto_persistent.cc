/*!
 * \file auto_persistent.cc
 * \brief Fold an Ascend logical launch grid onto the physical NPU cores.
 *
 * AutoPersistent runs before MaterializeKernelLaunch, while
 * T.PersistentKernel is still a thread-binding For loop. It folds the 1-D
 * logical grid onto the required ``num_cores`` physical cores and executes
 * additional logical tasks in wave-major order:
 *
 *   logical_id = wave * physical_core_count + physical_block_id
 *
 * Consequently adjacent logical tasks in each wave are striped across
 * different physical cores.  This is the desired L2-friendly assignment for
 * workloads whose adjacent tasks usually touch adjacent memory.
 */

#include "op/builtin.h"
#include "support/check.h"
#include "transform/common/attr.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// Frontend contract with tilelang.ascend.language.PersistentKernel. Recorded on
// the tilelang_root launch block and consumed/stripped by AutoPersistent;
// defined here rather than a shared attr header because only this pass reads
// them.
constexpr const char *tilelang_persistent_kernel_num_cores =
    "tilelang.persistent_kernel_num_cores";
constexpr const char *tilelang_persistent_kernel_annotations =
    "tilelang.persistent_kernel_annotations";

// `tx = tl.launch_thread_idx(0); ty = ...; tz = ...` placeholders that the new
// KernelLaunch always emits between the blockIdx.x loop and the tilelang_root
// block.
bool IsLaunchThreadPlaceholder(const Stmt &stmt) {
  const BindNode *bind = stmt.as<BindNode>();
  if (!bind)
    return false;
  const CallNode *call = bind->value.as<CallNode>();
  return call && call->op.same_as(launch_thread_idx());
}

bool IsBlockBinding(const ForNode *op) {
  if (op->kind != ForKind::kThreadBinding || !op->thread_binding.defined())
    return false;
  std::string tag = op->thread_binding.value()->thread_tag;
  return tag.rfind("blockIdx.", 0) == 0;
}

class LaunchVarUseDetector : public StmtExprVisitor {
public:
  explicit LaunchVarUseDetector(const ffi::Array<Var> &launch_vars) {
    for (const Var &var : launch_vars)
      launch_vars_.insert(var);
  }

  void VisitExpr_(const VarNode *op) final {
    if (launch_vars_.count(GetRef<Var>(op)))
      found = true;
  }

  // StmtVisitor::VisitStmt_(SBlockNode) visits iter_vars / alloc_buffers /
  // reads / writes / match_buffers / init / body but not `annotations`, so a
  // launch variable referenced only by an annotation value would be missed
  // here. Scan the annotation values explicitly as well.
  void VisitStmt_(const SBlockNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    for (const auto &kv : op->annotations)
      VisitAnnotationValue(kv.second);
  }

  bool found{false};

private:
  // An annotation value is a type-erased ffi::Any; only PrimExpr / Array / Map
  // can carry IR expressions, so recurse through those and ignore the rest
  // (int, bool, String, ...).
  void VisitAnnotationValue(const ffi::Any &value) {
    if (const auto expr = value.as<PrimExpr>()) {
      this->VisitExpr(expr.value());
    } else if (const auto arr = value.as<ffi::Array<ffi::ObjectRef>>()) {
      for (const ffi::ObjectRef &elem : arr.value())
        VisitAnnotationValue(ffi::Any(elem));
    } else if (const auto map = value.as<ffi::Map<ffi::String, ffi::Any>>()) {
      for (const auto &kv : map.value())
        VisitAnnotationValue(kv.second);
    }
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> launch_vars_;
};

For RebuildFor(const For &loop, Stmt body) {
  return For(loop->loop_var, loop->min, loop->extent, loop->kind,
             std::move(body), loop->thread_binding, loop->annotations,
             loop->step, loop->span);
}

struct PersistentKernelConfig {
  int64_t core_count{0};
  ffi::Map<ffi::String, ffi::Any> loop_annotations;
};

bool IsPersistentKernelRoot(const SBlock &root) {
  return root->annotations.Get(tilelang_persistent_kernel_num_cores)
      .has_value();
}

PersistentKernelConfig ReadPersistentKernelConfig(const SBlock &root) {
  PersistentKernelConfig config;
  Optional<Any> num_cores =
      root->annotations.Get(tilelang_persistent_kernel_num_cores);
  ICHECK(num_cores.has_value())
      << "T.PersistentKernel requires num_cores; the Python frontend enforces "
         "this, so this check only triggers for hand-written/injected TIR";
  int64_t count = num_cores.value().cast<int64_t>();
  ICHECK_GT(count, 0)
      << "T.PersistentKernel num_cores must be a positive integer";
  config.core_count = count;
  if (Optional<Any> value =
          root->annotations.Get(tilelang_persistent_kernel_annotations)) {
    Optional<Map<String, Any>> annotations =
        value.value().as<Map<String, Any>>();
    ICHECK(annotations.has_value())
        << "T.PersistentKernel annotations must be a string-keyed map";
    config.loop_annotations = annotations.value();
  }
  return config;
}

SBlockRealize StripPersistentKernelConfig(const SBlockRealize &realize) {
  SBlock root = realize->block;
  ffi::Map<ffi::String, ffi::Any> annotations = root->annotations;
  annotations.erase(tilelang_persistent_kernel_num_cores);
  annotations.erase(tilelang_persistent_kernel_annotations);
  root.CopyOnWrite()->annotations = std::move(annotations);
  return SBlockRealize(realize->iter_values, realize->predicate, root,
                       realize->span);
}

bool RootMetadataUsesLaunchVars(const SBlockRealize &realize,
                                const ffi::Array<Var> &launch_vars) {
  // The new logical coordinates are Bind statements inside the root body.
  // They cannot legally appear in the realization predicate, allocation
  // shapes, or other root metadata.
  const SBlock &block = realize->block;
  SBlock metadata_only(block->iter_vars, block->reads, block->writes,
                       block->name_hint, Evaluate(0), block->init,
                       block->alloc_buffers, block->match_buffers,
                       block->annotations, block->span);
  SBlockRealize metadata_realize(realize->iter_values, realize->predicate,
                                 metadata_only, realize->span);
  LaunchVarUseDetector detector(launch_vars);
  detector(metadata_realize);
  return detector.found;
}

class AutoPersistentRewriter : public StmtMutator {
public:
  AutoPersistentRewriter() = default;

  PrimFunc Rewrite(PrimFunc func) {
    Stmt launch_stmt = func->body;
    Optional<SBlockRealize> host_root;
    if (const auto *realize = launch_stmt.as<SBlockRealizeNode>();
        realize && IsHostMainBlock(realize->block.get())) {
      host_root = GetRef<SBlockRealize>(realize);
      launch_stmt = realize->block->body;
    }

    auto wrap_host_root = [&](Stmt body) -> Stmt {
      if (!host_root.defined())
        return body;
      SBlockRealize host = host_root.value();
      SBlock block = host->block;
      block.CopyOnWrite()->body = std::move(body);
      return SBlockRealize(host->iter_values, host->predicate, block,
                           host->span);
    };

    Stmt transformed = (*this)(launch_stmt);
    func.CopyOnWrite()->body = wrap_host_root(std::move(transformed));
    return func;
  }

  // Fold a T.PersistentKernel launch in place. Other launches are left
  // unchanged, and — like MaterializeKernelLaunch — the mutator recurses
  // through host control flow so a launch nested in an IfThenElse or other
  // statement is still folded.
  Stmt VisitStmt_(const ForNode *op) final {
    if (IsBlockBinding(op)) {
      if (Optional<For> folded = FoldPersistentLaunch(GetRef<For>(op)))
        return folded.value();
      return GetRef<For>(op);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  Optional<For> FoldPersistentLaunch(const For &logical_loop) {
    // KernelLaunch always emits tx/ty/tz = tl.launch_thread_idx(...)
    // placeholders between the blockIdx.x loop and the tilelang_root block, so
    // peel them (keeping the prefix to re-wrap when rebuilding) and locate the
    // launch root below them.
    ffi::Array<Stmt> placeholder_prefix;
    const SBlockRealizeNode *root_realize_node = nullptr;
    if (const auto *seq = logical_loop->body.as<SeqStmtNode>()) {
      size_t i = 0;
      while (i < seq->size() && IsLaunchThreadPlaceholder(seq->seq[i])) {
        placeholder_prefix.push_back(seq->seq[i]);
        ++i;
      }
      if (i < seq->size())
        root_realize_node = seq->seq[i].as<SBlockRealizeNode>();
    }
    if (!root_realize_node ||
        !IsDeviceMainBlock(root_realize_node->block.get()) ||
        !IsPersistentKernelRoot(root_realize_node->block)) {
      return std::nullopt;
    }

    auto wrap = [&](Stmt rebuilt) -> Stmt {
      if (placeholder_prefix.empty())
        return rebuilt;
      ffi::Array<Stmt> seq = placeholder_prefix;
      seq.push_back(std::move(rebuilt));
      return SeqStmt(seq);
    };

    ICHECK(logical_loop->thread_binding.value()->thread_tag == "blockIdx.x")
        << "AutoPersistent expects the T.PersistentKernel dimension to be "
           "blockIdx.x";
    ICHECK(is_zero(logical_loop->min))
        << "AutoPersistent only supports a zero-based T.PersistentKernel grid";
    ICHECK(logical_loop->HasTrivialStep())
        << "AutoPersistent does not support a non-unit launch step";

    SBlockRealize root_realize = GetRef<SBlockRealize>(root_realize_node);
    PersistentKernelConfig persistent_config =
        ReadPersistentKernelConfig(root_realize->block);
    SBlockRealize clean_root = StripPersistentKernelConfig(root_realize);

    int64_t core_count = persistent_config.core_count;

    DataType index_dtype = logical_loop->loop_var.dtype();
    ICHECK(index_dtype.is_int())
        << "AutoPersistent requires signed integer launch extents";
    PrimExpr logical_extent = logical_loop->extent;
    if (logical_extent.dtype() != index_dtype)
      logical_extent = cast(index_dtype, logical_extent);
    if (const int64_t *value = as_const_int(logical_extent)) {
      ICHECK_GT(*value, 0)
          << "AutoPersistent requires a positive launch extent";
    }
    logical_extent = analyzer_.Simplify(logical_extent);

    ffi::Array<Var> launch_vars;
    launch_vars.push_back(logical_loop->loop_var);

    PrimExpr configured_cores = make_const(index_dtype, core_count);
    bool fits_physical_grid;
    if (const int64_t *value = as_const_int(logical_extent)) {
      fits_physical_grid = *value <= core_count;
    } else {
      fits_physical_grid =
          analyzer_.CanProve(logical_extent <= configured_cores);
    }

    if (fits_physical_grid) {
      return RebuildFor(logical_loop, wrap(clean_root));
    }

    ICHECK(!RootMetadataUsesLaunchVars(root_realize, launch_vars))
        << "AutoPersistent found a T.PersistentKernel grid variable in "
           "tilelang_root metadata (for example an allocation shape). Logical "
           "grid variables may only be used in the executable kernel body. To "
           "migrate, make the root metadata grid-invariant and move "
           "grid-dependent expressions into the body, or use T.Kernel to keep "
           "the direct logical-grid launch";

    // Every rewritten grid uses the same wave representation. Special-casing
    // an equality proof here would only avoid a single-iteration loop when a
    // prior `logical_extent <= configured_cores` proof was inconclusive, while
    // making the rewrite depend on the form of the Analyzer query.
    PrimExpr physical_extent = configured_cores;
    Var physical_block("block_id", index_dtype);
    Var wave("block_wave", index_dtype);
    PrimExpr logical_id = wave * physical_extent + physical_block;

    ffi::Array<Stmt> iteration_sequence;
    const Var &old_var = launch_vars[0];
    PrimExpr logical_block_value = logical_id;
    if (logical_block_value.dtype() != old_var.dtype())
      logical_block_value = cast(old_var.dtype(), logical_block_value);
    // The logical launch loop is removed from the output, so its variable can
    // be rebound directly. Reusing it keeps the original kernel body shared
    // instead of traversing and rebuilding the body with Substitute.
    // Scalar `local`/`local.var` buffers (T.alloc_var without an explicit
    // initializer) are zero-initialized by the codegen at their declaration,
    // which after folding lives outside the wave loop and would run only once
    // per physical core, leaking state across logical tasks. Re-initialize
    // each such buffer to zero at the start of every task so the implicit and
    // explicit (init=0) forms agree.
    for (const Buffer &buffer : clean_root->block->alloc_buffers) {
      std::string scope = buffer.scope();
      if (buffer->shape.size() == 1 && is_one(buffer->shape[0]) &&
          (scope == "local" || scope == "local.var")) {
        PrimExpr zero = IntImm(DataType::Int(32), 0);
        iteration_sequence.push_back(
            BufferStore(buffer, make_zero(buffer->dtype), {zero}));
      }
    }
    iteration_sequence.push_back(Bind(old_var, logical_block_value));
    // T.Persistent has already expanded into an ordinary serial loop at this
    // point. It is intentionally left nested and receives no coordination or
    // deduplication from AutoPersistent; combining both APIs is supported as
    // ordinary loop composition but is not recommended.
    iteration_sequence.push_back(clean_root->block->body);
    Stmt iteration_body = SeqStmt(iteration_sequence);

    PrimExpr wave_extent =
        analyzer_.Simplify(ceildiv(logical_extent, physical_extent));
    if (!analyzer_.CanProveEqual(wave_extent * physical_extent,
                                 logical_extent)) {
      iteration_body = IfThenElse(logical_id < logical_extent, iteration_body);
    }
    ffi::Map<ffi::String, ffi::Any> wave_annotations =
        persistent_config.loop_annotations;
    iteration_body = For(wave, make_const(index_dtype, 0), wave_extent,
                         ForKind::kSerial, iteration_body,
                         /*thread_binding=*/std::nullopt,
                         /*annotations=*/wave_annotations);

    SBlock new_root = clean_root->block;
    new_root.CopyOnWrite()->body = iteration_body;
    Stmt transformed =
        SBlockRealize(clean_root->iter_values, clean_root->predicate, new_root,
                      clean_root->span);

    ffi::Map<ffi::String, ffi::Any> physical_annotations =
        logical_loop->annotations;
    IterVar physical_iter(
        Range::FromMinExtent(make_const(index_dtype, 0), physical_extent),
        physical_block, IterVarType::kThreadIndex, "blockIdx.x");
    return For(physical_block, make_const(index_dtype, 0), physical_extent,
               ForKind::kThreadBinding, wrap(std::move(transformed)),
               physical_iter, physical_annotations);
  }

  arith::Analyzer analyzer_;
};

} // namespace

tvm::transform::Pass AutoPersistent() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    return AutoPersistentRewriter().Rewrite(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AutoPersistent", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoPersistent", AutoPersistent);
}

} // namespace tl
} // namespace tvm
