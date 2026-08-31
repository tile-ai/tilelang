/*!
 * \file materialize_cpu_parallel_grid.cc
 * \brief Convert the annotated CPU grid loop nest to parallel loops.
 *
 * When the ``tl.cpu_parallel`` pass config is enabled, the CPU branch of
 * MaterializeKernelLaunch tags each grid (blockIdx) loop with the
 * ``tl.cpu_grid_dim`` annotation (value = grid dimension index). The
 * annotation rides through the whole pipeline without semantic effect —
 * every mid-pipeline conflict site (LayoutInference, LowerTileOp, the
 * vectorizer) only reacts to ForKind::kParallel, never to this annotation.
 * This pass runs at the tail of the CPU pipeline (after UnrollLoop /
 * LoopUnswitching / HoistIfThenElse, before AnnotateDeviceRegions), where
 * loop structure is final, and:
 *
 *  1. Finds the outermost contiguous chain of annotated grid loops.
 *  2. Gates on total trip count (``tl.cpu_parallel_min_trip``, default 0 —
 *     the master switch itself is the gate).
 *  3. Converts the chain to ForKind::kParallel:
 *     - target ``c``: every grid dim, so the C codegen can emit
 *       ``#pragma omp parallel for collapse(n)``;
 *     - target ``llvm``: a single dim — the first non-unit dim (TVM's LLVM
 *       codegen lowers kParallel to TVMBackendParallelLaunch and
 *       hard-rejects nesting).
 *  4. Sinks AllocBuffer statements whose uses all lie inside the parallel
 *     region into the parallel loop body, giving each worker a private copy
 *     (a C declaration inside the loop body / an alloca in the llvm closure
 *     is thread-private by construction). For the ``c`` target the sink
 *     level is the innermost parallelized loop so collapse(n) perfect
 *     nesting is preserved; uses spanning grid levels keep the allocation
 *     hoisted (with a warning) instead of risking a wrong sink position.
 *
 * Everything is opt-in: without the annotation (config off) this pass leaves
 * the IR untouched, and the pass is not even inserted into the pipeline.
 */

#include "op/builtin.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tvm::ffi;

namespace {

bool HasGridAnnotation(const ForNode *op) {
  return op->annotations.count(attr::kCPUGridDim) > 0;
}

Map<ffi::String, ffi::Any>
StripGridAnnotation(const Map<ffi::String, ffi::Any> &annotations) {
  Map<ffi::String, ffi::Any> out;
  for (const auto &kv : annotations) {
    if (kv.first != attr::kCPUGridDim) {
      out.Set(kv.first, kv.second);
    }
  }
  return out;
}

/*! \brief For each buffer (keyed by its data Var): the shallowest grid-nest
 * depth of any access (0 = accessed outside the nest) and whether it is
 * accessed inside the nest at all. */
class GridAccessAnalysis : public StmtExprVisitor {
public:
  struct AccessInfo {
    int min_depth = 0;
    bool inside = false;
  };

  AccessInfo Lookup(const Var &data) const {
    static const AccessInfo kNone{0, false};
    auto it = info_.find(data);
    return it == info_.end() ? kNone : it->second;
  }

private:
  void VisitStmt_(const ForNode *op) override {
    bool grid = HasGridAnnotation(op);
    if (grid)
      ++grid_depth_;
    StmtExprVisitor::VisitStmt_(op);
    if (grid)
      --grid_depth_;
  }
  void VisitStmt_(const BufferStoreNode *op) override {
    Touch(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) override {
    Touch(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void Touch(const Buffer &buffer) {
    AccessInfo &entry = info_[buffer->data];
    if (grid_depth_ == 0) {
      entry.min_depth = 0;
    } else {
      // First inside access records its depth; later ones take the minimum.
      entry.min_depth =
          entry.inside ? std::min(entry.min_depth, grid_depth_) : grid_depth_;
      entry.inside = true;
    }
  }

  std::unordered_map<Var, AccessInfo, ObjectPtrHash, ObjectPtrEqual> info_;
  int grid_depth_ = 0;
};

/*! \brief A missing For step is the implicit default of 1; materialize it so
 * downstream checks (e.g. TVM's llvm parallel-loop lowering) see a literal. */
PrimExpr NormalizedStep(const ForNode *op) {
  if (op->step.defined()) {
    return op->step.value();
  }
  return IntImm(op->loop_var.dtype(), 1);
}

/*! \brief True if the subtree contains a kParallel loop (the llvm backend
 * hard-rejects nested parallel loops). */
class ParallelLoopFinder : public StmtVisitor {
public:
  bool found = false;

private:
  void VisitStmt_(const ForNode *op) override {
    if (op->kind == ForKind::kParallel)
      found = true;
    StmtVisitor::VisitStmt_(op);
  }
};

/*! \brief True if the subtree still carries a grid annotation — i.e. a nest
 * this pass did not convert (wrapped in an unrecognized construct, or a
 * second T.Kernel launch nest). */
class GridAnnotationFinder : public StmtVisitor {
public:
  bool found = false;

private:
  void VisitStmt_(const ForNode *op) override {
    if (HasGridAnnotation(op))
      found = true;
    StmtVisitor::VisitStmt_(op);
  }
};

struct GridRewriter {
  GridRewriter(bool collapse_all_dims, int64_t min_trip,
               const GridAccessAnalysis &analysis)
      : collapse_all_dims_(collapse_all_dims), min_trip_(min_trip),
        analysis_(analysis) {}

  Stmt Rewrite(const Stmt &stmt) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> out;
      for (const Stmt &elem : seq->seq) {
        if (!converted_ && elem->IsInstance<ForNode>() &&
            HasGridAnnotation(elem.as<ForNode>())) {
          std::vector<Stmt> sunk =
              CollectSinkableAllocs(out, elem.as<ForNode>());
          out.push_back(ConvertGridNest(elem.as<ForNode>(), std::move(sunk)));
          continue;
        }
        out.push_back(Rewrite(elem));
      }
      if (out.size() == 1) {
        return out[0];
      }
      return SeqStmt(std::move(out));
    }
    if (!converted_ && stmt->IsInstance<ForNode>() &&
        HasGridAnnotation(stmt.as<ForNode>())) {
      return ConvertGridNest(stmt.as<ForNode>(), {});
    }
    return stmt;
  }

private:
  static std::vector<const ForNode *> CollectChain(const ForNode *head) {
    std::vector<const ForNode *> loops;
    const ForNode *cur = head;
    while (cur && HasGridAnnotation(cur)) {
      loops.push_back(cur);
      cur = cur->body.as<ForNode>();
    }
    return loops;
  }

  /*! \brief Index of the chain member whose body is the allocation sink
   * target: the innermost dim on ``c`` (keeps collapse(n) perfect nesting),
   * the single parallelized dim on ``llvm``. */
  static size_t SinkIndex(const std::vector<const ForNode *> &loops,
                          bool collapse_all_dims) {
    if (collapse_all_dims) {
      return loops.size() - 1;
    }
    for (size_t i = 0; i < loops.size(); ++i) {
      const auto *extent = loops[i]->extent.as<IntImmNode>();
      if (!extent || extent->value != 1) {
        return i;
      }
    }
    return 0;
  }

  /*! \brief Among the already-emitted sequence elements, decide which
   * function-scope allocations sink into the parallel region, remove them
   * from `out` and return them. */
  std::vector<Stmt> CollectSinkableAllocs(Array<Stmt> &out,
                                          const ForNode *grid_head) {
    if (out.empty())
      return {};

    int sink_depth = static_cast<int>(SinkIndex(CollectChain(grid_head),
                                                collapse_all_dims_)) +
                     1;
    std::vector<Stmt> sunk;
    Array<Stmt> kept;
    for (const Stmt &elem : out) {
      const auto *alloc = elem.as<AllocBufferNode>();
      if (!alloc) {
        kept.push_back(elem);
        continue;
      }
      GridAccessAnalysis::AccessInfo info =
          analysis_.Lookup(alloc->buffer->data);
      if (info.min_depth >= sink_depth) {
        // All uses are deep enough: private per-worker copy is safe and
        // collapse-perfect.
        sunk.push_back(elem);
      } else if (info.min_depth >= 1) {
        LOG(WARNING) << "tl.cpu_parallel: buffer `" << alloc->buffer->name
                     << "` is used across grid levels; it stays shared "
                        "across workers and may race";
        kept.push_back(elem);
      } else if (info.inside) {
        LOG(WARNING) << "tl.cpu_parallel: buffer `" << alloc->buffer->name
                     << "` is used both inside and outside the parallel "
                        "region; it stays shared across workers and may race";
        kept.push_back(elem);
      } else {
        // Only used outside the nest (or dead): nothing to do.
        kept.push_back(elem);
      }
    }
    out = std::move(kept);
    return sunk;
  }

  /*! \brief Rewrite the outermost annotated grid nest `head`. */
  Stmt ConvertGridNest(const ForNode *head, std::vector<Stmt> sunk) {
    converted_ = true;

    std::vector<const ForNode *> loops = CollectChain(head);

    // Rebuild with annotations stripped; failure paths return this with the
    // sunk allocations re-attached in front of the nest (they were removed
    // from their original position by the caller already).
    auto rebuild_serial = [&loops, &sunk]() -> Stmt {
      Stmt body = loops.back()->body;
      for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
        const ForNode *op = loops[i];
        PrimExpr step = NormalizedStep(op);
        body = For(op->loop_var, op->min, op->extent, ForKind::kSerial,
                   std::move(body), std::nullopt,
                   StripGridAnnotation(op->annotations), std::move(step),
                   op->span);
      }
      if (!sunk.empty()) {
        Array<Stmt> elements(sunk.begin(), sunk.end());
        elements.push_back(body);
        return SeqStmt(std::move(elements));
      }
      return body;
    };

    // Total trip count must be statically known and above the threshold.
    int64_t trip = 1;
    for (const ForNode *op : loops) {
      const auto *extent = op->extent.as<IntImmNode>();
      if (!extent || extent->value <= 0)
        return rebuild_serial();
      trip *= extent->value;
    }
    if (trip < min_trip_)
      return rebuild_serial();

    // Which chain members become kParallel: all of them on ``c`` (for
    // collapse(n)); exactly one — the first non-unit dim — on ``llvm``.
    size_t parallel_idx = SinkIndex(loops, collapse_all_dims_);

    if (!collapse_all_dims_) {
      // TVM's llvm codegen requires min=0 / step=1 on the parallel loop.
      // A missing step is the implicit default of 1 and is normalized to a
      // literal during the rebuild below.
      const ForNode *marked = loops[parallel_idx];
      const auto *min = marked->min.as<IntImmNode>();
      ICHECK(min && min->value == 0)
          << "tl.cpu_parallel: the parallelized grid loop must start at 0 on "
             "the llvm target, got min="
          << marked->min;
      if (marked->step.defined()) {
        const auto *step = marked->step.as<IntImmNode>();
        ICHECK(step && step->value == 1)
            << "tl.cpu_parallel: the parallelized grid loop must have step 1 "
               "on the llvm target, got step="
            << marked->step;
      }
    }

    // Sunk allocations become per-worker private copies at the head of the
    // sink loop's body (c: innermost grid dim, keeps collapse(n) perfect
    // nesting; llvm: the single parallel dim).
    Stmt body = loops[parallel_idx]->body;
    if (!sunk.empty()) {
      Array<Stmt> elements(sunk.begin(), sunk.end());
      elements.push_back(body);
      body = SeqStmt(std::move(elements));
    }
    for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
      const ForNode *op = loops[i];
      ForKind kind =
          (collapse_all_dims_ || static_cast<size_t>(i) == parallel_idx)
              ? ForKind::kParallel
              : ForKind::kSerial;
      Map<ffi::String, ffi::Any> annotations =
          StripGridAnnotation(op->annotations);
      PrimExpr step = NormalizedStep(op);
      body =
          For(op->loop_var, op->min, op->extent, kind, std::move(body),
              std::nullopt, std::move(annotations), std::move(step), op->span);
    }

    if (!collapse_all_dims_) {
      ParallelLoopFinder finder;
      finder(loops[parallel_idx]->body);
      ICHECK(!finder.found) << "tl.cpu_parallel: nested parallel loops are "
                               "not supported on the llvm target";
    }

    return body;
  }

  bool collapse_all_dims_;
  int64_t min_trip_;
  const GridAccessAnalysis &analysis_;
  bool converted_ = false;
};

} // namespace

namespace transform {

using namespace tirx::transform;

tvm::transform::Pass MaterializeCPUParallelGrid() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    auto opt_target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target)
      return func;
    std::string kind = opt_target.value()->kind->name;
    bool collapse_all_dims;
    if (kind == "c") {
      collapse_all_dims = true;
    } else if (kind == "llvm") {
      collapse_all_dims = false;
    } else {
      return func;
    }

    GridAccessAnalysis analysis;
    analysis(func->body);

    int64_t min_trip = ctx->GetConfig<IntImm>(kCPUParallelMinTrip,
                                              IntImm(DataType::Int(64), 0))
                           .value()
                           ->value;

    GridRewriter rewriter(collapse_all_dims, min_trip, analysis);
    Stmt new_body = rewriter.Rewrite(func->body);
    GridAnnotationFinder residual;
    residual(new_body);
    if (residual.found) {
      // A nest this pass did not convert: wrapped in an unrecognized
      // construct, or a second launch nest. Surface the silent serial
      // fallback instead of leaving the user wondering why nothing sped up.
      LOG(WARNING) << "tl.cpu_parallel: a grid nest could not be converted "
                      "(wrapped in an unrecognized construct, or not the "
                      "first launch nest); it stays serial";
    }
    func.CopyOnWrite()->body = std::move(new_body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MaterializeCPUParallelGrid", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cpu.transform.MaterializeCPUParallelGrid",
                        MaterializeCPUParallelGrid);
}

} // namespace transform

} // namespace tl
} // namespace tvm
