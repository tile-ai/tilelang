/*!
 * \file materialize_cpu_parallel_grid.cc
 * \brief Convert the annotated CPU grid loop nests to parallel loops.
 *
 * When the ``tl.cpu_parallel`` pass config is enabled, MaterializeKernelLaunch
 * tags each grid (blockIdx) loop with the ``tl.cpu_grid_dim`` annotation. It
 * rides through the pipeline inertly — the mid-pipeline conflict sites
 * (LayoutInference, LowerTileOp, the vectorizer) only react to
 * ForKind::kParallel — until this tail pass (after HoistIfThenElse, before
 * AnnotateDeviceRegions, where loop structure is final):
 *
 *  1. Finds every annotated nest, looking through transparent wrappers
 *     (assume AttrStmt, hoisted IfThenElse). Sibling nests convert
 *     independently; a nest inside an already-parallel one stays serial.
 *  2. Gates on total trip count (``tl.cpu_parallel_min_trip``, default 0).
 *     Dynamic extents skip the gate: both OpenMP and TVM's parallel launch
 *     handle runtime trip counts.
 *  3. Converts the chain to kParallel: every dim on ``c`` (for
 *     ``collapse(n)``), the first non-unit dim on ``llvm`` (its codegen
 *     rejects nested parallel loops).
 *  4. Sinks AllocBuffers into the parallel body (per-worker private copies;
 *     on ``c`` at the innermost parallelized dim to keep collapse(n) perfect
 *     nesting). Only allocs whose uses are all plain load/store inside this
 *     nest sink; load-only buffers (e.g. a table initialized before the
 *     nest) may stay shared. A nest whose buffer is mutated inside but
 *     cannot be privatized (opaque/cross-level/cross-nest/outside uses) is
 *     refused with a warning rather than raced.
 *
 * Opt-in only: without the annotation this pass is not even in the pipeline.
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
#include <unordered_set>
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

/*! \brief Per-buffer access census: shallowest in-nest depth of plain
 * loads/stores (0 = accessed outside every nest), owning annotated nest(s),
 * and an opaque-use flag (data var in a Call argument — provably not
 * iteration-private, pins the buffer shared). */
class GridAccessAnalysis : public StmtExprVisitor {
public:
  struct AccessInfo {
    int min_depth = 0;
    bool inside = false;        // plain load/store inside a nest
    bool outside = false;       // plain load/store outside every nest
    bool opaque = false;        // data var in a Call argument, anywhere
    bool opaque_inside = false; // opaque use inside a nest
    bool store_inside = false;  // plain store inside a nest
    std::unordered_set<const ForNode *> nests;
  };

  AccessInfo Lookup(const Var &data) const {
    static const AccessInfo kNone{};
    auto it = info_.find(data);
    return it == info_.end() ? kNone : it->second;
  }

  /*! \brief AllocBuffer data vars with grid-nest depth at declaration
   * (0 = function scope; only those may sink). */
  const std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> &
  AllocDepths() const {
    return alloc_depths_;
  }

private:
  void VisitStmt_(const ForNode *op) override {
    bool grid = HasGridAnnotation(op);
    if (grid) {
      if (grid_depth_ == 0) {
        current_nest_ = op; // chain head of this annotated nest
      }
      ++grid_depth_;
    }
    StmtExprVisitor::VisitStmt_(op);
    if (grid) {
      --grid_depth_;
      if (grid_depth_ == 0) {
        current_nest_ = nullptr;
      }
    }
  }
  void VisitStmt_(const AllocBufferNode *op) override {
    // A declaration is not a use.
    known_alloc_vars_.insert(op->buffer->data.get());
    alloc_depths_[op->buffer->data] = grid_depth_;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) override {
    if (grid_depth_ > 0) {
      info_[op->buffer->data].store_inside = true;
    }
    Touch(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) override {
    Touch(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const VarNode *op) override {
    if (known_alloc_vars_.count(op)) {
      AccessInfo &entry = info_[GetRef<Var>(op)];
      entry.opaque = true;
      entry.min_depth = 0;
      if (grid_depth_ > 0) {
        entry.opaque_inside = true;
        entry.inside = true;
        entry.nests.insert(current_nest_);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void Touch(const Buffer &buffer) {
    AccessInfo &entry = info_[buffer->data];
    if (grid_depth_ == 0) {
      entry.min_depth = 0;
      entry.outside = true;
    } else {
      entry.min_depth =
          entry.inside ? std::min(entry.min_depth, grid_depth_) : grid_depth_;
      entry.inside = true;
      entry.nests.insert(current_nest_);
    }
  }

  std::unordered_map<Var, AccessInfo, ObjectPtrHash, ObjectPtrEqual> info_;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> alloc_depths_;
  std::unordered_set<const VarNode *> known_alloc_vars_;
  const ForNode *current_nest_ = nullptr;
  int grid_depth_ = 0;
};

/*! \brief A missing For step is the implicit default of 1; materialize it so
 * downstream checks (e.g. llvm's parallel lowering) see a literal. */
PrimExpr NormalizedStep(const ForNode *op) {
  if (op->step.defined()) {
    return op->step.value();
  }
  return IntImm(op->loop_var.dtype(), 1);
}

//! True if the subtree contains a kParallel loop.
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

//! True if the subtree still carries a grid annotation (unconverted nest).
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

  Stmt Rewrite(const Stmt &stmt) { return Rewrite(stmt, false); }

private:
  //! The annotated chain head reachable through transparent wrappers, if any.
  static const ForNode *TransparentNestHead(const Stmt &stmt) {
    if (const auto *f = stmt.as<ForNode>()) {
      return HasGridAnnotation(f) ? f : nullptr;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return TransparentNestHead(attr->body);
    }
    return nullptr;
  }

  Stmt Rewrite(const Stmt &stmt, bool in_parallel) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> out;
      for (const Stmt &elem : seq->seq) {
        const ForNode *head = in_parallel ? nullptr : TransparentNestHead(elem);
        if (head != nullptr) {
          // Collect sibling allocations here even for a wrapped nest.
          std::vector<Stmt> sunk = CollectSinkableAllocs(out, head);
          out.push_back(ConvertNestElement(elem, head, std::move(sunk)));
          continue;
        }
        out.push_back(Rewrite(elem, in_parallel));
      }
      if (out.size() == 1) {
        return out[0];
      }
      return SeqStmt(std::move(out));
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      Stmt body = Rewrite(attr->body, in_parallel);
      if (body.same_as(attr->body)) {
        return stmt;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, std::move(body),
                      attr->span);
    }
    if (const auto *ite = stmt.as<IfThenElseNode>()) {
      Stmt then_case = Rewrite(ite->then_case, in_parallel);
      Optional<Stmt> else_case = ite->else_case;
      if (else_case.defined()) {
        Stmt new_else = Rewrite(else_case.value(), in_parallel);
        if (!new_else.same_as(else_case)) {
          else_case = new_else;
        }
      }
      if (then_case.same_as(ite->then_case) &&
          else_case.same_as(ite->else_case)) {
        return stmt;
      }
      return IfThenElse(ite->condition, std::move(then_case),
                        std::move(else_case), ite->span);
    }
    if (!in_parallel && stmt->IsInstance<ForNode>() &&
        HasGridAnnotation(stmt.as<ForNode>())) {
      return ConvertGridNest(stmt.as<ForNode>(), {});
    }
    // An annotated nest inside an already-parallel region stays serial; its
    // annotation is kept so the residual-annotation warning fires.
    return stmt;
  }

  //! Convert the nest headed by `head`, rebuilding transparent wrappers.
  Stmt ConvertNestElement(const Stmt &elem, const ForNode *head,
                          std::vector<Stmt> sunk) {
    if (const auto *attr = elem.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value,
                      ConvertNestElement(attr->body, head, std::move(sunk)),
                      attr->span);
    }
    return ConvertGridNest(head, std::move(sunk));
  }

  static std::vector<const ForNode *> CollectChain(const ForNode *head) {
    std::vector<const ForNode *> loops;
    const ForNode *cur = head;
    while (cur && HasGridAnnotation(cur)) {
      loops.push_back(cur);
      cur = cur->body.as<ForNode>();
    }
    return loops;
  }

  /*! \brief Sink target within the chain: the innermost dim on ``c`` (keeps
   * collapse(n) perfect nesting), the single parallelized dim on ``llvm``. */
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

  //! True when every use is a plain load/store inside the `head` nest.
  static bool FullyOwnedBy(const GridAccessAnalysis::AccessInfo &info,
                           const ForNode *head) {
    return !info.opaque && !info.outside && info.inside &&
           info.nests.size() == 1 && *info.nests.begin() == head;
  }

  //! True when the buffer may be sunk into the `head` nest.
  static bool SinkableAlloc(const GridAccessAnalysis::AccessInfo &info,
                            const ForNode *head, int sink_depth) {
    return FullyOwnedBy(info, head) && info.min_depth >= sink_depth;
  }

  //! Move sinkable allocations out of `out` and return them; kept buffers
  //! are reconsidered when their own nest converts.
  std::vector<Stmt> CollectSinkableAllocs(Array<Stmt> &out,
                                          const ForNode *grid_head) {
    if (out.empty()) {
      return {};
    }

    int sink_depth = static_cast<int>(SinkIndex(CollectChain(grid_head),
                                                collapse_all_dims_)) +
                     1;
    std::vector<Stmt> sunk;
    Array<Stmt> kept;
    for (const Stmt &elem : out) {
      const auto *alloc = elem.as<AllocBufferNode>();
      if (alloc != nullptr &&
          SinkableAlloc(analysis_.Lookup(alloc->buffer->data), grid_head,
                        sink_depth)) {
        sunk.push_back(elem);
      } else {
        kept.push_back(elem);
      }
    }
    out = std::move(kept);
    return sunk;
  }

  //! Rewrite the outermost annotated grid nest `head`.
  Stmt ConvertGridNest(const ForNode *head, std::vector<Stmt> sunk) {
    std::vector<const ForNode *> loops = CollectChain(head);
    int sink_depth = static_cast<int>(SinkIndex(loops, collapse_all_dims_)) + 1;

    // Failure paths: rebuild serial, annotations stripped, sunk allocations
    // re-attached in front of the nest.
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

    int64_t trip = 1;
    bool dynamic_extents = false;
    for (const ForNode *op : loops) {
      const auto *extent = op->extent.as<IntImmNode>();
      if (!extent) {
        dynamic_extents = true;
        break;
      }
      if (extent->value <= 0) {
        return rebuild_serial();
      }
      trip *= extent->value;
    }
    if (dynamic_extents && min_trip_ > 0) {
      LOG(WARNING) << "tl.cpu_parallel: cannot evaluate "
                      "tl.cpu_parallel_min_trip against a dynamic grid "
                      "extent; the nest stays serial";
      return rebuild_serial();
    }
    if (!dynamic_extents && trip < min_trip_) {
      return rebuild_serial();
    }

    size_t parallel_idx = SinkIndex(loops, collapse_all_dims_);

    if (!collapse_all_dims_) {
      // The llvm parallel launch requires min=0 / step=1 and no nesting;
      // fall back to serial instead of failing the compile.
      const ForNode *marked = loops[parallel_idx];
      const auto *min = marked->min.as<IntImmNode>();
      if (!min || min->value != 0) {
        LOG(WARNING) << "tl.cpu_parallel: the grid loop does not start at 0, "
                        "which the llvm parallel launch requires; the nest "
                        "stays serial";
        return rebuild_serial();
      }
      if (marked->step.defined()) {
        const auto *step = marked->step.as<IntImmNode>();
        if (!step || step->value != 1) {
          LOG(WARNING) << "tl.cpu_parallel: the grid loop has a non-unit "
                          "step, which the llvm parallel launch requires to "
                          "be 1; the nest stays serial";
          return rebuild_serial();
        }
      }
      ParallelLoopFinder finder;
      finder(loops[parallel_idx]->body);
      if (finder.found) {
        LOG(WARNING) << "tl.cpu_parallel: the grid nest already contains a "
                        "parallel loop, which the llvm backend rejects; the "
                        "nest stays serial";
        return rebuild_serial();
      }
    }

    // Refuse to parallelize when a function-scope buffer is mutated (or
    // opaquely used) inside the nest but cannot be privatized into it —
    // running it shared across workers would be a data race. Load-only
    // sharing (e.g. a table initialized before the nest) is race-free.
    for (const auto &kv : analysis_.AllocDepths()) {
      const Var &data = kv.first;
      if (kv.second != 0) {
        continue;
      }
      GridAccessAnalysis::AccessInfo info = analysis_.Lookup(data);
      if (!info.nests.count(head)) {
        continue; // not used inside this nest
      }
      if (SinkableAlloc(info, head, sink_depth)) {
        bool was_sunk = false;
        for (const Stmt &s : sunk) {
          if (s.as<AllocBufferNode>() &&
              s.as<AllocBufferNode>()->buffer->data.same_as(data)) {
            was_sunk = true;
            break;
          }
        }
        if (!was_sunk) {
          LOG(WARNING) << "tl.cpu_parallel: buffer `" << data->name_hint
                       << "` is used only inside the grid nest but its "
                          "allocation is not adjacent to it, so it cannot be "
                          "privatized; the nest stays serial";
          return rebuild_serial();
        }
        continue;
      }
      if (info.store_inside || info.opaque_inside) {
        const char *reason =
            info.opaque_inside
                ? "it is referenced through opaque accesses (call_extern / "
                  "address_of / access_ptr)"
            : info.outside          ? "it is also used outside the nest"
            : info.nests.size() > 1 ? "it is shared by multiple grid nests"
                                    : "it is used across grid levels";
        LOG(WARNING) << "tl.cpu_parallel: buffer `" << data->name_hint
                     << "` is mutated inside the grid nest but cannot be "
                        "privatized ("
                     << reason << "); the nest stays serial";
        return rebuild_serial();
      }
    }

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

    return body;
  }

  bool collapse_all_dims_;
  int64_t min_trip_;
  const GridAccessAnalysis &analysis_;
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
      LOG(WARNING) << "tl.cpu_parallel: a grid nest could not be converted "
                      "(nested inside another parallelized nest, or wrapped "
                      "in an unrecognized construct); it stays serial";
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
