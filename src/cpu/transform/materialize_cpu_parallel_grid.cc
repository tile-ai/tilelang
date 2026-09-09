/*!
 * \file materialize_cpu_parallel_grid.cc
 * \brief Convert the annotated CPU grid loop nests to parallel loops.
 *
 * When the ``tl.cpu_parallel`` pass config is enabled, MaterializeKernelLaunch
 * tags each grid (blockIdx) loop with the ``tl.cpu_grid_dim`` annotation. The
 * annotation rides through the pipeline inertly (mid-pipeline passes only
 * react to ForKind::kParallel) until this tail pass, where loop structure is
 * final. For every annotated nest it then:
 *
 *  1. Gates on total trip count (``tl.cpu_parallel_min_trip``, default 0;
 *     dynamic extents skip the gate — both OpenMP and TVM's parallel launch
 *     handle runtime trip counts).
 *  2. Converts the chain to kParallel: every dim on ``c`` (for
 *     ``collapse(n)``), the first non-unit dim on ``llvm`` (its codegen
 *     rejects nested parallel loops).
 *  3. Sinks AllocBuffers into the parallel body (per-worker private copies;
 *     on ``c`` at the innermost parallelized dim to keep collapse(n) perfect
 *     nesting). Only allocs whose uses are all plain load/store inside this
 *     nest sink; load-only buffers may stay shared. A nest whose buffer is
 *     mutated inside but cannot be privatized (opaque/cross-level/
 *     cross-nest/outside uses) is refused with a warning rather than raced.
 *
 * Sibling nests convert independently; a nest inside an already-parallel one
 * stays serial. Invariant: no ``tl.cpu_grid_dim`` survives this pass — every
 * nest is either converted or stripped with an in-place warning naming the
 * loop and the reason. Opt-in only: without the annotation this pass is not
 * even in the pipeline.
 */

#include "op/builtin.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
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
 * loads/stores, owning annotated nest(s), and outside/opaque/store flags. */
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

  //! AllocBuffer data vars with grid-nest depth at declaration (0 = function
  //! scope; only those may sink).
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
      TouchOpaque(GetRef<Var>(op));
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const CallNode *op) override {
    // address_of wraps its BufferLoad argument, hiding the buffer from the
    // bare-Var check above.
    if (op->op.same_as(builtin::address_of()) && !op->args.empty()) {
      if (const auto *load = op->args[0].as<BufferLoadNode>();
          load && known_alloc_vars_.count(load->buffer->data.get())) {
        TouchOpaque(load->buffer->data);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void TouchOpaque(const Var &data) {
    AccessInfo &entry = info_[data];
    entry.opaque = true;
    entry.min_depth = 0;
    if (grid_depth_ > 0) {
      entry.opaque_inside = true;
      entry.inside = true;
      entry.nests.insert(current_nest_);
    }
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

//! A missing For step is the implicit default of 1; materialize the literal.
PrimExpr NormalizedStep(const ForNode *op) {
  if (op->step.defined()) {
    return op->step.value();
  }
  return IntImm(op->loop_var.dtype(), 1);
}

struct GridRewriter : public StmtMutator {
  GridRewriter(bool collapse_all_dims, int64_t min_trip,
               const GridAccessAnalysis &analysis)
      : collapse_all_dims_(collapse_all_dims), min_trip_(min_trip),
        analysis_(analysis) {}

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

  // SeqStmt is the only place where sibling allocations can be collected
  // for sinking; everything else uses the mutator's default recursion.
  Stmt VisitStmt_(const SeqStmtNode *op) override {
    Array<Stmt> out;
    for (const Stmt &elem : op->seq) {
      const ForNode *head = in_parallel_ ? nullptr : TransparentNestHead(elem);
      if (head != nullptr) {
        std::vector<Stmt> sunk = CollectSinkableAllocs(out, head);
        out.push_back(ConvertNestElement(elem, head, std::move(sunk)));
        continue;
      }
      out.push_back(VisitStmt(elem));
    }
    if (out.size() == 1) {
      return out[0];
    }
    return SeqStmt(std::move(out));
  }

  Stmt VisitStmt_(const ForNode *op) override {
    if (!HasGridAnnotation(op)) {
      return StmtMutator::VisitStmt_(op);
    }
    if (in_parallel_) {
      // Nested inside an already-parallel region: keep serial, report in
      // place, strip the annotation (deeper nests are handled likewise).
      LOG(WARNING) << "tl.cpu_parallel: grid loop `" << op->loop_var->name_hint
                   << "` is nested inside another parallelized nest; it "
                      "stays serial";
      return For(op->loop_var, op->min, op->extent, ForKind::kSerial,
                 VisitStmt(op->body), std::nullopt,
                 StripGridAnnotation(op->annotations), NormalizedStep(op),
                 op->span);
    }
    auto [result, converted] = ConvertGridNest(op, {});
    // Whether or not this nest converted, its body may hide further
    // annotated nests: if this one is now parallel they must stay serial,
    // otherwise they convert independently.
    bool was_in_parallel = in_parallel_;
    in_parallel_ = converted;
    Stmt out = VisitStmt(result);
    in_parallel_ = was_in_parallel;
    return out;
  }

  //! Convert the nest headed by `head`, rebuilding transparent wrappers.
  Stmt ConvertNestElement(const Stmt &elem, const ForNode *head,
                          std::vector<Stmt> sunk) {
    if (const auto *attr = elem.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value,
                      ConvertNestElement(attr->body, head, std::move(sunk)),
                      attr->span);
    }
    auto [result, converted] = ConvertGridNest(head, std::move(sunk));
    bool was_in_parallel = in_parallel_;
    in_parallel_ = converted;
    Stmt out = VisitStmt(result);
    in_parallel_ = was_in_parallel;
    return out;
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

  //! Move sinkable allocations out of `out` and return them.
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

  /*! \brief Rewrite the outermost annotated grid nest `head`; the bool
   * reports whether it was converted to parallel. */
  std::pair<Stmt, bool> ConvertGridNest(const ForNode *head,
                                        std::vector<Stmt> sunk) {
    std::vector<const ForNode *> loops = CollectChain(head);
    int sink_depth = static_cast<int>(SinkIndex(loops, collapse_all_dims_)) + 1;

    // Failure paths: rebuild serial with annotations stripped.
    auto rebuild_serial = [&loops, &sunk]() -> std::pair<Stmt, bool> {
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
        return {SeqStmt(std::move(elements)), false};
      }
      return {body, false};
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
                      "extent; grid loop `"
                   << head->loop_var->name_hint << "` stays serial";
      return rebuild_serial();
    }
    if (!dynamic_extents && trip < min_trip_) {
      return rebuild_serial();
    }

    size_t parallel_idx = SinkIndex(loops, collapse_all_dims_);

    if (!collapse_all_dims_) {
      // The llvm parallel launch requires min=0 / step=1 and no nesting.
      const ForNode *marked = loops[parallel_idx];
      const auto *min = marked->min.as<IntImmNode>();
      if (!min || min->value != 0) {
        LOG(WARNING) << "tl.cpu_parallel: grid loop `"
                     << marked->loop_var->name_hint
                     << "` does not start at 0, which the llvm parallel "
                        "launch requires; it stays serial";
        return rebuild_serial();
      }
      if (marked->step.defined()) {
        const auto *step = marked->step.as<IntImmNode>();
        if (!step || step->value != 1) {
          LOG(WARNING) << "tl.cpu_parallel: grid loop `"
                       << marked->loop_var->name_hint
                       << "` has a non-unit step, which the llvm parallel "
                          "launch requires to be 1; it stays serial";
          return rebuild_serial();
        }
      }
      bool nested_parallel = false;
      PostOrderVisit(loops[parallel_idx]->body, [&](const ObjectRef &node) {
        if (const auto *f = node.as<ForNode>()) {
          nested_parallel |= f->kind == ForKind::kParallel;
        }
      });
      if (nested_parallel) {
        LOG(WARNING) << "tl.cpu_parallel: the nest of grid loop `"
                     << head->loop_var->name_hint
                     << "` already contains a parallel loop, which the llvm "
                        "backend rejects; it stays serial";
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
                       << "` is used only inside the nest of grid loop `"
                       << head->loop_var->name_hint
                       << "` but its allocation is not adjacent to it, so it "
                          "cannot be privatized; the nest stays serial";
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
                     << "` is mutated inside the nest of grid loop `"
                     << head->loop_var->name_hint
                     << "` but cannot be privatized (" << reason
                     << "); the nest stays serial";
        return rebuild_serial();
      }
    }

    // Rebuild from the true innermost body so loops below the parallelized
    // dim are not duplicated; splice the sunk allocations when the wrap
    // reaches the parallel dim.
    Stmt body = loops.back()->body;
    for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
      if (static_cast<size_t>(i) == parallel_idx && !sunk.empty()) {
        Array<Stmt> elements(sunk.begin(), sunk.end());
        elements.push_back(body);
        body = SeqStmt(std::move(elements));
      }
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

    return {body, true};
  }

  bool collapse_all_dims_;
  int64_t min_trip_;
  const GridAccessAnalysis &analysis_;
  // True while visiting the body of a successfully parallelized nest:
  // annotated nests found there must stay serial.
  bool in_parallel_ = false;
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
    func.CopyOnWrite()->body = rewriter(func->body);
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
