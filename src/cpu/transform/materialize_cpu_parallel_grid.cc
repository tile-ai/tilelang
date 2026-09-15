/*!
 * \file materialize_cpu_parallel_grid.cc
 * \brief Convert the annotated CPU grid loop nests to parallel loops.
 *
 * MaterializeKernelLaunch tags each grid (blockIdx) loop with
 * ``tl.cpu_grid_dim`` when ``tl.cpu_parallel`` is enabled; the annotation
 * rides the pipeline inertly until this tail pass, where loop structure is
 * final. For every annotated nest this pass:
 *
 *  1. Gates on total trip count (``tl.cpu_parallel_min_trip``, default 0;
 *     dynamic extents skip the gate).
 *  2. Converts the chain to kParallel: every dim on ``c`` (for
 *     ``collapse(n)``), the first non-unit dim on ``llvm`` (its codegen
 *     rejects nested parallel loops).
 *  3. Sinks AllocBuffers into the parallel body (per-worker private copies)
 *     when every use is a plain load/store inside the nest. A buffer mutated
 *     inside but not privatizable refuses the nest with a warning.
 *
 * Invariant: no ``tl.cpu_grid_dim`` survives this pass — every nest is
 * converted or stripped with a warning naming the loop and the reason.
 */

#include "op/builtin.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
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

  //! True for data vars declared by an AllocBuffer anywhere (parameter and
  //! global buffers have no declaration).
  bool HasAllocation(const Var &data) const {
    return alloc_depths_.count(data) > 0;
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
  return op->step.value_or(IntImm(op->loop_var.dtype(), 1));
}

/*! \brief Scalarize a possibly-vectorized flat index for affine analysis:
 * a Ramp gets a synthetic lane variable, anything else is used as-is. */
PrimExpr ScalarizedIndex(const PrimExpr &index, ffi::Map<Var, Range> *ranges) {
  if (const auto *ramp = index.as<RampNode>()) {
    Var lane("affine_lane", DataType::Int(32));
    ranges->Set(
        lane, Range::FromMinExtent(IntImm(DataType::Int(32), 0), ramp->lanes));
    return ramp->base + lane * ramp->stride;
  }
  return index;
}

/*! \brief Replace every integer cast of a variable by a fresh variable of
 * the cast's dtype (the expression stays well-typed; DetectIterMap rejects
 * Cast nodes). The fresh variable inherits the original variable's range. */
class CastVarFreshener : public StmtExprMutator {
public:
  const std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> &
  FreshVars() const {
    return fresh_;
  }

  PrimExpr VisitExpr_(const CastNode *op) override {
    if (const auto *var = op->value.as<VarNode>();
        var && op->dtype.is_int() && var->dtype.is_int()) {
      Var orig = GetRef<Var>(var);
      auto it = fresh_.find(orig);
      if (it == fresh_.end()) {
        it = fresh_
                 .emplace(orig, Var(orig->name_hint + "_i" +
                                        std::to_string(op->dtype.bits()),
                                    op->dtype))
                 .first;
      }
      return it->second;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> fresh_;
};

/*! \brief Prove the flat index is injective over the given variable scopes
 * (affine IterMap analysis at the bijective level). */
bool ProveInjectiveIndex(const PrimExpr &index,
                         const ffi::Map<Var, Range> &ranges) {
  arith::Analyzer analyzer;
  CastVarFreshener freshener;
  PrimExpr normalized = analyzer.Simplify(freshener(index));
  ffi::Map<Var, Range> norm_ranges;
  for (const auto &[var, range] : ranges) {
    auto it = freshener.FreshVars().find(var);
    // Use the fresh cast-typed var *instead of* the original: DetectIterMap
    // treats every range entry as an input iterator, so leaving both in
    // would make the map trivially non-bijective.
    norm_ranges.Set(it != freshener.FreshVars().end() ? it->second : var,
                    range);
  }
  auto res = arith::DetectIterMap({normalized}, norm_ranges, /*predicate=*/1,
                                  arith::IterMapLevel::Bijective, &analyzer);
  return res->errors.empty();
}

bool StoreReadsSameBuffer(const BufferStoreNode *op) {
  bool reads = false;
  PostOrderVisit(op->value, [&](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>();
        load && load->buffer->data.same_as(op->buffer->data)) {
      reads = true;
    }
  });
  return reads;
}

/*! \brief Prove that a store rewrites the whole buffer once per execution of
 * some non-empty enclosing loop suffix: constant extents, unit steps,
 * trip×lanes == numel, and an injective flat index. Partial writes, dynamic
 * or zero-trip loops, and non-affine indices do not count as a reset. Outer
 * loops may repeat the full rewrite (e.g. per-stage shared-buffer copies). */
bool StoreCoversWholeBuffer(const BufferStoreNode *op,
                            const std::vector<const ForNode *> &enclosing) {
  int64_t numel = 1;
  for (const PrimExpr &dim : op->buffer->shape) {
    const auto *imm = dim.as<IntImmNode>();
    if (!imm || imm->value <= 0) {
      return false;
    }
    numel *= imm->value;
  }
  if (op->indices.size() != 1) {
    return false;
  }
  PrimExpr index = op->indices[0];
  int64_t lanes = 1;
  if (const auto *ramp = index.as<RampNode>()) {
    const int64_t *l = as_const_int(ramp->lanes);
    if (l == nullptr || *l <= 0) {
      return false;
    }
    lanes = *l;
  } else if (index.as<BroadcastNode>()) {
    return false;
  }

  if (enclosing.empty()) {
    // The store always executes; it is a reset only for a scalar buffer.
    return numel == 1;
  }

  // Any non-empty suffix of the enclosing nest may be the level that fully
  // rewrites the buffer; check innermost-outward.
  for (size_t start = 0; start < enclosing.size(); ++start) {
    ffi::Map<Var, Range> ranges;
    int64_t trip = 1;
    bool ok = true;
    for (size_t i = start; i < enclosing.size(); ++i) {
      const ForNode *loop = enclosing[i];
      const auto *extent = loop->extent.as<IntImmNode>();
      if (!extent || extent->value <= 0 || !is_zero(loop->min) ||
          !is_one(NormalizedStep(loop))) {
        ok = false;
        break;
      }
      ranges.Set(loop->loop_var, Range::FromMinExtent(
                                     IntImm(DataType::Int(32), 0),
                                     IntImm(DataType::Int(32), extent->value)));
      trip *= extent->value;
    }
    if (!ok || trip * lanes != numel) {
      continue;
    }
    if (ProveInjectiveIndex(ScalarizedIndex(index, &ranges), ranges)) {
      return true;
    }
  }
  return false;
}

/*! \brief Per-nest check: does a buffer read observe the previous grid
 * iteration's value? A load is iteration-private only when a store that
 * covers the whole buffer, neither reads the buffer nor is if-guarded,
 * precedes it in the sink body (e.g. the T.clear before gemm's RMW
 * accumulation). */
class IterationPrivacyChecker : public StmtExprVisitor {
public:
  bool ReadsPrevious(const Var &data) const {
    auto it = state_.find(data);
    return it != state_.end() && it->second.reads_previous;
  }

private:
  struct St {
    bool reset_seen = false;
    bool reads_previous = false;
  };
  void VisitStmt_(const ForNode *op) override {
    loop_stack_.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
  }
  void VisitStmt_(const IfThenElseNode *op) override {
    ++if_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --if_depth_;
  }
  void VisitStmt_(const BufferStoreNode *op) override {
    StmtExprVisitor::VisitStmt_(op);
    if (StoreReadsSameBuffer(op) || if_depth_ > 0) {
      return;
    }
    St &st = state_[op->buffer->data];
    if (!st.reset_seen && StoreCoversWholeBuffer(op, loop_stack_)) {
      st.reset_seen = true;
    }
  }
  void VisitExpr_(const BufferLoadNode *op) override {
    if (!state_[op->buffer->data].reset_seen) {
      state_[op->buffer->data].reads_previous = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  std::unordered_map<Var, St, ObjectPtrHash, ObjectPtrEqual> state_;
  std::vector<const ForNode *> loop_stack_;
  int if_depth_ = 0;
};

/*! \brief Per-nest check for parameter/global buffers (those without an
 * AllocBuffer): every write-relevant access must be provably race-free.
 * Accesses are collected during the traversal and evaluated in Finish():
 * opaque uses are unanalyzable (a bare data var in a Call argument, or
 * address_of — the callee may write past the addressed element); all store
 * addresses must agree and be injective over the parallel scope and the
 * per-iteration serial scopes (concurrent writes to one address race even
 * when the values agree); loads of a written buffer must use that same
 * address. Read-only buffers are always fine. */
class OverlapStoreChecker : public StmtExprVisitor {
public:
  OverlapStoreChecker(const GridAccessAnalysis &analysis,
                      std::vector<std::pair<Var, PrimExpr>> parallel_scope,
                      std::vector<std::pair<Var, PrimExpr>> outer_serial_scope)
      : analysis_(analysis), parallel_scope_(std::move(parallel_scope)),
        outer_serial_scope_(std::move(outer_serial_scope)) {}

  bool found() const { return found_; }
  const std::string &buffer_name() const { return buffer_name_; }
  const char *reason() const { return reason_; }

  //! Evaluate the collected accesses; call after the traversal.
  void Finish() {
    arith::Analyzer analyzer;
    for (const auto &[data, acc] : access_) {
      bool fail = acc.opaque_unanalyzable;
      if (fail) {
        reason_ = "opaque pointer use (call_extern / address_of / "
                  "access_ptr)";
      }
      PrimExpr uniform;
      for (const StoreRec &rec : acc.stores) {
        PrimExpr simplified =
            rec.index.defined() ? analyzer.Simplify(rec.index) : PrimExpr();
        if (!simplified.defined()) {
          fail = true;
          reason_ = "unanalyzable store address";
          break;
        }
        if (!uniform.defined()) {
          uniform = simplified;
        } else if (!StructuralEqual()(uniform, simplified)) {
          fail = true; // stores to one buffer cover each other's addresses
          reason_ = "stores to one buffer cover each other's addresses";
          break;
        }
        auto ranges = rec.ranges;
        if (!ProveInjectiveIndex(ScalarizedIndex(rec.index, &ranges), ranges)) {
          fail = true; // colliding write across grid iterations
          reason_ = "write addresses collide across grid iterations";
          break;
        }
      }
      if (!fail && uniform.defined()) {
        // A written buffer may only be read at the same per-iteration
        // address; anything else is a loop-carried dependency.
        for (const PrimExpr &ld : acc.load_indices) {
          if (!ld.defined() ||
              !StructuralEqual()(uniform, analyzer.Simplify(ld))) {
            fail = true;
            reason_ = "cross-iteration dependency (loop-carried load)";
            break;
          }
        }
      }
      if (fail) {
        found_ = true;
        buffer_name_ = acc.name;
        return;
      }
    }
  }

private:
  struct StoreRec {
    PrimExpr index;
    ffi::Map<Var, Range> ranges;
  };
  struct BufAccess {
    std::string name;
    std::vector<StoreRec> stores; // plain BufferStore addresses
    std::vector<PrimExpr> load_indices;
    bool opaque_unanalyzable = false;
  };

  ffi::Map<Var, Range> CurrentRanges() const {
    ffi::Map<Var, Range> ranges;
    for (const auto *scope :
         {&parallel_scope_, &outer_serial_scope_, &serial_scope_}) {
      for (const auto &[var, extent] : *scope) {
        ranges.Set(var,
                   Range::FromMinExtent(IntImm(DataType::Int(32), 0), extent));
      }
    }
    return ranges;
  }

  void VisitStmt_(const ForNode *op) override {
    serial_scope_.push_back({op->loop_var, op->extent});
    StmtExprVisitor::VisitStmt_(op);
    serial_scope_.pop_back();
  }
  void VisitStmt_(const BufferStoreNode *op) override {
    if (!analysis_.HasAllocation(op->buffer->data)) {
      BufAccess &acc = access_[op->buffer->data];
      acc.name = op->buffer->name;
      acc.stores.push_back(
          {op->indices.size() == 1 ? op->indices[0] : PrimExpr(),
           CurrentRanges()});
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) override {
    if (!analysis_.HasAllocation(op->buffer->data)) {
      BufAccess &acc = access_[op->buffer->data];
      acc.name = op->buffer->name;
      acc.load_indices.push_back(op->indices.size() == 1 ? op->indices[0]
                                                         : PrimExpr());
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const CallNode *op) override {
    // address_of hands the callee a raw pointer into the buffer; the write
    // range through that pointer is unanalyzable (the callee may store past
    // the addressed element), so it takes the same conservative path as
    // access_ptr: refuse.
    if (op->op.same_as(builtin::address_of()) && !op->args.empty()) {
      if (const auto *load = op->args[0].as<BufferLoadNode>();
          load && !analysis_.HasAllocation(load->buffer->data)) {
        BufAccess &acc = access_[load->buffer->data];
        acc.name = load->buffer->name;
        acc.opaque_unanalyzable = true;
      }
    }
    ++in_call_;
    StmtExprVisitor::VisitExpr_(op);
    --in_call_;
  }
  void VisitExpr_(const VarNode *op) override {
    // A bare data var of a parameter/global buffer in a call argument is an
    // unanalyzable opaque use (call_extern / access_ptr). Create the entry
    // on demand: the buffer may have no plain load/store inside the nest.
    if (in_call_ > 0 && op->dtype.is_handle() &&
        !analysis_.HasAllocation(GetRef<Var>(op))) {
      BufAccess &acc = access_[GetRef<Var>(op)];
      acc.name = op->name_hint;
      acc.opaque_unanalyzable = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  const GridAccessAnalysis &analysis_;
  std::vector<std::pair<Var, PrimExpr>> parallel_scope_;
  std::vector<std::pair<Var, PrimExpr>> outer_serial_scope_;
  std::vector<std::pair<Var, PrimExpr>> serial_scope_;
  std::unordered_map<Var, BufAccess, ObjectPtrHash, ObjectPtrEqual> access_;
  int in_call_ = 0;
  bool found_ = false;
  std::string buffer_name_;
  const char *reason_ = "";
};

struct GridRewriter : public StmtMutator {
  GridRewriter(bool collapse_all_dims, int64_t min_trip,
               const GridAccessAnalysis &analysis, bool force_serial = false)
      : collapse_all_dims_(collapse_all_dims), min_trip_(min_trip),
        analysis_(analysis), force_serial_(force_serial) {}

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
      // Nested inside an already-parallel region: strip and stay serial.
      LOG(WARNING) << "tl.cpu_parallel: grid loop `" << op->loop_var->name_hint
                   << "` is nested inside another parallelized nest; it "
                      "stays serial";
      return For(op->loop_var, op->min, op->extent, ForKind::kSerial,
                 VisitStmt(op->body), std::nullopt,
                 StripGridAnnotation(op->annotations), NormalizedStep(op),
                 op->span);
    }
    return ConvertThenVisit(op, {});
  }

  //! Convert the nest headed by `head`, rebuilding transparent wrappers.
  Stmt ConvertNestElement(const Stmt &elem, const ForNode *head,
                          std::vector<Stmt> sunk) {
    if (const auto *attr = elem.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value,
                      ConvertNestElement(attr->body, head, std::move(sunk)),
                      attr->span);
    }
    return ConvertThenVisit(head, std::move(sunk));
  }

  //! Convert the nest headed by `head`, then keep visiting the result: an
  //! annotated nest hidden deeper must stay serial when this one turned
  //! parallel, and converts independently otherwise.
  Stmt ConvertThenVisit(const ForNode *head, std::vector<Stmt> sunk) {
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
                        sink_depth) &&
          !Privacy(grid_head).ReadsPrevious(alloc->buffer->data)) {
        sunk.push_back(elem);
      } else {
        kept.push_back(elem);
      }
    }
    out = std::move(kept);
    return sunk;
  }

  /*! \brief Lazily run (and cache) the iteration-privacy check over the
   * sink body of the nest headed by `head`. */
  IterationPrivacyChecker &Privacy(const ForNode *head) {
    auto it = privacy_cache_.find(head);
    if (it == privacy_cache_.end()) {
      it = privacy_cache_.emplace(head, IterationPrivacyChecker{}).first;
      std::vector<const ForNode *> loops = CollectChain(head);
      it->second(loops[SinkIndex(loops, collapse_all_dims_)]->body);
    }
    return it->second;
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

    // Atomic kernels keep the serial lowering (see pass_func): strip the
    // annotations without converting.
    if (force_serial_) {
      return rebuild_serial();
    }

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

    // Refuse writes to parameter/global buffers that cannot be proven
    // race-free across grid iterations.
    {
      std::vector<std::pair<Var, PrimExpr>> parallel_scope, outer_serial;
      for (size_t i = 0; i < loops.size(); ++i) {
        bool parallel_dim =
            collapse_all_dims_ || static_cast<size_t>(i) == parallel_idx;
        (parallel_dim ? parallel_scope : outer_serial)
            .emplace_back(loops[i]->loop_var, loops[i]->extent);
      }
      OverlapStoreChecker overlap(analysis_, std::move(parallel_scope),
                                  std::move(outer_serial));
      overlap(loops.back()->body);
      overlap.Finish();
      if (overlap.found()) {
        LOG(WARNING) << "tl.cpu_parallel: buffer `" << overlap.buffer_name()
                     << "` cannot be proven race-free across grid iterations "
                        "("
                     << overlap.reason() << "); grid loop `"
                     << head->loop_var->name_hint << "` stays serial";
        return rebuild_serial();
      }
    }

    // Refuse to parallelize when a function-scope buffer is mutated (or
    // opaquely used) inside the nest but cannot be privatized into it;
    // load-only sharing (e.g. a table initialized before the nest) is
    // race-free.
    for (const auto &kv : analysis_.AllocDepths()) {
      const Var &data = kv.first;
      if (kv.second != 0) {
        continue;
      }
      GridAccessAnalysis::AccessInfo info = analysis_.Lookup(data);
      if (!info.nests.count(head)) {
        continue; // not used inside this nest
      }
      if (SinkableAlloc(info, head, sink_depth) &&
          !Privacy(head).ReadsPrevious(data)) {
        bool was_sunk =
            std::any_of(sunk.begin(), sunk.end(), [&](const Stmt &s) {
              const auto *alloc = s.as<AllocBufferNode>();
              return alloc && alloc->buffer->data.same_as(data);
            });
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
            Privacy(head).ReadsPrevious(data)
                ? "it carries state across grid iterations"
            : info.opaque_inside
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
  std::unordered_map<const ForNode *, IterationPrivacyChecker> privacy_cache_;
  // Strip annotations without converting (kernels marked tl.cpu_had_atomics).
  bool force_serial_;
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

    bool force_serial =
        func->GetAttr<Bool>(attr::kCPUHadAtomics).value_or(Bool(false))->value;
    if (force_serial) {
      LOG(WARNING) << "tl.cpu_parallel: the kernel calls atomic ops, which "
                      "are lowered to serial read-modify-write on CPU; the "
                      "grid nest stays serial to avoid a data race";
    }

    GridRewriter rewriter(collapse_all_dims, min_trip, analysis, force_serial);
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
