/*!
 * \file infer_memory_scope.cc
 *
 * Infer concrete memory scopes for buffers allocated with the virtual scope
 * "auto" (T.auto_alloc, issue #277).
 *
 * The pass runs early: before VerifyBufferInit, warp specialization,
 * pipeline planning and layout inference, so that downstream passes that
 * branch on IsSharedBuffer (multi-versioning, cp.async/TMA staging, layout
 * inference) already see the final scope.
 *
 * Decision rules (conservative, correctness first):
 *   R1  tl.gemm accumulator (C)            -> local.fragment (hard)
 *   R2  tl.gemm A/B operands               -> shared.dyn     (hard)
 *   R3  T.copy(global -> X) dst inside a pipelined (num_stages) loop
 *                                          -> shared.dyn     (hard)
 *   R4  tl.cumsum / tl.cummax src or dst   -> shared.dyn     (hard)
 *   R5  tl.copy / tl.fill / tl.atomic_add  -> no constraint
 *   R6  all accesses are plain loads/stores inside T.Parallel nests, with a
 *       consistent (up to loop-var renaming) bijective index mapping
 *                                          -> local.fragment
 *   R7  all accesses are plain loads/stores outside parallel loops
 *                                          -> local
 *       (but accesses whose indices reference a per-thread threadIdx binding
 *       var, or which sit under a per-thread condition, are never R6/R7:
 *       a per-thread scope would give every thread a private copy with only
 *       its own slots written, silently breaking cross-thread reads)
 *   R8  anything else                      -> shared.dyn
 *   R9  R1 conflicting with R2/R3/R4       -> compile error with per-use
 *                                             diagnostics
 *   R10 dead buffer (no accesses)          -> local
 *   R11 bool dtype decided as shared       -> shared (MergeSharedMemory-
 *                                             Allocations cannot merge bool)
 *
 * On CPU targets every shared choice degrades to "local": the CPU backend has
 * no shared-memory hierarchy and its op lowerings reject shared scopes.
 *
 * The rewrite follows the CanonicalizeLegacyReducer precedent: for each auto
 * buffer a new data Var (with the new PointerType storage scope) and a new
 * Buffer are created, the owning sblock's alloc_buffers entry is replaced, and
 * all references in the body are rewritten.
 */

#include "../layout/layout.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/gemm.h"
#include "../op/gemm_sp.h"
#include "../op/operator.h"
#include "../op/scan.h"
#include "../op/utils.h"
#include "backend/common/target_utils.h"
#include "common/pipeline_utils.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr const char *kAutoScope = "auto";

/*! \brief Whether this call is a tile op (or a region bridge), however
 *  well-formed its arguments are. */
bool IsTileOpCall(const CallNode *op) {
  if (op->op.same_as(tl::region())) {
    return true;
  }
  auto opt_op = op->op.as<Op>();
  if (!opt_op.has_value()) {
    return false;
  }
  const std::string &name = opt_op.value()->name;
  return name.rfind("tl.tileop.", 0) == 0;
}

/*! \brief Parse a call as a tile op, or return null if it cannot be parsed.
 *
 * The call is not always in its final form at this point in the pipeline; a
 * builder that cannot interpret its arguments throws. Builders raise
 * tvm::ffi::Error, which derives from std::exception; catching the base covers
 * the rest by the same rule (same approach as VerifyBufferInit).
 */
TileOperator TryParseOperator(const Call &call) {
  try {
    return ParseOperator(call);
  } catch (const std::exception &) {
    return TileOperator();
  }
}

/*! \brief Accumulated facts about one auto-scope buffer. */
struct BufferScopeAnalysis {
  /*! \brief The buffer is read or written at least once. */
  bool has_access = false;
  /*! \brief Uses that require local.fragment, as human-readable reasons. */
  std::vector<std::string> fragment_reasons;
  /*! \brief Uses that require shared memory, as human-readable reasons. */
  std::vector<std::string> shared_reasons;
  /*! \brief Every access is a plain BufferLoad/BufferStore (no tile op). */
  bool all_plain = true;
  /*! \brief Every access sits inside an all-parallel loop nest. */
  bool all_parallel_nest = true;
  /*! \brief Every access sits outside parallel loops entirely. */
  bool all_sequential = true;
  /*! \brief Every parallel-nest access has a bijective index mapping. */
  bool bijective = true;
  /*! \brief Every parallel-nest access has the same index mapping (up to
   *  loop-var renaming) and the same loop extents. */
  bool mapping_consistent = true;
  /*! \brief Whether canonical_indices/canonical_extents hold a value. */
  bool has_canonical = false;
  /*! \brief Normalized indices of the first parallel-nest access. */
  Array<PrimExpr> canonical_indices;
  /*! \brief Parallel loop extents of the first parallel-nest access. */
  Array<PrimExpr> canonical_extents;
};

/*!
 * \brief Collect the auto-scope buffers of a function and how each is used.
 *
 * The whole function body is scanned once; buffers are keyed by their data
 * Var, so the allocating sblock's nesting does not matter.
 */
class AutoScopeCollector : public StmtExprVisitor {
public:
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> auto_buffers_;
  std::unordered_map<Var, BufferScopeAnalysis, ObjectPtrHash, ObjectPtrEqual>
      analyses_;

  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      if (buffer.scope() == kAutoScope) {
        auto_buffers_.emplace(buffer->data, buffer);
        analyses_.emplace(buffer->data, BufferScopeAnalysis{});
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    // Pre-launch-binding IR (e.g. pass-level use) carries thread bindings as
    // kThreadBinding For loops.
    const bool is_per_thread_axis =
        op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        IsPerThreadTag(op->thread_binding.value()->thread_tag);
    if (is_per_thread_axis) {
      thread_vars_.push_back(op->loop_var.get());
    }
    loop_stack_.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
    if (is_per_thread_axis) {
      thread_vars_.pop_back();
    }
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    // After MaterializeKernelLaunch, thread bindings are thread_extent
    // AttrStmts instead of For loops.
    bool pushed = false;
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>();
          iv != nullptr && IsPerThreadTag(iv->thread_tag)) {
        thread_vars_.push_back(iv->var.get());
        pushed = true;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
    if (pushed) {
      thread_vars_.pop_back();
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    // A write guarded by a per-thread condition (e.g. `if tx == 0`) only lands
    // in some threads' private copies under a per-thread scope.
    const bool thread_conditional = UsesAnyThreadVar(op->condition);
    if (thread_conditional) {
      ++thread_condition_depth_;
    }
    StmtExprVisitor::VisitStmt_(op);
    if (thread_conditional) {
      --thread_condition_depth_;
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    RecordPlainAccess(op->buffer, op->indices);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    RecordPlainAccess(op->buffer, op->indices);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (IsTileOpCall(op)) {
      if (TileOperator tile_op = TryParseOperator(GetRef<Call>(op));
          tile_op.defined()) {
        VisitTileOp(tile_op);
      } else {
        RecordOpaqueTileOp(op);
      }
      // Do not recurse: a tl.region argument wraps a BufferLoad that marks the
      // region, not an element access, and a parsed op's access regions already
      // cover its arguments.
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

private:
  /*! \brief The For nodes enclosing the node currently being visited. */
  std::vector<const ForNode *> loop_stack_;
  /*! \brief Per-thread (threadIdx.*) binding vars currently in scope. */
  std::vector<const VarNode *> thread_vars_;
  /*! \brief Enclosing if-conditions that reference a per-thread var. */
  int thread_condition_depth_ = 0;
  arith::Analyzer analyzer_;
  /*! \brief Canonical loop-var placeholders, one per (position, dtype). */
  std::unordered_map<std::string, Var> placeholders_;

  /*! \brief threadIdx.* axes are per-thread; blockIdx.* are block-uniform and
   *  do not by themselves make a local scope unsafe. */
  static bool IsPerThreadTag(const String &tag) {
    const std::string &s = tag;
    return s.rfind("threadIdx", 0) == 0;
  }

  /*! \brief Whether the expression references an in-scope per-thread var. */
  bool UsesAnyThreadVar(const PrimExpr &expr) const {
    if (thread_vars_.empty()) {
      return false;
    }
    return UsesVar(expr, [this](const VarNode *var) {
      return std::find(thread_vars_.begin(), thread_vars_.end(), var) !=
             thread_vars_.end();
    });
  }

  Var PlaceholderVar(size_t position, DataType dtype) {
    std::string key = std::to_string(position) + "_" +
                      std::to_string(dtype.code()) + "_" +
                      std::to_string(dtype.bits());
    auto it = placeholders_.find(key);
    if (it != placeholders_.end()) {
      return it->second;
    }
    Var var("pv" + std::to_string(position), dtype);
    placeholders_.emplace(key, var);
    return var;
  }

  BufferScopeAnalysis *Lookup(const Buffer &buffer) {
    auto it = analyses_.find(buffer->data);
    return it == analyses_.end() ? nullptr : &it->second;
  }

  /*! \brief Whether any enclosing loop is pipelined (carries num_stages). */
  bool InPipelinedLoop() const {
    for (const ForNode *loop : loop_stack_) {
      if (GetPipelineNumStages(loop).defined()) {
        return true;
      }
    }
    return false;
  }

  void RequireFragment(const Buffer &buffer, std::string reason) {
    if (BufferScopeAnalysis *a = Lookup(buffer)) {
      a->fragment_reasons.push_back(std::move(reason));
    }
  }

  void RequireShared(const Buffer &buffer, std::string reason) {
    if (BufferScopeAnalysis *a = Lookup(buffer)) {
      a->shared_reasons.push_back(std::move(reason));
    }
  }

  /*! \brief A tile op touches this buffer: it is accessed, but not through a
   *  plain per-element load/store. */
  void RecordTileOpAccess(const Buffer &buffer) {
    if (BufferScopeAnalysis *a = Lookup(buffer)) {
      a->has_access = true;
      a->all_plain = false;
      a->all_parallel_nest = false;
      a->all_sequential = false;
    }
  }

  /*! \brief A tile op call that could not be parsed: mark every buffer it
   *  references as opaquely accessed. */
  void RecordOpaqueTileOp(const CallNode *op) {
    for (const PrimExpr &arg : op->args) {
      if (const auto *load = arg.as<BufferLoadNode>()) {
        RecordTileOpAccess(load->buffer);
      } else if (const auto *var = arg.as<VarNode>()) {
        if (var->dtype.is_handle()) {
          auto it = analyses_.find(GetRef<Var>(var));
          if (it != analyses_.end()) {
            it->second.has_access = true;
            it->second.all_plain = false;
            it->second.all_parallel_nest = false;
            it->second.all_sequential = false;
          }
        }
      } else if (const auto *call = arg.as<CallNode>()) {
        RecordOpaqueTileOp(call);
      }
    }
  }

  void VisitTileOp(const TileOperator &tile_op) {
    // Role-specific hard constraints first.
    if (const auto *gemm = tile_op.as<GemmNode>()) {
      RequireFragment(gemm->cRegion_->buffer,
                      "used as T.gemm accumulator (requires local.fragment)");
      RequireShared(gemm->aRegion_->buffer,
                    "used as T.gemm A operand (requires shared.dyn)");
      RequireShared(gemm->bRegion_->buffer,
                    "used as T.gemm B operand (requires shared.dyn)");
    } else if (const auto *gemm_sp = tile_op.as<GemmSPNode>()) {
      RequireFragment(
          gemm_sp->cRegion_->buffer,
          "used as T.gemm_sp accumulator (requires local.fragment)");
      RequireShared(gemm_sp->aRegion_->buffer,
                    "used as T.gemm_sp A operand (requires shared.dyn)");
      RequireShared(gemm_sp->bRegion_->buffer,
                    "used as T.gemm_sp B operand (requires shared.dyn)");
    } else if (const auto *copy = tile_op.as<CopyNode>()) {
      if (InPipelinedLoop() && IsGlobalBuffer(copy->src)) {
        RequireShared(copy->dst,
                      "used as T.copy destination fed from global memory "
                      "inside a pipelined loop (requires shared.dyn)");
      }
    } else if (const auto *cumsum = tile_op.as<CumSumOpNode>()) {
      RequireShared(cumsum->src,
                    "used as T.cumsum operand (requires shared.dyn)");
      RequireShared(cumsum->dst,
                    "used as T.cumsum operand (requires shared.dyn)");
    } else if (const auto *cummax = tile_op.as<CumMaxOpNode>()) {
      RequireShared(cummax->src,
                    "used as T.cummax operand (requires shared.dyn)");
      RequireShared(cummax->dst,
                    "used as T.cummax operand (requires shared.dyn)");
    }
    // Generic: every region this op touches is a non-plain access.
    AccessRegions regions = tile_op->GetAccessRegions();
    for (const BufferRegion &region : regions.reads) {
      RecordTileOpAccess(region->buffer);
    }
    for (const BufferRegion &region : regions.writes) {
      RecordTileOpAccess(region->buffer);
    }
  }

  /*! \brief Record a plain element access and classify its loop nest. */
  void RecordPlainAccess(const Buffer &buffer, const Array<PrimExpr> &indices) {
    BufferScopeAnalysis *a = Lookup(buffer);
    if (a == nullptr) {
      return;
    }
    a->has_access = true;

    // An access whose indices reference a per-thread (threadIdx) binding var,
    // or one guarded by a per-thread condition, is not a scope-safe sequential
    // or parallel-nest pattern: under a per-thread scope every thread owns a
    // private copy with only its own slots written, so cross-thread reads (or
    // reads by threads the condition excluded) silently see garbage. Force the
    // shared fallback. (Values referencing thread vars are fine: each thread's
    // private copy stays self-consistent.)
    bool uses_thread_var = thread_condition_depth_ > 0;
    for (const PrimExpr &index : indices) {
      uses_thread_var = uses_thread_var || UsesAnyThreadVar(index);
    }
    if (uses_thread_var) {
      a->all_sequential = false;
      a->all_parallel_nest = false;
      return;
    }

    // Thread-binding loops are launch geometry, not compute loops; ignore them
    // when classifying the nest.
    std::vector<const ForNode *> compute_loops;
    for (const ForNode *loop : loop_stack_) {
      if (loop->kind != ForKind::kThreadBinding) {
        compute_loops.push_back(loop);
      }
    }
    Array<Var> parallel_vars;
    Array<PrimExpr> parallel_extents;
    bool all_parallel = !compute_loops.empty();
    for (const ForNode *loop : compute_loops) {
      if (loop->kind == ForKind::kParallel && is_zero(loop->min)) {
        parallel_vars.push_back(loop->loop_var);
        parallel_extents.push_back(loop->extent);
      } else {
        all_parallel = false;
      }
    }

    if (parallel_vars.empty()) {
      a->all_parallel_nest = false;
      return;
    }
    a->all_sequential = false;
    if (!all_parallel) {
      a->all_parallel_nest = false;
      return;
    }
    CheckParallelMapping(a, indices, parallel_vars, parallel_extents);
  }

  /*! \brief Check one parallel-nest access for bijectivity (DetectIterMap) and
   *  for mapping consistency with the accesses seen so far. */
  void CheckParallelMapping(BufferScopeAnalysis *a,
                            const Array<PrimExpr> &indices,
                            const Array<Var> &parallel_vars,
                            const Array<PrimExpr> &parallel_extents) {
    Map<Var, Range> input_iters;
    for (size_t i = 0; i < parallel_vars.size(); ++i) {
      input_iters.Set(parallel_vars[i],
                      Range::FromMinExtent(0, parallel_extents[i]));
    }
    auto result = arith::DetectIterMap(
        indices, input_iters, 1, arith::IterMapLevel::Bijective, &analyzer_);
    if (!result->errors.empty()) {
      a->bijective = false;
      return;
    }
    // Normalize loop vars to positional placeholders so mappings from
    // different nests can be compared structurally. The placeholder Vars are
    // created once per (position, dtype) and reused: ExprDeepEqual compares
    // free Vars by identity, so fresh objects would never compare equal.
    std::unordered_map<const VarNode *, PrimExpr> subst;
    for (size_t i = 0; i < parallel_vars.size(); ++i) {
      subst.emplace(parallel_vars[i].get(),
                    PlaceholderVar(i, parallel_vars[i].dtype()));
    }
    auto remap = [&subst](const Var &var) -> Optional<PrimExpr> {
      auto it = subst.find(var.get());
      if (it != subst.end()) {
        return it->second;
      }
      return std::nullopt;
    };
    Array<PrimExpr> normalized;
    normalized.reserve(indices.size());
    for (const PrimExpr &index : indices) {
      normalized.push_back(Substitute(index, remap));
    }
    if (!a->has_canonical) {
      a->has_canonical = true;
      a->canonical_indices = normalized;
      a->canonical_extents = parallel_extents;
      return;
    }
    if (!ArrayDeepEqual(a->canonical_indices, normalized) ||
        !ArrayDeepEqual(a->canonical_extents, parallel_extents)) {
      a->mapping_consistent = false;
    }
  }

  static bool ArrayDeepEqual(const Array<PrimExpr> &lhs,
                             const Array<PrimExpr> &rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    ExprDeepEqual equal;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!equal(lhs[i], rhs[i])) {
        return false;
      }
    }
    return true;
  }
};

/*! \brief Decide the concrete scope of one auto buffer from its analysis. */
String DecideScope(const Buffer &buffer, const BufferScopeAnalysis &a,
                   bool is_cpu_target) {
  // CPU targets have no shared-memory hierarchy and their op lowerings reject
  // shared scopes (e.g. src/cpu/op/fill.cc); every would-be-shared decision
  // degrades to thread-local storage there.
  const bool is_bool = buffer->dtype.is_bool();
  const char *shared_choice =
      is_cpu_target ? "local" : (is_bool ? "shared" : "shared.dyn");
  if (!a.fragment_reasons.empty() && !a.shared_reasons.empty()) {
    std::ostringstream os;
    os << "Error: cannot infer memory scope for buffer '" << buffer->name
       << "'\n";
    for (const std::string &reason : a.fragment_reasons) {
      os << "  - " << reason << "\n";
    }
    for (const std::string &reason : a.shared_reasons) {
      os << "  - " << reason << "\n";
    }
    os << "Hint: use T.alloc_shared / T.alloc_fragment explicitly.";
    LOG(FATAL) << os.str();
  }
  if (!a.fragment_reasons.empty()) {
    return "local.fragment";
  }
  if (!a.shared_reasons.empty()) {
    return shared_choice;
  }
  if (!a.has_access) {
    return "local";
  }
  if (a.all_plain && a.all_parallel_nest && a.bijective &&
      a.mapping_consistent) {
    return "local.fragment";
  }
  if (a.all_plain && a.all_sequential) {
    return "local";
  }
  return shared_choice;
}

/*!
 * \brief Rewrite buffer references after the scope decisions.
 *
 * Follows the CanonicalizeLegacyReducer precedent: the owning sblock's
 * alloc_buffers entry is swapped for a buffer whose data Var carries the new
 * PointerType storage scope, and BufferLoad/BufferStore/bare-Var references in
 * the body are rewritten. Tile op arguments go through tl.region bridge calls
 * whose region marker is a BufferLoad, so they are covered by the BufferLoad
 * hook. sblock reads/writes regions are remapped as well for completeness.
 */
class AutoScopeRewriter : public StmtExprMutator {
public:
  AutoScopeRewriter(Map<Var, Var> var_remap, Map<Buffer, Buffer> buffer_remap)
      : var_remap_(std::move(var_remap)),
        buffer_remap_(std::move(buffer_remap)) {}

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    auto *p_block = block.CopyOnWrite();
    p_block->alloc_buffers = RemapBuffers(op->alloc_buffers);
    p_block->reads = RemapRegions(op->reads);
    p_block->writes = RemapRegions(op->writes);
    // Annotations keyed by a buffer's data Var must follow the rewritten
    // buffer: T.annotate_layout (layout_map) and T.annotate_safe_value
    // (safe_value_map) can both reference an auto-scope buffer.
    RemapVarKeyedAnnotation<Layout>(p_block, attr::kLayoutMap);
    RemapVarKeyedAnnotation<PrimExpr>(p_block, attr::kSafeValueMap);
    // barrier_init is intentionally not remapped: barrier buffers carry their
    // own dedicated scope (T.alloc_barrier) and can never be auto.
    return block;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_remap_.find(store->buffer);
    if (it == buffer_remap_.end()) {
      return store;
    }
    return BufferStore((*it).second, store->value, store->indices);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = buffer_remap_.find(load->buffer);
    if (it == buffer_remap_.end()) {
      return load;
    }
    return BufferLoad((*it).second, load->indices);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto it = var_remap_.find(GetRef<Var>(op));
    if (it != var_remap_.end()) {
      return (*it).second;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  /*! \brief Rewrite the keys of a Var-keyed map annotation (e.g. layout_map).
   *
   * Uses the exact static value type so that unrelated Map-typed annotations
   * with non-Var keys are left untouched.
   */
  template <typename V>
  void RemapVarKeyedAnnotation(SBlockNode *p_block, const char *key) const {
    auto ref = p_block->annotations.Get(key);
    if (!ref.has_value()) {
      return;
    }
    auto map = ref.value().as<Map<Var, V>>();
    if (!map.has_value()) {
      return;
    }
    Map<Var, V> updated;
    bool changed = false;
    for (const auto &[var, value] : map.value()) {
      auto it = var_remap_.find(var);
      if (it != var_remap_.end()) {
        updated.Set((*it).second, value);
        changed = true;
      } else {
        updated.Set(var, value);
      }
    }
    if (changed) {
      p_block->annotations.Set(key, updated);
    }
  }

  Array<Buffer> RemapBuffers(const Array<Buffer> &buffers) const {
    Array<Buffer> result;
    result.reserve(buffers.size());
    bool changed = false;
    for (const Buffer &buffer : buffers) {
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        result.push_back((*it).second);
        changed = true;
      } else {
        result.push_back(buffer);
      }
    }
    return changed ? result : buffers;
  }

  Array<BufferRegion> RemapRegions(const Array<BufferRegion> &regions) const {
    Array<BufferRegion> result;
    result.reserve(regions.size());
    bool changed = false;
    for (const BufferRegion &region : regions) {
      auto it = buffer_remap_.find(region->buffer);
      if (it != buffer_remap_.end()) {
        result.push_back(BufferRegion((*it).second, region->region));
        changed = true;
      } else {
        result.push_back(region);
      }
    }
    return changed ? result : regions;
  }

  Map<Var, Var> var_remap_;
  Map<Buffer, Buffer> buffer_remap_;
};

using namespace tirx::transform;

tvm::transform::Pass InferMemoryScope() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    AutoScopeCollector collector;
    collector(f->body);
    if (collector.auto_buffers_.empty()) {
      return f;
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    const bool is_cpu_target = target.defined() && TargetIsCPU(target.value());
    Map<Var, Var> var_remap;
    Map<Buffer, Buffer> buffer_remap;
    for (const auto &[var, buffer] : collector.auto_buffers_) {
      String new_scope =
          DecideScope(buffer, collector.analyses_.at(var), is_cpu_target);
      // Preserve the span so later diagnostics still point at the user's
      // T.auto_alloc line.
      Var new_var(buffer->data->name_hint,
                  PointerType(PrimType(buffer->dtype), new_scope),
                  buffer->data->span);
      Buffer new_buffer(new_var, buffer->dtype, buffer->shape, buffer->strides,
                        buffer->elem_offset, buffer->name,
                        buffer->data_alignment, buffer->offset_factor,
                        buffer->buffer_type);
      var_remap.Set(buffer->data, new_var);
      buffer_remap.Set(buffer, new_buffer);
    }
    AutoScopeRewriter rewriter(var_remap, buffer_remap);
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InferMemoryScope", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.InferMemoryScope", InferMemoryScope);
}

} // namespace

} // namespace tvm::tl
