/*!
 * \file materialize_ws_schedule.cc
 * \brief Materialize a user-provided warp-specialization schedule.
 *
 * The schedule — a T.WSSchedule annotation on the kernel's block (see
 * T.annotate_ws_schedule) — declares roles (named warp ranges with
 * register budgets), pipelines, and scopes (the scheduled loops plus an
 * implicit root; each gives participating roles a per-iteration body of
 * ops, child scopes, and sync points). The pass rewrites the kernel
 * into explicit form — role branches, versioned buffers, mbarrier
 * waits/arrives, asynchronous copies and MMAs — before
 * PipelinePlanning / LayoutInference / LowerTileOp, so downstream
 * passes see the same tile-op IR a hand-written kernel would produce.
 *
 * A pipeline is one producer/consumer handshake: a "full" and an
 * "empty" mbarrier array protecting a set of buffers. It synchronizes
 * exactly ONE scope, between exactly two roles. A buffer may be bound
 * to several pipelines with strictly nested scopes; the analysis is
 * hierarchical — at the enclosing level, the deeper binding's scope is
 * one opaque use, synchronous by default (a tcgen05 watermark covers
 * the role's own in-flight MMA issue), and inner synchronization never
 * penetrates outward. Each level replicates the buffer by its own depth
 * and hands ONE version down, so the buffer holds the PRODUCT of the
 * depths, outer-major.
 *
 * Ids: a schedulable op is one statement, named by a "tl.ws_op_id"
 * annotation or a `with T.ws_op(id):` wrapper. Scope loops are serial
 * loops or wrapped while loops (runtime phase counters; role-uniform
 * condition). Every statement must be placed; several roles may place
 * an op only when it touches no pipeline buffers. Dependencies are
 * computed, not declared (QueryAccess).
 *
 * Synchronization: acquire/commit and wait/release bracket a role's
 * work on a pipeline; an op runs inside an open span of the pipeline
 * owning each access, and accesses rebind to version (phase % depth).
 * Sync entries carry a software-pipeline stage: an entry at stage s
 * runs (s - s_min) iterations behind, unrolled into prologue/epilogue
 * steps around a boundary-check-free steady-state loop. Arrive counts
 * derive from the signaling ops' atoms; an op inside several brackets
 * signals each of them (a tcgen05 commit is a watermark on the role's
 * in-order queue, so one MMA validly signals several sides). Source
 * `if`s become per-op guards; sync entries stay unconditional; an `if`
 * around a scope loop guards the whole scope, role-uniformly.
 * VerifySchedule rejects broken schedules at compile time: span
 * coverage, cycle balance, deadlock freedom under the mbarrier parity
 * model.
 *
 * TODO: cluster launch control; data-dependent synchronization; 2-CTA
 * GEMM; epilogue sub-tiling; try_acquire/try_wait split sync entries;
 * multi-signer barrier sides (FA4's merged S/P/O barrier) and, with
 * them, pipelines sharing one storage (the P-in-S TMEM overlay); sync
 * elision on the synchronization dependence graph (issue-order-
 * dependent; elided empty sides need a lap-freedom proof; proxy fences
 * must be re-placed); cross-scope software pipelining of an op INSIDE a
 * child scope relative to its siblings; else branches; region-granular
 * pipeline protection.
 */

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cuda/op/builtin.h"
#include "cuda/op/copy.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/utils.h"
#include "transform/common/mbarrier.h"
#include "transform/common/warp_specialize.h"

#include "./auto_schedule/memory_detector.h"
#include "./ws_analysis.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// Below this register budget a role donates registers (setmaxnreg.dec);
// at or above it the role receives (setmaxnreg.inc).
constexpr int kNregIncThreshold = 128;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
// An id value arrives as an ffi String (loop annotations) or as a
// StringImm (call annotations, whose values must be objects).
String ExtractWSOpId(const Any &a) {
  if (auto v = a.try_cast<String>())
    return *std::move(v);
  if (const auto *s = a.as<StringImmNode>())
    return s->value;
  LOG(FATAL) << "ws_schedule: expected string for " << kWSOpIdKey << ", got "
             << a.GetTypeKey();
  return "";
}

// Rebuild an annotations map, keeping only the keys `keep` accepts. The
// value type is templated because For/SBlock annotations are Any-valued
// while Call annotations are ObjectRef-valued.
template <typename V, typename Pred>
Map<String, V> FilterAnnotations(const Map<String, V> &ann, Pred keep) {
  Map<String, V> result;
  for (const auto &[k, v] : ann) {
    if (keep(k))
      result.Set(k, v);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Tile op helpers
// ---------------------------------------------------------------------------
const Op &TmaCopyOp() {
  static const Op &op = Op::Get("tl.tileop.tma_copy");
  return op;
}
const Op &Tcgen05GemmOp() {
  static const Op &op = Op::Get("tl.tileop.tcgen05_gemm");
  return op;
}

// ---------------------------------------------------------------------------
// The schedule object model. Everything a scope body contains derives
// from BodyEntry; a Scope IS an Operation (at its parent's level it is
// one opaque op). Polymorphic body entries are shared_ptr-owned (the
// registries and every referencing body share them); the flat specs
// (RoleSpec, PipelineSpec) are uniquely owned by the materializer and
// referenced by plain pointer.
// ---------------------------------------------------------------------------
struct RoleSpec;
struct PipelineSpec;
struct Synchronization;
struct Operation;
struct Scope;

struct RoleSpec {
  String name;
  int warp_lo = 0, warp_hi = 0; // [lo, hi) in warps
  int nreg = 0;                 // 0 = absent
  int index = -1;               // position in the sorted role list
  int NumThreads() const { return (warp_hi - warp_lo) * 32; }
  int NregAction() const { return nreg >= kNregIncThreshold ? 1 : 0; }
};

// Identity-keyed maps on Buffer handles. Iteration order is hash order
// and never observable: consumers renormalize into ordered containers.
// BufferDefMap maps each buffer a schedulable op accesses to its defs:
// the pipelines protecting the buffer, outermost scope first (resolved
// by BuildUseChains; empty = unprotected).
using BufferDefMap = std::unordered_map<Buffer, std::vector<PipelineSpec *>,
                                        ObjectPtrHash, ObjectPtrEqual>;
using BufferVersionMap =
    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
// Pipeline ownership is storage-keyed: T.view / T.reshape aliases that share
// one data var join the same pipelines and physical versioning. The
// bindings are sorted outermost scope first (a strictly nested chain).
using StoragePipelineMap = std::unordered_map<Var, std::vector<PipelineSpec *>,
                                              ObjectPtrHash, ObjectPtrEqual>;

// Orders pipeline-keyed containers by declaration position, never by
// address: generated kernels must be byte-identical across runs.
struct PipelineOrder {
  bool operator()(const PipelineSpec *a, const PipelineSpec *b) const;
};
using PipelineSet = std::set<const PipelineSpec *, PipelineOrder>;

// ---------------------------------------------------------------------------
// Op atoms: the asynchronous instruction (if any) an op lowers to,
// classified once so planner and emitter cannot disagree. kSync arrives
// per thread; kTmaCopy rides the barrier's transaction count;
// kTcgen05Gemm signals through tcgen05.commit (a watermark on the role's
// in-order MMA queue); kCpAsyncCopy through a deferred per-thread
// cp.async.mbarrier.arrive.
// ---------------------------------------------------------------------------
enum class OpAtom : uint8_t {
  kSync = 0,
  kTmaCopy = 1,
  kTcgen05Gemm = 2,
  kCpAsyncCopy = 3,
};

struct AccessSet {
  // QueryAccess records the accesses; BuildUseChains resolves the defs.
  BufferDefMap reads, writes;

  // The defs of all accesses, unprotected buffers excluded.
  PipelineSet Defs() const;
};

// Collects the statement's accesses with the same MemoryAccessDetector the
// automatic scheduler uses, so both layers share one oracle. Pipeline
// protection is whole-buffer.
void QueryAccess(const Stmt &stmt, AccessSet *acc) {
  MemoryAccessDetector detector;
  detector.Analyze(stmt);
  for (const BufferRegion &r : detector.GetReadRegions())
    acc->reads.emplace(r->buffer, std::vector<PipelineSpec *>());
  for (const BufferRegion &r : detector.GetWriteRegions())
    acc->writes.emplace(r->buffer, std::vector<PipelineSpec *>());
  // TODO: Add variable-induced dependencies
}

// One entry of a scope body: a Synchronization, a leaf Operation, or a
// child Scope (which is itself an Operation).
struct BodyEntry {
  virtual ~BodyEntry() = default;
  virtual Synchronization *AsSync() { return nullptr; }
  virtual Operation *AsOperation() { return nullptr; }
  virtual Scope *AsScope() { return nullptr; }
  const Synchronization *AsSync() const {
    return const_cast<BodyEntry *>(this)->AsSync();
  }
  const Operation *AsOperation() const {
    return const_cast<BodyEntry *>(this)->AsOperation();
  }
  const Scope *AsScope() const {
    return const_cast<BodyEntry *>(this)->AsScope();
  }
};
using BodyEntryPtr = std::shared_ptr<BodyEntry>;

struct Synchronization : BodyEntry {
  WSSyncKind kind; // set at parse time; never null afterwards
  PipelineSpec *pipeline = nullptr;
  int stage = 0;
  Synchronization *AsSync() override { return this; }
};

// One schedulable op: declared by ParseSchedule, completed by
// RecordOpStmt with the matched statement, filled in by later phases.
struct Operation : BodyEntry {
  String id;
  Stmt stmt;                   // the single matched original statement
  Optional<PrimExpr> guard;    // original guard, if any
  AccessSet access;            // buffer accesses and their defs
  OpAtom atom = OpAtom::kSync; // classified once by BuildOpAtoms
  // The written storage's binding at the op's level: the pipeline whose
  // barrier carries the op's completion signal (a TMA copy's
  // transaction). Always null for a Scope — a scope is never
  // completion-wired.
  const PipelineSpec *write_def = nullptr;
  std::set<const RoleSpec *> roles; // the placing roles
  Scope *sched_scope = nullptr;     // scope whose body references this
  bool uses_async_proxy = false;    // touches buffers through the async proxy
  bool fused_arrive = false;        // this copy carries its cycle's arrive

  Operation *AsOperation() override { return this; }
  virtual bool PlacedBy(const RoleSpec &role) const {
    return roles.count(&role) != 0;
  }
  // The atom the role's arrive must match; a leaf op has one placing
  // role, so its atom is role-independent.
  virtual OpAtom AtomFor(const RoleSpec &role) const { return atom; }
  // Whether anything below touches buffers through the async proxy.
  virtual bool TouchesAsyncProxy() const { return uses_async_proxy; }
};

struct Scope : Operation {
  For orig_loop; // undefined for the root scope (until matched, for others)
  // Set when the scope is a T.ws_op-wrapped while loop: phases under it
  // use runtime counters, and the condition must be role-uniform.
  While orig_while;
  // Per-role instruction sequence (by RoleSpec::index); empty when the
  // role has no body here. The parent scope is Operation::sched_scope.
  std::vector<std::vector<BodyEntryPtr>> bodies;

  Scope *AsScope() override { return this; }

  // The implicit root is the only scope without a loop.
  bool IsRoot() const { return !orig_loop.defined() && !orig_while.defined(); }

  const std::vector<BodyEntryPtr> &BodyOf(const RoleSpec &role) const {
    return bodies[role.index];
  }

  // Whether this scope is `ancestor` or nested somewhere below it.
  bool IsNestedIn(const Scope *ancestor) const {
    for (const Scope *s = this; s != nullptr; s = s->sched_scope)
      if (s == ancestor)
        return true;
    return false;
  }

  // The scope's nesting depth: the root is 0, each level below adds one.
  int Depth() const {
    int depth = 0;
    for (const Scope *s = sched_scope; s != nullptr; s = s->sched_scope)
      ++depth;
    return depth;
  }

  // Whether cycles in this scope advance once per counted iteration of
  // the enclosing for-loop nest, so a phase can be linearized over it:
  // no while ancestor (no iteration expression), no guard on this scope
  // or an ancestor (a skipped cycle would still advance the linear
  // phase), and a rectangular nest (no bound reading an outer loop var).
  bool HasLinearPhase() const {
    std::vector<const Scope *> chain; // this -> root
    for (const Scope *s = this; s != nullptr; s = s->sched_scope)
      chain.push_back(s);
    std::unordered_set<const VarNode *> outer;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
      const Scope *s = *it;
      if (s->orig_while.defined() || s->guard.defined())
        return false;
      if (!s->orig_loop.defined())
        continue; // the root
      auto uses_outer = [&outer](const PrimExpr &e) {
        return UsesVar(
            e, [&outer](const VarNode *v) { return outer.count(v) != 0; });
      };
      if (uses_outer(s->orig_loop->min) || uses_outer(s->orig_loop->extent))
        return false;
      outer.insert(s->orig_loop->loop_var.get());
    }
    return true;
  }

  bool PlacedBy(const RoleSpec &role) const override {
    return !bodies[role.index].empty();
  }

  // To its parent a scope is one opaque op; BuildOpAtoms folds its
  // summary once: per role, the watermarkable work (tcgen05 commit /
  // cp.async deferred arrive) possibly still in flight at exit — TMA
  // transactions ride their issue's own barrier and leave no residue —
  // and the async-proxy bit into the inherited uses_async_proxy.
  // TODO: reject the unobservable corner — a doubly-bound storage whose
  // inner binding is written by a TMA copy while a different role
  // signals the outer side; nothing orders that arrive after the data
  // lands.
  struct Residue {
    bool tcgen05 = false;
    bool cp_async = false;
  };
  std::vector<Residue> role_residues;
  OpAtom AtomFor(const RoleSpec &role) const override {
    const Residue &r = role_residues[role.index];
    return r.tcgen05    ? OpAtom::kTcgen05Gemm
           : r.cp_async ? OpAtom::kCpAsyncCopy
                        : OpAtom::kSync;
  }
};

// Orders a pipeline's uses by op id: deterministic, one entry per op.
struct OpIdLess {
  bool operator()(const Operation *a, const Operation *b) const {
    return a->id < b->id;
  }
};
using UseSet = std::set<const Operation *, OpIdLess>;

// How one side of a pipeline's barrier pair is signaled, derived from
// the signaling role's uses:
//   count = has_tcgen05_arrival + fused_tma_arrive
//         + (has_cpasync_arrival + has_thread_arrive) * role threads.
// The deferred cp.async arrive orders only prior cp.async ops, so it
// never replaces the per-thread arrive of synchronous work.
// Cross-proxy invariant (PTX 9.7.9.26.2): generic- and async-proxy
// accesses to one location are unordered without a fence.proxy.async in
// the synchronization chain, executed by the generic-access thread. A
// role publishing generic WRITES to pipeline buffers therefore fences
// before its arrive when the other side touches them through the async
// proxy; per-thread InjectFenceProxy cannot see a cross-role handoff.
// TMEM ordering is InjectTcgen05Fence's job.
struct BarrierSidePlan {
  bool has_transaction = false;     // TMA transactions ride the tx-count
  bool has_tcgen05_arrival = false; // one tcgen05.commit covers all MMAs
  bool has_cpasync_arrival = false; // per-thread deferred cp.async arrive
  bool has_thread_arrive = false;   // per-thread arrive at the sync entry
  bool fused_tma_arrive = false;    // one elected arrive rides the last copy
  bool needs_proxy_fence = false;   // generic writes published by the arrive
  const RoleSpec *signal_role = nullptr; // the signaling role
  int64_t count = 0;                     // resulting mbarrier.init count
};

struct PipelineSpec {
  String name;
  int depth = 1;
  int index = -1; // declaration position; keys deterministic ordering
  // The single scope whose bodies hold this pipeline's sync entries
  // (resolved by ResolvePipelineScopes; a pipeline synchronizes exactly
  // one scope).
  Scope *sync_scope = nullptr;
  std::vector<Buffer> buffers; // original buffers
  Buffer full, empty;          // materialized barrier buffers
  // Uses at THIS pipeline's level: leaf ops it owns (their innermost
  // containing binding) plus the collapsed scopes of deeper bindings.
  // Producers write the storage, consumers read it.
  UseSet producers, consumers;
  const UseSet &Uses(bool producer_side) const {
    return producer_side ? producers : consumers;
  }
  // Derived from the signaling ops' atoms (never user-specified).
  BarrierSidePlan full_plan, empty_plan;
};

bool PipelineOrder::operator()(const PipelineSpec *a,
                               const PipelineSpec *b) const {
  return a->index < b->index;
}

PipelineSet AccessSet::Defs() const {
  PipelineSet defs;
  for (const auto *side : {&reads, &writes})
    for (const auto &[buf, buf_defs] : *side)
      for (const PipelineSpec *def : buf_defs)
        defs.insert(def);
  return defs;
}

// The innermost binding whose scope contains `scope` (null if none):
// the pipeline that owns an access there. Enclosing bindings see the
// access only through their level's collapsed scope.
PipelineSpec *
InnermostContainingBinding(const std::vector<PipelineSpec *> &bindings,
                           const Scope *scope) {
  PipelineSpec *found = nullptr;
  if (scope == nullptr)
    return found;
  for (PipelineSpec *p : bindings) // outermost first
    if (scope->IsNestedIn(p->sync_scope))
      found = p;
  return found;
}

// Classify one parsed tile op with the same instruction-selection helpers
// the lowering uses, so the atom matches what the op actually lowers to.
OpAtom ClassifyTileOp(const TileOperator &tile_op, const Target &target) {
  if (!tile_op.defined())
    return OpAtom::kSync; // a non-tile-op intrinsic
  if (const auto *copy = tile_op.as<CopyNode>()) {
    cuda::CopyInstSelection sel =
        cuda::ClassifyWarpSpecializedCopy(*copy, target);
    ICHECK(sel.supported) << "ws_schedule: copy instruction selection failed: "
                          << sel.reason;
    // Only TMA loads ride the pipeline barrier's transaction count; TMA
    // stores complete through the commit-group machinery and stay kSync.
    if (cuda::CopyInstIsTMALoad(sel.inst))
      return OpAtom::kTmaCopy;
    if (cuda::CopyInstIsCPAsync(sel.inst))
      return OpAtom::kCpAsyncCopy;
    return OpAtom::kSync;
  }
  if (const auto *gemm = tile_op.as<GemmNode>()) {
    return IsTmemBuffer(gemm->cRegion_->buffer) ? OpAtom::kTcgen05Gemm
                                                : OpAtom::kSync;
  }
  return OpAtom::kSync;
}

// Whether a tile op touches its buffers through the async proxy: TMA
// copies, tcgen05 MMAs, and — conservatively — every gemm (wgmma reads
// shared operands through the async proxy). TODO: exempt Ampere MMAs.
bool UsesAsyncProxy(const Call &call, const Target &target) {
  switch (ClassifyStmt(Evaluate(call), target)) {
  case TileStmtKind::kTmaProducer:
  case TileStmtKind::kTmaStore:
  case TileStmtKind::kTcgen05Mma:
    return true;
  default:
    return ParseOperator(call).as<GemmNode>() != nullptr;
  }
}

// ---------------------------------------------------------------------------
// The materializer
// ---------------------------------------------------------------------------
class WSScheduleMaterializer {
public:
  WSScheduleMaterializer(SBlock block, Var thread_var, Target target)
      : block_(std::move(block)), thread_var_(std::move(thread_var)),
        target_(std::move(target)) {}

  int NumThreads() const { return num_warps_ * 32; }

  SBlock Run() {
    ParseSchedule();
    ResolvePipelineScopes();
    MatchKernel();
    PlanVersionedBuffers();
    BuildOpAtoms(); // before BuildUseChains: scope atoms fold the leaf atoms
    BuildUseChains();
    PlanArriveCounts();
    VerifySchedule();
    return RebuildBlock(EmitRoleBranches());
  }

private:
  // ---- schedule parsing ---------------------------------------------------

  // Resolve a schedule-referenced buffer to the block's own
  // allocation: by identity, then data var, then name.
  Buffer FindBlockBuffer(const Buffer &ref) {
    for (const Buffer &b : block_->alloc_buffers) {
      if (b.same_as(ref) || b->data.same_as(ref->data))
        return b;
    }
    for (const Buffer &b : block_->alloc_buffers) {
      if (b->name == ref->name)
        return b;
    }
    LOG(FATAL) << "ws_schedule: pipeline buffer '" << ref->name
               << "' is not allocated in this kernel";
    return Buffer();
  }

  void ParseSchedule() {
    auto ann = block_->annotations.Get(kWSScheduleKey);
    ICHECK(ann.has_value());
    auto sched_opt = ann.value().try_cast<WSSchedule>();
    ICHECK(sched_opt.has_value())
        << "ws_schedule: annotation must be a tl.WSSchedule object, got "
        << ann.value().GetTypeKey();
    WSSchedule sched = *std::move(sched_opt);

    num_warps_ = static_cast<int>(sched->num_warps);
    ICHECK_GT(num_warps_, 0) << "ws_schedule: num_warps must be positive";
    ICHECK_EQ(num_warps_ % 4, 0)
        << "ws_schedule: num_warps must be a multiple of 4 (warps are "
           "managed in warpgroups of 4), got "
        << num_warps_;

    // Roles.
    for (const WSRole &r : sched->roles) {
      auto role = std::make_unique<RoleSpec>();
      role->name = r->name;
      role->warp_lo = static_cast<int>(r->warp_lo);
      role->warp_hi = static_cast<int>(r->warp_hi);
      role->nreg = static_cast<int>(r->max_nreg);
      ICHECK_GE(role->warp_lo, 0) << "ws_schedule: role " << role->name
                                  << " has a negative warp range start";
      ICHECK_LT(role->warp_lo, role->warp_hi)
          << "ws_schedule: role " << role->name << " has an empty warp range";
      ICHECK_LE(role->warp_hi, num_warps_)
          << "ws_schedule: role " << role->name << " exceeds num_warps";
      for (const auto &other : roles_)
        ICHECK(other->name != role->name)
            << "ws_schedule: duplicate role name '" << role->name << "'";
      roles_.push_back(std::move(role));
    }
    std::stable_sort(roles_.begin(), roles_.end(),
                     [](const std::unique_ptr<RoleSpec> &a,
                        const std::unique_ptr<RoleSpec> &b) {
                       return a->warp_lo < b->warp_lo;
                     });
    for (size_t i = 0; i < roles_.size(); ++i)
      roles_[i]->index = static_cast<int>(i);
    for (size_t i = 1; i < roles_.size(); ++i) {
      ICHECK_LE(roles_[i - 1]->warp_hi, roles_[i]->warp_lo)
          << "ws_schedule: roles " << roles_[i - 1]->name << " ["
          << roles_[i - 1]->warp_lo << ", " << roles_[i - 1]->warp_hi
          << ") and " << roles_[i]->name << " [" << roles_[i]->warp_lo << ", "
          << roles_[i]->warp_hi << ") have overlapping warp ranges";
    }

    // setmaxnreg allocates per warpgroup: roles sharing one must agree
    // on max_nreg. Idle warps adopt their warpgroup's request; a fully
    // idle warpgroup donates down to the smallest donor budget.
    warpgroup_nreg_.assign(num_warps_ / 4, -1); // -1 = no covering role
    for (const auto &r : roles_) {
      for (int w = r->warp_lo; w < r->warp_hi; ++w) {
        int &wg = warpgroup_nreg_[w / 4];
        if (wg == -1) {
          wg = r->nreg;
        } else {
          ICHECK_EQ(wg, r->nreg)
              << "ws_schedule: role " << r->name << " requests max_nreg "
              << r->nreg << " but another role of warpgroup " << w / 4
              << " (warps " << w / 4 * 4 << ".." << w / 4 * 4 + 3
              << ") requests " << wg << " (0 = none); setmaxnreg "
              << "allocates per warpgroup, so all four warps must "
              << "allocate or deallocate the same number of registers";
        }
      }
    }
    int donor = 0;
    for (const auto &r : roles_) {
      if (r->nreg > 0 && r->NregAction() == 0)
        donor = donor == 0 ? r->nreg : std::min(donor, r->nreg);
    }
    for (int &v : warpgroup_nreg_) {
      if (v == -1)
        v = donor;
    }

    // Pipelines.
    for (const WSPipeline &p : sched->pipelines) {
      ICHECK(FindPipeline(p->name) == nullptr)
          << "ws_schedule: duplicate pipeline name '" << p->name << "'";
      auto pipeline = std::make_unique<PipelineSpec>();
      pipeline->name = p->name;
      pipeline->depth = static_cast<int>(p->depth);
      pipeline->index = static_cast<int>(pipelines_.size());
      ICHECK_GE(pipeline->depth, 1);
      for (const tirx::Buffer &b : p->buffers) {
        Buffer buffer = FindBlockBuffer(b);
        ICHECK(IsSharedBuffer(buffer) || IsTmemBuffer(buffer))
            << "ws_schedule: pipeline '" << p->name << "' buffer '"
            << buffer->name << "' must live in shared or tensor memory; "
            << "each role has its own instance of private storage";
        pipeline->buffers.push_back(std::move(buffer));
      }
      pipelines_.push_back(std::move(pipeline));
    }

    // Scopes: create the shells first so body references resolve in
    // any declaration order, then build the typed bodies.
    for (const WSScope &s : sched->scopes) {
      ICHECK(FindScope(s->id) == nullptr)
          << "ws_schedule: duplicate scope id '" << s->id << "'";
      auto scope = std::make_shared<Scope>();
      scope->id = s->id;
      scope->bodies.resize(roles_.size());
      scopes_.push_back(std::move(scope));
    }
    // A body entry names a sync point, a child scope, or an op; ops are
    // created at first reference. Validate the reference graph: an op is
    // placed in one scope, at most once per role; a child scope has one
    // parent, is entered at most once per role, and is never the root.
    std::set<std::pair<const Scope *, int>> scope_refs;
    for (size_t si = 0; si < scopes_.size(); ++si) {
      const std::shared_ptr<Scope> &scope = scopes_[si];
      const WSScope &s = sched->scopes[si];
      for (const auto &[role_name, instrs] : s->bodies) {
        const std::unique_ptr<RoleSpec> *role_it = nullptr;
        for (const auto &role : roles_)
          if (role->name == role_name)
            role_it = &role;
        ICHECK(role_it != nullptr)
            << "ws_schedule: scope " << scope->id
            << " has a body for unknown role '" << role_name << "'";
        const RoleSpec &role = **role_it;
        std::vector<BodyEntryPtr> entries;
        for (const WSInstr &instr : instrs) {
          if (const auto *sync_node = instr.as<WSSyncNode>()) {
            auto sync = std::make_shared<Synchronization>();
            sync->kind = sync_node->kind;
            sync->stage = static_cast<int>(sync_node->stage);
            sync->pipeline = FindPipeline(sync_node->pipeline);
            ICHECK(sync->pipeline != nullptr)
                << "ws_schedule: unknown pipeline '" << sync_node->pipeline
                << "' in scope " << scope->id;
            entries.push_back(std::move(sync));
            continue;
          }
          const auto *op_ref = instr.as<WSOpRefNode>();
          ICHECK(op_ref != nullptr) << "ws_schedule: unknown instruction type "
                                    << instr->GetTypeKey();
          if (std::shared_ptr<Scope> child = FindScope(op_ref->id)) {
            ICHECK(child->id != kWSRootScopeId)
                << "ws_schedule: the root scope cannot be referenced from a "
                   "scope body";
            if (child->sched_scope == nullptr) {
              child->sched_scope = scope.get();
            } else {
              ICHECK_EQ(child->sched_scope, scope.get())
                  << "ws_schedule: scope '" << child->id
                  << "' is referenced from multiple parent scopes ('"
                  << child->sched_scope->id << "' and '" << scope->id << "')";
            }
            ICHECK(scope_refs.insert({child.get(), role.index}).second)
                << "ws_schedule: scope '" << child->id
                << "' is referenced more than once by role " << role.name
                << "; each participating role enters a scope exactly once "
                   "per parent iteration";
            entries.push_back(child);
            continue;
          }
          std::shared_ptr<Operation> op;
          auto it = ops_.find(op_ref->id);
          if (it == ops_.end()) {
            op = std::make_shared<Operation>();
            op->id = op_ref->id;
            op->sched_scope = scope.get();
            ops_.emplace(op->id, op);
          } else {
            op = it->second;
            ICHECK_EQ(op->sched_scope, scope.get())
                << "ws_schedule: op '" << op->id
                << "' is scheduled in multiple scopes";
          }
          ICHECK(op->roles.insert(&role).second)
              << "ws_schedule: op '" << op->id
              << "' is referenced more than once in the body of scope '"
              << scope->id << "' by role " << role.name
              << "; a role places an op at most once";
          entries.push_back(op);
        }
        scope->bodies[role.index] = std::move(entries);
      }
    }
    // An unreferenced scope would silently drop its scheduled work; a
    // body whose role never enters the scope would plan barriers that
    // are never emitted.
    for (const auto &scope : scopes_) {
      if (scope->id == kWSRootScopeId) {
        root_scope_ = scope.get();
        continue;
      }
      ICHECK(scope->sched_scope != nullptr)
          << "ws_schedule: scope '" << scope->id
          << "' is never referenced from a parent scope body";
      for (size_t r = 0; r < scope->bodies.size(); ++r) {
        ICHECK(scope->bodies[r].empty() ||
               scope_refs.count({scope.get(), static_cast<int>(r)}))
            << "ws_schedule: scope '" << scope->id << "' has a body for role "
            << roles_[r]->name
            << ", but that role never enters it from the parent scope";
      }
    }
    ICHECK(root_scope_ != nullptr) << "ws_schedule: missing root scope";
  }

  std::shared_ptr<Scope> FindScope(const String &id) const {
    for (const auto &scope : scopes_)
      if (scope->id == id)
        return scope;
    return nullptr;
  }

  PipelineSpec *FindPipeline(const String &name) const {
    for (const auto &pipeline : pipelines_)
      if (pipeline->name == name)
        return pipeline.get();
    return nullptr;
  }

  // Every pipeline synchronizes exactly one scope: resolve it from the
  // sync entries and reject entries spread across scopes. Nesting of
  // multi-bound buffers is validated in PlanVersionedBuffers, once the
  // buffers are resolved.
  void ResolvePipelineScopes() {
    for (const auto &scope : scopes_) {
      for (const std::vector<BodyEntryPtr> &body : scope->bodies) {
        for (const BodyEntryPtr &entry : body) {
          const Synchronization *sync = entry->AsSync();
          if (sync == nullptr)
            continue;
          PipelineSpec &pipeline = *sync->pipeline;
          if (pipeline.sync_scope == nullptr) {
            pipeline.sync_scope = scope.get();
          } else {
            ICHECK_EQ(pipeline.sync_scope, scope.get())
                << "ws_schedule: pipeline '" << pipeline.name
                << "' has sync entries in scopes '" << pipeline.sync_scope->id
                << "' and '" << scope->id
                << "'; a pipeline synchronizes exactly one "
                << "scope — protect the buffer with a second pipeline in the "
                << "outer scope instead (nested pipelines multiply the "
                << "buffer's versions)";
          }
        }
      }
    }
    for (const auto &pipeline : pipelines_) {
      ICHECK(pipeline->sync_scope != nullptr)
          << "ws_schedule: pipeline '" << pipeline->name
          << "' has no sync entries in any scope body";
    }
  }

  // ---- op matching --------------------------------------------------------

  // Walk the kernel body against the schedule. Coverage is two-sided:
  // every declared op and scope must match a statement, and (enforced in
  // MatchOp) every kernel statement must carry an id.
  void MatchKernel() {
    MatchScopeBody(root_scope_, block_->body);
    for (auto &[id, op] : ops_) {
      ICHECK(op->stmt.defined())
          << "ws_schedule: op '" << id
          << "' is scheduled but no statement in the kernel carries this id";
      QueryAccess(op->stmt, &op->access);
      // Buffers read by the op's guard are accesses too.
      if (op->guard.defined())
        QueryAccess(Evaluate(op->guard.value()), &op->access);
    }
    for (const auto &scope : scopes_) {
      ICHECK(!scope->IsRoot() || scope->id == kWSRootScopeId)
          << "ws_schedule: scope '" << scope->id
          << "' has no matching loop in the kernel";
      // The scope's own access set: as its parent's use it is one big op
      // over the loop statement (the hierarchical collapse).
      if (!scope->IsRoot())
        QueryAccess(scope->stmt, &scope->access);
    }
  }

  void MatchScopeBody(Scope *scope, const Stmt &body) {
    for (const Stmt &stmt : BodyStmts(body))
      MatchOp(scope, stmt, Optional<PrimExpr>());
  }

  // The statements of a body: a SeqStmt's list, or the single statement
  // itself. Not recursive — traced TIR does not nest SeqStmts.
  static Array<Stmt> BodyStmts(const Stmt &s) {
    if (const auto *seq = s.as<SeqStmtNode>())
      return seq->seq;
    return {s};
  }

  // A while scope has no extent: phases under it use runtime counters.
  void RecordScopeWhileLoop(Scope *scope, const While &loop) {
    ICHECK(scope->IsRoot())
        << "ws_schedule: scope " << scope->id << " matched twice";
    scope->orig_while = loop;
    scope->stmt = loop;
    MatchScopeBody(scope, loop->body);
  }

  void RecordScopeForLoop(Scope *scope, const For &loop) {
    ICHECK(scope->IsRoot())
        << "ws_schedule: scope " << scope->id << " matched twice";
    // Any sequential loop can be a scope (T.Pipelined is a serial loop
    // with a num_stages annotation; a T.unroll scope keeps its unroll
    // kind).
    ICHECK(loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)
        << "ws_schedule: scope '" << scope->id
        << "' must be a sequential loop (T.Pipelined, T.serial, or "
           "T.unroll); T.Parallel loops are scheduled as single ops";
    // Phases count iterations as (loop_var - min); a non-unit step
    // would break them. TODO: divide by the step.
    ICHECK(loop->HasTrivialStep())
        << "ws_schedule: scope '" << scope->id
        << "' has a non-unit loop step; write the loop with unit step and "
           "scale the indices in the body instead";
    scope->orig_loop = loop;
    scope->stmt = loop;
    MatchScopeBody(scope, loop->body);
  }

  // Consume the id marker as the statement is recorded, so the emitted
  // clones never leak scheduling metadata downstream.
  static Stmt StripOpId(const Stmt &stmt) {
    if (const auto *ev = stmt.as<EvaluateNode>()) {
      const auto *call = ev->value.as<CallNode>();
      if (call && call->annotations.count(kWSOpIdKey)) {
        Map<String, ObjectRef> ann =
            FilterAnnotations(call->annotations, [](const String &key) {
              return key != kWSOpIdKey;
            });
        return Evaluate(Call(call->dtype, call->op, call->args, std::move(ann),
                             call->span));
      }
      return stmt;
    }
    if (stmt.as<ForNode>()) {
      For loop = Downcast<For>(stmt);
      if (loop->annotations.count(kWSOpIdKey)) {
        loop.CopyOnWrite()->annotations =
            FilterAnnotations(loop->annotations, [](const String &key) {
              return key != kWSOpIdKey;
            });
      }
      return loop;
    }
    return stmt;
  }

  void RecordOpStmt(Scope *scope, const String &id, const Stmt &stmt,
                    const Optional<PrimExpr> &guard) {
    auto it = ops_.find(id);
    ICHECK(it != ops_.end()) << "ws_schedule: statement carries ws op id '"
                             << id << "' but the schedule never places it:\n"
                             << stmt;
    Operation &op = *it->second;
    ICHECK(!op.stmt.defined())
        << "ws_schedule: two statements carry ws op id '" << id
        << "'; every scheduled statement needs its own id";
    ICHECK_EQ(scope, op.sched_scope)
        << "ws_schedule: op '" << id << "' lives in scope '" << scope->id
        << "' of the kernel but is scheduled by " << "scope '"
        << op.sched_scope->id << "'";
    op.stmt = StripOpId(stmt);
    if (guard.defined())
      op.guard = guard;
  }

  void MatchOp(Scope *scope, const Stmt &stmt, Optional<PrimExpr> guard) {
    // A `with T.ws_op(id):` wrapper. With a scope id it wraps a while
    // loop and opens that scope; otherwise the wrapped statements become
    // ONE opaque op. The wrapper is consumed here.
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == kWSOpIdKey) {
        String id = ExtractWSOpId(Any(attr->value));
        if (std::shared_ptr<Scope> child = FindScope(id)) {
          const auto *wl = attr->body.as<WhileNode>();
          ICHECK(wl) << "ws_schedule: scope id '" << id << "' on a T.ws_op "
                     << "wrapper must wrap a while loop; serial loops carry "
                     << "the id in their own annotations";
          ICHECK_EQ(child->sched_scope, scope)
              << "ws_schedule: scope '" << id << "' lives in scope '"
              << scope->id << "' of the kernel but is referenced from scope '"
              << child->sched_scope->id << "'";
          if (guard.defined())
            child->guard = guard;
          RecordScopeWhileLoop(child.get(), GetRef<While>(wl));
          return;
        }
        RecordOpStmt(scope, id, attr->body, guard);
        return;
      }
      // A T.annotate_ws_pipeline_depth wrapper: scheduler-only metadata;
      // consume it and match the statements it wraps.
      if (attr->attr_key == kWSPipelineDepthKey) {
        for (const Stmt &sub : BodyStmts(attr->body))
          MatchOp(scope, sub, guard);
        return;
      }
      // Kernel-level metadata (T.use_swizzle,
      // T.annotate_min_blocks_per_sm, ...) wraps the statements that
      // follow it. Record the wrapper, keep matching inside it, and
      // re-wrap the rebuilt kernel body (RebuildBlock).
      ICHECK(scope->id == kWSRootScopeId && !guard.defined())
          << "ws_schedule: AttrStmt '" << attr->attr_key
          << "' must be at the top level of the kernel:\n"
          << stmt;
      metadata_attrs_.push_back(GetRef<AttrStmt>(attr));
      for (const Stmt &sub : BodyStmts(attr->body))
        MatchOp(scope, sub, guard);
      return;
    }

    // An annotated loop: a serial loop whose id names a scope opens
    // that scope; any other annotated loop is one opaque op.
    if (const auto *loop = stmt.as<ForNode>()) {
      if (auto id_ann = loop->annotations.Get(kWSOpIdKey)) {
        String id = ExtractWSOpId(id_ann.value());
        if (std::shared_ptr<Scope> child = FindScope(id)) {
          ICHECK_EQ(child->sched_scope, scope)
              << "ws_schedule: scope '" << id << "' lives in scope '"
              << scope->id << "' of the kernel but is referenced from scope '"
              << child->sched_scope->id << "'";
          if (guard.defined())
            child->guard = guard;
          RecordScopeForLoop(child.get(), GetRef<For>(loop));
          return;
        }
        RecordOpStmt(scope, id, stmt, guard);
        return;
      }
    }

    // A tile op carrying its id in the call annotations.
    if (const auto *ev = stmt.as<EvaluateNode>()) {
      if (const auto *call = ev->value.as<CallNode>()) {
        if (auto id = call->annotations.Get(kWSOpIdKey)) {
          RecordOpStmt(scope, ExtractWSOpId(id.value()), stmt, guard);
          return;
        }
      }
    }

    // An op inside an IfThenElse records the accumulated condition as
    // its guard.
    if (const auto *ite = stmt.as<IfThenElseNode>()) {
      ICHECK(!ite->else_case.defined())
          << "ws_schedule: else branches are not supported yet";
      PrimExpr cond =
          guard.defined() ? (guard.value() && ite->condition) : ite->condition;
      for (const Stmt &sub : BodyStmts(ite->then_case))
        MatchOp(scope, sub, cond);
      return;
    }

    // Every statement of a scheduled kernel must be placed by the
    // schedule; nothing is silently dropped or replicated.
    LOG(FATAL) << "ws_schedule: statement in scope '" << scope->id
               << "' carries no ws op id; tile ops and loops take "
                  "annotations={\"tl.ws_op_id\": ...}, other statements "
                  "(a scalar Bind) are wrapped in T.ws_op(...):\n"
               << stmt;
  }

  // ---- planning -----------------------------------------------------------

  // Give one logical buffer (an allocation, or a view/reshape alias of it)
  // its versioned counterpart with one leading dimension PER BINDING,
  // outermost first — each level replicates the buffer by its own depth
  // and hands one version down. Every binding gets its dimension, extent
  // 1 included, so downstream never distinguishes versioned from
  // unversioned. Views cover their whole allocation (T.view / T.reshape
  // enforce equal sizes), so every alias shares the version strides.
  void EnsureVersionedAlias(const Buffer &buf) {
    if (versioned_.count(buf))
      return;
    const std::vector<PipelineSpec *> &bindings =
        buffer_pipeline_.at(buf->data);
    ObjectPtr<BufferNode> n = make_object<BufferNode>(*buf.get());
    for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
      if (!n->strides.empty()) {
        PrimExpr stride0 = n->strides[0] * n->shape[0];
        n->strides.insert(n->strides.begin(), std::move(stride0));
      }
      n->shape.insert(n->shape.begin(),
                      IntImm(DataType::Int(32), (*it)->depth));
    }
    versioned_[buf] = Buffer(std::move(n));
  }

  void PlanVersionedBuffers() {
    for (const auto &pipeline : pipelines_) {
      for (const Buffer &buf : pipeline->buffers)
        buffer_pipeline_[buf->data].push_back(pipeline.get());
      pipeline->full =
          CreateMBarrierBuffer(pipeline->name + "_full", pipeline->depth);
      pipeline->empty =
          CreateMBarrierBuffer(pipeline->name + "_empty", pipeline->depth);
    }
    // A multi-bound storage's pipelines must form a strictly nested scope
    // chain; sort them outermost first so version indices compose
    // outer-major.
    for (auto &[storage, bindings] : buffer_pipeline_) {
      std::stable_sort(bindings.begin(), bindings.end(),
                       [](const PipelineSpec *a, const PipelineSpec *b) {
                         return a->sync_scope->Depth() < b->sync_scope->Depth();
                       });
      for (size_t i = 1; i < bindings.size(); ++i) {
        const PipelineSpec &outer = *bindings[i - 1];
        const PipelineSpec &inner = *bindings[i];
        ICHECK(outer.sync_scope != inner.sync_scope &&
               inner.sync_scope->IsNestedIn(outer.sync_scope))
            << "ws_schedule: pipelines '" << outer.name << "' (scope '"
            << outer.sync_scope->id << "') and '" << inner.name << "' (scope '"
            << inner.sync_scope->id
            << "') both protect one storage but their scopes are not "
            << "strictly nested";
      }
    }
    for (const auto &pipeline : pipelines_)
      for (const Buffer &buf : pipeline->buffers)
        EnsureVersionedAlias(buf);
  }

  // ---- def/use chains -------------------------------------------------------

  // Resolve one access set's defining pipelines from the storage map.
  void ResolveAccessDefs(AccessSet *access) {
    for (auto *side : {&access->writes, &access->reads}) {
      for (auto &[buf, defs] : *side) {
        auto it = buffer_pipeline_.find(buf->data);
        if (it != buffer_pipeline_.end()) {
          defs = it->second;
          EnsureVersionedAlias(buf);
        }
      }
    }
  }

  // Join the use sets of the innermost binding containing the
  // operation's position — one rule for leaves and scopes alike (a
  // scope's detector-summarized accesses make it the enclosing
  // binding's opaque producer/consumer). An access with no containing
  // binding is rejected by VerifySpanCoverage.
  void RegisterUses(Operation *op) {
    for (auto &[buf, defs] : op->access.writes) {
      PipelineSpec *owner = InnermostContainingBinding(defs, op->sched_scope);
      if (owner != nullptr)
        owner->producers.insert(op);
    }
    for (auto &[buf, defs] : op->access.reads) {
      PipelineSpec *owner = InnermostContainingBinding(defs, op->sched_scope);
      if (owner != nullptr)
        owner->consumers.insert(op);
    }
  }

  // Build the pipelines' use sets HIERARCHICALLY: every operation —
  // leaf or scope — joins the innermost binding whose scope contains
  // its position. An enclosing binding sees a nested binding's work
  // only as the collapsed scope operation (a role that merely observed
  // the inner writes still publishes them with its arrive); inner
  // synchronization never penetrates outward. Leaf-only rules: an op's
  // written buffers must share ONE owning binding (one completion
  // signal), and a multi-role op must touch no pipeline buffers.
  void BuildUseChains() {
    for (auto &[id, op_ptr] : ops_) {
      Operation &op = *op_ptr;
      Buffer write_buf;
      ResolveAccessDefs(&op.access);
      RegisterUses(&op);
      for (auto &[buf, defs] : op.access.writes) {
        PipelineSpec *owner = InnermostContainingBinding(defs, op.sched_scope);
        if (owner == nullptr)
          continue; // unprotected, or rejected by VerifySpanCoverage
        if (op.write_def == nullptr) {
          op.write_def = owner;
          write_buf = buf;
        } else {
          ICHECK_EQ(op.write_def, owner)
              << "ws_schedule: op '" << id << "' writes " << write_buf->name
              << " of pipeline '" << op.write_def->name << "' and " << buf->name
              << " of pipeline '" << owner->name
              << "'; an op's synchronization can only trigger one pipeline, "
                 "so split the op";
        }
      }
      if (op.roles.size() > 1) {
        PipelineSet defs = op.access.Defs();
        ICHECK(defs.empty())
            << "ws_schedule: op '" << id << "' is placed by " << op.roles.size()
            << " roles but touches buffer(s) of " << "pipeline '"
            << (*defs.begin())->name
            << "'; only ops touching no pipeline buffers may be duplicated "
               "across roles";
        for (const auto &[buf, def] : op.access.writes) {
          ICHECK((IsLocalBuffer(buf, true) || IsFragmentBuffer(buf)))
              << "ws_schedule: op '" << id << "' is placed by "
              << op.roles.size() << " roles but writes non-local buffer '"
              << buf->name << "'; every role would repeat the write";
        }
      }
    }
    for (const auto &scope : scopes_) {
      ResolveAccessDefs(&scope->access);
      RegisterUses(scope.get()); // the root has no position: a no-op
    }
    // A scope guard or while condition must be uniform across roles: it may
    // read role-private locals or storage no op writes.
    VarSet written;
    for (const auto &[id, op] : ops_)
      for (const auto &[buf, def] : op->access.writes)
        written.insert(buf->data);
    for (const auto &scope : scopes_) {
      auto check_uniform = [&](const PrimExpr &expr, const char *what) {
        AccessSet access;
        QueryAccess(Evaluate(expr), &access);
        for (const auto &[buf, def] : access.reads) {
          ICHECK(!buffer_pipeline_.count(buf->data))
              << "ws_schedule: the " << what << " of scope '" << scope->id
              << "' touches pipeline buffer " << buf->name << "; it must "
              << "be uniform across roles";
          ICHECK((IsLocalBuffer(buf, true) || IsFragmentBuffer(buf)) ||
                 !written.count(buf->data))
              << "ws_schedule: the " << what << " of scope '" << scope->id
              << "' reads buffer '" << buf->name << "' that scheduled ops "
              << "write; its value could diverge between roles";
        }
      };
      if (scope->guard.defined())
        check_uniform(scope->guard.value(), "guard");
      if (scope->orig_while.defined())
        check_uniform(scope->orig_while->condition, "condition");
    }
  }

  // ---- op atoms -------------------------------------------------------------

  void BuildOpAtoms() {
    for (auto &[id, op_ptr] : ops_) {
      Operation &op = *op_ptr;
      // Classify every call once: the direct tile-op call sets the op's
      // atom, while an asynchronous instruction nested below a compound
      // statement (an op-node loop, a T.ws_op group) is rejected — its
      // barrier could not be wired. Either way each call contributes its
      // async-proxy accesses. (Locals because C++17 lambdas cannot
      // capture structured bindings.)
      const auto *ev = op.stmt.as<EvaluateNode>();
      const CallNode *direct = ev ? ev->value.as<CallNode>() : nullptr;
      const String &op_id = id;
      PostOrderVisit(op.stmt, [&](const ObjectRef &node) {
        const auto *call = node.as<CallNode>();
        if (!call || call->op.same_as(region()))
          return;
        TileOperator tile_op = ParseOperator(GetRef<Call>(call));
        OpAtom atom = ClassifyTileOp(tile_op, target_);
        if (call == direct) {
          op.atom = atom;
        } else {
          ICHECK(atom == OpAtom::kSync)
              << "ws_schedule: op '" << op_id << "' nests an asynchronous "
              << "instruction inside a compound statement; make it a "
              << "directly scheduled op so its barrier can be wired:\n"
              << op.stmt;
        }
        op.uses_async_proxy =
            op.uses_async_proxy || UsesAsyncProxy(GetRef<Call>(call), target_);
      });
    }
    // Fold the scope summaries, children first.
    std::vector<Scope *> by_depth;
    for (const auto &scope : scopes_)
      by_depth.push_back(scope.get());
    std::stable_sort(
        by_depth.begin(), by_depth.end(),
        [](const Scope *a, const Scope *b) { return a->Depth() > b->Depth(); });
    for (Scope *scope : by_depth) {
      scope->role_residues.assign(roles_.size(), {});
      for (size_t r = 0; r < roles_.size(); ++r) {
        Scope::Residue &folded = scope->role_residues[r];
        for (const BodyEntryPtr &entry : scope->bodies[r]) {
          if (entry->AsSync())
            continue;
          scope->uses_async_proxy = scope->uses_async_proxy ||
                                    entry->AsOperation()->TouchesAsyncProxy();
          if (const Scope *child = entry->AsScope()) {
            folded.tcgen05 = folded.tcgen05 || child->role_residues[r].tcgen05;
            folded.cp_async =
                folded.cp_async || child->role_residues[r].cp_async;
          } else {
            OpAtom atom = entry->AsOperation()->atom;
            folded.tcgen05 = folded.tcgen05 || atom == OpAtom::kTcgen05Gemm;
            folded.cp_async = folded.cp_async || atom == OpAtom::kCpAsyncCopy;
          }
        }
      }
    }
  }

  // Fuse the full-side arrive into each commit cycle's last TMA copy:
  // init(1) plus one elected arrive replace the per-thread arrives.
  // Requires every cycle of the signaling role to end with an unguarded
  // copy in the same body — a guarded copy may skip its arrive; a copy
  // inside a child scope would arrive once per child iteration.
  bool TryFuseTmaArrival(const PipelineSpec *p, const BarrierSidePlan &plan) {
    // All of p's brackets live in its one scope; a producing entry is
    // either a copy owned by p or a collapsed scope (a use of p).
    std::vector<Operation *> cycle_copies;
    Operation *last = nullptr; // last pipeline-writing op of the open cycle
    bool open = false;
    for (const BodyEntryPtr &entry :
         p->sync_scope->bodies[plan.signal_role->index]) {
      if (const Synchronization *sync = entry->AsSync()) {
        if (sync->pipeline != p)
          continue;
        if (sync->kind.IsProducerAcquire()) {
          open = true;
          last = nullptr;
        } else if (sync->kind.IsProducerCommit()) {
          if (last == nullptr || last->guard.defined())
            return false;
          cycle_copies.push_back(last);
          open = false;
        }
        continue;
      }
      if (!open)
        continue;
      Operation *op = entry->AsOperation();
      if (op->AsScope()) {
        if (p->producers.count(op))
          return false;
        continue;
      }
      if (op->write_def == p)
        last = op;
    }
    for (Operation *copy : cycle_copies)
      copy->fused_arrive = true;
    return true;
  }

  // Fill each pipeline's two BarrierSidePlans: find the unique
  // signaling role of each side and derive the count from its uses.
  void PlanArriveCounts() {
    // The signaling role of a side is the role holding its commit /
    // release entries (all in the pipeline's one scope); it must be
    // unique.
    for (const auto &pipeline : pipelines_) {
      for (const auto &role : roles_) {
        for (const BodyEntryPtr &entry :
             pipeline->sync_scope->bodies[role->index]) {
          const Synchronization *sync = entry->AsSync();
          if (sync == nullptr || sync->pipeline != pipeline.get() ||
              sync->kind.IsWait())
            continue;
          bool producer_side = sync->kind.IsProducerCommit();
          BarrierSidePlan &plan =
              producer_side ? pipeline->full_plan : pipeline->empty_plan;
          if (plan.signal_role == nullptr) {
            plan.signal_role = role.get();
          } else {
            ICHECK_EQ(plan.signal_role, role.get())
                << "ws_schedule: pipeline " << pipeline->name
                << " is committed/released by multiple roles";
          }
        }
      }
    }
    // An asynchronous op's completion signal is wired to a barrier of the
    // pipeline protecting its destination: a copy's transaction / deferred
    // arrive rides the full side, a tcgen05 commit rides whichever side
    // counted the op. The op must therefore appear in the use set of a
    // side its own role signals, or the protocol never observes it
    // finishing (an unmatched expect_tx hangs the barrier at run time).
    for (const auto &[id, op] : ops_) {
      if (op->atom == OpAtom::kSync)
        continue;
      if (op->write_def == nullptr) {
        // Copies fall back to their own completion machinery
        // (ConvertAtomCall); a tcgen05 gemm has none.
        ICHECK(op->atom != OpAtom::kTcgen05Gemm)
            << "ws_schedule: tcgen05 gemm '" << id << "' accumulates into an "
            << "unprotected buffer; nothing would observe its completion";
        continue;
      }
      const PipelineSpec &pipeline = *op->write_def;
      if (pipeline.full_plan.signal_role == nullptr ||
          pipeline.empty_plan.signal_role == nullptr)
        continue; // "has no producer / consumer" is reported below
      bool counted = false;
      for (bool producer_side : {true, false}) {
        const BarrierSidePlan &plan =
            producer_side ? pipeline.full_plan : pipeline.empty_plan;
        if (op->PlacedBy(*plan.signal_role) &&
            pipeline.Uses(producer_side).count(op.get()))
          counted = true;
      }
      ICHECK(counted)
          << "ws_schedule: asynchronous op '" << id << "' writes pipeline '"
          << pipeline.name << "', but no barrier side signaled by its role "
          << "counts the op; the protocol never observes it finishing";
    }

    for (const auto &pipeline_ptr : pipelines_) {
      PipelineSpec &pipeline = *pipeline_ptr;
      for (bool producer_side : {true, false}) {
        BarrierSidePlan &plan =
            producer_side ? pipeline.full_plan : pipeline.empty_plan;
        if (plan.signal_role == nullptr)
          continue; // reported below
        const RoleSpec &signal_role = *plan.signal_role;
        // Selects the signaling role's uses (leaf ops have exactly one
        // placing role; a collapsed scope answers per role).
        for (const Operation *op : pipeline.Uses(producer_side)) {
          if (!op->PlacedBy(signal_role))
            continue;
          // A scope contributes every residue it holds for the role.
          if (const Scope *scope = op->AsScope()) {
            const Scope::Residue &res = scope->role_residues[signal_role.index];
            plan.has_tcgen05_arrival = plan.has_tcgen05_arrival || res.tcgen05;
            plan.has_cpasync_arrival = plan.has_cpasync_arrival || res.cp_async;
            if (!res.tcgen05 && !res.cp_async)
              plan.has_thread_arrive = true;
            continue;
          }
          switch (op->AtomFor(signal_role)) {
          case OpAtom::kTmaCopy:
            plan.has_transaction = true;
            break;
          case OpAtom::kTcgen05Gemm:
            plan.has_tcgen05_arrival = true;
            break;
          case OpAtom::kCpAsyncCopy:
            plan.has_cpasync_arrival = true;
            break;
          case OpAtom::kSync:
            plan.has_thread_arrive = true;
            break;
          }
        }
        // Transactions alone cannot complete a phase. A pure-TMA producer
        // side fuses one elected arrive into each cycle's last copy (the
        // legacy protocol); any other side with TMA copies keeps the
        // per-thread arrive.
        if (plan.has_transaction && !plan.has_thread_arrive &&
            !plan.has_cpasync_arrival && !plan.has_tcgen05_arrival &&
            producer_side && TryFuseTmaArrival(&pipeline, plan))
          plan.fused_tma_arrive = true;
        else if (plan.has_transaction)
          plan.has_thread_arrive = true;
        // The arrive publishes the signaling role's generic-proxy writes
        // to the pipeline's shared buffers; a fence is required when the
        // observing side touches them through the async proxy (see
        // BarrierSidePlan).
        bool generic_writes = false;
        for (const Operation *op : pipeline.producers) {
          if (generic_writes)
            break;
          if (!op->PlacedBy(signal_role) ||
              op->AtomFor(signal_role) != OpAtom::kSync)
            continue;
          for (const auto &[buf, defs] : op->access.writes) {
            bool of_pipeline = false;
            for (const PipelineSpec *def : defs)
              of_pipeline = of_pipeline || def == &pipeline;
            if (of_pipeline && !IsTmemBuffer(buf)) {
              generic_writes = true;
              break;
            }
          }
        }
        if (generic_writes) {
          for (const Operation *op : pipeline.Uses(!producer_side)) {
            if (op->TouchesAsyncProxy()) {
              plan.needs_proxy_fence = true;
              break;
            }
          }
        }
        // tcgen05_mma_arrive elects one lane PER WARP; a wider role
        // would arrive once per warp against a count of one.
        ICHECK(!plan.has_tcgen05_arrival ||
               signal_role.warp_hi - signal_role.warp_lo == 1)
            << "ws_schedule: role " << signal_role.name << " signals pipeline "
            << pipeline.name << " through a tcgen05 commit but spans "
            << (signal_role.warp_hi - signal_role.warp_lo)
            << " warps; tcgen05 issue and commit require a single-warp role";
        plan.count = (plan.has_tcgen05_arrival ? 1 : 0) +
                     (plan.fused_tma_arrive ? 1 : 0) +
                     ((plan.has_cpasync_arrival ? 1 : 0) +
                      (plan.has_thread_arrive ? 1 : 0)) *
                         static_cast<int64_t>(signal_role.NumThreads());
      }
    }
    for (const auto &pipeline : pipelines_) {
      ICHECK_GT(pipeline->full_plan.count, 0)
          << "ws_schedule: pipeline " << pipeline->name << " has no producer";
      ICHECK_GT(pipeline->empty_plan.count, 0)
          << "ws_schedule: pipeline " << pipeline->name << " has no consumer";
    }
  }

  // ---- schedule verification ------------------------------------------------
  //
  // A broken schedule would hang or race on the GPU; these checks reject
  // it at compile time instead.

  // Per role: sync brackets pair up within one scope body, a role is
  // never both producer and consumer of a pipeline, and every access
  // runs inside an open span of the innermost binding whose scope
  // contains the op. A producer-flavored role additionally holds every
  // enclosing binding (its nested writes publish through the outer
  // commit); a consumer's nested accesses ride the nested pipeline plus
  // same-role program order.
  void VerifySpanCoverage() const {
    for (const auto &role_ptr : roles_) {
      const RoleSpec &role = *role_ptr;
      // pipeline -> role is producer, resolved up front from the
      // pipeline's one scope: an op in a nested scope may need its
      // role's flavor for a pipeline whose entries only appear later in
      // an outer body.
      std::map<const PipelineSpec *, bool, PipelineOrder> flavor;
      for (const auto &p : pipelines_) {
        for (const BodyEntryPtr &entry : p->sync_scope->bodies[role.index]) {
          const Synchronization *sync = entry->AsSync();
          if (sync == nullptr || sync->pipeline != p.get())
            continue;
          auto [fit, first] = flavor.emplace(p.get(), sync->kind.IsProducer());
          ICHECK(first || fit->second == sync->kind.IsProducer())
              << "ws_schedule: role " << role.name
              << " is both a producer and a consumer of pipeline '" << p->name
              << "'; the flavors are the two parties of the handshake, and "
                 "a role handing data to itself needs no pipeline";
        }
      }
      PipelineSet open;   // union of all enclosing bodies' spans
      PipelineSet synced; // this role's sync entries already passed:
                          // afterwards its phase names the NEXT cycle
                          // (see OpRewriter::BindingPhase)
      std::function<void(const Scope &)> walk = [&](const Scope &scope) {
        PipelineSet local; // spans opened in THIS body
        for (const BodyEntryPtr &entry : scope.bodies[role.index]) {
          if (const Synchronization *sync = entry->AsSync()) {
            const PipelineSpec *p = sync->pipeline;
            synced.insert(p);
            if (sync->kind.IsWait()) {
              ICHECK(!open.count(p))
                  << "ws_schedule: pipeline '" << p->name << "' acquired twice "
                  << "without an intervening commit/release in role "
                  << role.name;
              open.insert(p);
              local.insert(p);
            } else {
              ICHECK(local.count(p))
                  << "ws_schedule: commit/release of pipeline '" << p->name
                  << "' in role " << role.name
                  << " has no matching acquire/wait in the same scope body";
              open.erase(p);
              local.erase(p);
            }
            continue;
          }
          if (const Scope *child = entry->AsScope()) {
            walk(*child);
            continue;
          }
          const Operation &op = *entry->AsOperation();
          auto check_access = [&](const Buffer &buf,
                                  const std::vector<PipelineSpec *> &defs,
                                  const char *how) {
            if (defs.empty())
              return;
            // The innermost binding whose scope contains the op owns the
            // alternation at the op's level.
            const PipelineSpec *innermost =
                InnermostContainingBinding(defs, op.sched_scope);
            ICHECK(innermost != nullptr)
                << "ws_schedule: op '" << op.id << "' in role " << role.name
                << " " << how << " " << buf->name
                << " outside the scope of every pipeline protecting it";
            ICHECK(open.count(innermost))
                << "ws_schedule: op '" << op.id << "' in role " << role.name
                << " " << how << " " << buf->name
                << " outside an open span of pipeline '" << innermost->name
                << "'; bracket the op with producer_acquire/producer_commit "
                << "or consumer_wait/consumer_release (the stage may be "
                << "concurrently overwritten or still read by an "
                << "asynchronous op)";
            for (const PipelineSpec *def : defs) {
              if (def == innermost)
                continue;
              if (!op.sched_scope->IsNestedIn(def->sync_scope)) {
                // Resolved to the adjacent version (a read the
                // last-completed slot, a write the next-produced one);
                // needs the binding's scope below the access's.
                ICHECK(def->depth == 1 ||
                       def->sync_scope->IsNestedIn(op.sched_scope))
                    << "ws_schedule: op '" << op.id << "' in role " << role.name
                    << " " << how << " " << buf->name
                    << " outside the scope of pipeline '" << def->name
                    << "' (depth " << def->depth
                    << "), which is not nested below the op's scope; no "
                    << "version is adjacent there";
                continue;
              }
              auto fit = flavor.find(def);
              if (fit != flavor.end() && fit->second) {
                ICHECK(open.count(def))
                    << "ws_schedule: op '" << op.id << "' in role " << role.name
                    << " " << how << " " << buf->name
                    << " outside an open span of enclosing pipeline '"
                    << def->name << "', which this role produces; "
                    << "bracket the nested scope reference with "
                    << "producer_acquire/producer_commit";
                continue;
              }
              // Consumer side, span not held: the emitted access derives
              // the slot from the role's own phase, which names the
              // in-production version only until the role's sync entries
              // of the pipeline run.
              if (def->depth > 1 && !open.count(def)) {
                ICHECK(!synced.count(def))
                    << "ws_schedule: op '" << op.id << "' in role " << role.name
                    << " " << how << " " << buf->name << " of pipeline '"
                    << def->name << "' (depth " << def->depth
                    << ") after this role's sync "
                    << "entries; the role's phase no longer names the "
                    << "in-production version — move the access before the "
                    << "wait or hold the span across it";
              }
            }
          };
          for (const auto &[buf, defs] : op.access.writes)
            check_access(buf, defs, "writes");
          for (const auto &[buf, defs] : op.access.reads)
            check_access(buf, defs, "reads");
        }
        ICHECK(local.empty())
            << "ws_schedule: role " << role.name << " leaves pipeline '"
            << (*local.begin())->name
            << "' acquired at the end of a scope body; every acquire/wait "
               "must be paired with a commit/release in the same body";
      };
      walk(*root_scope_);
    }
  }

  // Within every loop scope, a pipeline's producer and consumer sides
  // must cycle equally often per iteration; an imbalance drifts the
  // full/empty parity apart every trip. Balance also keeps the deadlock
  // model's bounded unrolling faithful.
  void VerifyCycleBalance() const {
    for (const auto &scope : scopes_) {
      if (scope->IsRoot())
        continue; // the root runs once; the deadlock model covers it exactly
      std::map<const PipelineSpec *, std::array<int, 2>, PipelineOrder>
          cycles; // [consumer, producer]
      for (const auto &body : scope->bodies) {
        for (const BodyEntryPtr &entry : body) {
          const Synchronization *sync = entry->AsSync();
          if (sync && sync->kind.IsCommit())
            cycles[sync->pipeline][sync->kind.IsProducer() ? 1 : 0]++;
        }
      }
      for (const auto &[pipeline, sides] : cycles) {
        ICHECK_EQ(sides[1], sides[0])
            << "ws_schedule: pipeline '" << pipeline->name << "' cycles "
            << sides[1] << " time(s) on the producer side but " << sides[0]
            << " time(s) on the consumer side per iteration of " << "scope '"
            << scope->id << "'; the full/empty parity diverges "
            << "as the loop advances — cycle both sides equally often "
            << "within the scope";
      }
    }
  }

  // Flatten one role's sync entries into emitted-code order: loops are
  // modeled for `loop_trips` iterations, and the stage deltas reproduce
  // the software-pipelined reordering (an entry at delta d executes at
  // steps d .. d + trips - 1).
  std::vector<Synchronization> FlattenRoleEvents(const RoleSpec &role,
                                                 int loop_trips) const {
    std::vector<Synchronization> events;
    std::function<void(const Scope &)> emit = [&](const Scope &scope) {
      const std::vector<BodyEntryPtr> &entries = scope.bodies[role.index];
      if (entries.empty())
        return;
      StagePlan plan = PlanStages(role, entries);
      int trips = scope.IsRoot() ? 1 : loop_trips;
      int steps = scope.IsRoot() ? 1 : trips + plan.shift;
      for (int t = 0; t < steps; ++t) {
        for (size_t i = 0; i < entries.size(); ++i) {
          if (!scope.IsRoot() &&
              (t < plan.delta[i] || t >= plan.delta[i] + trips))
            continue; // outside this entry's step window
          if (const Scope *child = entries[i]->AsScope())
            emit(*child);
          else if (const Synchronization *sync = entries[i]->AsSync())
            events.push_back(*sync);
        }
      }
    };
    emit(*root_scope_);
    return events;
  }

  // Execute all roles' sync events under the mbarrier parity model: a
  // role's (k+1)-th wait needs k+1 total commits, its (k+1)-th acquire
  // needs depth + total releases >= k+1, commits/releases never block.
  // Enabled events never become disabled, so one greedy execution
  // suffices; a role still blocked at quiescence is a real hang.
  void VerifyDeadlockFree() const {
    // A capacity deadlock (a producer run-ahead exhausting a pipeline's
    // depth) surfaces within depth+1 cycles, so model each loop for
    // max-depth+1 iterations.
    int trips = 2;
    for (const auto &pipeline : pipelines_)
      trips = std::max(trips, pipeline->depth + 1);
    int n_roles = static_cast<int>(roles_.size());
    std::vector<std::vector<Synchronization>> events(n_roles);
    for (int r = 0; r < n_roles; ++r)
      events[r] = FlattenRoleEvents(*roles_[r], trips);

    std::vector<int> cursors(n_roles, 0);
    // commits[p] / releases[p]: totals across all roles so far; waits and
    // acquires are counted per (role, pipeline) observer.
    std::map<const PipelineSpec *, int> commits, releases;
    std::vector<std::map<const PipelineSpec *, int>> waits(n_roles),
        acquires(n_roles);

    auto enabled = [&](int r) {
      const Synchronization &e = events[r][cursors[r]];
      if (e.kind.IsConsumerWait())
        return waits[r][e.pipeline] < commits[e.pipeline];
      if (e.kind.IsProducerAcquire())
        return acquires[r][e.pipeline] <
               e.pipeline->depth + releases[e.pipeline];
      return true; // commit / release never block
    };

    bool progressed = true;
    while (progressed) {
      progressed = false;
      for (int r = 0; r < n_roles; ++r) {
        while (cursors[r] < static_cast<int>(events[r].size()) && enabled(r)) {
          const Synchronization &e = events[r][cursors[r]];
          if (e.kind.IsProducerAcquire())
            acquires[r][e.pipeline] += 1;
          else if (e.kind.IsProducerCommit())
            commits[e.pipeline] += 1;
          else if (e.kind.IsConsumerWait())
            waits[r][e.pipeline] += 1;
          else
            releases[e.pipeline] += 1;
          cursors[r] += 1;
          progressed = true;
        }
      }
    }

    std::ostringstream blocked;
    bool deadlocked = false;
    for (int r = 0; r < n_roles; ++r) {
      if (cursors[r] >= static_cast<int>(events[r].size()))
        continue;
      deadlocked = true;
      const Synchronization &e = events[r][cursors[r]];
      blocked << "\n  role " << roles_[r]->name << " blocked at " << e.kind
              << "(\"" << e.pipeline->name << "\") after " << cursors[r]
              << " of " << events[r].size() << " sync events";
    }
    ICHECK(!deadlocked)
        << "ws_schedule: schedule deadlocks: the roles' sync events reach a "
           "state where every unfinished role is blocked (mbarrier parity "
           "model; loops modeled at "
        << trips << " iterations):" << blocked.str();
  }

  // Scalar Binds inside ops define vars that later ops (or their guards) may
  // use. Each role re-evaluates only the ops placed in it, so a use is
  // well-defined only when the defining op runs in the same role, earlier,
  // unguarded (a guarded definition would sit in its own `if` scope), and —
  // under a stage shift — in the same software-pipeline step, or the unrolled
  // prologue/epilogue would split the definition from the use.
  void VerifyVarDefUse() const {
    std::unordered_map<Var, const Operation *, ObjectPtrHash, ObjectPtrEqual>
        def_op;
    for (const auto &[id, op] : ops_) {
      const Operation *op_ptr = op.get();
      PostOrderVisit(op->stmt, [&](const ObjectRef &node) {
        if (const auto *bind = node.as<BindNode>()) {
          auto [it, inserted] = def_op.emplace(bind->var, op_ptr);
          ICHECK(inserted || it->second == op_ptr)
              << "ws_schedule: var '" << bind->var->name_hint
              << "' is bound by ops '" << it->second->id << "' and '"
              << op_ptr->id << "'";
        }
      });
    }
    if (def_op.empty())
      return;

    // The vars an op (or a guard expression) uses that other ops define.
    auto scan_uses = [&](const ObjectRef &root, const Operation *self,
                         std::vector<Var> *uses) {
      PostOrderVisit(root, [&](const ObjectRef &node) {
        if (const auto *v = node.as<VarNode>()) {
          Var var = GetRef<Var>(v);
          auto it = def_op.find(var);
          if (it == def_op.end() || it->second == self)
            return;
          auto seen =
              std::find_if(uses->begin(), uses->end(),
                           [&](const Var &u) { return u.same_as(var); });
          if (seen == uses->end())
            uses->push_back(var);
        }
      });
    };

    for (const auto &role_ptr : roles_) {
      const RoleSpec &role = *role_ptr;
      auto check_use = [&](const String &where, const Var &v,
                           const VarSet &defined) {
        const Operation *def = def_op.at(v);
        ICHECK(defined.count(v))
            << "ws_schedule: " << where << " in role " << role.name
            << " uses var '" << v->name_hint << "' defined by op '" << def->id
            << "', which does not run earlier in that role";
        // TODO: legalize guarded scalar definitions instead of rejecting
        // them — when every use shares the definition's guard and stays in
        // one contiguous guarded run, the emitted `if` already scopes them
        // together; escaping uses need the scalar lowered to a role-local
        // local.var (allocation outside the guard, guarded store, loads at
        // the uses).
        ICHECK(!def->guard.defined())
            << "ws_schedule: op '" << def->id << "' defines var '"
            << v->name_hint << "' used by " << where
            << " but runs under a source guard; the definition would not be "
               "visible outside its branch";
      };

      // Defs are C-scoped: a child scope sees the defs accumulated so far,
      // but its own defs do not escape back to the parent body.
      std::function<void(const Scope &, VarSet)> walk = [&](const Scope &scope,
                                                            VarSet defined) {
        const std::vector<BodyEntryPtr> &entries = scope.bodies[role.index];
        if (entries.empty())
          return;
        StagePlan plan = PlanStages(role, entries);
        std::map<String, int> body_delta;
        for (size_t i = 0; i < entries.size(); ++i) {
          if (!entries[i]->AsSync() && !entries[i]->AsScope())
            body_delta[entries[i]->AsOperation()->id] = plan.delta[i];
        }
        for (size_t i = 0; i < entries.size(); ++i) {
          if (entries[i]->AsSync())
            continue;
          if (const Scope *child = entries[i]->AsScope()) {
            std::vector<Var> uses;
            if (child->guard.defined())
              scan_uses(child->guard.value(), nullptr, &uses);
            if (child->orig_while.defined())
              scan_uses(child->orig_while->condition, nullptr, &uses);
            for (const Var &v : uses)
              check_use("scope '" + std::string(child->id) + "'", v, defined);
            walk(*child, defined);
            continue;
          }
          const Operation &op = *entries[i]->AsOperation();
          std::vector<Var> uses;
          scan_uses(op.stmt, &op, &uses);
          if (op.guard.defined())
            scan_uses(op.guard.value(), &op, &uses);
          for (const Var &v : uses) {
            check_use("op '" + std::string(op.id) + "'", v, defined);
            auto dit = body_delta.find(def_op.at(v)->id);
            if (plan.shift > 0 && dit != body_delta.end()) {
              ICHECK_EQ(dit->second, plan.delta[i])
                  << "ws_schedule: op '" << op.id << "' uses var '"
                  << v->name_hint << "' defined by op '" << def_op.at(v)->id
                  << "' at a different stage; the "
                  << "unrolled prologue/epilogue steps would split the "
                  << "definition from this use";
            }
          }
          PostOrderVisit(op.stmt, [&](const ObjectRef &node) {
            if (const auto *bind = node.as<BindNode>())
              defined.insert(bind->var);
          });
        }
      };
      walk(*root_scope_, {});
    }
  }

  void VerifySchedule() {
    VerifySpanCoverage();
    VerifyCycleBalance();
    VerifyDeadlockFree();
    VerifyVarDefUse();
  }

  // ---- stage analysis -------------------------------------------------------

  // Pipelines transitively touched by a role's entries under a scope.
  PipelineSet ScopePipelines(const RoleSpec &role, const Scope &scope) const {
    PipelineSet result;
    for (const BodyEntryPtr &entry : scope.bodies[role.index]) {
      if (entry->AsSync())
        continue;
      if (const Scope *child = entry->AsScope()) {
        PipelineSet pipelines = ScopePipelines(role, *child);
        result.insert(pipelines.begin(), pipelines.end());
      } else {
        PipelineSet defs = entry->AsOperation()->access.Defs();
        result.insert(defs.begin(), defs.end());
      }
    }
    return result;
  }

  // Per-entry stage offsets for one (role, scope) body. delta = stage -
  // s_min: an entry at delta d executes logical iteration (i - d) at
  // step i. A sync entry uses its own stage; an op or child scope takes
  // the stage of the open spans covering its pipelines (which must
  // agree); entries under no open span run at delta 0.
  struct StagePlan {
    std::vector<int> delta; // per entry
    int shift = 0;          // max delta; unrolled steps on each side
  };

  StagePlan PlanStages(const RoleSpec &role,
                       const std::vector<BodyEntryPtr> &entries) const {
    StagePlan plan;
    plan.delta.assign(entries.size(), 0);

    int s_min = std::numeric_limits<int>::max();
    for (const BodyEntryPtr &entry : entries) {
      if (const Synchronization *sync = entry->AsSync())
        s_min = std::min(s_min, sync->stage);
    }
    if (s_min == std::numeric_limits<int>::max())
      return plan; // no syncs in this body

    // pipeline -> stage of its open span; only the pair's stage
    // agreement still needs checking here.
    std::map<const PipelineSpec *, int, PipelineOrder> pipeline_stage;
    auto span_delta = [&](const PipelineSet &touched,
                          const String &what) -> int {
      int stage = std::numeric_limits<int>::min();
      for (const auto &[pipeline, open_stage] : pipeline_stage) {
        if (!touched.count(pipeline))
          continue;
        if (stage == std::numeric_limits<int>::min()) {
          stage = open_stage;
        } else {
          ICHECK_EQ(stage, open_stage)
              << "ws_schedule: " << what << " in role " << role.name
              << " touches pipelines whose open spans sit at different "
                 "stages; split the op or align the stages";
        }
      }
      return stage == std::numeric_limits<int>::min() ? s_min : stage;
    };

    for (size_t i = 0; i < entries.size(); ++i) {
      if (const Synchronization *sync = entries[i]->AsSync()) {
        plan.delta[i] = sync->stage - s_min;
        if (sync->kind.IsWait()) {
          pipeline_stage[sync->pipeline] = sync->stage;
        } else {
          // Bracket verified: the matching open exists in this body.
          auto oit = pipeline_stage.find(sync->pipeline);
          ICHECK_EQ(oit->second, sync->stage)
              << "ws_schedule: acquire/wait of pipeline '"
              << sync->pipeline->name << "' in role " << role.name
              << " is at stage " << oit->second
              << " but its commit/release is at stage " << sync->stage
              << "; pairs must share one stage";
          pipeline_stage.erase(oit);
        }
      } else if (const Scope *child = entries[i]->AsScope()) {
        plan.delta[i] =
            span_delta(ScopePipelines(role, *child), child->id) - s_min;
      } else {
        const Operation &op = *entries[i]->AsOperation();
        plan.delta[i] = span_delta(op.access.Defs(), op.id) - s_min;
      }
    }
    // Bracket verified: every span opened in this body has closed.

    for (int d : plan.delta)
      plan.shift = std::max(plan.shift, d);
    return plan;
  }

  // ---- emission -----------------------------------------------------------

  struct LoopLevel {
    PrimExpr iter;   // logical iteration of the current entry at this level
    PrimExpr extent; // ORIGINAL loop extent (cycles per outer iteration)
    PrimExpr min;    // ORIGINAL loop min (phase counts iterations from it)
  };

  struct RoleCtx {
    const RoleSpec *role = nullptr;
    Map<Var, PrimExpr> subs;      // orig vars -> per-role expressions
    std::vector<LoopLevel> chain; // enclosing loops, outer->inner
    // Pipeline -> phase at acquire/wait / runtime phase counter.
    // Lookup-only maps: pointer keys never order any emitted output.
    std::map<const PipelineSpec *, PrimExpr> pipeline_phase;
    std::map<const PipelineSpec *, Buffer> counters;
  };

  // Linearize the phase over the outermost `levels` loops of the chain
  // (all of them for a sync entry at the current level; a prefix when a
  // deeper access resolves an enclosing pipeline's phase).
  static PrimExpr LinearPhase(const RoleCtx &ctx, size_t levels) {
    PrimExpr phase;
    for (size_t i = 0; i < levels; ++i) {
      const LoopLevel &lvl = ctx.chain[i];
      // The phase counts completed iterations, so a non-zero loop min is
      // subtracted (T.Pipelined(3, 7) starts at phase 0, not 3).
      PrimExpr iter = is_zero(lvl.min) ? lvl.iter : lvl.iter - lvl.min;
      phase = phase.defined() ? phase * lvl.extent + iter : iter;
    }
    return phase.defined() ? phase : PrimExpr(IntImm(DataType::Int(32), 0));
  }
  static PrimExpr LinearPhase(const RoleCtx &ctx) {
    return LinearPhase(ctx, ctx.chain.size());
  }

  // A (role, pipeline) pair needs a runtime phase counter when its
  // cycles are not one-per-counted-iteration of the enclosing for
  // chain: several cycles per body, or a scope with no linearizable
  // phase (while ancestor, guard, non-rectangular nest). All of the
  // pipeline's sync entries live in its one scope; a role without any
  // never counts.
  bool NeedsCounter(const RoleSpec &role, const PipelineSpec &pipeline) const {
    int closes = 0;
    for (const BodyEntryPtr &entry : pipeline.sync_scope->bodies[role.index]) {
      const Synchronization *sync = entry->AsSync();
      if (sync && sync->pipeline == &pipeline && sync->kind.IsCommit())
        ++closes;
    }
    if (closes == 0)
      return false; // the role never syncs this pipeline
    return closes > 1 || !pipeline.sync_scope->HasLinearPhase();
  }

  PrimExpr PipelinePhase(const RoleCtx &ctx,
                         const PipelineSpec *pipeline) const {
    auto cit = ctx.counters.find(pipeline);
    if (cit != ctx.counters.end())
      return BufferLoad(cit->second, {IntImm(DataType::Int(32), 0)});
    return LinearPhase(ctx);
  }

  static Stmt MakeWait(const Buffer &bar, PrimExpr idx, PrimExpr parity) {
    return Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(),
             {BufferLoad(bar, {std::move(idx)}), std::move(parity)}));
  }

  // Arrivals for one commit/release entry — exactly plan.count arrivals
  // per phase.
  void MakeArrive(const PipelineSpec &pipeline, bool full, PrimExpr idx,
                  bool &proxy_fenced, Array<Stmt> *out) const {
    const Buffer &bar = full ? pipeline.full : pipeline.empty;
    const BarrierSidePlan &plan =
        full ? pipeline.full_plan : pipeline.empty_plan;
    // fence.proxy.async orders ALL of the thread's prior generic accesses
    // against its subsequent async-proxy accesses (PTX Proxies /
    // membar.proxy), so one fence covers every arrive until the next
    // emitted op can write again.
    if (plan.needs_proxy_fence && !proxy_fenced) {
      out->push_back(
          Evaluate(Call(DataType::Handle(), fence_proxy_async(), {})));
      proxy_fenced = true;
    }
    if (plan.has_tcgen05_arrival) {
      PrimExpr ptr = bar.access_ptr(3, DataType::Handle(), 1, idx);
      out->push_back(
          Evaluate(Call(DataType::Void(), tcgen05_mma_arrive(), {ptr})));
    }
    if (plan.has_cpasync_arrival) {
      // Group the thread's outstanding cp.asyncs, then arrive deferred:
      // the arrive fires once they complete.
      out->push_back(
          Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {})));
      out->push_back(
          Evaluate(Call(DataType::Handle(), ptx_cp_async_barrier_noinc(),
                        {BufferLoad(bar, {idx})})));
    }
    if (plan.has_thread_arrive) {
      out->push_back(
          Evaluate(Call(DataType::Void(), builtin::ptx_arrive_barrier(),
                        {BufferLoad(bar, {std::move(idx)})})));
    }
  }

  void EmitSync(RoleCtx &ctx, const Synchronization &sync, bool &proxy_fenced,
                Array<Stmt> *out) {
    const PipelineSpec &pipeline = *sync.pipeline;
    PrimExpr depth = IntImm(DataType::Int(32), pipeline.depth);
    PrimExpr phase = PipelinePhase(ctx, &pipeline);
    PrimExpr idx = floormod(phase, depth);
    PrimExpr parity =
        bitwise_and(floordiv(phase, depth), IntImm(DataType::Int(32), 1));

    if (sync.kind.IsProducerAcquire()) {
      ctx.pipeline_phase[&pipeline] = std::move(phase);
      out->push_back(MakeWait(
          pipeline.empty, std::move(idx),
          bitwise_xor(std::move(parity), IntImm(DataType::Int(32), 1))));
    } else if (sync.kind.IsConsumerWait()) {
      ctx.pipeline_phase[&pipeline] = std::move(phase);
      out->push_back(
          MakeWait(pipeline.full, std::move(idx), std::move(parity)));
    } else if (sync.kind.IsProducerCommit()) {
      MakeArrive(pipeline, /*full=*/true, std::move(idx), proxy_fenced, out);
    } else { // consumer release
      MakeArrive(pipeline, /*full=*/false, std::move(idx), proxy_fenced, out);
    }
    if (sync.kind.IsCommit() && ctx.counters.count(&pipeline)) {
      Buffer cnt = ctx.counters.at(&pipeline);
      PrimExpr zero = IntImm(DataType::Int(32), 0);
      out->push_back(BufferStore(cnt, BufferLoad(cnt, {zero}) + 1, {zero}));
    }
  }

  // Rebind every access of a multi-versioned buffer to the acquired
  // version. Buffer versioning only; call conversion is EmitOp's job.
  struct OpRewriter : public StmtExprMutator {
    const WSScheduleMaterializer &self;
    const RoleCtx &ctx;
    // The scope of the op being rewritten (null for guard/condition
    // expressions, which may not touch versioned buffers).
    const Scope *sched_scope;
    bool region_write_ = false; // rewriting a write region's root load
    OpRewriter(const WSScheduleMaterializer &self, const RoleCtx &ctx,
               const Scope *sched_scope)
        : self(self), ctx(ctx), sched_scope(sched_scope) {}

    // The phase of one binding at this access: the role's acquire/wait
    // bound it, or — for an unheld access cooperating inside the
    // pipeline's scope (FA's rescale) — the role's own phase names the
    // in-production slot. Sound only before the role's sync entries of
    // the pipeline (VerifySpanCoverage checked).
    PrimExpr BindingPhase(const PipelineSpec &pipeline, const Buffer &orig) {
      auto pit = ctx.pipeline_phase.find(&pipeline);
      if (pit != ctx.pipeline_phase.end())
        return pit->second;
      auto cit = ctx.counters.find(&pipeline);
      if (cit != ctx.counters.end())
        return BufferLoad(cit->second, {IntImm(DataType::Int(32), 0)});
      ICHECK(pipeline.sync_scope->HasLinearPhase())
          << "ws_schedule: an access to " << orig->name << " needs pipeline '"
          << pipeline.name << "'s phase, which cannot be linearized here, "
          << "and this role has no phase counter for it (no sync entries)";
      return LinearPhase(ctx, pipeline.sync_scope->Depth());
    }

    // The binding's completed-cycle count at an access AFTER its scope:
    // the role's counter, or (enclosing iteration + 1) x the cycles the
    // scope runs per enclosing iteration (static For extents only —
    // VerifySpanCoverage admitted the access).
    PrimExpr CompletedCycles(const PipelineSpec &pipeline, const Buffer &orig) {
      auto cit = ctx.counters.find(&pipeline);
      if (cit != ctx.counters.end())
        return BufferLoad(cit->second, {IntImm(DataType::Int(32), 0)});
      int64_t below = 1;
      for (const Scope *s = pipeline.sync_scope;
           s != nullptr && s != sched_scope; s = s->sched_scope) {
        const auto *extent = s->orig_loop.defined()
                                 ? s->orig_loop->extent.as<IntImmNode>()
                                 : nullptr;
        ICHECK(extent) << "ws_schedule: an access to " << orig->name
                       << " after pipeline '" << pipeline.name
                       << "'s scope needs its completed cycle count, but "
                       << "the scope's trip count is not static";
        below *= extent->value;
      }
      return (LinearPhase(ctx) + IntImm(DataType::Int(32), 1)) *
             IntImm(DataType::Int(32), below);
    }

    // One index per binding, outermost first. A depth-1 binding's slot
    // is 0 without consulting any phase; an access outside a deeper
    // binding's scope takes its LAST-COMPLETED version (reads only —
    // VerifySpanCoverage rejects the rest).
    std::vector<PrimExpr> VersionIndices(const Buffer &orig, bool is_read) {
      std::vector<PrimExpr> out;
      for (const PipelineSpec *p : self.buffer_pipeline_.at(orig->data)) {
        int depth = p->depth;
        if (depth == 1) {
          out.push_back(IntImm(DataType::Int(32), 0));
        } else if (sched_scope != nullptr &&
                   sched_scope->IsNestedIn(p->sync_scope)) {
          out.push_back(floormod(BindingPhase(*p, orig),
                                 IntImm(DataType::Int(32), depth)));
        } else {
          // Outside the binding's scope, the ADJACENT version: a read
          // the last-completed slot, a write the next-produced one.
          PrimExpr cycle = CompletedCycles(*p, orig);
          if (is_read)
            cycle = cycle - IntImm(DataType::Int(32), 1);
          out.push_back(floormod(cycle, IntImm(DataType::Int(32), depth)));
        }
      }
      return out;
    }

    // The BufferLoad path below prepends the version indices to a
    // versioned region's root load; insert matching unit extents to keep
    // the region's indices-per-extent invariant (the op touches exactly
    // the one acquired version of each binding).
    PrimExpr VisitExpr_(const CallNode *op) final {
      bool is_region = op->op.same_as(region()) && op->args.size() >= 2;
      if (is_region) {
        // The root BufferLoad stands for the whole region; its version
        // resolution needs the region's read/write direction.
        const auto *mask = op->args[1].as<IntImmNode>();
        region_write_ = mask && (mask->value & kAccessWrite);
      }
      Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
      if (is_region)
        region_write_ = false;
      if (call->op.same_as(region()) && call->args.size() >= 2) {
        if (const auto *load = call->args[0].as<BufferLoadNode>()) {
          size_t num_extents = call->args.size() - 2;
          if (load->indices.size() > num_extents) {
            Array<PrimExpr> args;
            args.push_back(call->args[0]);
            args.push_back(call->args[1]);
            for (size_t i = num_extents; i < load->indices.size(); ++i)
              args.push_back(IntImm(DataType::Int(32), 1));
            for (size_t i = 2; i < call->args.size(); ++i)
              args.push_back(call->args[i]);
            return Call(call->dtype, call->op, std::move(args),
                        call->annotations, call->span);
          }
        }
      }
      return call;
    }

    // versioned_ is identity-keyed: a miss must mean an unpipelined
    // storage, never an alias object the analysis did not see (that
    // access would silently stay at version 0).
    void CheckMissIsUnpipelined(const Buffer &buf) const {
      ICHECK(!self.buffer_pipeline_.count(buf->data))
          << "ws_schedule: access to '" << buf->name << "' aliases pipelined "
          << "storage '" << buf->data->name_hint << "' through a buffer "
          << "object the access analysis never saw; it cannot be versioned";
    }

    PrimExpr VisitExpr_(const BufferLoadNode *op) final {
      BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
      auto vit = self.versioned_.find(load->buffer);
      if (vit == self.versioned_.end()) {
        CheckMissIsUnpipelined(load->buffer);
        return load;
      }
      Array<PrimExpr> indices;
      for (PrimExpr &v : VersionIndices(load->buffer, !region_write_))
        indices.push_back(std::move(v));
      for (const PrimExpr &i : load->indices)
        indices.push_back(i);
      return BufferLoad(vit->second, std::move(indices), load->predicate,
                        load->span);
    }

    Stmt VisitStmt_(const BufferStoreNode *op) final {
      BufferStore store =
          Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
      auto vit = self.versioned_.find(store->buffer);
      if (vit == self.versioned_.end()) {
        CheckMissIsUnpipelined(store->buffer);
        return store;
      }
      Array<PrimExpr> indices;
      for (PrimExpr &v : VersionIndices(store->buffer, false))
        indices.push_back(std::move(v));
      for (const PrimExpr &i : store->indices)
        indices.push_back(i);
      return BufferStore(vit->second, store->value, std::move(indices),
                         store->predicate, store->span);
    }
  };

  Stmt RewriteOpStmt(const RoleCtx &ctx, Stmt stmt,
                     const Scope *sched_scope) const {
    OpRewriter rewriter(*this, ctx, sched_scope);
    return rewriter(std::move(stmt));
  }

  PrimExpr RewriteOpExpr(const RoleCtx &ctx, PrimExpr expr,
                         const Scope *sched_scope = nullptr) const {
    OpRewriter rewriter(*this, ctx, sched_scope);
    return rewriter(std::move(expr));
  }

  // Rewrite an asynchronous atom's call: swap it to its explicit async
  // op (TMA, tcgen05) or annotate it (cp.async).
  Stmt ConvertAtomCall(const RoleCtx &ctx, const Operation &op,
                       Stmt stmt) const {
    const auto *ev = stmt.as<EvaluateNode>();
    ICHECK(ev);
    Call call = Downcast<Call>(ev->value);
    auto ann = call->annotations;
    if (op.atom == OpAtom::kTmaCopy) {
      // The transaction completes the full barrier of the pipeline
      // protecting the destination. A copy into an unprotected buffer
      // has no barrier to wire and stays a plain copy.
      if (op.write_def == nullptr)
        return stmt;
      const PipelineSpec &pipeline = *op.write_def;
      // Span coverage verified: the pipeline was acquired.
      PrimExpr idx = floormod(ctx.pipeline_phase.at(&pipeline),
                              IntImm(DataType::Int(32), pipeline.depth));
      ann.Set("barrier", BufferLoad(pipeline.full, {std::move(idx)}));
      ann.Set("is_tma_copy", IntImm(DataType::Int(32), 1));
      if (op.fused_arrive)
        ann.Set("emit_arrive", IntImm(DataType::Int(32), 1));
      return Evaluate(Call(call->dtype, TmaCopyOp(), call->args, std::move(ann),
                           call->span));
    } else if (op.atom == OpAtom::kCpAsyncCopy) {
      // A copy into an unprotected buffer keeps its own commit/wait.
      if (op.write_def == nullptr)
        return stmt;
      // Skip the copy's implicit commit_group + wait_group(0):
      // completion rides the commit entry's deferred arrive instead.
      ann.Set(attr::kAsyncCopyNoImplicitCommitWait,
              IntImm(DataType::Int(32), 1));
      return Evaluate(
          Call(call->dtype, call->op, call->args, std::move(ann), call->span));
    } else if (op.atom == OpAtom::kTcgen05Gemm) {
      ann.Set("is_tcgen05", IntImm(DataType::Int(32), 1));
      return Evaluate(Call(call->dtype, Tcgen05GemmOp(), call->args,
                           std::move(ann), call->span));
    }
    LOG(FATAL) << "ws_schedule: unknown async atom "
               << static_cast<int>(op.atom);
  }

  // The op's source guard is NOT applied here: EmitScopeBody folds it
  // into the entry's guard.
  Stmt EmitOp(const RoleCtx &ctx, const Operation &op) const {
    Stmt body = RewriteOpStmt(
        ctx, RebindBufferData(Substitute(op.stmt, ctx.subs), ctx.subs),
        op.sched_scope);
    if (op.atom != OpAtom::kSync)
      body = ConvertAtomCall(ctx, op, std::move(body));
    return body;
  }

  // Rebind an emitted statement's scalar defs and their uses within it.
  struct BindDefRewriter : public StmtExprMutator {
    const Map<Var, Var> &fresh;
    explicit BindDefRewriter(const Map<Var, Var> &fresh) : fresh(fresh) {}

    Stmt VisitStmt_(const BindNode *op) final {
      Bind bind = Downcast<Bind>(StmtExprMutator::VisitStmt_(op));
      if (auto replacement = fresh.Get(bind->var))
        return Bind(replacement.value(), bind->value, bind->span);
      return bind;
    }
    PrimExpr VisitExpr_(const VarNode *op) final {
      if (auto replacement = fresh.Get(GetRef<Var>(op)))
        return replacement.value();
      return GetRef<PrimExpr>(op);
    }
  };

  // tirx::Substitute rebinds a buffer's data var only where the buffer is
  // DEFINED (IRSubstitute::VisitBufferDef); ops are emitted one statement
  // at a time and make_tensor views carry no definition statement, so
  // buffers on substituted handle vars are rebuilt at their use sites.
  static Stmt RebindBufferData(Stmt stmt, const Map<Var, PrimExpr> &subs) {
    BufferRemap remap;
    PostOrderVisit(stmt, [&](const ffi::ObjectRef &node) {
      Buffer buffer;
      if (const auto *load = node.as<BufferLoadNode>())
        buffer = load->buffer;
      else if (const auto *store = node.as<BufferStoreNode>())
        buffer = store->buffer;
      else
        return;
      if (remap.count(buffer))
        return;
      auto replacement = subs.Get(buffer->data);
      if (!replacement.has_value())
        return;
      if (const auto *var = replacement->as<VarNode>()) {
        auto n = make_object<BufferNode>(*buffer.get());
        n->data = GetRef<Var>(var);
        remap.emplace(std::move(buffer), Buffer(std::move(n)));
      }
    });
    return RemapBuffers(std::move(stmt), remap);
  }

  // Every emitted copy of an op re-binds its scalar defs with fresh vars:
  // role branches and unrolled pipeline steps re-emit the same source
  // statement, and each TIR var must have a single definition. Later ops
  // of the same emitted slice see the fresh names through ctx.subs, which
  // EmitScopeBody scopes per slice.
  Stmt FreshenBindDefs(RoleCtx &ctx, Stmt stmt) {
    std::vector<Var> bound;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (const auto *bind = node.as<BindNode>())
        bound.push_back(bind->var);
    });
    if (bound.empty())
      return stmt;
    Map<Var, Var> fresh;
    for (const Var &var : bound) {
      Var replacement =
          var.copy_with_suffix("_ws" + std::to_string(fresh_bind_count_++));
      fresh.Set(var, replacement);
      ctx.subs.Set(var, replacement);
    }
    return RebindBufferData(BindDefRewriter(fresh)(std::move(stmt)), ctx.subs);
  }

  // Emit one (role, scope) body, keeping only entries with stage delta
  // in [delta_lo, delta_hi] — prologue/epilogue steps select their
  // slice of the software pipeline this way. `base` is the step
  // expression of a for scope (an entry at delta d runs iteration
  // base - d); undefined for the root and for while scopes.
  Array<Stmt> EmitScopeBody(RoleCtx &ctx, const Scope &scope,
                            const StagePlan &plan,
                            const Optional<PrimExpr> &base,
                            const Var &orig_loop_var, int delta_lo,
                            int delta_hi) {
    const std::vector<BodyEntryPtr> &entries = scope.bodies[ctx.role->index];
    if (entries.empty())
      return {};

    Map<Var, PrimExpr> saved_subs = ctx.subs;
    Array<Stmt> stmts;
    // Whether a proxy fence was emitted since the last op; only ops can
    // add generic accesses, so subsequent arrives need no second fence.
    bool proxy_fenced = false;

    // Point the substitution and phase chain at entry i's iteration
    // and return its guard (iteration bound + the op's source guard).
    auto focus_entry = [&](size_t i) -> Optional<PrimExpr> {
      Optional<PrimExpr> guard;
      if (base.defined()) {
        int delta = plan.delta[i];
        PrimExpr iter = delta == 0
                            ? base.value()
                            : base.value() - IntImm(DataType::Int(32), delta);
        ctx.subs.Set(orig_loop_var, iter);
        ctx.chain.back().iter = iter;
        // A prologue step (finite delta window) bounds every entry by
        // the loop extent, covering loops shorter than their shift;
        // static bounds fold to constant guards the simplifier removes.
        if (delta_hi < plan.shift)
          guard = iter < ctx.chain.back().extent;
      }
      const Operation *op =
          entries[i]->AsScope() ? nullptr : entries[i]->AsOperation();
      if (op != nullptr && op->guard.defined()) {
        PrimExpr src = RewriteOpExpr(
            ctx, Substitute(op->guard.value(), ctx.subs), op->sched_scope);
        guard = guard.defined() ? guard.value() && src : src;
      }
      return guard;
    };
    // Runs of entries under a structurally identical guard close as one
    // IfThenElse.
    Optional<PrimExpr> cur_guard;
    Array<Stmt> guarded;
    StructuralEqual structural_equal;
    auto close_guard = [&]() {
      if (cur_guard.defined() && !guarded.empty()) {
        stmts.push_back(
            IfThenElse(cur_guard.value(), SeqOrSingle(std::move(guarded))));
      }
      cur_guard = Optional<PrimExpr>();
      guarded = Array<Stmt>();
    };

    for (size_t i = 0; i < entries.size(); ++i) {
      if (plan.delta[i] < delta_lo || plan.delta[i] > delta_hi)
        continue;
      Optional<PrimExpr> guard = focus_entry(i);
      bool reuse = guard.defined() == cur_guard.defined() &&
                   (!guard.defined() ||
                    structural_equal(guard.value(), cur_guard.value()));
      if (!reuse) {
        close_guard();
        cur_guard = guard;
      }
      Array<Stmt> *out = cur_guard.defined() ? &guarded : &stmts;
      if (const Scope *child = entries[i]->AsScope()) {
        Stmt emitted = EmitChildScope(ctx, *child);
        if (emitted.defined())
          out->push_back(std::move(emitted));
        proxy_fenced = false; // the child's ops may write generically
      } else if (const Synchronization *sync = entries[i]->AsSync()) {
        EmitSync(ctx, *sync, proxy_fenced, out);
      } else {
        out->push_back(
            FreshenBindDefs(ctx, EmitOp(ctx, *entries[i]->AsOperation())));
        proxy_fenced = false;
      }
    }
    close_guard();
    // Fresh scalar defs are slice-local; restore the caller's substitutions
    // and the steady-state iteration for any parent-level use.
    ctx.subs = saved_subs;
    if (base.defined())
      ctx.chain.back().iter = base.value();
    return stmts;
  }

  static Stmt SeqOrSingle(Array<Stmt> stmts) {
    return stmts.size() == 1 ? stmts[0] : SeqStmt(std::move(stmts));
  }

  // Emit one child scope for the current role. A stage shift is
  // unrolled explicitly, so no per-iteration boundary check remains in
  // the steady-state loop:
  //   prologue step t (t = 0 .. shift-1): the entries at delta <= t;
  //   steady state:                       every entry, unguarded;
  //   epilogue step t (t = 1 .. shift):   the entries at delta >= t.
  Stmt EmitChildScope(RoleCtx &ctx, const Scope &scope) {
    if (scope.bodies[ctx.role->index].empty())
      return Stmt(); // role does not participate in this scope
    StagePlan plan = PlanStages(*ctx.role, scope.bodies[ctx.role->index]);
    Stmt out = scope.orig_while.defined()
                   ? EmitChildScopeWhileLoop(ctx, scope, plan)
                   : EmitChildScopeForLoop(ctx, scope, plan);
    // The uniform source guard: false skips the scope in all roles
    // together.
    if (scope.guard.defined()) {
      out = IfThenElse(
          RewriteOpExpr(ctx, Substitute(scope.guard.value(), ctx.subs)),
          std::move(out));
    }
    return out;
  }

  // For scope: prologue step t runs iterations t - delta, the steady
  // state covers [shift, extent), and epilogue step t runs iterations
  // extent + t - 1 - delta iff extent > shift - t. Short and dynamic
  // extents are handled by the emitted bounds alone; the simplifier
  // folds the static cases.
  Stmt EmitChildScopeForLoop(RoleCtx &ctx, const Scope &scope,
                             const StagePlan &plan) {
    constexpr int kMaxDelta = std::numeric_limits<int>::max();
    PrimExpr zero = IntImm(DataType::Int(32), 0);

    const For &orig = scope.orig_loop;
    const Var &orig_var = orig->loop_var;
    Var fresh(orig_var->name_hint, orig_var->dtype);
    ctx.subs.Set(orig_var, fresh);
    PrimExpr extent = Substitute(orig->extent, ctx.subs);
    PrimExpr min = Substitute(orig->min, ctx.subs);
    if (plan.shift > 0) {
      ICHECK(is_zero(min))
          << "ws_schedule: stage offsets require 0-based loops";
    }
    ctx.chain.push_back({fresh, extent, min});

    Array<Stmt> result;
    for (int t = 0; t < plan.shift; ++t) {
      Array<Stmt> step = EmitScopeBody(
          ctx, scope, plan, IntImm(DataType::Int(32), t), orig_var, 0, t);
      for (const Stmt &stmt : step)
        result.push_back(stmt);
    }

    Array<Stmt> body =
        EmitScopeBody(ctx, scope, plan, fresh, orig_var, 0, kMaxDelta);
    // The steady state covers iterations [shift, extent). Preserve the
    // source loop kind and annotations, minus the consumed markers.
    Map<String, Any> ann =
        FilterAnnotations(scope.orig_loop->annotations, [](const String &k) {
          return k != kWSOpIdKey && k != "num_stages" &&
                 k.find("tl_pipeline_") != 0;
        });
    PrimExpr loop_min =
        plan.shift > 0 ? PrimExpr(IntImm(DataType::Int(32), plan.shift)) : min;
    PrimExpr loop_extent =
        plan.shift > 0 ? max(extent - plan.shift, zero) : extent;
    result.push_back(For(fresh, std::move(loop_min), std::move(loop_extent),
                         scope.orig_loop->kind, SeqOrSingle(std::move(body)),
                         std::nullopt, std::move(ann)));

    for (int t = 1; t <= plan.shift; ++t) {
      Array<Stmt> step = EmitScopeBody(
          ctx, scope, plan, extent + IntImm(DataType::Int(32), t - 1), orig_var,
          t, kMaxDelta);
      // The step runs iff extent > shift - t.
      result.push_back(
          IfThenElse(IntImm(DataType::Int(32), plan.shift - t) < extent,
                     SeqOrSingle(std::move(step))));
    }
    ctx.chain.pop_back();
    return SeqOrSingle(std::move(result));
  }

  // While scope: no iteration expression (phases under it are runtime
  // counters). Prologue steps run under the loop condition and bump a
  // completed-trip counter; epilogue step t runs iff t <= trips, which
  // drains exactly what a short loop started.
  Stmt EmitChildScopeWhileLoop(RoleCtx &ctx, const Scope &scope,
                               const StagePlan &plan) {
    constexpr int kMaxDelta = std::numeric_limits<int>::max();
    PrimExpr zero = IntImm(DataType::Int(32), 0);
    // The condition is re-evaluated at every use.
    auto cond = [&]() {
      return RewriteOpExpr(ctx,
                           Substitute(scope.orig_while->condition, ctx.subs));
    };

    Array<Stmt> result;
    Buffer trips;
    if (plan.shift > 0) {
      trips = decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                          scope.id + "_trips", "local");
      result.push_back(AllocBuffer(trips));
      result.push_back(BufferStore(trips, zero, {zero}));
    }
    auto bump_trips = [&]() {
      return BufferStore(trips, BufferLoad(trips, {zero}) + 1, {zero});
    };

    for (int t = 0; t < plan.shift; ++t) {
      Array<Stmt> step =
          EmitScopeBody(ctx, scope, plan, Optional<PrimExpr>(), Var(), 0, t);
      step.push_back(bump_trips());
      result.push_back(IfThenElse(cond(), SeqOrSingle(std::move(step))));
    }

    Array<Stmt> body = EmitScopeBody(ctx, scope, plan, Optional<PrimExpr>(),
                                     Var(), 0, kMaxDelta);
    if (plan.shift > 0)
      body.push_back(bump_trips());
    result.push_back(While(cond(), SeqOrSingle(std::move(body))));

    for (int t = 1; t <= plan.shift; ++t) {
      Array<Stmt> step = EmitScopeBody(ctx, scope, plan, Optional<PrimExpr>(),
                                       Var(), t, kMaxDelta);
      result.push_back(
          IfThenElse(IntImm(DataType::Int(32), t) <= BufferLoad(trips, {zero}),
                     SeqOrSingle(std::move(step))));
    }
    return SeqOrSingle(std::move(result));
  }

  Stmt EmitRole(const RoleSpec &role) {
    RoleCtx ctx;
    ctx.role = &role;

    Array<Stmt> stmts;
    if (role.nreg > 0) {
      stmts.push_back(
          Evaluate(Call(DataType::Handle(), set_max_nreg(),
                        {IntImm(DataType::Int(32), role.nreg),
                         IntImm(DataType::Int(32), role.NregAction())})));
    }

    // Runtime phase counters where linearization is unsound.
    for (const auto &pipeline : pipelines_) {
      if (NeedsCounter(role, *pipeline)) {
        Buffer cnt =
            decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                        pipeline->name + "_phase", "local");
        ctx.counters[pipeline.get()] = cnt;
        stmts.push_back(AllocBuffer(cnt));
        stmts.push_back(BufferStore(cnt, IntImm(DataType::Int(32), 0),
                                    {IntImm(DataType::Int(32), 0)}));
      }
    }

    StagePlan plan = PlanStages(role, root_scope_->bodies[role.index]);
    ICHECK_EQ(plan.shift, 0)
        << "ws_schedule: stage offsets require a loop scope; role " << role.name
        << " has offset sync stages in its root body";
    Array<Stmt> body =
        EmitScopeBody(ctx, *root_scope_, plan, Optional<PrimExpr>(), Var(), 0,
                      std::numeric_limits<int>::max());
    ICHECK(!body.empty()) << "ws_schedule: role " << role.name
                          << " has an empty root body";
    for (const Stmt &s : body)
      stmts.push_back(s);
    return SeqOrSingle(std::move(stmts));
  }

  // Idle warps execute their warpgroup's register request; a no-op
  // when it requests nothing.
  Stmt EmitIdle(int nreg) const {
    if (nreg == 0)
      return Evaluate(IntImm(DataType::Int(32), 0));
    return Evaluate(
        Call(DataType::Handle(), set_max_nreg(),
             {IntImm(DataType::Int(32), nreg),
              IntImm(DataType::Int(32), nreg >= kNregIncThreshold ? 1 : 0)}));
  }

  // Emit the `if (tx < ...)` role branches ordered by warp range,
  // filling gaps with idle branches split by warpgroup register
  // request.
  Stmt EmitRoleBranches() {
    Array<Stmt> branches;
    Array<PrimExpr> conds;
    auto fill_idle = [&](int lo, int hi) {
      for (int w = lo; w < hi;) {
        int nreg = warpgroup_nreg_[w / 4];
        int seg = w;
        while (seg < hi && warpgroup_nreg_[seg / 4] == nreg)
          ++seg;
        conds.push_back(thread_var_ < IntImm(DataType::Int(32), seg * 32));
        branches.push_back(EmitIdle(nreg));
        w = seg;
      }
    };
    int cursor = 0;
    for (const auto &role : roles_) {
      ICHECK_GE(role->warp_lo, cursor)
          << "ws_schedule: overlapping role warp ranges";
      fill_idle(cursor, role->warp_lo);
      conds.push_back(thread_var_ <
                      IntImm(DataType::Int(32), role->warp_hi * 32));
      branches.push_back(EmitRole(*role));
      cursor = role->warp_hi;
    }
    fill_idle(cursor, num_warps_);
    Stmt body = branches[branches.size() - 1];
    for (int i = static_cast<int>(branches.size()) - 2; i >= 0; --i)
      body = IfThenElse(conds[i], branches[i], std::move(body));
    return body;
  }

  // Rebuild the block: versioned buffers, barrier allocations with
  // their init counts, the consumed schedule annotation removed.
  SBlock RebuildBlock(Stmt body) {
    // Re-wrap the kernel-level metadata attrs consumed during matching.
    for (auto it = metadata_attrs_.rbegin(); it != metadata_attrs_.rend();
         ++it) {
      AttrStmt attr = *it;
      attr.CopyOnWrite()->body = std::move(body);
      body = attr;
    }
    Array<Buffer> allocs;
    for (const Buffer &buf : block_->alloc_buffers) {
      auto vit = versioned_.find(buf);
      allocs.push_back(vit == versioned_.end() ? buf : vit->second);
    }
    Map<Var, Array<PrimExpr>> barrier_init;
    for (const auto &pipeline : pipelines_) {
      allocs.push_back(pipeline->full);
      allocs.push_back(pipeline->empty);
      Array<PrimExpr> full_counts, empty_counts;
      for (int i = 0; i < pipeline->depth; ++i) {
        full_counts.push_back(
            IntImm(DataType::Int(32), pipeline->full_plan.count));
        empty_counts.push_back(
            IntImm(DataType::Int(32), pipeline->empty_plan.count));
      }
      barrier_init.Set(pipeline->full->data, std::move(full_counts));
      barrier_init.Set(pipeline->empty->data, std::move(empty_counts));
    }

    SBlock new_block = block_;
    auto *n = new_block.CopyOnWrite();
    n->body = std::move(body);
    n->alloc_buffers = std::move(allocs);
    Map<String, Any> ann =
        FilterAnnotations(n->annotations, [](const String &key) {
          return key != kWSScheduleKey;
        });
    if (auto existing = ann.Get("barrier_init")) {
      auto prev = Downcast<Map<Var, Array<PrimExpr>>>(existing.value());
      for (const auto &[var, counts] : prev)
        barrier_init.Set(var, counts);
    }
    ann.Set("barrier_init", std::move(barrier_init));
    // Annotated layouts of versioned buffers gain the version
    // dimension too.
    if (auto lm_ref = ann.Get(attr::kLayoutMap)) {
      if (auto lm_opt = lm_ref.value().as<Map<Var, Layout>>()) {
        Map<Var, Layout> layout_map = lm_opt.value();
        bool changed = false;
        VarSet expanded_storage;
        for (const auto &[orig, versioned] : versioned_) {
          if (!expanded_storage.insert(orig->data).second)
            continue;
          auto entry = layout_map.Get(orig->data);
          if (!entry.has_value())
            continue;
          // One leading dimension per binding.
          Array<PrimExpr> prefix;
          size_t n_bindings = buffer_pipeline_.at(orig->data).size();
          for (size_t i = 0; i < n_bindings; ++i)
            prefix.push_back(versioned->shape[i]);
          layout_map.Set(orig->data, entry.value()->Expand(prefix));
          changed = true;
        }
        if (changed)
          ann.Set(attr::kLayoutMap, layout_map);
      }
    }
    n->annotations = std::move(ann);
    return new_block;
  }

  // ---- state ---------------------------------------------------------------

  SBlock block_;
  Var thread_var_;
  Target target_;
  int num_warps_ = 0;
  std::vector<std::unique_ptr<RoleSpec>> roles_;
  std::vector<std::unique_ptr<PipelineSpec>> pipelines_;
  std::vector<std::shared_ptr<Scope>> scopes_;
  Scope *root_scope_ = nullptr;
  std::map<String, std::shared_ptr<Operation>> ops_;
  // Kernel-level metadata AttrStmts, re-wrapped around the rebuilt body.
  std::vector<AttrStmt> metadata_attrs_;
  // Per-warpgroup register request (index = warp / 4, 0 = none).
  std::vector<int> warpgroup_nreg_;
  // Suffix counter making every emitted Bind definition unique.
  int fresh_bind_count_ = 0;
  StoragePipelineMap buffer_pipeline_;
  BufferVersionMap versioned_;
};

// Rewrite every schedule-annotated SBlock in place — a function may
// hold several kernels, each with its own schedule. The enclosing
// threadIdx.x binding is widened to each schedule's warp count.
class WSScheduleRewriter : public StmtMutator {
public:
  explicit WSScheduleRewriter(Optional<Target> target)
      : target_(std::move(target)) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_iv_ = iv;
        new_extent_ = -1;
        AttrStmt attr = Downcast<AttrStmt>(StmtMutator::VisitStmt_(op));
        if (new_extent_ >= 0) {
          PrimExpr nt = IntImm(DataType::Int(32), new_extent_);
          thread_iv_.CopyOnWrite()->dom = {0, nt};
          attr.CopyOnWrite()->node = thread_iv_;
          attr.CopyOnWrite()->value = nt;
        }
        thread_iv_ = {};
        return attr;
      }
      // Warp roles partition threadIdx.x only.
      if (iv->thread_tag == "threadIdx.y" || iv->thread_tag == "threadIdx.z") {
        bool saved = multi_dim_threads_;
        multi_dim_threads_ = multi_dim_threads_ || !is_one(op->value);
        Stmt stmt = StmtMutator::VisitStmt_(op);
        multi_dim_threads_ = saved;
        return stmt;
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (!op->annotations.count(kWSScheduleKey))
      return StmtMutator::VisitStmt_(op);
    ICHECK(thread_iv_.defined())
        << "ws_schedule: threadIdx.x binding not found around the scheduled "
           "block";
    ICHECK(!multi_dim_threads_)
        << "ws_schedule: the scheduled kernel must launch threads in one "
           "dimension (threadIdx.x only); warp roles partition threadIdx.x";
    ICHECK(target_.defined())
        << "ws_schedule: PrimFunc has no bound target (BindTarget must run "
           "before MaterializeWSSchedule)";
    WSScheduleMaterializer materializer(GetRef<SBlock>(op), thread_iv_->var,
                                        target_.value());
    SBlock block = materializer.Run();
    new_extent_ = materializer.NumThreads();
    return block;
  }

private:
  Optional<Target> target_;
  IterVar thread_iv_;
  bool multi_dim_threads_ = false;
  int new_extent_ = -1;
};

// A function without a schedule annotation is returned untouched.
PrimFunc MaterializeWSScheduleImpl(PrimFunc f) {
  WSScheduleRewriter rewriter(f->GetAttr<Target>(tvm::attr::kTarget));
  Stmt body = rewriter(f->body);
  if (body.same_as(f->body))
    return f;
  f.CopyOnWrite()->body = std::move(body);
  return f;
}

} // namespace

tvm::transform::Pass MaterializeWSSchedule() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      tvm::transform::PassContext ctx) {
    return MaterializeWSScheduleImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MaterializeWSSchedule", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.cuda.transform.MaterializeWSSchedule",
                        MaterializeWSSchedule);
}

} // namespace tl
} // namespace tvm
