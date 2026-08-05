/*!
 * \file materialize_ws_schedule.cc
 * \brief Materialize a user-provided warp-specialization schedule.
 *
 * A warp-specialized kernel partitions its warps into roles that run
 * different code and hand data to each other through multi-buffered
 * staging buffers. The schedule — a T.WSSchedule annotation on the
 * kernel's block (see T.annotate_ws_schedule) — declares:
 *  - roles: named warp ranges, each with an optional register budget;
 *  - pipelines: one producer/consumer handshake each — a "full" and an
 *    "empty" mbarrier array protecting a set of buffers, which this pass
 *    replicates into `depth` versions;
 *  - scopes: the scheduled serial loops plus an implicit root. A scope
 *    gives each participating role a body: the sequence of ops, child
 *    scopes, and sync points that role executes per iteration.
 *
 * The pass rewrites the kernel into explicit form — `if (tx < ...)` role
 * branches, versioned buffers, mbarrier waits/arrives with correct
 * parity, TMA copies completing through barriers, asynchronous tcgen05
 * MMAs — before PipelinePlanning / LayoutInference / LowerTileOp, so
 * downstream passes see the same tile-op IR a hand-written
 * warp-specialized kernel would produce.
 *
 * Ids. The schedulable task is a single statement, named by a stable id:
 * a "tl.ws_op_id" entry in a tile op's call annotations or a loop's
 * annotations, or a `with T.ws_op(id):` wrapper for a scalar Bind (the
 * one statement form without annotations). Scope loops are serial
 * (T.Pipelined or T.serial; the pass consumes num_stages — the software
 * pipelining happens here). A T.unroll / T.Parallel loop is one opaque
 * op; statements inside it need no ids. Every statement of a scheduled
 * kernel must be placed by the schedule — anything unplaced is a fatal
 * error. A function may hold several kernels, each with its own
 * schedule; a function without any is returned untouched, and a
 * scheduled kernel's id markers are consumed during matching and
 * emission.
 *
 * Dependencies are computed, not declared: QueryAccess derives each op's
 * operands from the tile ops' own access regions. An operand's def is
 * the pipeline protecting its buffer; a pipeline's producer / consumer
 * uses are the ops writing / reading its buffers.
 *
 * Synchronization. producer_acquire / producer_commit and consumer_wait
 * / consumer_release bracket a role's work on a pipeline: the acquire or
 * wait opens a span, the matching commit or release closes it in the
 * same scope body. An op must run inside an open span of every pipeline
 * protecting one of its operands, and its accesses are rebound to buffer
 * version (phase % depth), where the role's phase counts its completed
 * sync cycles on that pipeline — an expression of the loop iteration, or
 * a runtime counter when cycles repeat within one body or span several
 * loops. Sync entries carry a software-pipeline stage: within one body,
 * an entry at stage s runs (s - s_min) iterations behind, the loop gains
 * that many extra steps, and boundary guards cover the ramp-up/drain.
 *
 * Arrive counts are derived from what each signaling op lowers to (its
 * atom): TMA data rides the barrier's transaction count, a tcgen05
 * commit arrives once for all outstanding MMAs, and synchronous compute
 * arrives per thread.
 *
 * Source `if`s around statements become per-op guards, folded together
 * with the stage boundary guard into one condition. Sync entries are
 * never guarded: a role that skips work skips only the op bodies, so
 * pipeline cycles stay unconditional in every role.
 *
 * VerifySchedule rejects broken schedules at compile time:
 *  - span coverage: brackets pair up, a role is the producer or the
 *    consumer of a pipeline (never both), and ops run inside the spans
 *    their operands need;
 *  - cycle balance: within every loop scope, a pipeline's producer and
 *    consumer sides cycle equally often per iteration;
 *  - deadlock freedom: the roles' sync events are executed under the
 *    mbarrier parity model, each loop modeled at two iterations (enough,
 *    because balanced iterations leave the barrier state unchanged).
 *
 * TODO: cluster launch control, while loops, 2-CTA GEMM, epilogue
 * sub-tiling, stage shifts on counter-tracked pipelines,
 * try_acquire/try_wait split sync entries, shared-barrier pipeline
 * groups.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
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
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cuda/op/builtin.h"
#include "cuda/op/copy.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "transform/common/mbarrier.h"
#include "transform/common/warp_specialize.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// Below this register budget a role donates registers (setmaxnreg.dec);
// at or above it the role receives (setmaxnreg.inc).
constexpr int kNregIncThreshold = 128;

// Positions in the materializer's roles_ / pipelines_ / scopes_ vectors,
// resolved once from schedule names at the parse boundary; -1 = none.
// Quantities that are not indices (stages, depths, warp counts) stay int.
using RoleIndex = int;
using PipelineIndex = int;
using ScopeIndex = int;

// Identity-keyed maps on Buffer handles. Iteration order is hash order
// and never observable: consumers renormalize into ordered containers.
using BufferDefMap =
    std::unordered_map<Buffer, PipelineIndex, ObjectPtrHash, ObjectPtrEqual>;
using BufferVersionMap =
    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>;

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
const Op &RegionOp() {
  static const Op &op = Op::Get("tl.tileop.region");
  return op;
}
const Op &TmaCopyOp() {
  static const Op &op = Op::Get("tl.tileop.tma_copy");
  return op;
}
const Op &Tcgen05GemmOp() {
  static const Op &op = Op::Get("tl.tileop.tcgen05_gemm");
  return op;
}

// ---------------------------------------------------------------------------
// QueryAccess: the buffers a statement reads and writes — the pass's
// dependency oracle. Schedules declare no dataflow; pipeline membership,
// arrive styles, and arrive counts are all derived from these operands.
//
// Tile ops report their own access regions (a gemm reads its accumulator
// only when it does not clear it, say); granularity is the whole buffer
// for now. Plain loads and stores count directly, except the
// address-only root load inside a region argument, which would surface
// write-only operands as reads. Global buffers are operands too — they
// simply never resolve to a pipeline.
// ---------------------------------------------------------------------------
struct AccessSet {
  // Operand buffer -> defining pipeline: the pipeline protecting the
  // buffer, or -1 while unresolved / unprotected. QueryAccess records
  // the operands; BuildUseChains resolves the defs.
  BufferDefMap reads, writes;

  // The defs of all operands, unprotected operands excluded.
  std::set<PipelineIndex> Defs() const {
    std::set<PipelineIndex> defs;
    for (const auto *operands : {&reads, &writes})
      for (const auto &[buf, def] : *operands)
        if (def >= 0)
          defs.insert(def);
    return defs;
  }

  bool HasDef(PipelineIndex pipeline_idx) const {
    for (const auto *operands : {&reads, &writes})
      for (const auto &[buf, def] : *operands)
        if (def == pipeline_idx)
          return true;
    return false;
  }
};

// Unions the statement's accesses into *acc. Region calls appear only as
// tile-op arguments and are skipped: the owning op reports their
// accesses, and their root loads are excluded from the plain-load pass.
void QueryAccess(const Stmt &stmt, AccessSet *acc) {
  std::unordered_set<BufferLoad, ObjectPtrHash, ObjectPtrEqual> region_roots;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (!call)
      return;
    if (call->op.same_as(RegionOp())) {
      const auto *load = call->args[0].as<BufferLoadNode>();
      ICHECK(load) << "ws_schedule: region arg0 must be a BufferLoad";
      region_roots.insert(GetRef<BufferLoad>(load));
      return;
    }
    TileOperator tile_op = ParseOperator(GetRef<Call>(call));
    if (!tile_op.defined())
      return; // a non-tile-op intrinsic (exp2, any_sync, ...)
    AccessRegions access = tile_op->GetAccessRegions();
    for (const BufferRegion &r : access.reads)
      acc->reads.emplace(r->buffer, -1);
    for (const BufferRegion &r : access.writes)
      acc->writes.emplace(r->buffer, -1);
  });
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>()) {
      if (!region_roots.count(GetRef<BufferLoad>(load)))
        acc->reads.emplace(load->buffer, -1);
    } else if (const auto *store = node.as<BufferStoreNode>()) {
      acc->writes.emplace(store->buffer, -1);
    }
  });
}

// ---------------------------------------------------------------------------
// Op atoms: which asynchronous instruction (if any) an op lowers to.
// Classified once per op and consumed twice — arrive-count planning reads
// each barrier side's completion signal off the atoms, and emission
// converts the call — so planner and emitter cannot disagree.
//  - kSync: synchronous compute; only the executing threads witness its
//    completion, so they arrive per thread.
//  - kTmaCopy: bulk-async copy; the data rides the barrier's transaction
//    count. It only writes its destination, so it holds producer uses.
//  - kTcgen05Gemm: asynchronous tensor core; one tcgen05.commit performs
//    a single mbarrier arrive covering all outstanding MMAs.
// ---------------------------------------------------------------------------
enum class OpAtom : uint8_t {
  kSync = 0,
  kTmaCopy = 1,
  kTcgen05Gemm = 2,
};

// Classify one call with the same instruction-selection helpers the
// lowering uses: ClassifyWarpSpecializedProducerCopy applies the real
// TMA legality rules (scope, swizzle, stride, dtype), and a gemm
// accumulating in tensor memory lowers to the asynchronous tcgen05 path.
OpAtom ClassifyCall(const Call &call, const Target &target) {
  TileOperator tile_op = ParseOperator(call);
  if (!tile_op.defined())
    return OpAtom::kSync; // a non-tile-op intrinsic
  if (const auto *copy = tile_op.as<CopyNode>()) {
    cuda::CopyInstSelection sel =
        cuda::ClassifyWarpSpecializedProducerCopy(*copy, target);
    return cuda::CopyInstIsTMA(sel.inst) ? OpAtom::kTmaCopy : OpAtom::kSync;
  }
  if (const auto *gemm = tile_op.as<GemmNode>()) {
    return gemm->cRegion_->buffer.scope() == "shared.tmem"
               ? OpAtom::kTcgen05Gemm
               : OpAtom::kSync;
  }
  return OpAtom::kSync;
}

// ---------------------------------------------------------------------------
// Parsed schedule structures
// ---------------------------------------------------------------------------
struct RoleSpec {
  String name;
  int warp_lo = 0, warp_hi = 0; // [lo, hi) in warps
  int nreg = 0;                 // 0 = absent
  int NumThreads() const { return (warp_hi - warp_lo) * 32; }
  int NregAction() const { return nreg >= kNregIncThreshold ? 1 : 0; }
};

// One schedulable op: declared by ParseSchedule (placed by exactly one
// role in exactly one scope body), completed by RecordOpStmt with the
// matched statement and its guard; later phases fill in the rest.
struct OpInfo {
  String id;
  Stmt stmt;                    // the single matched original statement
  Optional<PrimExpr> guard;     // original guard, if any
  AccessSet access;             // operands (whole buffers) and their defs
  OpAtom atom = OpAtom::kSync;  // classified once by BuildOpAtoms
  PipelineIndex write_def = -1; // the written operands' single def
  RoleIndex role_idx = -1;      // the single owning role
  ScopeIndex sched_scope = -1;  // scope whose body references this op
};

// Orders a pipeline's uses by op id: deterministic (pointer order varies
// run to run), and one entry per op — an op signals a pipeline once no
// matter how many of its operands the pipeline protects.
struct OpIdLess {
  bool operator()(const OpInfo *a, const OpInfo *b) const {
    return a->id < b->id;
  }
};
using UseSet = std::set<const OpInfo *, OpIdLess>;

// How one side (full or empty) of a pipeline's barrier pair is signaled.
// An mbarrier phase completes only when the pending arrival count AND
// the transaction count both reach zero, and the init count must be
// >= 1. So:
//  1. Collect the atoms of the signaling role's uses of the side
//     (producer uses signal full, consumer uses signal empty) into the
//     three flags below.
//  2. Transactions alone cannot close the arrival side, so a
//     transaction-only side falls back to the per-thread arrive. Then
//       count = has_async_arrival + has_thread_arrive * role threads.
// The commit/release entry emits exactly these arrivals: at most one
// tcgen05.commit (covering all outstanding MMAs) plus at most one
// per-thread ptx_arrive_barrier.
struct BarrierSidePlan {
  bool has_transaction = false;   // TMA transactions ride the tx-count
  bool has_async_arrival = false; // one tcgen05.commit arrive::one covers all
  bool has_thread_arrive = false; // per-thread arrive at the sync entry
  RoleIndex signal_role_idx = -1; // the signaling role
  int64_t count = 0;              // resulting mbarrier.init arrival count
};

struct PipelineSpec {
  String name;
  int depth = 1;
  std::vector<Buffer> buffers; // original buffers
  Buffer full, empty;          // materialized barrier buffers
  // Ops whose operands this pipeline protects: producers write them,
  // consumers read them.
  UseSet producers, consumers;
  const UseSet &Uses(bool producer_side) const {
    return producer_side ? producers : consumers;
  }
  // Derived from the signaling ops' atoms (never user-specified).
  BarrierSidePlan full_plan, empty_plan;
};

struct SyncEntry {
  WSSyncKind kind; // set at parse time; never null afterwards
  PipelineIndex pipeline_idx = -1;
  int stage = 0;
};

struct BodyEntry {
  enum Kind { kOp, kScope, kSync };
  Kind kind = kOp;
  String id; // op id or scope id
  SyncEntry sync;
};

struct ScopeSpec {
  String id;
  For orig_loop; // undefined for the root scope (until matched, for others)
  // The scope whose bodies reference this one (-1 for the root); unique,
  // and checked against the loop's lexical parent during matching.
  ScopeIndex parent = -1;
  // Per-role instruction sequence, indexed by role index (position in the
  // warp-sorted roles_ vector); empty when the role has no body here.
  std::vector<std::vector<BodyEntry>> bodies;

  // The kernel's implicit root is the only scope without a loop:
  // MatchKernel requires every other scope to match one.
  bool IsRoot() const { return !orig_loop.defined(); }
};

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
    MatchKernel();
    PlanVersionedBuffers();
    BuildUseChains();
    BuildOpAtoms();
    PlanArriveCounts();
    VerifySchedule();
    return RebuildBlock(EmitRoleBranches());
  }

private:
  // ---- schedule parsing ---------------------------------------------------

  // Resolve a schedule-referenced buffer to the block's own allocation. The
  // schedule holds the buffer object captured at trace time; match by
  // identity first, then by data var, then by name.
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
        << "ws_schedule: num_warps must be a multiple of 4 (setmaxnreg acts "
           "on whole warpgroups), got "
        << num_warps_;

    // Roles.
    for (const WSRole &r : sched->roles) {
      RoleSpec role;
      role.name = r->name;
      role.warp_lo = static_cast<int>(r->warp_lo);
      role.warp_hi = static_cast<int>(r->warp_hi);
      role.nreg = static_cast<int>(r->max_nreg);
      ICHECK_GE(role.warp_lo, 0) << "ws_schedule: role " << role.name
                                 << " has a negative warp range start";
      ICHECK_LT(role.warp_lo, role.warp_hi)
          << "ws_schedule: role " << role.name << " has an empty warp range";
      ICHECK_LE(role.warp_hi, num_warps_)
          << "ws_schedule: role " << role.name << " exceeds num_warps";
      roles_.push_back(std::move(role));
    }
    std::stable_sort(roles_.begin(), roles_.end(),
                     [](const RoleSpec &a, const RoleSpec &b) {
                       return a.warp_lo < b.warp_lo;
                     });
    for (size_t i = 1; i < roles_.size(); ++i) {
      ICHECK_LE(roles_[i - 1].warp_hi, roles_[i].warp_lo)
          << "ws_schedule: roles " << roles_[i - 1].name << " ["
          << roles_[i - 1].warp_lo << ", " << roles_[i - 1].warp_hi << ") and "
          << roles_[i].name << " [" << roles_[i].warp_lo << ", "
          << roles_[i].warp_hi << ") have overlapping warp ranges";
    }

    // setmaxnreg allocates registers for the whole warpgroup: all four
    // warps 4g..4g+3 must request the same budget (or none of them any).
    // Roles sharing a warpgroup must therefore agree on max_nreg; idle
    // warps adopt their warpgroup's request, and a fully idle warpgroup
    // donates down to the smallest donor budget.
    warpgroup_nreg_.assign(num_warps_ / 4, -1); // -1 = no covering role
    for (const RoleSpec &r : roles_) {
      for (int w = r.warp_lo; w < r.warp_hi; ++w) {
        int &wg = warpgroup_nreg_[w / 4];
        if (wg == -1) {
          wg = r.nreg;
        } else {
          ICHECK_EQ(wg, r.nreg)
              << "ws_schedule: role " << r.name << " requests max_nreg "
              << r.nreg << " but another role of warpgroup " << w / 4
              << " (warps " << w / 4 * 4 << ".." << w / 4 * 4 + 3
              << ") requests " << wg << " (0 = none); setmaxnreg "
              << "allocates per warpgroup, so all four warps must "
              << "allocate or deallocate the same number of registers";
        }
      }
    }
    int donor = 0;
    for (const RoleSpec &r : roles_) {
      if (r.nreg > 0 && r.NregAction() == 0)
        donor = donor == 0 ? r.nreg : std::min(donor, r.nreg);
    }
    for (int &v : warpgroup_nreg_) {
      if (v == -1)
        v = donor;
    }

    // Pipelines. Buffers are typed references; match them to the block's
    // allocations by data var (the schedule may hold the pre-trace proxy of
    // the same allocation).
    for (const WSPipeline &p : sched->pipelines) {
      PipelineSpec pipeline;
      pipeline.name = p->name;
      pipeline.depth = static_cast<int>(p->depth);
      ICHECK_GE(pipeline.depth, 1);
      for (const tirx::Buffer &b : p->buffers)
        pipeline.buffers.push_back(FindBlockBuffer(b));
      pipelines_.push_back(std::move(pipeline));
    }

    // Scopes.
    for (const WSScope &s : sched->scopes) {
      ScopeSpec scope;
      scope.id = s->id;
      scope.bodies.resize(roles_.size());
      for (const auto &[role_name, instrs] : s->bodies) {
        RoleIndex role_idx = RoleIndexOf(role_name);
        ICHECK_GE(role_idx, 0)
            << "ws_schedule: scope " << scope.id
            << " has a body for unknown role '" << role_name << "'";
        std::vector<BodyEntry> entries;
        for (const WSInstr &instr : instrs) {
          BodyEntry entry;
          if (const auto *op_ref = instr.as<WSOpRefNode>()) {
            entry.kind = BodyEntry::kOp; // refined to kScope below
            entry.id = op_ref->id;
          } else if (const auto *sync = instr.as<WSSyncNode>()) {
            entry.kind = BodyEntry::kSync;
            entry.sync.kind = sync->kind;
            entry.sync.stage = static_cast<int>(sync->stage);
            entry.sync.pipeline_idx = PipelineIndexOf(sync->pipeline);
            ICHECK_GE(entry.sync.pipeline_idx, 0)
                << "ws_schedule: unknown pipeline '" << sync->pipeline
                << "' in scope " << scope.id;
          } else {
            LOG(FATAL) << "ws_schedule: unknown instruction type "
                       << instr->GetTypeKey();
          }
          entries.push_back(std::move(entry));
        }
        scope.bodies[role_idx] = std::move(entries);
      }
      scopes_.push_back(std::move(scope));
    }

    // Classify body entries as op or child-scope references and validate
    // the reference graph: an op is placed exactly once (its statement is
    // cloned verbatim, so repetition would duplicate work); a child scope
    // has exactly one parent scope, is entered at most once per role, and
    // is never the root.
    std::set<std::pair<ScopeIndex, RoleIndex>> scope_refs;
    for (size_t si = 0; si < scopes_.size(); ++si) {
      ScopeSpec &scope = scopes_[si];
      for (size_t ri = 0; ri < scope.bodies.size(); ++ri) {
        for (BodyEntry &e : scope.bodies[ri]) {
          if (e.kind != BodyEntry::kOp)
            continue;
          ScopeIndex child_idx = ScopeIndexOf(e.id);
          if (child_idx >= 0) {
            e.kind = BodyEntry::kScope;
            ICHECK(e.id != kWSRootScopeId)
                << "ws_schedule: the root scope cannot be referenced from a "
                   "scope body";
            ScopeSpec &child = scopes_[child_idx];
            if (child.parent < 0) {
              child.parent = static_cast<ScopeIndex>(si);
            } else {
              ICHECK_EQ(child.parent, static_cast<ScopeIndex>(si))
                  << "ws_schedule: scope '" << e.id
                  << "' is referenced from multiple parent scopes ('"
                  << scopes_[child.parent].id << "' and '" << scope.id << "')";
            }
            ICHECK(scope_refs.insert({child_idx, static_cast<RoleIndex>(ri)})
                       .second)
                << "ws_schedule: scope '" << e.id
                << "' is referenced more than once by role " << roles_[ri].name
                << "; each participating role enters a scope exactly once "
                   "per parent iteration";
            continue;
          }
          auto it = ops_.find(e.id);
          if (it == ops_.end()) {
            OpInfo op;
            op.id = e.id;
            op.role_idx = static_cast<RoleIndex>(ri);
            op.sched_scope = static_cast<ScopeIndex>(si);
            ops_.emplace(e.id, std::move(op));
          } else {
            ICHECK_EQ(it->second.role_idx, static_cast<RoleIndex>(ri))
                << "ws_schedule: op '" << e.id
                << "' is scheduled by multiple roles";
            ICHECK_EQ(it->second.sched_scope, static_cast<ScopeIndex>(si))
                << "ws_schedule: op '" << e.id
                << "' is scheduled in multiple scopes";
            LOG(FATAL) << "ws_schedule: op '" << e.id
                       << "' is referenced more than once in the body of "
                          "scope '"
                       << scope.id << "'; an op is placed exactly once";
          }
        }
      }
    }
    // An unreferenced scope would never be emitted, silently dropping the
    // work its bodies schedule.
    for (const ScopeSpec &scope : scopes_) {
      ICHECK(scope.parent >= 0 || scope.id == kWSRootScopeId)
          << "ws_schedule: scope '" << scope.id
          << "' is never referenced from a parent scope body";
    }
  }

  ScopeIndex ScopeIndexOf(const String &id) const {
    for (size_t i = 0; i < scopes_.size(); ++i)
      if (scopes_[i].id == id)
        return static_cast<ScopeIndex>(i);
    return -1;
  }

  RoleIndex RoleIndexOf(const String &name) const {
    for (size_t i = 0; i < roles_.size(); ++i)
      if (roles_[i].name == name)
        return static_cast<RoleIndex>(i);
    return -1;
  }

  PipelineIndex PipelineIndexOf(const String &name) const {
    for (size_t i = 0; i < pipelines_.size(); ++i)
      if (pipelines_[i].name == name)
        return static_cast<PipelineIndex>(i);
    return -1;
  }

  // ---- op matching --------------------------------------------------------

  // Walk the kernel body against the schedule. Coverage is two-sided:
  // every declared op must match exactly one statement in its scheduling
  // scope, every scope must find its loop, and — enforced in MatchTask —
  // every kernel statement must carry an id, except inside op-node loops,
  // which their loop's id covers.
  void MatchKernel() {
    ScopeIndex root = ScopeIndexOf(kWSRootScopeId);
    ICHECK_GE(root, 0) << "ws_schedule: missing root scope";
    MatchScopeBody(root, block_->body);
    for (auto &[id, op] : ops_) {
      ICHECK(op.stmt.defined())
          << "ws_schedule: op '" << id
          << "' is scheduled but no statement in the kernel carries this id";
      QueryAccess(op.stmt, &op.access);
      // A buffer read only by the op's guard is an operand too: it needs
      // span protection and version rebinding like the body.
      if (op.guard.defined())
        QueryAccess(Evaluate(op.guard.value()), &op.access);
    }
    for (const ScopeSpec &scope : scopes_) {
      ICHECK(scope.orig_loop.defined() || scope.id == kWSRootScopeId)
          << "ws_schedule: scope '" << scope.id
          << "' has no matching loop in the kernel";
    }
  }

  void MatchScopeBody(ScopeIndex scope_idx, const Stmt &body) {
    for (const Stmt &task : BodyStmts(body))
      MatchTask(scope_idx, task, Optional<PrimExpr>());
  }

  // The statements of a body: a SeqStmt's list, or the single statement
  // itself. Not recursive — traced TIR does not nest SeqStmts.
  static Array<Stmt> BodyStmts(const Stmt &s) {
    if (const auto *seq = s.as<SeqStmtNode>())
      return seq->seq;
    return {s};
  }

  void RecordScopeLoop(ScopeIndex scope_idx, const For &loop) {
    ICHECK(!scopes_[scope_idx].orig_loop.defined())
        << "ws_schedule: scope " << scopes_[scope_idx].id << " matched twice";
    // Any serial loop can be a scope (T.Pipelined is just a serial loop
    // with a num_stages annotation, which this pass consumes — the
    // software pipelining happens here). T.unroll / T.Parallel loops are
    // single op nodes.
    ICHECK(loop->kind == ForKind::kSerial)
        << "ws_schedule: scope '" << scopes_[scope_idx].id
        << "' must be a serial loop (T.Pipelined or T.serial); T.unroll / "
           "T.Parallel loops are scheduled as single ops";
    // Phases count iterations as (loop_var - min); a non-unit step would
    // break version indices and parities. TODO: divide by the step.
    ICHECK(loop->HasTrivialStep())
        << "ws_schedule: scope '" << scopes_[scope_idx].id
        << "' has a non-unit loop step; write the loop with unit step and "
           "scale the indices in the body instead";
    scopes_[scope_idx].orig_loop = loop;
    MatchScopeBody(scope_idx, loop->body);
  }

  // Consume the id marker as the statement is recorded, so the emitted
  // clones never leak scheduling metadata downstream. Ids live in tile-op
  // call annotations or loop annotations; the T.ws_op wrapper was already
  // consumed by the matcher, so a recorded Bind carries nothing.
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

  void RecordOpStmt(ScopeIndex scope_idx, const String &id, const Stmt &stmt,
                    const Optional<PrimExpr> &guard) {
    auto it = ops_.find(id);
    ICHECK(it != ops_.end()) << "ws_schedule: statement carries ws op id '"
                             << id << "' but the schedule never places it:\n"
                             << stmt;
    OpInfo &op = it->second;
    ICHECK(!op.stmt.defined())
        << "ws_schedule: two statements carry ws op id '" << id
        << "'; every scheduled statement needs its own id";
    ICHECK_EQ(scope_idx, op.sched_scope)
        << "ws_schedule: op '" << id << "' lives in scope '"
        << scopes_[scope_idx].id << "' of the kernel but is scheduled by "
        << "scope '" << scopes_[op.sched_scope].id << "'";
    op.stmt = StripOpId(stmt);
    if (guard.defined())
      op.guard = guard;
  }

  void MatchTask(ScopeIndex scope_idx, const Stmt &task,
                 Optional<PrimExpr> guard) {
    ScopeSpec &scope = scopes_[scope_idx];

    // A manual annotation (`with T.ws_op(id):`) for the one statement kind
    // that cannot carry annotations= itself — the scalar Bind. The wrapper
    // is consumed here; the op records the wrapped statement.
    if (const auto *attr = task.as<AttrStmtNode>()) {
      if (attr->attr_key == kWSOpIdKey) {
        String id = ExtractWSOpId(Any(attr->value));
        ICHECK(!attr->body.as<SeqStmtNode>())
            << "ws_schedule: T.ws_op('" << id << "') must annotate exactly "
            << "one statement; every scheduled statement needs its own id";
        ICHECK_LT(ScopeIndexOf(id), 0)
            << "ws_schedule: scope id '" << id << "' belongs on the loop's "
            << "own annotations, not on a T.ws_op wrapper";
        RecordOpStmt(scope_idx, id, attr->body, guard);
        return;
      }
      LOG(FATAL) << "ws_schedule: unexpected AttrStmt '" << attr->attr_key
                 << "' in kernel body";
    }

    // Loop with a ws op id annotation. A serial loop whose id names a
    // schedule scope opens that scope; any other annotated loop (T.unroll,
    // T.Parallel) is one opaque op node — its body is cloned wholesale and
    // statements inside it need no ids of their own.
    if (const auto *loop = task.as<ForNode>()) {
      if (auto id_ann = loop->annotations.Get(kWSOpIdKey)) {
        String id = ExtractWSOpId(id_ann.value());
        ScopeIndex child = ScopeIndexOf(id);
        if (child >= 0) {
          ICHECK(!guard.defined())
              << "ws_schedule: guarded scope loops are not supported";
          ICHECK_EQ(scopes_[child].parent, scope_idx)
              << "ws_schedule: scope '" << id << "' lives in scope '"
              << scope.id << "' of the kernel but is referenced from scope '"
              << scopes_[scopes_[child].parent].id << "'";
          RecordScopeLoop(child, GetRef<For>(loop));
          return;
        }
        RecordOpStmt(scope_idx, id, task, guard);
        return;
      }
    }

    // A single tile-op statement carrying its id in the call annotations
    // is an op, treated as an opaque node.
    if (const auto *ev = task.as<EvaluateNode>()) {
      if (const auto *call = ev->value.as<CallNode>()) {
        if (auto id = call->annotations.Get(kWSOpIdKey)) {
          RecordOpStmt(scope_idx, ExtractWSOpId(id.value()), task, guard);
          return;
        }
      }
    }

    // Dig through source guards: an op inside an IfThenElse records the
    // accumulated condition, which emission folds into its entry guard.
    if (const auto *ite = task.as<IfThenElseNode>()) {
      ICHECK(!ite->else_case.defined())
          << "ws_schedule: else branches are not supported yet";
      PrimExpr cond =
          guard.defined() ? (guard.value() && ite->condition) : ite->condition;
      for (const Stmt &sub : BodyStmts(ite->then_case))
        MatchTask(scope_idx, sub, cond);
      return;
    }

    // Nothing is silently dropped or implicitly replicated: every
    // statement of a scheduled kernel must be placed by the schedule.
    LOG(FATAL) << "ws_schedule: statement in scope '" << scope.id
               << "' carries no ws op id; tile ops and loops take "
                  "annotations={\"tl.ws_op_id\": ...}, other statements "
                  "(a scalar Bind) are wrapped in T.ws_op(...):\n"
               << task;
  }

  // ---- planning -----------------------------------------------------------

  void PlanVersionedBuffers() {
    for (PipelineIndex i = 0; i < static_cast<PipelineIndex>(pipelines_.size());
         ++i) {
      PipelineSpec &pipeline = pipelines_[i];
      for (const Buffer &buf : pipeline.buffers) {
        ICHECK(!buffer_pipeline_.count(buf))
            << "ws_schedule: buffer " << buf->name
            << " belongs to multiple pipelines";
        buffer_pipeline_[buf] = i;
        if (pipeline.depth > 1) {
          ObjectPtr<BufferNode> n = make_object<BufferNode>(*buf.get());
          n->shape.insert(n->shape.begin(),
                          IntImm(DataType::Int(32), pipeline.depth));
          if (!n->strides.empty()) {
            PrimExpr stride0 = n->strides[0] * n->shape[1];
            n->strides.insert(n->strides.begin(), std::move(stride0));
          }
          versioned_[buf] = Buffer(std::move(n));
        }
      }
      pipeline.full =
          CreateMBarrierBuffer(pipeline.name + "_full", pipeline.depth);
      pipeline.empty =
          CreateMBarrierBuffer(pipeline.name + "_empty", pipeline.depth);
    }
  }

  // ---- def/use chains -------------------------------------------------------

  // Resolve each operand's defining pipeline and build the pipelines'
  // reverse use sets. An op's written operands must all resolve to ONE
  // pipeline: the completion signal behind a producer commit (a TMA
  // transaction in particular is bound to a single mbarrier at issue
  // time) can only trigger one. Read operands are unconstrained — one
  // asynchronous commit may back releases of several pipelines.
  void BuildUseChains() {
    for (auto &[id, op] : ops_) {
      Buffer write_buf;
      for (auto &[buf, def] : op.access.writes) {
        auto it = buffer_pipeline_.find(buf);
        if (it == buffer_pipeline_.end())
          continue;
        def = it->second;
        pipelines_[def].producers.insert(&op);
        if (op.write_def < 0) {
          op.write_def = def;
          write_buf = buf;
        } else {
          ICHECK_EQ(op.write_def, def)
              << "ws_schedule: op '" << id << "' writes " << write_buf->name
              << " of pipeline '" << pipelines_[op.write_def].name << "' and "
              << buf->name << " of pipeline '" << pipelines_[def].name
              << "'; an op's synchronization can only trigger one pipeline, "
                 "so split the op";
        }
      }
      for (auto &[buf, def] : op.access.reads) {
        auto it = buffer_pipeline_.find(buf);
        if (it == buffer_pipeline_.end())
          continue;
        def = it->second;
        pipelines_[def].consumers.insert(&op);
      }
    }
  }

  // ---- op atoms -------------------------------------------------------------

  void BuildOpAtoms() {
    for (auto &[id, op] : ops_) {
      const auto *ev = op.stmt.as<EvaluateNode>();
      if (ev && ev->value.as<CallNode>()) {
        op.atom = ClassifyCall(Downcast<Call>(ev->value), target_);
        continue;
      }
      // A compound statement (an op-node loop) stays synchronous; an
      // asynchronous instruction nested inside one could not be wired to
      // its barrier, so it is rejected. Plain reference locals: C++17
      // lambdas cannot capture structured bindings.
      const String &op_id = id;
      const Stmt &stmt = op.stmt;
      PostOrderVisit(stmt, [&](const ObjectRef &node) {
        const auto *call = node.as<CallNode>();
        if (!call || call->op.same_as(RegionOp()))
          return;
        ICHECK(ClassifyCall(GetRef<Call>(call), target_) == OpAtom::kSync)
            << "ws_schedule: op '" << op_id << "' nests an asynchronous "
            << "instruction inside a compound statement; make it a "
            << "directly scheduled op so its barrier can be wired:\n"
            << stmt;
      });
    }
  }

  // Fill each pipeline's two BarrierSidePlans (see the struct comment):
  // find the unique signaling role of each side, collect the atoms of
  // its uses, and derive the arrival count.
  void PlanArriveCounts() {
    // The signaling role of a side is the role holding its commit /
    // release entries; it must be unique.
    for (const ScopeSpec &scope : scopes_) {
      for (size_t ri = 0; ri < scope.bodies.size(); ++ri) {
        for (const BodyEntry &e : scope.bodies[ri]) {
          if (e.kind != BodyEntry::kSync)
            continue;
          if (e.sync.kind.IsWait())
            continue;
          PipelineSpec &pipeline = pipelines_[e.sync.pipeline_idx];
          bool producer_side = e.sync.kind.IsProducerCommit();
          BarrierSidePlan &plan =
              producer_side ? pipeline.full_plan : pipeline.empty_plan;
          if (plan.signal_role_idx < 0) {
            plan.signal_role_idx = static_cast<RoleIndex>(ri);
          } else {
            ICHECK_EQ(plan.signal_role_idx, static_cast<RoleIndex>(ri))
                << "ws_schedule: pipeline " << pipeline.name
                << " is committed/released by multiple roles";
          }
        }
      }
    }
    for (PipelineIndex p = 0; p < static_cast<PipelineIndex>(pipelines_.size());
         ++p) {
      PipelineSpec &pipeline = pipelines_[p];
      for (bool producer_side : {true, false}) {
        BarrierSidePlan &plan =
            producer_side ? pipeline.full_plan : pipeline.empty_plan;
        if (plan.signal_role_idx < 0)
          continue; // reported below
        for (const OpInfo *op : pipeline.Uses(producer_side)) {
          if (op->role_idx != plan.signal_role_idx)
            continue;
          switch (op->atom) {
          case OpAtom::kTmaCopy:
            plan.has_transaction = true;
            break;
          case OpAtom::kTcgen05Gemm:
            plan.has_async_arrival = true;
            break;
          case OpAtom::kSync:
            plan.has_thread_arrive = true;
            break;
          }
        }
        // Transactions cannot close the arrival side by themselves;
        // without any other arrival, fall back to the per-thread arrive.
        if (plan.has_transaction && !plan.has_async_arrival &&
            !plan.has_thread_arrive) {
          plan.has_thread_arrive = true;
        }
        const RoleSpec &role = roles_[plan.signal_role_idx];
        plan.count = (plan.has_async_arrival ? 1 : 0) +
                     (plan.has_thread_arrive ? role.NumThreads() : 0);
      }
    }
    for (PipelineSpec &pipeline : pipelines_) {
      ICHECK_GT(pipeline.full_plan.count, 0)
          << "ws_schedule: pipeline " << pipeline.name << " has no producer";
      ICHECK_GT(pipeline.empty_plan.count, 0)
          << "ws_schedule: pipeline " << pipeline.name << " has no consumer";
    }
  }

  // ---- schedule verification ------------------------------------------------
  //
  // A broken schedule would hang or race on the GPU; these checks reject
  // it at compile time instead.

  // Per role: sync brackets must pair up (an acquire/wait opens a span,
  // the matching commit/release closes it in the same scope body), a
  // role is the producer or the consumer of a pipeline but never both (a
  // role handing data to itself would need no pipeline), and every op
  // must run inside an open span of each pipeline protecting one of its
  // operands — an access outside the span races with the other side of
  // the handshake, before the wait or after the release alike.
  //
  // A span covers reads and writes both: the flavor is the role's
  // handshake direction, not the op's access direction (a consumer may
  // write what it waited on, a producer may read what it acquired).
  // Spans opened in an enclosing scope cover ops in child scopes.
  void VerifySpanCoverage() const {
    for (RoleIndex ri = 0; ri < static_cast<RoleIndex>(roles_.size()); ++ri) {
      std::map<PipelineIndex, bool> flavor; // pipeline -> role is producer
      std::set<PipelineIndex> open; // union of all enclosing bodies' spans
      std::function<void(ScopeIndex)> walk = [&](ScopeIndex scope_idx) {
        const ScopeSpec &scope = scopes_[scope_idx];
        std::set<PipelineIndex> local; // spans opened in THIS body
        for (const BodyEntry &e : scope.bodies[ri]) {
          if (e.kind == BodyEntry::kSync) {
            PipelineIndex p = e.sync.pipeline_idx;
            const String &pname = pipelines_[p].name;
            auto [fit, first] = flavor.emplace(p, e.sync.kind.IsProducer());
            ICHECK(first || fit->second == e.sync.kind.IsProducer())
                << "ws_schedule: role " << roles_[ri].name
                << " is both a producer and a consumer of pipeline '" << pname
                << "'; the flavors are the two parties of the handshake, and "
                   "a role handing data to itself needs no pipeline";
            if (e.sync.kind.IsWait()) {
              ICHECK(!open.count(p))
                  << "ws_schedule: pipeline '" << pname << "' acquired twice "
                  << "without an intervening commit/release in role "
                  << roles_[ri].name;
              open.insert(p);
              local.insert(p);
            } else {
              ICHECK(local.count(p))
                  << "ws_schedule: commit/release of pipeline '" << pname
                  << "' in role " << roles_[ri].name
                  << " has no matching acquire/wait in the same scope body";
              open.erase(p);
              local.erase(p);
            }
            continue;
          }
          if (e.kind == BodyEntry::kScope) {
            walk(ScopeIndexOf(e.id));
            continue;
          }
          const OpInfo &op = ops_.at(e.id);
          auto check_operand = [&](const Buffer &buf, PipelineIndex def,
                                   const char *how) {
            if (def < 0)
              return;
            ICHECK(open.count(def))
                << "ws_schedule: op '" << op.id << "' in role "
                << roles_[ri].name << " " << how << " " << buf->name
                << " outside an open span of pipeline '" << pipelines_[def].name
                << "'; bracket the op with producer_acquire/producer_commit "
                << "or consumer_wait/consumer_release (the stage may be "
                << "concurrently overwritten or still read by an "
                << "asynchronous task)";
          };
          for (const auto &[buf, def] : op.access.writes)
            check_operand(buf, def, "writes");
          for (const auto &[buf, def] : op.access.reads)
            check_operand(buf, def, "reads");
        }
        ICHECK(local.empty())
            << "ws_schedule: role " << roles_[ri].name << " leaves pipeline '"
            << pipelines_[*local.begin()].name
            << "' acquired at the end of a scope body; every acquire/wait "
               "must be paired with a commit/release in the same body";
      };
      walk(ScopeIndexOf(kWSRootScopeId));
    }
  }

  // Within every loop scope, a pipeline's producer and consumer sides
  // must cycle equally often per iteration: an imbalance drifts the
  // full/empty parity further apart every trip, and a side whose
  // counterpart cycles in a different loop diverges with the extents.
  // Balance also makes each iteration leave the barrier state unchanged,
  // which is what lets the deadlock model run every loop for just two
  // iterations. Deliberately strict: schedules balanced only through
  // particular loop extents are rejected too.
  void VerifyCycleBalance() const {
    for (size_t si = 0; si < scopes_.size(); ++si) {
      const ScopeSpec &scope = scopes_[si];
      if (scope.IsRoot())
        continue; // the root runs once; the deadlock model covers it exactly
      std::map<PipelineIndex, std::array<int, 2>>
          cycles; // [consumer, producer]
      for (RoleIndex ri = 0; ri < static_cast<RoleIndex>(roles_.size()); ++ri) {
        for (const BodyEntry &e : scope.bodies[ri]) {
          if (e.kind == BodyEntry::kSync && e.sync.kind.IsCommit())
            cycles[e.sync.pipeline_idx][e.sync.kind.IsProducer() ? 1 : 0]++;
        }
      }
      for (const auto &[pipeline_idx, sides] : cycles) {
        ICHECK_EQ(sides[1], sides[0])
            << "ws_schedule: pipeline '" << pipelines_[pipeline_idx].name
            << "' cycles " << sides[1] << " time(s) on the producer side but "
            << sides[0] << " time(s) on the consumer side per iteration of "
            << "scope '" << scope.id << "'; the full/empty parity diverges "
            << "as the loop advances — cycle both sides equally often "
            << "within the scope";
      }
    }
  }

  // Flatten one role's sync entries into the order the emitted code
  // executes them. Every loop is modeled for two iterations — warm-up
  // plus one steady-state trip; VerifyCycleBalance makes further
  // iterations replay the steady state — and the stage deltas reproduce
  // the software-pipelined reordering rather than the textual order (an
  // entry at delta d executes at steps d .. d + trips - 1). Sync entries
  // are unconditional in the emitted code, so the model needs no
  // conditional events.
  std::vector<SyncEntry> FlattenRoleEvents(RoleIndex role_idx) const {
    std::vector<SyncEntry> events;
    std::function<void(ScopeIndex)> emit = [&](ScopeIndex scope_idx) {
      const ScopeSpec &scope = scopes_[scope_idx];
      const std::vector<BodyEntry> &entries = scope.bodies[role_idx];
      if (entries.empty())
        return;
      StagePlan plan = PlanStages(role_idx, entries);
      int trips = scope.IsRoot() ? 1 : 2;
      int steps = scope.IsRoot() ? 1 : trips + plan.shift;
      for (int t = 0; t < steps; ++t) {
        for (size_t i = 0; i < entries.size(); ++i) {
          if (!scope.IsRoot() &&
              (t < plan.delta[i] || t >= plan.delta[i] + trips))
            continue; // outside this entry's boundary guards
          const BodyEntry &e = entries[i];
          if (e.kind == BodyEntry::kScope)
            emit(ScopeIndexOf(e.id));
          else if (e.kind == BodyEntry::kSync)
            events.push_back(e.sync);
        }
      }
    };
    emit(ScopeIndexOf(kWSRootScopeId));
    return events;
  }

  // Execute all roles' sync events under the mbarrier parity model:
  //  - a role's (k+1)-th wait on a pipeline is enabled once total commits
  //    reach k+1 (the (k+1)-th full phase completed);
  //  - its (k+1)-th acquire is enabled once depth + total releases reach
  //    k+1 (the first `depth` acquires pass on the initial parity);
  //  - commits and releases never block.
  // Parity waits consume nothing and the totals only grow, so an enabled
  // event can never become disabled: one greedy execution reaches the
  // same end state as every interleaving, and a role still blocked at
  // quiescence is a real hang, reported with its blocking event.
  void VerifyDeadlockFree() const {
    int n_roles = static_cast<int>(roles_.size());
    std::vector<std::vector<SyncEntry>> events(n_roles);
    for (RoleIndex r = 0; r < n_roles; ++r)
      events[r] = FlattenRoleEvents(r);

    std::vector<int> cursors(n_roles, 0);
    // commits[p] / releases[p]: totals across all roles so far; waits and
    // acquires are counted per (role, pipeline) observer.
    std::map<PipelineIndex, int> commits, releases;
    std::vector<std::map<PipelineIndex, int>> waits(n_roles), acquires(n_roles);

    auto enabled = [&](int r) {
      const SyncEntry &e = events[r][cursors[r]];
      if (e.kind.IsConsumerWait())
        return waits[r][e.pipeline_idx] < commits[e.pipeline_idx];
      if (e.kind.IsProducerAcquire())
        return acquires[r][e.pipeline_idx] <
               static_cast<int>(pipelines_[e.pipeline_idx].depth) +
                   releases[e.pipeline_idx];
      return true; // commit / release never block
    };

    bool progressed = true;
    while (progressed) {
      progressed = false;
      for (int r = 0; r < n_roles; ++r) {
        while (cursors[r] < static_cast<int>(events[r].size()) && enabled(r)) {
          const SyncEntry &e = events[r][cursors[r]];
          if (e.kind.IsProducerAcquire())
            acquires[r][e.pipeline_idx] += 1;
          else if (e.kind.IsProducerCommit())
            commits[e.pipeline_idx] += 1;
          else if (e.kind.IsConsumerWait())
            waits[r][e.pipeline_idx] += 1;
          else
            releases[e.pipeline_idx] += 1;
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
      const SyncEntry &e = events[r][cursors[r]];
      blocked << "\n  role " << roles_[r].name << " blocked at " << e.kind
              << "(\"" << pipelines_[e.pipeline_idx].name << "\") after "
              << cursors[r] << " of " << events[r].size() << " sync events";
    }
    ICHECK(!deadlocked)
        << "ws_schedule: schedule deadlocks: the roles' sync events reach a "
           "state where every unfinished role is blocked (mbarrier parity "
           "model; loops modeled at two iterations):"
        << blocked.str();
  }

  void VerifySchedule() {
    VerifySpanCoverage();
    VerifyCycleBalance();
    VerifyDeadlockFree();
  }

  // ---- stage analysis -------------------------------------------------------

  // Pipelines transitively touched by a role's entries under a scope.
  std::set<PipelineIndex> ScopePipelines(RoleIndex role_idx,
                                         ScopeIndex scope_idx) const {
    std::set<PipelineIndex> result;
    const ScopeSpec &scope = scopes_[scope_idx];
    for (const BodyEntry &e : scope.bodies[role_idx]) {
      if (e.kind == BodyEntry::kOp) {
        std::set<PipelineIndex> defs = ops_.at(e.id).access.Defs();
        result.insert(defs.begin(), defs.end());
      } else if (e.kind == BodyEntry::kScope) {
        std::set<PipelineIndex> pipelines =
            ScopePipelines(role_idx, ScopeIndexOf(e.id));
        result.insert(pipelines.begin(), pipelines.end());
      }
    }
    return result;
  }

  // Per-entry stage offsets for one (role, scope) body. delta = stage -
  // s_min: an entry at delta d executes logical iteration (i - d) at
  // loop step i, so the loop gains `shift` extra steps and per-entry
  // boundary guards. A sync entry uses its own stage; an op or child
  // scope takes the stage of the open spans covering the pipelines it
  // touches, which must agree (a mix would make the loop-var
  // substitution ambiguous); entries under no open span run at delta 0.
  struct StagePlan {
    std::vector<int> delta; // per entry
    int shift = 0;          // max delta; loop extent grows by this much
  };

  StagePlan PlanStages(RoleIndex role_idx,
                       const std::vector<BodyEntry> &entries) const {
    StagePlan plan;
    plan.delta.assign(entries.size(), 0);

    int s_min = std::numeric_limits<int>::max();
    for (const BodyEntry &e : entries) {
      if (e.kind == BodyEntry::kSync)
        s_min = std::min(s_min, e.sync.stage);
    }
    if (s_min == std::numeric_limits<int>::max())
      return plan; // no syncs in this body

    // pipeline -> stage of its open span. Bracket pairing and role
    // flavor exclusivity were verified (VerifySpanCoverage); only the
    // pair's stage agreement is checked here, where stages live.
    std::map<PipelineIndex, int> pipeline_stage;
    auto span_delta = [&](const std::set<PipelineIndex> &touched_pipelines,
                          const String &what) -> int {
      int stage = std::numeric_limits<int>::min();
      for (const auto &[pipeline_idx, open_stage] : pipeline_stage) {
        if (!touched_pipelines.count(pipeline_idx))
          continue;
        if (stage == std::numeric_limits<int>::min()) {
          stage = open_stage;
        } else {
          ICHECK_EQ(stage, open_stage)
              << "ws_schedule: " << what << " in role " << roles_[role_idx].name
              << " touches pipelines whose open spans sit at different "
                 "stages; split the op or align the stages";
        }
      }
      return stage == std::numeric_limits<int>::min() ? s_min : stage;
    };

    for (size_t i = 0; i < entries.size(); ++i) {
      const BodyEntry &e = entries[i];
      if (e.kind == BodyEntry::kSync) {
        plan.delta[i] = e.sync.stage - s_min;
        if (e.sync.kind.IsWait()) {
          pipeline_stage[e.sync.pipeline_idx] = e.sync.stage;
        } else {
          // Bracket verified: the matching open exists in this body.
          auto oit = pipeline_stage.find(e.sync.pipeline_idx);
          ICHECK_EQ(oit->second, e.sync.stage)
              << "ws_schedule: acquire/wait of pipeline '"
              << pipelines_[e.sync.pipeline_idx].name << "' in role "
              << roles_[role_idx].name << " is at stage " << oit->second
              << " but its commit/release is at stage " << e.sync.stage
              << "; pairs must share one stage";
          pipeline_stage.erase(oit);
        }
      } else if (e.kind == BodyEntry::kOp) {
        plan.delta[i] = span_delta(ops_.at(e.id).access.Defs(), e.id) - s_min;
      } else { // kScope
        plan.delta[i] =
            span_delta(ScopePipelines(role_idx, ScopeIndexOf(e.id)), e.id) -
            s_min;
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
    RoleIndex role_idx = -1;
    Map<Var, PrimExpr> subs;      // orig vars -> per-role expressions
    std::vector<LoopLevel> chain; // enclosing loops, outer->inner
    // Pipeline -> phase at acquire/wait / runtime phase counter.
    std::map<PipelineIndex, PrimExpr> pipeline_phase;
    std::map<PipelineIndex, Buffer> counters;
  };

  static PrimExpr LinearPhase(const RoleCtx &ctx) {
    PrimExpr phase;
    for (const LoopLevel &lvl : ctx.chain) {
      // The phase counts completed iterations, so a non-zero loop min is
      // subtracted (T.Pipelined(3, 7) starts at phase 0, not 3).
      PrimExpr iter = is_zero(lvl.min) ? lvl.iter : lvl.iter - lvl.min;
      phase = phase.defined() ? phase * lvl.extent + iter : iter;
    }
    return phase.defined() ? phase : PrimExpr(IntImm(DataType::Int(32), 0));
  }

  // A (role, pipeline) pair needs a runtime phase counter when its
  // cycles are not one-per-iteration of a single loop: sync points at
  // several loop depths, or several cycles in one body — the loop-based
  // phase would give such cycles the same version and parity. Counters
  // are per (role, pipeline): sharing one between pipelines would
  // require them to cycle in lockstep, since every read of the counter
  // must see that pipeline's own completed-cycle count.
  bool NeedsCounter(RoleIndex role_idx, PipelineIndex pipeline_idx) const {
    std::set<ScopeIndex> sync_scopes;
    bool multi_cycle = false;
    std::function<void(ScopeIndex)> scan = [&](ScopeIndex scope_idx) {
      int closes_here = 0;
      for (const BodyEntry &e : scopes_[scope_idx].bodies[role_idx]) {
        if (e.kind == BodyEntry::kScope) {
          scan(ScopeIndexOf(e.id));
        } else if (e.kind == BodyEntry::kSync &&
                   e.sync.pipeline_idx == pipeline_idx) {
          sync_scopes.insert(scope_idx);
          if (e.sync.kind.IsCommit())
            closes_here += 1;
        }
      }
      if (closes_here > 1)
        multi_cycle = true;
    };
    scan(ScopeIndexOf(kWSRootScopeId));
    return multi_cycle || sync_scopes.size() > 1;
  }

  PrimExpr PipelinePhase(const RoleCtx &ctx, PipelineIndex pipeline_idx) const {
    auto cit = ctx.counters.find(pipeline_idx);
    if (cit != ctx.counters.end())
      return BufferLoad(cit->second, {IntImm(DataType::Int(32), 0)});
    return LinearPhase(ctx);
  }

  static Stmt MakeWait(const Buffer &bar, PrimExpr idx, PrimExpr parity) {
    return Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(),
             {BufferLoad(bar, {std::move(idx)}), std::move(parity)}));
  }

  // Arrivals for one commit/release entry, per its side plan: at most
  // one tcgen05.commit (covering all outstanding MMAs) plus at most one
  // per-thread arrive — exactly plan.count arrivals per phase.
  void MakeArrive(const PipelineSpec &pipeline, bool full, PrimExpr idx,
                  Array<Stmt> *out) const {
    const Buffer &bar = full ? pipeline.full : pipeline.empty;
    const BarrierSidePlan &plan =
        full ? pipeline.full_plan : pipeline.empty_plan;
    if (plan.has_async_arrival) {
      PrimExpr ptr = bar.access_ptr(3, DataType::Handle(), 1, idx);
      out->push_back(
          Evaluate(Call(DataType::Void(), tcgen05_mma_arrive(), {ptr})));
    }
    if (plan.has_thread_arrive) {
      out->push_back(
          Evaluate(Call(DataType::Void(), builtin::ptx_arrive_barrier(),
                        {BufferLoad(bar, {std::move(idx)})})));
    }
  }

  void EmitSync(RoleCtx &ctx, const SyncEntry &sync, Array<Stmt> *out) {
    PipelineSpec &pipeline = pipelines_[sync.pipeline_idx];
    PrimExpr depth = IntImm(DataType::Int(32), pipeline.depth);
    PrimExpr phase = PipelinePhase(ctx, sync.pipeline_idx);
    PrimExpr idx = floormod(phase, depth);
    PrimExpr parity =
        bitwise_and(floordiv(phase, depth), IntImm(DataType::Int(32), 1));

    if (sync.kind.IsProducerAcquire()) {
      ctx.pipeline_phase[sync.pipeline_idx] = std::move(phase);
      out->push_back(MakeWait(
          pipeline.empty, std::move(idx),
          bitwise_xor(std::move(parity), IntImm(DataType::Int(32), 1))));
    } else if (sync.kind.IsConsumerWait()) {
      ctx.pipeline_phase[sync.pipeline_idx] = std::move(phase);
      out->push_back(
          MakeWait(pipeline.full, std::move(idx), std::move(parity)));
    } else if (sync.kind.IsProducerCommit()) {
      MakeArrive(pipeline, /*full=*/true, std::move(idx), out);
    } else { // consumer release
      MakeArrive(pipeline, /*full=*/false, std::move(idx), out);
    }
    if (sync.kind.IsCommit() && ctx.counters.count(sync.pipeline_idx)) {
      Buffer cnt = ctx.counters.at(sync.pipeline_idx);
      PrimExpr zero = IntImm(DataType::Int(32), 0);
      out->push_back(BufferStore(cnt, BufferLoad(cnt, {zero}) + 1, {zero}));
    }
  }

  // Rebind every access of a multi-versioned buffer to the acquired
  // version. Buffer versioning only; call conversion is EmitOp's job.
  struct OpRewriter : public StmtExprMutator {
    const WSScheduleMaterializer &self;
    const RoleCtx &ctx;
    OpRewriter(const WSScheduleMaterializer &self, const RoleCtx &ctx)
        : self(self), ctx(ctx) {}

    PrimExpr VersionIndex(const Buffer &orig) {
      // versioned_ keys are pipeline buffers (PlanVersionedBuffers), and
      // span coverage verified the op runs inside an open span, so the
      // pipeline and its phase are always present.
      PipelineIndex p = self.buffer_pipeline_.at(orig);
      return floormod(ctx.pipeline_phase.at(p),
                      IntImm(DataType::Int(32), self.pipelines_[p].depth));
    }

    // The BufferLoad path below prepends the version index to a
    // versioned region's root load; detect the extra index here and
    // insert the matching unit extent to keep the region's
    // indices-per-extent invariant.
    PrimExpr VisitExpr_(const CallNode *op) final {
      Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
      if (call->op.same_as(RegionOp()) && call->args.size() >= 2) {
        if (const auto *load = call->args[0].as<BufferLoadNode>()) {
          size_t num_extents = call->args.size() - 2;
          if (load->indices.size() == num_extents + 1) {
            Array<PrimExpr> args;
            args.push_back(call->args[0]);
            args.push_back(call->args[1]);
            // The region is the op's footprint: origin = the root load's
            // indices, size = the extents. The op touches exactly the ONE
            // acquired version (origin phase % depth), so the version
            // dimension's extent is 1 — the depth lives in the versioned
            // buffer's shape, not here.
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

    PrimExpr VisitExpr_(const BufferLoadNode *op) final {
      BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
      auto vit = self.versioned_.find(load->buffer);
      if (vit == self.versioned_.end())
        return load;
      Array<PrimExpr> indices;
      indices.push_back(VersionIndex(load->buffer));
      for (const PrimExpr &i : load->indices)
        indices.push_back(i);
      return BufferLoad(vit->second, std::move(indices));
    }

    Stmt VisitStmt_(const BufferStoreNode *op) final {
      BufferStore store =
          Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
      auto vit = self.versioned_.find(store->buffer);
      if (vit == self.versioned_.end())
        return store;
      Array<PrimExpr> indices;
      indices.push_back(VersionIndex(store->buffer));
      for (const PrimExpr &i : store->indices)
        indices.push_back(i);
      return BufferStore(vit->second, store->value, std::move(indices));
    }
  };

  Stmt RewriteOpStmt(const RoleCtx &ctx, Stmt stmt) const {
    OpRewriter rewriter(*this, ctx);
    return rewriter(std::move(stmt));
  }

  PrimExpr RewriteOpExpr(const RoleCtx &ctx, PrimExpr expr) const {
    OpRewriter rewriter(*this, ctx);
    return rewriter(std::move(expr));
  }

  // Swap an asynchronous atom's call to its explicit async op. The
  // statement shape — a direct Evaluate of the call — was enforced by
  // BuildOpAtoms.
  Stmt ConvertAtomCall(const RoleCtx &ctx, const OpInfo &op, Stmt stmt) const {
    const auto *ev = stmt.as<EvaluateNode>();
    ICHECK(ev);
    Call call = Downcast<Call>(ev->value);
    auto ann = call->annotations;
    if (op.atom == OpAtom::kTmaCopy) {
      // The TMA transaction completes the full barrier of the pipeline
      // protecting the destination: the op's single write def. A TMA copy
      // into an unprotected buffer has no barrier to wire and stays a
      // plain copy for the generic lowering.
      if (op.write_def < 0)
        return stmt;
      const PipelineSpec &pipeline = pipelines_[op.write_def];
      // Span coverage verified: the pipeline was acquired.
      PrimExpr idx = floormod(ctx.pipeline_phase.at(op.write_def),
                              IntImm(DataType::Int(32), pipeline.depth));
      ann.Set("barrier", BufferLoad(pipeline.full, {std::move(idx)}));
      ann.Set("is_tma_copy", IntImm(DataType::Int(32), 1));
      return Evaluate(Call(call->dtype, TmaCopyOp(), call->args, std::move(ann),
                           call->span));
    } else if (op.atom == OpAtom::kTcgen05Gemm) {
      ann.Set("is_tcgen05", IntImm(DataType::Int(32), 1));
      return Evaluate(Call(call->dtype, Tcgen05GemmOp(), call->args,
                           std::move(ann), call->span));
    }
    LOG(FATAL) << "ws_schedule: unknown async atom "
               << static_cast<int>(op.atom);
  }

  // The op's source guard is NOT applied here: EmitScopeBody folds it into
  // the entry's guard, combined with any stage boundary guard.
  Stmt EmitOp(const RoleCtx &ctx, const OpInfo &op) const {
    Stmt body = RewriteOpStmt(ctx, Substitute(op.stmt, ctx.subs));
    if (op.atom != OpAtom::kSync)
      body = ConvertAtomCall(ctx, op, std::move(body));
    return body;
  }

  // Emit one (role, scope) body as a flat statement list for the caller
  // to splice; a SeqStmt is built only where an IfThenElse needs a single
  // body. `plan` holds the per-entry stage deltas; `loop_var` /
  // `orig_extent` describe the enclosing loop (undefined for the root),
  // whose chain level the caller has already pushed.
  Array<Stmt> EmitScopeBody(RoleCtx &ctx, ScopeIndex scope_idx,
                            const StagePlan &plan,
                            const Optional<Var> &loop_var,
                            const Optional<PrimExpr> &orig_extent,
                            const Var &orig_loop_var) {
    ScopeSpec &scope = scopes_[scope_idx];
    const std::vector<BodyEntry> &entries = scope.bodies[ctx.role_idx];
    if (entries.empty())
      return {};
    if (plan.shift > 0) {
      ICHECK(loop_var.defined())
          << "ws_schedule: stage offsets require a loop scope, but scope "
          << scope.id << " is the root";
    }

    Array<Stmt> stmts;

    // Point the substitution and phase chain at entry i's logical
    // iteration and return its guard: the stage boundary condition
    // combined with the op's own source guard (substituted with the
    // shifted iteration and version-rebound like the body). Sync entries
    // have no source guard.
    auto focus_entry = [&](size_t i) -> Optional<PrimExpr> {
      Optional<PrimExpr> guard;
      if (loop_var.defined()) {
        int delta = plan.delta[i];
        PrimExpr iter =
            delta == 0 ? PrimExpr(loop_var.value())
                       : loop_var.value() - IntImm(DataType::Int(32), delta);
        ctx.subs.Set(orig_loop_var, iter);
        ctx.chain.back().iter = std::move(iter);
        if (delta > 0)
          guard = loop_var.value() >= IntImm(DataType::Int(32), delta);
        if (delta < plan.shift) {
          PrimExpr upper =
              loop_var.value() <
              orig_extent.value() + IntImm(DataType::Int(32), delta);
          guard = guard.defined() ? guard.value() && upper : upper;
        }
      }
      const BodyEntry &e = entries[i];
      if (e.kind == BodyEntry::kOp) {
        const OpInfo &op = ops_.at(e.id);
        if (op.guard.defined()) {
          PrimExpr src =
              RewriteOpExpr(ctx, Substitute(op.guard.value(), ctx.subs));
          guard = guard.defined() ? guard.value() && src : src;
        }
      }
      return guard;
    };
    // Unguarded entries append straight to `stmts`; guarded entries keep
    // emitting into `guarded` while the guard stays structurally
    // identical, and the run closes as one IfThenElse when it changes.
    Optional<PrimExpr> cur_guard;
    Array<Stmt> guarded;
    StructuralEqual structural_equal;
    auto close_guard = [&]() {
      if (cur_guard.defined() && !guarded.empty()) {
        stmts.push_back(IfThenElse(
            cur_guard.value(),
            guarded.size() == 1 ? guarded[0] : SeqStmt(std::move(guarded))));
      }
      cur_guard = Optional<PrimExpr>();
      guarded = Array<Stmt>();
    };

    for (size_t i = 0; i < entries.size(); ++i) {
      const BodyEntry &e = entries[i];
      Optional<PrimExpr> guard = focus_entry(i);
      bool reuse = guard.defined() == cur_guard.defined() &&
                   (!guard.defined() ||
                    structural_equal(guard.value(), cur_guard.value()));
      if (!reuse) {
        close_guard();
        cur_guard = guard;
      }
      Array<Stmt> *out = cur_guard.defined() ? &guarded : &stmts;
      if (e.kind == BodyEntry::kScope) {
        Stmt child = EmitChildScope(ctx, ScopeIndexOf(e.id));
        if (child.defined())
          out->push_back(std::move(child));
      } else if (e.kind == BodyEntry::kSync) {
        EmitSync(ctx, e.sync, out);
      } else {
        out->push_back(EmitOp(ctx, ops_.at(e.id)));
      }
    }
    close_guard();
    // Restore the unshifted iteration for any parent-level use.
    if (loop_var.defined()) {
      ctx.subs.Set(orig_loop_var, loop_var.value());
      ctx.chain.back().iter = loop_var.value();
    }
    return stmts;
  }

  Stmt EmitChildScope(RoleCtx &ctx, ScopeIndex scope_idx) {
    ScopeSpec &scope = scopes_[scope_idx];
    // Matched (MatchKernel): every non-root scope has its kernel loop.
    const For &orig = scope.orig_loop;

    if (scope.bodies[ctx.role_idx].empty())
      return Stmt();

    // The loop extent may need to grow by the scope's stage shift.
    StagePlan plan = PlanStages(ctx.role_idx, scope.bodies[ctx.role_idx]);

    Var fresh(orig->loop_var->name_hint, orig->loop_var->dtype);
    ctx.subs.Set(orig->loop_var, fresh);
    PrimExpr extent = Substitute(orig->extent, ctx.subs);
    PrimExpr min = Substitute(orig->min, ctx.subs);
    if (plan.shift > 0) {
      ICHECK(is_zero(orig->min))
          << "ws_schedule: stage offsets require 0-based loops";
    }
    ctx.chain.push_back({fresh, extent, min});
    Array<Stmt> body =
        EmitScopeBody(ctx, scope_idx, plan, fresh, extent, orig->loop_var);
    ctx.chain.pop_back();
    if (body.empty())
      return Stmt(); // role does not participate in this scope
    PrimExpr emit_extent = plan.shift > 0
                               ? extent + IntImm(DataType::Int(32), plan.shift)
                               : extent;
    // Preserve the source loop kind (e.g. an unrolled epilogue subtile loop)
    // and its annotations, minus the consumed scheduling markers.
    Map<String, Any> ann =
        FilterAnnotations(orig->annotations, [](const String &key) {
          return key != kWSOpIdKey && key != "num_stages" &&
                 key.find("tl_pipeline_") != 0;
        });
    return For(fresh, Substitute(orig->min, ctx.subs), std::move(emit_extent),
               orig->kind,
               body.size() == 1 ? body[0] : SeqStmt(std::move(body)),
               std::nullopt, std::move(ann));
  }

  Stmt EmitRole(RoleIndex role_idx) {
    const RoleSpec &role = roles_[role_idx];
    RoleCtx ctx;
    ctx.role = &role;
    ctx.role_idx = role_idx;

    Array<Stmt> stmts;
    if (role.nreg > 0) {
      stmts.push_back(
          Evaluate(Call(DataType::Handle(), set_max_nreg(),
                        {IntImm(DataType::Int(32), role.nreg),
                         IntImm(DataType::Int(32), role.NregAction())})));
    }

    // Materialize runtime phase counters where linearization is unsound.
    for (PipelineIndex p = 0; p < static_cast<PipelineIndex>(pipelines_.size());
         ++p) {
      bool used = false;
      for (const ScopeSpec &scope : scopes_) {
        for (const BodyEntry &e : scope.bodies[role_idx])
          if (e.kind == BodyEntry::kSync && e.sync.pipeline_idx == p)
            used = true;
      }
      if (used && NeedsCounter(role_idx, p)) {
        Buffer cnt =
            decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                        pipelines_[p].name + "_phase", "local");
        ctx.counters[p] = cnt;
        stmts.push_back(AllocBuffer(cnt));
        stmts.push_back(BufferStore(cnt, IntImm(DataType::Int(32), 0),
                                    {IntImm(DataType::Int(32), 0)}));
      }
    }

    ScopeIndex root = ScopeIndexOf(kWSRootScopeId);
    StagePlan plan = PlanStages(role_idx, scopes_[root].bodies[role_idx]);
    Array<Stmt> body = EmitScopeBody(ctx, root, plan, Optional<Var>(),
                                     Optional<PrimExpr>(), Var());
    ICHECK(!body.empty()) << "ws_schedule: role " << role.name
                          << " has an empty root body";
    for (const Stmt &s : body)
      stmts.push_back(s);
    // The role branch is one arm of an IfThenElse: inline seq-wrap.
    return stmts.size() == 1 ? stmts[0] : SeqStmt(std::move(stmts));
  }

  // Idle warps execute their warpgroup's register request
  // (warpgroup_nreg_); a no-op when the warpgroup requests nothing.
  Stmt EmitIdle(int nreg) const {
    if (nreg == 0)
      return Evaluate(IntImm(DataType::Int(32), 0));
    return Evaluate(
        Call(DataType::Handle(), set_max_nreg(),
             {IntImm(DataType::Int(32), nreg),
              IntImm(DataType::Int(32), nreg >= kNregIncThreshold ? 1 : 0)}));
  }

  // Emit the `if (tx < ...)` role branches ordered by warp range, filling
  // gaps with idle branches — split where the warpgroup register request
  // changes, so every idle warp executes exactly its warpgroup's request.
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
    for (size_t ri = 0; ri < roles_.size(); ++ri) {
      const RoleSpec &role = roles_[ri];
      ICHECK_GE(role.warp_lo, cursor)
          << "ws_schedule: overlapping role warp ranges";
      fill_idle(cursor, role.warp_lo);
      conds.push_back(thread_var_ <
                      IntImm(DataType::Int(32), role.warp_hi * 32));
      branches.push_back(EmitRole(static_cast<RoleIndex>(ri)));
      cursor = role.warp_hi;
    }
    fill_idle(cursor, num_warps_);
    Stmt body = branches[branches.size() - 1];
    for (int i = static_cast<int>(branches.size()) - 2; i >= 0; --i)
      body = IfThenElse(conds[i], branches[i], std::move(body));
    return body;
  }

  // Rebuild the block around the emitted body: versioned buffer
  // replacements, barrier allocations with their derived init counts, and
  // the consumed schedule annotation removed.
  SBlock RebuildBlock(Stmt body) {
    Array<Buffer> allocs;
    for (const Buffer &buf : block_->alloc_buffers) {
      auto vit = versioned_.find(buf);
      allocs.push_back(vit == versioned_.end() ? buf : vit->second);
    }
    Map<Var, Array<PrimExpr>> barrier_init;
    for (const PipelineSpec &pipeline : pipelines_) {
      allocs.push_back(pipeline.full);
      allocs.push_back(pipeline.empty);
      Array<PrimExpr> full_counts, empty_counts;
      for (int i = 0; i < pipeline.depth; ++i) {
        full_counts.push_back(
            IntImm(DataType::Int(32), pipeline.full_plan.count));
        empty_counts.push_back(
            IntImm(DataType::Int(32), pipeline.empty_plan.count));
      }
      barrier_init.Set(pipeline.full->data, std::move(full_counts));
      barrier_init.Set(pipeline.empty->data, std::move(empty_counts));
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
    n->annotations = std::move(ann);
    return new_block;
  }

  // ---- state ---------------------------------------------------------------

  SBlock block_;
  Var thread_var_;
  Target target_;
  int num_warps_ = 0;
  std::vector<RoleSpec> roles_;
  std::vector<PipelineSpec> pipelines_;
  std::vector<ScopeSpec> scopes_;
  std::map<String, OpInfo> ops_;
  // Per-warpgroup register request (index = warp / 4, 0 = none), agreed
  // on by every covering role; idle warps execute it too.
  std::vector<int> warpgroup_nreg_;
  BufferDefMap buffer_pipeline_;
  BufferVersionMap versioned_;
};

// Rewrite every schedule-annotated SBlock in place — a function may hold
// several kernels, each with its own schedule. Tracking the threadIdx.x
// binding on the way down hands each materializer the binding that
// actually encloses its block, and exactly that binding is widened to
// that schedule's warp count on the way back up.
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
      // Warp roles partition threadIdx.x only: a scheduled block under a
      // multi-dimensional thread launch would mis-assign warps.
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

// Schedule and op ids come together (both present or neither): a function
// without the schedule annotation is returned untouched, and the
// materializer consumes the ids of a scheduled kernel as it clones the
// statements.
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
