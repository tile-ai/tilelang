/*!
 * \file role_based_scheduler.cc
 * \brief The "role_based" automatic warp-specialization scheduler.
 *
 * Plans a schedule in four steps:
 *
 *  1. classify: each op gets a fixed role from its lowering eligibility,
 *     using the same statement classification as ProducerConsumerWS
 *     (ws_analysis.h) — Load (a global->shared tile-op copy selecting TMA
 *     with a TMA-compatible annotated layout, or selecting cp.async, whose
 *     completion the materializer binds to the pipeline barrier via a
 *     deferred cp.async.mbarrier.arrive), MMA (tcgen05 GEMM), Store
 *     (shared->global TMA copy), Worker (everything else). Ops touching
 *     only warp-private state (scalar Binds; local / local.var /
 *     local.fragment buffers) start with no role.
 *
 *  2. role bodies are backward slices: a role runs its fixed ops plus
 *     everything they transitively read — the reaching definition of every
 *     scalar read, every writer of every private-buffer read (register
 *     values cannot cross roles), and, once a slice enters a scope, the
 *     reads of that scope's bounds and guards. Slices overlap freely (the
 *     materializer re-emits and SSA-freshens a shared private chain per
 *     role); a slice reaching an op fixed to another role rejects the
 *     kernel. Private state in no slice runs in every active role.
 *
 *  3. pipelines: shared / shared.tmem storage touched by exactly two roles
 *     alternates OWNERSHIP through per-scope pipelines, and
 *     synchronization is exactly ownership movement. Every touch of the
 *     storage has an initial and a final owning role: a leaf op owns in
 *     its single role throughout; a scope hosting a pipeline receives
 *     ownership in the producer role and surrenders it in the consumer
 *     role (the transfer inside is its own handshake, opaque to the
 *     parent). Resolving the scope tree bottom-up, a scope whose touches
 *     involve both roles hosts a pipeline: a handoff is emitted wherever
 *     the role owning at one touch's end differs from the next touch's
 *     start — including across the iteration boundary, where the last
 *     touch's final role hands back to the first touch's initial role
 *     through the barrier's cyclic phases. The producer is the role
 *     opening the iteration's first bracket: the boundary transfer's
 *     receiver when final != initial, else the other role than the
 *     bracket-free boundary-crossing one. A scope with moving ownership
 *     is spanned by the producer (its interior accesses need the
 *     enclosing hold; the consumer's ride the scope's handshake plus
 *     same-role program order). One storage may thus be bound to several
 *     pipelines whose scopes are strictly nested (the materializer
 *     multiplies their depths into the version count). Declines: a
 *     consumer touch before the producer's first (its wait's commit only
 *     happens later that iteration), and a final role that never
 *     surrenders within the scope (an unconsumed producer-run; several
 *     self-cycling same-role subtrees land here — chaining their
 *     barriers across the seam is unsupported). depth = the hosting
 *     scope's num_stages (1 elsewhere), pinned to 1 when state carries
 *     across cycles within the scope (the consumer writes or the
 *     producer reads the storage — multi-versioning would fork an
 *     accumulator's recurrence) and for every non-outermost binding of a
 *     nested storage (an enclosing access cannot name an inner slot);
 *     every sync is stage 0.
 *
 *  4. emit the typed WSSchedule. Versioning, barriers, arrive counts, and
 *     proxy fences are MaterializeWSSchedule's job.
 *
 * TODO: if-else branches; sibling-pipeline chaining (a storage cycling
 * in two sequential scopes needs the first pipeline's last release to
 * arm the second's empty side instead of pre-arming); software-pipeline
 * stage offsets (every sync is stage 0); multi-consumer fan-out; a
 * multi-warp copy role so register-staged SIMT producers can leave the
 * workers; delayed wgmma waits — wg_wait != 0 rejects today because wgmma
 * completion is the warpgroup-local wgmma.wait_group watermark (no mbarrier
 * hook), so hiding MMA latency means a wait_group with a nonzero lag before
 * a consumer release that trails its wait by an iteration, which the span
 * model cannot express yet.
 */

#include "./role_based_scheduler.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "./memory_detector.h"
#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "cuda/transform/ws_analysis.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/utils.h"
#include "transform/common/constr_visitor.h"
#include "transform/common/warp_specialize.h"

namespace tvm {
namespace tl {

namespace {

// Fixed heuristic roles, in warp order.
enum class Role : uint8_t { kWorker = 0, kLoad = 1, kMma = 2, kStore = 3 };
constexpr int kNumRoles = 4;
using RoleMask = uint8_t;

constexpr RoleMask Bit(Role role) {
  return RoleMask{1} << static_cast<unsigned>(role);
}

const char *RoleName(Role role) {
  switch (role) {
  case Role::kWorker:
    return "Worker";
  case Role::kLoad:
    return "Load";
  case Role::kMma:
    return "MMA";
  case Role::kStore:
    return "Store";
  }
  LOG(FATAL) << "invalid role";
  return "";
}

bool HasSingleBit(RoleMask mask) { return mask && !(mask & (mask - 1)); }

Role SingleRole(RoleMask mask) {
  ICHECK(HasSingleBit(mask));
  return static_cast<Role>(__builtin_ctz(mask));
}

// Warp-private storage: values there cannot be handed between roles.
bool IsPrivateBuffer(const Buffer &buffer) {
  return IsLocalBuffer(buffer, true) || IsFragmentBuffer(buffer);
}

// Storage that may carry a cross-role handoff through a pipeline.
bool IsPipelineBuffer(const Buffer &buffer) {
  return IsSharedBuffer(buffer) || IsTmemBuffer(buffer);
}

// The pipeline depth a scope contributes: its loop's num_stages, else 1.
int ScopeStages(const For &loop) {
  if (!loop.defined())
    return 1;
  if (auto value = loop->annotations.Get("num_stages")) {
    if (auto integer = value.value().try_cast<Integer>())
      return std::max<int64_t>(1, integer.value()->value);
    if (const auto *imm = value.value().as<IntImmNode>())
      return std::max<int64_t>(1, imm->value);
  }
  return 1;
}

struct SchedScope;

// One schedulable statement, reduced to the facts scheduling needs; the
// statement itself stays in the kernel body, addressed by id. A scope IS
// an op (the hierarchical collapse: at its parent's level, a scope is
// one opaque entry): SchedScope derives from SchedOp, with `roles`
// holding the participating roles and `reads` the bound/guard reads.
// Ops and scopes are uniquely owned by the scheduler and referenced by
// plain pointer.
struct SchedOp {
  ffi::String id;
  TileStmtKind kind{TileStmtKind::kConsumer};
  TileOperator tile_op;        // parsed once for a direct tile-op call
  SchedScope *parent{nullptr}; // scope whose entries reference this
  bool is_bind{false};  // scalar Bind: duplicable even when it reads globals
  bool guarded{false};  // under an if: a skipped write carries state
  bool roleless{false}; // touches only warp-private state
  RoleMask roles{0};    // a fixed single bit, or the grown set when roleless
  std::vector<Buffer> reads, writes;      // buffer accesses, guards included
  std::vector<Var> read_vars, write_vars; // scalar uses / Bind defs

  virtual ~SchedOp() = default;
  virtual SchedScope *AsScope() { return nullptr; }
  const SchedScope *AsScope() const {
    return const_cast<SchedOp *>(this)->AsScope();
  }
};

struct SchedScope : SchedOp {
  For loop;         // a sequential-loop (serial / unrolled) scope
  While while_loop; // a T.ws_op-wrapped while scope; both undefined = root
  // Ops and child scopes, in source order.
  std::vector<SchedOp *> entries;

  SchedScope *AsScope() override { return this; }

  bool IsRoot() const { return parent == nullptr; }

  // Whether this scope is `ancestor` or nested somewhere below it.
  bool IsNestedIn(const SchedScope *ancestor) const {
    for (const SchedScope *s = this; s != nullptr; s = s->parent)
      if (s == ancestor)
        return true;
    return false;
  }
};

std::string ScopeName(const SchedScope *scope) {
  return scope->IsRoot() ? "root" : std::string(scope->id);
}

struct Pipeline {
  ffi::String name;
  std::vector<Buffer> allocations; // all buffers share one barrier pair
  int depth{1};
  Role producer{Role::kWorker};
  Role consumer{Role::kWorker};
  // The single scope hosting this pipeline's brackets, keyed by entry
  // boundary (boundary b sits just before entry b). At a shared boundary,
  // closes (commit / release) emit before opens (acquire / wait); within
  // one boundary list, push order is emission order.
  const SchedScope *scope{nullptr};
  int handoffs{0};    // producer->consumer transitions: the merge shape
  bool nested{false}; // one of several bindings of its storage: never merged
  std::map<int, std::vector<WSSyncKind>> closes, opens;
};

// Append the detector's results for `stmt` to the given vectors.
void CollectAccess(const Stmt &stmt, std::vector<Buffer> *reads,
                   std::vector<Buffer> *writes, std::vector<Var> *read_vars,
                   std::vector<Var> *write_vars) {
  MemoryAccessDetector detector;
  detector.Analyze(stmt);
  for (const BufferRegion &region : detector.GetReadRegions())
    reads->push_back(region->buffer);
  if (writes) {
    for (const BufferRegion &region : detector.GetWriteRegions())
      writes->push_back(region->buffer);
  }
  for (const Var &var : detector.GetReadVars())
    read_vars->push_back(var);
  if (write_vars) {
    for (const Var &var : detector.GetWriteVars())
      write_vars->push_back(var);
  }
}

// Builds the op/scope structure from the normalized body. ConstrVisitor
// maintains the lexical constraint stack — if guards, loop ranges, while
// conditions, assumes — so an op's guards are simply the conditions on the
// stack where it appears. Ops are leaves: their statements are recorded,
// not recursed into.
class StructureBuilder : public ConstrVisitor {
public:
  explicit StructureBuilder(Target target) : target_(std::move(target)) {}

  std::vector<std::unique_ptr<SchedOp>> ops;
  std::vector<std::unique_ptr<SchedScope>> scopes; // [0] is the root

  bool Build(const Stmt &body) {
    auto root = std::make_unique<SchedScope>();
    root->id = kWSRootScopeId;
    scope_stack_.push_back(root.get());
    scopes.push_back(std::move(root));
    VisitStmt(body);
    return ok_;
  }

private:
  // Conditions evaluated on every control path to the current statement.
  // Assumes are facts, not evaluated code; binds and ranges charge their
  // reads through the statements that evaluate them.
  void ChargeConditions(std::vector<Buffer> *reads,
                        std::vector<Var> *read_vars) {
    for (const Constr &constr : constr_stack_) {
      if (constr.kind == Constr::kConstr && !constr.is_assume)
        CollectAccess(Evaluate(constr.value), reads, nullptr, read_vars,
                      nullptr);
    }
  }

  void MakeOp(const ffi::String &id, const Stmt &stmt) {
    ICHECK(op_ids_.insert(id).second)
        << "two statements carry ws op id '" << id << "'";
    auto op = std::make_unique<SchedOp>();
    op->id = id;
    op->parent = scope_stack_.back();
    op->is_bind = stmt.as<BindNode>() != nullptr;
    op->guarded = if_depth_ > 0;
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (eval->value.as<CallNode>())
        op->tile_op = ParseOperator(Downcast<Call>(eval->value));
    }
    op->kind = ClassifyStmt(stmt, target_);
    CollectAccess(stmt, &op->reads, &op->writes, &op->read_vars,
                  &op->write_vars);
    ChargeConditions(&op->reads, &op->read_vars);
    scope_stack_.back()->entries.push_back(op.get());
    ops.push_back(std::move(op));
  }

  SchedScope *OpenScope(const ffi::String &id) {
    auto scope = std::make_unique<SchedScope>();
    SchedScope *child = scope.get();
    child->id = id;
    child->parent = scope_stack_.back();
    scope_stack_.back()->entries.push_back(child);
    scopes.push_back(std::move(scope));
    scope_stack_.push_back(child);
    return child;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if (!ok_)
      return;
    if (op->else_case.defined()) {
      LOG(WARNING)
          << "AutoSchedule skipped: if-else branches are not supported";
      ok_ = false;
      return;
    }
    ++if_depth_;
    ConstrVisitor::VisitStmt_(op); // guards the branch with its condition
    --if_depth_;
  }

  void VisitStmt_(const ForNode *op) final {
    if (!ok_)
      return;
    auto id = op->annotations.Get(kWSOpIdKey);
    ICHECK(id.has_value()) << "AutoSchedule normalized loop is missing "
                           << kWSOpIdKey;
    // Sequential loops are scopes; parallel / vectorized loops are one op.
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
      MakeOp(ExtractOpId(id.value()), GetRef<For>(op));
      return;
    }
    SchedScope *child = OpenScope(ExtractOpId(id.value()));
    child->loop = GetRef<For>(op);
    CollectAccess(Evaluate(op->min), &child->reads, nullptr, &child->read_vars,
                  nullptr);
    CollectAccess(Evaluate(op->extent), &child->reads, nullptr,
                  &child->read_vars, nullptr);
    if (op->step.defined())
      CollectAccess(Evaluate(GetRef<PrimExpr>(op->step.get())), &child->reads,
                    nullptr, &child->read_vars, nullptr);
    ChargeConditions(&child->reads, &child->read_vars);
    ConstrVisitor::VisitStmt_(op); // guards the body with the loop range
    scope_stack_.pop_back();
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (!ok_)
      return;
    if (op->attr_key == kWSOpIdKey) {
      // A wrapped while is a scope; every participating role re-evaluates
      // the condition on its own duplicated state. Anything else is one
      // opaque op.
      if (const auto *wl = op->body.as<WhileNode>()) {
        SchedScope *child = OpenScope(ExtractOpId(ffi::Any(op->value)));
        child->while_loop = GetRef<While>(wl);
        CollectAccess(Evaluate(wl->condition), &child->reads, nullptr,
                      &child->read_vars, nullptr);
        ChargeConditions(&child->reads, &child->read_vars);
        ConstrVisitor::VisitStmt_(wl); // guards the body with the condition
        scope_stack_.pop_back();
        return;
      }
      MakeOp(ExtractOpId(ffi::Any(op->value)), op->body);
      return;
    }
    // Assumptions enter the constraint stack; other wrappers (kernel
    // metadata) are transparent to scheduling.
    ConstrVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (!ok_)
      return;
    if (const auto *call = op->value.as<CallNode>()) {
      if (auto id = call->annotations.Get(kWSOpIdKey)) {
        MakeOp(ExtractOpId(ffi::Any(id.value())), GetRef<Stmt>(op));
        return;
      }
    }
    LOG(FATAL) << "AutoSchedule: Evaluate carries no ws op id";
  }

  // The normalizer wraps these statement forms with an id; reaching one
  // bare is its bug.
  void VisitStmt_(const BufferStoreNode *op) final {
    LOG(FATAL) << "AutoSchedule: BufferStore carries no ws op id";
  }
  void VisitStmt_(const BindNode *op) final {
    LOG(FATAL) << "AutoSchedule: Bind carries no ws op id";
  }
  void VisitStmt_(const WhileNode *op) final {
    LOG(FATAL) << "AutoSchedule: while loop carries no ws op id";
  }
  void VisitStmt_(const SBlockNode *op) final {
    LOG(FATAL) << "AutoSchedule: block carries no ws op id";
  }

  Target target_;
  bool ok_ = true;   // cleared when the kernel shape is unsupported
  int if_depth_ = 0; // enclosing IfThenElse frames
  std::vector<SchedScope *> scope_stack_;
  std::set<ffi::String> op_ids_;
};

// Plans the schedule for one normalized kernel body. Every unsupported
// shape declines the kernel with an on-site warning: auto scheduling is
// opt-in, so the user expects it to fire.
class RoleBasedSchedulerImpl {
public:
  RoleBasedSchedulerImpl(Target target, int worker_threads)
      : target_(std::move(target)), worker_threads_(worker_threads) {}

  Optional<WSSchedule> Run(const SBlock &block, const Stmt &body) {
    if (!TargetHasBulkCopy(target_)) {
      LOG(WARNING) << "AutoSchedule skipped: target has no bulk-copy (TMA) "
                      "support";
      return Optional<WSSchedule>();
    }
    if (worker_threads_ % 128 != 0) {
      LOG(WARNING) << "AutoSchedule skipped: worker threads must be a multiple "
                      "of 128 so issuer warps start a fresh warpgroup";
      return Optional<WSSchedule>();
    }
    CollectAnnotatedLayouts(block, layout_map_);
    if (!BuildStructure(body) || !ClassifyOps())
      return Optional<WSSchedule>();
    IndexBufferUses();
    if (!PlaceSlices())
      return Optional<WSSchedule>();
    // A scope's roles are its participants: every op marks its ancestor
    // chain, making the scope one opaque entry of those roles.
    for (const auto &op : ops_) {
      for (SchedScope *s = op->parent; s != nullptr; s = s->parent)
        s->roles |= op->roles;
    }
    if (!BuildPipelines(block))
      return Optional<WSSchedule>();
    if (pipelines_.empty()) {
      LOG(WARNING) << "AutoSchedule skipped: no cross-role on-chip handoff to "
                      "pipeline";
      return Optional<WSSchedule>();
    }
    return Emit();
  }

private:
  // ---- structure --------------------------------------------------------

  bool BuildStructure(const Stmt &body) {
    StructureBuilder builder(target_);
    if (!builder.Build(body))
      return false;
    ops_ = std::move(builder.ops);
    scopes_ = std::move(builder.scopes);
    root_ = scopes_.front().get();
    return true;
  }

  // ---- classification -----------------------------------------------------

  // A TMA-classified copy is only a Load when its destination's annotated
  // layout (if any) is TMA-expressible; otherwise lowering falls back to a
  // normal copy and the op is ordinary worker work.
  bool TmaDstLayoutCompatible(const SchedOp &op) const {
    const auto *copy = op.tile_op.as<CopyNode>();
    if (copy == nullptr)
      return true; // Im2Col carries no annotated shared layout
    auto it = layout_map_.find(copy->dst->data);
    if (it == layout_map_.end())
      return true;
    const auto &[buffer, layout] = it->second;
    return layout.defined() && IsTmaCompatibleLayout(layout, buffer);
  }

  bool ClassifyOps() {
    for (const auto &op_ptr : ops_) {
      SchedOp &op = *op_ptr;
      switch (op.kind) {
      case TileStmtKind::kTmaProducer:
        op.roles =
            TmaDstLayoutCompatible(op) ? Bit(Role::kLoad) : Bit(Role::kWorker);
        continue;
      case TileStmtKind::kCpAsyncProducer:
        // Asynchronous: the materializer suppresses the copy's implicit
        // commit/wait and the pipeline commit arrives through a deferred
        // cp.async.mbarrier.arrive.
        op.roles = Bit(Role::kLoad);
        continue;
      case TileStmtKind::kCpAsyncRaw:
        // The source may rely on the legacy pass to synthesize the
        // thread-local completion protocol this scheduler cannot see.
        LOG(WARNING) << "AutoSchedule skipped: op '" << op.id
                     << "': raw cp.async statements carry their own "
                        "thread-local completion protocol";
        return false;
      case TileStmtKind::kSimtProducer:
        // Register-staged and blocking; a one-warp Load role would
        // serialize the copy. TODO: multi-warp copy role.
        op.roles = Bit(Role::kWorker);
        continue;
      case TileStmtKind::kTmaStore:
        op.roles = Bit(Role::kStore);
        continue;
      case TileStmtKind::kTcgen05Mma:
        op.roles = Bit(Role::kMma);
        continue;
      case TileStmtKind::kConsumer:
        break;
      }

      if (op.tile_op.defined()) {
        // A wgmma gemm with wg_wait != 0 returns with MMAs still reading
        // its operands (completion is the manual T.wait_wgmma watermark);
        // the release this scheduler would place right after the gemm
        // races with them.
        if (const auto *gemm = op.tile_op.as<GemmNode>()) {
          if (gemm->wgWait_ != 0) {
            LOG(WARNING) << "AutoSchedule skipped: op '" << op.id
                         << "': gemm with wg_wait != 0 completes "
                            "asynchronously; delayed wgmma waits are not "
                            "supported yet";
            return false;
          }
        }
        op.roles = Bit(Role::kWorker); // any other tile op is SIMT work
        continue;
      }

      // Compound or plain statement op. PreprocessIR enforced that it
      // hosts no asynchronous tile op.
      bool private_writes = true;
      for (const Buffer &buffer : op.writes)
        private_writes = private_writes && IsPrivateBuffer(buffer);
      bool duplicable_reads = true;
      for (const Buffer &buffer : op.reads) {
        duplicable_reads =
            duplicable_reads &&
            (IsPrivateBuffer(buffer) || (op.is_bind && IsGlobalBuffer(buffer)));
      }
      if (private_writes && duplicable_reads)
        op.roleless = true; // placed by demand propagation
      else
        op.roles = Bit(Role::kWorker);
    }
    return true;
  }

  // ---- role placement: backward slices --------------------------------------

  // A role's body is the backward slice of its fixed ops: everything they
  // transitively read. An op's dependencies are the reaching definition of
  // each scalar read (a Bind var has one def) and every writer of each
  // private-buffer read (its value may come from any of them); the moment a
  // slice enters a scope, the reads of that scope's bounds and guards join
  // too (every participating role evaluates them). Slices may overlap: the
  // materializer re-emits and SSA-freshens a shared private chain per role.
  bool PlaceSlices() {
    // The reaching definition of each Bind var; private-buffer writers
    // come from the def-use index.
    std::unordered_map<Var, SchedOp *, ObjectPtrHash, ObjectPtrEqual> bind_def;
    for (const auto &op : ops_) {
      for (const Var &var : op->write_vars)
        bind_def.emplace(var, op.get());
    }

    auto slice = [&](Role role, const std::vector<SchedOp *> &seeds) {
      RoleMask bit = Bit(role);
      std::vector<SchedOp *> queue;
      for (SchedOp *op : seeds) {
        op->roles |= bit;
        queue.push_back(op);
      }
      auto enqueue = [&](SchedOp *def, const ffi::String &user) {
        if (def->roles & bit)
          return true;
        if (!def->roleless) {
          LOG(WARNING) << "AutoSchedule skipped: '" << user << "' needs op '"
                       << def->id << "' in role " << RoleName(role)
                       << ", but that op is fixed to role "
                       << RoleName(SingleRole(def->roles));
          return false;
        }
        def->roles |= bit;
        queue.push_back(def);
        return true;
      };
      auto enqueue_reads = [&](const std::vector<Buffer> &reads,
                               const std::vector<Var> &read_vars,
                               const SchedOp *self, const ffi::String &user) {
        for (const Var &var : read_vars) {
          auto it = bind_def.find(var);
          if (it != bind_def.end() && it->second != self &&
              !enqueue(it->second, user))
            return false;
        }
        for (const Buffer &buffer : reads) {
          if (!IsPrivateBuffer(buffer))
            continue;
          auto it = buffer_uses_.find(buffer->data);
          if (it == buffer_uses_.end())
            continue;
          for (SchedOp *writer : it->second.writers) {
            if (writer != self && !enqueue(writer, user))
              return false;
          }
        }
        return true;
      };
      std::unordered_set<const SchedScope *> entered;
      while (!queue.empty()) {
        SchedOp *op = queue.back();
        queue.pop_back();
        // An entered scope implies every ancestor is entered; the root
        // has no bounds to charge.
        for (SchedScope *s = op->parent; s != nullptr && !entered.count(s);
             s = s->parent) {
          entered.insert(s);
          if (!enqueue_reads(s->reads, s->read_vars, /*self=*/nullptr, s->id))
            return false;
        }
        if (!enqueue_reads(op->reads, op->read_vars, op, op->id))
          return false;
      }
      return true;
    };

    for (int r = 0; r < kNumRoles; ++r) {
      Role role = static_cast<Role>(r);
      std::vector<SchedOp *> seeds;
      for (const auto &op : ops_) {
        if (!op->roleless && op->roles == Bit(role))
          seeds.push_back(op.get());
      }
      if (!seeds.empty() && !slice(role, seeds))
        return false;
    }

    // Private state in no slice runs in every active role; its own reads
    // follow through the same closure.
    RoleMask active = 0;
    for (const auto &op : ops_)
      active |= op->roles;
    if (active == 0) {
      LOG(WARNING) << "AutoSchedule skipped: kernel has no schedulable work";
      return false;
    }
    std::vector<SchedOp *> leftovers;
    for (const auto &op : ops_) {
      if (op->roles == 0)
        leftovers.push_back(op.get());
    }
    for (int r = 0; !leftovers.empty() && r < kNumRoles; ++r) {
      Role role = static_cast<Role>(r);
      if ((active & Bit(role)) && !slice(role, leftovers))
        return false;
    }
    return true;
  }

  // ---- def-use index --------------------------------------------------------

  // Program-ordered users of one buffer, indexed once from the ops'
  // collected accesses; every later phase queries these lists instead of
  // re-scanning ops or the IR.
  struct BufferUses {
    std::vector<SchedOp *> touches; // readers and writers, program order
    std::vector<SchedOp *> writers;
    std::vector<SchedOp *> readers;
  };

  void IndexBufferUses() {
    auto touch = [&](const Buffer &buffer, SchedOp *op) -> BufferUses & {
      BufferUses &uses = buffer_uses_[buffer->data];
      if (uses.touches.empty() || uses.touches.back() != op)
        uses.touches.push_back(op);
      return uses;
    };
    for (const auto &op : ops_) {
      for (const Buffer &buffer : op->reads)
        touch(buffer, op.get()).readers.push_back(op.get());
      for (const Buffer &buffer : op->writes)
        touch(buffer, op.get()).writers.push_back(op.get());
    }
  }

  // ---- pipelines -----------------------------------------------------------

  // The ownership of one storage across an entry's lifetime: the role
  // that must own the storage when the entry begins and the role owning
  // it when the entry ends. A leaf op owns in its single role
  // throughout. A scope hosting a pipeline receives ownership in the
  // producer role and surrenders it in the consumer role — the transfer
  // inside is the scope's own handshake, opaque to its parent. Zero
  // masks = the storage is untouched below the entry.
  struct Ownership {
    RoleMask initial{0};
    RoleMask final{0};
  };

  // Resolves one storage's ownership movement (or the union of several
  // storages when probing a merge) into pipelines, bottom-up over the
  // scope tree; `pipelines` collects them innermost first.
  struct StorageResolution {
    ffi::String name; // for decline messages
    std::vector<Buffer> allocations;
    std::unordered_set<const SchedOp *> users;
    RoleMask parties{0};             // the storage's two roles
    std::vector<Pipeline> pipelines; // discovered pipelines
    bool failed{false};
  };

  StorageResolution MakeResolution(ffi::String name,
                                   std::vector<Buffer> allocations) const {
    StorageResolution resolution;
    resolution.name = std::move(name);
    resolution.allocations = std::move(allocations);
    for (const Buffer &allocation : resolution.allocations) {
      auto it = buffer_uses_.find(allocation->data);
      if (it == buffer_uses_.end())
        continue;
      resolution.users.insert(it->second.touches.begin(),
                              it->second.touches.end());
    }
    for (const SchedOp *op : resolution.users)
      resolution.parties |= op->roles;
    return resolution;
  }

  // Synchronization is ownership movement. Walking the scope tree
  // bottom-up once for all resolutions, each scope body is a cyclic
  // sequence of touches: a handoff belongs wherever the role owning at
  // one touch's end differs from the role owning at the next touch's
  // start — INCLUDING across the iteration boundary, where the last
  // touch's final role hands back to the first touch's initial role
  // through the barrier's cyclic phases. A scope whose touches involve
  // both roles hosts a pipeline here; otherwise its subtree's ownership
  // passes to its parent as one opaque touch. Returns each resolution's
  // ownership of the scope's subtree.
  std::vector<Ownership>
  ResolveOwnership(std::vector<StorageResolution> *resolutions,
                   const SchedScope &scope, bool log_reject) const {
    struct Touch {
      int pos;
      Ownership own;
    };
    size_t n = resolutions->size();
    std::vector<std::vector<Touch>> touches(n);
    for (int e = 0; e < static_cast<int>(scope.entries.size()); ++e) {
      SchedOp *entry = scope.entries[e];
      if (const SchedScope *child = entry->AsScope()) {
        std::vector<Ownership> below =
            ResolveOwnership(resolutions, *child, log_reject);
        for (size_t k = 0; k < n; ++k) {
          if (below[k].initial)
            touches[k].push_back({e, below[k]});
        }
      } else {
        for (size_t k = 0; k < n; ++k) {
          if ((*resolutions)[k].users.count(entry))
            touches[k].push_back({e, {entry->roles, entry->roles}});
        }
      }
    }

    std::vector<Ownership> proj(n);
    for (size_t k = 0; k < n; ++k) {
      StorageResolution &resolution = (*resolutions)[k];
      if (resolution.failed || touches[k].empty())
        continue;
      // A lone touch cycles by itself: a leaf holds throughout, and a
      // nested pipeline's barriers already carry its final role back to
      // its initial role across iterations.
      if (touches[k].size() == 1) {
        proj[k] = touches[k].front().own;
        continue;
      }
      RoleMask roles = 0;
      for (const Touch &t : touches[k])
        roles |= t.own.initial | t.own.final;
      if (HasSingleBit(roles)) {
        // One role throughout: its own program order carries ownership;
        // an ancestor may bracket the whole subtree as one touch.
        proj[k] = {roles, roles};
        continue;
      }
      // Both roles: this scope hosts a pipeline. The producer is the
      // role opening the iteration's first bracket: when the iteration
      // boundary is itself a transfer (final of the last touch differs
      // from initial of the first), the receiving role opens first;
      // otherwise the boundary-crossing role carries ownership
      // bracket-free — its own program order and its scopes' handshakes
      // order it — and the other role opens.
      RoleMask initial = touches[k].front().own.initial;
      RoleMask final = touches[k].back().own.final;
      RoleMask producer_bit =
          final != initial ? initial : resolution.parties & ~initial;
      ICHECK(HasSingleBit(producer_bit));
      Pipeline pipeline;
      pipeline.allocations = resolution.allocations;
      pipeline.scope = &scope;
      pipeline.producer = SingleRole(producer_bit);
      pipeline.consumer = SingleRole(resolution.parties & ~producer_bit);
      int state = 0; // 1 = producer-run open, 2 = consumer-run open
      int run_last = -1;
      bool ok = true;
      for (const Touch &t : touches[k]) {
        // A leaf is held by its role. A scope whose ownership moves
        // inside is spanned by the PRODUCER: the producer's interior
        // accesses need the enclosing hold, while the consumer's ride
        // the scope's own handshake plus program order.
        RoleMask holder =
            t.own.initial == t.own.final ? t.own.initial : producer_bit;
        int side = holder == producer_bit ? 1 : 2;
        if (side == state) {
          run_last = t.pos;
          continue;
        }
        if (state == 0 && side == 2) {
          // The consumer's touch would wait for a commit that only
          // happens later in the same iteration.
          if (log_reject)
            LOG(WARNING) << "AutoSchedule skipped: storage '" << resolution.name
                         << "' is touched by its consumer role before its "
                            "producer role in scope '"
                         << scope.id << "'";
          resolution.failed = true;
          ok = false;
          break;
        }
        if (state == 1)
          pipeline.closes[run_last + 1].push_back(WSSyncKind::ProducerCommit());
        if (state == 2)
          pipeline.closes[run_last + 1].push_back(
              WSSyncKind::ConsumerRelease());
        if (side == 1) {
          pipeline.opens[t.pos].push_back(WSSyncKind::ProducerAcquire());
        } else {
          pipeline.opens[t.pos].push_back(WSSyncKind::ConsumerWait());
          ++pipeline.handoffs;
        }
        state = side;
        run_last = t.pos;
      }
      if (!ok)
        continue;
      if (state == 1) {
        // The final role never surrenders ownership within the scope;
        // the commit would only be consumed on the next iteration and
        // the parity never returns to empty. (Several self-cycling
        // subtrees of one role land here too: chaining their barriers
        // across the seam is not supported.)
        if (log_reject)
          LOG(WARNING) << "AutoSchedule skipped: storage '" << resolution.name
                       << "' ends scope '" << scope.id
                       << "' with an unconsumed producer-run; the consumer "
                          "never takes the data within the scope";
        resolution.failed = true;
        continue;
      }
      pipeline.closes[run_last + 1].push_back(WSSyncKind::ConsumerRelease());
      proj[k] = {initial, final};
      resolution.pipelines.push_back(std::move(pipeline));
    }
    return proj;
  }

  // Depth and the guarded-writer check for one hosted pipeline, from the
  // storage's uses inside the hosting scope's subtree. `pin_single` keeps
  // a non-outermost binding of a nested storage at depth 1: an access at
  // the enclosing level could not name an inner slot.
  bool FinalizePipeline(Pipeline *pipeline, bool pin_single,
                        bool log_reject) const {
    RoleMask writer_mask = 0, reader_mask = 0;
    bool guarded_writer = false;
    for (const Buffer &allocation : pipeline->allocations) {
      auto it = buffer_uses_.find(allocation->data);
      if (it == buffer_uses_.end())
        continue;
      for (const SchedOp *op : it->second.writers) {
        if (!op->parent->IsNestedIn(pipeline->scope))
          continue;
        writer_mask |= op->roles;
        guarded_writer = guarded_writer || op->guarded;
      }
      for (const SchedOp *op : it->second.readers) {
        if (op->parent->IsNestedIn(pipeline->scope))
          reader_mask |= op->roles;
      }
    }
    // State carrying across cycles (the consumer writes the storage, or
    // the producer reads it) pins the pipeline single-buffered:
    // multi-versioning would fork an accumulator's recurrence.
    bool carries_state = (writer_mask & Bit(pipeline->consumer)) != 0 ||
                         (reader_mask & Bit(pipeline->producer)) != 0;
    if (!carries_state && !pin_single)
      pipeline->depth = ScopeStages(pipeline->scope->loop);
    // Versioning changes the meaning of a skipped write: the surviving
    // value would be the slot from `depth` iterations ago, not the
    // previous iteration's. Single-buffered pipelines keep source
    // semantics exactly.
    if (guarded_writer && pipeline->depth > 1) {
      if (log_reject)
        LOG(WARNING) << "AutoSchedule skipped: storage '" << pipeline->name
                     << "' is written under a guard and would be "
                     << pipeline->depth
                     << "-way versioned; a skipped write would expose a "
                        "stale version instead of the previous value";
      return false;
    }
    return true;
  }

  bool BuildPipelines(const SBlock &block) {
    // A global buffer written by one role must not be touched by another;
    // there is no cross-role synchronization mechanism for globals.
    for (const auto &writer : ops_) {
      for (const Buffer &buffer : writer->writes) {
        if (!IsGlobalBuffer(buffer))
          continue;
        RoleMask others = 0;
        for (const SchedOp *op : buffer_uses_[buffer->data].touches)
          others |= op->roles;
        if (others & ~writer->roles) {
          LOG(WARNING) << "AutoSchedule skipped: global buffer '"
                       << buffer->name << "' is written by op '" << writer->id
                       << "' and touched by another role";
          return false;
        }
      }
    }

    std::vector<StorageResolution> resolutions;
    for (const Buffer &allocation : block->alloc_buffers) {
      if (!IsPipelineBuffer(allocation))
        continue;
      auto it = buffer_uses_.find(allocation->data);
      if (it == buffer_uses_.end() || it->second.writers.empty())
        continue; // never written by a scheduled op
      RoleMask toucher_mask = 0;
      for (const SchedOp *op : it->second.touches) {
        ICHECK(HasSingleBit(op->roles))
            << "op '" << op->id << "' touches shared storage '"
            << allocation->name << "' but is placed by several roles";
        toucher_mask |= op->roles;
      }
      if (HasSingleBit(toucher_mask))
        continue; // role-private storage needs no pipeline
      if (__builtin_popcount(toucher_mask) > 2) {
        LOG(WARNING) << "AutoSchedule skipped: storage '" << allocation->name
                     << "' is shared by more than two roles";
        return false;
      }
      resolutions.push_back(MakeResolution(allocation->name, {allocation}));
    }
    if (resolutions.empty())
      return true; // Run declines when nothing needs a pipeline

    ResolveOwnership(&resolutions, *root_, /*log_reject=*/true);
    for (StorageResolution &resolution : resolutions) {
      if (resolution.failed)
        return false;
      // A two-role storage's ownership must alternate at some scope of
      // the walk or fail it.
      ICHECK(!resolution.pipelines.empty());
      for (size_t h = 0; h < resolution.pipelines.size(); ++h) {
        Pipeline &pipeline = resolution.pipelines[h];
        // The innermost pipeline keeps the storage name; enclosing
        // pipelines carry their scope's id.
        pipeline.name = h == 0 ? std::string(resolution.name)
                               : std::string(resolution.name) + "_" +
                                     ScopeName(pipeline.scope);
        pipeline.nested = resolution.pipelines.size() > 1;
        bool pin_single = h + 1 < resolution.pipelines.size();
        if (!FinalizePipeline(&pipeline, pin_single, /*log_reject=*/true))
          return false;
        pipelines_.push_back(std::move(pipeline));
      }
    }
    MergePipelines();
    return true;
  }

  // Two pipelines behave identically when they share roles, depth, scope,
  // and handoff count. Merging them puts all their buffers behind ONE
  // barrier pair — half the waits and arrives per iteration, the shape a
  // hand-written multi-buffer pipeline uses. The union access sequence is
  // re-walked; a group whose union interleaves irregularly (e.g. buffer B
  // is rewritten between buffer A's write and read) stays split, as do
  // the pipelines of a nested-bound storage.
  static bool SameShape(const Pipeline &a, const Pipeline &b) {
    return a.producer == b.producer && a.consumer == b.consumer &&
           a.depth == b.depth && a.scope == b.scope && a.handoffs == b.handoffs;
  }

  void MergePipelines() {
    std::vector<Pipeline> merged;
    std::vector<bool> used(pipelines_.size(), false);
    for (size_t i = 0; i < pipelines_.size(); ++i) {
      if (used[i])
        continue;
      std::vector<size_t> group{i};
      if (!pipelines_[i].nested) {
        for (size_t j = i + 1; j < pipelines_.size(); ++j) {
          if (!used[j] && !pipelines_[j].nested &&
              SameShape(pipelines_[i], pipelines_[j]))
            group.push_back(j);
        }
      }
      if (group.size() > 1) {
        std::string name;
        std::vector<Buffer> allocations;
        for (size_t j : group) {
          const Buffer &allocation = pipelines_[j].allocations.front();
          allocations.push_back(allocation);
          name += (name.empty() ? "" : "_") + std::string(allocation->name);
        }
        // A failed union is not a kernel rejection — keep the group split
        // (and log nothing).
        std::vector<StorageResolution> probe;
        probe.push_back(MakeResolution(name, std::move(allocations)));
        ResolveOwnership(&probe, *root_, /*log_reject=*/false);
        StorageResolution &resolution = probe.front();
        bool ok = !resolution.failed && resolution.pipelines.size() == 1;
        if (ok) {
          Pipeline &unioned = resolution.pipelines.front();
          unioned.name = name;
          ok = FinalizePipeline(&unioned, /*pin_single=*/false,
                                /*log_reject=*/false) &&
               SameShape(unioned, pipelines_[i]);
          if (ok) {
            for (size_t j : group)
              used[j] = true;
            merged.push_back(std::move(unioned));
            continue;
          }
        }
      }
      used[i] = true;
      merged.push_back(std::move(pipelines_[i]));
    }
    pipelines_ = std::move(merged);
  }

  // ---- emission ------------------------------------------------------------

  WSSchedule Emit() const {
    RoleMask active = 0;
    for (const auto &op : ops_)
      active |= op->roles;

    Array<WSRole> roles;
    int worker_warps = worker_threads_ / 32;
    int cursor = 0;
    if (active & Bit(Role::kWorker)) {
      roles.push_back(WSRole(RoleName(Role::kWorker), 0, worker_warps, 0));
      cursor = worker_warps;
    }
    for (Role role : {Role::kLoad, Role::kMma, Role::kStore}) {
      if (active & Bit(role)) {
        roles.push_back(WSRole(RoleName(role), cursor, cursor + 1, 32));
        ++cursor;
      }
    }
    int num_warps = (cursor + 3) / 4 * 4;

    Array<WSPipeline> ws_pipelines;
    for (const Pipeline &pipeline : pipelines_) {
      ws_pipelines.push_back(
          WSPipeline(pipeline.name,
                     Array<Buffer>(pipeline.allocations.begin(),
                                   pipeline.allocations.end()),
                     pipeline.depth));
    }

    // Scopes, innermost first, root last.
    Array<WSScope> scopes;
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it)
      scopes.push_back(EmitScope(**it, active));

    return WSSchedule(num_warps, roles, ws_pipelines, scopes);
  }

  // A bracket belongs to the producer or the consumer by its kind.
  static Role SyncRole(const Pipeline &pipeline, const WSSyncKind &kind) {
    return kind.IsProducerAcquire() || kind.IsProducerCommit()
               ? pipeline.producer
               : pipeline.consumer;
  }

  WSScope EmitScope(const SchedScope &scope, RoleMask active) const {
    int num_entries = static_cast<int>(scope.entries.size());
    Map<ffi::String, Array<WSInstr>> bodies;
    for (int r = 0; r < kNumRoles; ++r) {
      Role role = static_cast<Role>(r);
      if (!(active & Bit(role)))
        continue;
      std::vector<char> mine(num_entries, 0);
      int count = 0;
      for (int e = 0; e < num_entries; ++e) {
        mine[e] = (scope.entries[e]->roles & Bit(role)) != 0;
        count += mine[e];
      }
      if (count == 0)
        continue;

      // At every boundary: this role's closes (from all pipelines, in
      // order), then its opens, then the entry itself when it is ours.
      Array<WSInstr> body;
      auto append = [&](int boundary, bool close) {
        for (const Pipeline &pipeline : pipelines_) {
          if (pipeline.scope != &scope)
            continue;
          const auto &at = close ? pipeline.closes : pipeline.opens;
          auto it = at.find(boundary);
          if (it == at.end())
            continue;
          for (const WSSyncKind &kind : it->second) {
            if (SyncRole(pipeline, kind) == role)
              body.push_back(WSSync(kind, pipeline.name, /*stage=*/0));
          }
        }
      };
      for (int b = 0; b <= num_entries; ++b) {
        append(b, /*close=*/true);
        append(b, /*close=*/false);
        if (b < num_entries && mine[b])
          body.push_back(WSOpRef(scope.entries[b]->id));
      }
      bodies.Set(RoleName(role), body);
    }
    return WSScope(scope.id, bodies);
  }

  Target target_;
  int worker_threads_;
  BufferLayoutMap layout_map_;
  std::vector<std::unique_ptr<SchedOp>> ops_;
  std::vector<std::unique_ptr<SchedScope>> scopes_; // [0] is the root
  SchedScope *root_ = nullptr;
  std::unordered_map<Var, BufferUses, ObjectPtrHash, ObjectPtrEqual>
      buffer_uses_;
  std::vector<Pipeline> pipelines_;
};

} // namespace

ffi::Optional<WSSchedule> RoleBasedSchedule(const SBlock &block,
                                            const Stmt &body,
                                            int worker_threads,
                                            const Target &target) {
  return RoleBasedSchedulerImpl(target, worker_threads).Run(block, body);
}

} // namespace tl
} // namespace tvm
