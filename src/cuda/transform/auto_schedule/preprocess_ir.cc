/*!
 * \file preprocess_ir.cc
 * \brief Give every schedulable statement a stable "tl.ws_op_id" marker.
 *
 * The markers use the same surface forms a hand-written schedule uses:
 * call annotations on tile ops, loop annotations on For loops (sequential
 * kinds — serial, unrolled — are scopes; parallel and vectorized loops are
 * single opaque ops), and a `T.ws_op` AttrStmt wrapper on everything else
 * (Binds, stores, nested blocks, while-loop scopes).
 * Generated ids carry a name hint — the callee's op name, the bound var,
 * the stored buffer — so schedules and diagnostics stay readable.
 * Existing ids are reused, so normalization is idempotent.
 *
 * Normalization also enforces the schedulability contract, failing hard
 * (auto scheduling is opt-in): no loop_break, no atomics, and no
 * asynchronous tile op nested inside an opaque op — the scheduler must
 * see it as a directly scheduled statement to wire its barriers.
 */

#include "./preprocess_ir.h"

#include <tvm/ir/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include "./common.h"
#include "cuda/transform/ws_analysis.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// The callee's op name with "." replaced by "_".
std::string CallKind(const CallNode *call) {
  if (const auto *op = call->op.as<OpNode>()) {
    std::string name = op->name;
    std::replace(name.begin(), name.end(), '.', '_');
    return name;
  }
  return "op";
}

// Constructs the materializer cannot schedule at all, anywhere in the body.
void VerifyCalls(const Stmt &body) {
  PostOrderVisit(body, [](const ffi::ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr)
      return;
    ICHECK(!call->op.same_as(tl::loop_break()))
        << "AutoSchedule cannot schedule loop_break";
    ICHECK(!IsBarrierOrTmaControlCall(call))
        << "AutoSchedule cannot schedule hand-written synchronization or "
           "TMA control ('"
        << call->op << "'): block-wide syncs cannot be duplicated into "
        << "roles, and hand-managed barrier protocols are invisible to "
        << "the schedule";
    if (const auto *op = call->op.as<OpNode>()) {
      ICHECK(std::string(op->name).find("atomic") == std::string::npos)
          << "AutoSchedule cannot schedule atomic op '" << op->name << "'";
    }
  });
}

// A tile op whose completion the schedule must track through pipeline
// barriers (g2s async producer, tmem or wg_wait gemm) cannot nest inside
// an opaque op; it must be a directly scheduled statement. TMA stores are
// fine: their completion rides the copy lowering's own commit-group
// machinery.
void VerifyHostsNoAsync(const Stmt &stmt, const Target &target) {
  PostOrderVisit(stmt, [&](const ffi::ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr)
      return;
    TileOperator tile_op = ParseOperator(GetRef<Call>(call));
    if (!tile_op.defined())
      return;
    if (const auto *copy = tile_op.as<CopyNode>()) {
      ICHECK(ClassifyCopy(copy, target) != TileStmtKind::kTmaProducer)
          << "AutoSchedule: an asynchronous global->shared copy is nested "
             "inside a compound statement; write it as its own statement "
             "so its completion barrier can be wired";
      return;
    }
    if (const auto *gemm = tile_op.as<GemmNode>()) {
      ICHECK(!IsTmemBuffer(gemm->cRegion_->buffer) && gemm->wgWait_ == 0)
          << "AutoSchedule: an asynchronous gemm is nested inside a "
             "compound statement; write it as its own statement";
    }
  });
}

class TaskNormalizer : public StmtMutator {
public:
  static Stmt Rewrite(Stmt body, const Target &target) {
    TaskNormalizer normalizer(target);
    normalizer.CollectIds(body);
    return normalizer(std::move(body));
  }

private:
  explicit TaskNormalizer(Target target) : target_(std::move(target)) {}

  // A simple wrapper around one call is a direct op (ClassifyStmt sees
  // through it); only genuinely compound statements are opaque.
  void VerifyOpaqueOp(const Stmt &stmt) const {
    if (!GetEvaluateCallInSimpleWrapper(stmt).defined())
      VerifyHostsNoAsync(stmt, target_);
  }
  void CollectIds(const Stmt &stmt) {
    PostOrderVisit(stmt, [&](const ffi::ObjectRef &node) {
      if (const auto *attr = node.as<AttrStmtNode>()) {
        if (attr->attr_key == kWSOpIdKey)
          used_ids_.insert(ExtractOpId(ffi::Any(attr->value)));
      } else if (const auto *loop = node.as<ForNode>()) {
        if (auto value = loop->annotations.Get(kWSOpIdKey))
          used_ids_.insert(ExtractOpId(value.value()));
      } else if (const auto *call = node.as<CallNode>()) {
        if (auto value = call->annotations.Get(kWSOpIdKey))
          used_ids_.insert(ExtractOpId(ffi::Any(value.value())));
      }
    });
  }

  ffi::String FreshId(const std::string &kind) {
    while (true) {
      std::ostringstream os;
      os << kind << "_" << next_id_[kind]++;
      ffi::String id(os.str());
      if (used_ids_.insert(id).second)
        return id;
    }
  }

  Stmt WrapWithId(const Stmt &stmt, const std::string &kind) {
    return AttrStmt(Integer(0), kWSOpIdKey, StringImm(FreshId(kind)), stmt,
                    stmt->span);
  }

  Stmt VisitStmt(const Stmt &stmt) final {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == kWSOpIdKey) {
        // A while loop under the wrapper is a scope: keep the wrapper, keep
        // the loop, and normalize only the loop body (re-dispatching the
        // while would wrap it a second time and turn the scope into an op).
        // Anything else is one indivisible op.
        if (const auto *wl = attr->body.as<WhileNode>()) {
          While loop = GetRef<While>(wl);
          loop.CopyOnWrite()->body = VisitStmt(wl->body);
          return AttrStmt(attr->node, attr->attr_key, attr->value,
                          std::move(loop), attr->span);
        }
        VerifyOpaqueOp(stmt);
        return stmt;
      }
      // Other wrappers (assumptions, kernel metadata) are transparent.
      return StmtMutator::VisitStmt(stmt);
    }

    if (const auto *evaluate = stmt.as<EvaluateNode>()) {
      if (const auto *call = evaluate->value.as<CallNode>()) {
        if (call->annotations.Get(kWSOpIdKey).has_value())
          return stmt;
        auto annotations = call->annotations;
        annotations.Set(kWSOpIdKey, StringImm(FreshId(CallKind(call))));
        return Evaluate(Call(call->dtype, call->op, call->args,
                             std::move(annotations), call->span));
      }
      return WrapWithId(stmt, "op");
    }

    if (const auto *wl = stmt.as<WhileNode>()) {
      While loop = GetRef<While>(wl);
      loop.CopyOnWrite()->body = VisitStmt(wl->body);
      return WrapWithId(loop, "while");
    }

    if (const auto *store = stmt.as<BufferStoreNode>())
      return WrapWithId(stmt, std::string(store->buffer->name));
    if (const auto *bind = stmt.as<BindNode>())
      return WrapWithId(stmt, std::string(bind->var->name_hint));
    if (stmt.as<SBlockNode>()) {
      VerifyOpaqueOp(stmt);
      return WrapWithId(stmt, "block");
    }

    return StmtMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Sequential loops (serial, unrolled) are scopes; parallel and
    // vectorized loops are single opaque ops. Both carry the id in the
    // loop annotations, so recurse only into scope bodies.
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
      VerifyOpaqueOp(GetRef<For>(op));
      if (op->annotations.count(kWSOpIdKey))
        return GetRef<For>(op);
      const char *kind = op->kind == ForKind::kParallel     ? "parallel"
                         : op->kind == ForKind::kVectorized ? "vectorize"
                                                            : "loop";
      For loop = GetRef<For>(op);
      loop.CopyOnWrite()->annotations.Set(kWSOpIdKey, FreshId(kind));
      return loop;
    }
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    if (!loop->annotations.count(kWSOpIdKey))
      loop.CopyOnWrite()->annotations.Set(
          kWSOpIdKey,
          FreshId(op->kind == ForKind::kUnrolled ? "unroll" : "loop"));
    return loop;
  }

  Target target_;
  std::set<ffi::String> used_ids_;
  std::map<std::string, int> next_id_;
};

} // namespace

Stmt PreprocessIR(Stmt body, const Target &target) {
  VerifyCalls(body);
  return TaskNormalizer::Rewrite(std::move(body), target);
}

} // namespace tl
} // namespace tvm
