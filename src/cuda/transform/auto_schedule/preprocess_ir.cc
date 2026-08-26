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
 * Normalization also checks the schedulability contract, declining the
 * kernel (warning + nullopt): no loop_break, no atomics, no hand-written
 * synchronization, and no asynchronous tile op nested inside an opaque
 * op.
 */

#include "./preprocess_ir.h"

#include <tvm/ir/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <map>
#include <set>
#include <string>

#include "cuda/transform/ws_analysis.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/operator.h"
#include "op/utils.h"
#include "transform/common/warp_specialize.h"

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

// Constructs the materializer cannot schedule at all; warns and returns
// false on the first violation.
bool CheckCalls(const Stmt &body) {
  bool ok = true;
  PostOrderVisit(body, [&ok](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr || !ok)
      return;
    if (call->op.same_as(tl::loop_break())) {
      LOG(WARNING) << "AutoSchedule skipped: cannot schedule loop_break";
      ok = false;
      return;
    }
    if (IsBarrierOrTmaControlCall(call)) {
      LOG(WARNING) << "AutoSchedule skipped: cannot schedule hand-written "
                      "synchronization or TMA control ('"
                   << call->op << "'): block-wide syncs cannot be duplicated "
                   << "into roles, and hand-managed barrier protocols are "
                   << "invisible to the schedule";
      ok = false;
      return;
    }
    if (const auto *op = call->op.as<OpNode>()) {
      if (std::string(op->name).find("atomic") != std::string::npos) {
        LOG(WARNING) << "AutoSchedule skipped: cannot schedule atomic op '"
                     << op->name << "'";
        ok = false;
      }
    }
  });
  return ok;
}

// A tile op whose completion the schedule must track (g2s async
// producer, tmem or wg_wait gemm) cannot nest inside an opaque op. TMA
// stores are fine: they complete through the copy lowering's own
// commit-group machinery.
bool CheckHostsNoAsync(const Stmt &stmt, const Target &target) {
  bool ok = true;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr || !ok)
      return;
    TileOperator tile_op = ParseOperator(GetRef<Call>(call));
    if (!tile_op.defined())
      return;
    if (const auto *copy = tile_op.as<CopyNode>()) {
      if (ClassifyCopy(copy, target) == TileStmtKind::kTmaProducer) {
        LOG(WARNING) << "AutoSchedule skipped: an asynchronous "
                        "global->shared copy is nested inside a compound "
                        "statement; write it as its own statement so its "
                        "completion barrier can be wired";
        ok = false;
      }
      return;
    }
    if (auto gemm = GetGemmInfo(tile_op)) {
      if (IsTmemBuffer(gemm->accumulator) || gemm->wg_wait != 0) {
        LOG(WARNING) << "AutoSchedule skipped: an asynchronous gemm is "
                        "nested inside a compound statement; write it as "
                        "its own statement";
        ok = false;
      }
    }
  });
  return ok;
}

class OpIdNormalizer : public StmtMutator {
public:
  static Optional<Stmt> Rewrite(Stmt body, const Target &target) {
    OpIdNormalizer normalizer(target);
    normalizer.CollectIds(body);
    Stmt result = normalizer(std::move(body));
    if (!normalizer.ok_)
      return std::nullopt;
    return result;
  }

private:
  explicit OpIdNormalizer(Target target) : target_(std::move(target)) {}

  // A simple wrapper around one call is a direct op (ClassifyStmt sees
  // through it); only genuinely compound statements are opaque.
  void CheckOpaqueOp(const Stmt &stmt) {
    if (!GetEvaluateCallInSimpleWrapper(stmt).defined())
      ok_ = ok_ && CheckHostsNoAsync(stmt, target_);
  }

  void CollectIds(const Stmt &stmt) {
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (const auto *attr = node.as<AttrStmtNode>()) {
        if (attr->attr_key == kWSOpIdKey)
          used_ids_.insert(ExtractOpId(Any(attr->value)));
      } else if (const auto *loop = node.as<ForNode>()) {
        if (auto value = loop->annotations.Get(kWSOpIdKey))
          used_ids_.insert(ExtractOpId(value.value()));
      } else if (const auto *call = node.as<CallNode>()) {
        if (auto value = call->annotations.Get(kWSOpIdKey))
          used_ids_.insert(ExtractOpId(Any(value.value())));
      }
    });
  }

  String FreshId(const std::string &kind) {
    while (true) {
      String id(kind + "_" + std::to_string(next_id_[kind]++));
      if (used_ids_.insert(id).second)
        return id;
    }
  }

  Stmt WrapWithId(const Stmt &stmt, const std::string &kind) {
    return AttrStmt(Integer(0), kWSOpIdKey, StringImm(FreshId(kind)), stmt,
                    stmt->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    // Other wrappers (assumptions, kernel metadata,
    // T.annotate_ws_pipeline_depth) are transparent.
    if (op->attr_key != kWSOpIdKey)
      return StmtMutator::VisitStmt_(op);
    // A while loop under an existing wrapper is a scope: keep the wrapper,
    // keep the loop, and normalize only the loop body (re-dispatching the
    // while would wrap it a second time and turn the scope into an op).
    // Anything else is one indivisible op.
    if (const auto *wl = op->body.as<WhileNode>()) {
      While loop = GetRef<While>(wl);
      loop.CopyOnWrite()->body = VisitStmt(wl->body);
      return AttrStmt(op->node, op->attr_key, op->value, std::move(loop),
                      op->span);
    }
    Stmt stmt = GetRef<Stmt>(op);
    CheckOpaqueOp(stmt);
    return stmt;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr)
      return WrapWithId(GetRef<Stmt>(op), "op");
    if (call->annotations.count(kWSOpIdKey))
      return GetRef<Stmt>(op);
    auto annotations = call->annotations;
    annotations.Set(kWSOpIdKey, StringImm(FreshId(CallKind(call))));
    return Evaluate(Call(call->dtype, call->op, call->args,
                         std::move(annotations), call->span));
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    While loop = GetRef<While>(op);
    loop.CopyOnWrite()->body = VisitStmt(op->body);
    return WrapWithId(loop, "while");
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    return WrapWithId(GetRef<Stmt>(op), std::string(op->buffer->name));
  }

  Stmt VisitStmt_(const BindNode *op) final {
    return WrapWithId(GetRef<Stmt>(op), std::string(op->var->name_hint));
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    Stmt stmt = GetRef<Stmt>(op);
    CheckOpaqueOp(stmt);
    return WrapWithId(stmt, "block");
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Sequential loops (serial, unrolled) are scopes; parallel and
    // vectorized loops are single opaque ops. Both carry the id in the
    // loop annotations, so recurse only into scope bodies.
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
      CheckOpaqueOp(GetRef<For>(op));
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
  bool ok_ = true;
  std::set<String> used_ids_;
  std::map<std::string, int> next_id_;
};

} // namespace

ffi::Optional<Stmt> PreprocessIR(Stmt body, const Target &target) {
  if (!CheckCalls(body))
    return std::nullopt;
  return OpIdNormalizer::Rewrite(std::move(body), target);
}

} // namespace tl
} // namespace tvm
