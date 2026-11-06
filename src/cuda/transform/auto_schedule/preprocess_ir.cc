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
#include "ir/annotations.h"
#include <tvm/ffi/extra/structural_visit.h>

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
namespace cuda {

using namespace tirx;
using namespace prim;
using namespace ffi;

namespace {

// The callee's op name with "." replaced by "_".
std::string CallKind(const CallObj *call) {
  if (const auto *op = call->op.as<OpObj>()) {
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
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      body, [&ok](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
        const auto *call = node.as<CallObj>();
        if (call == nullptr || !ok)
          return ffi::WalkResult::Advance();
        if (call->op.same_as(tl::loop_break())) {
          LOG(WARNING)
              << "AutoWarpSpecialization skipped: cannot schedule loop_break";
          ok = false;
          return ffi::WalkResult::Advance();
        }
        if (IsBarrierOrTmaControlCall(call)) {
          LOG(WARNING)
              << "AutoWarpSpecialization skipped: cannot schedule hand-written "
                 "synchronization or TMA control ('"
              << call->op << "'): block-wide syncs cannot be duplicated "
              << "into roles, and hand-managed barrier protocols are "
              << "invisible to the schedule";
          ok = false;
          return ffi::WalkResult::Advance();
        }
        if (const auto *op = call->op.as<OpObj>()) {
          if (std::string(op->name).find("atomic") != std::string::npos) {
            LOG(WARNING)
                << "AutoWarpSpecialization skipped: cannot schedule atomic op '"
                << op->name << "'";
            ok = false;
          }
        }
        return ffi::WalkResult::Advance();
      });
  return ok;
}

// A tile op whose completion the schedule must track (g2s async
// producer, tmem or wg_wait gemm) cannot nest inside an opaque op. TMA
// stores are fine: they complete through the copy lowering's own
// commit-group machinery.
bool CheckHostsNoAsync(const Stmt &stmt, const Target &target) {
  bool ok = true;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      stmt, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
        const auto *call = node.as<CallObj>();
        if (call == nullptr || !ok)
          return ffi::WalkResult::Advance();
        TileOperator tile_op = ParseOperator(GetRef<Call>(call));
        if (!tile_op.defined())
          return ffi::WalkResult::Advance();
        if (const auto *copy = tile_op.as<CopyNode>()) {
          if (ClassifyCopy(copy, target) == TileStmtKind::kTmaProducer) {
            LOG(WARNING) << "AutoWarpSpecialization skipped: an asynchronous "
                            "global->shared copy is nested inside a compound "
                            "statement; write it as its own statement so its "
                            "completion barrier can be wired";
            ok = false;
          }
          return ffi::WalkResult::Advance();
        }
        if (auto gemm = GetGemmInfo(tile_op)) {
          if (IsTmemBuffer(gemm->accumulator) || gemm->wg_wait != 0) {
            LOG(WARNING)
                << "AutoWarpSpecialization skipped: an asynchronous gemm is "
                   "nested inside a compound statement; write it as "
                   "its own statement";
            ok = false;
          }
        }
        return ffi::WalkResult::Advance();
      });
  return ok;
}

class OpIdNormalizer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  static Optional<Stmt> Rewrite(Stmt body, const Target &target) {
    auto normalizer = make_object<OpIdNormalizer>(target);
    normalizer->CollectIds(body);
    Stmt result = normalizer->Mutate(body).ValueOrUnchanged(body);
    if (!normalizer->ok_)
      return std::nullopt;
    return result;
  }
  explicit OpIdNormalizer(Target target) : target_(std::move(target)) {}

private:
  void CheckOpaqueOp(const Stmt &stmt) {
    ok_ = ok_ && CheckHostsNoAsync(stmt, target_);
  }

  void CollectIds(const Stmt &stmt) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        stmt, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *region = node.as<RegionStmtObj>()) {
            if (auto attr = WSAttribute(region))
              if (attr.value().at("key").cast<String>() == kWSOpIdKey)
                used_ids_.insert(ExtractOpId(attr.value().at("value")));
          } else if (const auto *loop = node.as<ForObj>()) {
            if (auto value = loop->annotations.Get(kWSOpIdKey))
              used_ids_.insert(ExtractOpId(*value));
          } else if (const auto *evaluate = node.as<EvaluateObj>()) {
            if (auto id = WSNextOpId(GetRef<Stmt>(evaluate)))
              used_ids_.insert(*id);
          } else if (const auto *call = node.as<CallObj>()) {
            if (IsAnnotationCall(call))
              return ffi::WalkResult::Advance();
            if (auto value = ReadWSCall(call).annotations.Get(kWSOpIdKey))
              used_ids_.insert(ExtractOpId(*value));
          }
          return ffi::WalkResult::Advance();
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
    return MakeAttribute(nullptr, kWSOpIdKey, FreshId(kind), SeqStmt(stmt),
                         stmt->loc);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    Array<Stmt> statements;
    bool pending = false;
    for (const Stmt &statement : op->seq) {
      if (WSNextOpId(statement)) {
        TVM_FFI_CHECK(!pending, ValueError) << "Consecutive WS Bind markers";
        statements.push_back(statement);
        pending = true;
      } else if (pending) {
        TVM_FFI_CHECK(statement.as<BindObj>(), ValueError)
            << "A WS Bind marker must immediately precede a Bind";
        statements.push_back(statement);
        pending = false;
      } else if (const auto *bind = statement.as<BindObj>()) {
        statements.push_back(MakeAnnotation(
            {{kWSNextOpIdKey, FreshId(std::string(bind->var->name))}},
            bind->loc));
        statements.push_back(statement);
      } else {
        statements.push_back(
            Mutate(statement, mode).ValueOrUnchanged(statement));
      }
    }
    TVM_FFI_CHECK(!pending, ValueError) << "Dangling WS Bind marker";
    return SeqStmt(statements, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    auto attr = WSAttribute(op);
    if (!attr) {
      Stmt stmt = GetRef<Stmt>(op);
      CheckOpaqueOp(stmt);
      return WrapWithId(stmt, "block");
    }
    if (attr.value().at("key").cast<String>() != kWSOpIdKey)
      return StmtExprMutator::Mutate_(op, mode);
    // A single while is a scope; an explicit group remains one opaque op.
    if (op->body.size() == 1) {
      if (const auto *wl = op->body[0].as<WhileObj>()) {
        auto body = Mutate(wl->body, mode).ValueOrUnchanged(wl->body);
        auto loop = While(wl->condition, body, wl->loc);
        return RegionStmt(op->op, op->args, op->body_params, op->attrs,
                          SeqStmt(loop), op->result_vars, op->loc);
      }
    }
    CheckOpaqueOp(op->body);
    return Unchanged();
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode) final {
    const auto *call = op->value.as<CallObj>();
    if (!call)
      return WrapWithId(GetRef<Stmt>(op), "op");
    // Layout/schedule metadata is not a schedulable runtime operation.
    if (IsAnnotationCall(call))
      return Unchanged();
    auto annotations = ReadWSCall(call).annotations;
    if (annotations.count(kWSOpIdKey))
      return Unchanged();
    annotations.Set(kWSOpIdKey, FreshId(CallKind(call)));
    return Evaluate(RewriteWSCall(call, annotations), op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const WhileObj *op, InplaceMode mode) final {
    auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    return WrapWithId(While(op->condition, body, op->loc), "while");
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op, InplaceMode) final {
    auto tensor = op->dest.as<TensorVar>();
    return WrapWithId(GetRef<Stmt>(op),
                      tensor ? std::string(tensor->get()->name) : "store");
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode) final {
    // Normally handled by SeqStmt so the definition remains in its scope.
    return SeqStmt(
        {MakeAnnotation({{kWSNextOpIdKey, FreshId(std::string(op->var->name))}},
                        op->loc),
         GetRef<Stmt>(op)},
        op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const AssertStmtObj *op, InplaceMode) final {
    return WrapWithId(GetRef<Stmt>(op), "assert");
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    For loop = GetRef<For>(op);
    if (op->kind != ForKind::kDefault && op->kind != ForKind::kUnrolled) {
      CheckOpaqueOp(loop);
      if (op->annotations.count(kWSOpIdKey))
        return Unchanged();
      const char *kind = op->kind == ForKind::kParallel     ? "parallel"
                         : op->kind == ForKind::kVectorized ? "vectorize"
                                                            : "loop";
      loop.CopyOnWrite()->annotations.Set(kWSOpIdKey, FreshId(kind));
      return loop;
    }
    auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    loop.CopyOnWrite()->body = body;
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

} // namespace cuda
} // namespace tl
} // namespace tvm
