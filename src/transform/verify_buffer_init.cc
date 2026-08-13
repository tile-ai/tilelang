/*!
 * \file verify_buffer_init.cc
 *
 * Warn when a non-global-scope buffer is read before anything writes it.
 *
 * Reading a shared-memory or register buffer that nothing has written yields
 * whatever those locations last held. That is undefined behaviour: it can
 * silently produce correct results on one architecture and NaN on another
 * (see issue #2936). This pass reports it at compile time.
 *
 * The analysis is deliberately imprecise in one direction: a write anywhere
 * earlier in execution order counts, even under a conditional the read is not
 * guarded by. It detects "nothing writes this buffer at all", and nothing
 * finer. That keeps false positives rare enough for the check to be on by
 * default.
 */

#include "../op/builtin.h"
#include "../op/gemm.h"
#include "../op/gemm_sp.h"
#include "../op/operator.h"
#include "span_utils.h"
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

using namespace tirx;
using namespace ffi;

namespace {

/*!
 * \brief Scopes this check does not report.
 *
 * Global memory is the caller's responsibility. Barrier scopes hold mbarrier
 * state established by intrinsics the pass does not model, and reading them
 * before any visible write is normal (cf. inject_pipeline.cc).
 */
bool IsExemptScope(const String &scope) {
  return scope == "global" || scope == "shared.barrier" ||
         scope == "shared.cluster_barrier";
}

/*!
 * \brief Whether reads of this scope must ignore source order.
 *
 * Shared memory is written by one group of threads and read by another, with
 * correctness established by barriers rather than by program order. A
 * warp-specialized kernel routinely places the producer branch after the
 * consumer branch in the body, so "written earlier in the body" says nothing
 * about what has run. Per-thread storage has no such excuse: source order is
 * that thread's execution order.
 */
bool IsCrossThreadScope(const String &scope) {
  const std::string s = scope;
  return s.rfind("shared", 0) == 0;
}

/*! \brief Parse a call as a tile op, or return null if it cannot be.
 *
 * A tile op call is not always in its final form at this point in the
 * pipeline, and an op builder that indexes an argument it has not been given
 * throws. This pass only warns, so a call it cannot interpret must degrade to
 * the opaque-escape treatment rather than abort the build.
 */
TileOperator TryParseOperator(const Call &call) {
  try {
    return ParseOperator(call);
  } catch (const tvm::ffi::Error &) {
    return TileOperator();
  }
}

/*! \brief Whether this call is a tile op, however well-formed. */
bool IsTileOpCall(const CallNode *op) {
  auto opt_op = op->op.as<Op>();
  if (!opt_op.has_value()) {
    return false;
  }
  const std::string &name = opt_op.value()->name;
  return name.rfind("tl.tileop.", 0) == 0;
}

/*! \brief Whether this call may write through one of its arguments.
 *
 * A pure op only reads what it is given, so its BufferLoad arguments are
 * ordinary reads. Anything at kUpdateState (== kOpaque) or beyond may write,
 * as may an op that never registered an effect kind, or a call to something
 * other than an Op at all.
 */
bool MayWriteThroughArgs(const CallNode *op) {
  static const auto effect_map =
      Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
  auto opt_op = op->op.as<Op>();
  if (!opt_op.has_value()) {
    return true;
  }
  Op call_op = opt_op.value();
  if (!effect_map.count(call_op)) {
    return true;
  }
  return effect_map[call_op]->value >=
         static_cast<int>(CallEffectKind::kUpdateState);
}

/*! \brief Whether a buffer reaching this call may be written by it. */
bool EscapesThroughArgs(const CallNode *op) {
  return op->op.same_as(builtin::address_of()) ||
         op->op.same_as(builtin::tvm_access_ptr()) ||
         op->op.same_as(tl::access_ptr()) ||
         op->op.same_as(builtin::call_extern()) || IsTileOpCall(op) ||
         MayWriteThroughArgs(op);
}

/*!
 * \brief Report every buffer variable an uninterpretable call may write.
 *
 * A buffer whose pointer escapes into an opaque call may be written by it.
 * Assuming it is written trades recall for precision, which is the trade this
 * check is built on. The two walks below both route through here, so their
 * notions of what a call may write cannot drift apart.
 */
template <typename F>
void ForEachOpaqueWrite(const CallNode *op, const F &record) {
  if (op->op.same_as(builtin::address_of())) {
    if (!op->args.empty()) {
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        record(load->buffer->data);
      }
    }
    return;
  }
  if (op->op.same_as(tl::access_ptr())) {
    // access_ptr(base_load, extent, rw_mask)
    if (op->args.size() >= 3 &&
        (GetConservativeAccessMask(op->args[2]) & kAccessWrite)) {
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        record(load->buffer->data);
      }
    }
    return;
  }
  if (op->op.same_as(builtin::tvm_access_ptr())) {
    // args: [type_hint, data_var, offset, extent, rw_mask]
    if (op->args.size() >= 5 &&
        (GetConservativeAccessMask(op->args[4]) & kAccessWrite)) {
      if (const auto *var = op->args[1].as<VarNode>()) {
        record(GetRef<Var>(var));
      }
    }
    return;
  }
  for (const PrimExpr &arg : op->args) {
    if (const auto *var = arg.as<VarNode>()) {
      if (var->dtype.is_handle()) {
        record(GetRef<Var>(var));
      }
    } else if (const auto *load = arg.as<BufferLoadNode>()) {
      record(load->buffer->data);
    } else if (const auto *call = arg.as<CallNode>()) {
      // Nested address_of / tvm_access_ptr inside an extern argument list.
      ForEachOpaqueWrite(call, record);
    }
  }
}

/*! \brief Collects every buffer a function potentially writes.
 *
 * Records which node performed each write so that a read can discount the
 * reading operation's own write: a gemm that accumulates into an otherwise
 * untouched buffer must still be reported.
 */
struct PotentialWriteCollector : public StmtExprVisitor {
  std::unordered_map<const VarNode *, std::vector<const Object *>> writers;

  void Record(const Var &var, const Object *writer) {
    writers[var.get()].push_back(writer);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Record(op->buffer->data, op);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (TileOperator tile_op = TryParseOperator(GetRef<Call>(op));
        tile_op.defined()) {
      for (const BufferRegion &region : tile_op->GetAccessRegions().writes) {
        Record(region->buffer->data, op);
      }
      return;
    }
    if (EscapesThroughArgs(op)) {
      ForEachOpaqueWrite(op, [&](const Var &var) { Record(var, op); });
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

struct BufferInitVerifier : public StmtExprVisitor {
  /*! \brief Which remedy the warning should suggest. */
  enum class Kind { kGeneric, kGemmAccum };

  struct Report {
    Buffer buffer;
    Span span;
    Kind kind;
  };

  std::vector<Report> reports_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> written_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reported_;
  std::unordered_map<const VarNode *, std::vector<const Object *>>
      potential_writers_;
  Span current_span_;

  /*! \brief Treat every parameter buffer as written by the caller. */
  void SeedParams(const PrimFunc &f) {
    for (const auto &kv : f->buffer_map) {
      written_.insert(kv.second->data);
    }
  }

  /*! \brief Record what the whole function potentially writes. */
  void CollectPotentialWrites(const PrimFunc &f) {
    PotentialWriteCollector collector;
    collector(f->body);
    potential_writers_ = std::move(collector.writers);
  }

  /*! \brief Whether some node other than \p reader writes \p var. */
  bool WrittenByAnotherNode(const Var &var, const Object *reader) const {
    auto it = potential_writers_.find(var.get());
    if (it == potential_writers_.end()) {
      return false;
    }
    for (const Object *writer : it->second) {
      if (writer != reader) {
        return true;
      }
    }
    return false;
  }

  /*! \brief Record a read, reporting it if nothing has written the buffer.
   *
   * Each buffer is reported at most once, however many times it is read.
   */
  void CheckRead(const Buffer &buffer, Kind kind = Kind::kGeneric,
                 const Object *reader = nullptr) {
    if (IsExemptScope(buffer.scope())) {
      return;
    }
    const Var &var = buffer->data;
    if (reported_.count(var)) {
      return;
    }
    const bool satisfied = IsCrossThreadScope(buffer.scope())
                               ? WrittenByAnotherNode(var, reader)
                               : written_.count(var) > 0;
    if (satisfied) {
      return;
    }
    reported_.insert(var);
    reports_.push_back({buffer, current_span_, kind});
  }

  void VisitStmt_(const EvaluateNode *op) final {
    Span saved = current_span_;
    if (op->span.defined()) {
      current_span_ = op->span;
    }
    StmtExprVisitor::VisitStmt_(op);
    current_span_ = saved;
  }

  /*! \brief A direct store initializes, but only after it has been evaluated.
   *
   * The destination is recorded last, so `x[i] = x[i] + 1` reports x when
   * nothing wrote it earlier: a store reads its own right-hand side before it
   * establishes anything. Cross-thread scopes are unaffected, since they
   * ignore source order and ask only whether another node writes the buffer.
   */
  void VisitStmt_(const BufferStoreNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    written_.insert(op->buffer->data);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    CheckRead(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (TileOperator tile_op = TryParseOperator(GetRef<Call>(op));
        tile_op.defined()) {
      VisitTileOp(tile_op, op);
      // Do not recurse. A tl.region argument wraps a BufferLoad that marks the
      // region, not a real read; visiting it would double-count and ignore the
      // op's own semantics.
      return;
    }
    if (EscapesThroughArgs(op)) {
      ForEachOpaqueWrite(op, [&](const Var &var) { written_.insert(var); });
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  /*! \brief Check an op's definite reads, then apply its writes.
   *
   * Reads before writes, so a read-modify-write region (a gemm accumulating
   * into C) is reported when nothing wrote it earlier.
   */
  void VisitTileOp(const TileOperator &tile_op, const Object *reader) {
    // The analysis below is op-agnostic; this lookup only selects which remedy
    // the warning suggests.
    Buffer accum;
    if (const auto *gemm = tile_op.as<GemmNode>()) {
      accum = gemm->cRegion_->buffer;
    } else if (const auto *gemm_sp = tile_op.as<GemmSPNode>()) {
      accum = gemm_sp->cRegion_->buffer;
    }
    for (const BufferRegion &region : tile_op->GetReadBeforeWriteRegions()) {
      Kind kind = (accum.defined() && region->buffer.same_as(accum))
                      ? Kind::kGemmAccum
                      : Kind::kGeneric;
      CheckRead(region->buffer, kind, reader);
    }
    for (const BufferRegion &region : tile_op->GetAccessRegions().writes) {
      written_.insert(region->buffer->data);
    }
  }

  /*! \brief Emit all findings as one aggregated warning. */
  void EmitReport() const {
    if (reports_.empty()) {
      return;
    }
    std::ostringstream os;
    os << "Buffer read before initialization: " << reports_.size()
       << " buffer(s) read before anything writes them\n";
    for (size_t k = 0; k < reports_.size(); ++k) {
      const Report &report = reports_[k];
      os << "  [" << k + 1 << "] `" << report.buffer
         << "` is read before anything writes it"
         << SpanHintSuffix({report.span}) << "\n";
      if (report.kind == Kind::kGemmAccum) {
        os << "      T.gemm accumulates into C, so an uninitialized "
              "accumulator adds into stale contents. Add `T.clear(C)` before "
              "the gemm, or pass `clear_accum=True`.\n";
      }
    }
    os << "Initialize the buffer before its first read (for example with "
          "`T.clear`, `T.fill`, or a `T.copy` into it). If you believe this "
          "is a false positive, disable the check by setting "
          "`PassConfigKey.TL_DISABLE_BUFFER_INIT_CHECK` in the pass config.";
    LOG(WARNING) << os.str();
  }
};

using namespace tirx::transform;

tvm::transform::Pass VerifyBufferInit() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    BufferInitVerifier verifier;
    verifier.SeedParams(f);
    verifier.CollectPotentialWrites(f);
    verifier(f->body);
    verifier.EmitReport();
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyBufferInit", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.VerifyBufferInit", VerifyBufferInit);
}

} // namespace

} // namespace tvm::tl
