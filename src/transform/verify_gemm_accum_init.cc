/*!
 * \file verify_gemm_accum_init.cc
 *
 * Warn when a T.gemm accumulator is not provably initialized before use.
 *
 * T.gemm accumulates into its C operand, so a fragment that is never written
 * before the first T.gemm is read while uninitialized. That is undefined
 * behaviour: it silently yields correct results on some architectures and NaN
 * on others (see issue #2936). This pass reports it at compile time.
 */

#include "../op/gemm.h"
#include "../op/region.h"
#include "span_utils.h"
#include <optional>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

using namespace tirx;
using namespace ffi;

namespace {

/*! \brief Access-mask bit meaning "this region is written". */
constexpr int kRegionWriteBit = 2;

/*! \brief The buffer var a tl.region refers to, if `arg` is a region call. */
std::optional<Var> RegionBufferVar(const PrimExpr &arg) {
  const auto *call = arg.as<CallNode>();
  if (call == nullptr || !call->op.same_as(RegionOp::Get()) ||
      call->args.empty()) {
    return std::nullopt;
  }
  const auto *load = call->args[0].as<BufferLoadNode>();
  if (load == nullptr) {
    return std::nullopt;
  }
  return load->buffer->data;
}

/*! \brief Whether a tl.region call carries the write bit in its access mask. */
bool RegionIsWritten(const PrimExpr &arg) {
  const auto *call = arg.as<CallNode>();
  if (call == nullptr || !call->op.same_as(RegionOp::Get()) ||
      call->args.size() < 2) {
    return false;
  }
  const auto *mask = call->args[1].as<IntImmNode>();
  return mask != nullptr && (mask->value & kRegionWriteBit) != 0;
}

/*!
 * \brief Collect T.gemm calls whose accumulator has no preceding write.
 *
 * The body is visited in execution order, so any write recorded before a gemm
 * is reached genuinely precedes it. The check is deliberately permissive: a
 * write anywhere earlier counts, including inside a loop or conditional that
 * the gemm is not part of. Precision matters more than recall here, because a
 * false positive on correct code is worse than missing an exotic case.
 */
struct GemmAccumInitVerifier : public StmtExprVisitor {
  struct Report {
    Buffer accum;
    Span span;
  };

  std::vector<Report> reports_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> written_;
  Span current_span_;

  void VisitStmt_(const EvaluateNode *op) final {
    Span saved = current_span_;
    if (op->span.defined()) {
      current_span_ = op->span;
    }
    StmtExprVisitor::VisitStmt_(op);
    current_span_ = saved;
  }

  /*! \brief A direct store (e.g. a T.Parallel loop body) initializes. */
  void VisitStmt_(const BufferStoreNode *op) final {
    written_.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(Gemm::Get())) {
      CheckGemm(op);
      return;
    }
    // Every other tile op (T.fill, T.copy, ...) advertises what it writes
    // through the access mask on its region arguments.
    for (const PrimExpr &arg : op->args) {
      if (RegionIsWritten(arg)) {
        if (auto var = RegionBufferVar(arg)) {
          written_.insert(var.value());
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  /*!
   * \brief Report this gemm if its accumulator is unwritten, then mark the
   *        accumulator written for any later gemm on the same buffer.
   *
   * Argument layout is fixed by Gemm::Gemm: args[2] is the C region and
   * args[9] is clear_accum.
   */
  void CheckGemm(const CallNode *op) {
    if (op->args.size() <= 9) {
      return;
    }
    const PrimExpr &c_region = op->args[2];
    auto accum_var = RegionBufferVar(c_region);
    if (!accum_var) {
      return;
    }

    // Only a literal `false` is a missing initialization. The pipelined idiom
    // `clear_accum=(k == 0)` is a non-literal expression and is left alone.
    bool definitely_not_cleared = is_zero(op->args[9]);
    if (definitely_not_cleared && !written_.count(accum_var.value())) {
      const auto *load = c_region.as<CallNode>()->args[0].as<BufferLoadNode>();
      reports_.push_back({load->buffer, current_span_});
    }

    written_.insert(accum_var.value());
    // Still visit A and B so nested regions are accounted for, but do not let
    // this gemm's own read-write C region mark the accumulator retroactively.
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i != 2) {
        StmtExprVisitor::VisitExpr(op->args[i]);
      }
    }
  }

  /*! \brief Emit all findings as one aggregated warning. */
  void EmitReport() const {
    if (reports_.empty()) {
      return;
    }
    std::ostringstream os;
    os << "Uninitialized T.gemm accumulator: " << reports_.size()
       << " accumulator(s) read before being written\n";
    for (size_t k = 0; k < reports_.size(); ++k) {
      const Report &report = reports_[k];
      os << "  [" << k + 1 << "] `" << report.accum
         << "` is passed to T.gemm without being initialized"
         << SpanHintSuffix({report.span}) << "\n";
    }
    os << "T.gemm accumulates into C, so an uninitialized accumulator adds "
          "into stale register contents. Add `T.clear(C)` before the gemm, or "
          "pass `clear_accum=True`. If you believe this is a false positive, "
          "disable the check by setting "
          "`PassConfigKey.TL_DISABLE_GEMM_ACCUM_INIT_CHECK` in the pass "
          "config.";
    LOG(WARNING) << os.str();
  }
};

using namespace tirx::transform;

tvm::transform::Pass VerifyGemmAccumInit() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    GemmAccumInitVerifier verifier;
    verifier(f->body);
    verifier.EmitReport();
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyGemmAccumInit", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.VerifyGemmAccumInit",
                        VerifyGemmAccumInit);
}

} // namespace

} // namespace tvm::tl
