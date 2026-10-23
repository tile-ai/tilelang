/*!
 * \file ascend/costmodel/simtvf/analysis/ownership_effect.cc
 * \brief Adapt locally lowered ownership candidates to the SIMT UB model.
 *
 * Extract lane addresses and estimated machine instructions, fold proven
 * repetitions, and compose independent UB batches for comparison. This leaves
 * candidate legality and IR selection to the pass; it neither simulates the
 * backend scheduler nor prices missed downstream optimizations. See
 * ownership_effect.h for the model's scope and composition contract.
 */
#include "ownership_effect.h"
#include "op/copy.h"
#include "transform/loop_partition.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl::ascend::costmodel::simtvf {
using namespace tirx;
using namespace ffi;
namespace {
constexpr auto &geometry = kDav3510UbGeometry;
// Access extraction is separate from the shared latency model.
struct SimtUbAccess {
  bool store;
  std::vector<SimtUbInstructionAccess> instructions;
};

// Serial invocation count: enclosing loops may carry values or contain DMA,
// so repeat whole operator costs rather than overlap their memory requests.
// Factor invariant inner domains; enumerate only genuinely varying bounds
// (for example a triangular nest). Access-map invariance is checked by the
// caller. Unknown domains or overflowing counts cannot justify a selection.
std::optional<int64_t> CountInvocations(const std::vector<For> &loops) {
  arith::Analyzer analyzer;
  Map<Var, PrimExpr> values;
  std::function<std::optional<int64_t>(size_t)> count =
      [&](size_t depth) -> std::optional<int64_t> {
    if (depth == loops.size())
      return 1;
    const auto &loop = loops[depth];
    auto min = analyzer.Simplify(Substitute(loop->min, values));
    auto extent = analyzer.Simplify(Substitute(loop->extent, values));
    const auto *start = min.as<IntImmNode>();
    const auto *size = extent.as<IntImmNode>();
    if (!start || !size || size->value < 0)
      return std::nullopt;
    if (size->value == 0)
      return 0;
    if (start->value > std::numeric_limits<int64_t>::max() - (size->value - 1))
      return std::nullopt;
    Map<Var, PrimExpr> shift{{loop->loop_var, loop->loop_var + 1}};
    bool uniform = true;
    for (size_t i = depth + 1; i < loops.size(); ++i)
      for (const auto &bound : {loops[i]->min, loops[i]->extent})
        uniform &= analyzer.CanProveEqual(Substitute(bound, shift), bound);
    int64_t total = 0, repetitions = uniform ? size->value : 1;
    for (int64_t i = 0; i < (uniform ? 1 : size->value); ++i) {
      values.Set(loop->loop_var, Integer(start->value + i));
      auto inner = count(depth + 1);
      if (!inner ||
          *inner > (std::numeric_limits<int64_t>::max() - total) / repetitions)
        return std::nullopt;
      total += *inner * repetitions;
    }
    values.erase(loop->loop_var);
    return total;
  };
  return count(0);
}

class SimtUbAccessCollector : public StmtExprVisitor {
public:
  // Candidate lowering: turn the candidate loop layout into thread/local
  // coordinates using the same partitioning and vectorization as LowerTileOp.
  // LowerParallelLoop supplies the index expressions and tail guards; we do
  // not reconstruct the ownership mapping independently or commit this IR.
  std::optional<std::vector<SimtUbAccess>> Collect(const ParallelOpNode *op,
                                                   const Range &threads,
                                                   const LayoutMap &layouts) {
    Var tx("ub_access_thread");
    thread_ = tx;
    const auto *extent = threads->extent.as<IntImmNode>();
    const auto *minimum = threads->min.as<IntImmNode>();
    if (!extent || !minimum)
      return std::nullopt;
    thread_min_ = minimum->value;
    thread_count_ = extent->value;
    analyzer_.Bind(tx, threads);
    Stmt lowered = LowerParallelLoop(
        op->GetRoot(), op->GetLoopLayout(), tx, &analyzer_, layouts,
        op->GetPredicate(tx), true, op->LoopLayoutRequiresPaddingGuard());
    // Warp index is an ordinary serial coordinate for period compression.
    Var warp("ub_access_warp");
    warp_ = warp;
    loops_.push_back(For(
        warp, 0,
        Integer((thread_count_ + geometry.warp_size - 1) / geometry.warp_size),
        ForKind::kSerial, Evaluate(0)));
    VisitStmt(lowered);
    return valid_
               ? std::optional<std::vector<SimtUbAccess>>(std::move(accesses_))
               : std::nullopt;
  }

private:
  arith::Analyzer analyzer_;
  Var thread_, warp_;
  int64_t thread_min_{0}, thread_count_{0};
  std::vector<For> loops_;
  Optional<For> vector_loop_;
  PrimExpr predicate_{Bool(true)};
  bool valid_{true};
  std::vector<SimtUbAccess> accesses_;

  // Execution context: retain serial coordinates, one vector coordinate and
  // the enclosing branch mask for every UB load/store. These describe which
  // lane elements execute, not their eventual machine scheduling order.
  void VisitStmt_(const ForNode *op) final {
    if (!valid_)
      return;
    auto loop = GetRef<For>(op);
    if (op->kind == ForKind::kVectorized) {
      if (vector_loop_.defined()) {
        valid_ = false;
        return;
      }
      vector_loop_ = loop;
      VisitStmt(op->body);
      vector_loop_ = Optional<For>();
    } else {
      loops_.push_back(loop);
      VisitStmt(op->body);
      loops_.pop_back();
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    // The condition is evaluated under the enclosing mask, before either arm.
    VisitExpr(op->condition);
    PrimExpr saved = predicate_;
    predicate_ = saved && op->condition;
    VisitStmt(op->then_case);
    if (op->else_case.defined()) {
      predicate_ = saved && !op->condition;
      VisitStmt(op->else_case.value());
    }
    predicate_ = saved;
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Access(op->buffer, op->indices, false);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    // Value loads must precede the store that consumes them. Direction changes
    // delimit dependent batches; independent parallel elements within a batch
    // may overlap under the ideal downstream grouping assumption.
    StmtExprVisitor::VisitStmt_(op);
    Access(op->buffer, op->indices, true);
  }

  std::optional<int64_t> EvaluateInt(const PrimExpr &expr,
                                     const Map<Var, PrimExpr> &values) {
    PrimExpr value = analyzer_.Simplify(Substitute(expr, values));
    if (const auto *imm = value.as<IntImmNode>())
      return imm->value;
    return std::nullopt;
  }

  // Repetition proof: translating ALL lanes/vector elements by a common
  // multiple of 8B preserves packing, full-line equality and 4B coverage.
  // Masks and inner serial domains must also repeat. This permits evaluating
  // one representative and multiplying its service work, not merging warp
  // requests. Parallel elements may overlap; enclosing serial invocations
  // are counted separately. Lane/inner-coordinate shifts fail.
  bool Repeats(const PrimExpr &address, size_t depth, int64_t period) {
    const Var &var = loops_[depth]->loop_var;
    Map<Var, PrimExpr> shift{{var, var + Integer(period)}};
    // Equal address phases do not imply equal work in a triangular nest.
    // Only fold outer iterations when all inner loop domains also repeat.
    for (size_t i = depth + 1; i < loops_.size(); ++i) {
      const auto &loop = loops_[i];
      if (!analyzer_.CanProveEqual(Substitute(loop->min, shift), loop->min) ||
          !analyzer_.CanProveEqual(Substitute(loop->extent, shift),
                                   loop->extent))
        return false;
    }
    PrimExpr delta = analyzer_.Simplify(Substitute(address, shift) - address);
    Map<Var, PrimExpr> inner{{thread_, 0}};
    for (size_t i = depth + 1; i < loops_.size(); ++i)
      inner.Set(loops_[i]->loop_var, 0);
    if (vector_loop_.defined())
      inner.Set(vector_loop_.value()->loop_var, 0);
    return analyzer_.CanProveEqual(delta, Substitute(delta, inner)) &&
           analyzer_.CanProveEqual(floormod(delta, geometry.line_bytes), 0) &&
           analyzer_.CanProveEqual(Substitute(predicate_, shift), predicate_);
  }

  // Address formation: flatten the lowered UB indices using either the dense
  // shape or explicit strides, add elem_offset, then convert elements to
  // bytes. Only the common aligned buffer base is omitted. Non-UB accesses
  // are outside this score; the caller has already checked candidate scope.
  void Access(const Buffer &buffer, const Array<PrimExpr> &indices,
              bool store) {
    if (!valid_ ||
        (buffer.scope() != "shared" && buffer.scope() != "shared.dyn"))
      return;
    if (buffer->dtype.lanes() != 1 || buffer->dtype.bits() < 8) {
      valid_ = false;
      return;
    }
    // Unknown base low bits can change request packing. Require declared
    // 8B alignment; otherwise this access cannot be analyzed.
    if (buffer->data_alignment <= 0 ||
        buffer->data_alignment % geometry.line_bytes != 0) {
      valid_ = false;
      return;
    }
    PrimExpr address = 0;
    for (size_t i = 0; i < indices.size(); ++i)
      address = buffer->strides.empty()
                    ? address * buffer->shape[i] + indices[i]
                    : address + buffer->strides[i] * indices[i];
    int bytes = buffer->dtype.bytes();
    address = analyzer_.Simplify((address + buffer->elem_offset) * bytes);
    // A common 8B-aligned base translation does not change packing. Known
    // elem_offset/stride low bits remain part of the relative byte address.
    SimtUbAccess access{store, {}};
    Map<Var, PrimExpr> values;
    // Serial expansion: enumerate warp/local iterations, compressing only
    // periods established by Repeats. Each residue carries its execution
    // count in repeat; without a proof, visit every iteration. Unknown loop
    // domains invalidate the estimate rather than supplying a guessed count.
    std::function<void(size_t, int64_t)> walk = [&](size_t depth,
                                                    int64_t repeat) {
      if (!valid_)
        return;
      if (depth != loops_.size()) {
        const auto &loop = loops_[depth];
        auto min = EvaluateInt(loop->min, values);
        auto extent = EvaluateInt(loop->extent, values);
        if (!min || !extent || *extent < 0) {
          valid_ = false;
          return;
        }
        int64_t period = 1;
        while (period < *extent && !Repeats(address, depth, period))
          period = period > *extent / 2 ? *extent : period * 2;
        for (int64_t i = 0; i < std::min(period, *extent); ++i) {
          values.Set(loop->loop_var, Integer(*min + i));
          int64_t count = (*extent - 1 - i) / period + 1;
          if (repeat > std::numeric_limits<int64_t>::max() / count) {
            valid_ = false;
            return;
          }
          walk(depth + 1, repeat * count);
        }
        values.erase(loop->loop_var);
        return;
      }
      auto warp = EvaluateInt(warp_, values);
      int64_t vmin = 0, width = 1;
      if (vector_loop_.defined()) {
        auto start = EvaluateInt(vector_loop_.value()->min, values);
        auto count = EvaluateInt(vector_loop_.value()->extent, values);
        if (!start || !count || *count < 1) {
          valid_ = false;
          return;
        }
        vmin = *start;
        width = *count;
      }
      // Lane materialization: after fixing serial coordinates, evaluate the
      // mask and byte address for each vector element and each of 32 lanes.
      // addresses[v][lane] is absent for an inactive element, distinct from
      // address zero. Any unresolved active address or mask makes cost unknown.
      std::vector<SimtUbLaneAddresses> addresses(width);
      for (int lane = 0; lane < geometry.warp_size; ++lane) {
        int64_t tid = *warp * geometry.warp_size + lane;
        if (tid >= thread_count_)
          continue;
        values.Set(thread_, Integer(lane));
        for (int64_t v = 0; v < width; ++v) {
          if (vector_loop_.defined())
            values.Set(vector_loop_.value()->loop_var, Integer(vmin + v));
          auto active = EvaluateInt(predicate_, values);
          if (!active) {
            valid_ = false;
            return;
          }
          if (*active) {
            auto addr = EvaluateInt(address, values);
            if (!addr || *addr < 0) {
              valid_ = false;
              return;
            }
            addresses[v][lane] = *addr;
          }
        }
      }
      // Instruction organization: a repeated-address vector load collapses
      // to a scalar load; otherwise vector elements must be contiguous within
      // each active lane. Each lane's activity must agree across all vector
      // elements. Unsupported patterns remain unknown instead of guessing
      // scalarization or gather instructions.
      bool broadcast = !store, contiguous = true;
      for (int64_t v = 1; v < width; ++v)
        for (int lane = 0; lane < geometry.warp_size; ++lane) {
          auto first = addresses[0][lane], next = addresses[v][lane];
          if (first.has_value() != next.has_value()) {
            valid_ = false;
            return;
          }
          if (first) {
            broadcast &= *next == *first;
            contiguous &= *next == *first + v * bytes;
          }
        }
      if (!broadcast && !contiguous) {
        valid_ = false;
        return;
      }
      int64_t vector_bytes = bytes * (broadcast ? 1 : width);
      // Estimate backend memory widths: packed subword vectors use b32;
      // native >=32-bit vectors can use b64. This is a lowering approximation,
      // not an assertion that arbitrary TIR vectors become one ISA operation.
      // For example, f16x4 becomes two b32 instructions, while f32x2 becomes
      // one b64 instruction. Each emitted slice retains its own lane starts
      // and repetition count for independent pricing by the UB model.
      int instruction_bytes = bytes < 4 ? 4 : 8;
      for (int64_t slice = 0; slice < vector_bytes;
           slice += instruction_bytes) {
        auto request = addresses[0];
        for (auto &addr : request)
          if (addr)
            *addr += slice;
        access.instructions.push_back(
            {std::move(request),
             static_cast<int>(
                 std::min<int64_t>(instruction_bytes, vector_bytes - slice)),
             repeat});
      }
      values.erase(thread_);
      if (vector_loop_.defined())
        values.erase(vector_loop_.value()->loop_var);
    };
    // Bind thread coordinates before walking: tx = thread_min + warp*32 +
    // lane. Include the final partial warp's mask in the expression analyzed
    // by Repeats, so cost-equivalent warps can be counted together without
    // treating inactive lanes as accesses or combining separate requests.
    Var lane("ub_access_lane");
    PrimExpr saved_predicate = predicate_;
    PrimExpr thread = Integer(thread_min_) + warp_ * geometry.warp_size + lane;
    address = Substitute(address, {{thread_, thread}});
    predicate_ = Substitute(predicate_, {{thread_, thread}});
    if (thread_count_ % geometry.warp_size)
      predicate_ = predicate_ &&
                   (warp_ * geometry.warp_size + lane < Integer(thread_count_));
    Var saved_thread = thread_;
    thread_ = lane;
    walk(0, 1);
    thread_ = saved_thread;
    predicate_ = saved_predicate;
    if (valid_) {
      // Consecutive loads (or stores) feed one pipeline. A direction change
      // drains the preceding batch, conservatively preserving read/write
      // ordering even when different buffers could overlap in machine code.
      if (!accesses_.empty() && accesses_.back().store == store) {
        auto &batch = accesses_.back().instructions;
        batch.insert(batch.end(), access.instructions.begin(),
                     access.instructions.end());
      } else {
        accesses_.push_back(std::move(access));
      }
    }
  }
};

} // namespace

// Candidate aggregation: the UB model overlaps independent instructions in a
// batch. Direction changes and TileOp boundaries drain the pipeline; enclosing
// loops repeat those whole stages. This preserves serial dependencies without
// making every warp wait for the previous warp's completion. Arithmetic,
// instruction supply, register capacity and overlap across TileOps are outside
// this ideal UB component; it is not elapsed VF time.
std::optional<OwnershipEffect> AnalyzeOwnershipEffect(
    const std::vector<TileOperator> &operators, const std::vector<int> &members,
    const std::vector<Range> &thread_bounds, const LayoutMap &layouts,
    const std::vector<std::vector<For>> &outer_loops) {
  OwnershipEffect total;
  total.ub_access.estimated_latency = 0;
  total.ub_access.service_cycles = 0;
  for (int member : members) {
    auto invocations = CountInvocations(outer_loops[member]);
    if (!invocations)
      return std::nullopt;
    if (*invocations == 0)
      continue;
    const auto *loop = operators[member].as<ParallelOpNode>();
    // SIMT Copy inference already constructs and solves a ParallelOp. Reuse
    // its candidate layout so UB loads/stores at fragment boundaries are
    // priced together with computation, without a separate copy cost rule.
    if (const auto *copy = operators[member].as<CopyNode>();
        copy && copy->par_op_.defined())
      loop = copy->par_op_.get();
    if (!loop)
      return std::nullopt;
    const auto *extent = thread_bounds[member]->extent.as<IntImmNode>();
    if (!extent || extent->value <= 0 ||
        extent->value > std::numeric_limits<int>::max())
      return std::nullopt;
    auto accesses =
        SimtUbAccessCollector().Collect(loop, thread_bounds[member], layouts);
    if (!accesses)
      return std::nullopt;
    for (const auto &access : *accesses) {
      auto cost =
          EstimateSimtUbAccessLatency(access.instructions, access.store);
      if (!cost || !cost->estimated_latency || !cost->service_cycles)
        return std::nullopt;
      total.ub_access.estimated_latency = AddCycles(
          total.ub_access.estimated_latency,
          EstimateInstructionCycles({cost->estimated_latency}, *invocations));
      total.ub_access.service_cycles = AddCycles(
          total.ub_access.service_cycles,
          EstimateInstructionCycles({cost->service_cycles}, *invocations));
      if (!total.ub_access.estimated_latency || !total.ub_access.service_cycles)
        return std::nullopt;
      auto accumulate = [&](int64_t &sum, int64_t count) {
        if (count > (std::numeric_limits<int64_t>::max() - sum) / *invocations)
          return false;
        sum += count * *invocations;
        return true;
      };
      if (!accumulate(total.ub_access.instructions, cost->instructions) ||
          !accumulate(total.ub_access.requests, cost->requests) ||
          !accumulate(total.ub_access.entries, cost->entries) ||
          !accumulate(total.ub_access.partial, cost->partial))
        return std::nullopt;
    }
  }
  return total;
}

bool IsBetterOwnership(const OwnershipEffect &candidate,
                       const OwnershipEffect &current) {
  if (!candidate.ub_access.estimated_latency ||
      !current.ub_access.estimated_latency)
    return false;
  double before = *current.ub_access.estimated_latency;
  double after = *candidate.ub_access.estimated_latency;
  // Equivalent period decompositions can round differently. Numerical ties
  // retain the current candidate rather than changing ownership arbitrarily.
  return after < before - 1e-9 * std::max(1.0, before);
}

} // namespace tvm::tl::ascend::costmodel::simtvf
