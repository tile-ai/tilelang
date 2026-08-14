/*!
 * \file layout_cost_model.cc
 * \brief Statement-level bottleneck traffic model for free-mode layout
 *        attempts (layout RFC, design B2). See layout_cost_model.h.
 */

#include "layout_cost_model.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../layout/layout.h"
#include "../../layout/utils.h"
#include "../../op/copy.h"
#include "../../op/parallel.h"
#include "../../op/utils.h"
#include "../../span_utils.h"
#include "../loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

// ---------------------------------------------------------------------------
// Statement-level bottleneck traffic model (layout RFC, design B2)
//
// The unit of account is one STATEMENT that touches global memory — a
// fragment<->global tl.copy or a parallel loop with direct global accesses.
// The statement is measured in the FORWARD direction: enumerate every
// (logical point, replica) pair, evaluate the layout's own forward maps
//     thread = forward_thread(x, r)        slot = forward_index(x)
// plus the trivially affine global address addr(x), and materialize the
// exact (thread, slot) -> address table the lowered code will execute.
// No layout inversion (the forward maps are total and already built) and
// no witness sampling: the full walk is exact, with each point folded to
// a constant by the Analyzer.
//
// Execution time is bounded by two resources, both measured in BYTES so a
// fully busy block streaming maximal-width lanes scores exactly its
// logical byte count on either. The hardware geometry is parameterized:
// lane width from the vectorizer's shared MaxVectorLoadBits policy, warp
// size and coalescing-segment granularity from the target (see
// BindMemoryGeometry).
//     bw    = coalescing segments touched, counted exactly per (vector
//             step, warp); intra-warp broadcast merges into one segment,
//             store lanes holding a replica != 0 sleep behind the
//             replication guard and count nothing
//     issue = per-thread instruction depth x what a fully busy block
//             would stream at max lane width over that many steps:
//             steps x threads x lane_bytes. Idle lanes do not shorten the
//             depth, so thread-collapse pathologies surface here (#1729).
//     time(S) ~ max(bw, issue);      cost = sum over statements.
// ---------------------------------------------------------------------------

/*! \brief A statement-lifetime compiled form of an index expression.
 *
 *  The scoring walk executes each expression once per (logical point,
 *  replica) — 10^4..10^6 times — so per-point symbolic evaluation is the
 *  wrong shape. Instead: canonicalize ONCE with the Analyzer (TIR
 *  semantics stay the Analyzer's), flatten ONCE into a postfix program,
 *  then execute as straight-line int64 arithmetic. Vars outside the slot
 *  map (block indices inside region offsets) compile to the constant 0:
 *  they shift every address of the statement equally and cancel out of
 *  both contiguity and segment geometry. An unsupported node fails
 *  COMPILATION, marking the statement outside the model up front — the
 *  same protocol the per-point evaluation signalled, just earlier. */
class ExprProgram {
public:
  static std::optional<ExprProgram>
  Compile(const PrimExpr &expr,
          const std::unordered_map<const VarNode *, int> &var_slots,
          arith::Analyzer *analyzer) {
    // Zero out foreign vars BEFORE canonicalization: Simplify then folds
    // their dead terms away (e.g. `bx*16384 + rest` -> `rest`), so the
    // compiled program carries only live instructions into the hot loop.
    // Non-int foreign vars are kept and fail Emit below, as before.
    PrimExpr canon =
        Substitute(expr, [&](const Var &var) -> ffi::Optional<PrimExpr> {
          if (var_slots.count(var.get()) ||
              (!var->dtype.is_int() && !var->dtype.is_uint())) {
            return ffi::Optional<PrimExpr>();
          }
          return make_zero(var->dtype); // foreign additive offset
        });
    ExprProgram prog;
    if (!prog.Emit(analyzer->Simplify(canon), var_slots)) {
      return std::nullopt;
    }
    // Verify the fixed eval stack suffices (postfix depth simulation).
    int depth = 0, max_depth = 0;
    for (const Instr &instr : prog.code_) {
      if (instr.op == Op::kConst || instr.op == Op::kVar) {
        ++depth;
      } else if (instr.op == Op::kSelect) {
        depth -= 2;
      } else {
        --depth;
      }
      max_depth = std::max(max_depth, depth);
    }
    if (max_depth > kMaxStackDepth) {
      return std::nullopt;
    }
    return prog;
  }

  /*! \brief Max eval-stack depth any program may need; programs deeper
   *  than this fail compilation (an index expression this deep is not a
   *  layout expression). Keeps Eval on a fixed C array. */
  static constexpr int kMaxStackDepth = 64;

  /*! \brief Evaluate against `slots` (one value per var slot). Nullopt
   *  only on a division by a computed zero. */
  std::optional<int64_t> Eval(const int64_t *slots) const {
    int64_t stack[kMaxStackDepth];
    int top = 0; // index one past the top of the stack
    for (const Instr &instr : code_) {
      switch (instr.op) {
      case Op::kConst:
        stack[top++] = instr.imm;
        break;
      case Op::kVar:
        stack[top++] = slots[instr.imm];
        break;
      case Op::kSelect: {
        int64_t b = stack[--top], a = stack[--top], cond = stack[--top];
        stack[top++] = cond != 0 ? a : b;
        break;
      }
      default: {
        int64_t b = stack[--top], a = stack[--top];
        int64_t v = 0;
        switch (instr.op) {
        case Op::kAdd:
          v = a + b;
          break;
        case Op::kSub:
          v = a - b;
          break;
        case Op::kMul:
          v = a * b;
          break;
        case Op::kFloorDiv: {
          if (b == 0) {
            return std::nullopt;
          }
          v = a / b;
          if ((a % b != 0) && ((a < 0) != (b < 0))) {
            --v;
          }
          break;
        }
        case Op::kFloorMod: {
          if (b == 0) {
            return std::nullopt;
          }
          v = a % b;
          if (v != 0 && ((v < 0) != (b < 0))) {
            v += b;
          }
          break;
        }
        case Op::kMin:
          v = std::min(a, b);
          break;
        case Op::kMax:
          v = std::max(a, b);
          break;
        case Op::kLT:
          v = a < b ? 1 : 0;
          break;
        case Op::kLE:
          v = a <= b ? 1 : 0;
          break;
        case Op::kGT:
          v = a > b ? 1 : 0;
          break;
        case Op::kGE:
          v = a >= b ? 1 : 0;
          break;
        case Op::kEQ:
          v = a == b ? 1 : 0;
          break;
        case Op::kNE:
          v = a != b ? 1 : 0;
          break;
        case Op::kAnd:
          v = (a != 0 && b != 0) ? 1 : 0;
          break;
        case Op::kOr:
          v = (a != 0 || b != 0) ? 1 : 0;
          break;
        case Op::kBitAnd:
          v = a & b;
          break;
        case Op::kBitOr:
          v = a | b;
          break;
        case Op::kBitXor:
          v = a ^ b;
          break;
        case Op::kShl:
          if (b < 0 || b > 63) {
            return std::nullopt;
          }
          v = a << b;
          break;
        case Op::kShr:
          if (b < 0 || b > 63) {
            return std::nullopt;
          }
          v = a >> b; // arithmetic shift, matching TIR's shift_right on ints
          break;
        default:
          return std::nullopt; // unreachable
        }
        stack[top++] = v;
        break;
      }
      }
    }
    return stack[top - 1];
  }

private:
  enum class Op : uint8_t {
    kConst,
    kVar,
    kAdd,
    kSub,
    kMul,
    kFloorDiv,
    kFloorMod,
    kMin,
    kMax,
    kSelect,
    kLT,
    kLE,
    kGT,
    kGE,
    kEQ,
    kNE,
    kAnd,
    kOr,
    kBitAnd,
    kBitOr,
    kBitXor,
    kShl,
    kShr,
  };
  struct Instr {
    Op op;
    int64_t imm{0}; // kConst: value; kVar: slot index
  };
  std::vector<Instr> code_;

  bool Emit(const PrimExpr &e,
            const std::unordered_map<const VarNode *, int> &var_slots) {
    if (const auto *imm = e.as<IntImmNode>()) {
      code_.push_back({Op::kConst, imm->value});
      return true;
    }
    if (const auto *var = e.as<VarNode>()) {
      if (!var->dtype.is_int() && !var->dtype.is_uint()) {
        return false;
      }
      auto it = var_slots.find(var);
      if (it != var_slots.end()) {
        code_.push_back({Op::kVar, it->second});
      } else {
        code_.push_back({Op::kConst, 0}); // foreign additive offset
      }
      return true;
    }
    if (const auto *op = e.as<CastNode>()) {
      return Emit(op->value, var_slots);
    }
    if (const auto *op = e.as<SelectNode>()) {
      return Emit(op->condition, var_slots) &&
             Emit(op->true_value, var_slots) &&
             Emit(op->false_value, var_slots) &&
             (code_.push_back({Op::kSelect}), true);
    }
    if (const auto *op = e.as<CallNode>()) {
      auto call_binary = [&](Op op_code) {
        return op->args.size() == 2 && Emit(op->args[0], var_slots) &&
               Emit(op->args[1], var_slots) &&
               (code_.push_back({op_code}), true);
      };
      if (op->op.same_as(builtin::if_then_else()) && op->args.size() == 3) {
        return Emit(op->args[0], var_slots) && Emit(op->args[1], var_slots) &&
               Emit(op->args[2], var_slots) &&
               (code_.push_back({Op::kSelect}), true);
      }
      if (op->op.same_as(builtin::bitwise_and())) {
        return call_binary(Op::kBitAnd);
      }
      if (op->op.same_as(builtin::bitwise_or())) {
        return call_binary(Op::kBitOr);
      }
      if (op->op.same_as(builtin::bitwise_xor())) {
        return call_binary(Op::kBitXor);
      }
      if (op->op.same_as(builtin::shift_left())) {
        return call_binary(Op::kShl);
      }
      if (op->op.same_as(builtin::shift_right())) {
        return call_binary(Op::kShr);
      }
      if (op->op.same_as(builtin::bitwise_not()) && op->args.size() == 1) {
        // ~a == a ^ -1
        if (!Emit(op->args[0], var_slots)) {
          return false;
        }
        code_.push_back({Op::kConst, -1});
        code_.push_back({Op::kBitXor});
        return true;
      }
      return false;
    }
    if (const auto *op = e.as<NotNode>()) {
      // !a  ==  (a == 0)
      if (!Emit(op->a, var_slots)) {
        return false;
      }
      code_.push_back({Op::kConst, 0});
      code_.push_back({Op::kEQ});
      return true;
    }
    auto binary = [&](const PrimExpr &a, const PrimExpr &b, Op op_code) {
      if (!Emit(a, var_slots) || !Emit(b, var_slots)) {
        return false;
      }
      code_.push_back({op_code});
      return true;
    };
    if (const auto *op = e.as<AddNode>()) {
      return binary(op->a, op->b, Op::kAdd);
    }
    if (const auto *op = e.as<SubNode>()) {
      return binary(op->a, op->b, Op::kSub);
    }
    if (const auto *op = e.as<MulNode>()) {
      return binary(op->a, op->b, Op::kMul);
    }
    if (const auto *op = e.as<FloorDivNode>()) {
      return binary(op->a, op->b, Op::kFloorDiv);
    }
    if (const auto *op = e.as<FloorModNode>()) {
      return binary(op->a, op->b, Op::kFloorMod);
    }
    if (const auto *op = e.as<MinNode>()) {
      return binary(op->a, op->b, Op::kMin);
    }
    if (const auto *op = e.as<MaxNode>()) {
      return binary(op->a, op->b, Op::kMax);
    }
    if (const auto *op = e.as<LTNode>()) {
      return binary(op->a, op->b, Op::kLT);
    }
    if (const auto *op = e.as<LENode>()) {
      return binary(op->a, op->b, Op::kLE);
    }
    if (const auto *op = e.as<GTNode>()) {
      return binary(op->a, op->b, Op::kGT);
    }
    if (const auto *op = e.as<GENode>()) {
      return binary(op->a, op->b, Op::kGE);
    }
    if (const auto *op = e.as<EQNode>()) {
      return binary(op->a, op->b, Op::kEQ);
    }
    if (const auto *op = e.as<NENode>()) {
      return binary(op->a, op->b, Op::kNE);
    }
    if (const auto *op = e.as<AndNode>()) {
      return binary(op->a, op->b, Op::kAnd);
    }
    if (const auto *op = e.as<OrNode>()) {
      return binary(op->a, op->b, Op::kOr);
    }
    return false;
  }
};

/*! \brief Reference evaluation through the Analyzer: substitute concrete
 *  values (vars outside the slot map fold to 0) and constant-fold. This is
 *  the semantic ORACLE ExprProgram must agree with — used only to validate
 *  freshly compiled programs at a few witness points, never per point. */
std::optional<int64_t>
EvalByAnalyzer(const PrimExpr &e,
               const std::unordered_map<const VarNode *, int> &var_slots,
               const int64_t *slots, arith::Analyzer *analyzer) {
  PrimExpr bound =
      Substitute(e, [&](const Var &var) -> ffi::Optional<PrimExpr> {
        if (!var->dtype.is_int() && !var->dtype.is_uint()) {
          return ffi::Optional<PrimExpr>();
        }
        auto it = var_slots.find(var.get());
        int64_t value = it == var_slots.end() ? 0 : slots[it->second];
        return IntImm(var->dtype, value);
      });
  PrimExpr folded = analyzer->Simplify(bound);
  if (const auto *imm = folded.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

/*! \brief Cross-check a compiled program against the Analyzer oracle at
 *  the given witness points. A disagreement means ExprProgram mis-compiled
 *  (or mis-executes) a node: warn loudly and reject the program, so a
 *  compiler bug degrades to the conservative worst case instead of ever
 *  reaching a score. Witnesses the oracle cannot fold are inconclusive
 *  and skipped. */
bool ValidateProgram(const ExprProgram &prog, const PrimExpr &expr,
                     const std::unordered_map<const VarNode *, int> &var_slots,
                     const std::vector<std::vector<int64_t>> &witnesses,
                     arith::Analyzer *analyzer) {
  for (const auto &witness : witnesses) {
    auto expected = EvalByAnalyzer(expr, var_slots, witness.data(), analyzer);
    if (!expected.has_value()) {
      continue; // oracle is silent here: inconclusive
    }
    auto got = prog.Eval(witness.data());
    if (!got.has_value() || *got != *expected) {
      LOG(WARNING) << "[LayoutCost] compiled index program disagrees with "
                      "the Analyzer (expected "
                   << *expected << ", got "
                   << (got.has_value() ? std::to_string(*got) : "<none>")
                   << ") for expr: " << expr
                   << " — rejecting the program; statement falls back to "
                      "the conservative worst case.";
      return false;
    }
  }
  return true;
}

/*! \brief One global-memory access stream of a statement. `addr` is the
 *  flat element index into the global buffer, written in the probe's
 *  point_vars — replica-independent by construction. */
struct GlobalAccessProbe {
  PrimExpr addr;
  int64_t elem_bytes{4};
  bool is_store{false};
  int64_t repeat{1}; // enclosing serial trip count: the address pattern
                     // replays (shifted) that many times per step
};

/*! \brief A statement prepared for scoring. Three states, told apart by
 *  two fields (every consumer must honor this protocol):
 *    accesses.empty()   — the statement touches no global memory: charge 0;
 *    !measurable        — geometry outside the model: charge WorstCaseBytes
 *                         (sized by `worst_elements`);
 *    otherwise          — measure with ScoreStatement (which may still
 *                         fail and fall back to WorstCaseBytes). */
struct StatementProbe {
  std::vector<GlobalAccessProbe> accesses;
  bool measurable{false};
  // Forward-walk geometry, valid when `measurable`:
  std::vector<int64_t> extents; // logical iteration space, outermost first
  std::vector<Var> point_vars;  // one per extent; addr/thread/slot
                                // expressions are written in these
  Var rep_var;                  // replica index var of thread_expr
  int64_t rep{1};
  int64_t threads{1};
  int64_t slots{1};     // per-thread serial slots
  PrimExpr thread_expr; // forward thread map: point_vars + rep_var
  PrimExpr slot_expr;   // forward slot map: point_vars
  // Widest vector access in bits the vectorizer will plan for this
  // statement's memory mix (MaxVectorLoadBits — the shared policy).
  int64_t vector_bits{128};
  // Memory-system geometry of the target: how many threads issue one
  // coalesced request together, and the byte granularity each request is
  // charged at.
  int64_t warp_size{32};
  int64_t segment_bytes{128};
  int64_t worst_elements{0};
};

struct StatementTraffic {
  int64_t bw{0};
  int64_t issue{0};
  int64_t Time() const { return std::max(bw, issue); }
};

/*! \brief The conservative charge for a statement outside the model:
 *  every element its own full-segment transaction. An attempt must never
 *  profit from opacity — evaluability depends on the layout under test. */
int64_t WorstCaseBytes(const StatementProbe &probe) {
  int64_t total = 0;
  for (const auto &access : probe.accesses) {
    total += probe.worst_elements * access.repeat * probe.segment_bytes;
  }
  return total;
}

/*! \brief Table cap: statements whose (thread, slot) table would exceed
 *  this fall back to the worst case. Generous — a 128-thread x 128-slot
 *  fragment is 16K entries — while bounding pathological replications. */
constexpr int64_t kMaxTableEntries = int64_t(1) << 20;
constexpr int64_t kAbsentAddr = std::numeric_limits<int64_t>::min();

/*! \brief Measure one prepared statement.
 *
 *  Walk every (logical point, replica) pair, evaluate the forward maps,
 *  and fill the (thread, slot) -> address table; any unevaluable or
 *  out-of-range value, any collision (a non-injective candidate), or a
 *  table over the cap aborts to nullopt and the caller charges the
 *  conservative worst case. From the table, exactly:
 *    vector = widest power-of-two width every thread's aligned slot
 *             blocks sustain contiguously (the vectorizer's question)
 *    issue  = steps x repeat x threads x lane_bytes  (instruction depth)
 *    bw     = repeat x segments x segment_bytes      (transaction bytes)
 *  with segments summed over every (vector step, warp). */
std::optional<StatementTraffic>
ScoreStatementImpl(const StatementProbe &probe) {
  if (probe.accesses.empty()) {
    return StatementTraffic{};
  }
  if (!probe.measurable || probe.threads <= 0 || probe.slots <= 0 ||
      probe.rep <= 0 || probe.warp_size <= 0 || probe.segment_bytes <= 0 ||
      !probe.rep_var.defined()) {
    return std::nullopt;
  }

  int64_t points = 1;
  for (int64_t extent : probe.extents) {
    if (extent <= 0 || points > kMaxTableEntries / extent) {
      return std::nullopt;
    }
    points *= extent;
  }
  int64_t table_size = probe.threads * probe.slots;
  // A fragment the lowering accepts is a bijection between
  // (logical point, replica) and (thread, slot); anything else is
  // outside the model. The size identity is the cheap necessary half;
  // the collision check during the walk is the sufficient half.
  if (table_size > kMaxTableEntries || points > kMaxTableEntries / probe.rep ||
      points * probe.rep != table_size) {
    return std::nullopt;
  }

  size_t naccess = probe.accesses.size();
  std::vector<std::vector<int64_t>> addr_table(
      naccess, std::vector<int64_t>(table_size, kAbsentAddr));
  // Cells whose entry is the lead replica (r == 0): the only lanes that
  // stay active for stores under the replication guard.
  std::vector<uint8_t> lead_replica(table_size, 0);

  // Compile every expression once (Analyzer canonicalization included);
  // the walk below is then pure int64 arithmetic. Slot layout:
  // [point_vars..., rep_var].
  size_t ndim = probe.point_vars.size();
  arith::Analyzer analyzer;
  std::unordered_map<const VarNode *, int> var_slots;
  for (size_t d = 0; d < ndim; ++d) {
    var_slots[probe.point_vars[d].get()] = static_cast<int>(d);
  }
  var_slots[probe.rep_var.get()] = static_cast<int>(ndim);
  // Witness points for differential validation: origin, far corner, and an
  // interior point (rep slot included). Compilation failure and oracle
  // disagreement both reject to the worst case — never a wrong score.
  std::vector<std::vector<int64_t>> witnesses;
  for (int kind = 0; kind < 3; ++kind) {
    std::vector<int64_t> w(ndim + 1, 0);
    for (size_t d = 0; d < ndim; ++d) {
      int64_t last = probe.extents[d] - 1;
      w[d] = kind == 0 ? 0 : (kind == 1 ? last : last / 2);
    }
    w[ndim] = kind == 0 ? 0 : (kind == 1 ? probe.rep - 1 : probe.rep / 2);
    witnesses.push_back(std::move(w));
  }
  auto compile_checked =
      [&](const PrimExpr &expr) -> std::optional<ExprProgram> {
    auto prog = ExprProgram::Compile(expr, var_slots, &analyzer);
    if (!prog) {
      DLOG(INFO) << "[LayoutCost] unsupported node in index expr, "
                    "charged worst-case: "
                 << expr;
      return std::nullopt;
    }
    if (!ValidateProgram(*prog, expr, var_slots, witnesses, &analyzer)) {
      return std::nullopt;
    }
    return prog;
  };
  auto slot_prog = compile_checked(probe.slot_expr);
  auto thread_prog = compile_checked(probe.thread_expr);
  if (!slot_prog || !thread_prog) {
    return std::nullopt;
  }
  std::vector<ExprProgram> addr_progs;
  addr_progs.reserve(naccess);
  for (const auto &access : probe.accesses) {
    auto prog = compile_checked(access.addr);
    if (!prog) {
      return std::nullopt;
    }
    addr_progs.push_back(std::move(*prog));
  }

  std::vector<int64_t> slots_buf(ndim + 1, 0);
  std::vector<int64_t> point_addr(naccess, 0);
  for (int64_t flat = 0; flat < points; ++flat) {
    slots_buf[ndim] = 0;
    // Address and slot are replica-independent: evaluate once per point.
    for (size_t a = 0; a < naccess; ++a) {
      auto addr = addr_progs[a].Eval(slots_buf.data());
      if (!addr) {
        return std::nullopt;
      }
      point_addr[a] = *addr;
    }
    auto slot = slot_prog->Eval(slots_buf.data());
    if (!slot || *slot < 0 || *slot >= probe.slots) {
      return std::nullopt;
    }
    for (int64_t r = 0; r < probe.rep; ++r) {
      slots_buf[ndim] = r;
      auto thread = thread_prog->Eval(slots_buf.data());
      if (!thread || *thread < 0 || *thread >= probe.threads) {
        return std::nullopt;
      }
      int64_t cell = *thread * probe.slots + *slot;
      if (addr_table[0][cell] != kAbsentAddr) {
        return std::nullopt; // collision: candidate is not injective
      }
      for (size_t a = 0; a < naccess; ++a) {
        addr_table[a][cell] = point_addr[a];
      }
      if (r == 0) {
        lead_replica[cell] = 1;
      }
    }
    // Advance the multi-index (row-major, innermost fastest).
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
      if (++slots_buf[d] < probe.extents[d]) {
        break;
      }
      slots_buf[d] = 0;
    }
  }
  // points * rep == table_size and no collisions: every cell is filled.

  // Widest power-of-two vector width every access of the statement
  // sustains. This is the numeric mirror of IndicesCanVectorize (the
  // vectorizer's symbolic predicate), checked exhaustively on the table:
  //   - width cap: the shared MaxVectorLoadBits policy per access dtype;
  //   - extent divisibility: slots % cand == 0;
  //   - base alignment: every block base divisible by the width;
  //   - contiguity: slot + 1 advances every address by exactly 1 inside
  //     aligned width-sized blocks.
  int64_t vector_lane_bytes = probe.vector_bits / 8;
  int64_t max_vector = probe.slots;
  for (const auto &access : probe.accesses) {
    max_vector = std::min<int64_t>(max_vector,
                                   vector_lane_bytes /
                                       std::max<int64_t>(1, access.elem_bytes));
  }
  int64_t vector = 1;
  for (int64_t cand = 32; cand >= 2; cand /= 2) {
    if (cand > max_vector || probe.slots % cand != 0) {
      continue;
    }
    bool contiguous = true;
    for (size_t a = 0; a < naccess && contiguous; ++a) {
      for (int64_t t = 0; t < probe.threads && contiguous; ++t) {
        const int64_t *row = addr_table[a].data() + t * probe.slots;
        for (int64_t q = 0; q < probe.slots && contiguous; q += cand) {
          if (row[q] % cand != 0) {
            contiguous = false; // misaligned base: vectorizer would reject
            break;
          }
          for (int64_t offset = 1; offset < cand; ++offset) {
            if (row[q + offset] != row[q] + offset) {
              contiguous = false;
              break;
            }
          }
        }
      }
    }
    if (contiguous) {
      vector = cand;
      break;
    }
  }
  int64_t steps = probe.slots / vector;

  StatementTraffic traffic;
  int64_t num_warps = (probe.threads + probe.warp_size - 1) / probe.warp_size;
  // Distinct segments per (step, warp): a warp touches only a handful, so
  // a reused flat vector with linear dedupe beats a heap-allocating set.
  std::vector<int64_t> segments;
  segments.reserve(2 * probe.warp_size);
  for (size_t a = 0; a < naccess; ++a) {
    const auto &access = probe.accesses[a];
    traffic.issue += steps * access.repeat * probe.threads * vector_lane_bytes;

    int64_t segments_total = 0;
    for (int64_t q = 0; q < steps; ++q) {
      for (int64_t w = 0; w < num_warps; ++w) {
        segments.clear();
        for (int64_t lane = 0; lane < probe.warp_size; ++lane) {
          int64_t t = w * probe.warp_size + lane;
          if (t >= probe.threads) {
            break;
          }
          int64_t cell = t * probe.slots + q * vector;
          if (access.is_store && !lead_replica[cell]) {
            continue; // guarded replica: this lane is idle for stores
          }
          int64_t first_byte = addr_table[a][cell] * access.elem_bytes;
          int64_t last_byte = first_byte + vector * access.elem_bytes - 1;
          for (int64_t seg = first_byte / probe.segment_bytes;
               seg <= last_byte / probe.segment_bytes; ++seg) {
            if (std::find(segments.begin(), segments.end(), seg) ==
                segments.end()) {
              segments.push_back(seg);
            }
          }
        }
        segments_total += static_cast<int64_t>(segments.size());
      }
    }
    traffic.bw += access.repeat * segments_total * probe.segment_bytes;
  }
  return traffic;
}

/*! \brief Exception-safe wrapper: printing/substitution on pathological
 *  candidate layouts can throw deep inside the layout stack; every such
 *  case is simply outside the model. */
std::optional<StatementTraffic> ScoreStatement(const StatementProbe &probe) {
  try {
    return ScoreStatementImpl(probe);
  } catch (const std::exception &e) {
    DLOG(INFO) << "[LayoutCost] statement scoring threw: " << e.what();
    return std::nullopt;
  }
}

/*! \brief Row-major element strides; nullopt when the shape is symbolic
 *  or the buffer exceeds the int32 address algebra. */
std::optional<std::vector<int64_t>> RowMajorStrides(const Buffer &buffer) {
  size_t ndim = buffer->shape.size();
  std::vector<int64_t> strides(ndim, 1);
  for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
    const int64_t *extent = as_const_int(buffer->shape[d + 1]);
    if (!extent) {
      return std::nullopt;
    }
    strides[d] = strides[d + 1] * *extent;
  }
  const int64_t *outer = as_const_int(buffer->shape[0]);
  if (!outer || strides[0] * *outer >= (int64_t(1) << 31)) {
    return std::nullopt;
  }
  return strides;
}

/*! \brief Bind the target's memory-system geometry on a probe. The warp
 *  size comes from the target (32 CUDA lanes, 64 ROCm wavefront lanes);
 *  the segment granularity stays at 128B — the L1-line coalescing unit on
 *  NVIDIA and a serviceable approximation elsewhere — until calibration
 *  demands a per-target dispatch here. */
void BindMemoryGeometry(StatementProbe *probe, const Target &target) {
  probe->warp_size = target->GetAttr<Integer>("thread_warp_size", Integer(32))
                         .value()
                         .IntValue();
  probe->segment_bytes = 128;
}

/*! \brief Fresh point vars for a probe's logical space. */
std::vector<Var> MakePointVars(size_t ndim) {
  std::vector<Var> vars;
  vars.reserve(ndim);
  for (size_t d = 0; d < ndim; ++d) {
    vars.push_back(Var("_cost_x" + std::to_string(d), DataType::Int(32)));
  }
  return vars;
}

/*! \brief Install a fragment's forward maps on a probe: point_vars must
 *  already be set. Returns false (probe untouched otherwise) when the
 *  fragment's geometry is not constant-sized. */
bool BindForwardMaps(StatementProbe *probe, const Fragment &layout) {
  if (layout->OutputDim() != 1) {
    return false;
  }
  // Keep the expressions alive: as_const_int returns a pointer into the
  // node, and OutputShape()/ThreadExtent() build fresh temporaries.
  PrimExpr slots_expr = layout->OutputShape()[0];
  PrimExpr threads_expr = layout->ThreadExtent();
  PrimExpr rep_expr = layout->ReplicateExtent();
  const int64_t *slots_ptr = as_const_int(slots_expr);
  const int64_t *threads_ptr = as_const_int(threads_expr);
  const int64_t *rep_ptr = as_const_int(rep_expr);
  if (!slots_ptr || !threads_ptr || !rep_ptr || *slots_ptr <= 0 ||
      *threads_ptr <= 0 || *rep_ptr <= 0) {
    return false;
  }
  Array<PrimExpr> points;
  for (const Var &v : probe->point_vars) {
    points.push_back(v);
  }
  probe->rep_var = Var("_cost_rep", DataType::Int(32));
  probe->thread_expr = layout->ForwardThread(points, probe->rep_var);
  probe->slot_expr = layout->Forward(points)[0];
  probe->slots = *slots_ptr;
  probe->threads = *threads_ptr;
  probe->rep = *rep_ptr;
  return true;
}

/*! \brief Prepare a fragment<->global tl.copy for scoring: the logical
 *  walk is the fragment's full logical shape, the thread/slot maps are
 *  the fragment layout's own forward expressions, and the single access
 *  is the global side's affine address in the same logical point. */
StatementProbe BuildCopyProbe(const CopyNode *copy, const Fragment &frag_layout,
                              bool frag_is_src, const Target &target) {
  const Buffer &global = frag_is_src ? copy->dst : copy->src;
  const Array<Range> &frag_range =
      frag_is_src ? copy->src_range : copy->dst_range;
  const Array<Range> &global_range =
      frag_is_src ? copy->dst_range : copy->src_range;

  // Sized worst-case fallback for every shape outside the model: logical
  // elements from the fragment-side range (symbolic extents count as 1 —
  // worst geometry, optimistic count).
  auto worst_only = [&]() {
    StatementProbe probe;
    BindMemoryGeometry(&probe, target);
    probe.worst_elements = 1;
    for (const Range &r : frag_range) {
      const int64_t *extent = as_const_int(r->extent);
      probe.worst_elements *= extent ? *extent : 1;
    }
    GlobalAccessProbe access;
    access.is_store = frag_is_src;
    probe.accesses.push_back(std::move(access));
    return probe;
  };

  size_t ndim = frag_layout->InputShape().size();
  if (frag_range.size() != ndim || global_range.size() != ndim) {
    return worst_only();
  }
  // Whole-fragment copies only: the iteration space is the fragment's full
  // logical shape.
  std::vector<int64_t> extents;
  int64_t logical_elements = 1;
  for (size_t d = 0; d < ndim; ++d) {
    if (!is_zero(frag_range[d]->min)) {
      return worst_only();
    }
    const int64_t *frag_extent = as_const_int(frag_range[d]->extent);
    const int64_t *global_extent = as_const_int(global_range[d]->extent);
    const int64_t *shape_extent = as_const_int(frag_layout->InputShape()[d]);
    if (!frag_extent || !global_extent || !shape_extent ||
        *frag_extent != *shape_extent || *global_extent != *shape_extent) {
      return worst_only();
    }
    extents.push_back(*shape_extent);
    logical_elements *= *shape_extent;
  }

  int64_t elem_bits = global->dtype.bits() * global->dtype.lanes();
  if (elem_bits < 8) {
    return worst_only();
  }
  auto strides = RowMajorStrides(global);
  if (!strides.has_value()) {
    return worst_only();
  }

  StatementProbe probe;
  probe.worst_elements = logical_elements;
  probe.extents = std::move(extents);
  probe.point_vars = MakePointVars(ndim);
  if (!BindForwardMaps(&probe, frag_layout)) {
    return worst_only();
  }
  // A fragment<->global copy touches global memory and no shared memory.
  probe.vector_bits = MaxVectorLoadBits(target, /*global_only_access=*/true);
  BindMemoryGeometry(&probe, target);

  PrimExpr addr = make_zero(DataType::Int(32));
  for (size_t d = 0; d < ndim; ++d) {
    addr = addr + (global_range[d]->min + probe.point_vars[d]) *
                      IntImm(DataType::Int(32), (*strides)[d]);
  }
  GlobalAccessProbe access;
  access.addr = std::move(addr);
  access.elem_bytes = elem_bits / 8;
  access.is_store = frag_is_src;
  probe.accesses.push_back(std::move(access));
  probe.measurable = true;
  return probe;
}

/*! \brief Collect the direct global-memory accesses of a parallel loop
 *  body, with the trip count of any enclosing serial loops (the access
 *  pattern replays, shifted, once per serial iteration). */
class LoopGlobalAccessCollector : public StmtExprVisitor {
public:
  struct RawAccess {
    Buffer buffer;
    Array<PrimExpr> indices;
    bool is_store;
    int64_t repeat;
    bool symbolic_repeat;
  };
  std::vector<RawAccess> accesses;
  // Whether the loop body also touches shared memory: feeds the shared
  // width-cap policy (256-bit loads are global-only).
  bool touches_shared{false};

  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

private:
  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    const int64_t *extent = as_const_int(op->extent);
    serial_stack_.push_back(extent ? *extent : -1);
    StmtExprVisitor::VisitStmt_(op);
    serial_stack_.pop_back();
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    Record(op->buffer, op->indices, /*is_store=*/true);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    Record(op->buffer, op->indices, /*is_store=*/false);
    StmtExprVisitor::VisitExpr_(op);
  }
  void Record(const Buffer &buffer, const Array<PrimExpr> &indices,
              bool is_store) {
    if (IsSharedBuffer(buffer)) {
      touches_shared = true;
    }
    if (!IsGlobalBuffer(buffer)) {
      return;
    }
    int64_t repeat = 1;
    bool symbolic = false;
    for (int64_t extent : serial_stack_) {
      if (extent < 0) {
        symbolic = true;
      } else {
        repeat *= extent;
      }
    }
    accesses.push_back(RawAccess{buffer, indices, is_store, repeat, symbolic});
  }
  std::vector<int64_t> serial_stack_;
};

/*! \brief Prepare a parallel loop with direct global accesses for scoring
 *  (issue #1729, extension E1): the logical walk is the loop nest itself,
 *  the thread/slot maps are the loop layout's forward expressions bound
 *  to the loop vars, and each access keeps its raw index expressions —
 *  they are already written in the loop vars, so no substitution at all.
 *  Returns: probe with empty accesses (charge zero) when the loop touches
 *  no global memory; a full probe otherwise; nullopt when the loop cannot
 *  even be sized (skip — nothing sensible to charge). */
std::optional<StatementProbe> BuildLoopProbe(const ParallelOpNode *loop,
                                             const Target &target) {
  LoopGlobalAccessCollector collector;
  collector.Collect(loop->GetRoot());
  StatementProbe probe;
  if (collector.accesses.empty()) {
    return probe; // pure fragment/shared loop: no global traffic to model
  }

  // Loop nest geometry, taken from the For nest so it exists even when
  // the loop has no layout: an attempt must never profit from an unsolved
  // (opaque) loop.
  std::vector<int64_t> extents;
  std::vector<Var> nest_vars;
  int64_t domain = 1;
  {
    const ForNode *cur = loop->GetRoot().get();
    while (cur != nullptr && cur->kind == ForKind::kParallel) {
      const int64_t *value = as_const_int(cur->extent);
      if (!value) {
        return std::nullopt; // truly unsizeable: nothing sensible to charge
      }
      extents.push_back(*value);
      nest_vars.push_back(cur->loop_var);
      domain *= *value;
      cur = cur->body.as<ForNode>();
    }
  }
  probe.worst_elements = domain;

  // Mark the probe un-scoreable but sized (worst-case) when any structural
  // requirement fails. Symbolic serial trip counts fall back to repeat=1:
  // worst geometry, optimistic count.
  auto worst_only = [&]() -> std::optional<StatementProbe> {
    StatementProbe worst;
    BindMemoryGeometry(&worst, target);
    worst.worst_elements = probe.worst_elements;
    for (const auto &raw : collector.accesses) {
      GlobalAccessProbe access;
      access.elem_bytes = std::max<int64_t>(
          1, raw.buffer->dtype.bits() * raw.buffer->dtype.lanes() / 8);
      access.is_store = raw.is_store;
      access.repeat = raw.symbolic_repeat ? 1 : raw.repeat;
      worst.accesses.push_back(std::move(access));
    }
    return worst;
  };

  Fragment layout = loop->GetLoopLayout();
  if (!layout.defined()) {
    DLOG(INFO) << "[LayoutCost] loop layout undefined; charged worst-case";
    return worst_only();
  }
  if (nest_vars.size() != layout->InputShape().size()) {
    return worst_only();
  }
  probe.extents = std::move(extents);
  probe.point_vars = std::move(nest_vars);
  if (!BindForwardMaps(&probe, layout)) {
    return worst_only();
  }
  probe.vector_bits = MaxVectorLoadBits(
      target, /*global_only_access=*/!collector.touches_shared);
  BindMemoryGeometry(&probe, target);

  for (const auto &raw : collector.accesses) {
    if (raw.symbolic_repeat) {
      return worst_only();
    }
    int64_t elem_bits = raw.buffer->dtype.bits() * raw.buffer->dtype.lanes();
    auto strides = RowMajorStrides(raw.buffer);
    if (elem_bits < 8 || !strides.has_value() ||
        raw.indices.size() != raw.buffer->shape.size()) {
      return worst_only();
    }
    PrimExpr addr = make_zero(DataType::Int(32));
    for (size_t d = 0; d < raw.indices.size(); ++d) {
      addr = addr + raw.indices[d] * IntImm(DataType::Int(32), (*strides)[d]);
    }
    GlobalAccessProbe access;
    access.addr = std::move(addr);
    access.elem_bytes = elem_bits / 8;
    access.is_store = raw.is_store;
    access.repeat = raw.repeat;
    probe.accesses.push_back(std::move(access));
  }
  probe.measurable = true;
  return probe;
}

/*! \brief Total per-thread register slots across every fragment layout of
 *  the attempt. Shared by every cost model: it is the legacy score and the
 *  IO-aware model's tiebreak. */
int64_t CountRegisterSlots(const LayoutMap &tmp_layout_map) {
  int64_t regs = 0;
  for (const auto &[buffer, layout] : tmp_layout_map) {
    if (auto frag = layout.as<Fragment>()) {
      int64_t frag_reg_num = 1;
      for (auto i : frag.value()->OutputShape()) {
        auto pci = as_const_int(i);
        ICHECK(pci != nullptr) << "Can not use non-constant range to "
                                  "iterate over a fragment/local "
                                  "buffer. Non-constant shape expr is: "
                               << i
                               << ". This is possibly because you use "
                                  "symbolic shape when "
                                  "accessing a fragment/local buffer."
                               << SpanHintSuffix(buffer->span);
        frag_reg_num *= *pci;
      }
      regs += frag_reg_num;
    }
  }
  return regs;
}

/*! \brief Legacy policy: total register slots, nothing else. `mem` stays 0
 *  so the ordering is byte-identical to the historical register-count
 *  selection. */
class RegisterCountCostModel final : public LayoutCostModel {
public:
  AttemptCost Score(const std::vector<int> &members,
                    const std::vector<TileOperator> &infer_list,
                    const LayoutMap &tmp_layout_map) const final {
    (void)members;
    (void)infer_list;
    AttemptCost cost;
    cost.regs = CountRegisterSlots(tmp_layout_map);
    return cost;
  }
  const char *Name() const final { return "register-count"; }
};

/*! \brief IO-aware policy (layout RFC, design B2): every global-touching
 *  statement is charged max(bandwidth bytes, issue-equivalent bytes) under
 *  the attempt's tentative layouts; statements outside the model are
 *  charged a conservative worst case (an attempt must never profit from
 *  opacity). Registers remain the lexicographic tiebreak. */
class IOAwareCostModel final : public LayoutCostModel {
public:
  explicit IOAwareCostModel(Target target) : target_(std::move(target)) {}

  AttemptCost Score(const std::vector<int> &members,
                    const std::vector<TileOperator> &infer_list,
                    const LayoutMap &tmp_layout_map) const final {
    AttemptCost cost;
    cost.regs = CountRegisterSlots(tmp_layout_map);

    for (int idx : members) {
      if (const auto *copy = infer_list[idx].as<CopyNode>()) {
        bool src_frag = IsFragmentBuffer(copy->src);
        bool dst_frag = IsFragmentBuffer(copy->dst);
        Buffer frag;
        bool frag_is_src = false;
        if (src_frag && IsGlobalBuffer(copy->dst)) {
          frag = copy->src;
          frag_is_src = true;
        } else if (dst_frag && IsGlobalBuffer(copy->src)) {
          frag = copy->dst;
        } else {
          continue; // register moves / shared staging: out of the model
        }
        auto layout = tmp_layout_map.Get(frag);
        if (!layout.has_value()) {
          continue;
        }
        auto frag_layout = layout.value().as<Fragment>();
        if (!frag_layout.has_value()) {
          continue;
        }
        cost.mem += CachedStatementMem(idx, frag_layout.value(), [&]() {
          std::optional<StatementProbe> probe;
          try {
            probe =
                BuildCopyProbe(copy, frag_layout.value(), frag_is_src, target_);
          } catch (const std::exception &e) {
            probe = std::nullopt; // skipped below; builder-side fallbacks
                                  // cover every non-throwing failure
          }
          return ChargeStatement(probe, "copy");
        });
      } else if (const auto *loop = infer_list[idx].as<ParallelOpNode>()) {
        auto compute = [&]() {
          std::optional<StatementProbe> probe;
          try {
            probe = BuildLoopProbe(loop, target_);
          } catch (const std::exception &e) {
            probe = std::nullopt;
          }
          return ChargeStatement(probe, "parallel loop");
        };
        Fragment loop_layout = loop->GetLoopLayout();
        cost.mem += loop_layout.defined()
                        ? CachedStatementMem(idx, loop_layout, compute)
                        : compute();
      }
    }
    return cost;
  }

  const char *Name() const final { return "io-aware"; }

private:
  /*! \brief Final charge of one prepared statement, honoring the probe's
   *  three-state protocol (zero / worst-case / measured). */
  static int64_t ChargeStatement(const std::optional<StatementProbe> &probe,
                                 const char *what) {
    if (!probe.has_value() || probe->accesses.empty()) {
      return 0; // no global traffic to model / nothing sensible to charge
    }
    std::optional<StatementTraffic> traffic;
    if (probe->measurable) {
      traffic = ScoreStatement(*probe);
    }
    if (traffic.has_value()) {
      DLOG(INFO) << "[LayoutCost] " << what << ": bw=" << traffic->bw
                 << " issue=" << traffic->issue;
      return traffic->Time();
    }
    DLOG(INFO) << "[LayoutCost] " << what
               << " outside the model; charged worst-case.";
    return WorstCaseBytes(*probe);
  }

  /*! \brief Memoize a statement's charge by (op index, layout): the charge
   *  depends only on the op (fixed per index) and the layout under test,
   *  and different attempt roots frequently converge to the same layout —
   *  those attempts then score the statement for free. Structural layout
   *  equality; entries per index stay tiny (one per distinct layout). */
  template <typename F>
  int64_t CachedStatementMem(int idx, const Fragment &layout,
                             F &&compute) const {
    auto &entries = stmt_cache_[idx];
    for (const auto &[cached_layout, mem] : entries) {
      if (cached_layout->IsEqual(layout.get())) {
        return mem;
      }
    }
    int64_t mem = compute();
    entries.emplace_back(layout, mem);
    return mem;
  }

  Target target_;
  mutable std::unordered_map<int, std::vector<std::pair<Fragment, int64_t>>>
      stmt_cache_;
};

} // namespace

std::unique_ptr<LayoutCostModel> LayoutCostModel::Create(bool io_aware,
                                                         Target target) {
  if (io_aware) {
    return std::make_unique<IOAwareCostModel>(std::move(target));
  }
  return std::make_unique<RegisterCountCostModel>();
}

} // namespace tl
} // namespace tvm
