/*!
 * \file normalize_conflict_hints.cc
 * \brief Consume conflict schedule hints into loop/root annotations.
 *
 * The frontend emits a statement-form marker
 *   Evaluate(Call(tl.conflict_hint, a, b, level, cross, group,
 *                 is_conflict))
 * where `a`/`b` are each a `tl.region` Call (a concrete region) or a bare
 * buffer's data Var (the whole buffer, matched by storage key), `level` is an
 * IntImm (`-2` = tilelang_root plus every common enclosing loop, `-1` = only
 * tilelang_root's outermost sequence, non-negative = one common enclosing loop
 * counted from outermost),
 * `cross` is an IntImm (-1=any / 1=cross-iter / 0=same-iter), `group` is a
 * StringImm tag ("" = none), and `is_conflict` selects forced conflict rather
 * than forced non-conflict.
 *
 * The statement form (rather than an AttrStmt) is deliberate: its region Call
 * args are simplified/inlined by Simplify in lockstep with the real T.copy
 * accesses (an AttrStmt's `node` array is NOT descended into by Simplify, which
 * froze the region's symbolic vars and broke region matching).
 *
 * This pass runs before AutoSchedule and:
 *   - removes each marker (drops the Evaluate from its SeqStmt),
 *   - attaches each hint to the enclosing For(s) selected by `level` (for a
 *     `group`, `level` indexes the loops enclosing BOTH sites -- the common
 *     ancestor prefix -- where the cross-scope dependency is analyzed),
 *   - pairs the two half-declarations sharing a `group` tag (must appear
 *     exactly twice),
 *   - normalizes each hint into an
 *     `[a, b, IntImm(cross_code), Bool(is_conflict)]` entry appended to those
 *     Fors' `annotations["conflict_hint"]` (operands kept as Var or
 *     BufferRegion),
 *   - attaches `level=-1` hints, plus the root copy of `level=-2` hints, to
 *     tilelang_root under `tl.root_conflict_hints` for root-sequence analysis.
 *
 * AutoSchedule's dependency analysis reads the annotation at exactly the
 * sequence level being analyzed and applies the declared conflict polarity.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/scheduled_tir.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using namespace ascend;

// Op registered in src/op/schedule_hint.cc.
static constexpr const char *kConflictHintOpName = "tl.conflict_hint";
static constexpr const char *kConflictHintAnnotKey = "conflict_hint";
static constexpr int kAllEnclosingScopes = -2;
static constexpr int kRootSequence = -1;

class ConflictHintNormalizer : public StmtExprMutator {
public:
  static SBlock Rewrite(SBlock root) {
    ConflictHintNormalizer m;
    root.CopyOnWrite()->body = m(root->body);
    ICHECK(m.no_conflict_.stack.empty() && m.conflict_.stack.empty())
        << "unbalanced conflict-hint loop stack";
    m.CheckGroupsComplete(m.no_conflict_, HintKind::kNoConflict);
    m.CheckGroupsComplete(m.conflict_, HintKind::kConflict);
    if (!m.root_hints_.empty()) {
      root.CopyOnWrite()->annotations.Set(kRootConflictHints,
                                          std::move(m.root_hints_));
    }
    return root;
  }

private:
  enum class HintKind { kNoConflict, kConflict };

  // First-seen half of each group tag, awaiting its second (and only) partner:
  // its operand, its `level`, its `cross`, and the chain of loops enclosing it.
  struct PendingHalf {
    Any operand;
    int level;
    int cross;
    std::vector<For> chain;
  };

  struct HintState {
    // Per enclosing For (outermost first), the entries to attach on exit.
    std::vector<Array<Any>> stack;
    std::unordered_map<std::string, PendingHalf> pending_group;
    // Group tags already paired, to reject a third occurrence in one kernel.
    std::unordered_set<std::string> completed_group;
  };

  HintState no_conflict_;
  HintState conflict_;
  // The enclosing For identities (outermost first), retained as ObjectRef
  // handles because grouped declarations may outlive one visitor callback.
  std::vector<For> loop_chain_;
  // Hints for tilelang_root's outermost sequence, attached on exit.
  Array<Any> root_hints_;

  static const char *HintName(HintKind kind) {
    return kind == HintKind::kConflict ? "assume_conflict"
                                       : "assume_no_conflict";
  }

  HintState &State(HintKind kind) {
    return kind == HintKind::kConflict ? conflict_ : no_conflict_;
  }

  void CheckGroupsComplete(const HintState &state, HintKind kind) const {
    ICHECK(state.pending_group.empty())
        << HintName(kind) << " group '" << state.pending_group.begin()->first
        << "' appears only once; each group must appear exactly twice";
  }

  // A resolved entry [a, b, IntImm(cross_code), Bool(is_conflict)]; a/b are
  // Var or BufferRegion, kept verbatim.
  static Array<Any> HintEntry(HintKind kind, const Any &a, const Any &b,
                              int cross) {
    Array<Any> entry;
    entry.push_back(a);
    entry.push_back(b);
    entry.push_back(IntImm(DataType::Int(32), cross));
    entry.push_back(Bool(kind == HintKind::kConflict));
    return entry;
  }

  // Decode a marker operand: a bare buffer arrives as its data Var (a handle)
  // and is kept as a Var (matched by storage key downstream); a tl.region Call
  // is reconstructed into its (Simplify'd) BufferRegion.
  static Any DecodeOperand(const PrimExpr &arg) {
    if (arg.as<VarNode>())
      return arg;
    return NormalizeToBufferRegion(arg);
  }

  // Length of the longest common (outermost) prefix of two loop chains.
  static int CommonPrefix(const std::vector<For> &x,
                          const std::vector<For> &y) {
    int n = static_cast<int>(std::min(x.size(), y.size()));
    int i = 0;
    while (i < n && x[i].same_as(y[i]))
      ++i;
    return i;
  }

  // Attach an entry at the selected sequence level. `bound` caps the
  // addressable loop depth (for a group, the common-prefix length shared by
  // both halves; otherwise the full stack).
  void Attach(HintKind kind, int level, int bound, const Any &a, const Any &b,
              int cross) {
    HintState &state = State(kind);
    if (level == kAllEnclosingScopes) {
      for (int i = 0; i < bound; ++i)
        state.stack[i].push_back(HintEntry(kind, a, b, cross));
      root_hints_.push_back(HintEntry(kind, a, b, cross));
      return;
    }
    if (level == kRootSequence) {
      root_hints_.push_back(HintEntry(kind, a, b, cross));
      return;
    }
    ICHECK_GE(level, 0) << HintName(kind)
                        << " level must be >= " << kAllEnclosingScopes
                        << ", got " << level;
    ICHECK(level < bound) << HintName(kind) << " level " << level
                          << " is out of range (only " << bound
                          << " enclosing loop(s))";
    state.stack[level].push_back(HintEntry(kind, a, b, cross));
  }

  // Consume one marker Call. Returns nothing; drops the statement.
  void ConsumeMarker(const CallNode *call) {
    const auto &args = call->args;
    ICHECK_EQ(args.size(), 6u) << "conflict-hint marker expects 6 args";
    const auto *is_conflict = args[5].as<IntImmNode>();
    ICHECK(is_conflict != nullptr && is_conflict->dtype.is_bool())
        << "conflict-hint marker expects a constant boolean polarity";
    HintKind kind =
        is_conflict->value ? HintKind::kConflict : HintKind::kNoConflict;
    HintState &state = State(kind);
    Any a = DecodeOperand(args[0]);
    Any b = DecodeOperand(args[1]);
    int level = static_cast<int>(Downcast<IntImm>(args[2])->value);
    int cross = static_cast<int>(Downcast<IntImm>(args[3])->value);
    std::string group = Downcast<StringImm>(args[4])->value;
    int depth = static_cast<int>(state.stack.size());

    if (group.empty()) {
      Attach(kind, level, depth, a, b, cross);
      return;
    }
    // group pairing: a tag must appear exactly twice, symmetrically (which half
    // is seen first is irrelevant). Both must carry the same `level` and
    // `cross`. The first records its region, level, cross and enclosing loop
    // chain; the second attaches the combined hint, bounding `level` to the LCA
    // (longest common prefix of the two chains). A third occurrence is an
    // error.
    ICHECK(!state.completed_group.count(group))
        << HintName(kind) << " group '" << group
        << "' appears more than twice; each group must appear exactly twice";
    auto it = state.pending_group.find(group);
    if (it == state.pending_group.end()) {
      state.pending_group.emplace(group,
                                  PendingHalf{a, level, cross, loop_chain_});
    } else {
      ICHECK(it->second.level == level)
          << HintName(kind) << " group '" << group
          << "' has mismatched level between its two half-declarations ("
          << it->second.level << " vs " << level << ")";
      ICHECK(it->second.cross == cross)
          << HintName(kind) << " group '" << group
          << "' has mismatched cross between its two half-declarations ("
          << it->second.cross << " vs " << cross << ")";
      int lca = CommonPrefix(it->second.chain, loop_chain_);
      Attach(kind, level, lca, it->second.operand, a, cross);
      state.pending_group.erase(it);
      state.completed_group.insert(group);
    }
  }

  static bool IsZeroEvaluate(const Stmt &s) {
    if (const auto *eval = s.as<EvaluateNode>())
      if (auto imm = eval->value.as<IntImmNode>())
        return imm->value == 0;
    return false;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(Op::Get(kConflictHintOpName))) {
        ConsumeMarker(call);
        return Evaluate(0); // drop; SeqStmt visitor flattens no-ops away
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    for (const auto &stmt : op->seq) {
      Stmt s = VisitStmt(stmt);
      // Drop consumed markers (rewritten to Evaluate(0)).
      if (IsZeroEvaluate(s))
        continue;
      seq.push_back(s);
    }
    if (seq.empty())
      return Evaluate(0);
    if (seq.size() == 1)
      return seq[0];
    return SeqStmt(std::move(seq));
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Thread-binding loops (blockIdx/threadIdx) are not iteration loops the
    // hint's `level` counts -- and they are materialized to thread_extent
    // AttrStmts later anyway, so skip them for a consistent level numbering.
    if (op->kind == ForKind::kThreadBinding)
      return StmtExprMutator::VisitStmt_(op);
    no_conflict_.stack.emplace_back();
    conflict_.stack.emplace_back();
    loop_chain_.push_back(GetRef<For>(op));
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    loop_chain_.pop_back();
    Array<Any> hints = std::move(no_conflict_.stack.back());
    Array<Any> conflict = std::move(conflict_.stack.back());
    no_conflict_.stack.pop_back();
    conflict_.stack.pop_back();
    if (hints.empty() && conflict.empty())
      return stmt;
    for (const Any &entry : conflict)
      hints.push_back(entry);
    For for_node = Downcast<For>(stmt);
    for_node.CopyOnWrite()->annotations.Set(kConflictHintAnnotKey, hints);
    return for_node;
  }
};

using namespace tirx::transform;
tvm::transform::Pass NormalizeConflictHints() {
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &) {
    return RewriteTilelangKernels(
        std::move(f), "NormalizeConflictHints",
        [](const TilelangKernelContext &context) {
          return ConflictHintNormalizer::Rewrite(context.root);
        },
        /*require_kernel=*/false);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizeConflictHints", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.NormalizeConflictHints",
                        NormalizeConflictHints);
}

} // namespace tl
} // namespace tvm
