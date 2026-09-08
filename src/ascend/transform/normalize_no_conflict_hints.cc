/*!
 * \file normalize_no_conflict_hints.cc
 * \brief Consume `T.assume_no_conflict` markers into For annotations.
 *
 * The frontend emits a statement-form marker
 *   Evaluate(Call(tl.assume_no_conflict, a, b, level, cross, group))
 * inside a loop body, where `a`/`b` are each a `tl.region` Call (a concrete
 * region) or a bare buffer's data Var (the whole buffer, matched by storage
 * key), `level` is an IntImm loop index counted from the outermost enclosing
 * loop (`0` = outermost) or `-1` for "all enclosing loops", `cross` is an
 * IntImm (-1=any / 1=cross-iter / 0=same-iter), and `group` is a StringImm tag
 * ("" = none).
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
 *   - normalizes each hint into an `[a, b, IntImm(cross_code)]` triple appended
 *     to those Fors' `annotations["no_conflict"]` (operands kept as Buffer or
 *     BufferRegion).
 *
 * AutoSchedule's RegionsMayConflict reads the annotation off
 * `loop_stack.back()` and short-circuits matching dependencies to "disjoint".
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
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

// Op registered in src/op/schedule_hint.cc.
static constexpr const char *kHintOpName = "tl.assume_no_conflict";
static constexpr const char *kForAnnotKey = "no_conflict";

class NoConflictHintNormalizer : public StmtExprMutator {
public:
  static Stmt Rewrite(const Stmt &body) {
    NoConflictHintNormalizer m;
    Stmt rewritten = m(body);
    ICHECK(m.stack_.empty()) << "unbalanced loop stack";
    ICHECK(m.pending_group_.empty())
        << "assume_no_conflict group '" << m.pending_group_.begin()->first
        << "' appears only once; each group must appear exactly twice";
    return rewritten;
  }

private:
  // Per enclosing For (outermost first), the triples to attach on exit.
  std::vector<Array<Any>> stack_;
  // The enclosing For identities, parallel to stack_ (outermost first). Used to
  // compute a group's LCA as the longest common prefix of the two halves.
  std::vector<const ForNode *> loop_chain_;
  // First-seen half of each group tag, awaiting its second (and only) partner:
  // its operand, its `level`, its `cross`, and the chain of loops enclosing it.
  struct PendingHalf {
    Any operand;
    int level;
    int cross;
    std::vector<const ForNode *> chain;
  };
  std::unordered_map<std::string, PendingHalf> pending_group_;
  // Group tags already paired, to reject a third occurrence.
  std::unordered_set<std::string> completed_group_;

  // A resolved triple [a, b, IntImm(cross_code)]; a/b are Buffer or
  // BufferRegion, kept verbatim.
  static Array<Any> Triple(const Any &a, const Any &b, int cross) {
    Array<Any> t;
    t.push_back(a);
    t.push_back(b);
    t.push_back(IntImm(DataType::Int(32), cross));
    return t;
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
  static int CommonPrefix(const std::vector<const ForNode *> &x,
                          const std::vector<const ForNode *> &y) {
    int n = static_cast<int>(std::min(x.size(), y.size()));
    int i = 0;
    while (i < n && x[i] == y[i])
      ++i;
    return i;
  }

  // Attach a triple at loop index `level` (from outermost; -1 = all enclosing
  // loops). `bound` caps the addressable depth (for a group, the LCA prefix
  // length shared by both halves; otherwise the full stack).
  void Attach(int level, int bound, const Any &a, const Any &b, int cross) {
    if (level < 0) {
      for (int i = 0; i < bound; ++i)
        stack_[i].push_back(Triple(a, b, cross));
      return;
    }
    ICHECK(level < bound) << "assume_no_conflict level " << level
                          << " is out of range (only " << bound
                          << " enclosing loop(s))";
    stack_[level].push_back(Triple(a, b, cross));
  }

  // Consume one marker Call. Returns nothing; drops the statement.
  void ConsumeMarker(const CallNode *call) {
    const auto &args = call->args;
    ICHECK_EQ(args.size(), 5u) << "assume_no_conflict marker expects 5 args";
    Any a = DecodeOperand(args[0]);
    Any b = DecodeOperand(args[1]);
    int level = static_cast<int>(Downcast<IntImm>(args[2])->value);
    int cross = static_cast<int>(Downcast<IntImm>(args[3])->value);
    std::string group = Downcast<StringImm>(args[4])->value;
    int depth = static_cast<int>(stack_.size());

    ICHECK(depth > 0) << "assume_no_conflict must appear inside a loop body";

    if (group.empty()) {
      Attach(level, depth, a, b, cross);
      return;
    }
    // group pairing: a tag must appear exactly twice, symmetrically (which half
    // is seen first is irrelevant). Both must carry the same `level` and
    // `cross`. The first records its region, level, cross and enclosing loop
    // chain; the second attaches the combined hint, bounding `level` to the LCA
    // (longest common prefix of the two chains). A third occurrence is an
    // error.
    ICHECK(!completed_group_.count(group))
        << "assume_no_conflict group '" << group
        << "' appears more than twice; each group must appear exactly twice";
    auto it = pending_group_.find(group);
    if (it == pending_group_.end()) {
      pending_group_.emplace(group, PendingHalf{a, level, cross, loop_chain_});
    } else {
      ICHECK(it->second.level == level)
          << "assume_no_conflict group '" << group
          << "' has mismatched level between its two half-declarations ("
          << it->second.level << " vs " << level << ")";
      ICHECK(it->second.cross == cross)
          << "assume_no_conflict group '" << group
          << "' has mismatched cross between its two half-declarations ("
          << it->second.cross << " vs " << cross << ")";
      int lca = CommonPrefix(it->second.chain, loop_chain_);
      Attach(level, lca, it->second.operand, a, cross);
      pending_group_.erase(it);
      completed_group_.insert(group);
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
      if (call->op.same_as(Op::Get(kHintOpName))) {
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
    stack_.emplace_back();
    loop_chain_.push_back(op);
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    loop_chain_.pop_back();
    Array<Any> triples = std::move(stack_.back());
    stack_.pop_back();
    if (triples.empty())
      return stmt;
    For for_node = Downcast<For>(stmt);
    for_node.CopyOnWrite()->annotations.Set(kForAnnotKey, triples);
    return for_node;
  }
};

using namespace tirx::transform;
tvm::transform::Pass NormalizeNoConflictHints() {
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &) {
    return RewriteTilelangKernels(
        std::move(f), "NormalizeNoConflictHints",
        [](const TilelangKernelContext &context) {
          SBlock root = context.root;
          root.CopyOnWrite()->body =
              NoConflictHintNormalizer::Rewrite(root->body);
          return root;
        },
        /*require_kernel=*/false);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizeNoConflictHints", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.NormalizeNoConflictHints",
                        NormalizeNoConflictHints);
}

} // namespace tl
} // namespace tvm
