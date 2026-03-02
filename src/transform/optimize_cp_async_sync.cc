/*!
 * \file optimize_cp_async_sync.cc
 * \brief Optimize explicit cp.async synchronization intrinsics.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <optional>
#include <unordered_set>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace transform {

class CPAsyncSyncOptimizer : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> visited;
    visited.reserve(op->seq.size());
    for (const Stmt &stmt : op->seq) {
      visited.push_back(this->VisitStmt(stmt));
    }

    visited = MaybeMergeCommitsBeforeWait0(std::move(visited));

    visited = MaybeSplitEpilogueWait(std::move(visited));

    enum class UncommittedState { kUnknown, kZero, kNonZero };

    UncommittedState uncommitted_state = UncommittedState::kUnknown;
    std::optional<int> last_wait_n;
    bool last_wait_dynamic = false;
    std::optional<int> outstanding_committed_groups = 0;

    Array<Stmt> simplified;
    simplified.reserve(visited.size());
    for (size_t stmt_idx = 0; stmt_idx < visited.size(); ++stmt_idx) {
      const Stmt &stmt = visited[stmt_idx];
      Stmt current = stmt;
      if (const auto *loop = current.as<ForNode>()) {
        bool has_following_wait = false;
        for (size_t j = stmt_idx + 1; j < visited.size(); ++j) {
          AsyncIntrinSummary summary = SummarizeAsyncIntrinsics(visited[j]);
          if (summary.wait > 0) {
            has_following_wait = true;
            break;
          }
        }
        if (has_following_wait) {
          current = MaybeRelaxLoopFirstWait(Downcast<For>(current));
          current = MaybeRelaxLoopHeadWait(Downcast<For>(current),
                                           outstanding_committed_groups);
        }
      }

      ClassifiedStmt cls = ClassifySimpleAsyncStmt(current);
      switch (cls.kind) {
      case AsyncStmtKind::kCPAsync:
        uncommitted_state = UncommittedState::kNonZero;
        simplified.push_back(current);
        break;
      case AsyncStmtKind::kCommit: {
        if (uncommitted_state == UncommittedState::kZero) {
          // Proven redundant commit: no cp.async issued since the last commit.
          break;
        }
        bool commit_has_new_cpasync =
            (uncommitted_state == UncommittedState::kNonZero);
        simplified.push_back(current);
        uncommitted_state = UncommittedState::kZero;
        if (outstanding_committed_groups.has_value() &&
            commit_has_new_cpasync) {
          outstanding_committed_groups =
              outstanding_committed_groups.value() + 1;
        } else {
          outstanding_committed_groups = std::nullopt;
        }
        last_wait_n.reset();
        last_wait_dynamic = false;
        break;
      }
      case AsyncStmtKind::kWaitStatic:
        if (!last_wait_dynamic && last_wait_n.has_value() &&
            cls.wait_n >= *last_wait_n) {
          // A weaker (or equal) wait is redundant when no commit happened in
          // between.
          break;
        }
        simplified.push_back(current);
        last_wait_n = cls.wait_n;
        last_wait_dynamic = false;
        if (outstanding_committed_groups.has_value()) {
          outstanding_committed_groups =
              std::min(outstanding_committed_groups.value(), cls.wait_n);
        }
        break;
      case AsyncStmtKind::kWaitDynamic:
        simplified.push_back(current);
        last_wait_n.reset();
        last_wait_dynamic = true;
        outstanding_committed_groups = std::nullopt;
        break;
      case AsyncStmtKind::kOther:
        simplified.push_back(current);
        if (ContainsAsyncIntrinsics(current)) {
          AsyncIntrinSummary summary = SummarizeAsyncIntrinsics(current);
          if (summary.cp_async > 0 && summary.commit == 0 &&
              summary.wait == 0) {
            // Preserve pending cp.async state across cp.async-only wrappers
            // (e.g. prologue loops before a standalone commit).
            uncommitted_state = UncommittedState::kNonZero;
            break;
          }
          // Cross this unknown boundary conservatively.
          uncommitted_state = UncommittedState::kUnknown;
          last_wait_n.reset();
          last_wait_dynamic = false;
          outstanding_committed_groups = std::nullopt;
        }
        break;
      }
    }

    if (simplified.empty()) {
      return Evaluate(0);
    }
    if (simplified.size() == 1) {
      return simplified[0];
    }
    return SeqStmt(simplified);
  }

private:
  enum class AsyncStmtKind {
    kOther,
    kCPAsync,
    kCommit,
    kWaitStatic,
    kWaitDynamic
  };

  struct ClassifiedStmt {
    AsyncStmtKind kind{AsyncStmtKind::kOther};
    int wait_n{0};
  };

  struct AsyncIntrinSummary {
    int cp_async = 0;
    int commit = 0;
    int wait = 0;
  };

  Array<Stmt> MaybeMergeCommitsBeforeWait0(Array<Stmt> seq) const {
    // Merge adjacent cp.async commit groups when the program is still using a
    // full drain (wait_group(0)) as the synchronization point.
    //
    // Pattern (within a SeqStmt segment that ends at wait_group(0)):
    //   cp_async*; commit; cp_async*; commit; wait_group(0)
    // =>
    //   cp_async*;         cp_async*; commit; wait_group(0)
    //
    // This reduces the number of committed groups without weakening the drain,
    // and lets later wait relaxation derive the right retain count from the
    // new (smaller) group topology.
    const int n = static_cast<int>(seq.size());
    if (n < 4) {
      return seq;
    }

    auto is_direct_commit = [&](const Stmt &s) -> bool {
      const auto *eval = s.as<EvaluateNode>();
      if (!eval) {
        return false;
      }
      const auto *call = eval->value.as<CallNode>();
      return call && IsCommitCall(call);
    };

    Array<Stmt> out;
    out.reserve(n);
    int seg_start = 0;

    auto flush_segment = [&](int seg_end, bool merge_commits) {
      if (!merge_commits) {
        for (int j = seg_start; j < seg_end; ++j) {
          out.push_back(seq[j]);
        }
        return;
      }

      // We only want to merge commits in the immediately preceding cp.async
      // region, so we analyze a maximal suffix that contains only:
      //   - cp.async-only statements (possibly wrapped in loops/attrs), and
      //   - standalone commit_group statements.
      // This avoids blocking on unrelated statements (e.g. barriers) that may
      // appear earlier in the segment.
      auto is_cp_async_only_stmt = [&](const Stmt &s) -> bool {
        ClassifiedStmt cls = ClassifySimpleAsyncStmt(s);
        if (cls.kind == AsyncStmtKind::kCPAsync) {
          return true;
        }
        if (!ContainsAsyncIntrinsics(s)) {
          return false;
        }
        AsyncIntrinSummary summary = SummarizeAsyncIntrinsics(s);
        return (summary.cp_async > 0 && summary.commit == 0 &&
                summary.wait == 0);
      };
      auto is_direct_commit_stmt = [&](const Stmt &s) -> bool {
        ClassifiedStmt cls = ClassifySimpleAsyncStmt(s);
        return cls.kind == AsyncStmtKind::kCommit && is_direct_commit(s);
      };

      int merge_start = seg_end;
      bool saw_cp_async = false;
      for (int j = seg_end - 1; j >= seg_start; --j) {
        if (is_direct_commit_stmt(seq[j])) {
          merge_start = j;
          continue;
        }
        if (is_cp_async_only_stmt(seq[j])) {
          saw_cp_async = true;
          merge_start = j;
          continue;
        }
        // Stop at the first non-async statement.
        break;
      }

      std::vector<int> commit_indices;
      commit_indices.reserve(4);
      bool has_complex_async = false;
      for (int j = merge_start; j < seg_end; ++j) {
        if (is_direct_commit_stmt(seq[j])) {
          commit_indices.push_back(j);
          continue;
        }
        if (is_cp_async_only_stmt(seq[j])) {
          continue;
        }
        // Shouldn't happen given how merge_start is determined, but be safe.
        has_complex_async = true;
        break;
      }

      if (has_complex_async || !saw_cp_async || commit_indices.size() != 2) {
        for (int j = seg_start; j < seg_end; ++j) {
          out.push_back(seq[j]);
        }
        return;
      }

      // Emit prefix unchanged.
      for (int j = seg_start; j < merge_start; ++j) {
        out.push_back(seq[j]);
      }
      // Drop the first commit in the mergeable suffix so both copy regions are
      // committed as a single group by the second commit.
      int dropped_commit_idx = commit_indices[0];
      for (int j = merge_start; j < seg_end; ++j) {
        if (j == dropped_commit_idx) {
          continue;
        }
        out.push_back(seq[j]);
      }
    };

    for (int i = 0; i < n; ++i) {
      ClassifiedStmt cls = ClassifySimpleAsyncStmt(seq[i]);
      if (cls.kind == AsyncStmtKind::kWaitStatic && cls.wait_n == 0) {
        flush_segment(/*seg_end=*/i, /*merge_commits=*/true);
        out.push_back(seq[i]);
        seg_start = i + 1;
        continue;
      }
      if (cls.kind == AsyncStmtKind::kWaitStatic ||
          cls.kind == AsyncStmtKind::kWaitDynamic) {
        // For non-(wait0) waits, we don't attempt to merge commits because it
        // can weaken synchronization unless we also adjust wait counts.
        flush_segment(/*seg_end=*/i, /*merge_commits=*/false);
        out.push_back(seq[i]);
        seg_start = i + 1;
        continue;
      }
    }
    flush_segment(/*seg_end=*/n, /*merge_commits=*/false);
    return out;
  }

  Array<Stmt> MaybeSplitEpilogueWait(Array<Stmt> seq) const {
    // Schedule cp.async drains in a software-pipeline epilogue more formally.
    //
    // In TileLang software pipelining, async global->shared copies are
    // committed in the steady-state loop and consumed in one or more epilogue
    // "consumer phases". A conservative lowering often emits:
    //
    //   for ...:  (contains cp.async + commit)
    //   ptx_wait_group(0)              # full drain
    //   tvm_storage_sync("shared")
    //   ... consumer phase 0 ...
    //   tvm_storage_sync("shared")
    //   ... consumer phase 1 ...
    //
    // Draining all groups immediately after the loop can destroy overlap
    // between the work in phase 0 and the last in-flight committed group(s)
    // that are only needed in phase 1. We improve overlap by:
    //   - relaxing the post-loop wait_group(0) to keep some groups in flight,
    //   - inserting a final wait_group(0) right before the shared barrier that
    //     starts the next consumer phase.
    //
    // Unlike the earlier heuristic that looked for global stores, we identify
    // consumer phases by detecting reads from buffers written by ptx_cp_async.
    const int n = static_cast<int>(seq.size());
    if (n < 6) {
      return seq;
    }

    auto is_shared_storage_sync = [&](const Stmt &s) -> bool {
      const auto *eval = s.as<EvaluateNode>();
      if (!eval) {
        return false;
      }
      const auto *call = eval->value.as<CallNode>();
      if (!call || !call->op.same_as(builtin::tvm_storage_sync())) {
        return false;
      }
      if (call->args.size() != 1) {
        return false;
      }
      const auto *scope = call->args[0].as<StringImmNode>();
      return scope && scope->value == "shared";
    };

    auto make_wait_stmt = [&](int wait_n) -> Stmt {
      return Evaluate(Call(DataType::Handle(), builtin::ptx_wait_group(),
                           {IntImm(DataType::Int(32), wait_n)}));
    };

    auto access_ptr_buffer_var = [&](const PrimExpr &ptr) -> Optional<Var> {
      // Support both `tl.access_ptr(BufferLoad, extent, rw_mask)` (frontend)
      // and `tvm_access_ptr(ptype, data, offset, extent, rw_mask)` (lowered).
      const auto *call = ptr.as<CallNode>();
      if (!call) {
        return Optional<Var>();
      }
      if (call->op.same_as(tl::access_ptr())) {
        if (call->args.size() != 3) {
          return Optional<Var>();
        }
        const auto *base_load = call->args[0].as<BufferLoadNode>();
        if (!base_load) {
          return Optional<Var>();
        }
        return base_load->buffer->data;
      }
      if (call->op.same_as(builtin::tvm_access_ptr())) {
        if (call->args.size() != 5) {
          return Optional<Var>();
        }
        if (call->args[1].as<VarNode>()) {
          return Downcast<Var>(call->args[1]);
        }
      }
      return Optional<Var>();
    };

    auto collect_cp_async_dst_buffers = [&](const Stmt &s) {
      std::unordered_set<const VarNode *> vars;
      PostOrderVisit(s, [&](const ObjectRef &node) {
        const auto *call = node.as<CallNode>();
        if (!call || !IsCPAsyncCall(call)) {
          return;
        }
        if (call->args.empty()) {
          return;
        }
        if (Optional<Var> buf_var = access_ptr_buffer_var(call->args[0])) {
          vars.insert(buf_var.value().get());
        }
      });
      return vars;
    };

    auto contains_async_smem_read =
        [&](const Stmt &s,
            const std::unordered_set<const VarNode *> &async_smem_vars)
        -> bool {
      if (async_smem_vars.empty()) {
        return false;
      }
      bool found = false;
      PostOrderVisit(s, [&](const ObjectRef &node) {
        if (found) {
          return;
        }
        const auto *load = node.as<BufferLoadNode>();
        if (!load) {
          return;
        }
        if (async_smem_vars.count(load->buffer->data.get()) == 0) {
          return;
        }
        // Only treat shared memory reads as cp.async consumers.
        const String &scope = load->buffer.scope();
        if (scope == "shared" || scope == "shared.dyn") {
          found = true;
        }
      });
      return found;
    };

    for (int i = 1; i + 1 < n; ++i) {
      ClassifiedStmt cls = ClassifySimpleAsyncStmt(seq[i]);
      if (cls.kind != AsyncStmtKind::kWaitStatic || cls.wait_n != 0) {
        continue;
      }

      const auto *loop = seq[i - 1].as<ForNode>();
      if (!loop) {
        continue;
      }
      For loop_ref = Downcast<For>(seq[i - 1]);
      AsyncIntrinSummary loop_summary = SummarizeAsyncIntrinsics(loop_ref);
      if (loop_summary.cp_async <= 0 || loop_summary.commit <= 0) {
        continue;
      }

      if (!is_shared_storage_sync(seq[i + 1])) {
        continue;
      }

      int retain = PipelinedRetainGroups(loop_ref);
      if (retain <= 0) {
        continue;
      }

      // Avoid relaxing wait_group(0) into a no-op when we cannot prove there is
      // at least (retain + 1) committed groups that can be drained here.
      //
      // When loop extent is a compile-time constant, we can conservatively
      // lower-bound the total number of commit_group calls executed. Otherwise,
      // fall back to the per-iteration count (syntactic).
      int64_t min_commits = static_cast<int64_t>(loop_summary.commit);
      if (const auto *ext = loop_ref->extent.as<IntImmNode>()) {
        min_commits *= static_cast<int64_t>(ext->value);
      }
      if (min_commits < static_cast<int64_t>(retain + 1)) {
        continue;
      }

      std::unordered_set<const VarNode *> async_smem_vars =
          collect_cp_async_dst_buffers(loop_ref);
      if (async_smem_vars.empty()) {
        continue;
      }

      // Identify at least two "consumer phases" after the post-loop wait by
      // scanning barrier-separated regions for reads from async-written shared
      // buffers.
      int insert_before_sync = -1;
      int segment_start = i + 2; // after the first sync
      int prev_sync = i + 1;
      int found_phases = 0;
      for (int j = segment_start; j <= n; ++j) {
        bool end_segment = (j == n) || is_shared_storage_sync(seq[j]);
        if (!end_segment) {
          continue;
        }
        bool consumes = false;
        for (int k = segment_start; k < j; ++k) {
          if (contains_async_smem_read(seq[k], async_smem_vars)) {
            consumes = true;
            break;
          }
        }
        if (consumes) {
          ++found_phases;
          if (found_phases == 2) {
            // The barrier immediately before this segment starts the next
            // consumer phase.
            if (prev_sync > i + 1) {
              insert_before_sync = prev_sync;
            }
            break;
          }
        }
        // Start next segment after this sync (if any).
        if (j < n) {
          prev_sync = j;
          segment_start = j + 1;
        }
      }
      if (insert_before_sync == -1) {
        continue;
      }

      Array<Stmt> out;
      out.reserve(n + 1);
      for (int j = 0; j < n; ++j) {
        if (j == i) {
          bool changed = false;
          out.push_back(
              RewriteWaitStaticInSimpleWrapper(seq[j], retain, &changed));
          // If rewrite failed (non-simple wrapper), keep original.
          if (!changed) {
            out.Set(out.size() - 1, seq[j]);
          }
          continue;
        }
        if (j == insert_before_sync) {
          // Drain all groups before the next consumer phase.
          bool rewrote_existing = false;
          if (!out.empty()) {
            bool changed_prev = false;
            Stmt prev =
                RewriteWaitStaticInSimpleWrapper(out.back(), 0, &changed_prev);
            if (changed_prev) {
              out.Set(out.size() - 1, prev);
              rewrote_existing = true;
            }
          }
          if (!rewrote_existing) {
            out.push_back(make_wait_stmt(0));
          }
        }
        out.push_back(seq[j]);
      }
      return out;
    }

    return seq;
  }

  Stmt MakeStaticWaitStmtLike(const Stmt &stmt, int new_wait_n) const {
    const auto *eval = stmt.as<EvaluateNode>();
    if (!eval) {
      return stmt;
    }
    const auto *call = eval->value.as<CallNode>();
    if (!call || !IsWaitCall(call)) {
      return stmt;
    }

    DataType wait_dtype =
        call->args.empty() ? DataType::Int(32) : call->args[0].dtype();
    Array<PrimExpr> args{make_const(wait_dtype, new_wait_n)};
    return Evaluate(
        Call(call->dtype, call->op, args, call->annotations, call->span));
  }

  Stmt RewriteWaitStaticInSimpleWrapper(const Stmt &stmt, int new_wait_n,
                                        bool *changed) const {
    ClassifiedStmt cls = ClassifySimpleAsyncStmt(stmt);
    if (cls.kind != AsyncStmtKind::kWaitStatic) {
      return stmt;
    }

    if (const auto *eval = stmt.as<EvaluateNode>()) {
      const auto *call = eval->value.as<CallNode>();
      if (call && IsWaitCall(call)) {
        *changed = true;
        return MakeStaticWaitStmtLike(stmt, new_wait_n);
      }
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      Stmt new_body =
          RewriteWaitStaticInSimpleWrapper(let->body, new_wait_n, changed);
      if (*changed) {
        return LetStmt(let->var, let->value, new_body, let->span);
      }
      return stmt;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      Stmt new_body =
          RewriteWaitStaticInSimpleWrapper(attr->body, new_wait_n, changed);
      if (*changed) {
        return AttrStmt(attr->node, attr->attr_key, attr->value, new_body,
                        attr->span);
      }
      return stmt;
    }
    if (const auto *iff = stmt.as<IfThenElseNode>()) {
      if (!iff->else_case.defined()) {
        Stmt then_case = RewriteWaitStaticInSimpleWrapper(iff->then_case,
                                                          new_wait_n, changed);
        if (*changed) {
          return IfThenElse(iff->condition, then_case, Stmt(), iff->span);
        }
      }
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        Stmt inner =
            RewriteWaitStaticInSimpleWrapper(seq->seq[0], new_wait_n, changed);
        if (*changed) {
          return SeqStmt({inner});
        }
      }
      return stmt;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      Stmt inner =
          RewriteWaitStaticInSimpleWrapper(block->body, new_wait_n, changed);
      if (*changed) {
        Block new_block = Downcast<Block>(stmt);
        BlockNode *n = new_block.CopyOnWrite();
        n->body = inner;
        return new_block;
      }
      return stmt;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        Stmt inner = RewriteWaitStaticInSimpleWrapper(realize->block->body,
                                                      new_wait_n, changed);
        if (*changed) {
          Block block = realize->block;
          BlockNode *n = block.CopyOnWrite();
          n->body = inner;
          return BlockRealize(realize->iter_values, realize->predicate, block,
                              realize->span);
        }
      }
      return stmt;
    }

    return stmt;
  }

  Stmt MaybeRelaxLoopFirstWait(const For &loop) const {
    if (!loop.defined() || loop->kind != ForKind::kSerial) {
      return loop;
    }
    const auto *seq = loop->body.as<SeqStmtNode>();
    if (!seq) {
      return loop;
    }

    Array<Stmt> body = seq->seq;
    int cp_async_before_wait = 0;
    int commit_before_wait = 0;
    bool changed = false;
    for (int i = 0, n = static_cast<int>(body.size()); i < n; ++i) {
      ClassifiedStmt cls = ClassifySimpleAsyncStmt(body[i]);
      if (cls.kind == AsyncStmtKind::kCPAsync) {
        ++cp_async_before_wait;
        continue;
      }
      if (cls.kind == AsyncStmtKind::kCommit) {
        ++commit_before_wait;
        continue;
      }
      if (cls.kind == AsyncStmtKind::kWaitStatic) {
        if (cls.wait_n == 0 && cp_async_before_wait > 0 &&
            commit_before_wait > 0) {
          // In a pipelined loop, each iteration may commit multiple cp.async
          // groups (e.g. separate A/B shared-memory copies). For a K-stage
          // software pipeline, a good default is to keep (K - 1) iterations'
          // worth of committed groups in flight.
          //
          // `commit_before_wait` is a syntactic approximation of the number of
          // committed groups produced per iteration before the first wait.
          int new_wait_n = 1;
          if (loop->annotations.Get("tl_pipelined_num_stages")) {
            int per_iter_groups = std::max(1, commit_before_wait);
            int stage_retain = PipelinedRetainGroups(loop);
            new_wait_n = stage_retain * per_iter_groups;
            // `cp.async.wait_group` takes a small immediate operand. Clamp to a
            // conservative range to avoid generating illegal SASS/PTX.
            new_wait_n = std::max(0, std::min(new_wait_n, 7));
          }
          bool changed_wait = false;
          body.Set(i, RewriteWaitStaticInSimpleWrapper(body[i], new_wait_n,
                                                       &changed_wait));
          changed = changed || changed_wait;
        }
        break;
      }
      if (cls.kind == AsyncStmtKind::kWaitDynamic) {
        break;
      }
      if (cls.kind == AsyncStmtKind::kOther &&
          ContainsAsyncIntrinsics(body[i])) {
        AsyncIntrinSummary summary = SummarizeAsyncIntrinsics(body[i]);
        if (summary.cp_async > 0 && summary.commit == 0 && summary.wait == 0) {
          cp_async_before_wait += summary.cp_async;
          continue;
        }
        if (summary.cp_async == 0 && summary.commit > 0 && summary.wait == 0) {
          commit_before_wait += summary.commit;
          continue;
        }
        break;
      }
    }

    if (!changed) {
      return loop;
    }
    For new_loop = loop;
    ForNode *n = new_loop.CopyOnWrite();
    n->body = body.size() == 1 ? body[0] : SeqStmt(body);
    return new_loop;
  }

  Stmt MaybeRelaxLoopHeadWait(const For &loop,
                              const std::optional<int> &pre_outstanding) const {
    if (!loop.defined() || loop->kind != ForKind::kSerial) {
      return loop;
    }
    int retain = PipelinedRetainGroups(loop);
    if (retain <= 0) {
      return loop;
    }
    if (!pre_outstanding.has_value() ||
        pre_outstanding.value() < (retain + 1)) {
      // Without enough committed groups before the loop, relaxing a leading
      // wait_group(0) could turn it into a no-op and violate correctness.
      return loop;
    }

    const auto *seq = loop->body.as<SeqStmtNode>();
    if (!seq || seq->seq.empty()) {
      return loop;
    }

    // Only consider loops that start with a static wait_group(0).
    ClassifiedStmt first = ClassifySimpleAsyncStmt(seq->seq[0]);
    if (first.kind != AsyncStmtKind::kWaitStatic || first.wait_n != 0) {
      return loop;
    }

    // Require a prefetch pattern inside the loop (cp.async + commit after the
    // leading wait). Otherwise relaxing does not help and can be risky.
    int cp_async_after = 0;
    int commit_after = 0;
    for (int i = 1, n = static_cast<int>(seq->seq.size()); i < n; ++i) {
      AsyncIntrinSummary summary = SummarizeAsyncIntrinsics(seq->seq[i]);
      cp_async_after += summary.cp_async;
      commit_after += summary.commit;
      // No need to track waits here.
    }
    if (cp_async_after <= 0 || commit_after <= 0) {
      return loop;
    }

    Array<Stmt> body = seq->seq;
    bool changed = false;
    body.Set(0, RewriteWaitStaticInSimpleWrapper(body[0], retain, &changed));
    if (!changed) {
      return loop;
    }

    For new_loop = loop;
    ForNode *n = new_loop.CopyOnWrite();
    n->body = body.size() == 1 ? body[0] : SeqStmt(body);
    return new_loop;
  }

  int PipelinedRetainGroups(const For &loop) const {
    // Keep (num_stages - 1) committed groups in flight when possible.
    // This metadata is preserved by PipelinePlanning under the dedicated
    // annotation key "tl_pipelined_num_stages".
    int retain = 1;
    if (!loop.defined()) {
      return retain;
    }
    if (auto anno = loop->annotations.Get("tl_pipelined_num_stages")) {
      int num_stages = -1;
      if (const auto *imm = anno.value().as<IntImmNode>()) {
        num_stages = static_cast<int>(imm->value);
      }
      if (num_stages >= 1) {
        retain = std::max(0, num_stages - 1);
      }
    }
    return retain;
  }

  bool IsCPAsyncCall(const CallNode *call) const {
    return call && (call->op.same_as(builtin::ptx_cp_async()) ||
                    call->op.same_as(tl::ptx_cp_async()));
  }

  bool IsCommitCall(const CallNode *call) const {
    return call && call->op.same_as(builtin::ptx_commit_group());
  }

  bool IsWaitCall(const CallNode *call) const {
    return call && call->op.same_as(builtin::ptx_wait_group());
  }

  bool ContainsAsyncIntrinsics(const Stmt &stmt) const {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (found) {
        return;
      }
      const auto *call = node.as<CallNode>();
      if (!call) {
        return;
      }
      if (IsCPAsyncCall(call) || IsCommitCall(call) || IsWaitCall(call)) {
        found = true;
      }
    });
    return found;
  }

  AsyncIntrinSummary SummarizeAsyncIntrinsics(const Stmt &stmt) const {
    AsyncIntrinSummary summary;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      const auto *call = node.as<CallNode>();
      if (!call) {
        return;
      }
      if (IsCPAsyncCall(call)) {
        ++summary.cp_async;
      } else if (IsCommitCall(call)) {
        ++summary.commit;
      } else if (IsWaitCall(call)) {
        ++summary.wait;
      }
    });
    return summary;
  }

  ClassifiedStmt ClassifySimpleAsyncStmt(const Stmt &stmt) const {
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return ClassifySimpleAsyncStmt(let->body);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return ClassifySimpleAsyncStmt(attr->body);
    }
    // Do not treat IfThenElse as a "simple wrapper": conditional execution can
    // invalidate cp.async bookkeeping and make wait relaxation unsafe when the
    // prefetch path is skipped at runtime (e.g. blocksparse kernels).
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return ClassifySimpleAsyncStmt(seq->seq[0]);
      }
      return {};
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return ClassifySimpleAsyncStmt(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return ClassifySimpleAsyncStmt(realize->block->body);
      }
      return {};
    }

    const auto *eval = stmt.as<EvaluateNode>();
    if (!eval) {
      return {};
    }
    const auto *call = eval->value.as<CallNode>();
    if (!call) {
      return {};
    }
    if (IsCPAsyncCall(call)) {
      return {AsyncStmtKind::kCPAsync, 0};
    }
    if (IsCommitCall(call)) {
      return {AsyncStmtKind::kCommit, 0};
    }
    if (IsWaitCall(call)) {
      if (!call->args.empty()) {
        if (const auto *imm = call->args[0].as<IntImmNode>()) {
          return {AsyncStmtKind::kWaitStatic, static_cast<int>(imm->value)};
        }
      }
      return {AsyncStmtKind::kWaitDynamic, 0};
    }
    return {};
  }
};

tvm::transform::Pass OptimizeCPAsyncSync() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = CPAsyncSyncOptimizer()(std::move(fptr->body));
    return f;
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0,
                                                 "tl.OptimizeCPAsyncSync", {});
}

// Backward-compatible alias.
tvm::transform::Pass SimplifyCPAsyncSync() { return OptimizeCPAsyncSync(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.OptimizeCPAsyncSync",
                        OptimizeCPAsyncSync);
  refl::GlobalDef().def("tl.transform.SimplifyCPAsyncSync",
                        SimplifyCPAsyncSync);
}

} // namespace transform
} // namespace tl
} // namespace tvm
