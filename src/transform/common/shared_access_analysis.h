#ifndef TVM_TL_TRANSFORM_COMMON_SHARED_ACCESS_ANALYSIS_H_
#define TVM_TL_TRANSFORM_COMMON_SHARED_ACCESS_ANALYSIS_H_

#include "./constr_visitor.h"
#include "runtime/thread_storage_scope.h"

#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

namespace tvm::tl::shared_access_analysis {

enum AccessType : uint8_t { kRead, kWrite, kSync, kAlloc, kReadAcquire };

struct AccessEntry {
  ffi::Array<tir::IterVar> threads;
  ffi::Array<PrimExpr> buffer_indices;
  ConstrSet cset;
  ffi::Array<Range> buffer_ranges;
  tir::Var buffer = NullValue<tir::Var>();
  tir::Buffer buffer_name;
  DataType dtype;
  ffi::Array<arith::IntSet> touched;
  AccessType type;
  runtime::StorageScope scope;
  bool is_pointer_access = false;
  bool is_async_copy = false;
  bool is_atomic = false;
};

struct StmtEntry {
  const Object *stmt{};
  std::vector<AccessEntry> access;
};

struct SequenceSummaryResult {
  std::vector<AccessEntry> exposed_accesses;
  std::vector<const Object *> sync_before_stmts;
};

inline bool PointerAccessIsDisjoint(const AccessEntry &lhs,
                                    const AccessEntry &rhs) {
  if (lhs.touched.size() != 1 || rhs.touched.size() != 1) {
    return false;
  }
  ConstrSet prev_cset{lhs.cset};
  ConstrSet curr_cset{rhs.cset};
  arith::Analyzer analyzer;

  struct ThreadVarInfo {
    const char *name_prev;
    const char *name_curr;
  } thread_vars[] = {{"tx1", "tx2"}, {"ty1", "ty2"}, {"tz1", "tz2"}};

  PrimExpr lhs_min = analyzer.Simplify(lhs.touched[0].min());
  PrimExpr lhs_max = analyzer.Simplify(lhs.touched[0].max());
  PrimExpr rhs_min = analyzer.Simplify(rhs.touched[0].min());
  PrimExpr rhs_max = analyzer.Simplify(rhs.touched[0].max());

  for (unsigned idx = 0; idx != 3; ++idx) {
    auto &info = thread_vars[idx];
    tir::Var old_prev_var = lhs.threads[lhs.threads.size() + idx - 3]->var;
    tir::Var old_curr_var = rhs.threads[rhs.threads.size() + idx - 3]->var;
    tir::Var prev_var(info.name_prev, old_prev_var.dtype());
    tir::Var curr_var(info.name_curr, old_curr_var.dtype());
    lhs_min = tir::Substitute(lhs_min, {{old_prev_var, prev_var}});
    lhs_max = tir::Substitute(lhs_max, {{old_prev_var, prev_var}});
    prev_cset = prev_cset.Substitute({{old_prev_var, prev_var}});
    rhs_min = tir::Substitute(rhs_min, {{old_curr_var, curr_var}});
    rhs_max = tir::Substitute(rhs_max, {{old_curr_var, curr_var}});
    curr_cset = curr_cset.Substitute({{old_curr_var, curr_var}});
  }

  prev_cset.Populate(analyzer);
  curr_cset.Populate(analyzer);

  if (analyzer.CanProve(lhs_max < rhs_min,
                        arith::ProofStrength::kSymbolicBound)) {
    return true;
  }
  if (analyzer.CanProve(rhs_max < lhs_min,
                        arith::ProofStrength::kSymbolicBound)) {
    return true;
  }
  return false;
}

inline bool FindConflict(const AccessEntry &prev, const AccessEntry &curr,
                         const tir::ForNode *loop) {
  if (prev.type == kWrite && curr.type == kWrite && prev.is_async_copy &&
      curr.is_async_copy) {
    return false;
  }

  if (!prev.buffer.same_as(curr.buffer)) {
    return false;
  }

  if (prev.is_atomic && curr.is_atomic) {
    return false;
  }

  if (prev.buffer_indices.size() != curr.buffer_indices.size()) {
    return true;
  }

  if (prev.is_pointer_access || curr.is_pointer_access) {
    if (prev.is_pointer_access && curr.is_pointer_access &&
        PointerAccessIsDisjoint(prev, curr)) {
      return false;
    }
    return true;
  }

  ffi::Map<tir::Var, PrimExpr> loop_shift_sub;
  if (loop != nullptr) {
    PrimExpr step = tir::make_const(loop->loop_var.dtype(), 1);
    loop_shift_sub.Set(loop->loop_var, loop->loop_var + step);
  }

  bool has_same_index = true;
  for (size_t i = 0; i < prev.buffer_indices.size(); ++i) {
    const auto &prev_indice = prev.buffer_indices[i];
    PrimExpr curr_indice = curr.buffer_indices[i];
    if (loop != nullptr) {
      curr_indice = tir::Substitute(curr_indice, loop_shift_sub);
    }
    if (!tir::ExprDeepEqual()(prev_indice, curr_indice)) {
      has_same_index = false;
      break;
    }
  }

  if (has_same_index) {
    PrimExpr prev_constr = prev.cset.ToConjunction();
    PrimExpr curr_constr = curr.cset.ToConjunction();

    arith::Analyzer analyzer;
    for (const auto &iv : prev.threads) {
      if (iv->dom.defined()) {
        analyzer.Bind(iv->var, iv->dom);
      }
    }
    if (loop != nullptr) {
      PrimExpr adjusted_extent =
          loop->extent - tir::make_const(loop->extent.dtype(), 1);
      analyzer.Bind(loop->loop_var,
                    Range::FromMinExtent(loop->min, adjusted_extent));
    }

    bool prev_implies_curr = analyzer.z3_prover.CanProve(
        tir::Or(tir::Not(prev_constr), curr_constr));
    bool curr_implies_prev = analyzer.z3_prover.CanProve(
        tir::Or(tir::Not(curr_constr), prev_constr));

    return !(prev_implies_curr && curr_implies_prev);
  }

  bool range_is_overlap = true;

  for (size_t i = 0; i < prev.buffer_indices.size(); ++i) {
    auto prev_dtype = prev.dtype;
    auto curr_dtype = curr.dtype;

    const auto &prev_indice = prev.buffer_indices[i];
    PrimExpr curr_indice = curr.buffer_indices[i];
    if (loop != nullptr) {
      curr_indice = tir::Substitute(curr_indice, loop_shift_sub);
    }

    PrimExpr prev_indice_bytes = prev_indice * prev_dtype.bytes();
    PrimExpr curr_indice_bytes = curr_indice * curr_dtype.bytes();

    ConstrSet prev_cset{prev.cset};
    ConstrSet curr_cset{curr.cset};
    arith::Analyzer analyzer;

    if (loop != nullptr) {
      PrimExpr adjusted_extent =
          loop->extent - tir::make_const(loop->extent.dtype(), 1);
      analyzer.Bind(loop->loop_var,
                    Range::FromMinExtent(loop->min, adjusted_extent));
    }

    bool same_access_type = (prev.type == kWrite && curr.type == kWrite) ||
                            (prev.type == kRead && curr.type == kRead);

    PrimExpr thread_condition = Bool(false);
    ffi::Map<tir::Var, PrimExpr> prev_sub, curr_sub;

    const char *thread_names[] = {"tx", "ty", "tz"};
    for (unsigned idx = 0; idx != 3; ++idx) {
      tir::Var old_prev_var = prev.threads[prev.threads.size() + idx - 3]->var;
      tir::Var old_curr_var = curr.threads[curr.threads.size() + idx - 3]->var;

      if (same_access_type) {
        tir::Var shared_var(thread_names[idx], old_prev_var.dtype());
        prev_sub.Set(old_prev_var, shared_var);
        curr_sub.Set(old_curr_var, shared_var);
      } else {
        tir::Var prev_var(std::string(thread_names[idx]) + "1",
                          old_prev_var.dtype());
        tir::Var curr_var(std::string(thread_names[idx]) + "2",
                          old_curr_var.dtype());
        thread_condition =
            tir::Or(thread_condition, tir::NE(prev_var, curr_var));
        prev_sub.Set(old_prev_var, prev_var);
        curr_sub.Set(old_curr_var, curr_var);
      }
    }
    if (!same_access_type) {
      analyzer.EnterConstraint(thread_condition);
    }
    prev_cset.Substitute(prev_sub).Populate(analyzer);
    curr_cset.Substitute(curr_sub).Populate(analyzer);
    bool provably_disjoint = false;

    prev_indice_bytes =
        analyzer.Simplify(tir::Substitute(prev_indice_bytes, prev_sub));
    curr_indice_bytes =
        analyzer.Simplify(tir::Substitute(curr_indice_bytes, curr_sub));

    if (const auto *prev_ramp = prev_indice_bytes.as<tir::RampNode>()) {
      DataType prev_index_dtype = prev_ramp->base.dtype();
      tir::Var prev_idx("prev_idx", prev_index_dtype);
      analyzer.Bind(prev_idx, Range::FromMinExtent(0, prev_ramp->lanes));
      prev_indice_bytes = prev_ramp->base + prev_idx * prev_ramp->stride;
    }

    if (const auto *curr_ramp = curr_indice_bytes.as<tir::RampNode>()) {
      DataType curr_index_dtype = curr_ramp->base.dtype();
      tir::Var curr_idx("curr_idx", curr_index_dtype);
      analyzer.Bind(curr_idx, Range::FromMinExtent(0, curr_ramp->lanes));
      curr_indice_bytes = curr_ramp->base + curr_idx * curr_ramp->stride;
    }

    if (prev_indice_bytes.dtype().is_scalar() &&
        curr_indice_bytes.dtype().is_scalar()) {
      if (prev_indice_bytes.dtype() != curr_indice_bytes.dtype()) {
        if (prev_indice_bytes.dtype().bits() <
            curr_indice_bytes.dtype().bits()) {
          prev_indice_bytes =
              tir::Cast(curr_indice_bytes.dtype(), prev_indice_bytes);
        } else {
          curr_indice_bytes =
              tir::Cast(prev_indice_bytes.dtype(), curr_indice_bytes);
        }
      }
      ICHECK(prev_indice_bytes.dtype() == curr_indice_bytes.dtype());
      provably_disjoint =
          analyzer.CanProve(tir::NE(prev_indice_bytes, curr_indice_bytes));
    } else {
      try {
        auto prev_min = analyzer.Simplify(tir::Substitute(
            prev.touched[i].min() * prev_dtype.bytes(), prev_sub));
        auto prev_max = analyzer.Simplify(tir::Substitute(
            prev.touched[i].max() * prev_dtype.bytes(), prev_sub));
        auto curr_min = analyzer.Simplify(tir::Substitute(
            curr.touched[i].min() * curr_dtype.bytes(), curr_sub));
        auto curr_max = analyzer.Simplify(tir::Substitute(
            curr.touched[i].max() * curr_dtype.bytes(), curr_sub));
        provably_disjoint = analyzer.CanProve(analyzer.Simplify(
            tir::Or(prev_min > curr_max, curr_min > prev_max)));
      } catch (const std::exception &) {
        auto prev_bound = analyzer.const_int_bound(prev_indice_bytes);
        auto curr_bound = analyzer.const_int_bound(curr_indice_bytes);
        if (prev_bound.defined() && curr_bound.defined()) {
          if ((prev_bound->min_value) > (curr_bound->max_value) ||
              (curr_bound->min_value) > (prev_bound->max_value)) {
            range_is_overlap = false;
            break;
          }
        }
      }
    }

    if (provably_disjoint) {
      range_is_overlap = false;
      break;
    }
  }

  return range_is_overlap;
}

inline bool FindConflict(const std::vector<AccessEntry> &prev,
                         const AccessEntry &curr, const tir::ForNode *loop) {
  for (const AccessEntry &x : prev) {
    if (FindConflict(x, curr, loop)) {
      return true;
    }
  }
  return false;
}

inline SequenceSummaryResult SummarizeAccessSequence(
    std::vector<StmtEntry> seq, const tir::ForNode *loop,
    const runtime::StorageScope &sync_scope,
    const ffi::Array<tir::IterVar> &env_threads, const ConstrSet &current_cset,
    const std::unordered_set<const Object *> &initial_syncs = {},
    bool coalesce_dynamic_shared_buffers = false) {
  if (coalesce_dynamic_shared_buffers) {
    tir::Var shared_dyn_buf;
    for (StmtEntry &entry : seq) {
      for (AccessEntry &access : entry.access) {
        if (access.scope.rank == runtime::StorageRank::kShared &&
            access.scope.tag == ".dyn" && access.buffer.defined()) {
          if (!shared_dyn_buf.defined()) {
            shared_dyn_buf = access.buffer;
          } else {
            access.buffer = shared_dyn_buf;
          }
        }
      }
    }
  }

  std::vector<AccessEntry> reads;
  std::vector<AccessEntry> writes;
  std::unordered_set<const Object *> syncs_inserted = initial_syncs;
  std::vector<const Object *> new_syncs;

  auto insert_sync = [&](const Object *obj) {
    if (syncs_inserted.count(obj) != 0) {
      return;
    }
    syncs_inserted.insert(obj);
    new_syncs.push_back(obj);
  };

  for (size_t i = 0; i < seq.size(); ++i) {
    const StmtEntry &s = seq[i];
    bool sync_before_stmt = (syncs_inserted.count(s.stmt) != 0);

    if (sync_before_stmt) {
      reads.clear();
      writes.clear();
    }

    for (const AccessEntry &acc : s.access) {
      if (acc.type == kRead) {
        if (FindConflict(writes, acc, nullptr)) {
          sync_before_stmt = true;
          break;
        }
      } else if (acc.type == kWrite) {
        if (FindConflict(reads, acc, nullptr) ||
            FindConflict(writes, acc, nullptr)) {
          sync_before_stmt = true;
          break;
        }
      } else if (acc.type == kSync) {
        reads.clear();
        writes.clear();
      }
    }

    if (sync_before_stmt) {
      reads.clear();
      writes.clear();
    }

    for (const AccessEntry &acc : s.access) {
      if (acc.type == kRead) {
        reads.push_back(acc);
      } else if (acc.type == kWrite) {
        writes.push_back(acc);
      } else if (acc.type == kSync) {
        reads.clear();
        writes.clear();
      }
    }

    if (sync_before_stmt) {
      insert_sync(s.stmt);
    }
  }

  if (loop != nullptr) {
    bool has_read_in_scope = false;
    for (const StmtEntry &s : seq) {
      for (const AccessEntry &acc : s.access) {
        if (acc.type == kRead && acc.scope == sync_scope) {
          has_read_in_scope = true;
          break;
        }
      }
      if (has_read_in_scope) {
        break;
      }
    }

    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry &s = seq[i];
      if (syncs_inserted.count(s.stmt) != 0) {
        break;
      }
      if (reads.empty() && writes.empty()) {
        break;
      }
      bool need_loop_sync = false;
      for (const AccessEntry &acc : s.access) {
        if (acc.type == kRead) {
          if (FindConflict(writes, acc, loop)) {
            need_loop_sync = true;
            break;
          }
        } else if (acc.type == kWrite) {
          if (FindConflict(reads, acc, loop) ||
              FindConflict(writes, acc, loop)) {
            need_loop_sync = true;
            break;
          }
        } else if (acc.type == kSync) {
          reads.clear();
          writes.clear();
        }
      }
      if (need_loop_sync) {
        if (!has_read_in_scope) {
          insert_sync(loop);
        } else {
          insert_sync(s.stmt);
        }
        break;
      }
    }
  }

  int sync_count = 0;
  std::vector<AccessEntry> head, tail;
  AccessEntry esync{.cset = current_cset};
  esync.threads = env_threads;
  esync.type = kSync;
  esync.scope = sync_scope;

  for (const StmtEntry &s : seq) {
    if (syncs_inserted.count(s.stmt) != 0) {
      if (sync_count != 0) {
        tail.clear();
      } else {
        head.push_back(esync);
      }
      ++sync_count;
    }
    for (const AccessEntry &acc : s.access) {
      if (acc.type == kSync) {
        if (sync_count != 0) {
          tail.clear();
        } else {
          head.push_back(esync);
        }
        ++sync_count;
      } else {
        if (sync_count != 0) {
          tail.push_back(acc);
        } else {
          head.push_back(acc);
        }
      }
    }
  }
  head.insert(head.end(), tail.begin(), tail.end());

  return SequenceSummaryResult{std::move(head), std::move(new_syncs)};
}

} // namespace tvm::tl::shared_access_analysis

#endif // TVM_TL_TRANSFORM_COMMON_SHARED_ACCESS_ANALYSIS_H_
