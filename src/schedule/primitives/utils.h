/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file primitives/utils.h
 * \brief Shared utilities for tilelang schedule primitives.
 */

#ifndef TVM_TL_SCHEDULE_PRIMITIVES_UTILS_H_
#define TVM_TL_SCHEDULE_PRIMITIVES_UTILS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <limits>
#include <unordered_set>
#include <vector>

#include "runtime/thread_storage_scope.h"
#include "s_tir/schedule/error.h"
#include "s_tir/schedule/utils.h"

#include <tvm/s_tir/analysis.h>

#include "../../op/region.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tvm::s_tir;
using support::NDIntSet;

static inline void CheckLoopIsAncestorOfBlock(const ScheduleState &self,
                                              const StmtSRef &loop_sref,
                                              const StmtSRef &block_sref,
                                              const std::string &primitive) {
  for (const StmtSRefNode *sref = block_sref->parent; sref != nullptr;
       sref = sref->parent) {
    if (sref == loop_sref.get()) {
      return;
    }
  }
  throw LoopPositionError(
      self->mod, ffi::GetRef<For>(TVM_SREF_TO_FOR(loop_sref)),
      ffi::GetRef<SBlock>(TVM_SREF_TO_SBLOCK(block_sref)), primitive);
}

// ---------------------------------------------------------------------------
// MakeRegionCall: construct a tl.region() Call that encodes a BufferRegion as
// a PrimExpr for passing as an argument to tl.tileop.{copy,fill,reduce}.
// ---------------------------------------------------------------------------
static inline PrimExpr MatchBufferIndexDType(const Buffer &buf, size_t dim,
                                             PrimExpr expr) {
  DataType index_dtype = buf->shape[dim].dtype();
  if (expr.dtype() == index_dtype) {
    return expr;
  }
  if (const auto *imm = expr.as<IntImmNode>()) {
    return IntImm(index_dtype, imm->value);
  }
  return cast(index_dtype, expr);
}

static inline PrimExpr RebaseBufferIndex(PrimExpr index, PrimExpr min) {
  arith::Analyzer analyzer;
  return analyzer.Simplify(index - min);
}

static inline PrimExpr NormalizeStaticShapeDim(PrimExpr extent) {
  const auto *imm = extent.as<IntImmNode>();
  if (imm == nullptr || imm->dtype == DataType::Int(32)) {
    return extent;
  }
  ICHECK_GE(imm->value, 0) << "Buffer shape dimensions must be non-negative";
  ICHECK_LE(imm->value, std::numeric_limits<int32_t>::max())
      << "Buffer shape dimension exceeds the int32 range: " << imm->value;
  return IntImm(DataType::Int(32), imm->value);
}

static inline PrimExpr MakeRegionCall(const Buffer &buf,
                                      const ffi::Array<Range> &ranges,
                                      int access_mask) {
  ICHECK_EQ(buf->shape.size(), ranges.size());
  ffi::Array<PrimExpr> args;
  ffi::Array<PrimExpr> min_indices;
  ffi::Array<PrimExpr> extents;
  for (size_t i = 0; i < ranges.size(); ++i) {
    min_indices.push_back(MatchBufferIndexDType(buf, i, ranges[i]->min));
    extents.push_back(MatchBufferIndexDType(buf, i, ranges[i]->extent));
  }
  args.push_back(BufferLoad(buf, min_indices));
  args.push_back(IntImm(DataType::Int(32), access_mask));
  for (const PrimExpr &extent : extents) {
    args.push_back(extent);
  }
  return Call(DataType::Handle(), RegionOp::Get(), args);
}

// ---------------------------------------------------------------------------
// LoopDomainOfSRefTreePathSkipBlocks: variant of TVM's loop-domain collector
// that keeps traversing through intermediate Block/SBlockRealize wrappers.
//
// TileLang primitives often wrap loop bodies in opaque allocation blocks.
// Subsequent primitives still need to relax loop vars under the target loop,
// but TVM's default helper stops at the first non-For ancestor.  That makes
// later cache/reduce primitives think there are no inner loops, collapsing
// multi-element tiles to scalars.
// ---------------------------------------------------------------------------
static inline ffi::Map<Var, Range> LoopDomainOfSRefTreePathSkipBlocks(
    const StmtSRef &low_inclusive,
    const ffi::Optional<StmtSRef> &high_exclusive,
    const runtime::StorageScope &extra_relax_scope) {
  ffi::Map<Var, Range> result;
  const StmtSRefNode *p = low_inclusive.get();
  const StmtSRefNode *limit =
      static_cast<const StmtSRefNode *>(high_exclusive.get());
  for (; p != limit; p = p->parent) {
    if (const ForNode *loop = p->StmtAs<ForNode>()) {
      result.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
  }
  if (extra_relax_scope.rank != runtime::StorageRank::kGlobal) {
    for (; p; p = p->parent) {
      if (const ForNode *loop = p->StmtAs<ForNode>()) {
        if (loop->kind == ForKind::kThreadBinding) {
          const ffi::String &thread_tag =
              loop->thread_binding.value()->thread_tag;
          if (CanRelaxStorageUnderThread(
                  extra_relax_scope,
                  runtime::ThreadScope::Create(thread_tag))) {
            result.Set(loop->loop_var,
                       Range::FromMinExtent(loop->min, loop->extent));
          }
        }
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// LoopReplacer: rewrite a specific ForNode in the AST with a new one.
// ---------------------------------------------------------------------------
class LoopReplacer : public StmtMutator {
public:
  LoopReplacer(const ForNode *old_loop, For new_loop)
      : old_loop_(old_loop), new_loop_(std::move(new_loop)), found_(false) {}

  Stmt VisitStmt(const Stmt &stmt) final {
    return found_ ? stmt : StmtMutator::VisitStmt(stmt);
  }
  Stmt VisitStmt_(const ForNode *loop) final {
    if (loop == old_loop_) {
      found_ = true;
      return new_loop_;
    }
    return StmtMutator::VisitStmt_(loop);
  }

private:
  const ForNode *old_loop_;
  For new_loop_;
  bool found_;
};

// ---------------------------------------------------------------------------
// HasTileRegionCallOnBuffer: detect whether the given statement already
// contains tl.tileop.region calls that reference `buf`.  Used by cache
// primitives to skip dimension squeezing when region extents cannot be
// rewritten consistently.
// ---------------------------------------------------------------------------
static inline bool HasTileRegionCallOnBuffer(const Stmt &body,
                                             const Buffer &buf) {
  bool found = false;
  PostOrderVisit(body, [&found, &buf](const ObjectRef &obj) {
    if (found)
      return;
    const auto *call = obj.as<CallNode>();
    if (call == nullptr)
      return;
    const auto *op = call->op.as<OpNode>();
    if (op == nullptr || op->name != "tl.tileop.region" || call->args.empty()) {
      return;
    }
    const auto *load = call->args[0].as<BufferLoadNode>();
    if (load != nullptr && load->buffer.same_as(buf)) {
      found = true;
    }
  });
  return found;
}

// ---------------------------------------------------------------------------
// CacheBufferReplacer: rewrite all accesses to `src` with accesses to `dst`,
// shifting indices by subtracting per-axis region minimums.  Shared by
// CacheReadAt and CacheWriteAt.
// ---------------------------------------------------------------------------
class CacheBufferReplacer : public StmtExprMutator {
public:
  // kept_dims: indices of original dimensions that are kept (not squeezed).
  // target_blocks: the set of consumer SBlockNodes whose src accesses should
  //   be rewritten to dst.  An empty set means "rewrite everything under the
  //   loop" (legacy caller behaviour).  A non-empty set scopes the rewrite
  //   to those blocks only, allowing one CacheReadAt call to redirect
  //   multiple independent consumer blocks to a shared cache.
  CacheBufferReplacer(
      const Buffer &src, const Buffer &dst,
      const ffi::Array<PrimExpr> &region_mins,
      const std::vector<int> &kept_dims,
      ffi::Map<SBlock, SBlock> *block_sref_reuse,
      const std::unordered_set<const SBlockNode *> &target_blocks)
      : src_(src), dst_(dst), region_mins_(region_mins), kept_dims_(kept_dims),
        block_sref_reuse_(block_sref_reuse), target_blocks_(target_blocks),
        block_only_(!target_blocks.empty()), target_scope_depth_(0) {}

private:
  bool ShouldRewriteAccess() const {
    return !block_only_ || target_scope_depth_ > 0;
  }

  bool ValueUsesDstBuffer(const PrimExpr &expr) const {
    bool found = false;
    PostOrderVisit(expr, [this, &found](const ObjectRef &obj) {
      if (found) {
        return;
      }
      if (const auto *load = obj.as<BufferLoadNode>()) {
        found = load->buffer.same_as(dst_);
      }
    });
    return found;
  }

  // Build squeezed indices: for each kept dim, compute (original_idx - min).
  ffi::Array<PrimExpr> SqueezedIndices(const ffi::Array<PrimExpr> &indices) {
    ffi::Array<PrimExpr> new_indices;
    for (size_t i = 0; i < kept_dims_.size(); ++i) {
      int d = kept_dims_[i];
      new_indices.push_back(RebaseBufferIndex(indices[d], region_mins_[d]));
    }
    return new_indices;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *_load) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
    if (ShouldRewriteAccess() && load->buffer.same_as(src_)) {
      ObjectPtr<BufferLoadNode> n =
          ffi::make_object<BufferLoadNode>(*load.get());
      n->buffer = dst_;
      n->indices = SqueezedIndices(n->indices);
      return BufferLoad(n);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *_store) final {
    BufferStore store =
        Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
    if (ShouldRewriteAccess() && store->buffer.same_as(src_)) {
      ObjectPtr<BufferStoreNode> n =
          ffi::make_object<BufferStoreNode>(*store.get());
      n->buffer = dst_;
      n->indices = SqueezedIndices(n->indices);
      if (!ValueUsesDstBuffer(n->value) && n->value.dtype() != dst_->dtype) {
        arith::Analyzer analyzer;
        n->value = analyzer.Simplify(Cast(dst_->dtype, n->value));
      }
      return BufferStore(n);
    }
    return store;
  }

  Stmt VisitStmt_(const SBlockNode *_block) final {
    bool entered_target_scope = false;
    if (block_only_ && target_blocks_.count(_block) != 0) {
      ++target_scope_depth_;
      entered_target_scope = true;
    }
    SBlock old_block = ffi::GetRef<SBlock>(_block);
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(_block));
    ObjectPtr<SBlockNode> n = ffi::make_object<SBlockNode>(*block.get());
    if (ShouldRewriteAccess()) {
      n->reads = ReplaceBufferWithShift(n->reads);
      n->writes = ReplaceBufferWithShift(n->writes);
    }
    SBlock new_block(n);
    block_sref_reuse_->Set(old_block, new_block);
    if (entered_target_scope) {
      --target_scope_depth_;
    }
    return new_block;
  }

  ffi::Array<BufferRegion>
  ReplaceBufferWithShift(const ffi::Array<BufferRegion> &regions) {
    ffi::Array<BufferRegion> result;
    for (const auto &region : regions) {
      if (region->buffer.same_as(src_)) {
        ffi::Array<Range> new_ranges;
        for (size_t i = 0; i < kept_dims_.size(); ++i) {
          int d = kept_dims_[i];
          PrimExpr min =
              RebaseBufferIndex(region->region[d]->min, region_mins_[d]);
          new_ranges.push_back(
              Range::FromMinExtent(min, region->region[d]->extent));
        }
        result.push_back(BufferRegion(dst_, new_ranges));
      } else {
        result.push_back(region);
      }
    }
    return result;
  }

  const Buffer &src_;
  const Buffer &dst_;
  const ffi::Array<PrimExpr> &region_mins_;
  const std::vector<int> &kept_dims_;
  ffi::Map<SBlock, SBlock> *block_sref_reuse_;
  std::unordered_set<const SBlockNode *> target_blocks_;
  bool block_only_;
  int target_scope_depth_;
};

// ---------------------------------------------------------------------------
// ComputeRelaxedRegion: compute the access region of `buf` inside `block`,
// relaxed over all loops strictly inside `loop_sref`.  Shared by GemmAt and
// CopyAt to describe the operand and result tiles.
// ---------------------------------------------------------------------------
static inline ffi::Array<Range>
ComputeRelaxedRegion(ScheduleState self, const StmtSRef &loop_sref,
                     const StmtSRef &block_sref, const Buffer &buf,
                     BufferIndexType buffer_type,
                     const runtime::StorageScope &extra_relax_scope) {
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = GetBindings(realize);

  ffi::Map<Var, arith::IntSet> var_dom =
      arith::AsIntSet(LoopDomainOfSRefTreePathSkipBlocks(
          ffi::GetRef<StmtSRef>(self->stmt2ref.at(block)->parent), loop_sref,
          extra_relax_scope));

  const auto &regions =
      (buffer_type == BufferIndexType::kRead) ? block->reads : block->writes;

  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion &buffer_region : regions) {
    if (!buffer_region->buffer.same_as(buf)) {
      continue;
    }
    ffi::Array<arith::IntSet> relaxed =
        arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
    relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
  }
  ICHECK(!relaxed_regions.empty()) << "ValueError: buffer " << buf->name
                                   << " is not accessed in the specified block";

  NDIntSet unified = support::NDIntSetUnion(relaxed_regions);
  int ndim = static_cast<int>(unified.size());

  arith::Analyzer analyzer;
  ffi::Array<Range> result;
  result.reserve(ndim);
  for (int d = 0; d < ndim; ++d) {
    PrimExpr mn = analyzer.Simplify(unified[d].min());
    PrimExpr mx = analyzer.Simplify(unified[d].max());
    PrimExpr extent = analyzer.Simplify(mx - mn + 1);
    result.push_back(Range::FromMinExtent(mn, extent));
  }
  return result;
}

// ---------------------------------------------------------------------------
// ComputeNestReplacer: walk a SeqStmt and replace the subtree that contains
// a specific target SBlockNode with a given replacement statement.  Shared by
// GemmAt and CopyAt to replace a block's compute nest with a tile-level op.
// ---------------------------------------------------------------------------
namespace detail {
static inline bool ContainsTargetBlock(const Stmt &stmt,
                                       const SBlockNode *target) {
  bool found = false;
  PostOrderVisit(stmt, [&found, target](const ObjectRef &obj) {
    if (found) {
      return;
    }
    if (const auto *realize = obj.as<SBlockRealizeNode>()) {
      if (realize->block.get() == target) {
        found = true;
      }
    }
  });
  return found;
}
} // namespace detail

class ComputeNestReplacer : public StmtMutator {
public:
  ComputeNestReplacer(const SBlockNode *target, Stmt replacement)
      : target_(target), replacement_(std::move(replacement)),
        replaced_(false) {}

  Stmt VisitStmt(const Stmt &stmt) final {
    if (replaced_ || !stmt.defined()) {
      return stmt;
    }
    return StmtMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    ffi::Array<Stmt> new_seq;
    new_seq.reserve(op->seq.size());
    for (const Stmt &stmt : op->seq) {
      if (!replaced_ && detail::ContainsTargetBlock(stmt, target_)) {
        Stmt new_stmt = StmtMutator::VisitStmt(stmt);
        if (!replaced_ && detail::ContainsTargetBlock(new_stmt, target_)) {
          new_seq.push_back(replacement_);
          replaced_ = true;
        } else {
          new_seq.push_back(new_stmt);
        }
      } else {
        new_seq.push_back(stmt);
      }
    }
    return SeqStmt(new_seq);
  }

  bool replaced() const { return replaced_; }

private:
  const SBlockNode *target_;
  Stmt replacement_;
  bool replaced_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_SCHEDULE_PRIMITIVES_UTILS_H_
