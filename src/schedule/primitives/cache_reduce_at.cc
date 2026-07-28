#include "support/check.h"
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
 * \file cache_reduce_at.cc
 * \brief Implements the CacheReduceAt schedule primitive for tilelang.
 *
 * This is a combined primitive for reduction scheduling that:
 *
 * 1. Creates a compact accumulator buffer in a specified storage scope
 *    (like cache_write_at).
 * 2. Inserts a T.fill statement to initialize the accumulator with the
 *    identity value for the given reduction type (e.g., 0 for sum,
 *    -inf for max).
 * 3. Rewrites all write references to use the accumulator buffer.
 * 4. Inserts a T.copy to write back from the accumulator to the
 *    original buffer after the computation.
 *
 * This primitive is essential for reduction patterns where:
 * - An accumulator must be allocated in a fast memory scope
 * - The accumulator must be initialized before the reduction loop
 * - The final result must be written back after the loop
 *
 * Example usage (reduction c[i] = sum(a[i,k], axis=k)):
 *   sch.cache_reduce_at(outer_loop, block, 0, "local.fragment", 0.0)
 *   This creates: alloc c_frag → fill(c_frag, 0) → compute → copy(c_frag → c)
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "s_tir/schedule/transform.h"
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/utils.h>

#include "runtime/thread_storage_scope.h"
#include "s_tir/schedule/utils.h"
#include "s_tir/support/nd_int_set.h"
#include "utils.h"

#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tvm::s_tir;
using support::NDIntSet;

// ---------------------------------------------------------------------------
// Visitor that replaces accesses to `src_` with accesses to `dst_`,
// shifting indices by subtracting the per-axis region minimums.
// (Same approach as in cache_read_at.cc)
// ---------------------------------------------------------------------------
class CacheReduceBufferReplacer : public StmtExprMutator {
public:
  CacheReduceBufferReplacer(const Buffer &src, const Buffer &dst,
                            const ffi::Array<PrimExpr> &region_mins,
                            const std::vector<int> &kept_dims,
                            ffi::Map<SBlock, SBlock> *block_sref_reuse)
      : src_(src), dst_(dst), region_mins_(region_mins), kept_dims_(kept_dims),
        block_sref_reuse_(block_sref_reuse) {}

private:
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
    if (load->buffer.same_as(src_)) {
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
    if (store->buffer.same_as(src_)) {
      ObjectPtr<BufferStoreNode> n =
          ffi::make_object<BufferStoreNode>(*store.get());
      n->buffer = dst_;
      n->indices = SqueezedIndices(n->indices);
      // When the cache uses a wider dtype than the source (cache_dtype), the
      // stored value (computed in the source dtype) must be cast up to the
      // cache dtype.  Skip when the value already reads from dst (an in-place
      // accumulate already at dst dtype).
      if (!ValueUsesDstBuffer(n->value) && n->value.dtype() != dst_->dtype) {
        arith::Analyzer analyzer;
        n->value = analyzer.Simplify(Cast(dst_->dtype, n->value));
      }
      return BufferStore(n);
    }
    return store;
  }

  Stmt VisitStmt_(const SBlockNode *_block) final {
    SBlock old_block = ffi::GetRef<SBlock>(_block);
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(_block));
    tvm::ffi::ObjectPtr<SBlockNode> n =
        ffi::make_object<SBlockNode>(*block.get());
    n->reads = ReplaceBufferWithShift(n->reads);
    n->writes = ReplaceBufferWithShift(n->writes);
    SBlock new_block(n);
    block_sref_reuse_->Set(old_block, new_block);
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
};

// ---------------------------------------------------------------------------
// CacheReduceAt: main entry point
// ---------------------------------------------------------------------------
static void CacheReduceAt(s_tir::ScheduleState self, const StmtSRef &loop_sref,
                          const StmtSRef &block_sref, int write_buffer_index,
                          const ffi::String &storage_scope, double init_value,
                          bool write_back, const ffi::String &cache_dtype) {
  CheckLoopIsAncestorOfBlock(self, loop_sref, block_sref, "cache_reduce_at");
  // ---- Step 1: Obtain destination buffer and loop -------------------------
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);
  SBlock block_ref = ffi::GetRef<SBlock>(block);
  Buffer src = GetNthAccessBuffer(self, block_ref, write_buffer_index,
                                  s_tir::BufferIndexType::kWrite);

  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  // ---- Step 2: Gather inner-loop domains and block bindings ---------------
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = s_tir::GetBindings(realize);

  runtime::StorageScope scope = runtime::StorageScope::Create(storage_scope);
  ffi::Map<Var, arith::IntSet> var_dom =
      arith::AsIntSet(LoopDomainOfSRefTreePathSkipBlocks(
          ffi::GetRef<StmtSRef>(self->stmt2ref.at(block)->parent), loop_sref,
          scope));

  // ---- Step 3: Relax the buffer write region over the inner loops ---------
  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion &buffer_region : block->writes) {
    if (buffer_region->buffer.same_as(src)) {
      ffi::Array<arith::IntSet> relaxed =
          arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
      relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
    }
  }
  ICHECK(!relaxed_regions.empty()) << "ValueError: buffer " << src->name
                                   << " is not written in the specified block";

  NDIntSet unified = support::NDIntSetUnion(relaxed_regions);
  int ndim = static_cast<int>(unified.size());

  arith::Analyzer analyzer;
  ffi::Array<Range> cache_region;
  ffi::Array<PrimExpr> cache_shape;
  ffi::Array<PrimExpr> region_mins;
  cache_region.reserve(ndim);
  cache_shape.reserve(ndim);
  region_mins.reserve(ndim);

  for (int d = 0; d < ndim; ++d) {
    PrimExpr mn = analyzer.Simplify(unified[d].min());
    PrimExpr mx = analyzer.Simplify(unified[d].max());
    PrimExpr extent = analyzer.Simplify(mx - mn + 1);
    PrimExpr region_min = mn;
    PrimExpr cache_extent = NormalizeStaticShapeDim(extent);
    mn = MatchBufferIndexDType(src, d, mn);
    extent = MatchBufferIndexDType(src, d, extent);
    cache_region.push_back(Range::FromMinExtent(mn, extent));
    cache_shape.push_back(cache_extent);
    region_mins.push_back(region_min);
  }

  // ---- Step 4: Create the cache buffer ------------------------------------
  std::vector<int> kept_dims;
  ffi::Array<PrimExpr> squeezed_shape;
  for (int d = 0; d < ndim; ++d) {
    if (const auto *imm = cache_shape[d].as<IntImmNode>()) {
      if (imm->value == 1)
        continue;
    }
    kept_dims.push_back(d);
    squeezed_shape.push_back(cache_shape[d]);
  }
  if (kept_dims.empty()) {
    kept_dims.push_back(ndim - 1);
    squeezed_shape.push_back(cache_shape[ndim - 1]);
  }

  Buffer dst = s_tir::WithScope(src, storage_scope);
  {
    auto *w = dst.CopyOnWrite();
    w->shape = squeezed_shape;
    if (!cache_dtype.empty()) {
      w->dtype = DataType(ffi::StringToDLDataType(cache_dtype));
    }
    std::string scope_suffix = storage_scope;
    for (auto &c : scope_suffix) {
      if (c == '.')
        c = '_';
    }
    w->name = src->name + "_" + scope_suffix;
    w->strides = {};
  }

  // ---- Step 5: Build the T.fill call (initialization) ---------------------
  ffi::Array<Range> dst_ranges;
  dst_ranges.reserve(squeezed_shape.size());
  for (size_t i = 0; i < squeezed_shape.size(); ++i) {
    dst_ranges.push_back(
        Range::FromMinExtent(IntImm(DataType::Int(32), 0), squeezed_shape[i]));
  }

  PrimExpr fill_region_arg = MakeRegionCall(dst, dst_ranges, /*access_mask=*/2);
  PrimExpr fill_value = make_const(dst->dtype, init_value);
  Stmt fill_stmt = Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.fill"),
                                 {fill_region_arg, fill_value}));

  // ---- Step 6: Build optional T.copy call (write-back) --------------------
  Stmt copy_stmt;
  if (write_back) {
    PrimExpr cache_region_arg =
        MakeRegionCall(dst, dst_ranges, /*access_mask=*/1);
    PrimExpr orig_region_arg =
        MakeRegionCall(src, cache_region, /*access_mask=*/2);

    copy_stmt = Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.copy"),
                              {cache_region_arg, orig_region_arg}));
  }

  // ---- Step 7: Rewrite buffer references in the loop body -----------------
  ffi::Array<Stmt> subtrees = s_tir::AsArray(loop->body);
  ffi::Map<SBlock, SBlock> block_sref_reuse;
  CacheReduceBufferReplacer replacer(src, dst, region_mins, kept_dims,
                                     &block_sref_reuse);
  for (int i = 0; i < static_cast<int>(subtrees.size()); ++i) {
    Stmt old_stmt = subtrees[i];
    subtrees.Set(i, Stmt(nullptr));
    subtrees.Set(i, replacer(std::move(old_stmt)));
  }

  // ---- Step 8: Insert fill at start, optional copy at end -----------------
  // Order: [fill, ...original subtrees..., copy?]
  subtrees.insert(subtrees.begin(), fill_stmt);
  if (write_back) {
    subtrees.push_back(copy_stmt);
  }

  // Wrap in an opaque Block that owns the cache buffer allocation.
  SBlock alloc_block(
      /*iter_vars=*/{},
      /*reads=*/{},
      /*writes=*/{},
      /*name_hint=*/"",
      /*body=*/subtrees.size() == 1 ? subtrees[0] : SeqStmt(subtrees),
      /*init=*/std::nullopt,
      /*alloc_buffers=*/{dst});
  SBlockRealize alloc_realize(
      /*values=*/{},
      /*predicate=*/const_true(), alloc_block);

  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = std::move(alloc_realize);
  For new_loop(new_loop_node);

  // ---- Step 9: Replace in the scope root block ----------------------------
  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const SBlockNode *scope_block = TVM_SREF_TO_SBLOCK(scope_root_sref);

  SBlock new_scope_block = Downcast<SBlock>(
      LoopReplacer(loop, new_loop)(ffi::GetRef<SBlock>(scope_block)));

  if (!write_back) {
    // Intermediate reduction use-case: all consumers are rewritten to the
    // cache buffer, so the original tensor allocation can be removed from the
    // surrounding scope block.
    tvm::ffi::ObjectPtr<SBlockNode> scope_ptr =
        ffi::make_object<SBlockNode>(*new_scope_block.get());
    ffi::Array<Buffer> kept_alloc_buffers;
    kept_alloc_buffers.reserve(scope_ptr->alloc_buffers.size());
    for (const Buffer &buf : scope_ptr->alloc_buffers) {
      if (!buf.same_as(src)) {
        kept_alloc_buffers.push_back(buf);
      }
    }
    scope_ptr->alloc_buffers = std::move(kept_alloc_buffers);
    new_scope_block = SBlock(scope_ptr);
  }

  block_sref_reuse.Set(ffi::GetRef<SBlock>(scope_block), new_scope_block);
  self->Replace(scope_root_sref, new_scope_block, block_sref_reuse);
}

// ---------------------------------------------------------------------------
// FFI Registration
// ---------------------------------------------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.schedule.ScheduleCacheReduceAt",
                        [](s_tir::Schedule self, const s_tir::LoopRV &loop_rv,
                           const SBlockRV &block_rv, int write_buffer_index,
                           const ffi::String &storage_scope, double init_value,
                           bool write_back, const ffi::String &cache_dtype) {
                          CacheReduceAt(self->state(), self->GetSRef(loop_rv),
                                        self->GetSRef(block_rv),
                                        write_buffer_index, storage_scope,
                                        init_value, write_back, cache_dtype);
                        });
}

} // namespace tl
} // namespace tvm
