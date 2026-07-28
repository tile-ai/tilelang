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
 * \file cache_read_at.cc
 * \brief Implements the CacheReadAt schedule primitive for
 *        tilelang.  Given a consumer block, a buffer index, a loop where the
 *        cache should reside, and a target storage scope, this primitive:
 *
 *        1. Analyzes the buffer access region within one iteration of the
 *           specified loop (by relaxing over all inner loop variables).
 *        2. Creates a compact cache buffer whose shape equals the per-iteration
 *           access extents.
 *        3. Emits a T.copy (tl.tileop.copy) statement that transfers the
 *           accessed region from the original buffer into the cache.
 *        4. Rewrites all references to the original buffer inside the loop body
 *           so they use the cache buffer with shifted indices.
 *        5. Allocates the cache buffer locally inside the loop via an opaque
 *           Block wrapper, keeping it scoped to one iteration.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

// TVM internal headers (included via ${TVM_SOURCE}/src in the include path)
#include "runtime/thread_storage_scope.h"
#include "s_tir/schedule/transform.h"
#include "s_tir/schedule/utils.h"
#include "s_tir/support/nd_int_set.h"
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/utils.h>

// TileLang headers
#include "../../op/region.h"
#include "../../transform/layout_reducer.h"
#include "utils.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tvm::s_tir;
using support::NDIntSet;

// Build an in-place square transform:
//   for ...:
//     dst[idx] = dst[idx] * dst[idx]
static Stmt BuildSquareInplaceStmt(const Buffer &dst,
                                   const ffi::Array<PrimExpr> &shape) {
  int ndim = static_cast<int>(shape.size());
  std::vector<Var> iters;
  iters.reserve(ndim);

  ffi::Array<PrimExpr> indices;
  indices.reserve(ndim);
  for (int d = 0; d < ndim; ++d) {
    Var it("ax" + std::to_string(d), DataType::Int(32));
    iters.push_back(it);
    indices.push_back(it);
  }

  PrimExpr val = BufferLoad(dst, indices);
  Stmt body = BufferStore(dst, val * val, indices);
  for (int d = ndim - 1; d >= 0; --d) {
    body = For(iters[d],
               /*min=*/IntImm(DataType::Int(32), 0),
               /*extent=*/shape[d], ForKind::kSerial, body);
  }
  return body;
}

// ---------------------------------------------------------------------------
// CacheReadAt:  main entry point
// ---------------------------------------------------------------------------
static void CacheReadAt(s_tir::ScheduleState self, const StmtSRef &loop_sref,
                        const StmtSRef &block_sref, int read_buffer_index,
                        const ffi::String &storage_scope,
                        const ffi::String &transform,
                        const ffi::String &cache_dtype,
                        bool disable_tma = false,
                        const ffi::Array<StmtSRef> &consumer_block_srefs = {}) {
  // ---- Step 1: Obtain source buffer and loop --------------------------------
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);
  SBlock block_ref = ffi::GetRef<SBlock>(block);
  Buffer src = GetNthAccessBuffer(self, block_ref, read_buffer_index,
                                  s_tir::BufferIndexType::kRead);

  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  // Primary + extra consumer block srefs whose src accesses should all be
  // rewritten to use the shared cache.  The extras allow one CacheReadAt
  // call to create a single cache buffer shared across multiple sibling
  // blocks under the same loop (e.g. sum_x and sum_x_sq reductions that
  // both read x).
  std::vector<StmtSRef> all_consumer_srefs;
  all_consumer_srefs.push_back(block_sref);
  for (const StmtSRef &extra_sref : consumer_block_srefs) {
    const SBlockNode *extra_block = TVM_SREF_TO_SBLOCK(extra_sref);
    if (extra_block != block) {
      all_consumer_srefs.push_back(extra_sref);
    }
  }

  // ---- Step 2: Gather inner-loop domains and block bindings -----------------
  runtime::StorageScope scope = runtime::StorageScope::Create(storage_scope);

  for (const StmtSRef &cb_sref : all_consumer_srefs) {
    CheckLoopIsAncestorOfBlock(self, loop_sref, cb_sref, "cache_read_at");
  }

  // ---- Step 3: Relax the buffer read region over the inner loops ------------
  // Union the relaxed read regions of EVERY consumer block so the shared
  // cache covers all accessed slices of src.
  std::vector<NDIntSet> relaxed_regions;
  std::vector<const SBlockNode *> all_consumer_blocks;
  for (const StmtSRef &cb_sref : all_consumer_srefs) {
    const SBlockNode *cb = TVM_SREF_TO_SBLOCK(cb_sref);
    all_consumer_blocks.push_back(cb);
    SBlockRealize cb_realize = GetSBlockRealize(self, cb_sref);
    ffi::Map<Var, PrimExpr> cb_bindings = s_tir::GetBindings(cb_realize);
    ffi::Map<Var, arith::IntSet> cb_var_dom =
        arith::AsIntSet(LoopDomainOfSRefTreePathSkipBlocks(
            /*low_inclusive=*/ffi::GetRef<StmtSRef>(cb_sref->parent),
            /*high_exclusive=*/loop_sref,
            /*extra_relax_scope=*/scope));
    for (const BufferRegion &buffer_region : cb->reads) {
      if (buffer_region->buffer.same_as(src)) {
        ffi::Array<arith::IntSet> relaxed = arith::EvalSet(
            Substitute(buffer_region->region, cb_bindings), cb_var_dom);
        relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
      }
    }
  }
  ICHECK(!relaxed_regions.empty()) << "ValueError: buffer " << src->name
                                   << " is not read in the specified block";

  // Union all relaxed regions
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
    if (!extent->IsInstance<IntImmNode>()) {
      // Sliding windows whose size depends on the block index (e.g. the
      // `2 - fused % 2` windows of nearest upsampling) have no constant
      // extent, but fragment/local cache buffers require one.  Reading a
      // few extra elements is harmless, so pad the window to its constant
      // upper bound and clamp the start so the copy stays in bounds; the
      // consumer rebase uses the same clamped min, and
      // min(mn, dim - max_ext) + max_ext >= mn + true_ext keeps the true
      // window covered.
      arith::ConstIntBound bound = analyzer.const_int_bound(extent);
      if (bound->max_value != arith::ConstIntBound::kPosInf &&
          bound->max_value >= 1) {
        PrimExpr padded = IntImm(extent.dtype(), bound->max_value);
        PrimExpr dim = src->shape[d];
        if (dim.dtype() != mn.dtype()) {
          dim = cast(mn.dtype(), dim);
        }
        mn = analyzer.Simplify(
            max(min(mn, dim - padded), make_zero(mn.dtype())));
        extent = padded;
      }
    }
    cache_region.push_back(Range::FromMinExtent(mn, extent));
    cache_shape.push_back(extent);
    region_mins.push_back(mn);
  }

  // ---- Step 4: Create the cache buffer --------------------------------------
  std::vector<int> kept_dims;
  ffi::Array<PrimExpr> squeezed_shape;
  bool keep_full_rank = HasTileRegionCallOnBuffer(loop->body, src);
  if (keep_full_rank) {
    kept_dims.reserve(ndim);
    squeezed_shape.reserve(ndim);
    for (int d = 0; d < ndim; ++d) {
      kept_dims.push_back(d);
      squeezed_shape.push_back(cache_shape[d]);
    }
  } else {
    // Default behavior: squeeze unit extents for compact cache allocation.
    for (int d = 0; d < ndim; ++d) {
      if (const auto *imm = cache_shape[d].as<IntImmNode>()) {
        if (imm->value == 1)
          continue;
      }
      kept_dims.push_back(d);
      squeezed_shape.push_back(cache_shape[d]);
    }
    // If all dims are 1, keep the last one to avoid 0-d buffer.
    if (kept_dims.empty()) {
      kept_dims.push_back(ndim - 1);
      squeezed_shape.push_back(cache_shape[ndim - 1]);
    }
  }

  Buffer dst = s_tir::WithScope(src, storage_scope);
  {
    auto *w = dst.CopyOnWrite();
    w->shape = squeezed_shape;
    if (!cache_dtype.empty()) {
      w->dtype = DataType(ffi::StringToDLDataType(cache_dtype));
    }
    // Produce a readable name: e.g. a_shared_dyn  or  a_local_fragment
    std::string scope_suffix = storage_scope;
    // Replace '.' with '_' for valid identifier
    for (auto &c : scope_suffix) {
      if (c == '.')
        c = '_';
    }
    w->name = src->name + "_" + scope_suffix;
    // Strides can be empty (row-major) for the compact buffer.
    w->strides = {};
  }

  // ---- Step 5: Build the T.copy call ----------------------------------------
  // Source region: src[cache_region]  (read, full ndim)
  // Destination  : dst[0 : ext, ...]  (write, squeezed dims only)
  ffi::Array<Range> dst_ranges;
  dst_ranges.reserve(squeezed_shape.size());
  for (size_t i = 0; i < squeezed_shape.size(); ++i) {
    dst_ranges.push_back(
        Range::FromMinExtent(IntImm(DataType::Int(32), 0), squeezed_shape[i]));
  }

  PrimExpr src_region_arg =
      MakeRegionCall(src, cache_region, /*access_mask=*/1);
  PrimExpr dst_region_arg = MakeRegionCall(dst, dst_ranges, /*access_mask=*/2);

  ffi::Map<ffi::String, ObjectRef> copy_annotations;
  if (disable_tma) {
    copy_annotations.Set("disable_tma", Bool(true));
  }
  Stmt copy_stmt =
      Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.copy"),
                    {src_region_arg, dst_region_arg}, copy_annotations));

  std::string transform_mode = transform;
  ICHECK(transform_mode.empty() || transform_mode == "square")
      << "ValueError: unsupported cache_read_at transform `" << transform_mode
      << "`, expected one of {\"\", \"square\"}";
  bool do_square_transform = transform_mode == "square";
  Stmt square_stmt;
  if (do_square_transform) {
    square_stmt = BuildSquareInplaceStmt(dst, squeezed_shape);
  }

  // ---- Step 6: Rewrite buffer references in every consumer block -----------
  ffi::Array<Stmt> subtrees = s_tir::AsArray(loop->body);
  ffi::Map<SBlock, SBlock> block_sref_reuse;
  std::unordered_set<const SBlockNode *> target_block_set(
      all_consumer_blocks.begin(), all_consumer_blocks.end());
  CacheBufferReplacer replacer(src, dst, region_mins, kept_dims,
                               &block_sref_reuse, target_block_set);
  for (int i = 0; i < static_cast<int>(subtrees.size()); ++i) {
    Stmt old_stmt = subtrees[i];
    subtrees.Set(i, Stmt(nullptr));
    subtrees.Set(i, replacer(std::move(old_stmt)));
  }

  // ---- Step 7: Insert copy + allocation into the loop body ------------------
  // Prepend the copy statement before the existing subtrees.
  subtrees.insert(subtrees.begin(), copy_stmt);
  if (do_square_transform) {
    subtrees.insert(subtrees.begin() + 1, square_stmt);
  }

  // Wrap everything in an opaque Block that owns the cache buffer allocation.
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

  // Build the new For loop with the wrapper block as the body.
  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = std::move(alloc_realize);
  For new_loop(new_loop_node);

  // ---- Step 8: Replace in the scope root block ------------------------------
  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const SBlockNode *scope_block = TVM_SREF_TO_SBLOCK(scope_root_sref);

  SBlock new_scope_block = Downcast<SBlock>(
      LoopReplacer(loop, new_loop)(ffi::GetRef<SBlock>(scope_block)));

  block_sref_reuse.Set(ffi::GetRef<SBlock>(scope_block), new_scope_block);
  self->Replace(scope_root_sref, new_scope_block, block_sref_reuse);
}

// ---------------------------------------------------------------------------
// FFI Registration
// ---------------------------------------------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::GlobalDef().def(
      "tl.schedule.ScheduleCacheReadAt",
      [](s_tir::Schedule self, const s_tir::LoopRV &loop_rv,
         const SBlockRV &block_rv, int read_buffer_index,
         const ffi::String &storage_scope, const ffi::String &transform,
         const ffi::String &cache_dtype, bool disable_tma,
         const ffi::Array<SBlockRV> &consumer_block_rvs) {
        ffi::Array<StmtSRef> consumer_block_srefs;
        for (const SBlockRV &rv : consumer_block_rvs) {
          consumer_block_srefs.push_back(self->GetSRef(rv));
        }
        CacheReadAt(self->state(), self->GetSRef(loop_rv),
                    self->GetSRef(block_rv), read_buffer_index, storage_scope,
                    transform, cache_dtype, disable_tma, consumer_block_srefs);
      });
}

} // namespace tl
} // namespace tvm
