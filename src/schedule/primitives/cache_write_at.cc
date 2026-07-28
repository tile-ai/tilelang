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
 * \file cache_write_at.cc
 * \brief Implements the CacheWriteAt schedule primitive — the mirror of
 *        CacheReadAt for write buffers.  Stages a per-iteration slice of the
 *        producer's write region into `storage_scope` and writes it back to
 *        the original buffer at loop end.  Optionally marks the staged buffer
 *        as a TileLang reducer (for accumulator handling).
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <vector>

// TVM internal headers (included via ${TVM_SOURCE}/src in the include path)
#include "s_tir/schedule/transform.h"
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/utils.h>

// TileLang headers
#include "../../transform/layout_reducer.h"
#include "runtime/thread_storage_scope.h"
#include "s_tir/schedule/utils.h"
#include "s_tir/support/nd_int_set.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tvm::s_tir;
using namespace ffi;
using support::NDIntSet;

// ---------------------------------------------------------------------------
// CacheWriteAt: mirror of CacheReadAt for write buffers, with optional
// accumulator semantics (when reduce_type is set).
// ---------------------------------------------------------------------------
static void CacheWriteAt(s_tir::ScheduleState self, const StmtSRef &loop_sref,
                         const StmtSRef &block_sref, int write_buffer_index,
                         const ffi::String &storage_scope, bool write_back,
                         const ffi::String &reduce_type,
                         const ffi::String &reducer_replication,
                         const ffi::String &cache_dtype) {
  CheckLoopIsAncestorOfBlock(self, loop_sref, block_sref, "cache_write_at");
  // ---- Step 1: Obtain destination buffer and loop ---------------------------
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);
  SBlock block_ref = ffi::GetRef<SBlock>(block);
  Buffer src = GetNthAccessBuffer(self, block_ref, write_buffer_index,
                                  s_tir::BufferIndexType::kWrite);

  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  // ---- Step 2: Gather inner-loop domains and block bindings -----------------
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = s_tir::GetBindings(realize);

  runtime::StorageScope scope = runtime::StorageScope::Create(storage_scope);
  ffi::Map<Var, arith::IntSet> var_dom =
      arith::AsIntSet(LoopDomainOfSRefTreePathSkipBlocks(
          /*low_inclusive=*/ffi::GetRef<StmtSRef>(
              self->stmt2ref.at(block)->parent),
          /*high_exclusive=*/loop_sref,
          /*extra_relax_scope=*/scope));

  // ---- Step 3: Relax the buffer write region over the inner loops -----------
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
    std::string scope_suffix = storage_scope;
    for (auto &c : scope_suffix) {
      if (c == '.')
        c = '_';
    }
    w->name = src->name + "_" + scope_suffix;
    w->strides = {};
  }

  // ---- Step 5: Build optional T.copy call (write-back: cache → original) ---
  ffi::Array<Range> dst_ranges;
  dst_ranges.reserve(squeezed_shape.size());
  for (size_t i = 0; i < squeezed_shape.size(); ++i) {
    dst_ranges.push_back(
        Range::FromMinExtent(IntImm(DataType::Int(32), 0), squeezed_shape[i]));
  }

  Stmt copy_stmt;
  if (write_back) {
    // For write-back: source is the cache, destination is the original buffer
    PrimExpr cache_region_arg =
        MakeRegionCall(dst, dst_ranges, /*access_mask=*/1);
    PrimExpr orig_region_arg =
        MakeRegionCall(src, cache_region, /*access_mask=*/2);
    copy_stmt = Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.copy"),
                              {cache_region_arg, orig_region_arg}));
  }

  bool use_reducer = !reduce_type.empty();
  PrimExpr reducer_region_arg;
  Stmt finalize_stmt;
  if (use_reducer) {
    ICHECK(reduce_type == "sum")
        << "ValueError: reducer-backed cache_write_at currently only supports "
           "`sum` reductions";
    ICHECK(reducer_replication == "all" || reducer_replication == "none")
        << "ValueError: unsupported reducer replication `"
        << reducer_replication << "`, expected one of {\"all\", \"none\"}";
    reducer_region_arg = MakeRegionCall(dst, dst_ranges, /*access_mask=*/2);
    finalize_stmt =
        Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.finalize_reducer"),
                      {reducer_region_arg}));
  }

  // ---- Step 6: Rewrite buffer references in the loop body -------------------
  ffi::Array<Stmt> subtrees = s_tir::AsArray(loop->body);
  ffi::Map<SBlock, SBlock> block_sref_reuse;
  std::unordered_set<const SBlockNode *> empty_target_set;
  CacheBufferReplacer replacer(src, dst, region_mins, kept_dims,
                               &block_sref_reuse, empty_target_set);
  for (int i = 0; i < static_cast<int>(subtrees.size()); ++i) {
    Stmt old_stmt = subtrees[i];
    subtrees.Set(i, Stmt(nullptr));
    subtrees.Set(i, replacer(std::move(old_stmt)));
  }

  // ---- Step 7: Insert fill/finalize/write-back at correct positions
  // ----------
  if (use_reducer) {
    // Look for an init subtree (from decompose_reduction) that writes only a
    // constant to dst.  Replace it in-place with T.fill so the tile pipeline
    // handles initialization.  If no init block is found, insert T.fill(0) at
    // the beginning as a fallback.
    int init_idx = -1;
    double init_const = 0.0;
    for (int i = 0; i < static_cast<int>(subtrees.size()); ++i) {
      bool has_dst_store = false;
      bool all_const = true;
      bool has_dst_load = false;
      double found_const = 0.0;
      PostOrderVisit(subtrees[i], [&](const ObjectRef &obj) {
        if (!all_const)
          return;
        if (auto *store = obj.as<BufferStoreNode>()) {
          if (store->buffer.same_as(dst)) {
            has_dst_store = true;
            if (auto *fimm = store->value.as<FloatImmNode>()) {
              found_const = fimm->value;
            } else if (auto *iimm = store->value.as<IntImmNode>()) {
              found_const = static_cast<double>(iimm->value);
            } else {
              all_const = false;
            }
          }
        }
        if (auto *load = obj.as<BufferLoadNode>()) {
          if (load->buffer.same_as(dst)) {
            has_dst_load = true;
          }
        }
      });
      if (has_dst_store && all_const && !has_dst_load) {
        init_idx = i;
        init_const = found_const;
        break;
      }
    }
    if (init_idx >= 0) {
      subtrees.Set(init_idx,
                   Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.fill"),
                                 {reducer_region_arg,
                                  make_const(dst->dtype, init_const)})));
    } else {
      subtrees.insert(
          subtrees.begin(),
          Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.fill"),
                        {reducer_region_arg, make_const(dst->dtype, 0.0)})));
    }
  }
  if (use_reducer) {
    // Insert finalize_reducer BEFORE any epilogue subtrees that read from
    // the cache buffer but do not write to it.  This ensures the AllReduce
    // completes before the epilogue uses the reduced value.
    int finalize_pos = static_cast<int>(subtrees.size());
    for (int i = static_cast<int>(subtrees.size()) - 1; i >= 0; --i) {
      bool writes_dst = false;
      PostOrderVisit(subtrees[i], [&writes_dst, &dst](const ObjectRef &obj) {
        if (writes_dst)
          return;
        if (auto *store = obj.as<BufferStoreNode>()) {
          if (store->buffer.same_as(dst)) {
            writes_dst = true;
          }
        }
      });
      if (writes_dst) {
        finalize_pos = i + 1;
        break;
      }
    }
    subtrees.insert(subtrees.begin() + finalize_pos, finalize_stmt);
  }
  if (write_back) {
    subtrees.push_back(copy_stmt);
  }

  SBlock alloc_block(
      /*iter_vars=*/{},
      /*reads=*/{},
      /*writes=*/{},
      /*name_hint=*/"",
      /*body=*/subtrees.size() == 1 ? subtrees[0] : SeqStmt(subtrees),
      /*init=*/std::nullopt,
      /*alloc_buffers=*/{dst});
  if (use_reducer) {
    auto *block_ptr = alloc_block.CopyOnWrite();
    ffi::Map<Var, ffi::Map<ffi::String, ffi::String>> reducer_info;
    ffi::Map<ffi::String, ffi::String> reducer_meta;
    reducer_meta.Set("op", reduce_type);
    reducer_meta.Set("rep", reducer_replication);
    reducer_info.Set(dst->data, reducer_meta);
    block_ptr->annotations.Set(tvm::tl::attr::kReducerInfo, reducer_info);
  }
  SBlockRealize alloc_realize(
      /*values=*/{},
      /*predicate=*/const_true(), alloc_block);

  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = std::move(alloc_realize);
  For new_loop(new_loop_node);

  // ---- Step 8: Replace in the scope root block ------------------------------
  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const SBlockNode *scope_block = TVM_SREF_TO_SBLOCK(scope_root_sref);

  SBlock new_scope_block = Downcast<SBlock>(
      LoopReplacer(loop, new_loop)(ffi::GetRef<SBlock>(scope_block)));

  if (!write_back) {
    // Bridge/intermediate use-case: all consumers are rewritten to the cache
    // buffer, so the original tensor allocation can be removed from the
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

  refl::GlobalDef().def(
      "tl.schedule.ScheduleCacheWriteAt",
      [](s_tir::Schedule self, const s_tir::LoopRV &loop_rv,
         const SBlockRV &block_rv, int write_buffer_index,
         const ffi::String &storage_scope, bool write_back,
         const ffi::String &reduce_type, const ffi::String &reducer_replication,
         const ffi::String &cache_dtype) {
        CacheWriteAt(self->state(), self->GetSRef(loop_rv),
                     self->GetSRef(block_rv), write_buffer_index, storage_scope,
                     write_back, reduce_type, reducer_replication, cache_dtype);
      });
}

} // namespace tl
} // namespace tvm
