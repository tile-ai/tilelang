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
 * \file fill_at.cc
 * \brief Implements the FillAt schedule primitive for tilelang.
 *
 * Given a block and its write buffer index, a loop where the fill should
 * reside, and a fill value, this primitive:
 *
 * 1. Analyzes the buffer write region within one iteration of the
 *    specified loop (by relaxing over all inner loop variables).
 * 2. Emits a T.fill (tl.tileop.fill) statement that initializes the
 *    accessed region of the buffer to the given value.
 * 3. Inserts the fill statement at the beginning of the loop body.
 *
 * This is essential for reduction patterns where an accumulator buffer
 * must be initialized before the reduction loop.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "s_tir/schedule/analysis.h"
#include "s_tir/schedule/transform.h"
#include "s_tir/schedule/utils.h"

#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tvm::s_tir;
using support::NDIntSet;

// ---------------------------------------------------------------------------
// FillAt: main entry point
// ---------------------------------------------------------------------------
static void FillAt(s_tir::ScheduleState self, const StmtSRef &loop_sref,
                   const StmtSRef &block_sref, int write_buffer_index,
                   double value) {
  CheckLoopIsAncestorOfBlock(self, loop_sref, block_sref, "fill_at");
  // ---- Step 1: Obtain the write buffer and loop ----------------------------
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);
  SBlock block_ref = ffi::GetRef<SBlock>(block);
  Buffer buf = GetNthAccessBuffer(self, block_ref, write_buffer_index,
                                  s_tir::BufferIndexType::kWrite);

  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  // ---- Step 2: Gather inner-loop domains and block bindings ----------------
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = s_tir::GetBindings(realize);

  runtime::StorageScope scope = runtime::StorageScope::Create("local");
  ffi::Map<Var, arith::IntSet> var_dom =
      arith::AsIntSet(LoopDomainOfSRefTreePathSkipBlocks(
          /*low_inclusive=*/ffi::GetRef<StmtSRef>(
              self->stmt2ref.at(block)->parent),
          /*high_exclusive=*/loop_sref,
          /*extra_relax_scope=*/scope));

  // ---- Step 3: Relax the buffer write region over the inner loops ----------
  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion &buffer_region : block->writes) {
    if (buffer_region->buffer.same_as(buf)) {
      ffi::Array<arith::IntSet> relaxed =
          arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
      relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
    }
  }
  ICHECK(!relaxed_regions.empty()) << "ValueError: buffer " << buf->name
                                   << " is not written in the specified block";

  NDIntSet unified = support::NDIntSetUnion(relaxed_regions);
  int ndim = static_cast<int>(unified.size());

  arith::Analyzer analyzer;
  ffi::Array<Range> fill_region;
  fill_region.reserve(ndim);

  for (int d = 0; d < ndim; ++d) {
    PrimExpr mn = analyzer.Simplify(unified[d].min());
    PrimExpr mx = analyzer.Simplify(unified[d].max());
    PrimExpr extent = analyzer.Simplify(mx - mn + 1);
    fill_region.push_back(Range::FromMinExtent(mn, extent));
  }

  // ---- Step 4: Build the T.fill call ---------------------------------------
  PrimExpr region_arg = MakeRegionCall(buf, fill_region, /*access_mask=*/2);
  PrimExpr fill_value = make_const(buf->dtype, value);
  Stmt fill_stmt = Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.fill"),
                                 {region_arg, fill_value}));

  // ---- Step 5: Insert the fill at the beginning of the loop body -----------
  // If the loop body is a block realize whose block allocates `buf` (the
  // pattern produced by cache_write_at with write_back=false), insert the
  // fill inside that block so the buffer is used within its allocation
  // scope.  Otherwise prepend to the loop body directly.
  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  ffi::Map<SBlock, SBlock> block_sref_reuse;
  bool inserted_in_block = false;
  if (const auto *realize = loop->body.as<SBlockRealizeNode>()) {
    const SBlockNode *inner = realize->block.get();
    bool allocates_buf = false;
    for (const Buffer &alloc : inner->alloc_buffers) {
      if (alloc.same_as(buf)) {
        allocates_buf = true;
        break;
      }
    }
    if (allocates_buf) {
      ffi::Array<Stmt> inner_subtrees = s_tir::AsArray(inner->body);
      inner_subtrees.insert(inner_subtrees.begin(), fill_stmt);
      ObjectPtr<SBlockNode> new_block_node =
          ffi::make_object<SBlockNode>(*inner);
      new_block_node->body = inner_subtrees.size() == 1
                                 ? inner_subtrees[0]
                                 : SeqStmt(inner_subtrees);
      SBlock new_block(new_block_node);
      block_sref_reuse.Set(ffi::GetRef<SBlock>(inner), new_block);
      ObjectPtr<SBlockRealizeNode> new_realize_node =
          ffi::make_object<SBlockRealizeNode>(*realize);
      new_realize_node->block = new_block;
      new_loop_node->body = SBlockRealize(new_realize_node);
      inserted_in_block = true;
    }
  }
  if (!inserted_in_block) {
    ffi::Array<Stmt> subtrees = s_tir::AsArray(loop->body);
    subtrees.insert(subtrees.begin(), fill_stmt);
    new_loop_node->body =
        subtrees.size() == 1 ? subtrees[0] : SeqStmt(subtrees);
  }
  For new_loop(new_loop_node);

  // ---- Step 6: Replace in the scope root block -----------------------------
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
      "tl.schedule.ScheduleFillAt",
      [](s_tir::Schedule self, const s_tir::LoopRV &loop_rv,
         const SBlockRV &block_rv, int write_buffer_index, double value) {
        FillAt(self->state(), self->GetSRef(loop_rv), self->GetSRef(block_rv),
               write_buffer_index, value);
      });
}

} // namespace tl
} // namespace tvm
