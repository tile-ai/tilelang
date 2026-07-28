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
 * \file copy_at.cc
 * \brief Implements the CopyAt schedule primitive — replace the compute nest
 *        of a copy-like block with a tile-level T.copy call whose source and
 *        destination tiles are derived from the block's access regions.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <vector>

#include "s_tir/schedule/transform.h"
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/utils.h>

#include "runtime/thread_storage_scope.h"
#include "s_tir/schedule/utils.h"
#include "s_tir/support/nd_int_set.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tvm::s_tir;
using namespace ffi;

// ---------------------------------------------------------------------------
// CopyAt: replace a copy-like block's compute nest with a T.copy call.
// ---------------------------------------------------------------------------
static void CopyAt(s_tir::ScheduleState self, const StmtSRef &loop_sref,
                   const StmtSRef &block_sref, int read_buffer_index,
                   int write_buffer_index) {
  CheckLoopIsAncestorOfBlock(self, loop_sref, block_sref, "copy_at");
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);
  SBlock block_ref = ffi::GetRef<SBlock>(block);
  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  Buffer src = GetNthAccessBuffer(self, block_ref, read_buffer_index,
                                  s_tir::BufferIndexType::kRead);
  Buffer dst = GetNthAccessBuffer(self, block_ref, write_buffer_index,
                                  s_tir::BufferIndexType::kWrite);

  ffi::Array<Range> src_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, src, s_tir::BufferIndexType::kRead,
      runtime::StorageScope::Create(src.scope()));
  ffi::Array<Range> dst_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, dst, s_tir::BufferIndexType::kWrite,
      runtime::StorageScope::Create(dst.scope()));

  Stmt copy_stmt =
      Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.copy"),
                    {
                        MakeRegionCall(src, src_region, /*access_mask=*/1),
                        MakeRegionCall(dst, dst_region, /*access_mask=*/2),
                    }));

  ComputeNestReplacer replacer(block, copy_stmt);
  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = replacer(loop->body);
  ICHECK(replacer.replaced())
      << "ValueError: copy_at failed to isolate the tiled compute nest";
  For new_loop(new_loop_node);

  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const SBlockNode *scope_block = TVM_SREF_TO_SBLOCK(scope_root_sref);

  ffi::Map<SBlock, SBlock> block_sref_reuse;
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

  refl::GlobalDef().def("tl.schedule.ScheduleCopyAt",
                        [](s_tir::Schedule self, const s_tir::LoopRV &loop_rv,
                           const SBlockRV &block_rv, int read_buffer_index,
                           int write_buffer_index) {
                          CopyAt(self->state(), self->GetSRef(loop_rv),
                                 self->GetSRef(block_rv), read_buffer_index,
                                 write_buffer_index);
                        });
}

} // namespace tl
} // namespace tvm
