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
 * \file gemm_at.cc
 * \brief Implements the GemmAt schedule primitive — replace the compute nest
 *        of a matmul block with a tile-level T.gemm call whose operand tiles
 *        are derived from the block's read/write access regions.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
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

static int GetConstInt(const PrimExpr &expr, const char *what) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return static_cast<int>(imm->value);
  }
  ICHECK(false) << "ValueError: gemm_at expects a static " << what
                << ", but got " << expr;
  return 0;
}

static PrimExpr GetMatrixStride(const Buffer &buf) {
  if (buf->strides.size() >= 2) {
    return buf->strides[buf->strides.size() - 2];
  }
  ICHECK_GE(buf->shape.size(), 2U)
      << "ValueError: gemm_at expects at least a 2D buffer, but got "
      << buf->name;
  return buf->shape[buf->shape.size() - 1];
}

// ---------------------------------------------------------------------------
// GemmAt: replace a matmul block's compute nest with a T.gemm call.
// ---------------------------------------------------------------------------
static void GemmAt(s_tir::ScheduleState self, const StmtSRef &loop_sref,
                   const StmtSRef &block_sref, bool transpose_a,
                   bool transpose_b, bool clear_accum, int policy_type,
                   bool use_py) {
  CheckLoopIsAncestorOfBlock(self, loop_sref, block_sref, "gemm_at");
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);
  SBlock block_ref = ffi::GetRef<SBlock>(block);
  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  ICHECK_EQ(block->reads.size(), 2U)
      << "ValueError: gemm_at expects a matmul-like block with exactly two "
         "reads";
  ICHECK_EQ(block->writes.size(), 1U)
      << "ValueError: gemm_at expects a matmul-like block with exactly one "
         "write";

  Buffer a =
      GetNthAccessBuffer(self, block_ref, 0, s_tir::BufferIndexType::kRead);
  Buffer b =
      GetNthAccessBuffer(self, block_ref, 1, s_tir::BufferIndexType::kRead);
  Buffer c =
      GetNthAccessBuffer(self, block_ref, 0, s_tir::BufferIndexType::kWrite);

  ffi::Array<Range> a_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, a, s_tir::BufferIndexType::kRead,
      runtime::StorageScope::Create(a.scope()));
  ffi::Array<Range> b_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, b, s_tir::BufferIndexType::kRead,
      runtime::StorageScope::Create(b.scope()));
  ffi::Array<Range> c_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, c, s_tir::BufferIndexType::kWrite,
      runtime::StorageScope::Create(c.scope()));

  // Support 3D regions (normalize_to_matmul adds a leading batch dim).
  // For m/n/k extraction and c_row/c_col, work with 2D regions by stripping
  // a trivial leading batch dim (extent 1). For MakeRegionCall, keep the
  // original 3D regions so indices match the buffer dimensionality.
  ffi::Array<Range> a_region_2d = a_region;
  ffi::Array<Range> b_region_2d = b_region;
  ffi::Array<Range> c_region_2d = c_region;
  auto MaybeStripBatchDim = [](ffi::Array<Range> &region) {
    if (region.size() == 3) {
      if (auto *imm = region[0]->extent.as<IntImmNode>()) {
        if (imm->value == 1) {
          region.erase(region.begin());
        }
      }
    }
  };
  MaybeStripBatchDim(a_region_2d);
  MaybeStripBatchDim(b_region_2d);
  MaybeStripBatchDim(c_region_2d);

  // After stripping a trivial batch dim, the tiles must be exactly 2D.
  // Higher-rank regions (e.g. conv sliding windows that im2col did not
  // flatten) cannot be expressed as a single T.gemm — reject them so the
  // Matmul rule falls through to a generic schedule rule.
  ICHECK_EQ(a_region_2d.size(), 2U)
      << "ValueError: gemm_at currently expects a 2D lhs tile, but got rank "
      << a_region_2d.size();
  ICHECK_EQ(b_region_2d.size(), 2U)
      << "ValueError: gemm_at currently expects a 2D rhs tile, but got rank "
      << b_region_2d.size();
  ICHECK_EQ(c_region_2d.size(), 2U)
      << "ValueError: gemm_at currently expects a 2D accumulator tile, but "
         "got rank "
      << c_region_2d.size();

  int m = GetConstInt(c_region_2d[c_region_2d.size() - 2]->extent, "M extent");
  int n = GetConstInt(c_region_2d[c_region_2d.size() - 1]->extent, "N extent");
  int k = GetConstInt(transpose_a ? a_region_2d[a_region_2d.size() - 2]->extent
                                  : a_region_2d[a_region_2d.size() - 1]->extent,
                      "K extent");

  PrimExpr a_region_arg = MakeRegionCall(a, a_region, /*access_mask=*/1);
  PrimExpr b_region_arg = MakeRegionCall(b, b_region, /*access_mask=*/1);
  PrimExpr c_region_arg = MakeRegionCall(c, c_region, /*access_mask=*/3);

  PrimExpr stride_a = GetMatrixStride(a);
  PrimExpr stride_b = GetMatrixStride(b);
  int stride_a_value = GetConstInt(stride_a, "lhs stride");
  int stride_b_value = GetConstInt(stride_b, "rhs stride");
  PrimExpr offset_a = a_region_2d[a_region_2d.size() - 1]->min;
  PrimExpr offset_b = b_region_2d[b_region_2d.size() - 1]->min;
  int offset_a_value = GetConstInt(offset_a, "lhs offset");
  int offset_b_value = GetConstInt(offset_b, "rhs offset");
  // In the new TVM, region mins may be non-constant loop-variable
  // expressions (e.g. ax2_0 * 64 representing the absolute grid position).
  // The c_row/c_col values represent the RELATIVE offset of the C fragment
  // within the current tile, which is always 0.  When the expression is
  // a loop variable, the absolute position is irrelevant — fall back to 0.
  PrimExpr c_row = c_region_2d[c_region_2d.size() - 2]->min;
  PrimExpr c_col = c_region_2d[c_region_2d.size() - 1]->min;
  int c_row_value = 0;
  int c_col_value = 0;
  if (const auto *imm = c_row.as<IntImmNode>()) {
    c_row_value = static_cast<int>(imm->value);
  }
  if (const auto *imm = c_col.as<IntImmNode>()) {
    c_col_value = static_cast<int>(imm->value);
  }

  const char *op_name = use_py ? "tl.tileop.gemm_py" : "tl.tileop.gemm";
  Stmt gemm_stmt = Evaluate(Call(DataType::Handle(), Op::Get(op_name),
                                 {
                                     a_region_arg,
                                     b_region_arg,
                                     c_region_arg,
                                     Bool(transpose_a),
                                     Bool(transpose_b),
                                     IntImm(DataType::Int(32), m),
                                     IntImm(DataType::Int(32), n),
                                     IntImm(DataType::Int(32), k),
                                     IntImm(DataType::Int(32), policy_type),
                                     Bool(clear_accum),
                                     IntImm(DataType::Int(32), stride_a_value),
                                     IntImm(DataType::Int(32), stride_b_value),
                                     IntImm(DataType::Int(32), offset_a_value),
                                     IntImm(DataType::Int(32), offset_b_value),
                                     IntImm(DataType::Int(32), 1),
                                     IntImm(DataType::Int(32), 0),
                                     make_zero(DataType::UInt(32)),
                                     IntImm(DataType::Int(32), c_row_value),
                                     IntImm(DataType::Int(32), c_col_value),
                                 }));

  ComputeNestReplacer replacer(block, gemm_stmt);
  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = replacer(loop->body);
  ICHECK(replacer.replaced())
      << "ValueError: gemm_at failed to isolate the tiled compute nest";
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

  refl::GlobalDef().def(
      "tl.schedule.ScheduleGemmAt",
      [](s_tir::Schedule self, const s_tir::LoopRV &loop_rv,
         const SBlockRV &block_rv, bool transpose_a, bool transpose_b,
         bool clear_accum, int policy_type, bool use_py) {
        GemmAt(self->state(), self->GetSRef(loop_rv), self->GetSRef(block_rv),
               transpose_a, transpose_b, clear_accum, policy_type, use_py);
      });
}

} // namespace tl
} // namespace tvm
