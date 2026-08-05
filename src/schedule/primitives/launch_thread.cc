#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/s_tir/schedule/state.h>
#include <tvm/s_tir/utils.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "utils.h"

namespace tvm {
namespace tl {
using namespace tirx;
using namespace tvm::s_tir;

// ---------------------------------------------------------------------------
// LaunchThread: wrap a block's body in a For(kThreadBinding) loop.
// ---------------------------------------------------------------------------
static void LaunchThread(s_tir::ScheduleState self, const StmtSRef &block_sref,
                         int num_threads) {
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);

  Var tx("tx");
  IterVar thread_iter(Range(nullptr), Var("threadIdx.x"),
                      tirx::IterVarType::kThreadIndex, "threadIdx.x");
  Stmt new_body = For(tx, 0, num_threads, ForKind::kThreadBinding, block->body,
                      thread_iter, {}, std::nullopt);

  tvm::ffi::ObjectPtr<SBlockNode> new_block_node =
      ffi::make_object<SBlockNode>(*block);
  new_block_node->body = std::move(new_body);
  SBlock new_block(new_block_node);

  ffi::Map<SBlock, SBlock> block_sref_reuse;
  block_sref_reuse.Set(ffi::GetRef<SBlock>(block), new_block);
  self->Replace(block_sref, new_block, block_sref_reuse);
}

// ---------------------------------------------------------------------------
// ParallelizeLoop: change a For loop to parallel (cooperative thread) mode.
// ---------------------------------------------------------------------------
static void ParallelizeLoop(s_tir::ScheduleState self,
                            const StmtSRef &loop_sref) {
  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  tvm::ffi::ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->kind = ForKind::kParallel;
  new_loop_node->thread_binding = std::nullopt;
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
// PipelineLoop: annotate a loop with num_stages for software pipelining.
// ---------------------------------------------------------------------------
static void PipelineLoop(s_tir::ScheduleState self, const StmtSRef &loop_sref,
                         int num_stages) {
  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  tvm::ffi::ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->annotations.Set("num_stages",
                                 IntImm(DataType::Int(32), num_stages));
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
      "tl.schedule.ScheduleLaunchThread",
      [](s_tir::Schedule self, const SBlockRV &block_rv, int num_threads) {
        LaunchThread(self->state(), self->GetSRef(block_rv), num_threads);
      });
  refl::GlobalDef().def("tl.schedule.ScheduleParallelizeLoop",
                        [](s_tir::Schedule self, const s_tir::LoopRV &loop_rv) {
                          ParallelizeLoop(self->state(),
                                          self->GetSRef(loop_rv));
                        });
  refl::GlobalDef().def(
      "tl.schedule.SchedulePipelineLoop",
      [](s_tir::Schedule self, const s_tir::LoopRV &loop_rv, int num_stages) {
        PipelineLoop(self->state(), self->GetSRef(loop_rv), num_stages);
      });
}

} // namespace tl
} // namespace tvm
