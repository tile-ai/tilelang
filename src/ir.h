/*!
 * \file tl/ir.h
 * \brief Frame builders shared by the TileLang frontend and its dialects.
 *
 * src/ir.cc owns the target-neutral frontend frames that are registered as
 * tilelang builtins (tl.Parallel, tl.Pipelined, tl.Persistent, tl.KernelLaunch,
 * tl.WarpSpecialize, tl.SideEffect). A dialect owns the frames that only it
 * uses, in its own translation unit (e.g. src/ascend/ir.cc).
 *
 * A dialect launch variant still has to produce a tl.KernelLaunchFrame: the
 * Python dialect binds its launch surface to that object type, so a mixed
 * launch that declares extra launch dimensions must build the same node.
 * The launch frame and the two frame factories it is assembled from are
 * therefore declared here rather than being private to src/ir.cc.
 *
 * This header is included only by the frontend ir.cc translation units, so it
 * re-exports the script-builder and ffi namespaces they are written against.
 */

#ifndef TILELANG_IR_H_
#define TILELANG_IR_H_

#include "support/check.h"
#include <tvm/ffi/reflection/creator.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/ir_builder/tir/ir.h>
#include <tvm/tirx/stmt.h>

#include <string>

namespace tvm {
namespace tl {

using namespace script::ir_builder::tirx;
using namespace ffi;

// Build a ForFrame that emits a target-neutral kThreadBinding loop for one
// grid (program index) axis of a kernel launch. The launch nest is
// materialized into the target-specific form (thread_extent AttrStmt on GPU,
// serial For on CPU) by the tl.MaterializeKernelLaunch pass once the Target is
// known at compile time.
ForFrame MakeThreadBindingFrame(const std::string &name,
                                const String &thread_tag,
                                const PrimExpr &extent);

// Build a frame whose exit prefixes the body with
// `tx = tl.launch_thread_idx(0); ty = ...; tz = ...` Bind statements. The
// launch nest is traced before the Target is known, so the thread indices are
// only placeholders here: the Vars keep their identity through
// tl.MaterializeKernelLaunch, which rebinds them as threadIdx.* thread_extent
// scopes on SIMT backends and drops them elsewhere.
ForFrame MakeLaunchThreadFrame();

/*!
 * \brief A frame that represents a kernel launch.
 *
 * \sa KernelLaunchFrameNode
 */
class KernelLaunchFrameNode : public TIRFrameNode {
public:
  /*! \brief Grid loops, thread placeholders and the root block, outer to
   * inner. */
  Array<TIRFrame> frames;
  /*! \brief Program (grid) index vars, one per launch axis. */
  Array<tvm::tirx::Var> grid_vars;
  /*! \brief Grid extents, one per launch axis. */
  Array<PrimExpr> grid_extents;
  /*! \brief Placeholder thread index vars for the x, y and z axes. */
  Array<tvm::tirx::Var> thread_vars;
  /*! \brief Requested SIMT thread-block extents, when threads= was given. */
  Optional<Array<PrimExpr>> thread_extents;

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<KernelLaunchFrameNode>()
        .def_ro("frames", &KernelLaunchFrameNode::frames)
        .def_ro("grid_vars", &KernelLaunchFrameNode::grid_vars)
        .def_ro("grid_extents", &KernelLaunchFrameNode::grid_extents)
        .def_ro("thread_vars", &KernelLaunchFrameNode::thread_vars)
        .def_ro("thread_extents", &KernelLaunchFrameNode::thread_extents);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.KernelLaunchFrame",
                                    KernelLaunchFrameNode, TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    for (auto frame = frames.begin(); frame != frames.end(); ++frame)
      (*frame)->EnterWithScope();
  }
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

/*!
 * \brief Managed reference to KernelLaunchFrameNode.
 *
 * \sa KernelLaunchFrameNode
 */
class KernelLaunchFrame : public TIRFrame {
public:
  explicit KernelLaunchFrame(ObjectPtr<KernelLaunchFrameNode> data)
      : TIRFrame(UnsafeInit{}) {
    ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(KernelLaunchFrame, TIRFrame,
                                                KernelLaunchFrameNode);
};

} // namespace tl
} // namespace tvm

#endif // TILELANG_IR_H_
