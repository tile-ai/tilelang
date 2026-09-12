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

#include "op/builtin.h" // NOLINT(misc-include-cleaner) launch_thread_idx
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
inline ForFrame MakeThreadBindingFrame(const std::string &name,
                                       const String &thread_tag,
                                       const PrimExpr &extent) {
  using namespace tvm::tirx;
  Var var = Var(name, extent->dtype);
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.push_back(var);
  n->doms.push_back(Range(make_const(extent->dtype, 0), extent));
  n->f_make_for_loop =
      [thread_tag](const Array<Var> &vars, const Array<Range> &doms,
                   const Array<Optional<PrimExpr>> &steps, Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), 1);
    ICHECK_EQ(doms.size(), 1);
    IterVar iter_var(Range{nullptr}, Var(thread_tag, vars[0]->dtype),
                     IterVarType::kThreadIndex, thread_tag);
    Optional<PrimExpr> step =
        !steps.empty() ? steps[0] : Optional<PrimExpr>(std::nullopt);
    return For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kThreadBinding,
               body,
               /*thread_binding=*/iter_var,
               /*annotations=*/Map<String, Any>{},
               /*step=*/step);
  };
  return ForFrame(n);
}

// Build a frame whose exit prefixes the body with
// `tx = tl.launch_thread_idx(0); ty = ...; tz = ...` Bind statements. The
// launch nest is traced before the Target is known, so the thread indices are
// only placeholders here: the Vars keep their identity through
// tl.MaterializeKernelLaunch, which rebinds them as threadIdx.* thread_extent
// scopes on SIMT backends and drops them elsewhere.
inline ForFrame MakeLaunchThreadFrame() {
  using namespace tvm::tirx;
  static const char *kThreadVarNames[3] = {"tx", "ty", "tz"};
  DataType dtype = DataType::Int(32);
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  for (int axis = 0; axis < 3; axis++) {
    n->vars.push_back(Var(kThreadVarNames[axis], dtype));
    // The extent is decided by the backend at materialization; this dom only
    // keeps the ForFrame invariants satisfied.
    n->doms.push_back(Range(make_const(dtype, 0), make_const(dtype, 1)));
  }
  n->f_make_for_loop = [](const Array<Var> &vars, const Array<Range> &doms,
                          const Array<Optional<PrimExpr>> &steps,
                          Stmt body) -> Stmt {
    Array<Stmt> seq;
    for (int axis = 0; axis < static_cast<int>(vars.size()); axis++) {
      PrimExpr thread_idx = Call(vars[axis]->dtype, launch_thread_idx(),
                                 {IntImm(DataType::Int(32), axis)});
      seq.push_back(tvm::tirx::Bind(vars[axis], thread_idx));
    }
    seq.push_back(body);
    return SeqStmt::Flatten(seq);
  };
  return ForFrame(n);
}

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
