#ifndef TILELANG_IR_H_
#define TILELANG_IR_H_

#include "compat/expr.h"
#include "support/base.h"
#include <tvm/script/ir_builder/ir.h>
#include <tvm/tirx/script/ir_builder/ir.h>

namespace tvm::tl {

using namespace ffi;
using script::ir_builder::ir::ForFrame;
using script::ir_builder::ir::ForFrameObj;
using script::ir_builder::ir::RegionFrame;
using script::ir_builder::ir::StmtFrame;
using script::ir_builder::ir::StmtFrameObj;

TILELANG_DLL ForFrame MakeThreadBindingFrame(
    const std::string &name, const String &thread_tag, const PrimExpr &extent,
    const Map<String, Any> &annotations = {});
TILELANG_DLL ForFrame MakeLaunchThreadFrame();
TILELANG_DLL RegionFrame MakeNamedScope(const String &name,
                                        Map<String, Any> annotations = {});

class KernelLaunchFrameNode : public StmtFrameObj {
public:
  Array<StmtFrame> frames;
  Array<Var> grid_vars;
  Array<PrimExpr> grid_extents;
  Array<Var> thread_vars;
  Optional<Array<PrimExpr>> thread_extents;

  static void RegisterReflection() {
    reflection::ObjectDef<KernelLaunchFrameNode>()
        .def_ro("frames", &KernelLaunchFrameNode::frames)
        .def_ro("grid_vars", &KernelLaunchFrameNode::grid_vars)
        .def_ro("grid_extents", &KernelLaunchFrameNode::grid_extents)
        .def_ro("thread_vars", &KernelLaunchFrameNode::thread_vars)
        .def_ro("thread_extents", &KernelLaunchFrameNode::thread_extents);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.KernelLaunchFrame",
                                    KernelLaunchFrameNode, StmtFrameObj);

  void EnterWithScope() final {
    for (const auto &frame : frames)
      frame->EnterWithScope();
  }

  void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

class KernelLaunchFrame : public StmtFrame {
public:
  explicit KernelLaunchFrame(ObjectPtr<KernelLaunchFrameNode> node)
      : StmtFrame(std::move(node)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(KernelLaunchFrame, StmtFrame,
                                                KernelLaunchFrameNode);
};

} // namespace tvm::tl

#endif
