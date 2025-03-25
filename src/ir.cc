// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/ir.cc
 * \brief Extension for the tvm script frontend.
 *
 */

#include <tvm/arith/analyzer.h>
#include <tvm/script/ir_builder/tir/ir.h>

namespace tvm {
namespace tl {

constexpr const char *tilelang_is_cpu_kernel_frame =
    "tilelang.is_cpu_kernel_frame";

using namespace script::ir_builder::tir;

static Var CreateEnvThread(String name, String thread_tag, DataType dtype) {
  using namespace tvm::tir;
  using namespace tvm::script::ir_builder;
  IterVar iter_var(Range{nullptr}, Var(name, dtype),
                   tvm::tir::IterVarType::kThreadIndex, thread_tag);
  Var var = iter_var->var;
  if (Optional<PrimFuncFrame> opt_frame =
          IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    opt_frame.value()->env_threads.Set(var, iter_var);
  } else {
    LOG(FATAL) << "EnvThread can only be used inside a PrimFunc";
  }
  return var;
}

static ForFrame MakeIterVarFrame(std::string name, PrimExpr dom) {
  using namespace tvm::tir;
  Var var = Var(name);
  // Create a frame that represents a loop over the given domain.
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.push_back(var);
  n->doms.push_back(Range(0, dom));
  n->f_make_for_loop = [](Array<Var> vars, Array<Range> doms,
                          Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), 1);
    ICHECK_EQ(doms.size(), 1);
    return For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kSerial, body);
  };
  return ForFrame(n);
}

ForFrame ParallelFor(Array<PrimExpr> extents,
                     Map<String, ObjectRef> annotations) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  for (const auto &extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", extent.dtype()));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop = [annotations](Array<Var> vars, Array<Range> doms,
                                     Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      body =
          For(var, dom->min, dom->extent, ForKind::kParallel, std::move(body),
              /*thread_binding=*/NullOpt, /*annotations=*/annotations);
    }
    return body;
  };
  return ForFrame(n);
}

ForFrame PipelinedFor(PrimExpr start, PrimExpr stop, int num_stages,
                      Array<PrimExpr> order, Array<PrimExpr> stages,
                      Array<Array<PrimExpr>> sync,
                      Array<Array<PrimExpr>> groups) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  DataType dtype = stop.dtype();
  n->vars.push_back(Var("v", dtype));
  n->doms.push_back(Range(start, stop));
  n->f_make_for_loop = [=](Array<Var> vars, Array<Range> doms,
                           Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    ICHECK(n == 1);
    Map<String, ObjectRef> anno;
    if (num_stages > 0)
      anno.Set("num_stages", PrimExpr(num_stages));
    if (order.size() > 0)
      anno.Set("tl_pipeline_order", order);
    if (stages.size() > 0)
      anno.Set("tl_pipeline_stage", stages);
    if (sync.size() > 0)
      anno.Set("tl_pipeline_sync", sync);
    if (groups.size() > 0)
      anno.Set("tl_pipeline_group", groups);
    body = For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kSerial,
               std::move(body),
               /*thread_binding=*/NullOpt, /*annotations=*/anno);
    return body;
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
  Array<TIRFrame> frames;

  void VisitAttrs(tvm::AttrVisitor *v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("frames", &frames);
  }

  static constexpr const char *_type_key = "tl.KernelLaunchFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(KernelLaunchFrameNode, TIRFrameNode);

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
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(KernelLaunchFrame, TIRFrame,
                                                    KernelLaunchFrameNode);
};

KernelLaunchFrame KernelLaunch(Array<PrimExpr> grid_size,
                               Array<PrimExpr> block_size,
                               Map<String, ObjectRef> attrs) {
  ObjectPtr<KernelLaunchFrameNode> n = make_object<KernelLaunchFrameNode>();

  // If the kernel is a CPU kernel, we don't need to launch any threads.
  bool is_cpu_kernel_frame =
      attrs.defined() && attrs.count(tilelang_is_cpu_kernel_frame);

  if (is_cpu_kernel_frame) {
    ICHECK(grid_size.size() >= 0);
    ICHECK(block_size.size() == 0) << "CPU kernel cannot have block size";
    ICHECK(attrs.defined());
    // create grid loop var
    for (int i = 0; i < grid_size.size(); i++) {
      n->frames.push_back(
          MakeIterVarFrame("block_var_" + std::to_string(i), grid_size[i]));
    }
    // Launch CPU Kernel
  } else {
    // Launch GPU Kernel
    ICHECK(grid_size.size() <= 3);
    if (grid_size.size() > 0)
      n->frames.push_back(LaunchThread(
          CreateEnvThread("bx", "blockIdx.x", grid_size[0].dtype()),
          grid_size[0]));
    if (grid_size.size() > 1)
      n->frames.push_back(LaunchThread(
          CreateEnvThread("by", "blockIdx.y", grid_size[1].dtype()),
          grid_size[1]));
    if (grid_size.size() > 2)
      n->frames.push_back(LaunchThread(
          CreateEnvThread("bz", "blockIdx.z", grid_size[2].dtype()),
          grid_size[2]));
    if (block_size.defined()) {
      ICHECK(block_size.size() <= 3);
      if (block_size.size() > 0) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("tx", "threadIdx.x", block_size[0].dtype()),
            block_size[0]));
      }
      if (block_size.size() > 1) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("ty", "threadIdx.y", block_size[1].dtype()),
            block_size[1]));
      }
      if (block_size.size() > 2) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("tz", "threadIdx.z", block_size[2].dtype()),
            block_size[2]));
      }
    } else {
      n->frames.push_back(Block(""));
    }
  }

  if (attrs.defined()) {
    auto empty_block = Block("");
    empty_block->annotations = attrs;
    n->frames.push_back(empty_block);
  } else {
    n->frames.push_back(Block(""));
  }

  return KernelLaunchFrame(n);
}

TVM_REGISTER_NODE_TYPE(KernelLaunchFrameNode);

TVM_REGISTER_GLOBAL("tl.Parallel").set_body_typed(ParallelFor);
TVM_REGISTER_GLOBAL("tl.Pipelined").set_body_typed(PipelinedFor);
TVM_REGISTER_GLOBAL("tl.KernelLaunch").set_body_typed(KernelLaunch);

} // namespace tl
} // namespace tvm
