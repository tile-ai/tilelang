#include "ir.h"
#include "transform/common/attr.h"

#include <tvm/tirx/op/region.h>

namespace tvm::tl {
namespace builder = script::ir_builder::ir;

KernelLaunchFrame MixedKernelLaunch(const Array<PrimExpr> &grid_size,
                                    PrimExpr vector_extent,
                                    const Map<String, Any> &attributes) {
  TVM_FFI_CHECK_EQ(grid_size.size(), 1, ValueError)
      << "MixedKernel only supports a one-dimensional grid";
  auto vector_count = GetConstInt(vector_extent);
  TVM_FFI_CHECK(vector_count && (*vector_count == 1 || *vector_count == 2),
                ValueError)
      << "Mixed-kernel vector_count must be the constant integer 1 or 2";
  auto frame = make_object<KernelLaunchFrameNode>();
  ForFrame blocks = MakeThreadBindingFrame("bx", "blockIdx.x", grid_size[0]);
  ForFrame vectors = MakeThreadBindingFrame("sid", "cthread", vector_extent);
  frame->grid_vars = {blocks->vars[0], vectors->vars[0]};
  frame->grid_extents = {grid_size[0], vector_extent};
  frame->frames = {blocks, vectors};
  ForFrame threads = MakeLaunchThreadFrame();
  frame->thread_vars = threads->vars;
  frame->frames.push_back(threads);
  auto annotations = attributes;
  annotations.Set("vector_count", vector_extent);
  frame->frames.push_back(MakeNamedScope(DeviceMainBlockName, annotations));
  return KernelLaunchFrame(frame);
}

namespace {
RegionStmt NamedScope(const String &name, SeqStmt body,
                      Map<String, Any> annotations, Location location) {
  annotations.Set("name_hint", name);
  return RegionStmt(Op::Get("tl.scope"), {}, {}, DictAttrs(annotations), body,
                    {}, location);
}

Map<String, Any> VFAnnotations(int64_t latency, int64_t source_index) {
  Map<String, Any> annotations{
      {"tl.vf_source_index", IntImm::Int64(source_index)}};
  if (latency > 0)
    annotations.Set("tl.vf_latency", IntImm::Int64(latency));
  return annotations;
}
} // namespace

class SimtVFFrameNode : public StmtFrameObj {
public:
  Array<PrimExpr> thread_extents;
  Array<Var> thread_vars;
  int64_t vf_latency = 0;
  int64_t source_index = 0;

  static void RegisterReflection() {
    reflection::ObjectDef<SimtVFFrameNode>()
        .def_ro("thread_extents", &SimtVFFrameNode::thread_extents)
        .def_ro("thread_vars", &SimtVFFrameNode::thread_vars);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SimtVFFrame", SimtVFFrameNode,
                                    StmtFrameObj);

  void ExitWithScope() final {
    StmtFrameObj::ExitWithScope();
    SeqStmt body(NamedScope("simtvf", SeqStmt(stmts),
                            {{"tl.simtvf_scope", IntImm::Int32(1)}}, loc));
    for (size_t axis = thread_vars.size(); axis-- > 0;) {
      String tag = std::string("threadIdx.") + "xyz"[axis];
      body = SeqStmt(RegionStmt(tirx::launch_thread_op(),
                                {StringImm(tag), thread_extents[axis]},
                                {thread_vars[axis]}, {}, body, {}, loc));
    }
    builder::AddToParent(NamedScope("SIMT_VF", body,
                                    VFAnnotations(vf_latency, source_index),
                                    loc),
                         loc);
  }
};

class SimtVFFrame : public StmtFrame {
public:
  explicit SimtVFFrame(ObjectPtr<SimtVFFrameNode> frame)
      : StmtFrame(std::move(frame)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SimtVFFrame, StmtFrame,
                                                SimtVFFrameNode);
};

SimtVFFrame SimtVF(const Array<PrimExpr> &thread_extents, int64_t latency,
                   int64_t source_index) {
  TVM_FFI_CHECK_EQ(thread_extents.size(), 3, ValueError)
      << "SimtVF requires exactly three thread extents";
  TVM_FFI_CHECK(latency >= 0 && source_index >= 0, ValueError)
      << "VF latency and source index must be nonnegative";
  auto frame = make_object<SimtVFFrameNode>();
  for (size_t axis = 0; axis < thread_extents.size(); ++axis) {
    auto extent = GetConstInt(thread_extents[axis]);
    TVM_FFI_CHECK(extent && *extent > 0 && *extent <= INT32_MAX, ValueError)
        << "SimtVF thread extents must be positive int32 constants";
    frame->thread_extents.push_back(IntImm::Int32(*extent));
    frame->thread_vars.push_back(
        PrimVar(std::string("simtvf_t") + "xyz"[axis], PrimType::Int(32)));
  }
  frame->vf_latency = latency;
  frame->source_index = source_index;
  return SimtVFFrame(frame);
}

class SimdVFFrameNode : public StmtFrameObj {
public:
  int64_t vf_latency = 0;
  int64_t source_index = 0;
  static void RegisterReflection() { reflection::ObjectDef<SimdVFFrameNode>(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SimdVFFrame", SimdVFFrameNode,
                                    StmtFrameObj);

  void ExitWithScope() final {
    StmtFrameObj::ExitWithScope();
    SeqStmt body(NamedScope("simdvf", SeqStmt(stmts),
                            {{"tl.simdvf_scope", IntImm::Int32(1)}}, loc));
    builder::AddToParent(NamedScope("SIMD_VF", body,
                                    VFAnnotations(vf_latency, source_index),
                                    loc),
                         loc);
  }
};

class SimdVFFrame : public StmtFrame {
public:
  explicit SimdVFFrame(ObjectPtr<SimdVFFrameNode> frame)
      : StmtFrame(std::move(frame)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SimdVFFrame, StmtFrame,
                                                SimdVFFrameNode);
};

SimdVFFrame SimdVF(int64_t latency, int64_t source_index) {
  TVM_FFI_CHECK(latency >= 0 && source_index >= 0, ValueError)
      << "VF latency and source index must be nonnegative";
  auto frame = make_object<SimdVFFrameNode>();
  frame->vf_latency = latency;
  frame->source_index = source_index;
  return SimdVFFrame(frame);
}

class CubeFrameNode : public StmtFrameObj {
public:
  static void RegisterReflection() { reflection::ObjectDef<CubeFrameNode>(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.CubeFrame", CubeFrameNode,
                                    StmtFrameObj);

  void ExitWithScope() final {
    StmtFrameObj::ExitWithScope();
    builder::AddToParent(NamedScope("CUBE", SeqStmt(stmts), {}, loc), loc);
  }
};

class CubeFrame : public StmtFrame {
public:
  explicit CubeFrame(ObjectPtr<CubeFrameNode> frame)
      : StmtFrame(std::move(frame)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CubeFrame, StmtFrame,
                                                CubeFrameNode);
};

CubeFrame Cube() { return CubeFrame(make_object<CubeFrameNode>()); }

class VectorFrameNode : public StmtFrameObj {
public:
  int vector_count = 1;
  PrimVar sid_var{"sid", PrimType::Int(32)};
  static void RegisterReflection() {
    reflection::ObjectDef<VectorFrameNode>()
        .def_ro("vector_count", &VectorFrameNode::vector_count)
        .def_ro("sid_var", &VectorFrameNode::sid_var);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.VectorFrame", VectorFrameNode,
                                    StmtFrameObj);

  void ExitWithScope() final {
    StmtFrameObj::ExitWithScope();
    PrimExpr extent = IntImm::Int32(vector_count);
    SeqStmt body(RegionStmt(Op::Get("tl.launch_grid"),
                            {StringImm("cthread"), extent}, {sid_var}, {},
                            SeqStmt(stmts), {}, loc));
    builder::AddToParent(
        NamedScope("VECTOR", body, {{"vector_count", extent}}, loc), loc);
  }
};

class VectorFrame : public StmtFrame {
public:
  explicit VectorFrame(ObjectPtr<VectorFrameNode> frame)
      : StmtFrame(std::move(frame)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VectorFrame, StmtFrame,
                                                VectorFrameNode);
};

VectorFrame Vector(int vector_count) {
  TVM_FFI_CHECK(vector_count == 1 || vector_count == 2, ValueError)
      << "Vector core count must be 1 or 2";
  auto frame = make_object<VectorFrameNode>();
  frame->vector_count = vector_count;
  return VectorFrame(frame);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  SimtVFFrameNode::RegisterReflection();
  SimdVFFrameNode::RegisterReflection();
  CubeFrameNode::RegisterReflection();
  VectorFrameNode::RegisterReflection();
  reflection::GlobalDef()
      .def("tl.SimtVF", SimtVF)
      .def("tl.SimdVF", SimdVF)
      .def("tl.Cube", Cube)
      .def("tl.Vector", Vector)
      .def("tl.MixedKernelLaunch", MixedKernelLaunch);
}
} // namespace tvm::tl
