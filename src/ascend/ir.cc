/*!
 * \file tl/ascend/ir.cc
 * \brief Ascend-specific TileLang frontend frames.
 *
 * The NPU dialect owns its execution-region frames and its launch:
 *
 *  - T.SimtVF / T.SimdVF  -> tl.SimtVFFrame / tl.SimdVFFrame
 *  - T.Cube   / T.Vector  -> tl.CubeFrame   / tl.VectorFrame
 *  - T.MixedKernel        -> tl.MixedKernelLaunch, an AIC+AIV launch that
 *                            yields (bx, sid)
 *
 * None of these exist on any other backend, and
 * tilelang.ascend.language.frame binds the first four object types by name.
 * They live here rather than in the shared src/ir.cc so that the neutral
 * frontend carries only the frames every target uses.
 *
 * The mixed launch still produces a tl.KernelLaunchFrame -- the Python dialect
 * binds its launch surface to that object type -- so it reuses the neutral
 * launch frame machinery declared in "ir.h".
 */

#include "ir.h"

#include <tvm/ffi/reflection/creator.h>
#include <tvm/script/ir_builder/tir/ir.h>
#include <tvm/tirx/stmt.h>

#include "support/check.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace script::ir_builder::tirx;
using namespace ffi;

KernelLaunchFrame MixedKernelLaunch(const Array<PrimExpr> &grid_size,
                                    PrimExpr cthread_extent,
                                    const Map<String, Any> &attrs) {
  ObjectPtr<KernelLaunchFrameNode> n =
      tvm::ffi::make_object<KernelLaunchFrameNode>();

  ICHECK_EQ(grid_size.size(), 1) << "MixedKernel only supports 1-D grid";
  const auto *vector_count = cthread_extent.as<IntImmNode>();
  ICHECK(vector_count != nullptr &&
         (vector_count->value == 1 || vector_count->value == 2))
      << "Mixed-kernel vector_count must be the constant integer 1 or 2, got "
      << cthread_extent;

  // Frame 0: bx = blockIdx.x. Emit a target-neutral thread_binding For loop;
  // tl.MaterializeKernelLaunch turns it into a thread_extent AttrStmt.
  ForFrame bx_frame = MakeThreadBindingFrame("bx", "blockIdx.x", grid_size[0]);
  n->grid_vars.push_back(bx_frame->vars[0]);
  n->grid_extents.push_back(grid_size[0]);
  n->frames.push_back(bx_frame);

  // Frame 1: sid = asc_get_sub_block_id() via the Ascend "cthread" binding.
  // Also emitted as a thread_binding For loop; MaterializeKernelLaunch
  // recognizes the "cthread" tag (declared by the Ascend pipeline in
  // launch_dim_tags) and materializes it into a thread_extent AttrStmt. It is a
  // block-level launch dimension of the mixed kernel, so it is reported
  // alongside bx as one of the vars the launch yields
  // (`with T.MixedKernel(...) as (bx, sid)`).
  ForFrame sid_frame = MakeThreadBindingFrame("sid", "cthread", cthread_extent);
  n->grid_vars.push_back(sid_frame->vars[0]);
  n->grid_extents.push_back(cthread_extent);
  n->frames.push_back(sid_frame);

  // Frame 2: thread placeholders, dropped by the Ascend pipeline
  // (lower_thread_binding=false) exactly as for T.Kernel. They exist so a body
  // that references a thread index gets the actionable "no SIMT threads"
  // diagnostic instead of an out-of-range frame lookup.
  ForFrame thread_frame = MakeLaunchThreadFrame();
  n->thread_vars = thread_frame->vars;
  n->frames.push_back(thread_frame);

  // Frame 3: MainBlock with NPU marker
  auto main_block = tvm::script::ir_builder::tirx::Block(DeviceMainBlockName);
  main_block->reads = Array<tvm::tirx::BufferRegion>();
  main_block->writes = Array<tvm::tirx::BufferRegion>();
  Map<String, Any> block_annotations = attrs;
  block_annotations.Set("vector_count", cthread_extent);
  main_block->annotations = block_annotations;
  n->frames.push_back(main_block);

  return KernelLaunchFrame(n);
}

/**
 * Record the current number of alloc_buffers on the parent SBlockFrame.
 * Call this during EnterWithScope of SimtVF/Cube/Vector frames so that
 * ExitWithScope can later steal only the buffers added during this frame's
 * lifetime (i.e. those hoisted by TVM's AllocBuffer).
 */
static size_t RecordParentAllocCount() {
  script::ir_builder::IRBuilder builder =
      script::ir_builder::IRBuilder::Current();
  ffi::Optional<SBlockFrame> opt_parent = builder->FindFrame<SBlockFrame>();
  if (!opt_parent.defined()) {
    return 0;
  }
  return opt_parent.value()->alloc_buffers.size();
}

/**
 * Steal buffers that were added to the parent SBlockFrame during this frame's
 * lifetime.  TVM's AllocBuffer() hoists buffers to the nearest SBlockFrame,
 * but for SimtVF/Cube/Vector frames we want those buffers inside their own
 * Block node instead.
 *
 * @param parent_alloc_count  The alloc_buffers count recorded at
 *                            EnterWithScope time via RecordParentAllocCount().
 */
static ffi::Array<tvm::tirx::Buffer>
StealAllocBuffers(size_t parent_alloc_count) {
  using namespace tvm::tirx;
  script::ir_builder::IRBuilder builder =
      script::ir_builder::IRBuilder::Current();

  ffi::Optional<SBlockFrame> opt_parent = builder->FindFrame<SBlockFrame>();
  if (!opt_parent.defined()) {
    return {};
  }
  SBlockFrame parent = opt_parent.value();

  size_t total = parent->alloc_buffers.size();
  if (parent_alloc_count >= total) {
    return {};
  }

  ffi::Array<Buffer> stolen;
  ffi::Array<Buffer> remaining;
  for (size_t i = 0; i < total; ++i) {
    if (i < parent_alloc_count) {
      remaining.push_back(parent->alloc_buffers[i]);
    } else {
      stolen.push_back(parent->alloc_buffers[i]);
    }
  }

  parent->alloc_buffers = remaining;
  return stolen;
}

class SimtVFFrameNode : public TIRFrameNode {
public:
  Array<PrimExpr> thread_extents;
  Array<Var> thread_vars;
  size_t parent_alloc_count_{0};
  int64_t vf_latency_{0};
  int64_t source_index_{0};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SimtVFFrameNode>()
        .def_ro("thread_extents", &SimtVFFrameNode::thread_extents)
        .def_ro("thread_vars", &SimtVFFrameNode::thread_vars);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SimtVFFrame", SimtVFFrameNode,
                                    TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    TIRFrameNode::EnterWithScope();
    parent_alloc_count_ = RecordParentAllocCount();
  }

  TVM_DLL void ExitWithScope() final {
    using namespace tvm::tirx;
    TIRFrameNode::ExitWithScope();
    auto stolen_bufs = StealAllocBuffers(parent_alloc_count_);

    auto make_thread_extent = [](const Var &var, const String &thread_tag,
                                 const PrimExpr &extent, Stmt inner) {
      DataType dtype = extent.dtype();
      IterVar iv(Range::FromMinExtent(make_zero(dtype), extent), var,
                 IterVarType::kThreadIndex, thread_tag);
      return AttrStmt(iv, tirx::attr::thread_extent, extent, inner);
    };

    ICHECK_EQ(thread_extents.size(), 3)
        << "SimtVF requires exactly 3 thread extents [tx, ty, tz]";
    ICHECK_EQ(thread_vars.size(), 3)
        << "SimtVF requires exactly 3 thread vars [tx, ty, tz]";

    Stmt body = tvm::tirx::SeqStmt::Flatten(stmts);

    // SimtVF scope marker: placed inside thread extents (innermost),
    // so that StorageRewrite attaches local.fragment allocations here.
    // This keeps fragments inside the VF function as local variables.
    body = AttrStmt(StringImm("simtvf"), "tl.simtvf_scope",
                    IntImm(DataType::Int(32), 1), std::move(body));

    body = make_thread_extent(thread_vars[2], "threadIdx.z", thread_extents[2],
                              std::move(body));
    body = make_thread_extent(thread_vars[1], "threadIdx.y", thread_extents[1],
                              std::move(body));
    body = make_thread_extent(thread_vars[0], "threadIdx.x", thread_extents[0],
                              std::move(body));

    ffi::Map<ffi::String, ffi::Any> simtvf_annotations;
    simtvf_annotations.Set("tl.vf_source_index",
                           IntImm(DataType::Int(64), source_index_));
    if (vf_latency_ > 0) {
      simtvf_annotations.Set("tl.vf_latency",
                             IntImm(DataType::Int(64), vf_latency_));
    }
    Stmt stmt = SBlock({}, {}, {}, "SIMT_VF", body, std::nullopt, stolen_bufs,
                       {}, simtvf_annotations);

    script::ir_builder::IRBuilder builder =
        script::ir_builder::IRBuilder::Current();
    if (builder->frames.empty()) {
      ICHECK(!builder->result.defined())
          << "ValueError: Builder.result has already been set";
      builder->result = stmt;
    } else if (const auto *tir_frame =
                   builder->frames.back().as<TIRFrameNode>()) {
      ffi::GetRef<TIRFrame>(tir_frame)->stmts.push_back(stmt);
    } else {
      LOG(FATAL) << "TypeError: Unsupported frame type: "
                 << builder->frames.back();
    }
  }
};

class SimtVFFrame : public TIRFrame {
public:
  explicit SimtVFFrame(ObjectPtr<SimtVFFrameNode> data)
      : TIRFrame(::tvm::ffi::UnsafeInit{}) {
    ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SimtVFFrame, TIRFrame,
                                                SimtVFFrameNode);
};

SimtVFFrame SimtVF(const Array<PrimExpr> &thread_extents, int64_t vf_latency,
                   int64_t source_index) {
  ICHECK_EQ(thread_extents.size(), 3)
      << "SimtVF requires exactly 3 thread extents [tx, ty, tz]";
  ObjectPtr<SimtVFFrameNode> n = tvm::ffi::make_object<SimtVFFrameNode>();
  n->thread_extents = thread_extents;
  DataType dtype = DataType::Int(32);
  n->thread_vars = {
      Var("simtvf_tx", dtype),
      Var("simtvf_ty", dtype),
      Var("simtvf_tz", dtype),
  };
  n->vf_latency_ = vf_latency;
  n->source_index_ = source_index;
  return SimtVFFrame(n);
}

class SimdVFFrameNode : public TIRFrameNode {
public:
  size_t parent_alloc_count_{0};
  int64_t vf_latency_{0};
  int64_t source_index_{0};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SimdVFFrameNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SimdVFFrame", SimdVFFrameNode,
                                    TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    TIRFrameNode::EnterWithScope();
    parent_alloc_count_ = RecordParentAllocCount();
  }

  TVM_DLL void ExitWithScope() final {
    using namespace tvm::tirx;
    TIRFrameNode::ExitWithScope();
    auto stolen_bufs = StealAllocBuffers(parent_alloc_count_);

    Stmt body = tvm::tirx::SeqStmt::Flatten(stmts);

    // SimdVF scope marker: placed inside the Block body so that
    // fragment allocations (stolen into alloc_buffers) stay inside
    // the helper function during codegen.
    body = AttrStmt(StringImm("simdvf"), "tl.simdvf_scope",
                    IntImm(DataType::Int(32), 1), std::move(body));

    // No thread extent AttrStmts — SimdVF has no thread dimensions.

    ffi::Map<ffi::String, ffi::Any> simdvf_annotations;
    simdvf_annotations.Set("tl.vf_source_index",
                           IntImm(DataType::Int(64), source_index_));
    if (vf_latency_ > 0) {
      simdvf_annotations.Set("tl.vf_latency",
                             IntImm(DataType::Int(64), vf_latency_));
    }
    Stmt stmt = SBlock({}, {}, {}, "SIMD_VF", body, std::nullopt, stolen_bufs,
                       {}, simdvf_annotations);

    script::ir_builder::IRBuilder builder =
        script::ir_builder::IRBuilder::Current();
    if (builder->frames.empty()) {
      ICHECK(!builder->result.defined())
          << "ValueError: Builder.result has already been set";
      builder->result = stmt;
    } else if (const auto *tir_frame =
                   builder->frames.back().as<TIRFrameNode>()) {
      ffi::GetRef<TIRFrame>(tir_frame)->stmts.push_back(stmt);
    } else {
      LOG(FATAL) << "TypeError: Unsupported frame type: "
                 << builder->frames.back();
    }
  }
};

class SimdVFFrame : public TIRFrame {
public:
  explicit SimdVFFrame(ObjectPtr<SimdVFFrameNode> data)
      : TIRFrame(::tvm::ffi::UnsafeInit{}) {
    ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SimdVFFrame, TIRFrame,
                                                SimdVFFrameNode);
};

SimdVFFrame SimdVF(int64_t vf_latency, int64_t source_index) {
  ObjectPtr<SimdVFFrameNode> n = tvm::ffi::make_object<SimdVFFrameNode>();
  n->vf_latency_ = vf_latency;
  n->source_index_ = source_index;
  return SimdVFFrame(n);
}

class CubeFrameNode : public TIRFrameNode {
public:
  size_t parent_alloc_count_{0};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CubeFrameNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.CubeFrame", CubeFrameNode,
                                    TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    TIRFrameNode::EnterWithScope();
    parent_alloc_count_ = RecordParentAllocCount();
  }

  TVM_DLL void ExitWithScope() final {
    using namespace tvm::tirx;
    TIRFrameNode::ExitWithScope();
    auto stolen_bufs = StealAllocBuffers(parent_alloc_count_);

    Stmt body = tvm::tirx::SeqStmt::Flatten(stmts);
    Stmt stmt =
        SBlock({}, {}, {}, "CUBE", body, std::nullopt, stolen_bufs, {}, {});

    script::ir_builder::IRBuilder builder =
        script::ir_builder::IRBuilder::Current();
    if (builder->frames.empty()) {
      ICHECK(!builder->result.defined())
          << "ValueError: Builder.result has already been set";
      builder->result = stmt;
    } else if (const auto *tir_frame =
                   builder->frames.back().as<TIRFrameNode>()) {
      ffi::GetRef<TIRFrame>(tir_frame)->stmts.push_back(stmt);
    } else {
      LOG(FATAL) << "TypeError: Unsupported frame type: "
                 << builder->frames.back();
    }
  }
};

class CubeFrame : public TIRFrame {
public:
  explicit CubeFrame(ObjectPtr<CubeFrameNode> data)
      : TIRFrame(::tvm::ffi::UnsafeInit{}) {
    ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CubeFrame, TIRFrame,
                                                CubeFrameNode);
};

CubeFrame Cube() { return CubeFrame(tvm::ffi::make_object<CubeFrameNode>()); }

class VectorFrameNode : public TIRFrameNode {
public:
  int vector_count;
  tvm::tirx::Var sid_var;
  size_t parent_alloc_count_{0};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VectorFrameNode>()
        .def_ro("vector_count", &VectorFrameNode::vector_count)
        .def_ro("sid_var", &VectorFrameNode::sid_var);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.VectorFrame", VectorFrameNode,
                                    TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    TIRFrameNode::EnterWithScope();
    parent_alloc_count_ = RecordParentAllocCount();
  }

  TVM_DLL void ExitWithScope() final {
    using namespace tvm::tirx;
    TIRFrameNode::ExitWithScope();
    auto stolen_bufs = StealAllocBuffers(parent_alloc_count_);

    Stmt body = tvm::tirx::SeqStmt::Flatten(stmts);

    DataType dtype = DataType::Int(32);
    PrimExpr extent = IntImm(dtype, vector_count);
    IterVar iv(Range::FromMinExtent(make_zero(dtype), extent), sid_var,
               IterVarType::kThreadIndex, "cthread");
    body = AttrStmt(iv, tirx::attr::thread_extent, extent, body);

    Map<String, ObjectRef> annotations;
    annotations.Set("vector_count", IntImm(dtype, vector_count));
    Stmt stmt = SBlock({}, {}, {}, "VECTOR", body, std::nullopt, stolen_bufs,
                       {}, annotations);

    script::ir_builder::IRBuilder builder =
        script::ir_builder::IRBuilder::Current();
    if (builder->frames.empty()) {
      ICHECK(!builder->result.defined())
          << "ValueError: Builder.result has already been set";
      builder->result = stmt;
    } else if (const auto *tir_frame =
                   builder->frames.back().as<TIRFrameNode>()) {
      ffi::GetRef<TIRFrame>(tir_frame)->stmts.push_back(stmt);
    } else {
      LOG(FATAL) << "TypeError: Unsupported frame type: "
                 << builder->frames.back();
    }
  }
};

class VectorFrame : public TIRFrame {
public:
  explicit VectorFrame(ObjectPtr<VectorFrameNode> data)
      : TIRFrame(::tvm::ffi::UnsafeInit{}) {
    ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VectorFrame, TIRFrame,
                                                VectorFrameNode);
};

VectorFrame Vector(int vector_count) {
  ICHECK(vector_count >= 1 && vector_count <= 2)
      << "Vector core count must be 1 or 2, got " << vector_count;
  auto n = tvm::ffi::make_object<VectorFrameNode>();
  n->vector_count = vector_count;
  n->sid_var = tvm::tirx::Var("sid", DataType::Int(32));
  return VectorFrame(n);
}

// Registered here, with the rest of the Ascend surface. This is compiled for
// every target (TILE_LANG_ASCEND_ALWAYS_SRCS in src/ascend/CMakeLists.txt):
// tilelang/__init__.py imports tilelang.ascend unconditionally, and
// tilelang.ascend.language.frame binds the four frame object types at import
// time, so they have to exist even in a build without the Ascend backend.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("tl.SimtVF", SimtVF)
      .def("tl.SimdVF", SimdVF)
      .def("tl.Cube", Cube)
      .def("tl.Vector", Vector)
      .def("tl.MixedKernelLaunch", MixedKernelLaunch);
  SimtVFFrameNode::RegisterReflection();
  SimdVFFrameNode::RegisterReflection();
  CubeFrameNode::RegisterReflection();
  VectorFrameNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
