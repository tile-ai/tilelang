/*!
 * \file tl/ir.cc
 * \brief Extension for the tvm script frontend.
 *
 */

#include "./transform/common/attr.h"
#include "./transform/common/warp_specialize.h"
#include "op/builtin.h"
#include "support/check.h"
#include <tvm/ffi/reflection/creator.h>
#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt.h>

#include <tvm/arith/analyzer.h>
#include <tvm/script/ir_builder/tir/ir.h>
#include <tvm/tirx/analysis.h>

#include <utility>

namespace tvm {
namespace tl {

using namespace script::ir_builder::tirx;
using namespace ffi;

// Build a ForFrame that emits a target-neutral kThreadBinding loop for one
// grid (program index) axis of a kernel launch. The launch nest is
// materialized into the target-specific form (thread_extent AttrStmt on GPU,
// serial For on CPU) by the tl.MaterializeKernelLaunch pass once the Target is
// known at compile time.
static ForFrame MakeThreadBindingFrame(const std::string &name,
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

ForFrame ParallelFor(const Array<PrimExpr> &extents,
                     const Map<String, Any> &annotations) {
  using namespace tvm::tirx;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  for (const auto &extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", extent.dtype()));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop =
      [annotations](const Array<Var> &vars, const Array<Range> &doms,
                    const Array<Optional<PrimExpr>> &steps, Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      Optional<PrimExpr> step =
          i < steps.size() ? steps[i] : Optional<PrimExpr>(std::nullopt);
      // Only attach annotations to the outermost parallel loop.
      // Rationale: In TileLang's design, inner loops cannot govern or annotate
      // their outer loops, while the outermost loop can manage and transform
      // the entire nested region. Placing the layout on the outermost loop
      // lets lowering/validators reason about and rewrite the whole nest.
      // Layout annotations (like parallel_loop_layout) and other hints are
      // read from the outermost loop.
      Map<String, Any> loop_annotations;
      if (i == 0) {
        loop_annotations = annotations;
      }
      body = For(var, dom->min, dom->extent, ForKind::kParallel, body,
                 /*thread_binding=*/std::nullopt,
                 /*annotations=*/loop_annotations,
                 /*step=*/step);
    }
    return body;
  };
  return ForFrame(n);
}

ForFrame PipelinedFor(PrimExpr start, const PrimExpr &stop, int num_stages,
                      const Array<PrimExpr> &order,
                      const Array<PrimExpr> &stages,
                      const Map<String, Any> &annotations) {
  using namespace tvm::tirx;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  DataType dtype = stop.dtype();
  n->vars.push_back(Var("v", dtype));
  n->doms.push_back(Range(std::move(start), stop));
  n->f_make_for_loop = [=](const Array<Var> &vars, const Array<Range> &doms,
                           const Array<Optional<PrimExpr>> &steps,
                           Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    ICHECK(n == 1);
    Map<String, Any> anno = annotations;
    if (num_stages > 0)
      anno.Set("num_stages", PrimExpr(num_stages));
    if (!order.empty())
      anno.Set("tl_pipeline_order", order);
    if (!stages.empty())
      anno.Set("tl_pipeline_stage", stages);
    Optional<PrimExpr> step =
        !steps.empty() ? steps[0] : Optional<PrimExpr>(std::nullopt);
    body = For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kSerial, body,
               /*thread_binding=*/std::nullopt, /*annotations=*/anno,
               /*step=*/step);
    return body;
  };
  return ForFrame(n);
}

ForFrame PersistentFor(const Array<PrimExpr> &domain, const PrimExpr &wave_size,
                       const PrimExpr &index, PrimExpr group_size) {
  using namespace tvm::tirx;
  ICHECK(!domain.empty());
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(domain.size());
  n->doms.reserve(domain.size());
  PrimExpr domain_size = domain[0];
  for (int i = 1; i < domain.size(); i++) {
    domain_size *= domain[i];
  }

  PrimExpr last_extent = domain[domain.size() - 1];
  group_size =
      max(make_const(group_size.dtype(), 1), min(group_size, last_extent));
  Array<PrimExpr> grouped_domain;
  grouped_domain.push_back(ceildiv(last_extent, group_size));
  for (int i = 0; i < domain.size() - 1; ++i) {
    grouped_domain.push_back(domain[i]);
  }
  grouped_domain.push_back(group_size);
  PrimExpr padded_domain_size = grouped_domain[0];
  for (int i = 1; i < grouped_domain.size(); ++i) {
    padded_domain_size *= grouped_domain[i];
  }

  auto waves = ceildiv(padded_domain_size, wave_size);
  auto loop_var = Var("w", waves.dtype());
  Array<Var> coord_vars;

  for (int i = 0; i < domain.size(); ++i) {
    DataType dtype = domain[i].dtype();
    Var coord("v" + std::to_string(i), dtype);
    coord_vars.push_back(coord);
    n->vars.push_back(coord);
    n->doms.push_back(Range(make_const(dtype, 0), domain[i]));
  }

  n->f_make_for_loop = [=](const Array<Var> &vars, const Array<Range> &doms,
                           const Array<Optional<PrimExpr>> &steps,
                           Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    Map<String, Any> anno;
    Array<PrimExpr> idxs(grouped_domain.size(), PrimExpr());
    PrimExpr rem = loop_var * wave_size + index;

    for (int i = grouped_domain.size() - 1; i >= 1; --i) {
      idxs.Set(i, truncmod(rem, grouped_domain[i]));
      rem = truncdiv(rem, grouped_domain[i]);
    }
    idxs.Set(0, rem);
    PrimExpr last_coord =
        idxs[0] * group_size + idxs[grouped_domain.size() - 1];
    PrimExpr in_range = last_coord < domain[domain.size() - 1];
    auto out_if = tvm::tirx::IfThenElse(
        padded_domain_size <= (loop_var * wave_size + index),
        tvm::tirx::Evaluate(
            tvm::tirx::Call(DataType::Handle(), tvm::tl::loop_break(), {})),
        Stmt());
    Stmt guarded_body = tvm::tirx::IfThenElse(in_range, body, Stmt());

    arith::Analyzer analyzer;
    Stmt new_body = guarded_body;
    if (analyzer.CanProveGreaterEqual(waves, 2)) {
      new_body = SeqStmt({out_if, guarded_body});
    }
    Optional<PrimExpr> step =
        !steps.empty() ? steps[0] : Optional<PrimExpr>(std::nullopt);
    Stmt outer = For(loop_var, 0, waves, ForKind::kSerial, new_body,
                     /*thread_binding=*/std::nullopt, /*annotations=*/anno,
                     /*step=*/step);
    for (int i = 0; i < vars.size() - 1; ++i) {
      outer = SeqStmt({tirx::Bind(vars[i], idxs[i + 1]), outer});
    }
    outer = SeqStmt({tirx::Bind(vars[vars.size() - 1], last_coord), outer});
    return outer;
  };

  return ForFrame(n);
}

// Build a frame whose exit prefixes the body with
// `tx = tl.launch_thread_idx(0); ty = ...; tz = ...` Bind statements. The
// launch nest is traced before the Target is known, so the thread indices are
// only placeholders here: the Vars keep their identity through
// tl.MaterializeKernelLaunch, which rebinds them as threadIdx.* thread_extent
// scopes on SIMT backends and drops them elsewhere.
static ForFrame MakeLaunchThreadFrame() {
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

KernelLaunchFrame KernelLaunch(const Array<PrimExpr> &grid_size,
                               const Optional<Array<PrimExpr>> &block_size_opt,
                               const Map<String, Any> &attrs) {
  ObjectPtr<KernelLaunchFrameNode> n = make_object<KernelLaunchFrameNode>();

  ICHECK(grid_size.size() <= 3);

  static const char *kBlockVarNames[3] = {"bx", "by", "bz"};
  static const char *kBlockTags[3] = {"blockIdx.x", "blockIdx.y", "blockIdx.z"};

  for (size_t i = 0; i < grid_size.size(); i++) {
    ForFrame frame =
        MakeThreadBindingFrame(kBlockVarNames[i], kBlockTags[i], grid_size[i]);
    n->grid_vars.push_back(frame->vars[0]);
    n->grid_extents.push_back(grid_size[i]);
    n->frames.push_back(frame);
  }
  // Thread placeholders are always emitted so the body may reference a thread
  // index regardless of whether threads= was given; the backend decides what
  // they mean.
  ForFrame thread_frame = MakeLaunchThreadFrame();
  n->thread_vars = thread_frame->vars;
  n->frames.push_back(thread_frame);

  Map<String, Any> block_annotations =
      attrs.defined() ? attrs : Map<String, Any>{};
  if (block_size_opt.defined()) {
    Array<PrimExpr> block_size = block_size_opt.value();
    ICHECK(block_size.size() <= 3);
    while (block_size.size() < 3) {
      block_size.push_back(IntImm(DataType::Int(32), 1));
    }
    n->thread_extents = block_size;
    block_annotations.Set(attr::kLaunchThreads, block_size);
  }

  auto empty_block = tvm::script::ir_builder::tirx::Block(DeviceMainBlockName);
  empty_block->reads = Array<tvm::tirx::BufferRegion>();
  empty_block->writes = Array<tvm::tirx::BufferRegion>();
  empty_block->annotations = block_annotations;
  n->frames.push_back(empty_block);

  return KernelLaunchFrame(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("tl.Parallel", ParallelFor)
      .def("tl.Pipelined", PipelinedFor)
      .def("tl.Persistent", PersistentFor)
      .def("tl.KernelLaunch", KernelLaunch);
}

class WarpSpecializeFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<WarpSpecializeFrameNode>().def_ro(
        "frames", &WarpSpecializeFrameNode::frames);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WarpSpecializeFrame",
                                    WarpSpecializeFrameNode, TIRFrameNode);

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

class WarpSpecializeFrame : public TIRFrame {
public:
  explicit WarpSpecializeFrame(ObjectPtr<WarpSpecializeFrameNode> data)
      : TIRFrame(UnsafeInit{}) {
    ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(WarpSpecializeFrame, TIRFrame,
                                                WarpSpecializeFrameNode);
};

WarpSpecializeFrame WarpSpecialize(const Array<IntImm> &warp_group_ids,
                                   const PrimExpr &thread_idx,
                                   int warp_group_size = 128) {
  ObjectPtr<WarpSpecializeFrameNode> n = make_object<WarpSpecializeFrameNode>();
  PrimExpr condition;
  std::vector<int> warp_groups;
  warp_groups.reserve(warp_group_ids.size());
  for (int i = 0; i < warp_group_ids.size(); i++) {
    warp_groups.push_back(Downcast<IntImm>(warp_group_ids[i])->value);
  }
  std::sort(warp_groups.begin(), warp_groups.end());

  // Merge consecutive groups
  std::vector<std::pair<int, int>> merged;
  for (int group : warp_groups) {
    if (merged.empty() || group != merged.back().second) {
      merged.emplace_back(group, group + 1);
    } else {
      merged.back().second = group + 1;
    }
  }

  for (const auto &[start, end] : merged) {
    PrimExpr min_bound = IntImm(thread_idx.dtype(), start) * warp_group_size;
    PrimExpr max_bound = IntImm(thread_idx.dtype(), end) * warp_group_size;
    PrimExpr range_cond = (thread_idx >= min_bound) && (thread_idx < max_bound);

    if (condition.defined()) {
      condition = tirx::Or(condition, range_cond);
    } else {
      condition = range_cond;
    }
  }
  IfFrame if_frame = If(condition);
  AttrFrame attr_frame = Attr(Integer(0), "warp_specialize", Integer(1));
  n->frames.push_back(if_frame);
  n->frames.push_back(Then());
  n->frames.push_back(attr_frame);
  return WarpSpecializeFrame(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("tl.WarpSpecialize", WarpSpecialize)
      .def("tl.SideEffect", tirx::SideEffect);
  KernelLaunchFrameNode::RegisterReflection();
  WarpSpecializeFrameNode::RegisterReflection();
}

// ---------------------------------------------------------------------------
// Warp-specialization schedule objects (transform/common/warp_specialize.h).
// These are frontend IR constructs, not transformations: T.WSSchedule and
// friends are built by the tracer and attached as a block annotation for the
// MaterializeWSSchedule pass to consume.
// ---------------------------------------------------------------------------

WSRole::WSRole(String name, int64_t warp_lo, int64_t warp_hi,
               int64_t max_nreg) {
  auto n = make_object<WSRoleNode>();
  n->name = std::move(name);
  n->warp_lo = warp_lo;
  n->warp_hi = warp_hi;
  n->max_nreg = max_nreg;
  data_ = std::move(n);
}

void WSRoleNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<WSRoleNode>()
      .def_ro("name", &WSRoleNode::name)
      .def_ro("warp_lo", &WSRoleNode::warp_lo)
      .def_ro("warp_hi", &WSRoleNode::warp_hi)
      .def_ro("max_nreg", &WSRoleNode::max_nreg);
}

WSPipeline::WSPipeline(String name, Array<tirx::Buffer> buffers,
                       int64_t depth) {
  auto n = make_object<WSPipelineNode>();
  n->name = std::move(name);
  n->buffers = std::move(buffers);
  n->depth = depth;
  data_ = std::move(n);
}

void WSPipelineNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<WSPipelineNode>()
      .def_ro("name", &WSPipelineNode::name)
      .def_ro("buffers", &WSPipelineNode::buffers)
      .def_ro("depth", &WSPipelineNode::depth);
}

void WSInstrNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<WSInstrNode>();
}

WSOpRef::WSOpRef(String id) {
  auto n = make_object<WSOpRefNode>();
  n->id = std::move(id);
  data_ = std::move(n);
}

void WSOpRefNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<WSOpRefNode>().def_ro("id", &WSOpRefNode::id);
}

WSSync::WSSync(WSSyncKind kind, String pipeline, int64_t stage) {
  auto n = make_object<WSSyncNode>();
  n->kind = std::move(kind);
  n->pipeline = std::move(pipeline);
  n->stage = stage;
  data_ = std::move(n);
}

void WSSyncNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<WSSyncNode>()
      .def_ro("kind", &WSSyncNode::kind)
      .def_ro("pipeline", &WSSyncNode::pipeline)
      .def_ro("stage", &WSSyncNode::stage);
}

WSScope::WSScope(String id, Map<String, Array<WSInstr>> bodies) {
  auto n = make_object<WSScopeNode>();
  n->id = std::move(id);
  n->bodies = std::move(bodies);
  data_ = std::move(n);
}

void WSScopeNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<WSScopeNode>()
      .def_ro("id", &WSScopeNode::id)
      .def_ro("bodies", &WSScopeNode::bodies);
}

WSSchedule::WSSchedule(int64_t num_warps, Array<WSRole> roles,
                       Array<WSPipeline> pipelines, Array<WSScope> scopes) {
  auto n = make_object<WSScheduleNode>();
  n->num_warps = num_warps;
  n->roles = std::move(roles);
  n->pipelines = std::move(pipelines);
  n->scopes = std::move(scopes);
  data_ = std::move(n);
}

void WSScheduleNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<WSScheduleNode>()
      .def_ro("num_warps", &WSScheduleNode::num_warps)
      .def_ro("roles", &WSScheduleNode::roles)
      .def_ro("pipelines", &WSScheduleNode::pipelines)
      .def_ro("scopes", &WSScheduleNode::scopes);
}

// Register the sync-kind enum and its variants. Declaration order fixes the
// dense ordinals (0..3) consumed by WSSyncKind::CanonicalOrdinal().
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  // EnumObj subclasses have no __ffi_init__; allocate via init(false).
  refl::ObjectDef<WSSyncKindObj>(
      refl::init(false)); // NOLINT(bugprone-unused-raii)
  refl::TypeAttrDef<WSSyncKindObj>().def(
      refl::type_attr::kConvert,
      &refl::details::FFIConvertFromAnyViewToObjectRef<WSSyncKind>);
  refl::EnumDef<WSSyncKindObj>("PRODUCER_ACQUIRE"); // ordinal 0
  refl::EnumDef<WSSyncKindObj>("PRODUCER_COMMIT");  // ordinal 1
  refl::EnumDef<WSSyncKindObj>("CONSUMER_WAIT");    // ordinal 2
  refl::EnumDef<WSSyncKindObj>("CONSUMER_RELEASE"); // ordinal 3
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  WSRoleNode::RegisterReflection();
  WSPipelineNode::RegisterReflection();
  WSInstrNode::RegisterReflection();
  WSOpRefNode::RegisterReflection();
  WSSyncNode::RegisterReflection();
  WSScopeNode::RegisterReflection();
  WSScheduleNode::RegisterReflection();
  refl::GlobalDef()
      .def("tl.WSRole",
           [](String name, int64_t warp_lo, int64_t warp_hi, int64_t max_nreg) {
             return WSRole(std::move(name), warp_lo, warp_hi, max_nreg);
           })
      .def("tl.WSPipeline",
           [](String name, Array<tirx::Buffer> buffers, int64_t depth) {
             return WSPipeline(std::move(name), std::move(buffers), depth);
           })
      .def("tl.WSOpRef", [](String id) { return WSOpRef(std::move(id)); })
      .def("tl.WSSync",
           [](WSSyncKind kind, String pipeline, int64_t stage) {
             return WSSync(std::move(kind), std::move(pipeline), stage);
           })
      .def("tl.WSScope",
           [](String id, Map<String, Array<WSInstr>> bodies) {
             return WSScope(std::move(id), std::move(bodies));
           })
      .def("tl.WSSchedule",
           [](int64_t num_warps, Array<WSRole> roles,
              Array<WSPipeline> pipelines, Array<WSScope> scopes) {
             return WSSchedule(num_warps, std::move(roles),
                               std::move(pipelines), std::move(scopes));
           });
}

} // namespace tl
} // namespace tvm
