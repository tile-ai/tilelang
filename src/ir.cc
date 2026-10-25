#include "ir.h"
#include "op/builtin.h"
#include "transform/common/attr.h"
#include "transform/common/warp_specialize.h"

#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/creator.h>
#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/ir/analysis.h>
#include <tvm/sym/analyzer.h>

#include <algorithm>
#include <vector>

namespace tvm::tl {

using namespace ffi;
namespace builder = script::ir_builder::ir;

namespace {

Array<Var> EmptyBodyParams(const CallNode *) { return {}; }

Array<Var> GridBodyParams(const CallNode *call) {
  TVM_FFI_CHECK(!call->args[0].as_or_throw<StringImm>()->value.empty(),
                ValueError)
      << "A launch dimension requires a thread tag";
  return {PrimVar("", call->args[1].as_or_throw<IntExpr>().ty())};
}

void ValidateFrontendRegion(const RegionStmtNode *region) {
  TVM_FFI_CHECK(region->result_vars.empty(), ValueError)
      << region->op->name << " cannot return values";
}

PrimType IndexType(const Array<PrimExpr> &expressions) {
  int bits = 32;
  for (const PrimExpr &expression : expressions) {
    PrimType type = expression.ty();
    TVM_FFI_CHECK(type.IsScalar() &&
                      (type.MatchesCode(kDLInt) || type.MatchesCode(kDLUInt)) &&
                      type.bits() > 1,
                  TypeError)
        << "Loop extents must be scalar integers";
    bits = std::max(bits, type.bits());
  }
  return PrimType::Int(bits);
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.layout_hint",
        "Attach a TileLang layout to a previously declared tensor.")
      .set_attr(tvm::op_attr::kAttrsMayContainExpr, true)
      .signature(sig::arg<tirx::TensorVar>("tensor"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
  OpDef("tl.safe_value_hint", "Attach an out-of-bounds fill value to a tensor.")
      .signature(sig::arg<tirx::TensorVar>("tensor"),
                 sig::arg<PrimExpr>("value"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
  OpDef("tl.scope", "TileLang lexical scope and lowering annotations.")
      .signature()
      .set_native_fn_attr<&EmptyBodyParams>(tvm::op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateFrontendRegion>(
          tvm::op_attr::kRegionValidate);
  OpDef("tl.launch_grid", "Target-neutral program-index launch dimension.")
      .signature(sig::arg<StringImm>("tag"), sig::arg<IntExpr>("extent"))
      .set_native_fn_attr<&GridBodyParams>(tvm::op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateFrontendRegion>(
          tvm::op_attr::kRegionValidate);
}

RegionFrame MakeNamedScope(const String &name, Map<String, Any> annotations) {
  annotations.Set("name_hint", name);
  return builder::Region(Op::Get("tl.scope"), {}, std::nullopt,
                         DictAttrs(annotations));
}

void AnnotateScope(Map<String, Any> annotations) {
  if (annotations.size() == 1 && annotations.count("tl.ws_pipeline_depth")) {
    const auto &frames = script::ir_builder::IRBuilder::Current()->frames;
    for (auto iterator = frames.rbegin(); iterator != frames.rend();
         ++iterator) {
      if (auto loop_frame = (*iterator).as<ForFrame>()) {
        auto original = loop_frame.value()->f_make_for_loop;
        loop_frame.value()->f_make_for_loop =
            [original, annotations](Array<Var> variables, Array<Range> domains,
                                    Array<Optional<PrimExpr>> steps,
                                    SeqStmt body, Location location) -> Stmt {
          For loop = original(variables, domains, steps, body, location)
                         .as_or_throw<For>();
          auto options = loop->annotations;
          auto depths = options.Get("tl.ws_pipeline_depth")
                            .value_or(Map<Any, Any>())
                            .cast<Map<Any, Any>>();
          for (const auto &[tensor, depth] :
               annotations.at("tl.ws_pipeline_depth").cast<Map<Any, Any>>())
            depths.Set(tensor, depth);
          options.Set("tl.ws_pipeline_depth", depths);
          loop.CopyOnWrite()->annotations = options;
          return loop;
        };
        return;
      }
      if ((*iterator).as<RegionFrame>())
        break;
    }
  }
  auto frame =
      script::ir_builder::IRBuilder::Current()->FindFrame<RegionFrame>();
  TVM_FFI_CHECK(frame, ValueError)
      << "Scope annotations require an enclosing lexical region";
  auto attributes = frame.value()->attrs->dict;
  for (const auto &[name, value] : annotations) {
    auto existing = attributes.Get(name);
    auto incoming_map = value.as<Map<Any, Any>>();
    if (existing && incoming_map) {
      auto previous = existing.value().cast<Map<Any, Any>>();
      for (const auto &[key, entry] : incoming_map.value())
        previous.Set(key, entry);
      attributes.Set(name, previous);
    } else {
      attributes.Set(name, value);
    }
  }
  frame.value()->attrs = DictAttrs(attributes);
}

ForFrame MakeThreadBindingFrame(const std::string &name,
                                const String &thread_tag,
                                const PrimExpr &extent,
                                const Map<String, Any> &annotations) {
  PrimType type = IndexType({extent});
  auto node = make_object<ForFrameObj>();
  node->vars.push_back(PrimVar(name, type));
  node->doms.push_back(Range(MakeConst(type, 0), prim::cast(extent, type)));
  node->f_make_for_loop =
      [thread_tag, annotations](Array<Var> variables, Array<Range> domains,
                                Array<Optional<PrimExpr>>, SeqStmt body,
                                Location loc) -> Stmt {
    return RegionStmt(Op::Get("tl.launch_grid"),
                      {StringImm(thread_tag), domains[0]->extent}, variables,
                      DictAttrs(annotations), body, {}, loc);
  };
  return ForFrame(node);
}

ForFrame MakeLaunchThreadFrame() {
  auto node = make_object<ForFrameObj>();
  for (const char *name : {"tx", "ty", "tz"}) {
    node->vars.push_back(PrimVar(name, PrimType::Int(32)));
    node->doms.push_back(Range(PrimExpr(0), PrimExpr(1)));
  }
  node->f_make_for_loop = [](Array<Var> variables, Array<Range>,
                             Array<Optional<PrimExpr>>, SeqStmt body,
                             Location loc) -> Stmt {
    Array<Stmt> statements;
    for (size_t axis = 0; axis < variables.size(); ++axis) {
      statements.push_back(Bind(variables[axis],
                                Call(variables[axis]->ty, launch_thread_idx(),
                                     {PrimExpr(static_cast<int32_t>(axis))},
                                     std::nullopt, {}, loc),
                                loc));
    }
    statements.push_back(body);
    return SeqStmt(statements, loc);
  };
  return ForFrame(node);
}

ForFrame ParallelFor(const Array<PrimExpr> &extents,
                     const Map<String, Any> &annotations) {
  TVM_FFI_CHECK(!extents.empty(), ValueError)
      << "T.Parallel requires at least one extent";
  auto node = make_object<ForFrameObj>();
  for (size_t axis = 0; axis < extents.size(); ++axis) {
    PrimType type = IndexType({extents[axis]});
    node->vars.push_back(PrimVar("parallel_" + std::to_string(axis), type));
    node->doms.push_back(
        Range(MakeConst(type, 0), prim::cast(extents[axis], type)));
  }
  node->f_make_for_loop = [annotations](Array<Var> variables,
                                        Array<Range> domains,
                                        Array<Optional<PrimExpr>> steps,
                                        SeqStmt body, Location loc) -> Stmt {
    for (size_t axis = variables.size(); axis-- > 0;) {
      PrimVar variable = variables[axis].as_or_throw<PrimVar>();
      PrimExpr step = axis < steps.size() && steps[axis].has_value()
                          ? prim::cast(steps[axis].value(), variable.ty())
                          : MakeConst(variable.ty(), 1);
      body = SeqStmt(For(variable, domains[axis]->begin,
                         domains[axis]->begin + domains[axis]->extent, step,
                         ForKind::kParallel, body,
                         axis == 0 ? annotations : Map<String, Any>{}, loc));
    }
    return body;
  };
  return ForFrame(node);
}

ForFrame PipelinedFor(PrimExpr start, const PrimExpr &stop, int num_stages,
                      const Array<PrimExpr> &order,
                      const Array<PrimExpr> &stages,
                      const Map<String, Any> &annotations) {
  TVM_FFI_CHECK_GE(num_stages, 0, ValueError)
      << "Pipeline stages must be nonnegative";
  PrimType type = IndexType({start, stop});
  auto node = make_object<ForFrameObj>();
  node->vars.push_back(PrimVar("pipeline", type));
  node->doms.push_back(Range(prim::cast(start, type), prim::cast(stop, type)));
  node->f_make_for_loop = [=](Array<Var> variables, Array<Range> domains,
                              Array<Optional<PrimExpr>> steps, SeqStmt body,
                              Location loc) -> Stmt {
    Map<String, Any> options = annotations;
    if (num_stages > 0)
      options.Set("num_stages", num_stages);
    if (!order.empty())
      options.Set("tl_pipeline_order", order);
    if (!stages.empty())
      options.Set("tl_pipeline_stage", stages);
    PrimExpr step = !steps.empty() && steps[0].has_value()
                        ? prim::cast(steps[0].value(), type)
                        : MakeConst(type, 1);
    return For(variables[0].as_or_throw<PrimVar>(), domains[0]->begin,
               domains[0]->begin + domains[0]->extent, step, ForKind::kDefault,
               body, options, loc);
  };
  return ForFrame(node);
}

ForFrame PersistentFor(const Array<PrimExpr> &domain, const PrimExpr &wave_size,
                       const PrimExpr &index, PrimExpr group_size,
                       int num_stages, const Map<String, Any> &annotations) {
  TVM_FFI_CHECK(!domain.empty(), ValueError)
      << "T.Persistent requires a nonempty domain";
  TVM_FFI_CHECK_GE(num_stages, 0, ValueError)
      << "Pipeline stages must be nonnegative";
  PrimType type = IndexType(domain);
  type = IndexType({MakeConst(type, 0), wave_size, index, group_size});
  auto node = make_object<ForFrameObj>();
  Array<PrimExpr> shape;
  for (size_t axis = 0; axis < domain.size(); ++axis) {
    shape.push_back(prim::cast(domain[axis], type));
    node->vars.push_back(PrimVar("persistent_" + std::to_string(axis), type));
    node->doms.push_back(Range(MakeConst(type, 0), shape.back()));
  }
  PrimExpr wave = prim::cast(wave_size, type);
  if (auto constant = GetConstInt(wave)) {
    TVM_FFI_CHECK_GT(*constant, 0, ValueError)
        << "Persistent wave size must be positive";
  }
  group_size =
      Max(MakeConst(type, 1), Min(prim::cast(group_size, type), shape.back()));
  Array<PrimExpr> grouped{ceildiv(shape.back(), group_size)};
  for (size_t axis = 0; axis + 1 < shape.size(); ++axis)
    grouped.push_back(shape[axis]);
  grouped.push_back(group_size);
  PrimExpr total = MakeConst(type, 1);
  for (const PrimExpr &extent : grouped)
    total = total * extent;
  PrimExpr waves = ceildiv(total, wave);
  PrimVar wave_var("wave", type);
  node->f_make_for_loop = [=](Array<Var> variables, Array<Range>,
                              Array<Optional<PrimExpr>>, SeqStmt body,
                              Location loc) -> Stmt {
    Map<String, Any> options = annotations;
    if (num_stages > 0)
      options.Set("num_stages", num_stages);
    Array<PrimExpr> coordinates(grouped.size(), MakeConst(type, 0));
    PrimExpr linear = wave_var * wave + prim::cast(index, type);
    PrimExpr remaining = linear;
    for (size_t axis = grouped.size(); axis-- > 1;) {
      coordinates.Set(axis, FloorMod(remaining, grouped[axis]));
      remaining = FloorDiv(remaining, grouped[axis]);
    }
    coordinates.Set(0, remaining);
    PrimExpr last = coordinates[0] * group_size + coordinates.back();
    Array<Stmt> statements;
    for (size_t axis = 0; axis < variables.size(); ++axis) {
      statements.push_back(Bind(
          variables[axis],
          axis + 1 == variables.size() ? last : coordinates[axis + 1], loc));
    }
    statements.push_back(
        If((last < shape.back()) && (linear < total), body, std::nullopt, loc));
    return For(wave_var, MakeConst(type, 0), waves, MakeConst(type, 1),
               ForKind::kDefault, SeqStmt(statements), options, loc);
  };
  return ForFrame(node);
}

KernelLaunchFrame KernelLaunch(const Array<PrimExpr> &grid_size,
                               const Optional<Array<PrimExpr>> &block_size,
                               const Map<String, Any> &attributes) {
  TVM_FFI_CHECK_LE(grid_size.size(), 3U, ValueError)
      << "T.Kernel supports at most three grid axes";
  auto node = make_object<KernelLaunchFrameNode>();
  const char *grid_names[] = {"bx", "by", "bz"};
  const char *grid_tags[] = {"blockIdx.x", "blockIdx.y", "blockIdx.z"};
  for (size_t axis = 0; axis < grid_size.size(); ++axis) {
    ForFrame frame = MakeThreadBindingFrame(grid_names[axis], grid_tags[axis],
                                            grid_size[axis]);
    node->grid_vars.push_back(frame->vars[0]);
    node->grid_extents.push_back(grid_size[axis]);
    node->frames.push_back(frame);
  }
  ForFrame threads = MakeLaunchThreadFrame();
  node->thread_vars = threads->vars;
  node->frames.push_back(threads);
  Map<String, Any> options = attributes;
  if (block_size.has_value()) {
    Array<PrimExpr> extents = block_size.value();
    TVM_FFI_CHECK_LE(extents.size(), 3U, ValueError)
        << "T.Kernel supports at most three thread axes";
    while (extents.size() < 3)
      extents.push_back(PrimExpr(1));
    node->thread_extents = extents;
    options.Set(attr::kLaunchThreads, extents);
  }
  node->frames.push_back(MakeNamedScope(DeviceMainBlockName, options));
  return KernelLaunchFrame(node);
}

class WarpSpecializeFrameNode : public StmtFrameObj {
public:
  Array<StmtFrame> frames;
  static void RegisterReflection() {
    reflection::ObjectDef<WarpSpecializeFrameNode>().def_ro(
        "frames", &WarpSpecializeFrameNode::frames);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WarpSpecializeFrame",
                                    WarpSpecializeFrameNode, StmtFrameObj);
  void EnterWithScope() final {
    for (const auto &frame : frames)
      frame->EnterWithScope();
  }
  void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

class WarpSpecializeFrame : public StmtFrame {
public:
  explicit WarpSpecializeFrame(ObjectPtr<WarpSpecializeFrameNode> node)
      : StmtFrame(std::move(node)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(WarpSpecializeFrame, StmtFrame,
                                                WarpSpecializeFrameNode);
};

WarpSpecializeFrame WarpSpecialize(const Array<IntImm> &warp_group_ids,
                                   const PrimExpr &thread_index,
                                   int warp_group_size) {
  TVM_FFI_CHECK(!warp_group_ids.empty() && warp_group_size > 0, ValueError)
      << "Warp specialization requires nonempty roles and a positive group "
         "size";
  PrimExpr condition = MakeConst(PrimType::Bool(), false);
  for (const IntImm &group : warp_group_ids) {
    int64_t index = GetConstInt(group).value();
    TVM_FFI_CHECK_GE(index, 0, ValueError)
        << "Warp group indices must be nonnegative";
    PrimExpr begin = MakeConst(thread_index.ty(), index * warp_group_size);
    PrimExpr end = MakeConst(thread_index.ty(), (index + 1) * warp_group_size);
    condition = condition || ((thread_index >= begin) && (thread_index < end));
  }
  auto node = make_object<WarpSpecializeFrameNode>();
  node->frames = {builder::If(condition), builder::Then(),
                  MakeNamedScope("warp_specialize", {{"warp_specialize", 1}})};
  return WarpSpecializeFrame(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef()
      .def("tl.Parallel", ParallelFor)
      .def("tl.Pipelined", PipelinedFor)
      .def("tl.Persistent", PersistentFor)
      .def("tl.KernelLaunch", KernelLaunch)
      .def("tl.AnnotateScope", AnnotateScope)
      .def("tl.WarpSpecialize", WarpSpecialize)
      .def("tl.SideEffect", tvm::SideEffect);
  KernelLaunchFrameNode::RegisterReflection();
  WarpSpecializeFrameNode::RegisterReflection();
}

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

WSPipeline::WSPipeline(String name, Array<tirx::TensorVar> buffers,
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
  refl::TypeAttrDef<WSSyncKindObj>()
      .def(ffi::type_attr::kDataToJson.name,
           [](const WSSyncKindObj *node) -> String { return node->_str_index; })
      .def(ffi::type_attr::kDataFromJson.name, [](String name) {
        return ffi::EnumObj::_GetByStrIndex<WSSyncKindObj>(name)
            .as_or_throw<WSSyncKind>();
      });
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
           [](String name, Array<tirx::TensorVar> buffers, int64_t depth) {
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

} // namespace tvm::tl
