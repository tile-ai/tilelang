/*!
 * \file tl/ir/warp_specialize.cc
 * \brief Native objects describing TileLang warp-specialization schedules.
 */

#include "transform/common/warp_specialize.h"

#include <tvm/ffi/reflection/creator.h>
#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {
using namespace ffi;

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
  auto n = make_object<WSSyncNode>(std::move(kind));
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
