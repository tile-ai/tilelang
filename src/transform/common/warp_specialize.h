/*!
 * \file warp_specialize.h
 * \brief Typed warp-specialization schedule, exposed via TVM-FFI.
 *
 * A WSSchedule is the complete description of how to transform a
 * straight-line kernel into a warp-specialized one. These are frontend IR
 * constructs (implemented in src/ir.cc): they are
 * built on the Python side (T.WSSchedule / T.WSRole / T.WSPipeline /
 * T.WSScope), attached to the tilelang_root block by
 * T.annotate_ws_schedule, and consumed by the MaterializeWSSchedule pass.
 *
 * Structure:
 *  - WSRole: a contiguous warp range [warp_lo, warp_hi) with one duty and an
 *    optional register budget.
 *  - WSPipeline: a full/empty mbarrier pair protecting a set of buffers with
 *    `depth` versions. Buffers are referenced directly (matched to block
 *    allocations by their data var).
 *  - WSInstr: one step of a role's program. Either a WSOpRef, naming a tile
 *    op or child scope by its stable `tl.ws_op_id`, or a WSSync, a pipeline
 *    synchronization point (producer_acquire / producer_commit /
 *    consumer_wait / consumer_release) with a software-pipeline stage.
 *  - WSScope: a loop (or the kWSRootScopeId root scope) with one
 *    instruction sequence per role.
 *  - WSSchedule: warp count, roles, pipelines, scopes.
 *
 * Dependencies (which ops touch which pipeline buffers) are intentionally
 * not part of the schedule: the pass infers them from the kernel.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_WARP_SPECIALIZE_H_
#define TVM_TL_TRANSFORM_COMMON_WARP_SPECIALIZE_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/enum.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/buffer.h>

#include "support/check.h"

namespace tvm {
namespace tl {

/*!
 * \brief Block annotation key carrying the WSSchedule object.
 *
 * Set on the tilelang_root block by T.annotate_ws_schedule.
 */
static constexpr const char *kWSScheduleKey = "tl.ws_schedule";

/*!
 * \brief Annotation key carrying a stable op / scope id.
 *
 * Appears in tile-op call annotations
 * (T.copy(..., annotations={"tl.ws_op_id": ...})), loop annotations
 * (T.Pipelined / T.serial / T.unroll / T.Parallel), and AttrStmt wrappers
 * around statement groups (with T.ws_op(...)).
 */
static constexpr const char *kWSOpIdKey = "tl.ws_op_id";

// Attr wrapper from T.annotate_ws_pipeline_depth: node = buffer var, value =
// depth of the pipeline the automatic scheduler hosts for it at the
// enclosing scope. Scheduler-only metadata; the materializer drops it.
static constexpr const char *kWSPipelineDepthKey = "tl.ws_pipeline_depth";

/*! \brief Scope id of the kernel's implicit root scope (T.WSScope.ROOT). */
static constexpr const char *kWSRootScopeId = "tl.ws_scope_root";

/*! \brief A contiguous warp range with a single duty. */
class WSRoleNode : public ffi::Object {
public:
  /*! \brief Role name; keys the per-role bodies of every scope. */
  ffi::String name;
  /*! \brief Warp range [warp_lo, warp_hi). */
  int64_t warp_lo;
  int64_t warp_hi;
  /*! \brief setmaxnreg budget; 0 = unset. */
  int64_t max_nreg;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WSRole", WSRoleNode, ffi::Object);
};

class WSRole : public ffi::ObjectRef {
public:
  TVM_DLL WSRole(ffi::String name, int64_t warp_lo, int64_t warp_hi,
                 int64_t max_nreg);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WSRole, ffi::ObjectRef,
                                             WSRoleNode);
};

/*! \brief A full/empty barrier pair protecting multi-versioned buffers. */
class WSPipelineNode : public ffi::Object {
public:
  /*! \brief Pipeline name; referenced by WSSync instructions. */
  ffi::String name;
  /*! \brief The buffers this pipeline protects (and multi-versions). */
  ffi::Array<tirx::Buffer> buffers;
  /*! \brief Number of buffer versions. */
  int64_t depth;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WSPipeline", WSPipelineNode,
                                    ffi::Object);
};

class WSPipeline : public ffi::ObjectRef {
public:
  TVM_DLL WSPipeline(ffi::String name, ffi::Array<tirx::Buffer> buffers,
                     int64_t depth);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WSPipeline, ffi::ObjectRef,
                                             WSPipelineNode);
};

/*! \brief Base class of one step in a role's program. */
class WSInstrNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.WSInstr", WSInstrNode, ffi::Object);
};

class WSInstr : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WSInstr, ffi::ObjectRef,
                                             WSInstrNode);
};

/*! \brief Reference to a tile op or child scope by its `tl.ws_op_id`. */
class WSOpRefNode : public WSInstrNode {
public:
  ffi::String id;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WSOpRef", WSOpRefNode, WSInstrNode);
};

class WSOpRef : public WSInstr {
public:
  TVM_DLL explicit WSOpRef(ffi::String id);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WSOpRef, WSInstr, WSOpRefNode);
};

/*! \brief Pipeline synchronization kind, in TVM FFI enum convention. */
class WSSyncKindObj : public ffi::EnumObj {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WSSyncKind", WSSyncKindObj,
                                    ffi::EnumObj);
};

class WSSyncKind : public ffi::Enum {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WSSyncKind, ffi::Enum,
                                             WSSyncKindObj);

  static WSSyncKind ProducerAcquire() { return Get("PRODUCER_ACQUIRE"); }
  static WSSyncKind ProducerCommit() { return Get("PRODUCER_COMMIT"); }
  static WSSyncKind ConsumerWait() { return Get("CONSUMER_WAIT"); }
  static WSSyncKind ConsumerRelease() { return Get("CONSUMER_RELEASE"); }

  int CanonicalOrdinal() const {
    return static_cast<int>(operator->()->_value);
  }

  bool IsProducerAcquire() const { return CanonicalOrdinal() == 0; }
  bool IsProducerCommit() const { return CanonicalOrdinal() == 1; }
  bool IsConsumerWait() const { return CanonicalOrdinal() == 2; }
  bool IsConsumerRelease() const { return CanonicalOrdinal() == 3; }
  /*! \brief Waits (acquire / consumer-wait) open a span and block. */
  bool IsWait() const { return IsProducerAcquire() || IsConsumerWait(); }
  /*! \brief Commits (commit / release) close a span and signal a barrier. */
  bool IsCommit() const { return IsProducerCommit() || IsConsumerRelease(); }
  /*! \brief Producer-side syncs use the empty->full barrier direction. */
  bool IsProducer() const { return IsProducerAcquire() || IsProducerCommit(); }

private:
  static WSSyncKind Get(const ffi::String &name) {
    ffi::Enum e = ffi::EnumObj::Get<WSSyncKindObj>(name);
    const auto *node = e.as<WSSyncKindObj>();
    ICHECK(node != nullptr)
        << "WSSyncKind entry `" << name << "` is not a WSSyncKindObj";
    return ffi::GetRef<WSSyncKind>(node);
  }
};

/*! \brief A pipeline synchronization point. */
class WSSyncNode : public WSInstrNode {
public:
  WSSyncKind kind;
  /*! \brief Name of the pipeline being synchronized. */
  ffi::String pipeline;
  /*! \brief Software-pipeline stage of this sync point. */
  int64_t stage;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WSSync", WSSyncNode, WSInstrNode);
};

class WSSync : public WSInstr {
public:
  TVM_DLL WSSync(WSSyncKind kind, ffi::String pipeline, int64_t stage);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WSSync, WSInstr, WSSyncNode);
};

/*! \brief A loop (or the root scope) with per-role instruction lists. */
class WSScopeNode : public ffi::Object {
public:
  /*! \brief The `tl.ws_op_id` of the loop, or kWSRootScopeId. */
  ffi::String id;
  /*! \brief Role name -> instruction sequence. */
  ffi::Map<ffi::String, ffi::Array<WSInstr>> bodies;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WSScope", WSScopeNode, ffi::Object);
};

class WSScope : public ffi::ObjectRef {
public:
  TVM_DLL WSScope(ffi::String id,
                  ffi::Map<ffi::String, ffi::Array<WSInstr>> bodies);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WSScope, ffi::ObjectRef,
                                             WSScopeNode);
};

/*! \brief The complete warp-specialization schedule of one kernel. */
class WSScheduleNode : public ffi::Object {
public:
  /*! \brief Total warp count; overrides the kernel's thread extent. */
  int64_t num_warps;
  ffi::Array<WSRole> roles;
  ffi::Array<WSPipeline> pipelines;
  ffi::Array<WSScope> scopes;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WSSchedule", WSScheduleNode,
                                    ffi::Object);
};

class WSSchedule : public ffi::ObjectRef {
public:
  TVM_DLL WSSchedule(int64_t num_warps, ffi::Array<WSRole> roles,
                     ffi::Array<WSPipeline> pipelines,
                     ffi::Array<WSScope> scopes);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WSSchedule, ffi::ObjectRef,
                                             WSScheduleNode);
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_WARP_SPECIALIZE_H_
