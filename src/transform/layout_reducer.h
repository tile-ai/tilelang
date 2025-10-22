/*!
 * \file layout_reducer.h
 */

#ifndef TVM_TL_TRANSFORM_LAYOUT_REDUCER_H_
#define TVM_TL_TRANSFORM_LAYOUT_REDUCER_H_

#include <tvm/tir/op.h>

#include "../layout/layout.h"

namespace tvm {
/**
 * Types of reduction operations supported by TL transforms.
 *
 * SUM   - arithmetic sum reduction.
 * MAX   - elementwise maximum reduction.
 * MIN   - elementwise minimum reduction.
 */

/**
 * Representation semantics for a reducer.
 *
 * ALL  - reducer collapses all elements along the reduced axes.
 * NONE - reducer does not collapse (used to represent a placeholder/no-op).
 */

/**
 * Holds metadata describing a reducer used in layout transforms.
 *
 * Contains the reduction operation (`op`) and its representation semantics
 * (`rep`).
 */

/**
 * Construct a ReducerInfoNode from textual identifiers.
 *
 * @param op_str  String identifier for the reduction operation (e.g., "sum",
 * "max", "min").
 * @param rep_str String identifier for the representation semantics (e.g.,
 * "all", "none").
 */

/**
 * Handle type for ReducerInfoNode (ObjectRef wrapper).
 *
 * Constructed from string identifiers for operation and representation.
 *
 * @param op_str  String identifier for the reduction operation (e.g., "sum",
 * "max", "min").
 * @param rep_str String identifier for the representation semantics (e.g.,
 * "all", "none").
 */

/**
 * Attribute key used to attach ReducerInfo to IR nodes or other attribute maps.
 */
namespace tl {

enum class ReducerOpType { SUM, MAX, MIN };
enum class ReducerRepType { ALL, NONE };

struct ReducerInfoNode : Object {
  ReducerOpType op;
  ReducerRepType rep;

  ReducerInfoNode() = default;
  ReducerInfoNode(const String &op_str, const String &rep_str);
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ReducerInfo", ReducerInfoNode, Object);
};

struct ReducerInfo : ObjectRef {
public:
  TVM_DLL ReducerInfo(const String &op_str, const String &rep_str) {
    data_ = tvm::ffi::make_object<ReducerInfoNode>(op_str, rep_str);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReducerInfo, ObjectRef,
                                             ReducerInfoNode);
};

namespace attr {
constexpr const char *kReducerInfo = "reducer_info";
}

} // namespace tl
} // namespace tvm

#endif
