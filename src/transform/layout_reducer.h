/*!
 * \file layout_reducer.h
 */

#ifndef TVM_TL_TRANSFORM_LAYOUT_REDUCER_H_
#define TVM_TL_TRANSFORM_LAYOUT_REDUCER_H_

#include <tvm/tir/op.h>

#include "../layout/layout.h"

namespace tvm {
namespace tl {

enum class ReducerOpType { SUM, MAX, MIN };
enum class ReducerRepType { ALL, NONE };

struct ReducerInfoNode : Object {
  ReducerOpType op;
  ReducerRepType rep;

  ReducerInfoNode() = default;
  ReducerInfoNode(const String &op_str, const String &rep_str);
  static constexpr const char *_type_key = "tl.ReducerInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReducerInfoNode, Object);
};

struct ReducerInfo : ObjectRef {
public:
  TVM_DLL ReducerInfo(const String &op_str, const String &rep_str) {
    data_ = make_object<ReducerInfoNode>(op_str, rep_str);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(ReducerInfo, ObjectRef, ReducerInfoNode);
};

namespace attr {
constexpr const char *kReducerInfo = "reducer_info";
}

} // namespace tl
} // namespace tvm

#endif
