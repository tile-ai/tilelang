/*!
 * \file tl/op/builtin_registry.h
 * \brief Canonical lookup helpers; OpDef owns the actual native contracts.
 */
#ifndef TVM_TL_OP_BUILTIN_REGISTRY_H_
#define TVM_TL_OP_BUILTIN_REGISTRY_H_
#include <tvm/ir/op.h>
#include <tvm/tirx/op/attr.h>
#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op op = Op::Get("tl." #OpName);                               \
    return op;                                                                 \
  }
#endif // TVM_TL_OP_BUILTIN_REGISTRY_H_
