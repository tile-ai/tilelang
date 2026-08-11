/*!
 * \file tl/op/builtin_registry.h
 * \brief Internal registration helper for TileLang intrinsic Ops.
 */

#ifndef TVM_TL_OP_BUILTIN_REGISTRY_H_
#define TVM_TL_OP_BUILTIN_REGISTRY_H_

#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName", #OpName)

#endif // TVM_TL_OP_BUILTIN_REGISTRY_H_
