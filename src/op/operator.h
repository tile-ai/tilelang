/*!
 * \file tl/op/op.h
 * \brief Tile library operations.
 *
 */

#ifndef TVM_TL_OP_OP_H_
#define TVM_TL_OP_OP_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>

#include "../layout/layout.h"

namespace tvm {
namespace tl {

using namespace tir;

using AddWorkspaceCallback = std::function<PrimExpr(int, DataType)>;
using LayoutMap = Map<Buffer, Layout>;
using BufferMap = Map<Var, Buffer>;

enum class InferLevel : uint8_t {
  kFree = 0,
  kCommon = 1,
  kStrict = 2,
};

struct LowerArgs {
  Target target;
  Range thread_bounds;
  Var thread_var;
  AddWorkspaceCallback AddWorkspace;
  LayoutMap layout_map;
  Map<Buffer, Buffer> buffer_remap;
  Array<Var> buffer_var_gemm;
};

struct LayoutInferArgs {
  Target target;
  Range thread_bounds;
  LayoutMap layout_map;
  arith::Analyzer *analyzer;
  bool buffer_oob = false;
  Map<Buffer, Buffer> buffer_remap;
};

class TileOperator;

class TileOperatorNode : public Object {
public:
  virtual Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const = 0;

  virtual LayoutMap InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const = 0;

  virtual TileOperator Clone() const = 0;

  TVM_FFI_DECLARE_OBJECT_INFO("tl.TileOperator", TileOperatorNode, Object);
};

class TileOperator : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TileOperator, ObjectRef,
                                             TileOperatorNode);
};

Var GetVarFromAccessPtr(const PrimExpr &expr);

TileOperator ParseOperator(Call call, BufferMap vmap);
TileOperator ParseOperator(Stmt stmt, BufferMap vmap);

using OpBuilderFunc =
    ffi::TypedFunction<TileOperator(Array<PrimExpr>, BufferMap)>;

#define TIR_REGISTER_TL_OP(Entry, OpName)                                      \
  const Op &Entry::Get() {                                                     \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)             \
      .set_attr<OpBuilderFunc>("TLOpBuilder",                                  \
                               [](Array<PrimExpr> args, BufferMap vmap) {      \
                                 return Entry(args, vmap);                     \
                               })

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_OP_H_
