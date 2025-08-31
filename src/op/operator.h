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
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>

#include "../layout/layout.h"

namespace tvm {
namespace tl {

using namespace tir;

using AddWorkspaceCallback = std::function<PrimExpr(int, DataType)>;
using LayoutMap = Map<Buffer, Layout>;
using BufferMap = Map<Var, Buffer>;

enum class InferLevel {
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
  Map<Buffer, Buffer> buffer_remap;
};

class TileOperatorNode;
class TileOperator;

/**
 * Abstract base class for tile-level operators.
 *
 * Implementations must provide lowering to TIR, layout inference, and cloning.
 */

/**
 * Lower this tile operator to a TIR statement.
 *
 * @param T Lowering context and utilities (target, thread bounds, layout
 * mappings, buffer remapping, and AddWorkspace callback for requesting
 * temporary buffers).
 * @param analyzer Arithmetic analyzer used during lowering.
 * @return A TIR Stmt representing the lowered operator.
 */

/**
 * Infer buffer layouts for this operator.
 *
 * The returned LayoutMap associates input/output Buffers with inferred Layouts.
 * The `level` controls how strictly layouts are determined (kFree, kCommon,
 * kStrict).
 *
 * @param T Layout inference context (target, thread bounds, existing
 * layout_map, buffer_remap).
 * @param level Inference strictness level.
 * @return A LayoutMap mapping Buffers to their inferred Layouts.
 */

/**
 * Create a deep copy of this TileOperator.
 *
 * @return A TileOperator referencing a cloned operator instance.
 */

/**
 * Reference wrapper for TileOperatorNode.
 *
 * Use this ObjectRef to hold and pass tile operator instances within the
 * runtime.
 */

/**
 * Extract the underlying Var from an access pointer expression.
 *
 * If `expr` represents an access pointer that directly refers to a variable,
 * returns that Var; otherwise returns a null/default Var.
 *
 * @param expr The pointer/access expression to inspect.
 * @return The extracted Var, or a null Var if none can be found.
 */

/**
 * Parse a Call into a TileOperator using the provided buffer mapping.
 *
 * @param call The Call node representing a tile operator invocation.
 * @param vmap Mapping from TIR Vars to Buffers for resolving buffer arguments.
 * @return A TileOperator constructed from the call and buffer map.
 */

/**
 * Parse a Stmt into a TileOperator using the provided buffer mapping.
 *
 * @param stmt The Stmt representing a tile operator region or call.
 * @param vmap Mapping from TIR Vars to Buffers for resolving buffer references.
 * @return A TileOperator constructed from the statement and buffer map.
 */

/**
 * Function type for TL operator builders exposed to the FFI.
 *
 * Builder functions take an array of PrimExpr arguments and a BufferMap, and
 * return a constructed TileOperator.
 */

/**
 * Register a TL operator and its builder with TVM's op registry.
 *
 * Entry should be a type providing a static `Get()` and a constructor taking
 * `(Array<PrimExpr>, BufferMap)`. This macro registers the operator under the
 * name "tl.OpName" and sets an FFI builder attribute that constructs
 * Entry(args, vmap).
 *
 * Usage: TIR_REGISTER_TL_OP(MyOpEntry, MyOp)
 */
class TileOperatorNode : public Object {
public:
  virtual Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const = 0;

  virtual LayoutMap InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const = 0;

  virtual TileOperator Clone() const = 0;

  static constexpr const char *_type_key = "tl.TileOperator";

  TVM_DECLARE_BASE_OBJECT_INFO(TileOperatorNode, Object);
};

class TileOperator : public ObjectRef {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(TileOperator, ObjectRef, TileOperatorNode);
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
