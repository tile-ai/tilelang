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
#include <tvm/tir/stmt.h>
#include <tvm/tir/op_attr_types.h>

#include "../layout/layout.h"

namespace tvm {
namespace tl {

using namespace tir;

using AddWorkspaceCallback = std::function<PrimExpr(int, DataType)>;
using LayoutMap = Map<Buffer, Layout>;
using BufferMap = Map<Var, Buffer>;
using OpBuilderFunc = ffi::TypedFunction<void *(Array<PrimExpr>, BufferMap)>;

#define TIR_REGISTER_TL_OP(Entry, OpName)                                      \
  const Op &Entry::Get() {                                                     \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)             \
      .set_attr<OpBuilderFunc>("TLOpBuilder",                                  \
                               [](Array<PrimExpr> a, BufferMap b) {            \
                                 return (void *)(new Entry(a, b));             \
                               })

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
};

struct LayoutInferArgs {
  Target target;
  Range thread_bounds;
  LayoutMap layout_map;
  Map<Buffer, Buffer> buffer_remap;
};

class TileOperator {
 public:
  // Lower 接口
  virtual Stmt Lower(const LowerArgs& T, arith::Analyzer* analyzer) const {
    ICHECK(0) << "Not Implemented Lower method.";
    return Evaluate(0);
  }

  // InferLayout 接口
  virtual LayoutMap InferLayout(const LayoutInferArgs& T, InferLevel level) const {
    return {};
  }

  // Clone 接口
  virtual std::unique_ptr<TileOperator> Clone() const = 0;
  
  // 虚析构函数
  virtual ~TileOperator() = default;
};

Var GetVarFromAccessPtr(const PrimExpr &expr);

std::unique_ptr<TileOperator> ParseOperator(Call call, BufferMap vmap);
std::unique_ptr<TileOperator> ParseOperator(Stmt stmt, BufferMap vmap);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_OP_H_
