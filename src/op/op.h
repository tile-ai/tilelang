// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

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

#include "../layout/layout.h"

namespace tvm {
namespace tl {

using namespace tir;

using AddWorkspaceCallback = std::function<PrimExpr(int, DataType)>;
using LayoutMap = Map<Buffer, Layout>;
using BufferMap = Map<Var, Buffer>;
using OpBuilderFunc = TypedPackedFunc<void *(Array<PrimExpr>, BufferMap)>;

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
  size_t block_size;
  Var thread_var;
  AddWorkspaceCallback AddWorkspace;
  LayoutMap layout_map;
  Map<Buffer, Buffer> buffer_remap;
  bool disable_tma_lower;
};

struct LayoutInferArgs {
  Target target;
  size_t block_size;
  LayoutMap layout_map;
  Map<Buffer, Buffer> buffer_remap;
};

struct CanonializeArgs {
  Target target;
};

class Operator {
public:
  virtual Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;
  virtual Stmt Canonialize(const CanonializeArgs &T,
                           arith::Analyzer *analyzer) const;
  virtual LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level);
  virtual ~Operator() = default;
};

class RegionOp : public Operator {
public:
  RegionOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();

  const Buffer &GetBuffer() const { return buffer_; }
  const Array<Range> &GetRanges() const { return ranges_; }
  int GetAccessMask() const { return access_mask_; }
  bool IsFullRegion() const;

private:
  Buffer buffer_;
  Array<Range> ranges_;
  int access_mask_;
};

Var GetVarFromAccessPtr(const PrimExpr &expr);

std::unique_ptr<Operator> ParseOperator(Call call, BufferMap vmap);
std::unique_ptr<Operator> ParseOperator(Stmt stmt, BufferMap vmap);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_OP_H_
