/*!
 * \file tl/op/elem.h
 * \brief Define elment-wise operators.
 *
 */

#ifndef TVM_TL_OP_ELEM_H_
#define TVM_TL_OP_ELEM_H_

#include "operator.h"
#include "parallel.h"

/**
 * Lower the Fill operator into TIR statements.
 *
 * Produces a TIR Stmt that implements element-wise filling of `dst` over
 * `region` with `value`, using information from `T`.
 *
 * @param T Lowering inputs (buffers, shapes, and iteration info) used to
 * generate the IR.
 */

/**
 * Infer the memory layout mapping for the Fill operator.
 *
 * Returns a LayoutMap that describes how logical iteration axes map to memory
 * dimensions for the destination buffer. `level` controls the aggressiveness
 * of inference (e.g., relaxed vs. strict constraints).
 *
 * @param T Layout inference inputs (buffers, shapes, and related metadata).
 * @param level Inference level controlling precision of the returned mapping.
 */

/**
 * Return the global operator descriptor for tl.Fill.
 *
 * The returned Op can be used to look up operator-level metadata and to
 * register or query the operator within the TVM operator registry.
 */

/**
 * Create a copy of this operator node as a TileOperator reference.
 *
 * The returned TileOperator is an independent handle representing a clone of
 * the underlying FillNode.
 */

/**
 * Build a SIMT-style For loop that implements the fill.
 *
 * Constructs and returns a TIR `For` loop that iterates over the target region
 * in a SIMT-friendly ordering appropriate for `dst` and `region`.
 */

/**
 * Construct a Fill operator from argument expressions and a buffer mapping.
 *
 * @param args Positional PrimExpr arguments passed to the operator (e.g.,
 * indices or shape expressions required by the operator's specification).
 * @param vmap Mapping from named buffer parameters to concrete tir::Buffer
 * instances used by this operator instance.
 */

/**
 * Return the global operator descriptor for the public Fill wrapper.
 *
 * Mirrors FillNode::Get() and provides the operator descriptor for users of the
 * public TileOperator API.
 */
namespace tvm {
namespace tl {

using namespace tir;

class FillNode : public TileOperatorNode {
public:
  tir::Buffer dst;
  PrimExpr value;
  Array<Range> region;
  static constexpr const char *_type_key = "tl.Fill";
  TVM_DECLARE_FINAL_OBJECT_INFO(FillNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;
  static const Op &Get();

  TileOperator Clone() const;

private:
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
};

class Fill : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Fill, TileOperator, FillNode);
  TVM_DLL Fill(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ELEM_H_