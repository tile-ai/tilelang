/*!
 * \file tl/op/copy.h
 * \brief Copy operations
 */

#ifndef TVM_TL_OP_COPY_H_
#define TVM_TL_OP_COPY_H_

#include "builtin.h"
#include "operator.h"
#include "parallel.h"

#include <utility>

namespace tvm {
namespace tl {
using namespace tir;

/*!
 * \brief Get TVM Op handle for Conv2DIm2Col.
 */

/*!
 * \brief Clone this Conv2DIm2Col operator.
 *
 * Returns a TileOperator reference that is a shallow clone of this operator.
 */
class CopyNode : public TileOperatorNode {
public:
  Buffer src, dst;                   // Source and destination buffers
  Array<Range> src_range, dst_range; // Ranges for each dimension in src and dst
  Map<String, ObjectRef> annotations; // Backend/pass-specific annotations.
  // Common SIMT annotation keys:
  //   - "coalesced_width": IntImm, width for coalesced memory access.
  //   - attr::kParallelLoopLayout ("parallel_loop_layout"): Fragment, loop
  //     layout hint applied to the outermost generated parallel loop of this
  //     copy's SIMT loop nest.

  mutable ParallelOp par_op_; // Optional associated parallelization operator

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Copy", CopyNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CopyNode>()
        .def_ro("src", &CopyNode::src)
        .def_ro("dst", &CopyNode::dst)
        .def_ro("src_range", &CopyNode::src_range)
        .def_ro("dst_range", &CopyNode::dst_range)
        .def_ro("annotations", &CopyNode::annotations);
  }

  /*!
   * \brief Lower the copy operator to a TIR statement.
   * \param T        Arguments for lowering.
   * \param analyzer Analyzer for simplification and bounds checks.
   */
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  /*!
   * \brief Infer buffer layouts after applying this operator.
   * \param T     Arguments for layout inference.
   * \param level Level of inference (basic or detailed).
   */
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  /*!
   * \brief Infer layout through the generated SIMT copy loop.
   */
  LayoutMap InferSIMTLayout(const LayoutInferArgs &T, InferLevel level) const;

  /*!
   * \brief Generate SIMT (thread-level) loop for copying.
   */
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;

  /*!
   * \brief Create iterator variables for multi-dimensional copy loops.
   */
  Array<IterVar> MakeIterVars() const;

  /*!
   * \brief Calculate source or destination indices from iteration vars.
   * \param ivs      Iterator variables from MakeIterVars().
   * \param src_dst  0 = make source indices, 1 = make destination indices.
   */
  Array<PrimExpr> MakeIndices(const Array<IterVar> &ivs, int src_dst) const;

  /*!
   * \brief Construct the boundary predicate for valid copy (to avoid OOB).
   * \param analyzer  Arithmetic analyser for simplification.
   * \param ivs       Iterator variables.
   * \param extents   Extent expressions for the relevant buffer.
   * \param src_dst   0 = predicate for source, 1 = predicate for destination.
   */
  PrimExpr MakePredicate(arith::Analyzer *analyzer, const Array<IterVar> &ivs,
                         Array<PrimExpr> extents, int src_dst) const;

protected:
  /**
   * \brief Create a deep copy of this operator.
   *
   * Returns a TileOperator that is a copy of the current node, preserving all
   * configuration (buffers, parameters, and layout-related fields).
   * @return A TileOperator owning the cloned operator node.
   */

  /**
   * \brief Constructor.
   * \param args Expression arguments for the Conv2D im2col operator.
   * \param vmap Buffer variable mapping.
   */

  /**
   * \brief Get the TVM Op handle corresponding to this Conv2DIm2Col operator.
   * @return Reference to the singleton TVM Op representing this operator.
   */
  TileOperator Clone() const;
};

using CopyTargetPredicate = bool (*)(Target target);

struct CopyImpl {
  const char *name;
  CopyTargetPredicate match_target;
  int priority;

  LayoutMap (*infer_layout)(const CopyNode &op, const LayoutInferArgs &T,
                            InferLevel level);

  Stmt (*lower)(const CopyNode &op, const LowerArgs &T,
                arith::Analyzer *analyzer);
};

void RegisterCopyImpl(CopyImpl impl);

Stmt LowerNormalCopy(const CopyNode &op, const LowerArgs &T,
                     arith::Analyzer *analyzer);

class Copy : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Copy, TileOperator, CopyNode);

  /*!
   * \brief Constructor.
   * \param args  Expression arguments for the copy.
   * \param annotations  Annotations map from the Call node.
   */
  TVM_DLL Copy(Array<PrimExpr> args,
               Map<String, ObjectRef> annotations = Map<String, ObjectRef>());

  /*!
   * \brief Get the TVM Op handle corresponding to this Copy op.
   */
  static const Op &Get();
};

/*!
 * \brief Special operator for Conv2D im2col transformation.
 *
 * This operator converts input image layout into columnar format suitable
 * for matrix multiplication-based convolution lowering.
 */
class Conv2DIm2ColOpNode : public TileOperatorNode {
public:
  BufferRegion srcRegion_, dstRegion_;
  Buffer src_,
      dst_;      // Source (input feature map) and destination (im2col matrix)
  int stride_;   // Stride for convolution
  int padding_;  // Padding amount
  int dilation_; // Dilation factor
  int kernel_;   // Kernel size
  int eviction_policy_;                // Cache eviction policy
  PrimExpr nhw_step_;                  // Step size in NHW dimensions
  PrimExpr c_step_;                    // Step size in channel dimension
  Map<String, ObjectRef> annotations_; // Annotations from Call node

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Conv2DIm2Col", Conv2DIm2ColOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Conv2DIm2ColOpNode>()
        .def_ro("srcRegion", &Conv2DIm2ColOpNode::srcRegion_)
        .def_ro("dstRegion", &Conv2DIm2ColOpNode::dstRegion_)
        .def_ro("src", &Conv2DIm2ColOpNode::src_)
        .def_ro("dst", &Conv2DIm2ColOpNode::dst_)
        .def_ro("stride", &Conv2DIm2ColOpNode::stride_)
        .def_ro("padding", &Conv2DIm2ColOpNode::padding_)
        .def_ro("dilation", &Conv2DIm2ColOpNode::dilation_)
        .def_ro("kernel", &Conv2DIm2ColOpNode::kernel_)
        .def_ro("eviction_policy", &Conv2DIm2ColOpNode::eviction_policy_);
  }

  /*!
   * \brief Lower to TIR statement.
   */
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  /*!
   * \brief Infer layout for this operator.
   */
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  /*!
   * \brief Get TVM Op handle.
   */
  static const Op &Get();
  TileOperator Clone() const;
};

struct Conv2DIm2ColImpl {
  const char *name;
  CopyTargetPredicate match_target;
  int priority;

  Stmt (*lower)(const Conv2DIm2ColOpNode &op, const LowerArgs &T,
                arith::Analyzer *analyzer);
};

void RegisterConv2DIm2ColImpl(Conv2DIm2ColImpl impl);

class Conv2DIm2ColOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Conv2DIm2ColOp, TileOperator,
                                             Conv2DIm2ColOpNode);
  TVM_DLL
  Conv2DIm2ColOp(Array<PrimExpr> args,
                 Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_COPY_H_
