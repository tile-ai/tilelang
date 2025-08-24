/*!
 * \file tl/op/elem.h
 * \brief Define element-wise and copy-related operators for TVM TensorIR
 * Lowering.
 *
 * This header declares the Copy operator and related operator descriptors
 * such as TMADesc and TMAIm2ColDesc, as well as a Conv2DIm2Col special
 * operator.
 */

#ifndef TVM_TL_OP_COPY_H_
#define TVM_TL_OP_COPY_H_

#include "op.h"
#include "parallel.h"

namespace tvm {
namespace tl {
using namespace tir;

/*!
 * \brief Descriptor for Tensor Memory Access (TMA) copy operations.
 *
 * Contains meta-information required to perform global-to-shared memory copy
 * using Tensor Memory Accelerator (TMA) hardware instructions. It is mainly
 * used to describe the shape, strides, and data layout for both source and
 * shared memory buffers.
 */
struct TMADesc {
  size_t rank;                  // Tensor rank (number of dimensions)
  int data_type;                // Data type identifier (numeric code)
  Array<PrimExpr> global_shape; // Shape of the source tensor in global memory
  Array<PrimExpr>
      global_stride;           // Strides of the source tensor in global memory
  Array<PrimExpr> smem_box;    // Block shape in shared memory
  Array<PrimExpr> smem_stride; // Strides in shared memory layout
  PrimExpr global_addr;        // Base address in global memory
  int swizzle;                 // Swizzle parameter for memory layout transform
  int interleave;              // Interleave parameter for optimization
  int oob_fill;                // Out-of-bound fill policy
  int l2_promotion;            // Whether to promote data to L2 cache

  /*!
   * \brief Encode descriptor fields into an argument array for runtime calls.
   */
  Array<PrimExpr> EncodeCallArgs() const;
};

/*!
 * \brief Descriptor for TMA-based im2col transformation used in Conv2D.
 *
 * This supports extracting patches from the input image (im2col)
 * for convolution lowering, storing them in shared memory.
 */
struct TMAIm2ColDesc {
  size_t rank;                   // Rank of the tensor
  int data_type;                 // Data type identifier
  Array<PrimExpr> global_shape;  // Shape of input tensor in global memory
  Array<PrimExpr> global_stride; // Stride in global memory
  Array<PrimExpr> elem_stride;   // Stride at element level (per axis)
  Array<PrimExpr> lower_corner; // Lower bound offsets for the extraction window
                                // (rank - 2 dims)
  Array<PrimExpr> upper_corner; // Upper bound offsets for the extraction window
                                // (rank - 2 dims)
  PrimExpr global_addr;         // Base address in global memory
  int smem_box_pixel;           // Pixel dimension of shared memory box
  int smem_box_channel;         // Channel dimension of shared memory box
  int swizzle;                  // Memory swizzle setting
  int interleave;               // Memory interleaving setting
  int oob_fill;                 // Out-of-bound fill policy
  int l2_promotion;             // Whether to enable L2 cache promotion

  /*!
   * \brief Encode descriptor fields into runtime arguments.
   */
  Array<PrimExpr> EncodeCallArgs() const;
};

/*!
 * \brief Copy operator for transferring data between buffers.
 *
 * This class implements a generic copy operator in TensorIR Lowering for
 * block-wise or element-wise data transfer, possibly optimized with
 * parallelization or TMA hardware acceleration.
 */
class Copy : public Operator {
public:
  /*!
   * \brief Constructor.
   * \param args  Expression arguments for the copy.
   * \param vmap  Buffer variable mapping.
   */
  Copy(Array<PrimExpr> args, BufferMap vmap);

  /*!
   * \brief Lower the copy operator to a TIR statement.
   * \param T        Arguments for lowering.
   * \param analyzer Analyzer for simplification and bounds checks.
   */
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;

  /*!
   * \brief Infer buffer layouts after applying this operator.
   * \param T     Arguments for layout inference.
   * \param level Level of inference (basic or detailed).
   */
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;

  /*!
   * \brief Get the TVM Op handle corresponding to this Copy op.
   */
  static const Op &Get();

  /*!
   * \brief Copy instruction type.
   */
  enum class CopyInst {
    kNormal = 0,    // utilize ldg/stg or cpasync or any buffer copy
    kLDSM = 1,      // ldmatrix memory copy
    kSTSM = 2,      // stmatrix memory copy
    kBulkLoad = 3,  // utilize tma load
    kBulkStore = 4, // utilize tma store
  };

  /*!
   * \brief Check if bulk copy is supported.
   */
  bool CheckBulkLoad(Target target) const;

  /*!
   * \brief Check if bulk store is supported.
   */
  bool CheckBulkStore(Target target) const;

  /*!
   * \brief Check if lds memory copy is supported.
   */
  bool CheckLDSMCopy(Target target) const;

  /*!
   * \brief Check if stsm memory copy is supported.
   */
  bool CheckSTSMCopy(Target target) const;

  /*!
   * \brief Get the copy instruction type.
   */
  CopyInst GetCopyInst(Target target, bool disable_tma_lower) const;

  /*!
   * \brief Copy constructor (deep clones ParallelOp if present).
   */
  Copy(const Copy &other)
      : args_(other.args_), src(other.src), dst(other.dst),
        src_range(other.src_range), dst_range(other.dst_range),
        coalesced_width(other.coalesced_width), disable_tma(other.disable_tma) {
    // Deep copy ParallelOp if it exists
    if (other.par_op_)
      par_op_ = std::unique_ptr<ParallelOp>(
          static_cast<ParallelOp *>(other.par_op_->Clone().release()));
  }

  /*!
   * \brief Clone this copy operator.
   */
  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<Copy>(*this);
  }

protected:
  /*!
   * \brief Generate lowering for bulk/global-to-shared copy.
   */
  Stmt LowerBulkCopy(const LowerArgs &T, arith::Analyzer *analyzer,
                     CopyInst copy_inst) const;

  /*!
   * \brief Generate lowering for LDS Memory Copy (shared memory to shared
   * memory or smem usage).
   */
  Stmt LowerLDSMCopy(const LowerArgs &T, arith::Analyzer *analyzer,
                     CopyInst copy_inst) const;

  /*!
   * \brief Generate lowering for normal copy.
   */
  Stmt LowerNormalCopy(const LowerArgs &T, arith::Analyzer *analyzer) const;

  /*!
   * \brief Generate SIMT (thread-level) loop for copying.
   */
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;

  /*!
   * \brief Compute linear layout for tma copy.
   */
  Layout ComputeLinearLayout(const Buffer &shared_tensor) const;

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

  Array<PrimExpr> args_; // Copy parameters (indices, sizes, etc.)

  Buffer src, dst;                   // Source and destination buffers
  Array<Range> src_range, dst_range; // Ranges for each dimension in src and dst
  IntImm coalesced_width; // Width (in elements) for coalesced memory access
  Bool disable_tma = Bool(false); // Whether to disable TMA acceleration

  std::unique_ptr<ParallelOp>
      par_op_; // Optional associated parallelization operator

  enum class EvictionPolicy {
    kEvictNormal = 0,
    kEvictFirst = 1,
    kEvictLast = 2,
  };

  int eviction_policy; // Policy for cache eviction
};

/*!
 * \brief Special operator for Conv2D im2col transformation.
 *
 * This operator converts input image layout into columnar format suitable
 * for matrix multiplication-based convolution lowering.
 */
class Conv2DIm2ColOp : public Operator {
public:
  /*!
   * \brief Constructor.
   * \param args  Op arguments (convolution parameters, shapes, etc.)
   * \param vmap  Variable buffer mapping.
   */
  Conv2DIm2ColOp(Array<PrimExpr> args, BufferMap vmap);

  /*!
   * \brief Lower to TIR statement.
   */
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;

  /*!
   * \brief Get TVM Op handle.
   */
  static const Op &Get();

  /*!
   * \brief Clone this operator.
   */
  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<Conv2DIm2ColOp>(*this);
  }

private:
  Buffer src, dst; // Source (input feature map) and destination (im2col matrix)
  int stride;      // Stride for convolution
  int padding;     // Padding amount
  int dilation;    // Dilation factor
  int kernel;      // Kernel size
  int eviction_policy; // Cache eviction policy
  PrimExpr nhw_step;   // Step size in NHW dimensions
  PrimExpr c_step;     // Step size in channel dimension
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_COPY_H_