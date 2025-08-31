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

#include "operator.h"
#include "parallel.h"

namespace tvm {
namespace tl {
using namespace tir;

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
 * Performs element- or block-wise copies between `src` and `dst` buffers for
 * TensorIR lowering. The operator supports thread-level parallelization,
 * shared-memory layouts, and hardware-accelerated paths (TMA/LDSM/STMATRIX)
 * when available. Public fields describe the copy ranges and tuning knobs
 * (coalesced width, eviction policy, disable_tma).
 */

/*!
 * \brief Lower the copy operator to a TIR statement.
 *
 * Produces a TIR statement implementing the configured copy (normal, LDSM,
 * STSM, or bulk TMA-based) for the given lowering context.
 *
 * \param T        Lowering arguments that provide buffer bindings and context.
 * \param analyzer Analyzer used for expression simplification and bounds
 * checks. \return         A TIR `Stmt` implementing the copy.
 */

/*!
 * \brief Infer buffer layouts after applying this operator.
 *
 * Computes resulting layouts (shape/stride mappings) for buffers affected by
 * this copy operation.
 *
 * \param T     Arguments for layout inference (buffer maps, shapes).
 * \param level Granularity of inference to perform.
 * \return      A LayoutMap describing inferred layouts.
 */

/*!
 * \brief Check if bulk global->shared copy is supported on the target.
 *
 * Returns true if the target supports bulk (TMA) loads from global memory.
 *
 * \param target Target to query.
 */

/*!
 * \brief Check if bulk shared->global store is supported on the target.
 *
 * Returns true if the target supports bulk (TMA) stores to global memory.
 *
 * \param target Target to query.
 */

/*!
 * \brief Check if LDSM (LDMATRIX) memory-copy is supported on the target.
 *
 * \param target Target to query.
 */

/*!
 * \brief Check if STSM (STMATRIX) memory-copy is supported on the target.
 *
 * \param target Target to query.
 */

/*!
 * \brief Select the copy instruction type to use.
 *
 * Chooses between kNormal, kLDSM, kSTSM, kBulkLoad, and kBulkStore based on
 * the target capabilities and whether TMA lowering is disabled.
 *
 * \param target            Target to query.
 * \param disable_tma_lower When true, force non-TMA copy paths.
 * \return                  The selected CopyInst value.
 */

/*!
 * \brief Clone this copy operator.
 *
 * Returns a TileOperator reference that is a shallow clone of this operator
 * object suitable for further modifications in pass pipelines.
 */

/*!
 * \brief Generate lowering for bulk (global-to-shared or shared-to-global)
 * copy.
 *
 * Implements TMA-based bulk load/store lowering when `copy_inst` indicates a
 * bulk path. The function encodes TMA descriptors and produces calls or
 * loops required by the selected bulk mechanism.
 *
 * \param T         Lowering context.
 * \param analyzer  Analyzer for simplification.
 * \param copy_inst Copy instruction type indicating bulk load/store.
 * \return          A TIR `Stmt` implementing the bulk copy.
 */

/*!
 * \brief Generate lowering for LDS matrix-copy paths (LDMATRIX/STMATRIX).
 *
 * Emits the lowering for LDS-based matrix-copy instructions when the chosen
 * `copy_inst` is an LDSM or STSM variant.
 *
 * \param T         Lowering context.
 * \param analyzer  Analyzer for simplification.
 * \param copy_inst Copy instruction type indicating an LDS matrix path.
 * \return          A TIR `Stmt` implementing the matrix-copy.
 */

/*!
 * \brief Generate lowering for the normal (non-bulk, scalar/vec) copy path.
 *
 * Emits element-wise or vectorized loads/stores using the computed iteration
 * space and predicates to ensure in-bounds accesses.
 *
 * \param T        Lowering context.
 * \param analyzer Analyzer for simplification.
 * \return         A TIR `Stmt` implementing the normal copy.
 */

/*!
 * \brief Generate a SIMT-style thread-level loop for the copy.
 *
 * Produces a `For` loop that distributes copy work across SIMD/warp lanes or
 * CUDA threads according to the operator's iteration strategy.
 *
 * \param analyzer Analyzer for simplification.
 * \return         A `For` loop representing the thread-level iteration.
 */

/*!
 * \brief Compute a linear shared-memory layout suitable for TMA copies.
 *
 * Returns a `Layout` that maps the shared-memory `shared_tensor` into a
 * linearized representation required by bulk/TMA transfers.
 *
 * \param shared_tensor Buffer representing the shared-memory tensor.
 * \return              A `Layout` describing the linearized shared layout.
 */

/*!
 * \brief Create iterator variables for multi-dimensional copy loops.
 *
 * The returned `IterVar` array enumerates the loop indices used to traverse
 * the copy extents in each tensor dimension.
 *
 * \return Array of iterator variables.
 */

/*!
 * \brief Calculate source or destination indices from iteration variables.
 *
 * Converts the iterator variables (from MakeIterVars) into concrete index
 * expressions for either the source image or the destination tensor.
 *
 * \param ivs     Iterator variables returned by MakeIterVars().
 * \param src_dst 0 to produce source indices, 1 to produce destination indices.
 * \return        Array of `PrimExpr` index expressions.
 */

/*!
 * \brief Construct the boundary predicate ensuring in-bounds accesses.
 *
 * Builds a boolean expression that guards loads/stores so they only occur
 * when indices lie within the provided `extents`.
 *
 * \param analyzer Arithmetic analyzer used to simplify predicates.
 * \param ivs      Iterator variables.
 * \param extents  Extent expressions for the target buffer.
 * \param src_dst  0 = predicate for source indices, 1 = predicate for
 * destination. \return         A `PrimExpr` boolean predicate.
 */

/*!
 * \brief Constructor.
 *
 * \param args Expression arguments for the copy (indices, sizes, etc.).
 * \param vmap Buffer variable mapping for source and destination.
 */

/*!
 * \brief Get the TVM Op handle corresponding to this Copy op.
 */

/*!
 * \brief Special operator for Conv2D im2col transformation.
 *
 * Converts an input feature map into an im2col matrix layout used for GEMM-
 * based convolution lowering. Public fields configure kernel geometry,
 * stride/padding/dilation, and cache eviction behavior.
 */

/*!
 * \brief Lower to TIR statement.
 *
 * Emits TIR that performs the im2col extraction from `src` into `dst`
 * according to kernel, stride, padding, and dilation parameters.
 *
 * \param T        Lowering context with buffer bindings.
 * \param analyzer Analyzer for expression simplification and bounds reasoning.
 * \return         A TIR `Stmt` performing the im2col transform.
 */

/*!
 * \brief Infer layout for this operator.
 *
 * Produces the layout mapping for the destination im2col matrix given the
 * source layout and convolution parameters.
 *
 * \param T     Layout inference arguments.
 * \param level Inference granularity level.
 * \return      A LayoutMap with inferred layouts for affected buffers.
 */

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
  Array<PrimExpr> args_; // Copy parameters (indices, sizes, etc.)

  Buffer src, dst;                   // Source and destination buffers
  Array<Range> src_range, dst_range; // Ranges for each dimension in src and dst
  IntImm coalesced_width; // Width (in elements) for coalesced memory access
  Bool disable_tma = Bool(false); // Whether to disable TMA acceleration

  mutable ParallelOp par_op_; // Optional associated parallelization operator

  enum class EvictionPolicy {
    kEvictNormal = 0,
    kEvictFirst = 1,
    kEvictLast = 2,
  };

  int eviction_policy; // Policy for cache eviction
  static constexpr const char *_type_key = "tl.Copy";
  TVM_DECLARE_FINAL_OBJECT_INFO(CopyNode, TileOperatorNode);

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
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;

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
   * \brief Clone this copy operator.
   */
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

class Copy : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Copy, TileOperator, CopyNode);

  /*!
   * \brief Constructor.
   * \param args  Expression arguments for the copy.
   * \param vmap  Buffer variable mapping.
   */
  TVM_DLL Copy(Array<PrimExpr> args, BufferMap vmap);

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
  Buffer src, dst; // Source (input feature map) and destination (im2col matrix)
  int stride;      // Stride for convolution
  int padding;     // Padding amount
  int dilation;    // Dilation factor
  int kernel;      // Kernel size
  int eviction_policy; // Cache eviction policy
  PrimExpr nhw_step;   // Step size in NHW dimensions
  PrimExpr c_step;     // Step size in channel dimension

  static constexpr const char *_type_key = "tl.Conv2DIm2Col";
  TVM_DECLARE_FINAL_OBJECT_INFO(Conv2DIm2ColOpNode, TileOperatorNode);

  /*!
   * \brief Lower to TIR statement.
   */
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  /*!
   * \brief Infer layout for this operator.
   */
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;

  /*!
   * \brief Get TVM Op handle.
   */
  static const Op &Get();
  TileOperator Clone() const;
};

class Conv2DIm2ColOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Conv2DIm2ColOp, TileOperator,
                                Conv2DIm2ColOpNode);
  TVM_DLL Conv2DIm2ColOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_COPY_H_