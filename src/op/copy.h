/*!
 * \file tl/op/copy.h
 * \brief Copy operations and Tensor Memory Access (TMA) descriptors
 */

#ifndef TVM_TL_OP_COPY_H_
#define TVM_TL_OP_COPY_H_

#include "operator.h"
#include "parallel.h"

namespace tvm {
namespace tl {
using namespace tir;

/// Copy instruction types for different memory access patterns
enum class CopyInst : uint8_t {
  kNormal = 0,    // utilize ldg/stg or cpasync or any buffer copy
  kLDSM = 1,      // ldmatrix memory copy
  kSTSM = 2,      // stmatrix memory copy
  kBulkLoad = 3,  // utilize tma load
  kBulkStore = 4, // utilize tma store
  // we should separate the bulk load and store for 1d and multi-dim
  // as they have different memory access patterns
  kBulkLoad1D = 5,  // utilize tma load 1d
  kBulkStore1D = 6, // utilize tma store 1d
  kTMemLoad = 7,    // tcgen05.ld (tensor memory -> register)
  kTMemStore = 8,   // tcgen05.st (register -> tensor memory)
};

/// Convert CopyInst enum to string for debugging
inline const char *CopyInstToString(CopyInst inst) {
  switch (inst) {
  case CopyInst::kNormal:
    return "Normal";
  case CopyInst::kLDSM:
    return "LDSM";
  case CopyInst::kSTSM:
    return "STSM";
  case CopyInst::kBulkLoad:
    return "BulkLoad";
  case CopyInst::kBulkStore:
    return "BulkStore";
  case CopyInst::kBulkLoad1D:
    return "BulkLoad1D";
  case CopyInst::kBulkStore1D:
    return "BulkStore1D";
  case CopyInst::kTMemLoad:
    return "TMemLoad";
  case CopyInst::kTMemStore:
    return "TMemStore";
  default:
    return "Unknown";
  }
}

/// Descriptor for Tensor Memory Access (TMA) copy operations
struct TMADesc {
  size_t rank;                   ///< Tensor rank (number of dimensions)
  int data_type;                 ///< Data type identifier
  Array<PrimExpr> global_shape;  ///< Shape in global memory
  Array<PrimExpr> global_stride; ///< Strides in global memory
  Array<PrimExpr> smem_box;      ///< Block shape in shared memory
  Array<PrimExpr> smem_stride;   ///< Strides in shared memory
  PrimExpr global_addr;          ///< Base address in global memory
  int swizzle;                   ///< Memory layout swizzle parameter
  int interleave;                ///< Memory interleave parameter
  int oob_fill;                  ///< Out-of-bound fill policy
  int l2_promotion;              ///< L2 cache promotion flag

  /// Encode descriptor fields into runtime call arguments
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
  IntImm coalesced_width; // Width (in elements) for coalesced memory access
  Bool disable_tma = Bool(false); // Whether to disable TMA acceleration

  mutable ParallelOp par_op_; // Optional associated parallelization operator

  enum class EvictionPolicy : uint8_t {
    kEvictNormal = 0,
    kEvictFirst = 1,
    kEvictLast = 2,
  };

  uint8_t eviction_policy; // Policy for cache eviction
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Copy", CopyNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CopyNode>()
        .def_ro("src", &CopyNode::src)
        .def_ro("dst", &CopyNode::dst)
        .def_ro("src_range", &CopyNode::src_range)
        .def_ro("dst_range", &CopyNode::dst_range)
        .def_ro("coalesced_width", &CopyNode::coalesced_width);
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
   * \brief Check if bulk copy is supported.
   */
  bool CheckBulkLoad(Target target, arith::Analyzer *analyzer,
                     bool check_last_dim = true) const;

  /*!
   * \brief Check if bulk store is supported.
   */
  bool CheckBulkStore(Target target, arith::Analyzer *analyzer,
                      bool check_last_dim = true) const;

  /*!
   * \brief Check if bulk copy 1d load is supported.
   */
  bool CheckBulkLoad1D(Target target, const LayoutMap &layout_map,
                       arith::Analyzer *analyzer) const;

  /*!
   * \brief Check if bulk copy 1d store is supported.
   */
  bool CheckBulkStore1D(Target target, const LayoutMap &layout_map,
                        arith::Analyzer *analyzer) const;

  /*!
   * \brief Check if bulk copy 1d is supported.
   */
  bool CheckBulkCopy1D(const Buffer &global_tensor, const Buffer &shared_tensor,
                       const Array<Range> &global_range,
                       const Array<Range> &shared_range,
                       const LayoutMap &layout_map,
                       arith::Analyzer *analyzer) const;

  /*!
   * \brief Check if lds memory copy is supported.
   */
  bool CheckLDSMCopy(Target target) const;

  /*!
   * \brief Check if stsm memory copy is supported.
   */
  bool CheckSTSMCopy(Target target) const;

  /*!
   * \brief Check if tensor memory load is supported.
   */
  bool CheckTMemLoad(Target target) const;

  /*!
   * \brief Check if tensor memory store is supported.
   */
  bool CheckTMemStore(Target target) const;

  /*!
   * \brief Get the copy instruction type.
   */
  CopyInst GetCopyInst(Target target, bool disable_tma_lower,
                       const LayoutMap &layout_map, arith::Analyzer *analyzer,
                       bool buffer_oob) const;

protected:
  /*!
   * \brief Generate lowering for bulk/global-to-shared copy.
   */
  Stmt LowerBulkCopy(const LowerArgs &T, arith::Analyzer *analyzer,
                     CopyInst copy_inst) const;

  /*!
   * \brief Generate lowering for bulk copy 1d.
   */
  Stmt LowerBulkCopy1D(const LowerArgs &T, arith::Analyzer *analyzer,
                       CopyInst copy_inst) const;

  /*!
   * \brief Generate lowering for LDS Memory Copy (shared memory to shared
   * memory or smem usage).
   */
  Stmt LowerLDSMCopy(const LowerArgs &T, arith::Analyzer *analyzer,
                     CopyInst copy_inst) const;

  /*!
   * \brief Generate lowering for tensor memory copy (tcgen05.ld/st/cp).
   */
  Stmt LowerTmemCopy(const LowerArgs &T, arith::Analyzer *analyzer) const;

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

private:
  /*!
   * \brief Collect fragment buffers from expression and create fully replicated
   * layouts.
   *
   * Recursively searches the expression for BufferLoad nodes with
   * "local.fragment" scope, following let bindings. For each found fragment
   * buffer, creates a fully replicated layout and adds it to result_map.
   *
   * \param expr            Expression to search.
   * \param let_var_to_expr Map from let variables to their bound expressions.
   * \param existing_layouts Existing layout map to check for already-inferred
   * layouts. \param thread_extent   Number of threads for replication. \param
   * thread_bounds   Thread bounds for binding the layout. \param result_map
   * Output map to store collected fragment layouts.
   */
  void CollectFragmentLayouts(const PrimExpr &expr,
                              const Map<Var, PrimExpr> &let_var_to_expr,
                              const LayoutMap &existing_layouts,
                              PrimExpr thread_extent, Range thread_bounds,
                              Map<Buffer, Layout> &result_map) const;
};

class Copy : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Copy, TileOperator, CopyNode);

  /*!
   * \brief Constructor.
   * \param args  Expression arguments for the copy.
   * \param vmap  Buffer variable mapping.
   */
  TVM_DLL Copy(Array<PrimExpr> args);

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
  int eviction_policy_; // Cache eviction policy
  PrimExpr nhw_step_;   // Step size in NHW dimensions
  PrimExpr c_step_;     // Step size in channel dimension

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

class Conv2DIm2ColOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Conv2DIm2ColOp, TileOperator,
                                             Conv2DIm2ColOpNode);
  TVM_DLL Conv2DIm2ColOp(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_COPY_H_
