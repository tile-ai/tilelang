/*!
 * \file Layout.h
 *
 */

#ifndef TVM_TL_LAYOUT_LAYOUT_H_
#define TVM_TL_LAYOUT_LAYOUT_H_

#include <cstddef>
#include <exception>
#include <string>
#include <utility>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tirx/buffer.h>

#include "support/check.h"
#include "swizzle_mode.h"

namespace tvm {
namespace tl {

// Common layout-related exceptions
class LayoutConflictException : public std::exception {
public:
  const char *what() const noexcept override { return msg_.c_str(); }
  explicit LayoutConflictException(const std::string &msg) : msg_(msg) {}

private:
  std::string msg_;
};

class LoopLayoutInjectiveException : public std::exception {
public:
  const char *what() const noexcept override { return msg_.c_str(); }
  explicit LoopLayoutInjectiveException(const std::string &msg) : msg_(msg) {}

private:
  std::string msg_;
};

class Layout;
class Fragment;
class PartialFragment;

class LayoutNode : public ffi::Object {
public:
  LayoutNode() = default;
  LayoutNode(ffi::Array<PrimExpr> input_size,
             ffi::Array<PrimExpr> forward_index);

  size_t InputDim() const { return input_size_.size(); }

  size_t OutputDim() const { return forward_index_.size(); }

  ffi::Array<PrimExpr> InputShape() const { return input_size_; }

  ffi::Array<PrimExpr> OutputShape() const;

  ffi::Array<PrimExpr> GetForwardIndex() const { return forward_index_; }

  /*!
   * \brief Convert the physical output coordinates to a row-major linear
   * index.
   *
   * For output coordinates [f0, f1, ..., fn] with shape
   * [s0, s1, ..., sn], this returns
   * (((f0 * s1 + f1) * s2 + f2) ... * sn + fn).
   */
  PrimExpr GetLinearizedForwardIndex() const;

  virtual ffi::Array<PrimExpr> GetForwardVars() const;

  virtual ffi::Array<PrimExpr> Forward(const ffi::Array<PrimExpr> &vars) const;

  // Repeat the layout along a single input dimension and prepend a new output
  // dimension that indicates the repeat-group index.
  //
  // For a layout L with input shape S and forward index F, repeating along
  // dimension `dim` with `factor` constructs a new layout L' where:
  //   - New input shape: S'[dim] = S[dim] * factor
  //   - New forward index: [i_dim // S[dim]] + F(..., i_dim % S[dim], ...)
  virtual Layout Repeat(int dim, int factor) const;

  // Expand (lift) this layout by prepending new leading input dimensions that
  // are forwarded unchanged to the output.
  //
  // For example, given a 2D layout L: [J, K] -> F(J, K), calling
  // Expand([I]) produces a 3D layout L': [I, J, K] -> [I] + F(J, K).
  //
  // `leading_shape` can contain multiple dimensions.
  virtual Layout Expand(const ffi::Array<PrimExpr> &leading_shape) const;

  virtual Layout Inverse() const;

  // Reshape the layout to a new logical shape. When aliasing buffers of
  // different dtypes, the element count may change while the underlying
  // storage footprint stays equal. Use rescale_num/rescale_den to represent
  // the ratio between the old element size and the new element size in bits.
  // Specifically, define factor = rescale_num / rescale_den where:
  //   new_num_elems = old_num_elems * factor
  // For example, f32->i8 (32b -> 8b) uses rescale_num=32, rescale_den=8.
  // i8->f32 (8b -> 32b) uses rescale_num=8, rescale_den=32.
  // Reinterpreting views rescale the last (stride-1) output mode, keeping
  // the output in the view dtype's element units; widening requires storage
  // to be contiguous and aligned at the new element width.
  virtual Layout Reshape(const ffi::Array<PrimExpr> &shape,
                         arith::Analyzer *analyzer = nullptr,
                         const PrimExpr rescale_num = Integer(1),
                         const PrimExpr rescale_den = Integer(1)) const;

  virtual std::pair<Layout, arith::IterMapLevel>
  InverseWithLevel(bool require_padding_guard = false) const;

  /*!
   * \brief Verify that distinct logical coordinates map to distinct physical
   * coordinates.
   *
   * The returned errors array is empty on success. Exact fallback checks may
   * succeed without populating the normalized iter-map indices.
   */
  virtual arith::IterMapResult
  DetectInjective(bool require_padding_guard = false) const;

  virtual std::string DebugOutput() const;

  virtual bool IsEqual(const LayoutNode *other, bool skip_index = false) const;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.Layout", LayoutNode, ffi::Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

protected:
  virtual ffi::Map<tirx::Var, Range> GetVarMap() const;
  void UpdateAnalyzer(arith::Analyzer *analyzer) const;
  ffi::Array<PrimExpr> forward_index_;
  ffi::Array<PrimExpr> input_size_;
};

/*!
 * \brief Layout reference class.
 */
class Layout : public ffi::ObjectRef {
public:
  TVM_DLL Layout(ffi::Array<tirx::IterVar> forward_var,
                 ffi::Array<PrimExpr> forward_index);
  TVM_DLL Layout(ffi::Array<PrimExpr> input_size,
                 ffi::Array<PrimExpr> forward_index);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Layout, ffi::ObjectRef,
                                             LayoutNode);
};

class FragmentNode : public LayoutNode {
public:
  FragmentNode() = default;
  FragmentNode(ffi::Array<PrimExpr> input_size,
               ffi::Array<PrimExpr> forward_index, PrimExpr forward_thread,
               PrimExpr replicate_size);

  PrimExpr GetForwardThread() const { return forward_thread_; }

  ffi::Array<PrimExpr> GetForwardVars() const final;

  Layout Inverse() const final;

  Layout Reshape(const ffi::Array<PrimExpr> &shape,
                 arith::Analyzer *analyzer = nullptr,
                 const PrimExpr rescale_num = Integer(1),
                 const PrimExpr rescale_den = Integer(1)) const;

  std::pair<Layout, arith::IterMapLevel>
  InverseWithLevel(bool require_padding_guard = false) const final;

  PrimExpr ThreadExtent() const;

  PrimExpr ReplicateExtent() const { return replicate_size_; };

  PrimExpr ForwardThread(const ffi::Array<PrimExpr> &vars,
                         const ffi::Optional<PrimExpr> &rep_var) const;

  Fragment Repeat(const ffi::Array<PrimExpr> &repeats, bool repeat_on_thread,
                  bool lower_dim_first = true) const;

  Fragment Replicate(int repeats) const;

  Fragment DeReplicate() const;

  Fragment CondenseReplicateVar() const;

  std::string DebugOutput() const override;

  Fragment BindThreadRange(Range thread_range) const;

  Range ThreadRange() const { return thread_range_; }

  bool IsEqual(const FragmentNode *other, bool skip_index = false) const;

  /*! \brief Kind guard on the base signature: a plain Fragment never equals a
   *  reducer partial (the reverse direction lives in
   *  PartialFragmentNode::IsEqual); all other comparisons keep the historical
   *  LayoutNode behavior. */
  bool IsEqual(const LayoutNode *other, bool skip_index = false) const override;

  bool IsCompletedReplicated() const;

  arith::IterMapResult
  DetectInjective(bool require_padding_guard = false) const;

  static void RegisterReflection();

  TVM_FFI_DECLARE_OBJECT_INFO("tl.Fragment", FragmentNode, LayoutNode);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

protected:
  ffi::Map<tirx::Var, Range> GetVarMap() const final;
  Range thread_range_;
  PrimExpr forward_thread_;
  PrimExpr replicate_size_;
};

/*!
 * \brief Fragment reference class.
 */
class Fragment : public Layout {
public:
  TVM_DLL Fragment(ffi::Array<tirx::IterVar> forward_var,
                   ffi::Array<PrimExpr> forward_index, PrimExpr forward_thread,
                   tirx::IterVar thread_replicate);

  TVM_DLL Fragment(ffi::Array<PrimExpr> input_size,
                   ffi::Array<PrimExpr> forward_index, PrimExpr forward_thread,
                   PrimExpr replicate_size,
                   ffi::Optional<tirx::Var> replicate_var);

  /*!
   * \brief Create a fully replicated fragment layout.
   *
   * A fully replicated fragment means all threads hold identical copies of the
   * entire buffer. This is useful for index buffers or masks that need to be
   * accessed uniformly across all threads.
   *
   * \param shape The shape of the buffer.
   * \param thread_extent The number of threads.
   * \return A Fragment where each thread has a complete copy of all elements.
   */
  TVM_DLL static Fragment FullyReplicated(ffi::Array<PrimExpr> shape,
                                          PrimExpr thread_extent);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Fragment, Layout, FragmentNode);
};

/*!
 * \brief Layout of a reducer's per-thread partials.
 *
 * Shares Fragment's algebra, but the replication coordinate is NOT uniform
 * copies of a finished value: under the low-bits convention enforced at
 * construction, `_rep % combine_size` enumerates addend lanes the finalize
 * collective must reduce, and `_rep / combine_size` enumerates equal-value
 * copy groups (from loop replication) that must never be combined. The pair
 * (storage algebra, combine_size) therefore determines the complete
 * physical plan: `combine_size == ReplicateExtent()` is the FullParticipant
 * wide plan, `combine_size == 1` is the communication-free LocalComplete
 * plan, and the collective steps are derivable from the node alone
 * (CombineSteps).
 *
 * A partial never compares equal to a plain Fragment and must not flow
 * through replica-equality shortcuts (ProveFragmentContains,
 * canonical-replica predicates). Instances live only between
 * LayoutInference and ReducerPlanAndMaterialize, keyed by `local.reducer`
 * buffers.
 *
 * Fragment methods returning new Fragment copies (Repeat/Replicate/
 * BindThreadRange/...) degrade the result to a plain Fragment; the thread
 * range is set at construction instead.
 */
class PartialFragmentNode : public FragmentNode {
public:
  PartialFragmentNode() = default;
  PartialFragmentNode(ffi::Array<PrimExpr> input_size,
                      ffi::Array<PrimExpr> forward_index,
                      PrimExpr forward_thread, PrimExpr replicate_size,
                      PrimExpr combine_size, ffi::Optional<Range> thread_range);

  /*! \brief Width of the combine coordinate: `_rep % CombineSize()` are
   *  addend lanes, `_rep / CombineSize()` are equal-value copy groups. */
  PrimExpr CombineSize() const { return combine_size_; }

  /*! \brief The collective steps (reducing_threads, scale) the finalize
   *  needs, derived from this node alone: normalize the thread expression
   *  with `_rep = lo + CombineSize() * hi` and collect the splits sourced
   *  from `lo`. Requires constant extents. */
  std::vector<std::pair<int, int>> CombineSteps() const;

  std::string DebugOutput() const override;

  /*! \brief Kind-aware equality on the base signature: a partial is never
   *  equal to a plain Fragment (dispatch through a Layout reference would
   *  otherwise fall back to LayoutNode::IsEqual, which ignores the thread
   *  mapping entirely); partial-vs-partial compares the full Fragment
   *  algebra plus the combine decomposition. */
  bool IsEqual(const LayoutNode *other, bool skip_index = false) const override;

  static void RegisterReflection();

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.PartialFragment", PartialFragmentNode,
                                    FragmentNode);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

protected:
  PrimExpr combine_size_;
};

/*!
 * \brief PartialFragment reference class.
 */
class PartialFragment : public Fragment {
public:
  TVM_DLL PartialFragment(ffi::Array<PrimExpr> input_size,
                          ffi::Array<PrimExpr> forward_index,
                          PrimExpr forward_thread, PrimExpr replicate_size,
                          PrimExpr combine_size,
                          ffi::Optional<tirx::Var> replicate_var,
                          ffi::Optional<Range> thread_range = std::nullopt);

  /*! \brief Reinterpret a solved Fragment as per-thread partials whose
   *  ENTIRE replication coordinate is addend lanes (combine ==
   *  ReplicateExtent). This is the annotation semantics: the user declares
   *  W combine lanes, copy groups come only from loop replication. */
  TVM_DLL static PartialFragment FromFragment(const Fragment &fragment);

  /*! \brief Reinterpret an induced projection (built under the low-bits
   *  convention) as per-thread partials with `combine_size` addend lanes in
   *  the low bits of the replication coordinate. */
  TVM_DLL static PartialFragment FromInduced(const Fragment &fragment,
                                             PrimExpr combine_size);

  /*! \brief The wide plan: every participant holds one full-shape partial
   *  (combine == thread_extent, no copy groups). */
  TVM_DLL static PartialFragment
  FullyReplicated(ffi::Array<PrimExpr> shape, PrimExpr thread_extent,
                  ffi::Optional<Range> thread_range = std::nullopt);

  /*! \brief The same algebraic map read as a plain Fragment: after the
   *  finalize collective every replica holds the combined value. */
  TVM_DLL Fragment AsPostCollective() const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PartialFragment, Fragment,
                                             PartialFragmentNode);
};

/*!
 * \brief CondenseReplicateVar variant that preserves the combine/copy
 * boundary of a reducer projection. The input fragment's replication
 * coordinate must follow the low-bits convention (`_rep % combine_size` =
 * addend lanes); the two halves are compressed independently and recomposed
 * with the compressed combine part back in the low bits, so the convention
 * survives condensation. Returns the condensed fragment (thread range
 * preserved) and the compressed combine width.
 */
std::pair<Fragment, PrimExpr>
CondenseReplicateVarKeepingBoundary(const Fragment &fragment,
                                    const PrimExpr &combine_size);

tirx::Var InputPlaceholder(size_t idx);
tirx::Var ReplicationPlaceholder();
tirx::IterVar MakeIterVar(std::string name, PrimExpr dom);

Fragment MakeGemmFragment8x8();
Fragment MakeGemmFragment8x8Transposed();
Fragment MakeGemmFragmentC(const int block_m, const int block_n,
                           const int warp_m, const int warp_n,
                           const int element_size);
Fragment MakeGemmSparseFragmentC(const int block_m, const int block_n,
                                 const int warp_m, const int warp_n,
                                 const int element_size);
Fragment MakeGemmFragmentCCDNA(const int block_m, const int block_n,
                               const int warp_m, const int warp_n,
                               const int element_size);
Fragment MakeGemmFragmentCHopper(const int block_m, const int block_n,
                                 const int warp_m, const int warp_n,
                                 const int element_size);
Fragment MakeGemmFragmentA(const int block_m, const int block_n,
                           const int block_k, const int warp_m,
                           const int warp_n, const int element_size,
                           bool transposed = false);
Fragment MakeGemmFragmentB(const int block_m, const int block_n,
                           const int block_k, const int warp_m,
                           const int warp_n, bool transposed = false);

Fragment MakeGemmFragmentACDNA(const int block_m, const int block_n,
                               const int block_k, const int warp_m,
                               const int warp_n, const int element_size,
                               const int k_pack, bool transposed = false);

// Default Memory Layout (row-major linear layout for any dimension)
Layout MakeLinearLayout(ffi::Array<PrimExpr> shape);
Layout MakeGemmABLayoutPadded(int stride, int continuous, int element_size);
Layout MakeGemmABLayout(int mat_stride, int mat_continuous, int continuity,
                        int element_size, bool k_inner = true);
Layout MakeGemmABLayoutHopper(int mat_stride, int mat_continuous,
                              int continuity, int element_size,
                              bool k_inner = true);
Layout MakeGemmABLayoutSm100(int mat_stride, int mat_continuous, int continuity,
                             int element_size, bool k_inner = true);
Layout MakeGemmABLayoutCDNA(int stride, int continuous, int element_size,
                            int kPack);

Fragment MakeGemmVoltaFragmentC(const int block_m, const int block_n,
                                const int warp_m, const int warp_n,
                                const int element_size);
Fragment MakeGemmVoltaFragmentA(const int block_m, const int block_n,
                                const int block_k, const int warp_m,
                                const int warp_n);
Layout MakeGemmVoltaABLayout(int stride, int continuous, bool is_a,
                             bool k_inner = true);

Layout MakeTensorOpMultiplicand(int mat_stride, int mat_continuous,
                                int elementsize, int crosswise);
Layout MakeGemmSparseAmpereABLayout(int mat_stride, int mat_continuous,
                                    int elementsize);

Layout MakeSwizzledLayout(const tirx::Buffer &buffer, bool k_inner = true,
                          bool allow_pad = true);
Layout MakeVoltaSwizzledLayout(const tirx::Buffer &buffer, bool is_a = true,
                               bool k_inner = true);
Layout MakeWgmmaSwizzledLayout(const tirx::Buffer &buffer, int continuity = -1,
                               bool k_inner = true);
Layout MakeTcgen05MmaSwizzledLayout(const tirx::Buffer &buffer,
                                    int continuity = -1, bool k_inner = true);
Layout MakeFullBankSwizzleLayout(const tirx::Buffer &buffer);
Layout MakeHalfBankSwizzleLayout(const tirx::Buffer &buffer);
Layout MakeQuarterBankSwizzleLayout(const tirx::Buffer &buffer);

// Detect which swizzle mode a layout uses
SwizzleMode DetectSwizzleMode(const Layout &layout, const tirx::Buffer &buffer);

// Merge two swizzle layouts by taking the smaller granularity
// Returns NullOpt if either layout is not a swizzle layout
ffi::Optional<Layout> MergeSwizzleLayouts(const Layout &layout1,
                                          const Layout &layout2,
                                          const tirx::Buffer &buffer);

namespace attr {
// BlockAttr, Containing the layout for all the buffers in the block
constexpr const char *kLayoutMap = "layout_map";
// ForAttr, Containing the parallel loop layout for a parallel for loop
constexpr const char *kParallelLoopLayout = "parallel_loop_layout";
// ForAttr, Containing the predicate for a parallel for loop
constexpr const char *kParallelLoopPredicate = "parallel_loop_predicate";
// ForAttr, Marks a ragged SIMT loop layout that needs guarded inverse lowering
constexpr const char *kParallelLoopRequiresPaddingGuard =
    "parallel_loop_requires_padding_guard";
// ForAttr, Width (in elements) for coalesced memory access
constexpr const char *kCoalescedWidth = "coalesced_width";
} // namespace attr

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_LAYOUT_H_
