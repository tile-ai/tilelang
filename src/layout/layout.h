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
  // For sub-byte subtype views, the output layout may temporarily gain or drop
  // a trailing "pack lane" dimension so that the layout still describes how
  // multiple logical elements share the same physical storage slot.
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

  std::string DebugOutput() const final;

  Fragment BindThreadRange(Range thread_range) const;

  Range ThreadRange() const { return thread_range_; }

  bool IsEqual(const FragmentNode *other, bool skip_index = false) const;

  bool IsCompletedReplicated() const;

  arith::IterMapResult
  DetectInjective(bool require_padding_guard = false) const;

  static void RegisterReflection();

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Fragment", FragmentNode, LayoutNode);
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
