/*!
 * \file Layout.h
 *
 */

#ifndef TVM_TL_LAYOUT_LAYOUT_H_
#define TVM_TL_LAYOUT_LAYOUT_H_

#include <exception>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/ffi/object.h>
#include <utility>

#include "../support/ffi_aliases.h"

namespace tvm {
namespace tl {

using namespace tir;

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

class LayoutNode : public Object {
public:
  LayoutNode() = default;
  LayoutNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index);

  size_t InputDim() const { return input_size_.size(); }

  size_t OutputDim() const { return forward_index_.size(); }

  Array<PrimExpr> InputShape() const { return input_size_; }

  Array<PrimExpr> OutputShape() const;

  Array<PrimExpr> GetForwardIndex() const { return forward_index_; }

  virtual Array<PrimExpr> GetForwardVars() const;

  virtual Array<PrimExpr> Forward(const Array<PrimExpr> &vars) const;

  virtual Layout Inverse() const;

  // Reshape the layout to a new logical shape. When aliasing buffers of
  // different dtypes, the element count may change while the underlying
  // byte-size stays equal. Use rescale_num/rescale_den to represent the
  // ratio between the old element size and the new element size in bytes.
  // Specifically, define factor = rescale_num / rescale_den where:
  //   new_num_elems = old_num_elems * factor
  // For example, f32->i8 (4B -> 1B) uses rescale_num=4, rescale_den=1.
  // i8->f32 (1B -> 4B) uses rescale_num=1, rescale_den=4.
  virtual Layout Reshape(const Array<PrimExpr> &shape,
                         arith::Analyzer *analyzer,
                         const PrimExpr rescale_num = Integer(1),
                         const PrimExpr rescale_den = Integer(1)) const;

  virtual std::pair<Layout, arith::IterMapLevel> InverseWithLevel() const;

  virtual std::string DebugOutput() const;

  virtual bool IsEqual(const LayoutNode *other, bool skip_index = false) const;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.Layout", LayoutNode, Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

protected:
  virtual Map<Var, Range> getVarMap() const;
  void UpdateAnalyzer(arith::Analyzer *analyzer) const;
  Array<PrimExpr> forward_index_;
  Array<PrimExpr> input_size_;
};

/*!
 * \brief Layout reference class.
 */
class Layout : public ObjectRef {
public:
  TVM_DLL Layout(Array<IterVar> forward_var, Array<PrimExpr> forward_index);
  TVM_DLL Layout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Layout, ObjectRef, LayoutNode);
};

class FragmentNode : public LayoutNode {
public:
  FragmentNode() = default;
  FragmentNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
               PrimExpr forward_thread, PrimExpr replicate_size);

  PrimExpr GetForwardThread() const { return forward_thread_; }

  Array<PrimExpr> GetForwardVars() const final;

  Layout Inverse() const final;

  Layout Reshape(const Array<PrimExpr> &shape, arith::Analyzer *analyzer,
                 const PrimExpr rescale_num = Integer(1),
                 const PrimExpr rescale_den = Integer(1)) const;

  std::pair<Layout, arith::IterMapLevel> InverseWithLevel() const final;

  PrimExpr ThreadExtent() const;

  PrimExpr ReplicateExtent() const { return replicate_size_; };

  PrimExpr ForwardThread(const Array<PrimExpr> &vars,
                         const Optional<PrimExpr> &rep_var) const;

  Fragment Repeat(const Array<PrimExpr> &repeats, bool repeat_on_thread,
                  bool lower_dim_first = true) const;

  Fragment Replicate(int repeats) const;

  Fragment DeReplicate() const;

  Fragment CondenseReplicateVar() const;

  std::string DebugOutput() const final;

  Fragment BindThreadRange(Range thread_range) const;

  Range ThreadRange() const { return thread_range_; }

  bool IsEqual(const FragmentNode *other, bool skip_index = false) const;

  bool IsCompletedReplicated() const;

  arith::IterMapResult DetectInjective() const;

  static void RegisterReflection();

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Fragment", FragmentNode, LayoutNode);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

protected:
  Map<Var, Range> getVarMap() const final;
  Range thread_range_;
  PrimExpr forward_thread_;
  PrimExpr replicate_size_;
};

/*!
 * \brief Fragment reference class.
 */
class Fragment : public Layout {
public:
  TVM_DLL Fragment(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, IterVar thread_replicate);

  TVM_DLL Fragment(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, PrimExpr replicate_size,
                   Optional<Var> replicate_var);

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
  TVM_DLL static Fragment FullyReplicated(Array<PrimExpr> shape,
                                          PrimExpr thread_extent);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Fragment, Layout, FragmentNode);
};

Var InputPlaceholder(size_t idx);
Var ReplicationPlaceholder();
IterVar make_itervar(std::string name, PrimExpr dom);

Fragment makeGemmFragment8x8();
Fragment makeGemmFragment8x8Transposed();
Fragment makeGemmFragmentC(const int block_m, const int block_n,
                           const int warp_m, const int warp_n,
                           const int element_size);
Fragment makeGemmSparseFragmentC(const int block_m, const int block_n,
                                 const int warp_m, const int warp_n,
                                 const int element_size);
Fragment makeGemmFragmentCCDNA(const int block_m, const int block_n,
                               const int warp_m, const int warp_n,
                               const int element_size);
Fragment makeGemmFragmentCHopper(const int block_m, const int block_n,
                                 const int warp_m, const int warp_n,
                                 const int element_size);
Fragment makeGemmFragmentA(const int block_m, const int block_n,
                           const int block_k, const int warp_m,
                           const int warp_n, const int element_size,
                           bool transposed = false);
Fragment makeGemmFragmentB(const int block_m, const int block_n,
                           const int block_k, const int warp_m,
                           const int warp_n, bool transposed = false);

Fragment makeGemmFragmentACDNA(const int block_m, const int block_n,
                               const int block_k, const int warp_m,
                               const int warp_n, const int element_size,
                               const int k_pack, bool transposed = false);

// Default Memory Layout (row-major linear layout for any dimension)
Layout makeLinearLayout(Array<PrimExpr> shape);
Layout makeGemmABLayoutPadded(int stride, int continuous, int element_size);
Layout makeGemmABLayout(int mat_stride, int mat_continuous, int continuity,
                        int element_size, bool k_inner = true);
Layout makeGemmABLayoutHopper(int mat_stride, int mat_continuous,
                              int continuity, int element_size,
                              bool k_inner = true);
Layout makeGemmABLayoutSm100(int mat_stride, int mat_continuous, int continuity,
                             int element_size, bool k_inner = true);
Layout makeGemmABLayoutCDNA(int stride, int continuous, int element_size,
                            int kPack);

Fragment makeGemmVoltaFragmentC(const int block_m, const int block_n,
                                const int warp_m, const int warp_n,
                                const int element_size);
Fragment makeGemmVoltaFragmentA(const int block_m, const int block_n,
                                const int block_k, const int warp_m,
                                const int warp_n);
Layout makeGemmVoltaABLayout(int stride, int continuous, bool is_a,
                             bool k_inner = true);

Layout makeTensorOpMultiplicand(int mat_stride, int mat_continuous,
                                int elementsize, int crosswise);
Layout makeGemmSparseAmpereABLayout(int mat_stride, int mat_continuous,
                                    int elementsize);

Layout makeFullBankSwizzleLayout(int stride, int continuous, int element_size);
Layout makeHalfBankSwizzleLayout(int stride, int continuous, int element_size);
Layout makeQuarterBankSwizzleLayout(int stride, int continuous,
                                    int element_size);

namespace attr {
// BlockAttr, Containing the layout for all the buffers in the block
constexpr const char *kLayoutMap = "layout_map";
} // namespace attr

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_LAYOUT_H_
