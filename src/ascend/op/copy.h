/*!
 * \file tl/ascend/op/copy.h
 * \brief Ascend dialect copy operator.
 *
 * The Ascend frontend emits `tl.tileop.ascend_copy`, whose TLOpBuilder
 * constructs the typed AscendCopyNode below. The Call's annotations map stays
 * the durable IR encoding of every Ascend copy hint; this node is a transient
 * typed view decoded once per parse.
 */

#ifndef TVM_TL_ASCEND_OP_COPY_H_
#define TVM_TL_ASCEND_OP_COPY_H_

#include "op/copy.h"

#include <tvm/ir/cow.h>

namespace tvm {
namespace tl {

/*!
 * \brief Typed Ascend view of a tile copy.
 *
 * Extends the shared CopyNode with typed decodes of the Ascend lowering
 * hints that ride in the Call annotations.
 */
class AscendCopyNode : public CopyNode {
public:
  // Typed views of the Ascend copy hint annotations, decoded once at
  // construction. Deliberately not reflected: the reflected `annotations`
  // map on the base node remains the single durable encoding.
  int nd2nz{0};        // "nd2nz"
  int dual_dst_ctl{0}; // "dual_dst_ctl": L0C->UB dual-destination mode
  int transpose{0};    // "transpose": GM->L1 dn2nz / L1->L0 transpose request
  int data_select{0};  // "data_select": GM->UB hardware right-padding
  // Canonical allocation buffer selected from alloc_l0*_sf's IR binding.
  // Leading indices already include any materialized physical versions.
  Optional<Buffer> mx_sf_data;
  PrimExpr unit_flag_ctl;          // "unit_flag_ctrl", defaults to 0
  PrimExpr sub_blockid;            // "sub_blockid", defaults to 0
  Optional<Integer> l2_cache_ctrl; // "l2_cache_ctrl"; defaults differ per path
  Optional<PrimExpr> pad_value;    // "pad_value": GM->UB pad fill value

  /*!
   * \brief l2_cache_ctrl with a path-specific default (GM loads default to 0,
   *        UB->GM stores to NOTALLOC_KEEP).
   */
  int L2CacheCtrlOr(int default_value) const {
    if (l2_cache_ctrl.defined()) {
      return static_cast<int>(l2_cache_ctrl.value()->value);
    }
    return default_value;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.AscendCopy", AscendCopyNode, CopyNode);

  static void RegisterReflection() { reflection::ObjectDef<AscendCopyNode>(); }

  AscendCopyNode() = default;
  /*! \brief Copy the base state verbatim; annotation decoding happens in
   *  AscendCopy(const CopyNode &). */
  explicit AscendCopyNode(const CopyNode &base) : CopyNode(base) {}

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;

  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;

protected:
  TileOperator Clone() const override;
};

class AscendCopy : public Copy {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AscendCopy, Copy, AscendCopyNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AscendCopyNode);

  /*!
   * \brief Constructor from tl.tileop.ascend_copy call arguments.
   * \param args args[0]/args[1] are the source/destination regions.
   * \param annotations Annotations map from the Call node.
   */
  TVM_DLL
  AscendCopy(Array<PrimExpr> args,
             Map<String, ObjectRef> annotations = Map<String, ObjectRef>());

  /*!
   * \brief Upgrade a base copy into the typed Ascend node.
   *
   * Plain tl.tileop.copy calls synthesized by shared passes (e.g.
   * ReducerPlanAndMaterialize) parse into the base CopyNode; this re-decodes
   * their annotations so both spellings converge on one implementation.
   */
  TVM_DLL explicit AscendCopy(const CopyNode &base);

  /*!
   * \brief Get the TVM Op handle for tl.tileop.ascend_copy.
   */
  static const Op &Get();
};

/*!
 * \brief True for calls the Ascend pipeline must treat as tile copies: the
 *        dialect's tl.tileop.ascend_copy plus plain tl.tileop.copy calls
 *        synthesized by shared passes.
 */
bool IsAscendCopyCall(const tirx::CallNode *call);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_COPY_H_
