/*!
 * \file layout_reducer.cc
 *
 * Legacy (v1) reducer metadata support. The former LayoutReducer pass —
 * which forced fragment layouts onto reducer buffers before layout
 * inference — has been removed: legacy reducer syntax is canonicalized into
 * first-class reducer v2 ops by CanonicalizeLegacyReducer, and physical
 * storage is chosen by ReducerPlanAndMaterialize after LayoutInference.
 *
 * What remains here is the ReducerInfo annotation type used by the legacy
 * frontend annotation (consumed by CanonicalizeLegacyReducer) and by the
 * data-race verifier's reducer exemption.
 */

#include "layout_reducer.h"

#include <tvm/runtime/logging.h>

namespace tvm {
namespace tl {

using namespace ffi;

ReducerInfoNode::ReducerInfoNode(const String &op_str, const String &rep_str) {
  if (op_str == "sum")
    op = ReducerOpType::SUM;
  else if (op_str == "max")
    op = ReducerOpType::MAX;
  else if (op_str == "min")
    op = ReducerOpType::MIN;
  else
    ICHECK(false) << "Unrecognized reducer_info op: " << op_str;

  if (rep_str == "all")
    rep = ReducerRepType::ALL;
  else if (rep_str == "none")
    rep = ReducerRepType::NONE;
  else
    ICHECK(false) << "Unrecognized reducer_info rep: " << rep_str;
}

} // namespace tl
} // namespace tvm
