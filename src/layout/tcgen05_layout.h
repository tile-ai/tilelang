/*!
 * \file layout/tcgen05_layout.cc
 *
 */
#pragma once

#include "layout.h"

namespace tvm {
namespace tl {

// A structure encapsulating the metadata for a particular tcgen05.ld/st
// instruction.
struct Tcgen05Meta {
  std::string intrinsics_name;
  Fragment frag; // Physical tmem coord |-> (thread_id, val_id) in fragment
  int width;
};

// Obtain the metadata for tcgen05.ld instructions.
Tcgen05Meta GetTcgen05MetaLd32Dp32B();
Tcgen05Meta GetTcgen05MetaLd32Dp64B();
Tcgen05Meta GetTcgen05MetaLd32Dp128B();
Tcgen05Meta GetTcgen05MetaLd32Dp256B();

// Obtain the metadata for tcgen05.st instructions.
Tcgen05Meta GetTcgen05MetaSt32Dp32B();
Tcgen05Meta GetTcgen05MetaSt32Dp64B();
Tcgen05Meta GetTcgen05MetaSt32Dp128B();
Tcgen05Meta GetTcgen05MetaSt32Dp256B();

// Expand a tcgen05 layout along thread_idx/value_idx (T/V) dimensions.
// Return {is_success, fragment, num_chunks_each_wg}
std::tuple<bool, Fragment, int>
ExpandTcgen05Layout(const Tcgen05Meta &meta, int tmem_phy_col_extent,
                    int num_threads, Range row_dom, Range col_dom);

} // namespace tl
} // namespace tvm
