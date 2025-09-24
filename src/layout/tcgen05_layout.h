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

// Obtain the metadata for tcgen05.ld/st instructions.
Tcgen05Meta getTcgen05Meta_32dp32b();
Tcgen05Meta getTcgen05Meta_32dp64b();
Tcgen05Meta getTcgen05Meta_32dp128b();
Tcgen05Meta getTcgen05Meta_32dp256b();

// Expand a tcgen05 layout along thread_idx/value_idx (T/V) dimensions.
// Return {is_success, fragment, num_chunks_each_wg}
std::tuple<bool, Fragment, int>
expandTcgen05Layout(const Tcgen05Meta &meta, int tmem_phy_col_extent,
                    int num_threads, Range row_dom, Range col_dom);

} // namespace tl
} // namespace tvm
