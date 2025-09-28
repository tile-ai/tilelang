/*!
 * \file layout/tcgen05_layout.cc
 * \brief Define Layout used in tcgen05.ld/st.
 *
 */

#include <tvm/tir/stmt_functor.h>

#include <cmath>

#include "layout.h"
#include "tcgen05_layout.h"

namespace tvm {
namespace tl {

static IterVar make_itervar(std::string name, Range dom) {
  Var var = Var(name, dom->min->dtype);
  return IterVar(dom, var, IterVarType::kDataPar);
}

Tcgen05Meta getTcgen05Meta_32dp32b() {
  constexpr int INST_WIDTH = 1;
  IterVar inst_row = make_itervar("row", 128);
  IterVar inst_col = make_itervar("col", INST_WIDTH);
  return Tcgen05Meta{"tl::tcgen05_ld_32dp32bNx",
                     Fragment({inst_row, inst_col}, {inst_col}, {inst_row},
                              make_itervar("rep", Range(0, 1))),
                     INST_WIDTH};
}

Tcgen05Meta getTcgen05Meta_32dp64b() {
  constexpr int INST_WIDTH = 2;
  IterVar inst_row = make_itervar("row", 128);
  IterVar inst_col = make_itervar("col", INST_WIDTH);
  return Tcgen05Meta{
      "tl::tcgen05_ld_32dp64bNx",
      Fragment({inst_row, inst_col}, {FloorDiv(FloorMod(inst_row, 32), 16)},
               {FloorDiv(inst_row, 32) * 32 + FloorMod(inst_row, 8) * 4 +
                FloorDiv(FloorMod(inst_row, 16), 8) +
                FloorMod(inst_col, 2) * 2},
               make_itervar("rep", Range(0, 1))),
      INST_WIDTH};
}

Tcgen05Meta getTcgen05Meta_32dp128b() {
  constexpr int INST_WIDTH = 4;
  IterVar inst_row = make_itervar("row", 128);
  IterVar inst_col = make_itervar("col", INST_WIDTH);
  return Tcgen05Meta{
      "tl::tcgen05_ld_32dp128bNx",
      Fragment({inst_row, inst_col}, {FloorDiv(FloorMod(inst_row, 32), 8)},
               {FloorDiv(inst_row, 32) * 32 + FloorMod(inst_row, 8) * 4 +
                FloorMod(inst_col, 4)},
               make_itervar("rep", Range(0, 1))),
      INST_WIDTH};
}

Tcgen05Meta getTcgen05Meta_32dp256b() {
  constexpr int INST_WIDTH = 8;
  IterVar inst_row = make_itervar("row", 128);
  IterVar inst_col = make_itervar("col", INST_WIDTH);
  return Tcgen05Meta{
      "tl::tcgen05_ld_32dp256bNx",
      Fragment(
          {inst_row, inst_col},
          {FloorMod(inst_col, 2) + FloorDiv(FloorMod(inst_row, 32), 8) * 2},
          {FloorDiv(inst_row, 32) * 32 + FloorMod(inst_row, 8) * 4 +
           FloorDiv(FloorMod(inst_col, 8), 2)},
          make_itervar("rep", Range(0, 1))),
      INST_WIDTH};
}

std::tuple<bool, Fragment, int>
expandTcgen05Layout(const Tcgen05Meta &meta, int tmem_phy_col_extent,
                    int num_threads, Range row_dom, Range col_dom) {
  static constexpr int WARPGROUP_SIZE = 128;
  ICHECK(num_threads % WARPGROUP_SIZE == 0);
  int num_wgs = num_threads / WARPGROUP_SIZE;

#define FAIL_IF(cond)                                                          \
  if (cond) {                                                                  \
    return {false, Fragment(), 0};                                             \
  }

  FAIL_IF(tmem_phy_col_extent % meta.width != 0);
  int total_chunks = tmem_phy_col_extent / meta.width;
  FAIL_IF(total_chunks % num_wgs != 0); // Otherwise the layout is not bijective
  int num_chunks_each_wg = total_chunks / num_wgs;
  int num_cols_each_wg = num_chunks_each_wg * meta.width;
  int num_elems_each_thread_in_one_chunk = meta.width * 128 / WARPGROUP_SIZE;

  IterVar iter_row = make_itervar("row", row_dom);
  IterVar iter_col = make_itervar("col", col_dom);
  PrimExpr thread_idx =
      meta.frag->ForwardThread({iter_row, FloorMod(iter_col, meta.width)},
                               std::nullopt) +
      FloorDiv(iter_col, num_cols_each_wg) * WARPGROUP_SIZE;
  PrimExpr val_idx =
      meta.frag->Forward({iter_row, FloorMod(iter_col, meta.width)})[0] +
      FloorDiv(FloorMod(iter_col, num_cols_each_wg), meta.width) *
          num_elems_each_thread_in_one_chunk;

  return {true,
          Fragment({iter_row, iter_col}, {val_idx}, thread_idx,
                   make_itervar("rep", Range(0, 1))),
          num_chunks_each_wg};
}

} // namespace tl
} // namespace tvm
