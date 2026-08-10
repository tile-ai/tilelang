/*!
 * \file layout/tcgen05_layout.cc
 * \brief tcgen05.ld/st data-movement shapes as CuTe TV atoms.
 *
 * Each atom is the CUTLASS ``Copy_Traits<SM100_TMEM_LOAD_*>`` TV layout
 * over (datapath, b32 column), matching the PTX data-movement shapes
 * (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-layout).
 * Loads and stores share one shape per width.
 */

#include "support/check.h"
#include <algorithm>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "layout.h"
#include "tcgen05_layout.h"

namespace tvm {
namespace tl {

using namespace tirx;
using tvm::ffi::Array;

const cute::Layout kDpOnly = cute::Layout::Parse("(1,1):(1,0)");
const cute::Layout kColOnly = cute::Layout::Parse("(1,1):(0,1)");

namespace {

IterVar MakeIterVar(std::string name, Range dom) {
  Var var = Var(name, dom->min->dtype);
  return IterVar(dom, var, IterVarType::kDataPar);
}

// kSerialize: (datapath, column) -> serialized TMEM, the flat address
// datapath + 128 * column.
const cute::Layout kSerialize = cute::Layout::Parse("(1,1):(1,128)");

// The atoms are the CUTLASS ``Copy_Traits<...1x>`` TV layouts, written over
// axis 0 = datapath ("@0"), axis 1 = b32 column ("@1").  Each covers one
// PTX issue of one warp.

Tcgen05Meta MakeTcgen05Meta_32dp32b(bool is_store) {
  // PTX 32x32b: lane t -> datapath t; one register on one column per
  // repetition; wrapper chaining extends repetitions exactly, so any N.
  return Tcgen05Meta(is_store ? "tl::tcgen05_st_32dp32bNx"
                              : "tl::tcgen05_ld_32dp32bNx",
                     cute::Layout::Parse("(32,1):(1@0,0)"),
                     /*max_chunks=*/0);
}

Tcgen05Meta MakeTcgen05Meta_16dp64b(bool is_store) {
  // PTX 16x64b: lane t -> datapath 8*(t%2) + t/4, column (t/2)%2; one
  // register per repetition.
  return Tcgen05Meta{is_store ? "tl::tcgen05_st_32dp64bNx"
                              : "tl::tcgen05_ld_32dp64bNx",
                     cute::Layout::Parse("((2,2,8),1):((8@0,1@1,1@0),0)"),
                     /*max_chunks=*/128};
}

Tcgen05Meta MakeTcgen05Meta_16dp128b(bool is_store) {
  // PTX 16x128b: lane t -> datapath t/4, column t%4; two registers stepping
  // the 8-datapath half.
  return Tcgen05Meta{is_store ? "tl::tcgen05_st_32dp128bNx"
                              : "tl::tcgen05_ld_32dp128bNx",
                     cute::Layout::Parse("((4,8),2):((1@1,1@0),8@0)"),
                     /*max_chunks=*/64};
}

Tcgen05Meta MakeTcgen05Meta_16dp256b(bool is_store) {
  // PTX 16x256b: lane t -> datapath t/4, column 2*(t%4); four registers as
  // (adjacent column, 8-datapath half).
  return Tcgen05Meta{is_store ? "tl::tcgen05_st_32dp256bNx"
                              : "tl::tcgen05_ld_32dp256bNx",
                     cute::Layout::Parse("((4,8),(2,2)):((2@1,1@0),(1@1,8@0))"),
                     /*max_chunks=*/32};
}

} // namespace

Tcgen05Meta::Tcgen05Meta(ffi::String intrinsics_name, cute::Layout tv,
                         int64_t max_chunks) {
  auto node = ffi::make_object<Tcgen05MetaNode>();
  node->intrinsics_name = std::move(intrinsics_name);
  node->tv = std::move(tv);
  node->max_chunks = max_chunks;
  data_ = std::move(node);
}

void Tcgen05MetaNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Tcgen05MetaNode>()
      .def_ro("intrinsics_name", &Tcgen05MetaNode::intrinsics_name)
      .def_ro("tv", &Tcgen05MetaNode::tv)
      .def_ro("max_chunks", &Tcgen05MetaNode::max_chunks);
}

Tcgen05Meta GetTcgen05MetaLd32Dp32B() { return MakeTcgen05Meta_32dp32b(false); }
Tcgen05Meta GetTcgen05MetaLd16Dp64B() { return MakeTcgen05Meta_16dp64b(false); }
Tcgen05Meta GetTcgen05MetaLd16Dp128B() {
  return MakeTcgen05Meta_16dp128b(false);
}
Tcgen05Meta GetTcgen05MetaLd16Dp256B() {
  return MakeTcgen05Meta_16dp256b(false);
}

Tcgen05Meta GetTcgen05MetaSt32Dp32B() { return MakeTcgen05Meta_32dp32b(true); }
Tcgen05Meta GetTcgen05MetaSt16Dp64B() { return MakeTcgen05Meta_16dp64b(true); }
Tcgen05Meta GetTcgen05MetaSt16Dp128B() {
  return MakeTcgen05Meta_16dp128b(true);
}
Tcgen05Meta GetTcgen05MetaSt16Dp256B() {
  return MakeTcgen05Meta_16dp256b(true);
}

int64_t Tcgen05AtomDatapaths(const Tcgen05Meta &meta) {
  return cute::AsConst(cute::Cosize(cute::Composition(kDpOnly, meta->tv)));
}

int64_t Tcgen05AtomWidth(const Tcgen05Meta &meta) {
  return cute::AsConst(cute::Cosize(cute::Composition(kColOnly, meta->tv)));
}

Tcgen05CopyPlan::Tcgen05CopyPlan(cute::Layout fragment,
                                 int64_t num_chunks_each_wg,
                                 cute::Layout rest_domain, int64_t num_issues,
                                 int64_t vals_per_issue,
                                 int64_t datapaths_per_warp) {
  auto node = ffi::make_object<Tcgen05CopyPlanNode>();
  node->fragment = std::move(fragment);
  node->num_chunks_each_wg = num_chunks_each_wg;
  node->rest_domain = std::move(rest_domain);
  node->num_issues = num_issues;
  node->vals_per_issue = vals_per_issue;
  node->datapaths_per_warp = datapaths_per_warp;
  data_ = std::move(node);
}

void Tcgen05CopyPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Tcgen05CopyPlanNode>()
      .def_ro("fragment", &Tcgen05CopyPlanNode::fragment)
      .def_ro("num_chunks_each_wg", &Tcgen05CopyPlanNode::num_chunks_each_wg)
      .def_ro("rest_domain", &Tcgen05CopyPlanNode::rest_domain)
      .def_ro("num_issues", &Tcgen05CopyPlanNode::num_issues)
      .def_ro("vals_per_issue", &Tcgen05CopyPlanNode::vals_per_issue)
      .def_ro("datapaths_per_warp", &Tcgen05CopyPlanNode::datapaths_per_warp);
}

// Running example: 32dp32b, 128 threads, gapped tile from a column slice of
// a batched accumulator:
//   tmem_tile = (3,128,64):(128@1,1@0,1@1)   (batch, datapath, column)
Tcgen05CopyPlan ExpandTcgen05Layout(const Tcgen05Meta &meta,
                                    const cute::Layout &tmem_tile,
                                    int num_threads,
                                    int64_t values_per_column) {
  static constexpr int WARPGROUP_SIZE = 128;
  ICHECK(num_threads > 0 && num_threads % WARPGROUP_SIZE == 0)
      << "ExpandTcgen05Layout needs a positive multiple of " << WARPGROUP_SIZE
      << " threads, got " << num_threads;
  ICHECK_GE(values_per_column, 1);
  int num_wgs = num_threads / WARPGROUP_SIZE;

  // The TILE decides the datapath split: its contiguous datapath run,
  // clamped to one warp's 32 -- 32 for a dense tile, 16 for PTX Layout F
  // (1SM M=64).  The atom must divide it; ndup duplicates the atom onto the
  // high datapaths.
  const int64_t atom_datapaths = Tcgen05AtomDatapaths(meta);
  const int64_t dp_run = cute::AsConst(
      cute::Size(cute::RightInverse(cute::Composition(kDpOnly, tmem_tile))));
  const int64_t dp_per_warp = std::min<int64_t>(dp_run, 32);
  if (dp_per_warp % atom_datapaths != 0)
    return Tcgen05CopyPlan(nullptr);
  const int64_t ndup = dp_per_warp / atom_datapaths;
  const int64_t dp_per_issue = 4 * dp_per_warp; // 128, or 64 for Layout F
  const bool partial_subpartition = dp_per_warp < 32;

  // serialized_tmem_tile: logical tile -> (datapath, column) -> serialized
  //   TMEM.
  //   E.g. (3,128,64):(16384,1,128).
  cute::Layout serialized_tmem_tile = cute::Composition(kSerialize, tmem_tile);
  const int64_t size = cute::AsConst(cute::Size(serialized_tmem_tile));

  // One issue covers the whole datapath footprint by one contiguous column
  // run, so the tile splits along the two projections.
  // valid_dps:  datapath index -> datapath, Filter(kDpOnly o tmem_tile).
  //   E.g. 128:1.
  // valid_cols: column index -> column, Filter(kColOnly o tmem_tile).
  //   E.g. (3,64):(128,1).
  // issue_cols: per-issue column run -> column index,
  //   RightInverse(valid_cols), the maximal contiguous run.
  //   E.g. 64:3.
  cute::Layout valid_dps = cute::Filter(cute::Composition(kDpOnly, tmem_tile));
  cute::Layout valid_cols =
      cute::Filter(cute::Composition(kColOnly, tmem_tile));
  cute::Layout issue_cols = cute::RightInverse(valid_cols);
  const int64_t num_dps = cute::AsConst(cute::Size(valid_dps));
  const int64_t num_cols = cute::AsConst(cute::Size(valid_cols));
  const int64_t cols_per_issue = cute::AsConst(cute::Size(issue_cols));

  // The four warps must own the datapath footprint exactly, the tile must
  // be bijective onto (datapath, column), and the issues must tile the
  // columns.
  if (num_dps != dp_per_issue || num_dps * num_cols != size ||
      num_cols % cols_per_issue != 0)
    return Tcgen05CopyPlan(nullptr);
  const int64_t num_issues = num_cols / cols_per_issue;
  const int64_t elems_per_issue = dp_per_issue * cols_per_issue;

  // Per-warpgroup instruction shape.
  // E.g. width = 1, num_chunks_each_wg = 64.
  const int64_t width = Tcgen05AtomWidth(meta);
  if (cols_per_issue % width != 0)
    return Tcgen05CopyPlan(nullptr);
  const int64_t total_chunks = cols_per_issue / width;
  if (total_chunks % num_wgs != 0)
    return Tcgen05CopyPlan(nullptr);
  const int64_t num_chunks_each_wg = total_chunks / num_wgs;

  // The .xN the wrapper is handed: a partial sub-partition carries its
  // sub-word packing as its own mode, so the repetitions are what remains
  // (the full-datapath path plans in value columns; LowerTmem halves N).
  const int64_t pack = partial_subpartition ? values_per_column : 1;
  if (num_chunks_each_wg % pack != 0)
    return Tcgen05CopyPlan(nullptr);
  const int64_t reps = num_chunks_each_wg / pack;

  // The wrapper chains an oversized copy one column and one register per
  // repetition -- exact only for 32dp32b (max_chunks == 0).  Every other
  // atom must fit one .xN: a power of two, at most max_chunks.
  if (meta->max_chunks != 0) {
    if (reps & (reps - 1)) // reps must be a power of two
      return Tcgen05CopyPlan(nullptr);
    if (reps > meta->max_chunks)
      return Tcgen05CopyPlan(nullptr);
  }

  // atom: (lane..., reg...) -> (datapath, column), the TV atom with its
  //   column steps widened to `pack` value columns (the atom addresses b32
  //   columns, the tile counts value columns).
  //   E.g. (32,1):(1@0,0).
  cute::Layout atom = cute::Composition(
      cute::Layout(Array<int64_t>{1, 1},
                   Array<cute::IntTuple>{cute::E({0}), pack * cute::E({1})}),
      meta->tv);

  // full_t: (lane..., warp, warpgroup) -> (datapath, column); warps one
  //   sub-partition apart, warpgroups splitting the columns.
  //   E.g. ((32,1),4,1):((1@0,0),32@0,64@1).
  cute::Layout full_t = cute::MakeLayout(
      {atom[0], cute::Layout(4, 32 * cute::E({0})),
       cute::Layout(num_wgs, num_chunks_each_wg * width * cute::E({1}))});

  // full_v: (pack, reg..., rep, dup) -> (datapath, column), the wrapper's
  //   register order: packed partner fastest, then the atom's registers,
  //   then .xN repetitions, then the duplicate issue after ALL repetitions.
  //   E.g. (1,1,64,1):(1@1,0,1@1,16@0).
  cute::Layout full_v =
      cute::MakeLayout({cute::Layout(pack, cute::E({1})), atom[1],
                        cute::Layout(reps, width * pack * cute::E({1})),
                        cute::Layout(ndup, atom_datapaths * cute::E({0}))});

  // serialized_full_tv: (T, V) -> (datapath, column) -> serialized TMEM, one
  //   issue's warpgroup-wide stamp.
  cute::Layout serialized_full_tv =
      cute::Composition(kSerialize, cute::MakeLayout({full_t, full_v}));

  // to_logical: serialized TMEM -> flat logical tile,
  //   LeftInverse(serialized_tmem_tile); gaps in the tile's image fold into
  //   its modes and are never addressed.
  //   E.g. (128,128,3):(3,384,1).
  cute::Layout to_logical = cute::LeftInverse(serialized_tmem_tile);

  // rest_cols: issue -> column origin,
  //   LogicalDivide(valid_cols, issue_cols)[1].
  //   E.g. 3:128.
  // rest_domain: issue -> column origin -> serialized TMEM -> flat logical
  //   tile origin, to_logical o (128 * rest_cols).
  //   E.g. 3:1.
  cute::Layout rest_cols = cute::LogicalDivide(valid_cols, issue_cols)[1];
  cute::Layout rest_domain = cute::Composition(
      to_logical, cute::Layout(rest_cols->shape, rest_cols->stride * 128));
  if (cute::AsConst(cute::Size(rest_domain)) != num_issues)
    return Tcgen05CopyPlan(nullptr);

  // tile_tv: (T, V) -> (datapath, column) -> serialized TMEM -> flat logical
  //   tile, with rest_domain appended as the slowest value mode.
  //   E.g. ((32,4),(64,3)):((3,96),(384,1)).
  cute::Layout tile_tv = cute::MakeLayout(
      {cute::Composition(to_logical, serialized_full_tv[0]),
       cute::MakeLayout({cute::Composition(to_logical, serialized_full_tv[1]),
                         rest_domain})});
  const int64_t num_vals = cute::AsConst(cute::Size(tile_tv[1]));

  // fragment: logical tile -> (thread@0, value@1), RightInverse(tile_tv)
  //   under the (thread, value) identity, reshaped to the tile's modes (the
  //   make_tiled_copy right_inverse.with_shape idiom).  The size check
  //   proves bijectivity.
  //   E.g. (3,128,64):(64@1,1@0,1@1).
  cute::Layout inv_tv = cute::RightInverse(tile_tv);
  if (cute::AsConst(cute::Size(inv_tv)) != size)
    return Tcgen05CopyPlan(nullptr);
  Array<cute::IntTuple> tile_shape;
  for (int64_t i = 0, r = cute::Rank(tmem_tile); i < r; ++i)
    tile_shape.push_back(cute::Product(tmem_tile->shape[i]));
  cute::Layout fragment =
      cute::Composition(
          cute::MakeIdentityLayout(Array<int64_t>{num_threads, num_vals}),
          inv_tv)
          .WithShape(cute::IntTupleTuple(tile_shape));

  return Tcgen05CopyPlan(fragment, num_chunks_each_wg, rest_domain, num_issues,
                         elems_per_issue / num_threads, dp_per_warp);
}

Fragment FragmentToTileLang(const cute::Layout &layout) {
  int64_t r = cute::Rank(layout);
  Array<IterVar> ivs;
  Array<cute::IntTuple> coords;
  arith::Analyzer analyzer;
  for (int64_t i = 0; i < r; ++i) {
    int64_t size = cute::AsConst(cute::Product(layout->shape[i]));
    IterVar iv =
        MakeIterVar("i" + std::to_string(i), Range(0, static_cast<int>(size)));
    analyzer.Bind(iv->var, iv->dom);
    ivs.push_back(iv);
    coords.push_back(iv->var);
  }
  // Normalize the (thread@0, value@1) coordinate by adding the rank-2 zero
  // ArithmeticTuple, so an untouched axis materializes as a plain zero slot.
  cute::IntTuple tv_coord =
      layout(cute::IntTupleTuple(coords)) + cute::IntTupleTuple({0, 0});
  Array<cute::IntTuple> fields = cute::TupleFields(tv_coord);
  ICHECK_EQ(fields.size(), 2U)
      << "Fragment must map into (thread@0, value@1), got " << tv_coord;
  DataType dtype = DataType::Int(32);
  PrimExpr thread =
      analyzer.Simplify(cute::AsConstOrPrimExpr(fields[0], dtype));
  PrimExpr value = analyzer.Simplify(cute::AsConstOrPrimExpr(fields[1], dtype));
  return Fragment(ivs, {value}, thread, MakeIterVar("rep", Range(0, 1)));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  Tcgen05MetaNode::RegisterReflection();
  Tcgen05CopyPlanNode::RegisterReflection();
  refl::GlobalDef()
      .def("tl.get_tcgen05_meta_ld_32dp32b", GetTcgen05MetaLd32Dp32B)
      .def("tl.get_tcgen05_meta_ld_16dp64b", GetTcgen05MetaLd16Dp64B)
      .def("tl.get_tcgen05_meta_ld_16dp128b", GetTcgen05MetaLd16Dp128B)
      .def("tl.get_tcgen05_meta_ld_16dp256b", GetTcgen05MetaLd16Dp256B)
      .def("tl.get_tcgen05_meta_st_32dp32b", GetTcgen05MetaSt32Dp32B)
      .def("tl.get_tcgen05_meta_st_16dp64b", GetTcgen05MetaSt16Dp64B)
      .def("tl.get_tcgen05_meta_st_16dp128b", GetTcgen05MetaSt16Dp128B)
      .def("tl.get_tcgen05_meta_st_16dp256b", GetTcgen05MetaSt16Dp256B)
      .def("tl.ExpandTcgen05Layout",
           [](const Tcgen05Meta &meta, const cute::Layout &tmem_tile,
              int64_t num_threads) {
             return ExpandTcgen05Layout(meta, tmem_tile,
                                        static_cast<int>(num_threads));
           });
}

} // namespace tl
} // namespace tvm
