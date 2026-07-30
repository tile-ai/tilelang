/*!
 * \file layout/tcgen05_layout.cc
 * \brief tcgen05.ld/st data-movement shapes as CuTe TV atoms.
 *
 * Each atom below is the CUTLASS ``Copy_Traits<SM100_TMEM_LOAD_*>`` TV
 * layout (cute/atom/copy_traits_sm100.hpp) written over (datapath, b32
 * column) coordinates, cross-checked against the PTX data-movement-shape
 * figures
 * (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-layout).
 * Loads and stores share one data-movement shape per width.
 */

#include "support/check.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "layout.h"
#include "tcgen05_layout.h"

namespace tvm {
namespace tl {

using namespace tirx;
using tvm::ffi::Array;

namespace {

IterVar MakeIterVar(std::string name, Range dom) {
  Var var = Var(name, dom->min->dtype);
  return IterVar(dom, var, IterVarType::kDataPar);
}

// The atoms are the CUTLASS ``Copy_Traits<...1x>`` TV layouts verbatim
// (DstLayout over ValID, upcast to b32), written in CuTe spelling over the
// physical coordinate axes, axis 0 = datapath ("@0"), axis 1 = b32 column
// ("@1").  Each covers exactly one PTX issue of one warp: the 32x32b shape
// fills the warp's whole 32-datapath sub-partition, the 16x shapes only its
// low 16 datapaths.  ExpandTcgen05Layout replicates them over warps, the
// high-datapath duplicate issue, .xN repetitions, and warpgroups purely by
// layout algebra.

Tcgen05Meta MakeTcgen05Meta_32dp32b(bool is_store) {
  // PTX 32x32b (Copy_Traits 32dp32b1x, ValID (32,32):(1,DP_b)): lane t ->
  // datapath t; one register on one column per repetition, and the wrapper
  // chaining extends repetitions exactly, so any N is legal.
  return Tcgen05Meta(is_store ? "tl::tcgen05_st_32dp32bNx"
                              : "tl::tcgen05_ld_32dp32bNx",
                     cute::Layout::Parse("(32,1):(1@0,0)"),
                     /*max_chunks=*/0);
}

Tcgen05Meta MakeTcgen05Meta_16dp64b(bool is_store) {
  // PTX 16x64b (Copy_Traits 16dp64b1x, DstLayout ((2,2,8),32):((512,32,64),1)
  // over ValID (64,16):(1,DP_b)): lane t -> datapath 8*(t%2) + t/4, column
  // (t/2)%2; one register per issue.
  return Tcgen05Meta{is_store ? "tl::tcgen05_st_32dp64bNx"
                              : "tl::tcgen05_ld_32dp64bNx",
                     cute::Layout::Parse("((2,2,8),1):((8@0,1@1,1@0),0)"),
                     /*max_chunks=*/128};
}

Tcgen05Meta MakeTcgen05Meta_16dp128b(bool is_store) {
  // PTX 16x128b (Copy_Traits 16dp128b1x, DstLayout ((4,8),(32,2)):
  // ((32,128),(1,1024)) over ValID (128,16):(1,DP_b)): lane t -> column t%4,
  // datapath t/4; two registers stepping the 8-datapath half.
  return Tcgen05Meta{is_store ? "tl::tcgen05_st_32dp128bNx"
                              : "tl::tcgen05_ld_32dp128bNx",
                     cute::Layout::Parse("((4,8),2):((1@1,1@0),8@0)"),
                     /*max_chunks=*/64};
}

Tcgen05Meta MakeTcgen05Meta_16dp256b(bool is_store) {
  // PTX 16x256b (Copy_Traits 16dp256b1x, DstLayout ((4,8),(64,2)):
  // ((64,256),(1,2048)) over ValID (256,16):(1,DP_b)): lane t -> column
  // 2*(t%4), datapath t/4; four registers as (adjacent column, 8-datapath).
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

// Project one physical axis through the TV atom (keep that axis's basis
// strides, zero the other) and measure the footprint.  The datapath extent
// determines the wrapper's duplication factor; the column extent is the
// width of one .x1 repetition.
int64_t Tcgen05AtomDatapaths(const Tcgen05Meta &meta) {
  static const cute::Layout kDpOnly = cute::Layout::Parse("(1,1):(1,0)");
  return cute::AsConst(cute::Cosize(cute::Composition(kDpOnly, meta->tv)));
}

int64_t Tcgen05AtomWidth(const Tcgen05Meta &meta) {
  static const cute::Layout kColOnly = cute::Layout::Parse("(1,1):(0,1)");
  return cute::AsConst(cute::Cosize(cute::Composition(kColOnly, meta->tv)));
}

Tcgen05CopyPlan::Tcgen05CopyPlan(cute::Layout fragment,
                                 int64_t num_chunks_each_wg,
                                 cute::Layout rest_domain, int64_t num_issues,
                                 int64_t vals_per_issue) {
  auto node = ffi::make_object<Tcgen05CopyPlanNode>();
  node->fragment = std::move(fragment);
  node->num_chunks_each_wg = num_chunks_each_wg;
  node->rest_domain = std::move(rest_domain);
  node->num_issues = num_issues;
  node->vals_per_issue = vals_per_issue;
  data_ = std::move(node);
}

void Tcgen05CopyPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Tcgen05CopyPlanNode>()
      .def_ro("fragment", &Tcgen05CopyPlanNode::fragment)
      .def_ro("num_chunks_each_wg", &Tcgen05CopyPlanNode::num_chunks_each_wg)
      .def_ro("rest_domain", &Tcgen05CopyPlanNode::rest_domain)
      .def_ro("num_issues", &Tcgen05CopyPlanNode::num_issues)
      .def_ro("vals_per_issue", &Tcgen05CopyPlanNode::vals_per_issue);
}

// Running example: 32dp32b, 128 threads, gapped tile from a column slice of
// a batched accumulator:
//   tmem_tile = (3,128,64):(128@1,1@0,1@1)   (batch, datapath, column)
Tcgen05CopyPlan ExpandTcgen05Layout(const Tcgen05Meta &meta,
                                    const cute::Layout &tmem_tile,
                                    int num_threads) {
  static constexpr int WARPGROUP_SIZE = 128;
  ICHECK(num_threads > 0 && num_threads % WARPGROUP_SIZE == 0)
      << "ExpandTcgen05Layout needs a positive multiple of " << WARPGROUP_SIZE
      << " threads, got " << num_threads;
  int num_wgs = num_threads / WARPGROUP_SIZE;

  // Serialize (datapath, column) into the flat address datapath + 128*column
  // so everything below is algebra over one codomain.
  // serialized_fragment: atom (lane/warp, reg) -> serialized TMEM
  // serialized_tmem_tile: logical tile -> serialized TMEM
  // E.g., serialized_tmem_tile = (3,128,64):(16384,1,128)
  static const cute::Layout kSerialize = cute::Layout::Parse("(1,1):(1,128)");
  cute::Layout serialized_fragment = cute::Composition(kSerialize, meta->tv);
  cute::Layout serialized_tmem_tile = cute::Composition(kSerialize, tmem_tile);

  int64_t size = cute::AsConst(cute::Size(serialized_tmem_tile));

  // right_inverse's stride chain stops at the tile's first serialized gap,
  // so its size is the maximal contiguous chunk = one tcgen05 issue.
  // inv_prefix: serialized chunk -> flat logical tile
  // E.g., inv_prefix = 8192:3, chunk = 8192, num_issues = 3
  cute::Layout inv_prefix = cute::RightInverse(serialized_tmem_tile);
  int64_t chunk = cute::AsConst(cute::Size(inv_prefix));
  if (chunk % 128 != 0 || size % chunk != 0)
    return Tcgen05CopyPlan(nullptr);
  int64_t num_issues = size / chunk;

  // Divide the flat logical domain by the chunk; the rest mode locates each
  // issue's origin (CuTe's tiled-copy rest iteration).
  // rest_domain: issue -> flat logical tile origin
  // E.g., rest_domain = 3:1 -> origins 0, 1, 2 (idx2crd: batch 0, 1, 2)
  cute::Layout rest_domain =
      num_issues == 1 ? cute::Layout(1, 0)
                      : cute::LogicalDivide(
                            cute::MakeColumnMajorLayout(cute::Size(tmem_tile)),
                            inv_prefix)[1];
  if (cute::AsConst(cute::Size(rest_domain)) != num_issues)
    return Tcgen05CopyPlan(nullptr);

  // Instruction feasibility per issue.  The atom's column width and
  // datapath extent come from its own algebra; a 16-datapath atom is issued
  // twice per warp (low then high datapaths), so the whole per-warpgroup
  // copy must stay one .xN issue for the wrapper's register order to hold.
  // E.g., width = 1, ndup = 1, cols_per_issue = 64, num_chunks_each_wg = 64
  int64_t width = Tcgen05AtomWidth(meta);
  int64_t ndup = 32 / Tcgen05AtomDatapaths(meta);
  int64_t cols_per_issue = chunk / 128;
  if (cols_per_issue % width != 0)
    return Tcgen05CopyPlan(nullptr);
  int64_t total_chunks = cols_per_issue / width;
  if (total_chunks % num_wgs != 0)
    return Tcgen05CopyPlan(nullptr);
  int num_chunks_each_wg = static_cast<int>(total_chunks / num_wgs);
  if (ndup > 1) {
    // The wrapper appends the duplicate issue's registers after ALL
    // repetitions, so the per-warpgroup copy must be one .xN issue.
    if (num_chunks_each_wg & (num_chunks_each_wg - 1))
      return Tcgen05CopyPlan(nullptr);
    if (num_chunks_each_wg > meta->max_chunks)
      return Tcgen05CopyPlan(nullptr);
  }

  // Replicate the atom with one blocked product over the wrapper's
  // replication grid.  The blocked product's complement enumerates the
  // serialized gaps around the atom in ascending-stride order -- the
  // duplicate high-16-datapath issue first, then the warp sub-partitions,
  // then the .xN column repetitions, then the warpgroup column split -- so
  // the row-major (wg, rep, warp, dup) grid indexes exactly those slots.
  // Zipped per atom mode: thread gets (warp, wg), value gets (rep, dup);
  // the value replicas are FASTER than the thread replicas, so a thread's
  // values stay contiguous in TMEM, and the duplicate issue lands after the
  // repetitions exactly as the wrapper appends its registers.
  // tiled: ((lane, (warp, wg)), (reg, (rep, dup))) -> serialized chunk
  // E.g., tiled = ((32,(4,1)),(1,(64,1))):((1,(32,8192)),(0,(128,32)))
  cute::Layout grid = cute::MakeRowMajorLayout(
      Array<int64_t>{num_wgs, num_chunks_each_wg, 4, ndup});
  cute::Layout tiler = cute::MakeLayout({cute::MakeLayout({grid[2], grid[0]}),
                                         cute::MakeLayout({grid[1], grid[3]})});
  cute::Layout tiled = cute::BlockedProduct(serialized_fragment, tiler);

  // Map the copy back to flat logical indices piecewise, mirroring the
  // (chunk, rest) split of the divide above: every replica of `tiled`
  // addresses within one contiguous chunk, which inv_prefix inverts, and
  // rest_domain already locates each issue's flat logical origin -- so the
  // issue mode composes directly, without inverting the whole tile.
  // tile_tv: (thread, value) -> flat logical tile
  // E.g., tile_tv = ((32,(4,1)),((1,(64,1)),3)):((3,(96,1)),((0,(384,1)),1))
  cute::Layout tile_tv = cute::MakeLayout(
      {cute::Composition(inv_prefix, tiled[0]),
       cute::MakeLayout(
           {cute::Composition(inv_prefix, tiled[1]), rest_domain})});
  int64_t num_vals = cute::AsConst(cute::Size(tile_tv[1]));

  // Invert (the make_tiled_copy `right_inverse(...).with_shape(...)` idiom):
  // the identity layout tags the (thread@0, value@1) axes, with_shape
  // restores the tile's logical modes.
  // fragment: logical tile -> (thread@0, value@1)
  // E.g., fragment = (3,128,64):(64@1,1@0,1@1)
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
                         chunk / num_threads);
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
