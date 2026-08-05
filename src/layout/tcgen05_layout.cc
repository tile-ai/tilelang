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

  // Serialize (datapath, column) into the flat address datapath + 128*column
  // so everything below is algebra over one codomain.
  // serialized_tmem_tile: logical tile -> serialized TMEM
  // E.g., serialized_tmem_tile = (3,128,64):(16384,1,128)
  static const cute::Layout kSerialize = cute::Layout::Parse("(1,1):(1,128)");

  // How much of a warp's 32-lane sub-partition one warp covers is a property
  // of the TILE.  On the datapath axis alone a dense fragment runs 128 before
  // its first gap, a PTX Layout F fragment (1SM M=64) exactly 16.  A warp owns
  // 32, so clamp there and let the atom divide what is left -- an atom too
  // wide for the run (32dp32b against Layout F) falls out as a non-division.
  // Projecting the columns away also makes this independent of whether they
  // count values or b32 slots.
  static const cute::Layout kDpOnly = cute::Layout::Parse("(1,1):(1,0)");
  const int64_t atom_datapaths = Tcgen05AtomDatapaths(meta);
  const int64_t dp_run = cute::AsConst(
      cute::Size(cute::RightInverse(cute::Composition(kDpOnly, tmem_tile))));
  const int64_t dp_per_warp = dp_run < 32 ? dp_run : 32;
  if (dp_per_warp % atom_datapaths != 0)
    return Tcgen05CopyPlan(nullptr);
  const int64_t ndup = dp_per_warp / atom_datapaths;
  const int64_t dp_per_issue = 4 * dp_per_warp; // 128, or 64 for Layout F
  const bool partial_subpartition = dp_per_warp < 32;

  cute::Layout serialized_tmem_tile = cute::Composition(kSerialize, tmem_tile);
  int64_t size = cute::AsConst(cute::Size(serialized_tmem_tile));

  // A full-datapath issue is exactly the tile's maximal contiguous run, and
  // the divide's rest mode iterates the issues after it.  A Layout F issue is
  // not contiguous at all -- four datapath groups 32 apart -- so it is
  // assembled explicitly below and always covers the tile in one go.
  // inv_prefix: serialized chunk -> flat logical tile.  E.g. 8192:3.
  cute::Layout inv_prefix = cute::RightInverse(serialized_tmem_tile);
  int64_t elems_per_issue =
      partial_subpartition ? size : cute::AsConst(cute::Size(inv_prefix));
  if (elems_per_issue % dp_per_issue != 0 || size % elems_per_issue != 0)
    return Tcgen05CopyPlan(nullptr);
  int64_t num_issues = size / elems_per_issue;

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
  int64_t cols_per_issue = elems_per_issue / dp_per_issue;
  if (cols_per_issue % width != 0)
    return Tcgen05CopyPlan(nullptr);
  int64_t total_chunks = cols_per_issue / width;
  if (total_chunks % num_wgs != 0)
    return Tcgen05CopyPlan(nullptr);
  int num_chunks_each_wg = static_cast<int>(total_chunks / num_wgs);

  // The .xN the wrapper is actually handed: a partial sub-partition carries
  // its sub-word packing as its own mode, so the repetitions are what remains.
  // The full-datapath path plans in value columns and LowerTmem halves the
  // count itself, so bound the planned count there -- never optimistic.
  int64_t pack = partial_subpartition ? values_per_column : 1;
  if (num_chunks_each_wg % pack != 0)
    return Tcgen05CopyPlan(nullptr);
  int64_t reps = num_chunks_each_wg / pack;

  // core chains a copy too big for one instruction by advancing ONE column
  // and ONE register per repetition, exact only for an atom that is one of
  // each -- 32dp32b, flagged by max_chunks == 0.  Every other atom must fit
  // the per-warpgroup copy in a single .xN.  Lifting that needs the core
  // parameterised by columns/registers per repetition AND the 32dp
  // composites' nesting inverted so each half chains separately: fixing the
  // strides alone still leaves a chained duplicate interleaved per segment
  // instead of following all repetitions.  Until then a non-power-of-two .xN
  // finds no plan (an M=64 bf16 operand at K = 96, 192 or 320).
  if (meta->max_chunks != 0) {
    if (reps & (reps - 1))  // reps must be a power of two
      return Tcgen05CopyPlan(nullptr);
    if (reps > meta->max_chunks)
      return Tcgen05CopyPlan(nullptr);
  }

  // The stamp: what ONE wrapper call covers.  Warps sit one sub-partition
  // apart (32), a 16-datapath atom's duplicate issue on the high half of one
  // (16), .xN repetitions a column group apart, warpgroups split the columns.
  // The packed partner -- two sub-word values in one b32 column -- is the
  // FASTEST value, one value column along, since a register's two halves are
  // adjacent in the buffer; the atom addresses b32 columns, so its column step
  // is `pack` value columns wide.  Spelled out rather than left to a blocked
  // product's complement, whose fastest gap for Layout F is the *unoccupied*
  // datapaths -- it would pack the warps in at stride 16.
  // tiled: ((lane, (warp, wg)), (pack, reg, (rep, dup))) -> serialized TMEM
  int64_t b32_col = 128 * pack;
  cute::Layout atom = cute::Composition(
      cute::Layout(Array<int64_t>{1, 1}, Array<int64_t>{1, b32_col}), meta->tv);
  cute::Layout tiled = cute::MakeLayout(
      {cute::MakeLayout(
           {atom[0], cute::Layout(Array<int64_t>{4, num_wgs},
                                  Array<int64_t>{32, b32_col * width * reps})}),
       cute::MakeLayout(
           {cute::Layout(Array<int64_t>{pack}, Array<int64_t>{128}), atom[1],
            cute::Layout(Array<int64_t>{reps, ndup},
                         Array<int64_t>{b32_col * width, 16})})});

  // Physical -> logical.  inv_prefix spans the maximal contiguous chunk, one
  // whole full-datapath issue, and stays valid where the tile is not injective
  // (a pack::16b tile folds its unused sub-slot onto a stride-0 mode).  No
  // contiguous prefix covers a Layout F issue's four groups, but such a tile
  // IS injective, so left_inverse serves.  Either way the placement is derived
  // from the tile, not assumed of it.
  // tile_tv: (thread, value) -> flat logical tile
  cute::Layout to_logical = partial_subpartition
                                ? cute::LeftInverse(serialized_tmem_tile)
                                : inv_prefix;
  cute::Layout tile_tv = cute::MakeLayout(
      {cute::Composition(to_logical, tiled[0]),
       cute::MakeLayout(
           {cute::Composition(to_logical, tiled[1]), rest_domain})});
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
