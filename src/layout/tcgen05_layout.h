/*!
 * \file layout/tcgen05_layout.h
 * \brief tcgen05.ld/st data-movement shapes as CuTe TV atoms.
 */
#pragma once

#include "cute_layout.h"
#include "layout.h"

namespace tvm {
namespace tl {

// Metadata for one tcgen05.ld/st data-movement shape.
//
// `tv` is the CUTLASS ``Copy_Traits<SM100_TMEM_LOAD_*1x>`` TV atom verbatim
// (DstLayout over ValID, upcast to b32): a rank-2 layout (lane modes,
// register modes) whose ScaledBasis strides land in axis 0 = datapath and
// axis 1 = b32 column.  It covers exactly what one PTX issue of the x1
// shape covers -- one 32-lane warp and, for the 16-datapath shapes, only
// the LOW 16 datapaths of the warp's sub-partition.  All replication is
// applied algebraically by ExpandTcgen05Layout: the warp tiling
// (make_tmem_copy's atom_t_layout), the wrapper's duplicate issue on the
// high 16 datapaths, the .xN column repetitions, and the warpgroup split
// enter as one blocked product over a replication grid, so the atom itself
// stores none of them.
class Tcgen05MetaNode : public ffi::Object {
public:
  ffi::String intrinsics_name;
  cute::Layout tv;    // (lane..., reg...) -> (datapath, column), one issue
  int64_t max_chunks; // largest .xN of one issue; 0 = wrapper chaining exact

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Tcgen05Meta", Tcgen05MetaNode,
                                    ffi::Object);
};

class Tcgen05Meta : public ffi::ObjectRef {
public:
  TVM_DLL Tcgen05Meta(ffi::String intrinsics_name, cute::Layout tv,
                      int64_t max_chunks);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Tcgen05Meta, ffi::ObjectRef,
                                             Tcgen05MetaNode);
};

// Obtain the metadata for tcgen05.ld instructions.
Tcgen05Meta GetTcgen05MetaLd32Dp32B();
Tcgen05Meta GetTcgen05MetaLd16Dp64B();
Tcgen05Meta GetTcgen05MetaLd16Dp128B();
Tcgen05Meta GetTcgen05MetaLd16Dp256B();

// Obtain the metadata for tcgen05.st instructions.
Tcgen05Meta GetTcgen05MetaSt32Dp32B();
Tcgen05Meta GetTcgen05MetaSt16Dp64B();
Tcgen05Meta GetTcgen05MetaSt16Dp128B();
Tcgen05Meta GetTcgen05MetaSt16Dp256B();

// The atom's extent along one physical axis, from the CuTe algebra: compose
// an axis projection over the TV layout (keeping one axis's strides and
// zeroing the other's) and take the cosize.  Datapaths determine the
// wrapper's duplication factor (32 / datapaths issues fill a warp's
// sub-partition); the width is the b32 columns of one .x1 repetition.
int64_t Tcgen05AtomDatapaths(const Tcgen05Meta &meta);
int64_t Tcgen05AtomWidth(const Tcgen05Meta &meta);

// The tiled copy of one TMEM tile, built by ExpandTcgen05Layout.
//
// A tile whose serialized image is gapped (e.g. a column slice of a batched
// accumulator, (3,128,64):(16384,1,128) serialized) cannot be one
// instruction; the copy iterates its contiguous chunks instead, one issue
// per rest coordinate (CuTe's tiled-copy rest modes).
class Tcgen05CopyPlanNode : public ffi::Object {
public:
  cute::Layout fragment;      // logical tile coord -> (thread@0, value@1)
  int64_t num_chunks_each_wg; // .xN repetitions per warpgroup, per issue
  cute::Layout rest_domain;   // issue -> flat logical tile index of origin
  int64_t num_issues;         // size(rest_domain)
  int64_t vals_per_issue;     // registers per thread, per issue
  // Datapaths of a warp's sub-partition the tile occupies -- 32, or 16 for a
  // PTX Layout F fragment.  This is the number the wrapper is named after:
  // 16 means the atom is issued once per warp instead of being duplicated
  // onto the high datapaths.
  int64_t datapaths_per_warp;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Tcgen05CopyPlan", Tcgen05CopyPlanNode,
                                    ffi::Object);
};

class Tcgen05CopyPlan : public ffi::ObjectRef {
public:
  TVM_DLL Tcgen05CopyPlan(cute::Layout fragment, int64_t num_chunks_each_wg,
                          cute::Layout rest_domain, int64_t num_issues,
                          int64_t vals_per_issue, int64_t datapaths_per_warp);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Tcgen05CopyPlan, ffi::ObjectRef,
                                             Tcgen05CopyPlanNode);
};

// Build the copy plan for one TMEM tile, in CuTe algebra: serialize the
// atom TV layout and `tmem_tile` (logical tile coords -> physical
// (datapath@0, column@1)) into linear TMEM addressing via (1,1):(1,128);
// right_inverse of the serialized tile finds its maximal contiguous chunk,
// and logical_divide by that inverse splits the tile into (chunk, rest);
// tile the serialized atom over (warpgroups, repetitions) with a blocked
// product whose appended value mode is FASTER than the appended thread mode
// (the values a thread holds stay contiguous in TMEM for the longest
// vectors), and append the rest mode slowest of all (later issues append
// registers); compose with the left inverse of the serialized tile
// (register TV -> serialized TMEM -> logical tile); and invert once more.
//
// Returns a null ref when this instruction/warpgroup arrangement cannot
// express the tile's chunks bijectively.
// `values_per_column` is how many of the tile's columns share one b32 column
// -- 1 for 32-bit values or a tile already stated in b32 columns.  It is the
// one thing a value-column layout cannot say about itself.
Tcgen05CopyPlan ExpandTcgen05Layout(const Tcgen05Meta &meta,
                                    const cute::Layout &tmem_tile,
                                    int num_threads,
                                    int64_t values_per_column = 1);

// Convert a fragment (logical coord -> (thread@0, value@1), like
// Tcgen05CopyPlan::fragment) into a TileLang Fragment over its top-level
// modes (the single, final conversion out of CuTe algebra).
Fragment FragmentToTileLang(const cute::Layout &layout);

} // namespace tl
} // namespace tvm
