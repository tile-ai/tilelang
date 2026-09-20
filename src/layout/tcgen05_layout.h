/*!
 * \file layout/tcgen05_layout.h
 * \brief tcgen05.ld/st data-movement shapes as CuTe TV atoms.
 */
#pragma once

#include "cute_layout.h"
#include "layout.h"

namespace tvm {
namespace tl {

// Projections of the physical TMEM coordinate (datapath@0, column@1) onto
// one axis (the other axis's strides become 0).
// E.g. Composition(kColOnly, (128,64):(1@0,1@1)) == (128,64):(0,1).
TVM_DLL extern const cute::Layout kDpOnly;  // (datapath, column) -> datapath
TVM_DLL extern const cute::Layout kColOnly; // (datapath, column) -> column

// Metadata for one tcgen05.ld/st data-movement shape.
//
// `tv` is the CUTLASS ``Copy_Traits<...1x>`` TV atom: (lane..., reg...) ->
// (datapath@0, b32 column@1), covering one PTX issue of one warp (the
// 16-datapath shapes only the LOW 16 datapaths of its sub-partition).  All
// replication -- warps, the high-datapath duplicate issue, .xN repetitions,
// warpgroups -- is applied by ExpandTcgen05Layout.
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

// The atom's extent along one physical axis: Cosize of the axis projection
// of `tv`.  Datapaths set the wrapper's duplication factor; the width is
// the b32 columns of one .x1 repetition.
int64_t Tcgen05AtomDatapaths(const Tcgen05Meta &meta);
int64_t Tcgen05AtomWidth(const Tcgen05Meta &meta);

// The tiled copy of one TMEM tile, built by ExpandTcgen05Layout.  A tile
// whose serialized image is gapped cannot be one instruction; the copy
// issues once per contiguous chunk (CuTe's tiled-copy rest modes).
class Tcgen05CopyPlanNode : public ffi::Object {
public:
  cute::Layout fragment;      // logical tile coord -> (thread@0, value@1)
  int64_t num_chunks_each_wg; // .xN repetitions per warpgroup, per issue
  cute::Layout rest_domain;   // issue -> flat logical tile index of origin
  int64_t num_issues;         // size(rest_domain)
  int64_t vals_per_issue;     // registers per thread, per issue
  // Datapaths of a warp's sub-partition the tile occupies: 32, or 16 for a
  // PTX Layout F fragment (the atom is then issued once per warp).
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

// Build the copy plan for one TMEM tile (logical tile coords -> physical
// (datapath@0, column@1)).  One issue covers the tile's whole datapath
// footprint by its maximal contiguous column run; the atom TV layout is
// extended to that stamp and mapped back through the tile:
//   tile_tv:     (thread, value) -> (datapath, column) -> serialized TMEM
//                -> logical tile; its RightInverse is the fragment.
//   rest_domain: issue -> column origin -> serialized TMEM -> logical tile.
//
// Returns a null ref when this instruction/warpgroup arrangement cannot
// express the tile bijectively.  `values_per_column` is how many of the
// tile's columns share one b32 column (1 for 32-bit values or a tile
// already stated in b32 columns).
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
