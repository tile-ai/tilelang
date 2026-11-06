#ifndef TILELANG_IR_ANNOTATIONS_H_
#define TILELANG_IR_ANNOTATIONS_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/stmt.h>

#include "layout/layout.h"

namespace tvm::tl {

// Decode the operand envelope used by metadata other than layout_map.
// Layout declarations have a dedicated typed-attrs representation below.
TVM_TIRX_DLL ffi::Any DecodeAnnotations(const tvm::Tuple &payload,
                                        const ffi::Array<ffi::Any> &schema);

// Runtime Call operands exclude a declared trailing annotation Tuple.
// Validate the complete envelope and all schema references.
TVM_TIRX_DLL ffi::Array<Expr> DecodeCallOperands(const CallObj *call);

// Layout declarations own their TensorVar/Layout references in typed attrs.
// Generic upstream Call traversal does not rewrite these attrs: TileLang
// visitors/mutators must explicitly enter them using their current context.
TVM_TIRX_DLL const Op &layout_map_op();
TVM_TIRX_DLL bool IsAnnotationCall(const CallObj *call);
inline bool IsAnnotationCall(const Call &call) {
  return IsAnnotationCall(call.get());
}
TVM_TIRX_DLL ffi::Map<tirx::TensorVar, Layout>
ReadLayoutMapAnnotation(const CallObj *call);
TVM_TIRX_DLL Stmt
MakeLayoutMapAnnotation(const ffi::Map<tirx::TensorVar, Layout> &layouts,
                        bool source_block = false, Location loc = UnknownLoc());

// Read either a dedicated layout_map declaration or an operand envelope.
// The Array/Region overloads still describe only the operand protocol.
TVM_TIRX_DLL ffi::Map<ffi::String, ffi::Any>
ReadAnnotations(const ffi::Array<Expr> &args, const DictAttrs &attrs);
TVM_TIRX_DLL ffi::Map<ffi::String, ffi::Any>
ReadAnnotations(const CallObj *call);
TVM_TIRX_DLL ffi::Map<ffi::String, ffi::Any>
ReadAnnotations(const RegionStmtObj *region);
// A mixed map is emitted as adjacent dedicated-layout and ordinary metadata
// statements. The result may be SeqStmt; callers must not assume Evaluate.
TVM_TIRX_DLL Stmt
MakeAnnotation(const ffi::Map<ffi::String, ffi::Any> &annotations,
               Location loc = UnknownLoc());
TVM_TIRX_DLL RegionStmt MakeAttribute(ffi::Any node, ffi::String key,
                                      ffi::Any value, SeqStmt body,
                                      Location loc = UnknownLoc());

// Source metadata accepts Python bools and literal 0/1 values, including IR
// IntImm literals. Other integers are not boolean flags.
TVM_TIRX_DLL bool AnnotationAsBool(ffi::AnyView value);

struct EncodedAnnotations {
  tvm::Tuple payload;
  ffi::Array<ffi::Any> schema;
};

// This is the same version-1 schema as
// language._annotation_codec.encode_annotations. All expressions, including
// expressions inside Layout and WS values, are explicit uses in payload. The
// schema contains only static metadata. Dedicated layout_map declarations do
// not use this codec; other layout-bearing annotation contracts may still do.
TVM_TIRX_DLL EncodedAnnotations EncodeAnnotations(const ffi::Any &annotations);

// Literal values and containers of literals can live in native attributes.
// Other contracts containing variables need an explicit traversal policy;
// layout_map is one such dedicated contract, not a static annotation.
TVM_TIRX_DLL bool IsStaticAnnotationValue(const ffi::Any &value);

// Source annotation grammar shared by source lowering and final verification.
// These classifications do not depend on a target or a lowering stage. A
// codegen annotation may also belong to the reserved compiler namespace.
TVM_TIRX_DLL bool IsCodegenAnnotation(const ffi::String &key);
TVM_TIRX_DLL bool IsCompilerAnnotation(const ffi::String &key);

// Remove an ordinary Call's source metadata envelope, keeping static options
// in native attrs. Dynamic compiler/codegen options must already have an
// explicit operand representation; unknown dynamic hints are discarded without
// evaluation. Tile operators must have been consumed by LowerTileOp. This
// helper does not traverse children or consume other passes' metadata.
TVM_TIRX_DLL Call NormalizeCallAnnotations(Call call);

// Source sblock_attr declarations describe their entire nearest SBlock/Kernel,
// including statements before the declaration. Collect their owned metadata
// without moving definitions or uses in the IR. Nested source owners and
// function/lambda expressions are not traversed. Native unmarked annotations
// retain their ordinary lexical-sequence semantics.
TVM_TIRX_DLL ffi::Map<ffi::String, ffi::Any>
CollectSourceBlockAnnotations(const SeqStmt &body);

// Tile operators keep their analysis metadata in explicit operands. Ordinary
// intrinsics put static annotations in attrs; dynamic source hints are consumed
// by LowerSourceAnnotations before code generation. operands excludes metadata.
TVM_TIRX_DLL Call
MakeAnnotatedCall(Type ty, Op op, ffi::Array<Expr> operands,
                  const ffi::Map<ffi::String, ffi::Any> &annotations,
                  Location loc = UnknownLoc());

struct TileOpArguments {
  ffi::Array<Expr> operands;
  ffi::Map<ffi::String, ffi::Any> annotations;
};

TVM_TIRX_DLL TileOpArguments DecodeTileOpArguments(const CallObj *call);

} // namespace tvm::tl

#endif // TILELANG_IR_ANNOTATIONS_H_
