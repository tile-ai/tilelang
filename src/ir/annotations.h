#ifndef TILELANG_IR_ANNOTATIONS_H_
#define TILELANG_IR_ANNOTATIONS_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/stmt.h>

namespace tvm::tl {

// Decode the frontend's schema without moving expression uses into IR attrs.
// Callers may hold the result while analyzing a scope; any replacement IR must
// put dynamic values back into explicit operands.
TVM_TIRX_DLL ffi::Any DecodeAnnotations(const tvm::Tuple &payload,
                                        const ffi::Array<ffi::Any> &schema);

struct EncodedAnnotations {
  tvm::Tuple payload;
  ffi::Array<ffi::Any> schema;
};

// This is the same version-1 schema as language._native.encode_annotations.
// All expressions, including expressions inside Layout and WS values, are
// explicit uses in payload. The schema contains only static metadata.
TVM_TIRX_DLL EncodedAnnotations EncodeAnnotations(const ffi::Any &annotations);

// Source sblock_attr declarations describe their entire nearest SBlock/Kernel,
// including statements before the declaration. Collect their explicit payloads
// without moving definitions or uses in the IR. Nested source owners and
// function/lambda expressions are not traversed. Native unmarked annotations
// retain their ordinary lexical-sequence semantics.
TVM_TIRX_DLL ffi::Map<ffi::String, ffi::Any>
CollectSourceBlockAnnotations(const SeqStmt &body);

// Re-encode producer metadata after analysis before constructing replacement
// IR. operands excludes the final metadata Tuple; ty is the native result type.
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
