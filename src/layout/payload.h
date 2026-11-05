/*!
 * \file layout/payload.h
 * \brief Native expression payloads for TileLang layout annotations.
 */

#ifndef TVM_TL_LAYOUT_PAYLOAD_H_
#define TVM_TL_LAYOUT_PAYLOAD_H_

#include "layout.h"

namespace tvm::tl {

/*!
 * \brief A backend-owned codec for a concrete Layout subtype.
 *
 * Type keys and payload kinds must be unique and cannot replace the built-in
 * Layout, Fragment or PartialFragment codecs. Every dynamic expression in an
 * extension must remain an operand in the returned native Tuple. Registration
 * is independent of static-initialization order.
 */
struct LayoutPayloadCodec {
  ffi::String type_key;
  ffi::String kind;
  tvm::Tuple (*encode)(const Layout &layout);
  Layout (*decode)(const tvm::Tuple &payload, int64_t schema_version);
};

TVM_TIRX_DLL void RegisterLayoutPayloadCodec(LayoutPayloadCodec codec);

/*! \brief Get the lossless codec kind, rejecting unregistered subtypes. */
TVM_TIRX_DLL ffi::String LayoutPayloadKind(const Layout &layout);

/*!
 * \brief Encode a Layout as a native expression with explicit coordinate
 * binders.
 *
 * Schema version 1 has five slots:
 * (Tuple(shape), Lambda([rep, *coords]) -> (Tuple(indices), owner),
 *  replicate, combine, Tuple([] or [thread_begin, thread_extent])).
 * Only the coordinate placeholders are bound by the lambda. Symbolic shape
 * parameters remain explicit captures visible to native traversal and
 * rewriting. Kind is carried by the enclosing annotation: layout, fragment, or
 * partial. Registered subtypes define their own Tuple schema and kind.
 */
TVM_TIRX_DLL tvm::Tuple EncodeLayoutPayload(const Layout &layout);

/*! \brief Decode and validate version 1 of the native layout payload. */
TVM_TIRX_DLL Layout DecodeLayoutPayload(const tvm::Tuple &payload,
                                        const ffi::String &kind,
                                        int64_t schema_version = 1);

} // namespace tvm::tl

#endif // TVM_TL_LAYOUT_PAYLOAD_H_
