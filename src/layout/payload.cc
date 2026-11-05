/*!
 * \file layout/payload.cc
 * \brief Lossless layout annotation boundary using native shared IR
 * expressions.
 */

#include "payload.h"

#include "expr_utils.h"
#include "support/check.h"

#include <mutex>
#include <optional>
#include <vector>

namespace tvm::tl {
namespace {

struct PayloadCodecs {
  std::mutex mutex;
  std::vector<LayoutPayloadCodec> entries;
};

PayloadCodecs &Codecs() {
  // Backend static initializers may register before this translation unit's
  // FFI initializer. The first caller constructs the registry.
  static auto *registry = new PayloadCodecs();
  return *registry;
}

bool IsBuiltinKind(const ffi::String &kind) {
  return kind == "layout" || kind == "fragment" || kind == "partial";
}

std::optional<LayoutPayloadCodec> FindCodec(const ffi::String &key,
                                            bool by_type) {
  auto &registry = Codecs();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (const auto &codec : registry.entries) {
    if ((by_type ? codec.type_key : codec.kind) == key)
      return codec;
  }
  return std::nullopt;
}

tvm::Tuple PrimTuple(const ffi::Array<PrimExpr> &values) {
  return tvm::Tuple(
      values.Map([](const PrimExpr &value) -> Expr { return value; }));
}

ffi::Array<PrimExpr> ReadPrimTuple(const Expr &value) {
  const auto tuple = value.as_or_throw<tvm::Tuple>();
  return tuple->fields.Map(
      [](const Expr &field) { return field.as_or_throw<PrimExpr>(); });
}

} // namespace

void RegisterLayoutPayloadCodec(LayoutPayloadCodec codec) {
  CHECK(!codec.type_key.empty() && !codec.kind.empty() && codec.encode &&
            codec.decode,
        ValueError)
      << "A layout payload codec needs a type key, kind, encoder and decoder";
  CHECK(!IsBuiltinKind(codec.kind) && codec.type_key != "tl.Layout" &&
            codec.type_key != "tl.Fragment" &&
            codec.type_key != "tl.PartialFragment",
        ValueError)
      << "Cannot replace a built-in layout payload codec";
  auto &registry = Codecs();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (const auto &existing : registry.entries) {
    CHECK(existing.type_key != codec.type_key && existing.kind != codec.kind,
          ValueError)
        << "Duplicate layout payload codec type or kind: " << codec.type_key
        << ", " << codec.kind;
  }
  registry.entries.push_back(std::move(codec));
}

ffi::String LayoutPayloadKind(const Layout &layout) {
  CHECK(layout.defined(), ValueError) << "Cannot encode a null layout";
  ffi::String type_key = layout->GetTypeKey();
  if (type_key == "tl.Layout")
    return "layout";
  if (type_key == "tl.Fragment")
    return "fragment";
  if (type_key == "tl.PartialFragment")
    return "partial";
  auto codec = FindCodec(type_key, true);
  CHECK(codec.has_value(), ValueError)
      << "Unregistered layout payload subtype: " << type_key;
  return codec->kind;
}

tvm::Tuple EncodeLayoutPayload(const Layout &layout) {
  auto kind = LayoutPayloadKind(layout);
  if (!IsBuiltinKind(kind))
    return FindCodec(kind, false)->encode(layout);
  const auto *fragment = layout.as<FragmentNode>();
  const auto *partial = layout.as<PartialFragmentNode>();
  ffi::Array<Var> parameters;
  ffi::Map<PrimVar, PrimExpr> coordinates;
  PrimVar rep("rep", ReplicationPlaceholder().ty());
  parameters.push_back(rep);
  coordinates.Set(ReplicationPlaceholder(), rep);
  for (size_t i = 0; i < layout->InputDim(); ++i) {
    PrimVar index("i" + std::to_string(i), InputPlaceholder(i).ty());
    parameters.push_back(index);
    coordinates.Set(InputPlaceholder(i), index);
  }
  auto indices = SubstituteLayoutVars(layout->GetForwardIndex(), coordinates);
  PrimExpr owner =
      fragment ? SubstituteLayoutVars(fragment->GetForwardThread(), coordinates)
               : PrimExpr(0);
  LambdaExpr mapping(parameters, tvm::Tuple({PrimTuple(indices), owner}));
  PrimExpr replicate = fragment ? fragment->ReplicateExtent() : PrimExpr(1);
  PrimExpr combine = partial ? partial->CombineSize() : PrimExpr(1);
  ffi::Array<Expr> thread_range;
  if (fragment && fragment->ThreadRange().has_value()) {
    auto range = fragment->ThreadRange().value();
    thread_range = {range->begin, range->extent};
  }
  return tvm::Tuple({PrimTuple(layout->InputShape()), mapping, replicate,
                     combine, tvm::Tuple(thread_range)});
}

Layout DecodeLayoutPayload(const tvm::Tuple &payload, const ffi::String &kind,
                           int64_t schema_version) {
  CHECK_EQ(schema_version, 1, ValueError)
      << "Unsupported layout payload version";
  if (!IsBuiltinKind(kind)) {
    auto codec = FindCodec(kind, false);
    CHECK(codec.has_value(), ValueError)
        << "Unknown layout payload kind: " << kind;
    Layout decoded = codec->decode(payload, schema_version);
    CHECK(decoded.defined() && decoded->GetTypeKey() == codec->type_key,
          ValueError)
        << "Layout payload decoder returned a different subtype for " << kind;
    return decoded;
  }
  CHECK_EQ(payload->fields.size(), 5, ValueError)
      << "Layout payload must have five slots";
  auto shape = ReadPrimTuple(payload->fields[0]);
  auto mapping = payload->fields[1].as_or_throw<LambdaExpr>();
  CHECK_EQ(mapping->vars.size(), shape.size() + 1, ValueError)
      << "Layout mapping must bind replication followed by each logical "
         "coordinate";
  ffi::Array<Expr> coordinates{ReplicationPlaceholder()};
  for (size_t i = 0; i < shape.size(); ++i)
    coordinates.push_back(InputPlaceholder(i));
  auto body = mapping->Apply(coordinates).as_or_throw<tvm::Tuple>();
  CHECK_EQ(body->fields.size(), 2, ValueError)
      << "Layout mapping must return (Tuple(indices), owner)";
  auto indices = ReadPrimTuple(body->fields[0]);
  auto owner = body->fields[1].as_or_throw<PrimExpr>();
  auto replicate = payload->fields[2].as_or_throw<PrimExpr>();
  auto combine = payload->fields[3].as_or_throw<PrimExpr>();
  auto range_fields = ReadPrimTuple(payload->fields[4]);
  CHECK(range_fields.empty() || range_fields.size() == 2, ValueError)
      << "Thread range must be empty or (begin, extent)";
  ffi::Optional<Range> thread_range;
  if (!range_fields.empty()) {
    thread_range = Range::FromMinExtent(range_fields[0], range_fields[1]);
  }

  if (kind == "layout") {
    CHECK(prim::IsZero(owner) && prim::IsOne(replicate) &&
              prim::IsOne(combine) && !thread_range.has_value(),
          ValueError)
        << "A plain layout payload cannot carry fragment ownership or "
           "replication";
    return Layout(shape, indices);
  }
  if (kind == "fragment") {
    CHECK(prim::IsOne(combine), ValueError)
        << "Only a partial fragment can have multiple addend lanes";
    return Fragment(shape, indices, owner, replicate, std::nullopt)
        ->BindThreadRange(thread_range);
  }
  return PartialFragment(shape, indices, owner, replicate, combine,
                         std::nullopt, thread_range);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.LayoutPayloadKind", LayoutPayloadKind)
      .def("tl.EncodeLayoutPayload", EncodeLayoutPayload)
      .def_packed("tl.DecodeLayoutPayload",
                  [](ffi::PackedArgs args, ffi::Any *result) {
                    CHECK(args.size() == 2 || args.size() == 3, TypeError)
                        << "DecodeLayoutPayload expects payload, kind, and "
                           "optional schema_version";
                    *result = DecodeLayoutPayload(
                        args[0].cast<tvm::Tuple>(), args[1].cast<ffi::String>(),
                        args.size() == 3 ? args[2].cast<int64_t>() : 1);
                  });
}

} // namespace tvm::tl
