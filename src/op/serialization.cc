#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {

class PrimTypeEncodingNode : public ffi::Object {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ir.PrimTypeEncoding",
                                    PrimTypeEncodingNode, ffi::Object);
};

class DataTypeEncodingNode : public ffi::Object {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ir.DataTypeEncoding",
                                    DataTypeEncodingNode, ffi::Object);
};

class DataTypeImmEncodingNode : public ffi::Object {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ir.DataTypeImmEncoding",
                                    DataTypeImmEncodingNode, ffi::Object);
};

ffi::Array<int64_t> EncodeDType(DLDataType type) {
  return {type.code, type.bits, type.lanes};
}

DLDataType DecodeDType(const ffi::json::Value &data) {
  auto fields = data.cast<ffi::Array<int64_t>>();
  TVM_FFI_CHECK(fields.size() == 3, ValueError)
      << "PrimType encoding must contain [code, bits, lanes]";
  TVM_FFI_CHECK(fields[0] >= 0 && fields[0] <= 255 && fields[1] >= 0 &&
                    fields[1] <= 255 && fields[2] >= 0 && fields[2] <= 65535,
                ValueError)
      << "PrimType encoding is outside the DLPack field ranges";
  return DLDataType{static_cast<uint8_t>(fields[0]),
                    static_cast<uint8_t>(fields[1]),
                    static_cast<uint16_t>(fields[2])};
}

void CollectDTypes(ffi::AnyView value, ffi::Array<ffi::Any> *leaves,
                   std::unordered_set<const ffi::Object *> *seen) {
  if (value.as<DLDataType>()) {
    leaves->push_back(value);
    return;
  }
  const auto *object = value.as<ffi::Object>();
  if (!object || !seen->insert(object).second)
    return;
  if (value.as<PrimType>() || value.as<DataTypeImm>()) {
    leaves->push_back(value);
    return;
  }
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    for (const auto &element : *array)
      CollectDTypes(element, leaves, seen);
  } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
    for (const auto &[key, element] : *map) {
      CollectDTypes(key, leaves, seen);
      CollectDTypes(element, leaves, seen);
    }
  } else {
    ffi::reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(object->type_index()),
        [&](const TVMFFIFieldInfo *field) {
          CollectDTypes(ffi::reflection::FieldGetter(field)(object), leaves,
                        seen);
        });
  }
}

ffi::String SaveJSON(ffi::Any value) {
  ffi::Array<ffi::Any> leaves;
  std::unordered_set<const ffi::Object *> seen;
  CollectDTypes(value, &leaves, &seen);
  auto graph = ffi::ToJSONGraph(ffi::Array<ffi::Any>{value, leaves})
                   .cast<ffi::json::Object>();
  auto nodes = graph.at("nodes").cast<ffi::json::Array>();
  auto roots = nodes[graph.at("root_index").cast<int64_t>()]
                   .cast<ffi::json::Object>()
                   .at("data")
                   .cast<ffi::Array<int64_t>>();
  auto indices = nodes[roots[1]]
                     .cast<ffi::json::Object>()
                     .at("data")
                     .cast<ffi::Array<int64_t>>();
  for (size_t index = 0; index < leaves.size(); ++index) {
    auto node = nodes[indices[index]].cast<ffi::json::Object>();
    if (auto primitive = leaves[index].as<PrimType>()) {
      node.Set("type", "tl.ir.PrimTypeEncoding");
      node.Set("data", EncodeDType(primitive.value()->dtype));
    } else if (auto immediate = leaves[index].as<DataTypeImm>()) {
      node.Set("type", "tl.ir.DataTypeImmEncoding");
      node.Set("data", ffi::json::Object{
                           {"dtype", EncodeDType(immediate.value()->value)},
                           {"loc", ffi::json::Stringify(ffi::ToJSONGraph(
                                       immediate.value()->loc))}});
    } else {
      node.Set("type", "tl.ir.DataTypeEncoding");
      node.Set("data", EncodeDType(leaves[index].cast<DLDataType>()));
    }
    nodes.Set(indices[index], node);
  }
  graph.Set("root_index", roots[0]);
  graph.Set("nodes", nodes);
  graph.Set("metadata", ffi::json::Object{{"tilelang_dtype_encoding", 1}});
  return ffi::json::Stringify(graph);
}

ffi::Any LoadJSON(ffi::String text) {
  auto graph = ffi::json::Parse(text).cast<ffi::json::Object>();
  auto nodes = graph.at("nodes").cast<ffi::json::Array>();
  for (size_t index = 0; index < nodes.size(); ++index) {
    auto node = nodes[index].cast<ffi::json::Object>();
    if (node.at("type").cast<ffi::String>() == "ir.PrimType") {
      if (auto legacy = node.at("data").as<ffi::json::Object>()) {
        auto dtype =
            ffi::StringToDLDataType(legacy->at("dtype").cast<ffi::String>());
        node.Set("data", EncodeDType(dtype));
      }
      node.Set("type", "tl.ir.PrimTypeEncoding");
      nodes.Set(index, node);
    }
  }
  graph.Set("nodes", nodes);
  return ffi::FromJSONGraph(graph);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::ObjectDef<PrimTypeEncodingNode>();
  ffi::reflection::ObjectDef<DataTypeEncodingNode>();
  ffi::reflection::ObjectDef<DataTypeImmEncodingNode>();
  ffi::reflection::TypeAttrDef<PrimTypeEncodingNode>().def(
      ffi::type_attr::kDataFromJson.name,
      [](const ffi::json::Value &data) { return PrimType(DecodeDType(data)); });
  ffi::reflection::TypeAttrDef<DataTypeEncodingNode>().def(
      ffi::type_attr::kDataFromJson.name, DecodeDType);
  ffi::reflection::TypeAttrDef<DataTypeImmEncodingNode>().def(
      ffi::type_attr::kDataFromJson.name, [](const ffi::json::Value &data) {
        auto fields = data.cast<ffi::json::Object>();
        return DataTypeImm(
            DecodeDType(fields.at("dtype")),
            ffi::FromJSONGraph(
                ffi::json::Parse(fields.at("loc").cast<ffi::String>()))
                .cast<Location>());
      });
  ffi::reflection::GlobalDef()
      .def("tl.ir.SaveJSON", SaveJSON)
      .def("tl.ir.LoadJSON", LoadJSON);
}

} // namespace
} // namespace tvm::tl
