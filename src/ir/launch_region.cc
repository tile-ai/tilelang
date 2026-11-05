/*!
 * \file tl/ir/launch_region.cc
 * \brief Typed, target-independent TileLang kernel launch region.
 */

#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/runtime/memory_scope.h>

#include <string>

namespace tvm::tl {

namespace {

using ffi::Array;
using ffi::String;

const TupleObj *LaunchTuple(const Array<Expr> &args, size_t index) {
  TVM_FFI_CHECK_EQ(args.size(), 5, ValueError)
      << "tl.launch_kernel requires begins, ends, steps, threads and dynamic "
         "hints";
  const auto *tuple = args[index].as<TupleObj>();
  TVM_FFI_CHECK(tuple, TypeError)
      << "tl.launch_kernel operand " << index << " must be a Tuple";
  return tuple;
}

void CheckLaunchOperands(const Array<Expr> &args, const DictAttrs &attrs) {
  const auto *begins = LaunchTuple(args, 0);
  const auto *ends = LaunchTuple(args, 1);
  const auto *steps = LaunchTuple(args, 2);
  const auto *threads = LaunchTuple(args, 3);
  LaunchTuple(args, 4);
  TVM_FFI_CHECK_EQ(attrs.GetAttr<int64_t>("schema_version", 0).value(), 1,
                   ValueError)
      << "Unsupported tl.launch_kernel schema";
  TVM_FFI_CHECK_EQ(begins->fields.size(), ends->fields.size(), ValueError);
  TVM_FFI_CHECK_EQ(steps->fields.size(), ends->fields.size(), ValueError);
  for (size_t i = 0; i < ends->fields.size(); ++i) {
    auto begin = begins->fields[i].as_or_throw<IntExpr>();
    auto end = ends->fields[i].as_or_throw<IntExpr>();
    auto step = steps->fields[i].as_or_throw<IntExpr>();
    TVM_FFI_CHECK(ffi::StructuralEqual()(begin.ty(), end.ty()) &&
                      ffi::StructuralEqual()(step.ty(), end.ty()),
                  TypeError)
        << "Kernel range endpoints and step must have one integer type";
    if (const auto *value = step.as<IntImmObj>()) {
      TVM_FFI_CHECK_NE(value->value, 0, ValueError)
          << "Kernel grid step cannot be zero";
    }
  }
  bool specified = attrs.GetAttr<bool>("threads_was_specified", false).value();
  TVM_FFI_CHECK_EQ(threads->fields.size(), specified ? 3 : 0, ValueError)
      << "Kernel must preserve whether threads was specified; explicit threads "
         "has three axes";
  for (const auto &value : threads->fields) {
    auto extent = value.as_or_throw<IntExpr>();
    if (const auto *constant = extent.as<IntImmObj>()) {
      TVM_FFI_CHECK_GT(constant->value, 0, ValueError)
          << "Kernel thread extent must be positive";
    }
  }
  if (auto tags = attrs.GetAttr<Array<String>>("grid_tags")) {
    TVM_FFI_CHECK_EQ(tags.value().size(), ends->fields.size(), ValueError);
  }
  if (auto tags = attrs.GetAttr<Array<String>>("thread_tags")) {
    TVM_FFI_CHECK_EQ(tags.value().size(), 3, ValueError);
  }
}

Array<Var> LaunchBodyParams(const CallObj *call) {
  TVM_FFI_CHECK(call->attrs.has_value(), ValueError)
      << "Kernel requires schema attributes";
  CheckLaunchOperands(call->args, call->attrs.value().as_or_throw<DictAttrs>());
  Array<Var> vars;
  const auto *ends = LaunchTuple(call->args, 1);
  for (size_t i = 0; i < ends->fields.size(); ++i) {
    vars.push_back(PrimVar("block_" + std::to_string(i),
                           ends->fields[i].as_or_throw<IntExpr>().ty()));
  }
  for (const char *name : {"tx", "ty", "tz"}) {
    vars.push_back(PrimVar(name, PrimType::Int(32)));
  }
  return vars;
}

void ValidateLaunch(const RegionStmtObj *region) {
  CheckLaunchOperands(region->args, region->attrs);
  const auto *ends = LaunchTuple(region->args, 1);
  TVM_FFI_CHECK(region->result_vars.empty(), ValueError)
      << "Kernel regions do not return values";
  TVM_FFI_CHECK_EQ(region->body_params.size(), ends->fields.size() + 3,
                   ValueError);
  for (size_t i = 0; i < region->body_params.size(); ++i) {
    Type expected =
        i < ends->fields.size() ? ends->fields[i]->ty : PrimType::Int(32);
    TVM_FFI_CHECK(ffi::StructuralEqual()(region->body_params[i]->ty, expected),
                  TypeError)
        << "Kernel body parameter " << i << " has the wrong type";
    for (size_t j = 0; j < i; ++j) {
      TVM_FFI_CHECK(!region->body_params[i].same_as(region->body_params[j]),
                    ValueError)
          << "Kernel body parameters must be distinct definitions";
    }
  }
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  // These are TileLang storage classes.  Registration permits typed IR
  // construction; backend lowering still has to materialize each class before
  // a generic emitter can allocate it.
  runtime::MemoryScope::Global();
  // The existing CUDA/HIP ABI uses this scope for by-value kernel arguments.
  // Register its identity without giving it an allocation policy.
  ffi::reflection::EnumDef<runtime::MemoryScopeObj>("grid_constant");
  for (const char *name :
       {"local.var", "local.fragment", "local.reducer", "shared.barrier",
        "shared.cluster_barrier", "shared.descriptor", "shared.tmem",
        "local.descriptor.wgmma", "local.descriptor.tcgen05_smem",
        "local.descriptor.tcgen05_instr"}) {
    ffi::reflection::EnumDef<runtime::MemoryScopeObj>(name).set_attr(
        runtime::attr::kMemoryAllocProperty,
        static_cast<int>(runtime::MemoryAllocProperty::kSeparateAllocations));
  }
  OpDef("tl.launch_kernel",
        "TileLang kernel grid and requested SIMT launch configuration.")
      .signature(sig::arg<Tuple>("begins"), sig::arg<Tuple>("ends"),
                 sig::arg<Tuple>("steps"), sig::arg<Tuple>("threads"),
                 sig::arg<Tuple>("dynamic_hints"),
                 sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&LaunchBodyParams>(op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateLaunch>(op_attr::kRegionValidate);
  ffi::reflection::GlobalDef().def("tl.CompilerBuildInfo", []() {
    return ffi::Map<String, String>{
        {"ir", "tvm_tirx.ir"},
        {"launch_schema", "1"},
        {"ffi_ref", TILELANG_FFI_TARGET_REF},
        {"source_digest", TILELANG_COMPILER_SOURCE_DIGEST},
        {"provider_version", TILELANG_PROVIDER_VERSION},
        {"compiler", "tvm_tirx_compiler"}};
  });
  ffi::reflection::GlobalDef().def("tl.SideEffect", [](const Expr &expr) {
    return static_cast<int>(tvm::SideEffect(expr));
  });
}

} // namespace tvm::tl
