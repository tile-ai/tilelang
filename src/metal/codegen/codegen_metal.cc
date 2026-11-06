/*!
 * \file metal/codegen/codegen_metal.cc
 */
#include "codegen_metal.h"
#include "backend/common/codegen/codegen_ir_utils.h"
#include "metal/op/builtin.h"
#include "op/builtin.h"
#include <tvm/tirx/op/memory.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/metal/codegen/metal_fallback_module.h"
#include "codegen/build_common.h"
#include <tvm/ir/expr_functor.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/support/gpu_thread_tag.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace codegen {

namespace {

ffi::Optional<Var> TryMetalMatrixStorageVar(const Expr &expr) {
  if (auto var = expr.as<Var>())
    return var;
  if (const auto *call = expr.as<CallObj>();
      call && call->op.same_as(tirx::tensor_data_ptr_op()) &&
      call->args.size() == 1) {
    return call->args[0].as<Var>();
  }
  return std::nullopt;
}

Var MetalMatrixStorageVar(const Expr &expr) {
  auto var = TryMetalMatrixStorageVar(expr);
  TVM_FFI_CHECK(var.has_value(), TypeError)
      << "Metal matrix operand must name storage or its tensor_data_ptr";
  return var.value();
}

class CooperativeTensorUseCollector : public tirx::StmtExprVisitor {
public:
  ffi::Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    if (const auto *tensor = op->var->ty.as<tirx::TensorTypeObj>();
        tensor && tensor->scope.name() == "metal.cooperative_tensor") {
      uses_cooperative_tensor = true;
    }
    return StmtExprVisitor::Visit_(op);
  }
  ffi::Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (op->op.same_as(tl::cooperative_tensor_load()) ||
        op->op.same_as(tl::cooperative_tensor_store())) {
      uses_cooperative_tensor = true;
      needs_fragment_lane_vars = true;
    } else if (op->op.same_as(tl::cooperative_tensor_fill()) ||
               op->op.same_as(tl::cooperative_tensor_multiply_accumulate())) {
      uses_cooperative_tensor = true;
    }
    return StmtExprVisitor::Visit_(op);
  }
  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (op->op.same_as(tirx::launch_thread_op())) {
      auto tag = op->args[0].as_or_throw<StringImm>()->value;
      auto extent = op->args[1].as_or_throw<PrimExpr>();
      auto [it, inserted] = extents.emplace(tag, extent);
      TVM_FFI_CHECK(inserted || analyzer->CanProveEqual(it->second, extent),
                    ValueError)
          << "Conflicting Metal launch extents for " << tag;
    }
    return StmtExprVisitor::Visit_(op);
  }
  bool uses_cooperative_tensor{false};
  bool needs_fragment_lane_vars{false};
  std::unordered_map<ffi::String, PrimExpr> extents;
  sym::Analyzer analyzer;
};

std::string MetalAddressSpaceForStorageScope(const std::string &scope) {
  if (scope == "global") {
    return "device";
  }
  if (scope == "shared" || scope == "shared.dyn" || scope == "threadgroup") {
    return "threadgroup";
  }
  if (scope == "local" || scope == "local.var" || scope == "metal.simdgroup" ||
      scope == "metal.cooperative_tensor") {
    return "thread";
  }
  TVM_FFI_THROW(InternalError)
      << "Unknown Metal storage scope `" << scope << "`";
  return "";
}

} // namespace

void CodeGenTileLangMetal::InitFuncState(const tirx::Function &f) {
  CodeGenTileLangCBase::InitFuncState(f);
  emitted_frag_lane_vars_ = false;
  emitted_metal_simdgroup_id_ = false;
  thread_idx_x_var_ = std::nullopt;
  block_idx_x_var_ = std::nullopt;
  block_idx_y_var_ = std::nullopt;
  active_mlx_swizzle_panel_ = 0;
  active_mlx_swizzle_log_ = 0;
  ct_c_inlined_.clear();
  ct_c_storage_elided_.clear();
  cooperative_tensor_dtype_.clear();
  simdgroup_dtype_.clear();
  CooperativeTensorUseCollector ct_collector;
  ct_collector.Visit(f->body);
  uses_cooperative_tensor_ = ct_collector.uses_cooperative_tensor;
  needs_fragment_lane_vars_ = ct_collector.needs_fragment_lane_vars;
  // analyze the data;
  for (Var arg : f->params) {
    if (auto pointer = arg->ty.as<PtrType>()) {
      alloc_storage_scope_[arg.get()] = pointer.value()->scope;
    }
  }
}

CodeGenTileLangMetal::CodeGenTileLangMetal(Target target) : target_(target) {
  restrict_keyword_ = "__restrict";
  decl_stream << "union __TVMArgUnion {\n"
              << " int v_int[2];\n"
              << "};\n\n";
}

void CodeGenTileLangMetal::EmitPreamble_(std::ostream &os) const {
  os << "#include <metal_stdlib>\n";
  os << "#include <metal_simdgroup>\n";
  if (uses_cooperative_tensor_) {
    os << "#include "
          "<MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  }
  os << "#define TILELANG_PRAGMA_UNROLL _Pragma(\"clang loop "
        "unroll(full)\")\n";
  os << "using namespace metal;\n\n";
}

void CodeGenTileLangMetal::AddFunction(const GlobalVar &gvar,
                                       const tirx::Function &func) {
  // The Metal runtime binds leading pointers followed by one 8-byte-per-scalar
  // constant buffer. This is the runtime ABI, independent of tensor views.
  InitFuncState(func);
  name_supply_->FreshName("v_");
  auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol).value();
  CooperativeTensorUseCollector collector;
  collector.Visit(func->body);
  int work_dim = 0;
  int64_t total_threads = 1;
  bool has_thread_extent = false;
  bool static_thread_extent = true;
  for (const auto &[tag, extent] : collector.extents) {
    auto scope = support::GPUThreadTag::Parse(tag);
    TVM_FFI_CHECK(scope.rank == 0 || scope.rank == 1, ValueError)
        << "Unsupported Metal launch tag " << tag;
    work_dim = std::max(work_dim, scope.dim_index + 1);
    if (scope.rank == 1) {
      has_thread_extent = true;
      if (const auto *constant = extent.as<IntImmObj>()) {
        const int64_t n = CodegenInt(constant);
        TVM_FFI_CHECK_GT(n, 0, ValueError);
        TVM_FFI_CHECK_LE(n, std::numeric_limits<int64_t>::max() / total_threads,
                         ValueError);
        total_threads *= n;
      } else {
        static_thread_extent = false;
      }
    }
  }
  for (const auto &tag : func->GetAttr(tirx::attr::kKernelLaunchParams)
                             .value_or(ffi::Array<ffi::String>{})) {
    if (tag == runtime::launch_param::kUseDynamicSharedMemoryTag)
      continue;
    auto scope = support::GPUThreadTag::Parse(tag);
    TVM_FFI_CHECK(scope.rank == 0 || scope.rank == 1, ValueError);
    work_dim = std::max(work_dim, scope.dim_index + 1);
  }
  if (has_thread_extent && static_thread_extent) {
    stream << "[[kernel, max_total_threads_per_threadgroup(" << total_threads
           << ")]] void ";
  } else {
    stream << "kernel void ";
  }
  stream << global_symbol << "(";
  bool first_arg = true;
  auto next_arg = [&]() {
    if (!first_arg)
      stream << ",\n";
    first_arg = false;
    stream << "  ";
  };
  const size_t limit =
      target_->GetAttr<int64_t>("max_function_args").value_or(31);
  if (func->params.size() > limit)
    LOG(WARNING) << "Metal function exceeds max_function_args=" << limit;
  bool no_alias = func->HasNonzeroAttr(tirx::attr::kNoAlias.name);
  std::unordered_set<int64_t> readonly;
  if (auto values =
          func->GetAttr<ffi::Array<ffi::Any>>("tl.readonly_param_indices")) {
    for (const auto &value : values.value())
      readonly.insert(value.as<IntImmObj>() ? CodegenInt(value.as<IntImmObj>())
                                            : value.cast<int64_t>());
  }
  size_t num_buffer = 0;
  for (; num_buffer < func->params.size(); ++num_buffer) {
    Var var = func->params[num_buffer];
    auto pointer = var->ty.as<PtrType>();
    if (!pointer)
      break;
    TVM_FFI_CHECK_EQ(pointer.value()->scope.name(), "global", ValueError)
        << "Metal kernel buffer arguments require global pointer scope";
    next_arg();
    if (readonly.count(num_buffer))
      stream << "const ";
    PrintStorageScope(pointer.value()->scope, stream);
    PrintType(var->ty, stream);
    RegisterHandleType(var.get(),
                       pointer.value()->element_type.as_or_throw<PrimType>());
    if (no_alias)
      PrintRestrict(var, stream);
    stream << ' ' << AllocVarID(var.get()) << " [[ buffer(" << num_buffer
           << ") ]]";
  }
  if (num_buffer != func->params.size()) {
    std::string arg_name = name_supply_->FreshName("arg");
    std::string arg_type = std::string(global_symbol) + "_args_t";
    next_arg();
    stream << "constant " << arg_type << "& " << arg_name << " [[ buffer("
           << num_buffer << ") ]]";
    decl_stream << "struct " << arg_type << " {\n";
    for (size_t i = num_buffer; i < func->params.size(); ++i) {
      Var var = func->params[i];
      TVM_FFI_CHECK(!var->ty.as<PtrTypeObj>(), TypeError)
          << "Metal kernel arguments require all pointers before scalar "
             "parameters; run host/device ABI lowering before code generation";
      auto dtype = var->ty.as_or_throw<PrimType>();
      TVM_FFI_CHECK(dtype.IsScalar() && !dtype.IsVoid(), TypeError)
          << "Metal scalar parameter must have scalar primitive type";
      const std::string id = AllocVarID(var.get());
      PrimType storage_type =
          dtype.MatchesCode(kDLBool) ? PrimType::Int(32) : dtype;
      TVM_FFI_CHECK(storage_type == PrimType::Int(32) ||
                        storage_type == PrimType::UInt(32) ||
                        storage_type == PrimType::Float(32) ||
                        storage_type == PrimType::Int(64),
                    TypeError)
          << "Unsupported Metal runtime scalar argument type " << dtype;
      decl_stream << "  ";
      PrintType(storage_type, decl_stream);
      decl_stream << ' ' << id;
      std::string ref = arg_name + "." + id;
      if (storage_type.bits() == 32) {
        decl_stream << "[2]";
        ref += "[0]";
      }
      decl_stream << ";\n";
      var_idmap_[var.get()] =
          dtype.MatchesCode(kDLBool) ? "bool(" + ref + ")" : ref;
    }
    decl_stream << "};\n\n";
  }
  TVM_FFI_ICHECK_EQ(name_supply_->FreshName("threadIdx"), "threadIdx");
  TVM_FFI_ICHECK_EQ(name_supply_->FreshName("blockIdx"), "blockIdx");
  if (work_dim != 0) {
    for (const auto &[name, attribute] :
         std::vector<std::pair<const char *, const char *>>{
             {"blockIdx", "threadgroup_position_in_grid"},
             {"__gridDim", "threadgroups_per_grid"},
             {"threadIdx", "thread_position_in_threadgroup"}}) {
      next_arg();
      PrintType(PrimType::UInt(thread_index_bits_, work_dim), stream);
      stream << ' ' << name << " [[" << attribute << "]]";
    }
    next_arg();
    stream << "uint __simd_group_id [[simdgroup_index_in_threadgroup]]";
    emitted_metal_simdgroup_id_ = true;
  }
  thread_work_dim_ = work_dim;
  stream << ") {\n";
  int scope = BeginScope();
  if (needs_fragment_lane_vars_)
    EnsureFragmentLaneVars();
  PreFunctionBody(func);
  PrintStmt(func->body);
  EndScope(scope);
  PrintIndent();
  stream << "}\n\n";
}

void CodeGenTileLangMetal::BindThreadIndex(const PrimVar &var,
                                           const ffi::String &thread_tag) {
  TVM_FFI_ICHECK(!var_idmap_.count(var.get()));
  // if we only have threadIdx.x
  // metal will directly print as threadIdx
  std::string vname = thread_tag;
  if (thread_work_dim_ <= 1) {
    vname = vname.substr(0, thread_tag.length() - 2);
  }
  var_idmap_[var.get()] =
      CastFromTo(vname, PrimType::UInt(thread_index_bits_), var.ty());
  if (thread_tag == "threadIdx.x") {
    thread_idx_x_var_ = var;
  } else if (thread_tag == "blockIdx.x") {
    block_idx_x_var_ = var;
  } else if (thread_tag == "blockIdx.y") {
    block_idx_y_var_ = var;
  }
}

void CodeGenTileLangMetal::PrintType(const PrimType &t,
                                     std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.IsVoid()) {
    os << "void";
    return;
  }
  if (t.MatchesCode(kDLOpaqueHandle)) {
    TVM_FFI_ICHECK_EQ(lanes, 1) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t == PrimType::Bool()) {
    os << "bool";
    return;
  }
  if ((t.MatchesElementType(kDLFloat, 16) ||
       t.MatchesElementType(kDLBfloat, 16)) &&
      lanes > 4 && lanes <= 8 && lanes % 2 == 0) {
    os << "uint" << lanes / 2;
    return;
  }
  bool fail = false;
  if (t.MatchesCode(kDLFloat)) {
    if (lanes == 3) {
      os << "packed_";
    }
    switch (t.bits()) {
    case 16:
      os << "half";
      break;
    case 32:
      os << "float";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.MatchesCode(kDLUInt) || t.MatchesCode(kDLInt)) {
    if (t.MatchesCode(kDLUInt)) {
      os << 'u';
    }
    switch (t.bits()) {
    case 8:
      os << "char";
      break;
    case 16:
      os << "short";
      break;
    case 32:
      os << "int";
      break;
    case 64:
      os << "long";
      break;
    case 1:
      os << "bool";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    os << "bfloat";
    return;
  }
  TVM_FFI_THROW(InternalError)
      << "Cannot convert type " << t << " to Metal type";
}

void CodeGenTileLangMetal::PrintStorageSync(const CallObj *op) {
  TVM_FFI_CHECK_EQ(op->args.size(), 1U, ValueError)
      << "Metal code generation does not support named/partial barriers";
  const std::string &sync = op->args[0].as<StringImmObj>()->value;
  if (sync == "warp") {
    // Metal has no per-simdgroup barrier; simdgroup_barrier synchronizes
    // the entire threadgroup (same as threadgroup_barrier).  We emit the
    // narrower intrinsic so the source documents the intended scope even
    // though the hardware effect is identical.
    this->PrintIndent();
    this->stream << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    this->stream << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "global") {
    TVM_FFI_THROW(InternalError) << "global barrier not supported";
  }
}

void CodeGenTileLangMetal::PrintVecElemLoad(const std::string &vec,
                                            const PrimType &t, int i,
                                            std::ostream &os) { // NOLINT(*)
  if (t.MatchesElementType(kDLFloat, 16) && t.lanes() > 4) {
    os << "((thread half*)(&" << vec << "))[" << i << "]";
  } else if (t.MatchesElementType(kDLBfloat, 16) && t.lanes() > 4) {
    os << "((thread bfloat*)(&" << vec << "))[" << i << "]";
  } else {
    os << vec << "[" << i << "]";
  }
}

void CodeGenTileLangMetal::PrintVecElemStore(const std::string &vec,
                                             const PrimType &t, int i,
                                             const std::string &value) {
  this->PrintIndent();
  if (t.MatchesElementType(kDLFloat, 16) && t.lanes() > 4) {
    stream << "((thread half*)(&" << vec << "))[" << i << "] = " << value
           << ";\n";
  } else if (t.MatchesElementType(kDLBfloat, 16) && t.lanes() > 4) {
    stream << "((thread bfloat*)(&" << vec << "))[" << i << "] = " << value
           << ";\n";
  } else {
    stream << vec << "[" << i << "]"
           << " = " << value << ";\n";
  }
}

void CodeGenTileLangMetal::PrintStorageScope(const runtime::MemoryScope &scope,
                                             std::ostream &os) { // NOLINT(*)
  os << MetalAddressSpaceForStorageScope(scope.name()) << " ";
}

void CodeGenTileLangMetal::Dispatch_(const BindObj *op) {
  if (const auto *call = op->value.as<CallObj>();
      call && call->op.same_as(tirx::alloc_tensor_op())) {
    PrintTensorAllocation(op);
    return;
  }
  const auto *pointer = op->var->ty.as<PtrTypeObj>();
  if (!pointer) {
    CodeGenC::Dispatch_(op);
    return;
  }
  const PrimType element = pointer->element_type.as_or_throw<PrimType>();
  alloc_storage_scope_[op->var.get()] = pointer->scope;
  RegisterHandleType(op->var.get(), element);
  RegisterHandleTypeFromPointer(op->var, &op->value);
  const std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    TVM_FFI_ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
    return;
  }
  PrintIndent();
  PrintStorageScope(pointer->scope, stream);
  PrintType(element, stream);
  stream << "* " << AllocVarID(op->var.get()) << " = (";
  PrintStorageScope(pointer->scope, stream);
  PrintType(element, stream);
  stream << "*)" << value << ";\n";
}

void CodeGenTileLangMetal::Dispatch_(const RegionStmtObj *op) {
  if (op->op->name != "tl.attr") {
    CodeGenC::Dispatch_(op);
    return;
  }
  auto attr = DecodeTileLangAttribute(op);
  if (attr.key == "threadblock_swizzle_pattern") {
    std::string func_name;
    int panel_size = 0;
    if (auto tuple = attr.value.as<tvm::Tuple>()) {
      TVM_FFI_CHECK_EQ(tuple.value()->fields.size(), 2, ValueError);
      func_name = tuple.value()->fields[0].as_or_throw<StringImm>()->value;
      panel_size = CodegenInt(tuple.value()->fields[1]);
    } else {
      auto values = attr.value.cast<ffi::Array<ffi::Any>>();
      TVM_FFI_CHECK_EQ(values.size(), 2, ValueError);
      func_name = CodegenAnnotationString(values[0]);
      panel_size = values[1].as<IntImmObj>()
                       ? CodegenInt(values[1].as<IntImmObj>())
                       : values[1].cast<int>();
    }
    TVM_FFI_ICHECK(!func_name.empty() && panel_size > 0);

    this->PrintIndent();
    stream << "{ ";
    if (func_name == "rasterization2DRow") {
      stream << "const uint __bi = blockIdx.x + blockIdx.y * __gridDim.x; "
                "const uint __gs = __gridDim.x * __gridDim.y; "
                "const uint __ps = "
             << panel_size
             << "u * __gridDim.x; "
                "const uint __po = __bi % __ps; "
                "const uint __pi = __bi / __ps; "
                "const uint __tp = (__gs + __ps - 1u) / __ps; "
                "const uint __st = __pi + 1u < __tp ? "
             << panel_size
             << "u : (__gs - __pi * __ps) / __gridDim.x; "
                "const uint3 blockIdx = uint3("
                "(__pi & 1u) ? __gridDim.x - 1u - __po / __st : __po / __st, "
                "__po % __st + __pi * "
             << panel_size
             << "u, "
                "blockIdx.z);\n";
    } else if (func_name == "rasterization2DColumn") {
      stream << "const uint __bi = blockIdx.x + blockIdx.y * __gridDim.x; "
                "const uint __gs = __gridDim.x * __gridDim.y; "
                "const uint __ps = "
             << panel_size
             << "u * __gridDim.y; "
                "const uint __po = __bi % __ps; "
                "const uint __pi = __bi / __ps; "
                "const uint __tp = (__gs + __ps - 1u) / __ps; "
                "const uint __st = __pi + 1u < __tp ? "
             << panel_size
             << "u : (__gs - __pi * __ps) / __gridDim.y; "
                "const uint3 blockIdx = uint3("
                "__po % __st + __pi * "
             << panel_size
             << "u, "
                "(__pi & 1u) ? __gridDim.y - 1u - __po / __st : __po / __st, "
                "blockIdx.z);\n";
    } else if (func_name == "rasterization2DMLX") {
      TVM_FFI_ICHECK_EQ(panel_size & (panel_size - 1), 0)
          << "Metal MLX swizzle panel size must be a power of two, got "
          << panel_size;
      int swizzle_log = 0;
      while ((1 << swizzle_log) < panel_size) {
        swizzle_log++;
      }
      if (thread_work_dim_ <= 1) {
        stream << "const uint3 __physical_blockIdx = "
                  "uint3(blockIdx, 0u, 0u); ";
      } else if (thread_work_dim_ == 2) {
        stream << "const uint3 __physical_blockIdx = "
                  "uint3(blockIdx.x, blockIdx.y, 0u); ";
      } else {
        stream << "const uint3 __physical_blockIdx = "
                  "uint3(blockIdx.x, blockIdx.y, blockIdx.z); ";
      }
      stream << "const uint3 blockIdx = uint3("
             << "(__physical_blockIdx.x >> " << swizzle_log << "), "
             << "((__physical_blockIdx.y << " << swizzle_log
             << ") + (__physical_blockIdx.x & " << (panel_size - 1) << "u)), "
             << "__physical_blockIdx.z);\n";
      int old_panel = active_mlx_swizzle_panel_;
      int old_log = active_mlx_swizzle_log_;
      active_mlx_swizzle_panel_ = panel_size;
      active_mlx_swizzle_log_ = swizzle_log;
      this->Dispatch(op->body);
      active_mlx_swizzle_panel_ = old_panel;
      active_mlx_swizzle_log_ = old_log;
      this->PrintIndent();
      stream << "}\n";
      return;
    } else {
      TVM_FFI_THROW(InternalError)
          << "Unknown Metal threadblock swizzle pattern `" << func_name << "`";
    }
    this->Dispatch(op->body);
    this->PrintIndent();
    stream << "}\n";
    return;
  }
  if (attr.key == "volatile_scope")
    MarkVolatile(attr.node.cast<Var>().get());
  PrintStmt(op->body);
}

void CodeGenTileLangMetal::Dispatch_(const ForObj *op) {
  sym::Analyzer analyzer;
  PrimExpr extent = analyzer->Simplify(op->end - op->begin);
  const auto *min_imm = op->begin.as<IntImmObj>();
  const auto *ext_imm = extent.as<IntImmObj>();
  bool body_has_alloc = false;
  bool needs_ct_index_const = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      op->body,
      [&](const BindObj *bind) -> ffi::Expected<ffi::WalkResult> {
        if (const auto *call = bind->value.as<CallObj>();
            call && call->op.same_as(tirx::alloc_tensor_op()))
          body_has_alloc = true;
        return ffi::WalkResult::Advance();
      },
      [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        if (call->op.same_as(tl::cooperative_tensor_fill()) ||
            call->op.same_as(tl::cooperative_tensor_load()) ||
            call->op.same_as(tl::cooperative_tensor_store()) ||
            call->op.same_as(tl::cooperative_tensor_multiply_accumulate())) {
          auto args = call->args;
          if (args.size() >= 2 && !args[1].as<IntImmObj>()) {
            if (auto var = TryMetalMatrixStorageVar(args[0]))
              needs_ct_index_const |=
                  ct_c_inlined_.count(var.value()) ||
                  cooperative_tensor_dtype_.count(var.value());
          }
        }
        return ffi::WalkResult::Advance();
      });
  if (ext_imm && min_imm && ext_imm->value >= 2 && ext_imm->value <= 4 &&
      prim::IsOne(op->step) && !body_has_alloc && needs_ct_index_const) {
    const int64_t begin = CodegenInt(min_imm);
    const int64_t count = CodegenInt(ext_imm);
    for (int64_t i = 0; i < count; ++i) {
      auto substitute = [&](const Var &var, TVMFFIDefRegionKind kind)
          -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (kind == kTVMFFIDefRegionKindNone && var.same_as(op->var))
          return ffi::Any(IntImm(op->var.ty(), begin + i));
        return ffi::Unchanged();
      };
      Stmt body =
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(op->body, substitute)
              .as_or_throw<Stmt>();
      body = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                 body,
                 [&](const PrimExpr &expr) -> PrimExpr {
                   return analyzer->Simplify(expr);
                 })
                 .as_or_throw<Stmt>();
      auto saved_vars = var_idmap_;
      PrintStmt(body);
      var_idmap_ = std::move(saved_vars);
    }
    return;
  }
  if (ext_imm && ext_imm->value > 4) {
    PrintIndent();
    stream << "#pragma clang loop unroll(disable)\n";
  }
  CodeGenC::Dispatch_(op);
}

void CodeGenTileLangMetal::PrintTensorAllocation(const BindObj *op) {
  const auto *allocation = op->value.as<CallObj>();
  const auto tensor = op->var.as_or_throw<tirx::TensorVar>();
  const auto *attrs = allocation->attrs.as<DictAttrsObj>();
  const std::string vid = AllocVarID(tensor.get());
  PrintIndent();
  size_t constant_size = CodegenAllocationElements(tensor);
  PrimType dtype = tensor.dtype();
  auto scope = tensor.scope().name();
  alloc_storage_scope_[tensor.get()] = tensor.scope();
  if (scope == "metal.cooperative_tensor") {
    uses_cooperative_tensor_ = true;
    TVM_FFI_ICHECK(dtype == PrimType::Float(16) ||
                   dtype == PrimType::Float(32) ||
                   dtype == PrimType::BFloat(16))
        << "Only float16, float32, and bfloat16 are supported for "
           "cooperative_tensor, but got "
        << dtype;
    TVM_FFI_ICHECK(constant_size % 64 == 0)
        << "cooperative_tensor buffer size must be multiple of 64, got "
        << constant_size;

    std::ostringstream dtype_os;
    PrintType(dtype, dtype_os);
    std::string dtype_str = dtype_os.str();
    cooperative_tensor_dtype_[tensor] = dtype_str;
    int elems_per_thread = constant_size / 32;
    bool can_inline_c = dtype_str == "float" && elems_per_thread >= 16 &&
                        elems_per_thread % 16 == 0;
    int num_c_tiles = can_inline_c ? elems_per_thread / 16 : 0;
    bool elide_c_storage = can_inline_c && num_c_tiles <= 4;
    if (!elide_c_storage) {
      stream << "thread " << dtype_str << " " << vid << '[' << elems_per_thread
             << "];\n";
    }
    if (can_inline_c) {
      ct_c_inlined_.insert(tensor);
      if (elide_c_storage) {
        ct_c_storage_elided_.insert(tensor);
      }
      this->PrintIndent();
      stream
          << "constexpr auto __pct_desc = mpp::tensor_ops::matmul2d_descriptor("
          << "16, 32, 16, false, false, true, "
          << "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);"
             "\n";
      this->PrintIndent();
      stream << "mpp::tensor_ops::matmul2d<__pct_desc, "
                "metal::execution_simdgroup> __pct_op;\n";
      for (int t = 0; t < num_c_tiles; t++) {
        this->PrintIndent();
        stream << "auto __pct_c" << t
               << " = __pct_op.get_destination_cooperative_tensor<"
               << "decltype(__pct_op.get_left_input_cooperative_tensor<half, "
                  "half, float>()), "
               << "decltype(__pct_op.get_right_input_cooperative_tensor<half, "
                  "half, float>()), float>();\n";
      }
    }
  } else if (scope == "metal.simdgroup") {
    TVM_FFI_ICHECK(dtype == PrimType::Float(16) ||
                   dtype == PrimType::Float(32) ||
                   dtype == PrimType::BFloat(16))
        << "Only float16, float32, and bfloat16 are supported, but got "
        << dtype;
    TVM_FFI_ICHECK(constant_size % 64 == 0)
        << "Only 8x8 matrix is supported, but got " << constant_size
        << " bytes\n";

    std::ostringstream dtype_os;
    PrintType(dtype, dtype_os);
    std::string dtype_str = dtype_os.str();
    simdgroup_dtype_[tensor] = dtype_str;
    stream << "simdgroup_" << dtype_str << "8x8 " << vid << '['
           << constant_size / 64 << "];\n";
  } else {
    // Apply 16-byte alignment padding to shared/threadgroup memory
    // to avoid bank conflicts on Apple GPUs (following MLX practice).
    // Apple GPU shared memory has banks of 16 bytes; padding prevents
    // the same bank being accessed by multiple threads simultaneously.
    if (scope == "shared" || scope == "threadgroup") {
      int dtype_bytes = dtype.StorageBytes();
      int padding_elems = (dtype_bytes > 0) ? (16 / dtype_bytes) : 0;
      constant_size += padding_elems;
    }
    PrintStorageScope(tensor.scope(), stream);
    PrintType(dtype, stream);
    if (scope == "local.var") {
      PrimExpr init = prim::MakeConst(tensor.dtype(), 0);
      if (auto initializer =
              attrs ? attrs->dict.Get("tl.local_var_init") : std::nullopt) {
        PrimExpr user_init = initializer->cast<PrimExpr>();
        if (!user_init.ty().IsVoid() && user_init.ty() != tensor.dtype()) {
          user_init = prim::Cast(tensor.dtype(), user_init);
        }
        init = user_init;
      }
      stream << ' ' << vid << " = " << PrintExpr(init) << ";\n";
    } else {
      stream << ' ' << vid << '[' << constant_size << "];\n";
    }
  }

  RegisterHandleType(tensor.get(), dtype);
  if (attrs &&
      CodegenBool(attrs->dict.Get(tirx::attr::kVolatile.name).value_or(false)))
    MarkVolatile(tensor.get());
}

void CodeGenTileLangMetal::Dispatch_(const TensorLoadObj *op,
                                     std::ostream &os) {
  if (CodegenStorageScope(op->source.as_or_throw<Var>()) == "local.var") {
    TVM_FFI_ICHECK_EQ(op->indices.size(), 1)
        << "Load from non-flat local.var not supported.";
    os << GetVarID(op->source.as_or_throw<Var>().get());
    return;
  }
  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangMetal::Dispatch_(const TensorStoreObj *op) {
  if (CodegenStorageScope(op->dest.as_or_throw<Var>()) == "local.var") {
    TVM_FFI_ICHECK_EQ(op->indices.size(), 1)
        << "Store to non-flat local.var not supported.";
    PrintIndent();
    stream << GetVarID(op->dest.as_or_throw<Var>().get()) << " = "
           << PrintExpr(op->value) << ";\n";
    return;
  }
  CodeGenC::Dispatch_(op);
}

void CodeGenTileLangMetal::EnsureCooperativeTensorBuffer(const Var &var) {
  if (cooperative_tensor_dtype_.count(var) != 0) {
    return;
  }
  auto type_it = handle_data_type_.find(var.get());
  TVM_FFI_ICHECK(type_it != handle_data_type_.end())
      << "Cannot find variable allocation for cooperative_tensor: " << var;
  std::ostringstream dtype_os;
  PrintType(type_it->second, dtype_os);
  cooperative_tensor_dtype_[var] = dtype_os.str();
}

void CodeGenTileLangMetal::EnsureFragmentLaneVars() {
  if (emitted_frag_lane_vars_) {
    return;
  }
  this->PrintIndent();
  stream << "const ushort __lane = "
            "__metal_get_thread_index_in_simdgroup(ushort());\n";
  this->PrintIndent();
  stream << "const ushort __qid = __lane >> 2;\n";
  this->PrintIndent();
  stream << "const ushort __base_row = (__qid & 4) | ((__lane >> 1) & 3);\n";
  this->PrintIndent();
  stream << "const ushort __base_col = ((__qid & 2) | (__lane & 1)) * 4;\n";
  emitted_frag_lane_vars_ = true;
}

void CodeGenTileLangMetal::Dispatch_(const prim::SelectObj *op,
                                     std::ostream &os) { // NOLINT(*)
  os << "select(" << PrintExpr(op->false_value) << ", "
     << PrintExpr(op->true_value) << ", " << PrintExpr(op->condition) << ")";
}

void CodeGenTileLangMetal::Dispatch_(const prim::BroadcastObj *op,
                                     std::ostream &os) { // NOLINT(*)
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  std::string v = PrintExpr(op->value);
  int lanes = dtype.lanes();
  if ((dtype.MatchesElementType(kDLFloat, 16) ||
       dtype.MatchesElementType(kDLBfloat, 16)) &&
      lanes > 4 && lanes % 2 == 0) {
    os << "uint" << lanes / 2 << "(";
    for (int i = 0; i < lanes / 2; ++i) {
      if (i != 0)
        os << ", ";
      if (dtype.MatchesElementType(kDLFloat, 16)) {
        os << "as_type<uint>(half2(" << v << ", " << v << "))";
      } else if (dtype.MatchesElementType(kDLBfloat, 16)) {
        os << "as_type<uint>(bfloat2(" << v << ", " << v << "))";
      }
    }
    os << ')';
  } else {
    PrintType(dtype, os);
    os << "(";
    for (int i = 0; i < lanes; ++i) {
      if (i != 0)
        os << ", ";
      os << v;
    }
    os << ')';
  }
}

void CodeGenTileLangMetal::Dispatch_(const prim::AddObj *op,
                                     std::ostream &os) { // NOLINT(*)
  if (TryPrintMlxSwizzleExpr(ffi::GetRef<Expr>(op), os)) {
    return;
  }
  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangMetal::Dispatch_(const prim::CastObj *op,
                                     std::ostream &os) { // NOLINT(*)
  std::ostringstream value;
  if (TryPrintMlxSwizzleExpr(op->value, value)) {
    os << CastFromTo(value.str(), op->value.ty(),
                     op->ty.as_or_throw<PrimType>());
    return;
  }
  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangMetal::Dispatch_(const prim::RShiftObj *op,
                                     std::ostream &os) { // NOLINT(*)
  if (TryPrintMlxSwizzleExpr(ffi::GetRef<Expr>(op), os)) {
    return;
  }
  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangMetal::Dispatch_(const prim::BitwiseAndObj *op,
                                     std::ostream &os) { // NOLINT(*)
  if (TryPrintMlxSwizzleExpr(ffi::GetRef<Expr>(op), os)) {
    return;
  }
  CodeGenC::Dispatch_(op, os);
}

std::string
CodeGenTileLangMetal::GetStorageScopeOf(const Expr &ptr_expr) const {
  if (const auto *pointer = ptr_expr->ty.as<PtrTypeObj>())
    return pointer->scope.name();
  if (const auto *tensor = ptr_expr->ty.as<tirx::TensorTypeObj>())
    return tensor->scope.name();
  TVM_FFI_THROW(TypeError)
      << "Expected a typed pointer or tensor for Metal address space, got "
      << ptr_expr->ty;
}

std::string CodeGenTileLangMetal::GetAddrSpaceOf(const Expr &ptr_expr) const {
  std::string storage_scope = GetStorageScopeOf(ptr_expr);
  TVM_FFI_ICHECK(!storage_scope.empty())
      << "Cannot determine Metal storage scope for pointer expression "
      << ptr_expr;
  return MetalAddressSpaceForStorageScope(storage_scope);
}

std::string
CodeGenTileLangMetal::GetPointeeTypeOf(const Expr &ptr_expr,
                                       const std::string &fallback) {
  std::ostringstream os;
  if (const auto *pointer = ptr_expr->ty.as<PtrTypeObj>()) {
    PrintType(pointer->element_type, os);
    return os.str();
  }
  if (const auto *tensor = ptr_expr->ty.as<tirx::TensorTypeObj>()) {
    PrintType(tensor->dtype, os);
    return os.str();
  }
  return fallback;
}

bool CodeGenTileLangMetal::IsThreadIdxXExpr(const Expr &expr) const {
  if (!thread_idx_x_var_.has_value()) {
    return false;
  }
  if (const auto *cast = expr.as<prim::CastObj>()) {
    return IsThreadIdxXExpr(cast->value);
  }
  if (const auto *var = expr.as<VarObj>()) {
    return ffi::GetRef<Var>(var).same_as(thread_idx_x_var_.value());
  }
  return false;
}

bool CodeGenTileLangMetal::IsBlockIdxExpr(const Expr &expr, int dim) const {
  const Expr *current = &expr;
  while (const auto *cast = current->as<prim::CastObj>()) {
    current = &cast->value;
  }
  const auto *var = current->as<VarObj>();
  if (!var) {
    return false;
  }
  Var var_ref = ffi::GetRef<Var>(var);
  if (dim == 0) {
    return block_idx_x_var_.has_value() &&
           var_ref.same_as(block_idx_x_var_.value());
  }
  if (dim == 1) {
    return block_idx_y_var_.has_value() &&
           var_ref.same_as(block_idx_y_var_.value());
  }
  return false;
}

bool CodeGenTileLangMetal::IsConstIntExpr(const Expr &expr,
                                          int64_t value) const {
  const auto *imm = expr.as<IntImmObj>();
  return imm && imm->value == value;
}

bool CodeGenTileLangMetal::IsMlxPanelRemainderExpr(const Expr &expr) const {
  if (active_mlx_swizzle_panel_ <= 0) {
    return false;
  }
  int64_t mask = active_mlx_swizzle_panel_ - 1;
  if (const auto *bitwise = expr.as<prim::BitwiseAndObj>()) {
    return (IsBlockIdxExpr(bitwise->a, 0) &&
            IsConstIntExpr(bitwise->b, mask)) ||
           (IsBlockIdxExpr(bitwise->b, 0) && IsConstIntExpr(bitwise->a, mask));
  }
  if (const auto *mod = expr.as<prim::ModObj>()) {
    return IsBlockIdxExpr(mod->a, 0) &&
           IsConstIntExpr(mod->b, active_mlx_swizzle_panel_);
  }
  if (const auto *mod = expr.as<prim::FloorModObj>()) {
    return IsBlockIdxExpr(mod->a, 0) &&
           IsConstIntExpr(mod->b, active_mlx_swizzle_panel_);
  }
  return false;
}

bool CodeGenTileLangMetal::IsMlxPanelRowExpr(const Expr &expr) const {
  if (active_mlx_swizzle_panel_ <= 0) {
    return false;
  }
  if (active_mlx_swizzle_log_ > 0) {
    if (const auto *shift = expr.as<prim::LShiftObj>()) {
      return IsBlockIdxExpr(shift->a, 1) &&
             IsConstIntExpr(shift->b, active_mlx_swizzle_log_);
    }
  }
  if (const auto *mul = expr.as<prim::MulObj>()) {
    return (IsBlockIdxExpr(mul->a, 1) &&
            IsConstIntExpr(mul->b, active_mlx_swizzle_panel_)) ||
           (IsBlockIdxExpr(mul->b, 1) &&
            IsConstIntExpr(mul->a, active_mlx_swizzle_panel_));
  }
  return active_mlx_swizzle_panel_ == 1 && IsBlockIdxExpr(expr, 1);
}

bool CodeGenTileLangMetal::IsMlxLogicalBlockXExpr(const Expr &expr) const {
  if (active_mlx_swizzle_panel_ <= 0) {
    return false;
  }
  if (const auto *shift = expr.as<prim::RShiftObj>()) {
    return IsBlockIdxExpr(shift->a, 0) &&
           IsConstIntExpr(shift->b, active_mlx_swizzle_log_);
  }
  if (const auto *div = expr.as<prim::DivObj>()) {
    return IsBlockIdxExpr(div->a, 0) &&
           IsConstIntExpr(div->b, active_mlx_swizzle_panel_);
  }
  if (const auto *div = expr.as<prim::FloorDivObj>()) {
    return IsBlockIdxExpr(div->a, 0) &&
           IsConstIntExpr(div->b, active_mlx_swizzle_panel_);
  }
  return active_mlx_swizzle_panel_ == 1 && IsBlockIdxExpr(expr, 0);
}

bool CodeGenTileLangMetal::IsMlxLogicalBlockYExpr(const Expr &expr) const {
  if (active_mlx_swizzle_panel_ <= 0) {
    return false;
  }
  if (IsMlxPanelRemainderExpr(expr)) {
    return true;
  }
  if (const auto *add = expr.as<prim::AddObj>()) {
    return (IsMlxPanelRowExpr(add->a) && IsMlxPanelRemainderExpr(add->b)) ||
           (IsMlxPanelRowExpr(add->b) && IsMlxPanelRemainderExpr(add->a));
  }
  return false;
}

bool CodeGenTileLangMetal::TryPrintMlxLogicalYAffineExpr(const Expr &expr,
                                                         std::ostream &os) {
  if (active_mlx_swizzle_panel_ <= 0) {
    return false;
  }

  std::vector<Expr> terms;
  std::function<void(const Expr &)> flatten_add = [&](const Expr &current) {
    if (const auto *add = current.as<prim::AddObj>()) {
      flatten_add(add->a);
      flatten_add(add->b);
    } else {
      terms.push_back(current);
    }
  };
  flatten_add(expr);
  if (terms.size() < 2) {
    return false;
  }

  enum class TermKind { kOther, kPanelRow, kRemainder };
  struct Term {
    TermKind kind{TermKind::kOther};
    int64_t coeff{1};
    Expr expr{IntImm::Int32(0)};
    bool used{false};
  };

  auto strip_cast = [](Expr current) {
    while (const auto *cast = current.as<prim::CastObj>()) {
      current = cast->value;
    }
    return current;
  };

  auto parse_const = [](const Expr &current, int64_t *value) {
    if (const auto *imm = current.as<IntImmObj>()) {
      *value = CodegenInt(imm);
      return true;
    }
    return false;
  };

  std::vector<Term> classified;
  classified.reserve(terms.size());
  bool saw_panel_row = false;
  for (const Expr &term_expr : terms) {
    Term term;
    term.expr = term_expr;
    Expr base = strip_cast(term_expr);
    int64_t coeff = 1;
    if (const auto *mul = base.as<prim::MulObj>()) {
      int64_t lhs_const = 0;
      int64_t rhs_const = 0;
      if (parse_const(mul->a, &lhs_const)) {
        coeff = lhs_const;
        base = strip_cast(mul->b);
      } else if (parse_const(mul->b, &rhs_const)) {
        coeff = rhs_const;
        base = strip_cast(mul->a);
      }
    }

    if (IsMlxPanelRowExpr(base)) {
      term.kind = TermKind::kPanelRow;
      term.coeff = coeff;
      saw_panel_row = true;
    } else if (IsBlockIdxExpr(base, 1) &&
               coeff % active_mlx_swizzle_panel_ == 0) {
      term.kind = TermKind::kPanelRow;
      term.coeff = coeff / active_mlx_swizzle_panel_;
      saw_panel_row = true;
    } else if (IsMlxPanelRemainderExpr(base)) {
      term.kind = TermKind::kRemainder;
      term.coeff = coeff;
    }
    classified.push_back(std::move(term));
  }

  std::vector<std::string> output_terms;
  for (size_t i = 0; i < classified.size(); ++i) {
    Term &row = classified[i];
    if (row.used || row.kind != TermKind::kPanelRow) {
      continue;
    }
    for (size_t j = 0; j < classified.size(); ++j) {
      Term &rem = classified[j];
      if (rem.used || rem.kind != TermKind::kRemainder ||
          rem.coeff != row.coeff) {
        continue;
      }
      std::ostringstream term_os;
      if (row.coeff == 1) {
        term_os << "blockIdx.y";
      } else {
        term_os << "(blockIdx.y * " << row.coeff << ")";
      }
      output_terms.push_back(term_os.str());
      row.used = true;
      rem.used = true;
      break;
    }
  }

  if (!saw_panel_row) {
    for (Term &term : classified) {
      if (term.used || term.kind != TermKind::kRemainder) {
        continue;
      }
      std::ostringstream term_os;
      if (term.coeff == 1) {
        term_os << "blockIdx.y";
      } else {
        term_os << "(blockIdx.y * " << term.coeff << ")";
      }
      output_terms.push_back(term_os.str());
      term.used = true;
    }
  }

  bool combined_mlx_y = !output_terms.empty();
  if (!combined_mlx_y) {
    return false;
  }
  for (const Term &term : classified) {
    if (term.used) {
      continue;
    }
    output_terms.push_back(PrintExpr(term.expr));
  }
  os << "(";
  for (size_t i = 0; i < output_terms.size(); ++i) {
    if (i != 0) {
      os << " + ";
    }
    os << output_terms[i];
  }
  os << ")";
  return true;
}

bool CodeGenTileLangMetal::TryPrintMlxSwizzleExpr(const Expr &expr,
                                                  std::ostream &os) {
  if (TryPrintMlxLogicalYAffineExpr(expr, os)) {
    return true;
  }
  if (IsMlxLogicalBlockXExpr(expr)) {
    os << "blockIdx.x";
    return true;
  }
  if (IsMlxLogicalBlockYExpr(expr)) {
    os << "blockIdx.y";
    return true;
  }
  return false;
}

void CodeGenTileLangMetal::Dispatch_(const CallObj *op, std::ostream &os) {
  static const Op metal_make_filled_simdgroup_matrix_op =
      Op::Get("tirx.metal.make_filled_simdgroup_matrix");
  static const Op metal_simdgroup_load_op =
      Op::Get("tirx.metal.simdgroup_load");
  static const Op metal_simdgroup_multiply_accumulate_op =
      Op::Get("tirx.metal.simdgroup_multiply_accumulate");
  static const Op metal_simdgroup_store_op =
      Op::Get("tirx.metal.simdgroup_store");
  const auto args = op->args; // NOLINT(*)
  TVM_FFI_ICHECK(!op->op.as<GlobalVarObj>())
      << "CodegenMetal does not support inter-function calls, "
      << "but expression " << ffi::GetRef<Call>(op) << " calls PrimFunc "
      << op->op;
  if (TryPrintMlxSwizzleExpr(ffi::GetRef<Expr>(op), os)) {
    return;
  }
  auto f_check_simdgroup_shape = [](const Expr &col, const Expr &row) {
    TVM_FFI_ICHECK(col->IsInstance<IntImmObj>() && row->IsInstance<IntImmObj>())
        << "Only constant shape is supported for simdgroup matrix, but got "
        << col << "x" << row;
    int col_val = CodegenInt(col);
    int row_val = CodegenInt(row);
    TVM_FFI_ICHECK(col_val == 8 && row_val == 8)
        << "Only 8x8 matrix is supported, but got " << col_val << "x"
        << row_val;
  };
  if (op->op.same_as(metal_make_filled_simdgroup_matrix_op)) {
    TVM_FFI_ICHECK_EQ(args.size(), 5);
    Var var = MetalMatrixStorageVar(args[0]);
    // Get the data type of the simdgroup matrix
    auto it = simdgroup_dtype_.find(var);
    TVM_FFI_ICHECK(it != simdgroup_dtype_.end())
        << "Cannot find variable allocation for simdgroup: " << var;
    const std::string &dtype_str = it->second;
    f_check_simdgroup_shape(args[3], args[4]);
    os << PrintExpr(var) << "[" << PrintExpr(args[1])
       << "] = make_filled_simdgroup_matrix<" << dtype_str << ", "
       << PrintExpr(args[3]) << ", " << PrintExpr(args[4]) << ">("
       << PrintExpr(args[2]) << ")";
  } else if (op->op.same_as(metal_simdgroup_load_op)) {
    TVM_FFI_ICHECK_EQ(args.size(), 7);
    f_check_simdgroup_shape(args[4], args[5]);
    os << "simdgroup_load(" << PrintExpr(args[0]) << "[" << PrintExpr(args[1])
       << "], " << PrintExpr(args[2]) << ", " << PrintExpr(args[3]) << ", 0, "
       << PrintExpr(args[6]) << ")";
  } else if (op->op.same_as(metal_simdgroup_store_op)) {
    TVM_FFI_ICHECK_EQ(args.size(), 7);
    f_check_simdgroup_shape(args[4], args[5]);
    os << "simdgroup_store(" << PrintExpr(args[0]) << "[" << PrintExpr(args[1])
       << "], " << PrintExpr(args[2]) << ", " << PrintExpr(args[3]) << ", 0, "
       << PrintExpr(args[6]) << ")";
  } else if (op->op.same_as(metal_simdgroup_multiply_accumulate_op)) {
    TVM_FFI_ICHECK_EQ(args.size(), 8);
    os << "simdgroup_multiply_accumulate("                         //
       << PrintExpr(args[0]) << "[" << PrintExpr(args[1]) << "], " //
       << PrintExpr(args[2]) << "[" << PrintExpr(args[3]) << "], " //
       << PrintExpr(args[4]) << "[" << PrintExpr(args[5]) << "], " //
       << PrintExpr(args[6]) << "[" << PrintExpr(args[7]) << "])";
  } else if (op->op.same_as(tl::cooperative_tensor_fill())) {
    TVM_FFI_ICHECK_EQ(args.size(), 5);
    std::string var = PrintExpr(args[0]);
    std::string idx = PrintExpr(args[1]);
    std::string val = PrintExpr(args[2]);
    int rows = CodegenInt(args[3]);
    int cols = CodegenInt(args[4]);
    int elems_per_tile = rows * cols / 32;
    Var fill_v = MetalMatrixStorageVar(args[0]);
    EnsureCooperativeTensorBuffer(fill_v);
    bool is_inlined = ct_c_inlined_.count(fill_v) > 0;
    bool storage_elided = ct_c_storage_elided_.count(fill_v) > 0;
    auto *fill_idx_imm = args[1].as<IntImmObj>();
    if (is_inlined && fill_idx_imm) {
      os << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < "
         << elems_per_tile << "; __i++) "
         << "__pct_c" << fill_idx_imm->value << "[__i] = " << val;
    } else {
      TVM_FFI_ICHECK(!storage_elided)
          << "Elided cooperative tensor C storage requires constant fill index";
      os << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < "
         << elems_per_tile << "; __i++) " << var << "[" << idx << " * "
         << elems_per_tile << " + __i] = " << val;
    }
  } else if (op->op.same_as(tl::cooperative_tensor_load())) {
    uses_cooperative_tensor_ = true;
    TVM_FFI_ICHECK_GE(args.size(), 11);
    std::string var = PrintExpr(args[0]);
    std::string idx = PrintExpr(args[1]);
    std::string src_ptr = PrintExpr(args[2]);
    std::string stride = PrintExpr(args[3]);
    int rows = CodegenInt(args[4]);
    int cols = CodegenInt(args[5]);
    Var v = MetalMatrixStorageVar(args[0]);
    EnsureCooperativeTensorBuffer(v);
    auto it = cooperative_tensor_dtype_.find(v);
    TVM_FFI_ICHECK(it != cooperative_tensor_dtype_.end());
    std::string dtype = it->second;
    std::string addr_space = GetAddrSpaceOf(args[2]);
    std::string src_dtype = GetPointeeTypeOf(args[2], dtype);
    int frag_rows = 16, frag_cols = 16;
    int nfrag_r = rows / frag_rows;
    int nfrag_c = cols / frag_cols;
    int total_elems = nfrag_r * nfrag_c * 8;
    bool is_inlined = ct_c_inlined_.count(v) > 0;
    bool storage_elided = ct_c_storage_elided_.count(v) > 0;
    auto *load_idx_imm = args[1].as<IntImmObj>();
    std::string direct_c_load_name;
    if (storage_elided) {
      TVM_FFI_ICHECK(is_inlined && load_idx_imm)
          << "Elided cooperative tensor C storage requires constant load index";
      direct_c_load_name =
          std::string("__pct_c") + std::to_string(CodegenInt(load_idx_imm));
    }
    std::string src_addr_space = "const " + addr_space;
    os << "{ " << src_addr_space << " " << src_dtype << "* __src = ("
       << src_addr_space << " " << src_dtype << "*)" << src_ptr << "; ";
    int elem_offset = 0;
    for (int fr = 0; fr < nfrag_r; fr++) {
      for (int fc = 0; fc < nfrag_c; fc++) {
        int row_off = fr * frag_rows;
        int col_off = fc * frag_cols;
        os << "{ "
           << "ushort __r0 = __base_row + " << row_off << "; "
           << "ushort __r1 = __r0 + 8; "
           << "ushort __c0 = __base_col + " << col_off << "; "
           << "*(thread " << dtype << "4*)(&";
        if (!direct_c_load_name.empty()) {
          os << direct_c_load_name << "[" << elem_offset << "]";
        } else {
          os << var << "[" << idx << " * " << (nfrag_r * nfrag_c * 8) << " + "
             << elem_offset << "]";
        }
        os << ") = ";
        std::string load0 = std::string("*(") + src_addr_space + " " +
                            src_dtype + "4*)(&__src[__r0 * " + stride +
                            " + __c0])";
        if (src_dtype == dtype) {
          os << load0;
        } else {
          os << dtype << "4(" << load0 << ")";
        }
        os << "; "
           << "*(thread " << dtype << "4*)(&";
        if (!direct_c_load_name.empty()) {
          os << direct_c_load_name << "[" << (elem_offset + 4) << "]";
        } else {
          os << var << "[" << idx << " * " << (nfrag_r * nfrag_c * 8) << " + "
             << (elem_offset + 4) << "]";
        }
        os << ") = ";
        std::string load1 = std::string("*(") + src_addr_space + " " +
                            src_dtype + "4*)(&__src[__r1 * " + stride +
                            " + __c0])";
        if (src_dtype == dtype) {
          os << load1;
        } else {
          os << dtype << "4(" << load1 << ")";
        }
        os << "; } ";
        elem_offset += 8;
      }
    }
    os << "}";
    if (is_inlined && load_idx_imm && direct_c_load_name.empty()) {
      int mma_tiles_per_load = total_elems / 16;
      int base_pct = CodegenInt(load_idx_imm) * mma_tiles_per_load;
      for (int t = 0; t < mma_tiles_per_load; t++) {
        os << "; TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < 16; __i++) "
              "__pct_c"
           << (base_pct + t) << "[__i] = " << var << "[" << idx << " * "
           << total_elems << " + " << (t * 16) << " + __i];";
      }
    }
  } else if (op->op.same_as(tl::cooperative_tensor_store())) {
    uses_cooperative_tensor_ = true;
    TVM_FFI_ICHECK_GE(args.size(), 11);
    std::string var = PrintExpr(args[0]);
    std::string idx = PrintExpr(args[1]);
    std::string dst_ptr = PrintExpr(args[2]);
    std::string stride = PrintExpr(args[3]);
    int rows = CodegenInt(args[4]);
    int cols = CodegenInt(args[5]);
    Var v = MetalMatrixStorageVar(args[0]);
    EnsureCooperativeTensorBuffer(v);
    auto it = cooperative_tensor_dtype_.find(v);
    TVM_FFI_ICHECK(it != cooperative_tensor_dtype_.end());
    std::string dtype = it->second;
    std::string addr_space = GetAddrSpaceOf(args[2]);
    std::string dst_dtype = GetPointeeTypeOf(args[2], dtype);
    int frag_rows = 16, frag_cols = 16;
    int nfrag_r = rows / frag_rows;
    int nfrag_c = cols / frag_cols;
    int total_elems = nfrag_r * nfrag_c * 8;
    bool is_inlined = ct_c_inlined_.count(v) > 0;
    bool storage_elided = ct_c_storage_elided_.count(v) > 0;
    auto *store_idx_imm = args[1].as<IntImmObj>();
    TVM_FFI_ICHECK(!(storage_elided && !(is_inlined && store_idx_imm)))
        << "Elided cooperative tensor C storage requires constant store index";
    os << "{ " << addr_space << " " << dst_dtype << "* __dst = (" << addr_space
       << " " << dst_dtype << "*)" << dst_ptr << "; ";
    int elem_offset = 0;
    for (int fr = 0; fr < nfrag_r; fr++) {
      for (int fc = 0; fc < nfrag_c; fc++) {
        int row_off = fr * frag_rows;
        int col_off = fc * frag_cols;
        auto emit_store_value = [&](const std::string &value) {
          if (dst_dtype == dtype) {
            os << value;
          } else {
            os << dst_dtype << "4(" << value << ")";
          }
        };
        os << "{ "
           << "ushort __r0 = __base_row + " << row_off << "; "
           << "ushort __r1 = __r0 + 8; "
           << "ushort __c0 = __base_col + " << col_off << "; "
           << "*(" << addr_space << " " << dst_dtype << "4*)(&__dst[__r0 * "
           << stride << " + __c0]) = ";
        if (is_inlined && store_idx_imm) {
          int pct_idx =
              CodegenInt(store_idx_imm) * (total_elems / 16) + elem_offset / 16;
          int pct_elem = elem_offset % 16;
          std::string value0 = std::string("*(thread ") + dtype +
                               "4*)(&__pct_c" + std::to_string(pct_idx) + "[" +
                               std::to_string(pct_elem) + "])";
          std::string value1 = std::string("*(thread ") + dtype +
                               "4*)(&__pct_c" + std::to_string(pct_idx) + "[" +
                               std::to_string(pct_elem + 4) + "])";
          emit_store_value(value0);
          os << "; "
             << "*(" << addr_space << " " << dst_dtype << "4*)(&__dst[__r1 * "
             << stride << " + __c0]) = ";
          emit_store_value(value1);
          os << "; } ";
        } else {
          std::string value0 = std::string("*(thread ") + dtype + "4*)(&" +
                               var + "[" + idx + " * " +
                               std::to_string(total_elems) + " + " +
                               std::to_string(elem_offset) + "])";
          std::string value1 = std::string("*(thread ") + dtype + "4*)(&" +
                               var + "[" + idx + " * " +
                               std::to_string(total_elems) + " + " +
                               std::to_string(elem_offset + 4) + "])";
          emit_store_value(value0);
          os << "; "
             << "*(" << addr_space << " " << dst_dtype << "4*)(&__dst[__r1 * "
             << stride << " + __c0]) = ";
          emit_store_value(value1);
          os << "; } ";
        }
        elem_offset += 8;
      }
    }
    os << "}";
  } else if (op->op.same_as(tl::cooperative_tensor_multiply_accumulate())) {
    uses_cooperative_tensor_ = true;
    TVM_FFI_ICHECK_GE(args.size(), 13);
    int M = CodegenInt(args[8]);
    int N = CodegenInt(args[9]);
    int K = CodegenInt(args[10]);
    bool trans_a = CodegenInt(args[11]) != 0;
    bool trans_b = CodegenInt(args[12]) != 0;

    std::string a_var = PrintExpr(args[2]);
    std::string a_idx = PrintExpr(args[3]);
    std::string b_var = PrintExpr(args[4]);
    std::string b_idx = PrintExpr(args[5]);
    std::string c_var = PrintExpr(args[0]);
    std::string c_idx = PrintExpr(args[1]);

    Var a_v = MetalMatrixStorageVar(args[2]);
    Var b_v = MetalMatrixStorageVar(args[4]);
    Var c_v = MetalMatrixStorageVar(args[0]);
    EnsureCooperativeTensorBuffer(a_v);
    EnsureCooperativeTensorBuffer(b_v);
    EnsureCooperativeTensorBuffer(c_v);
    auto a_it = cooperative_tensor_dtype_.find(a_v);
    auto b_it = cooperative_tensor_dtype_.find(b_v);
    auto c_it = cooperative_tensor_dtype_.find(c_v);
    TVM_FFI_ICHECK(a_it != cooperative_tensor_dtype_.end());
    TVM_FFI_ICHECK(b_it != cooperative_tensor_dtype_.end());
    TVM_FFI_ICHECK(c_it != cooperative_tensor_dtype_.end());
    std::string a_dtype = a_it->second;
    std::string b_dtype = b_it->second;
    std::string c_dtype = c_it->second;

    int a_elems = M * K / 32;
    int b_elems = K * N / 32;
    int c_elems = M * N / 32;

    TVM_FFI_ICHECK(M == 32 || N == 32 || K == 32)
        << "MPP matmul2d requires at least one of M, N, K to be 32, got " << M
        << "x" << N << "x" << K;

    bool c_inlined = ct_c_inlined_.count(c_v) > 0;
    bool c_storage_elided = ct_c_storage_elided_.count(c_v) > 0;
    auto *c_idx_imm = args[1].as<IntImmObj>();
    bool c_idx_const = c_inlined && c_idx_imm != nullptr;
    TVM_FFI_ICHECK(!(c_storage_elided && !c_idx_const))
        << "Elided cooperative tensor C storage requires constant MMA index";
    bool can_reuse_pct_op =
        c_idx_const && M == 16 && N == 32 && K == 16 && !trans_a && !trans_b;
    if (c_idx_const) {
      os << "{ ";
      if (can_reuse_pct_op) {
        os << "auto __ct_a = __pct_op.get_left_input_cooperative_tensor<"
           << a_dtype << ", " << b_dtype << ", " << c_dtype << ">(); "
           << "auto __ct_b = __pct_op.get_right_input_cooperative_tensor<"
           << a_dtype << ", " << b_dtype << ", " << c_dtype << ">(); ";
      } else {
        os << "constexpr auto __desc = mpp::tensor_ops::matmul2d_descriptor("
           << M << ", " << N << ", " << K << ", "
           << (trans_a ? "true" : "false") << ", "
           << (trans_b ? "true" : "false") << ", true, "
           << "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate)"
              "; "
           << "mpp::tensor_ops::matmul2d<__desc, metal::execution_simdgroup> "
              "__op; "
           << "auto __ct_a = __op.get_left_input_cooperative_tensor<" << a_dtype
           << ", " << b_dtype << ", " << c_dtype << ">(); "
           << "auto __ct_b = __op.get_right_input_cooperative_tensor<"
           << a_dtype << ", " << b_dtype << ", " << c_dtype << ">(); ";
      }
      os << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << a_elems
         << "; __i++) "
         << "__ct_a[__i] = " << a_var << "[" << a_idx << " * " << a_elems
         << " + __i]; "
         << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << b_elems
         << "; __i++) "
         << "__ct_b[__i] = " << b_var << "[" << b_idx << " * " << b_elems
         << " + __i]; " << (can_reuse_pct_op ? "__pct_op" : "__op")
         << ".run(__ct_a, __ct_b, __pct_c" << c_idx_imm->value << "); }";
    } else {
      os << "{ "
         << "constexpr auto __desc = mpp::tensor_ops::matmul2d_descriptor(" << M
         << ", " << N << ", " << K << ", " << (trans_a ? "true" : "false")
         << ", " << (trans_b ? "true" : "false") << ", true, "
         << "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate)"
            "; "
         << "mpp::tensor_ops::matmul2d<__desc, metal::execution_simdgroup> "
            "__op; "
         << "auto __ct_a = __op.get_left_input_cooperative_tensor<" << a_dtype
         << ", " << b_dtype << ", " << c_dtype << ">(); "
         << "auto __ct_b = __op.get_right_input_cooperative_tensor<" << a_dtype
         << ", " << b_dtype << ", " << c_dtype << ">(); "
         << "auto __ct_c = __op.get_destination_cooperative_tensor<"
         << "decltype(__ct_a), decltype(__ct_b), " << c_dtype << ">(); "
         << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << a_elems
         << "; __i++) "
         << "__ct_a[__i] = " << a_var << "[" << a_idx << " * " << a_elems
         << " + __i]; "
         << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << b_elems
         << "; __i++) "
         << "__ct_b[__i] = " << b_var << "[" << b_idx << " * " << b_elems
         << " + __i]; "
         << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << c_elems
         << "; __i++) "
         << "__ct_c[__i] = " << c_var << "[" << c_idx << " * " << c_elems
         << " + __i]; "
         << "__op.run(__ct_a, __ct_b, __ct_c); "
         << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << c_elems
         << "; __i++) " << c_var << "[" << c_idx << " * " << c_elems
         << " + __i] = __ct_c[__i]; }";
    }
  } else if (op->op.same_as(tl::atomic_add_elem_op()) ||
             op->op.same_as(tl::atomic_add_ret_elem_op())) {
    TVM_FFI_ICHECK_GE(args.size(), 2U);
    PrimType value_dtype = args[1]->ty.as_or_throw<PrimType>();
    TVM_FFI_ICHECK(
        value_dtype.IsScalar() && value_dtype.bits() == 32 &&
        (value_dtype.MatchesCode(kDLInt) || value_dtype.MatchesCode(kDLUInt)))
        << "Metal scalar atomic add supports int32 and uint32, got "
        << value_dtype;
    const char *atomic_type =
        value_dtype.MatchesCode(kDLInt) ? "atomic_int" : "atomic_uint";
    const std::string address_space = GetAddrSpaceOf(args[0]);
    TVM_FFI_ICHECK(address_space == "device" || address_space == "threadgroup")
        << "Metal scalar atomic add requires device or threadgroup storage, "
           "got "
        << address_space;
    os << "atomic_fetch_add_explicit(reinterpret_cast<" << address_space << " "
       << atomic_type << " *>(";
    this->PrintExpr(args[0], os);
    os << "), ";
    this->PrintExpr(args[1], os);
    os << ", memory_order_relaxed)";
  } else if (op->op.same_as(tirx::reinterpret_op())) {
    const auto *pointer = op->ty.as<PtrTypeObj>();
    const auto *source_pointer = args[0]->ty.as<PtrTypeObj>();
    if (pointer || source_pointer) {
      if (pointer && source_pointer) {
        TVM_FFI_CHECK_EQ(source_pointer->scope.name(), pointer->scope.name(),
                         TypeError)
            << "Metal pointer reinterpret cannot change address space";
      } else {
        const auto integer = (pointer ? args[0]->ty : op->ty).as<PrimType>();
        TVM_FFI_CHECK(integer && integer.value().IsScalar() &&
                          integer.value().MatchesCode(kDLInt, kDLUInt) &&
                          integer.value().bits() == 64,
                      TypeError)
            << "Metal pointer reinterpret requires a 64-bit scalar integer";
      }
      // MSL as_type explicitly excludes pointers. Use the C++ pointer cast,
      // taking the address space from the native destination PtrType when the
      // source is an integer address carrier.
      os << "reinterpret_cast<";
      if (pointer) {
        PrintStorageScope(pointer->scope, os);
        PrintType(pointer->element_type, os);
        os << "*";
      } else {
        PrintType(op->ty, os);
      }
      os << ">(";
      PrintExpr(args[0], os);
      os << ")";
      return;
    }
    // generate as_type<TYPE>(ARG)
    os << "(as_type<";
    this->PrintType(op->ty.as_or_throw<PrimType>(), os);
    os << ">(";
    this->PrintExpr(args[0], os);
    os << "))";
  } else if (op->op.same_as(tirx::handle_add_byte_offset_op())) {
    TVM_FFI_ICHECK_EQ(args.size(), 2U);
    std::string addr_space = GetAddrSpaceOf(args[0]);
    os << "((" << addr_space << ' ';
    PrintType(op->ty.as_or_throw<PtrType>()->element_type, os);
    os << "*)((" << addr_space << " char*)";
    this->PrintExpr(args[0], os);
    os << " + ";
    this->PrintExpr(args[1], os);
    os << "))";
  } else {
    CodeGenC::Dispatch_(op, os);
  }
}

void CodeGenTileLangMetal::Dispatch_(const FloatImmObj *op,
                                     std::ostream &os) { // NOLINT(*)
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  std::ostringstream temp;
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      temp << "-";
    }
    temp << "INFINITY";
  } else if (std::isnan(op->value)) {
    temp << "NAN";
  } else {
    temp << std::scientific << op->value;
    if (dtype.bits() == 32)
      temp << 'f';
    else if (dtype.bits() == 16)
      temp << 'h';
  }
  MarkConst(temp.str());
  os << temp.str();
}

ffi::Module BuildTileLangMetal(IRModule mod, Target target) {
  bool output_ssa = false;
  mod = tirx::transform::PointerValueTypeRewrite()(std::move(mod));

  std::ostringstream source_maker;
  ffi::Map<ffi::String, ffi::Bytes> smap;
  const auto fmetal_postproc =
      tvm::ffi::Function::GetGlobal("tilelang_callback_metal_postproc");
  std::string fmt = "metal";

  for (auto kv : mod->functions) {
    TVM_FFI_ICHECK(kv.second->IsInstance<tirx::FunctionObj>())
        << "CodeGenTileLangMetal: Can only take PrimFunc";
    auto global_symbol =
        kv.second->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    TVM_FFI_ICHECK(global_symbol.has_value());
    std::string func_name = global_symbol.value();

    source_maker << "// Function: " << func_name << "\n";
    CodeGenTileLangMetal cg(target);
    cg.Init(output_ssa);
    cg.SetEmitLineDirectives(TileLangEmitLineDirectivesEnabled());
    auto f = kv.second.as_or_throw<tirx::Function>();
    auto calling_conv =
        f->GetAttr(tvm::attr::kCallingConv).value_or(CallingConv::kDefault);
    TVM_FFI_ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenTileLangMetal: expect calling_conv equals "
           "CallingConv::kDeviceKernelLaunch";

    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    source_maker << fsource << "\n";
    if (fmetal_postproc) {
      fsource = (*fmetal_postproc)(fsource, target).cast<std::string>();
    }
    smap.Set(func_name, ffi::Bytes(std::move(fsource)));
  }

  ffi::Map<ffi::String, ffi::String> source;
  source.Set("metal", source_maker.str());
  return tvm::target::MetalModuleCreateWithFallback(
      std::move(smap), ffi::String(fmt), ExtractFuncInfo(mod),
      std::move(source));
}

ffi::Module BuildTileLangMetalWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  mod = tirx::transform::PointerValueTypeRewrite()(std::move(mod));

  std::ostringstream source_maker;
  ffi::Map<ffi::String, ffi::Bytes> smap;

  for (auto kv : mod->functions) {
    TVM_FFI_ICHECK(kv.second->IsInstance<tirx::FunctionObj>())
        << "CodeGenTileLangMetal: Can only take PrimFunc";
    auto global_symbol =
        kv.second->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    TVM_FFI_ICHECK(global_symbol.has_value());
    std::string func_name = global_symbol.value();

    source_maker << "// Function: " << func_name << "\n";
    CodeGenTileLangMetal cg(target);
    cg.Init(output_ssa);
    cg.SetEmitLineDirectives(TileLangEmitLineDirectivesEnabled());
    auto f = kv.second.as_or_throw<tirx::Function>();
    auto calling_conv =
        f->GetAttr(tvm::attr::kCallingConv).value_or(CallingConv::kDefault);
    TVM_FFI_ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenTileLangMetal: expect calling_conv equals "
           "CallingConv::kDeviceKernelLaunch";

    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    source_maker << fsource << "\n";
    smap.Set(func_name, ffi::Bytes(std::move(fsource)));
  }

  ffi::Map<ffi::String, ffi::String> source;
  source.Set("metal", source_maker.str());
  return tvm::target::MetalFallbackModuleCreate(
      std::move(smap), ffi::String("metal"), ExtractFuncInfo(mod),
      std::move(source));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_metal", BuildTileLangMetal)
      .def("target.build.tilelang_metal_without_compile",
           BuildTileLangMetalWithoutCompile);
}
} // namespace codegen
} // namespace tvm
