// PPU: native HGGC runtime module — uses hg* driver APIs and hggc* runtime APIs
// directly.
#include "codegen_ppu.h"
#include "config.h"
#include "runtime/pack_args.h"
#include "runtime/thread_storage_scope.h"
#include "support/bytes_io.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/ir/transform.h>

#include <hggc.h>
#include <hggc_runtime_api.h>

#include <array>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace tvm {
namespace codegen {

using namespace ffi;
using ppu::CodeGenTileLangPPU;

// ---------------------------------------------------------------------------
// HGGC driver-API error-checking macro
// ---------------------------------------------------------------------------
#define HGGC_DRIVER_CALL(x)                                                    \
  {                                                                            \
    HGresult result = x;                                                       \
    if (result != HGGC_SUCCESS && result != HGGC_ERROR_DEINITIALIZED) {        \
      const char *msg;                                                         \
      hgGetErrorName(result, &msg);                                            \
      TVM_FFI_THROW(InternalError)                                             \
          << "" #x " failed with error: " << (msg ? msg : "unknown");          \
    }                                                                          \
  }

// HGGC runtime-API error-checking macro
#define HGGC_RT_CALL(func)                                                     \
  {                                                                            \
    hggcError_t e = (func);                                                    \
    TVM_FFI_ICHECK(e == hggcSuccess || e == hggcErrorHggcrtUnloading)          \
        << "HGGC: " << hggcGetErrorString(e);                                  \
  }

// ---------------------------------------------------------------------------
// PPUModuleNode — runtime module using native HGGC driver APIs
// ---------------------------------------------------------------------------
namespace {

static constexpr const int kMaxNumPPUs = 32;

inline void EnsureCurrentPPUContext(int device_id) {
  ICHECK_GE(device_id, 0) << "Invalid device_id: " << device_id;
  ICHECK_LT(device_id, kMaxNumPPUs)
      << "device_id " << device_id << " exceeds maximum " << kMaxNumPPUs;
  // hggcSetDevice implicitly initializes the driver.
  // Retain primary context per-device; reuse for subsequent driver API calls.
  HGGC_RT_CALL(hggcSetDevice(device_id));
  static std::array<HGcontext, kMaxNumPPUs> primary_ctxs = {};
  static std::array<std::once_flag, kMaxNumPPUs> init_flags = {};
  std::call_once(init_flags[device_id], [&]() {
    HGGC_DRIVER_CALL(
        hgDevicePrimaryCtxRetain(&primary_ctxs[device_id], device_id));
  });
  HGGC_DRIVER_CALL(hgCtxSetCurrent(primary_ctxs[device_id]));
}

class PPUModuleNode : public ffi::ModuleObj {
public:
  PPUModuleNode(ffi::Bytes code, ffi::String fmt,
                ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                ffi::Map<ffi::String, ffi::String> source)
      : code_(code), fmt_(fmt), fmap_(fmap), source_(source) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }

  ~PPUModuleNode() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        hggcError_t set_err = hggcSetDevice(static_cast<int>(i));
        if (set_err != hggcSuccess && set_err != hggcErrorHggcrtUnloading) {
          continue;
        }
        HGresult result = hgModuleUnload(module_[i]);
        (void)result;
      }
    }
  }

  const char *kind() const final { return "ppu"; }

  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String &name) final;

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(code_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String &format) const final {
    if (format == fmt_) {
      return ffi::String(code_.data(), code_.size());
    }
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    if (format.empty()) {
      if (auto it = source_.find("ppu"); it != source_.end()) {
        return (*it).second;
      }
      if (fmt_ == "hgbin" || fmt_ == "ppu") {
        return ffi::String(code_.data(), code_.size());
      }
    }
    return ffi::String();
  }

  // Get a HGfunction from primary context in device_id (lazily loads module)
  HGfunction GetFunc(int device_id, const std::string &func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureCurrentPPUContext(device_id);
    if (module_[device_id] == nullptr) {
      HGGC_DRIVER_CALL(hgModuleLoadData(&(module_[device_id]), code_.data()));
    }
    HGfunction func;
    HGresult result =
        hgModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != HGGC_SUCCESS) {
      const char *msg;
      hgGetErrorName(result, &msg);
      TVM_FFI_THROW(InternalError) << "hgModuleGetFunction " << func_name
                                   << " failed with error: " << msg;
    }
    return func;
  }

private:
  ffi::Bytes code_;
  ffi::String fmt_;
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap_;
  ffi::Map<ffi::String, ffi::String> source_;
  std::array<HGmodule, kMaxNumPPUs> module_;
  std::mutex mutex_;
};

// ---------------------------------------------------------------------------
// PPUWrappedFunc — wrapped kernel launcher
// ---------------------------------------------------------------------------
class PPUWrappedFunc {
public:
  void Init(PPUModuleNode *m, ffi::ObjectPtr<ffi::Object> sptr,
            const std::string &func_name, size_t num_void_args,
            const ffi::Array<ffi::String> &launch_param_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    std::fill(dyn_smem_initialized_.begin(), dyn_smem_initialized_.end(),
              false);
    std::fill(cluster_attr_initialized_.begin(),
              cluster_attr_initialized_.end(), false);
    use_dyn_shared_memory_ = false;
    for (const auto &tag : launch_param_tags) {
      if (tag == runtime::launch_param::kUseDynamicSharedMemoryTag) {
        use_dyn_shared_memory_ = true;
        break;
      }
    }
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }

  void operator()(ffi::PackedArgs args, ffi::Any *rv, void **void_args) const {
    int device_id;
    HGGC_RT_CALL(hggcGetDevice(&device_id));
    ICHECK_LT(device_id, kMaxNumPPUs)
        << "device_id " << device_id << " exceeds maximum " << kMaxNumPPUs;
    EnsureCurrentPPUContext(device_id);
    runtime::ThreadWorkLoad wl = launch_param_config_.Extract(args);

    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }

    bool need_dyn_attr = use_dyn_shared_memory_ || (wl.dyn_shmem_size > 0);
    if (need_dyn_attr) {
      if (!dyn_smem_initialized_[device_id] ||
          dyn_smem_last_[device_id] != wl.dyn_shmem_size) {
        HGresult attr_set = hgFuncSetAttribute(
            fcache_[device_id], HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            static_cast<int>(wl.dyn_shmem_size));
        if (attr_set != HGGC_SUCCESS) {
          TVM_FFI_THROW(InternalError)
              << "Failed to set the allowed dynamic shared memory size to "
              << wl.dyn_shmem_size;
        }
        dyn_smem_last_[device_id] = wl.dyn_shmem_size;
        dyn_smem_initialized_[device_id] = true;
      }
    }

    HGstream strm = static_cast<HGstream>(
        TVMFFIEnvGetStream(static_cast<int>(DLDeviceType::kDLPPU), device_id));
    HGresult result;

    TVM_FFI_ICHECK(wl.grid_dim(0) > 0 && wl.grid_dim(1) > 0 &&
                   wl.grid_dim(2) > 0)
        << "PPULaunch Error: grid dimension must be positive, but got"
        << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << ","
        << wl.grid_dim(2) << ")"
        << " in kernel " << func_name_
        << ". A zero grid dimension is often caused by a dynamic shape"
        << " (e.g. num_tokens) being 0 at runtime.";

    if (wl.use_cluster_launch()) {
      HGlaunchConfig config{};
      HGlaunchAttribute attribute[2]{};
      attribute[0].id = HG_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attribute[0].value.clusterDim.x = wl.cluster_dim[0];
      attribute[0].value.clusterDim.y = wl.cluster_dim[1];
      attribute[0].value.clusterDim.z = wl.cluster_dim[2];
      attribute[1].id = HG_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute[1].value.programmaticStreamSerializationAllowed = 1;

      config.attrs = attribute;
      config.numAttrs = 2;
      config.hStream = strm;
      config.gridDimX = wl.grid_dim(0);
      config.gridDimY = wl.grid_dim(1);
      config.gridDimZ = wl.grid_dim(2);
      config.blockDimX = wl.block_dim(0);
      config.blockDimY = wl.block_dim(1);
      config.blockDimZ = wl.block_dim(2);
      config.sharedMemBytes = wl.dyn_shmem_size;

      if (!cluster_attr_initialized_[device_id]) {
        HGresult attr_result = hgFuncSetAttribute(
            fcache_[device_id],
            HG_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1);
        if (attr_result != HGGC_SUCCESS) {
          const char *msg;
          hgGetErrorName(attr_result, &msg);
          TVM_FFI_THROW(InternalError) << "Failed to set cluster attribute for "
                                       << func_name_ << ": " << msg;
        }
        cluster_attr_initialized_[device_id] = true;
      }

      result =
          hgLaunchKernelEx(&config, fcache_[device_id], void_args, nullptr);
    } else if (launch_param_config_.use_programtic_dependent_launch()) {
      HGlaunchConfig config{};
      HGlaunchAttribute attribute[1]{};
      attribute[0].id = HG_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute[0].value.programmaticStreamSerializationAllowed = 1;

      config.attrs = attribute;
      config.numAttrs = 1;
      config.hStream = strm;
      config.gridDimX = wl.grid_dim(0);
      config.gridDimY = wl.grid_dim(1);
      config.gridDimZ = wl.grid_dim(2);
      config.blockDimX = wl.block_dim(0);
      config.blockDimY = wl.block_dim(1);
      config.blockDimZ = wl.block_dim(2);
      config.sharedMemBytes = wl.dyn_shmem_size;

      result =
          hgLaunchKernelEx(&config, fcache_[device_id], void_args, nullptr);
    } else if (launch_param_config_.use_cooperative_launch()) {
      result = hgLaunchCooperativeKernel(
          fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2),
          wl.block_dim(0), wl.block_dim(1), wl.block_dim(2), wl.dyn_shmem_size,
          strm, void_args);
    } else {
      result = hgLaunchKernel(fcache_[device_id], wl.grid_dim(0),
                              wl.grid_dim(1), wl.grid_dim(2), wl.block_dim(0),
                              wl.block_dim(1), wl.block_dim(2),
                              wl.dyn_shmem_size, strm, void_args, nullptr);
    }

    if (result != HGGC_SUCCESS && result != HGGC_ERROR_DEINITIALIZED) {
      const char *msg;
      hgGetErrorName(result, &msg);
      std::ostringstream os;
      os << "PPULaunch Error: " << (msg ? msg : "unknown") << "\n"
         << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << ","
         << wl.grid_dim(2) << "), "
         << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << ","
         << wl.block_dim(2) << ")"
         << " dyn_smem_bytes=" << wl.dyn_shmem_size << "\n";
      ffi::String ppu_src = m_->InspectSource("");
      if (ppu_src.length() != 0) {
        os << "// func_name=" << func_name_ << "\n"
           << "// PPU Source\n"
           << "// -----------\n"
           << ppu_src;
      }
      TVM_FFI_THROW(InternalError) << os.str();
    }

    // Check for asynchronous errors
    if (result == HGGC_SUCCESS) {
      hggcError_t last_err = hggcPeekAtLastError();
      if (last_err != hggcSuccess) {
        const char *err_name = hggcGetErrorName(last_err);
        const char *err_str = hggcGetErrorString(last_err);
        hggcGetLastError(); // Clear sticky error
        TVM_FFI_THROW(InternalError)
            << func_name_ << ": " << (err_name ? err_name : "unknown") << " - "
            << (err_str ? err_str : "unknown");
      }
    }
  }

private:
  PPUModuleNode *m_;
  ffi::ObjectPtr<ffi::Object> sptr_;
  std::string func_name_;
  mutable std::array<HGfunction, kMaxNumPPUs> fcache_;
  runtime::LaunchParamConfig launch_param_config_;
  bool use_dyn_shared_memory_{false};
  mutable std::array<size_t, kMaxNumPPUs> dyn_smem_last_;
  mutable std::array<bool, kMaxNumPPUs> dyn_smem_initialized_;
  mutable std::array<bool, kMaxNumPPUs> cluster_attr_initialized_;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// PPUModuleNode::GetFunction — defined after PPUWrappedFunc
// ---------------------------------------------------------------------------
ffi::Optional<ffi::Function>
PPUModuleNode::GetFunction(const ffi::String &name) {
  ffi::ObjectPtr<ffi::Object> sptr_to_self =
      ffi::GetObjectPtr<ffi::Object>(this);
  TVM_FFI_ICHECK_EQ(sptr_to_self.get(), this);
  auto opt_info = fmap_.Get(name);
  if (!opt_info.has_value())
    return ffi::Function();
  runtime::FunctionInfo info = opt_info.value();
  PPUWrappedFunc f;
  f.Init(this, sptr_to_self, name, info->arg_types.size(),
         info->launch_param_tags);
  return runtime::PackFuncVoidAddr(f, info->arg_types, info->arg_extra_tags);
}

// ---------------------------------------------------------------------------
// PPUModuleCreate — factory
// ---------------------------------------------------------------------------
static ffi::Module
PPUModuleCreateImpl(ffi::Bytes code, ffi::String fmt,
                    ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                    ffi::Map<ffi::String, ffi::String> source) {
  auto n = ffi::make_object<PPUModuleNode>(code, fmt, fmap, source);
  return ffi::Module(n);
}

// Public factory used by BuildTileLangPPU
ffi::Module PPUModuleCreate(ffi::Bytes code, ffi::String fmt,
                            ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                            ffi::Map<ffi::String, ffi::String> source) {
  return PPUModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                             std::move(source));
}

// Load from bytes (for save/load round-trip)
static ffi::Module PPUModuleLoadFromBytes(const ffi::Bytes &bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap;
  ffi::Bytes code;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&code);
  return PPUModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                             ffi::Map<ffi::String, ffi::String>());
}

// ---------------------------------------------------------------------------
// Helper functions (from original rt_mod_ppu.cc)
// ---------------------------------------------------------------------------
static std::string GetDeviceGlobalSymbol(const GlobalVar &gvar,
                                         const tirx::PrimFunc &f) {
  if (auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
    return static_cast<std::string>(global_symbol.value());
  }
  return gvar->name_hint;
}

static void ValidateUniqueDeviceGlobalSymbols(const IRModule &mod) {
  std::unordered_map<std::string, std::string> symbol_to_gvar;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tirx::PrimFunc>(kv.second);
    std::string global_symbol = GetDeviceGlobalSymbol(gvar, f);
    auto [it, inserted] =
        symbol_to_gvar.emplace(global_symbol, gvar->name_hint);
    ICHECK(inserted) << "Duplicate PPU kernel global_symbol `" << global_symbol
                     << "` found on PrimFuncs `" << it->second << "` and `"
                     << gvar->name_hint << "`.";
  }
}

static Map<String, runtime::FunctionInfo> ExtractFuncInfo(const IRModule &mod) {
  Map<String, runtime::FunctionInfo> fmap;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tirx::PrimFunc>(kv.second);

    Array<DLDataType> arg_types;
    Array<String> launch_param_tags;

    for (size_t i = 0; i < f->params.size(); ++i) {
      if (f->params[i]->dtype.is_handle()) {
        auto ptr = f->params[i]->type_annotation.as<PointerTypeNode>();
        if (ptr && ptr->storage_scope == "grid_constant") {
          arg_types.push_back(DataType(runtime::kDLGridConstant, 64, 1));
          continue;
        }
      }
      DataType dtype = f->params[i].dtype();
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      arg_types.push_back(dtype);
    }
    if (f->HasNonzeroAttr(tl::attr::kHasGridSync)) {
      launch_param_tags.push_back(
          runtime::launch_param::kUseProgramaticDependentLaunch);
    }
    if (f->HasNonzeroAttr("use_cooperative_groups")) {
      launch_param_tags.push_back(runtime::launch_param::kUseCooperativeLaunch);
    }
    if (f->GetAttr<Array<Integer>>("cluster_dims").defined()) {
      launch_param_tags.push_back(runtime::launch_param::kClusterDimX);
      launch_param_tags.push_back(runtime::launch_param::kClusterDimY);
      launch_param_tags.push_back(runtime::launch_param::kClusterDimZ);
    }
    if (auto opt = f->GetAttr<Array<String>>(tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        if (tag != runtime::launch_param::kClusterDimX &&
            tag != runtime::launch_param::kClusterDimY &&
            tag != runtime::launch_param::kClusterDimZ) {
          launch_param_tags.push_back(tag);
        }
      }
    }
    std::string sym = GetDeviceGlobalSymbol(Downcast<GlobalVar>(kv.first), f);
    fmap.Set(String(sym), runtime::FunctionInfo(String(sym), arg_types,
                                                launch_param_tags, {}));
  }
  return fmap;
}

// ---------------------------------------------------------------------------
// BuildTileLangPPU — compile + create PPU module
// ---------------------------------------------------------------------------
Module BuildTileLangPPU(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangPPU cg;
  cg.Init(output_ssa);
  cg.SetEmitLineDirectives(tl::tl_config::EmitLineDirectivesEnabled());

  ValidateUniqueDeviceGlobalSymbols(mod);
  if (const auto f = Function::GetGlobal("tilelang_callback_ppu_validate")) {
    (*f)(mod, target);
  }

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangPPU: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f = Function::GetGlobal("tilelang_callback_ppu_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  std::string fmt = "hgbin";
  std::string binary;
  if (const auto f = Function::GetGlobal("tilelang_callback_ppu_compile")) {
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    binary = (*f)(code, target, pass_ctx->config).cast<std::string>();
    fmt = "hgbin"; // always binary data, not file path
  } else {
    ICHECK(0) << "tilelang_callback_ppu_compile not registered";
  }

  Map<String, String> source_map;
  source_map.Set("ppu", code);

  return PPUModuleCreate(Bytes(binary.data(), binary.size()), String(fmt),
                         ExtractFuncInfo(mod), source_map);
}

// ---------------------------------------------------------------------------
// BuildTileLangPPUWithoutCompile — source-only, no binary
// ---------------------------------------------------------------------------
Module BuildTileLangPPUWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangPPU cg;
  cg.Init(output_ssa);
  cg.SetEmitLineDirectives(tl::tl_config::EmitLineDirectivesEnabled());

  ValidateUniqueDeviceGlobalSymbols(mod);
  if (const auto f = Function::GetGlobal("tilelang_callback_ppu_validate")) {
    (*f)(mod, target);
  }

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangPPU: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f = Function::GetGlobal("tilelang_callback_ppu_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  Map<String, String> source_map;
  source_map.Set("ppu", code);

  static constexpr const char kDummy[] = "hgbin";
  return PPUModuleCreate(Bytes(kDummy, sizeof(kDummy) - 1), String("hgbin"),
                         ExtractFuncInfo(mod), source_map);
}

// ---------------------------------------------------------------------------
// FFI registration
// ---------------------------------------------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_ppu", BuildTileLangPPU)
      .def("target.build.tilelang_ppu_without_compile",
           BuildTileLangPPUWithoutCompile)
      .def("ffi.Module.create.ppu",
           [](ffi::Bytes code, ffi::String fmt,
              ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
              ffi::Map<ffi::String, ffi::String> source) {
             return PPUModuleCreateImpl(std::move(code), std::move(fmt),
                                        std::move(fmap), std::move(source));
           })
      .def("ffi.Module.load_from_bytes.ppu", PPUModuleLoadFromBytes);
}

} // namespace codegen
} // namespace tvm
