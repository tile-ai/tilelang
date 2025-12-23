"""CuTeDSL Source Wrapper for TileLang.

This module provides C++ kernel launcher generation for the CuTeDSL backend.

Key features:
- Automatic C++ launcher generation with CUDA Driver API
- TMA descriptors on HOST memory, passed via __grid_constant__ (no device copy needed)
- cuLaunchKernel automatically copies 128-byte CUtensorMap to kernel param space
- Support for single and multiple kernel launches
- Complete cache system integration
"""

from __future__ import annotations
from typing import Any, ClassVar

from tvm import IRModule
from tvm.target import Target
from tvm.tir.stmt_functor import post_order_visit

from tilelang import tvm as tvm
from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper
from tilelang.jit.adapter.utils import (
    extract_python_func_declaration,
    pythonic_expr,
    parse_tma_descriptor_args,
)

# =============================================================================
# C++ LAUNCHER TEMPLATES (using named parameters for clarity)
# =============================================================================

# TMA single descriptor initialization template (writes to caller-provided host array)
# No device copy needed - cuLaunchKernel handles __grid_constant__ params automatically
CPP_TMA_DESC_INIT_TEMPLATE = """\
  // Descriptor {desc_idx}: {desc_name} (tensor: {tensor_name})
  {{
    uint64_t globalDim[{rank}] = {{{global_dim_values}}};
    uint64_t globalStrides[{stride_rank}] = {{{global_stride_values}}};
    uint32_t boxDim[{rank}] = {{{box_dim_values}}};
    uint32_t elemStrides[{rank}] = {{{elem_stride_values}}};

    result = cuTensorMapEncodeTiled(
        &tma_descs[{desc_idx}],
        static_cast<CUtensorMapDataType>({dtype}),
        {rank},
        reinterpret_cast<void*>({tensor_name}_ptr),
        globalDim,
        globalStrides,
        boxDim,
        elemStrides,
        static_cast<CUtensorMapInterleave>({interleave}),
        static_cast<CUtensorMapSwizzle>({swizzle}),
        static_cast<CUtensorMapL2promotion>({l2_promotion}),
        static_cast<CUtensorMapFloatOOBfill>({oob_fill})
    );

    if (result != CUDA_SUCCESS) {{
      std::cerr << "Failed to encode TMA descriptor {desc_idx}: " << result << "\\n";
      return result;
    }}
  }}
"""

# TMA single im2col descriptor initialization template (writes to caller-provided host array)
# Align field ordering with NVRTC wrapper (cuTensorMapEncodeIm2col signature).
CPP_TMA_IM2COL_DESC_INIT_TEMPLATE = """\
  // Descriptor {desc_idx}: {desc_name} (tensor: {tensor_name}) [im2col]
  {{
    uint64_t globalDim[{rank}] = {{{global_dim_values}}};
    uint64_t globalStrides[{stride_rank}] = {{{global_stride_values}}};
    uint32_t elemStrides[{rank}] = {{{elem_stride_values}}};
    int32_t lowerCorner[{rank_minus_two}] = {{{lower_corner_values}}};
    int32_t upperCorner[{rank_minus_two}] = {{{upper_corner_values}}};

    result = cuTensorMapEncodeIm2col(
        &tma_descs[{desc_idx}],
        static_cast<CUtensorMapDataType>({dtype}),
        {rank},
        reinterpret_cast<void*>({tensor_name}_ptr),
        globalDim,
        globalStrides,
        lowerCorner,
        upperCorner,
        static_cast<uint32_t>({channels_per_pixel}),
        static_cast<uint32_t>({pixels_per_column}),
        elemStrides,
        static_cast<CUtensorMapInterleave>({interleave}),
        static_cast<CUtensorMapSwizzle>({swizzle}),
        static_cast<CUtensorMapL2promotion>({l2_promotion}),
        static_cast<CUtensorMapFloatOOBfill>({oob_fill})
    );

    if (result != CUDA_SUCCESS) {{
      std::cerr << "Failed to encode TMA im2col descriptor {desc_idx}: " << result << "\\n";
      return result;
    }}
  }}
"""

# TMA initialization function template (writes to caller-provided host array)
# __grid_constant__ allows kernel to receive TMA descriptor by value via param space
CPP_TMA_INIT_FUNC_TEMPLATE = """\
CUresult tma_init(CUtensorMap* tma_descs, {func_args}) {{
  // Initialize {num_descs} TMA descriptor(s) in caller-provided host array
  // cuLaunchKernel will copy 128-byte CUtensorMap to kernel param space automatically
  CUresult result;

{desc_init_code}

  return CUDA_SUCCESS;
}}
"""

# Kernel initialization template
CPP_KERNEL_INIT_TEMPLATE = """\
  // Find and configure kernel {kernel_idx}: {kernel_name}
  result = find_kernel_by_pattern(module, "{kernel_name}", &kernels[{kernel_idx}]);
  if (result != CUDA_SUCCESS) {{
    std::cerr << "Failed to find kernel {kernel_name} on device " << device_id << ": " << result << "\\n";
    return result;
  }}

  if ({smem_size} > 0) {{
    result = cuFuncSetAttribute(kernels[{kernel_idx}],
                                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                {smem_size});
    if (result != CUDA_SUCCESS) {{
      std::cerr << "Failed to set smem for {kernel_name} on device " << device_id << ": " << result << "\\n";
      return result;
    }}
  }}
"""

# TMA launch initialization template (host memory mode - uses __grid_constant__)
# Kernel receives TMA descriptor by value: .param .align 128 .b8 xxx_param[128]
CPP_TMA_LAUNCH_INIT_TEMPLATE = """\
  // Declare stack-local TMA descriptor array (eliminates concurrency race)
  CUtensorMap tma_descs[{num_tma_descs}];

  // Initialize TMA descriptors (HOST memory - passed via __grid_constant__)
  // NOTE: We intentionally do NOT reuse/cached descriptors across launches.
  // Pointer-only reuse is a correctness trap (shape/stride may change with same ptr),
  // and correctness beats micro-optimizations.
  result = tma_init(tma_descs, {tma_tensor_args});
  if (result != CUDA_SUCCESS) {{
    std::cerr << "Failed to initialize TMA descriptors: " << result << "\\n";
    return result;
  }}
"""

# Kernel launch template
CPP_KERNEL_LAUNCH_TEMPLATE = """\
  // Launch kernel {kernel_idx}: {kernel_name}
  {{
    // Get the kernel for current device
    auto kernels_it = g_device_kernels.find(device_id);
    if (kernels_it == g_device_kernels.end()) {{
      std::cerr << "Kernels not initialized for device " << device_id << "\\n";
      return CUDA_ERROR_NOT_INITIALIZED;
    }}
    const std::vector<CUfunction>& kernels = kernels_it->second;

    void* args[] = {{{kernel_args}}};
    result = cuLaunchKernel(
        kernels[{kernel_idx}],
        {grid_x}, {grid_y}, {grid_z},
        {block_x}, {block_y}, {block_z},
        {smem_size},
        stream,
        args,
        nullptr
    );
    if (result != CUDA_SUCCESS) {{
      std::cerr << "Failed to launch kernel {kernel_name} on device " << device_id << ": " << result << "\\n";
      return result;
    }}
  }}
"""

# Complete C++ launcher template
CPP_LAUNCHER_TEMPLATE = """\
#include <cuda.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <mutex>
#include <unordered_map>

// TVM Headers
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

// Per-device module and kernel storage for multi-GPU support
// Each device needs its own CUmodule because modules are tied to CUDA contexts
static std::unordered_map<int, CUmodule> g_device_modules;
static std::unordered_map<int, std::vector<CUfunction>> g_device_kernels;
static std::unordered_map<int, CUcontext> g_device_contexts;  // Track retained contexts for cleanup
static std::mutex g_devices_mutex;

// Find kernel by pattern (substring match, prefer base name over _N variants)
CUresult find_kernel_by_pattern(CUmodule module, const char* pattern, CUfunction* out_func) {{
  CUresult result;
  unsigned int num_funcs = 0;

  result = cuModuleGetFunctionCount(&num_funcs, module);
  if (result != CUDA_SUCCESS) {{
    std::cerr << "Failed to get function count: " << result << "\\n";
    return result;
  }}

  std::vector<CUfunction> func_list(num_funcs);
  result = cuModuleEnumerateFunctions(func_list.data(), num_funcs, module);
  if (result != CUDA_SUCCESS) {{
    std::cerr << "Failed to enumerate functions: " << result << "\\n";
    return result;
  }}

  // Collect substring matches, separating base name from _N variants
  std::vector<std::pair<std::string, CUfunction>> base_matches;     // pattern not followed by _digit
  std::vector<std::pair<std::string, CUfunction>> variant_matches;  // pattern followed by _digit

  size_t pattern_len = std::strlen(pattern);

  for (unsigned int i = 0; i < num_funcs; i++) {{
    const char* func_name = nullptr;
    result = cuFuncGetName(&func_name, func_list[i]);
    if (result != CUDA_SUCCESS || func_name == nullptr) {{
      std::cerr << "Failed to get function name: " << result << "\\n";
      return result;
    }}

    std::string name_str(func_name);
    size_t pos = name_str.find(pattern);

    if (pos != std::string::npos) {{
      // Found substring match
      size_t after_pattern = pos + pattern_len;

      // Check what follows the pattern
      if (after_pattern < name_str.length() &&
          name_str[after_pattern] == '_' &&
          after_pattern + 1 < name_str.length() &&
          std::isdigit(name_str[after_pattern + 1])) {{
        // Pattern followed by _digit (e.g., "main_kernel_1")
        variant_matches.push_back({{name_str, func_list[i]}});
      }} else {{
        // Pattern not followed by _digit (e.g., "main_kernel" itself)
        base_matches.push_back({{name_str, func_list[i]}});
      }}
    }}
  }}

  // Decision logic: prefer base matches over variant matches
  if (!base_matches.empty()) {{
    if (base_matches.size() == 1) {{
      *out_func = base_matches[0].second;
      return CUDA_SUCCESS;
    }}

    // Multiple base matches - ambiguous
    std::cerr << "Error: Pattern '" << pattern << "' matched " << base_matches.size()
              << " base kernels (ambiguous). Matches found:\\n";
    for (const auto& match : base_matches) {{
      std::cerr << "  - " << match.first << "\\n";
    }}
    std::cerr << "Please use a more specific pattern.\\n";
    return CUDA_ERROR_NOT_FOUND;
  }}

  // No base matches, try variant matches
  if (!variant_matches.empty()) {{
    if (variant_matches.size() == 1) {{
      *out_func = variant_matches[0].second;
      return CUDA_SUCCESS;
    }}

    // Multiple variant matches - ambiguous
    std::cerr << "Error: Pattern '" << pattern << "' matched " << variant_matches.size()
              << " variant kernels (ambiguous). Matches found:\\n";
    for (const auto& match : variant_matches) {{
      std::cerr << "  - " << match.first << "\\n";
    }}
    std::cerr << "Please use a more specific pattern (e.g., '" << pattern << "_1').\\n";
    return CUDA_ERROR_NOT_FOUND;
  }}

  // No matches at all
  std::cerr << "Failed to find kernel matching pattern '" << pattern << "'\\n";
  return CUDA_ERROR_NOT_FOUND;
}}


// Initialize CUDA module for a specific device (called once per device)
// Thread-safe and supports multi-GPU by tracking modules per device
// device_id: PyTorch CUDA device ID (e.g., 0, 1, 2...)
static CUresult tilelang_init_cuda_module(const std::string& cubin_path, int device_id) {{
  std::lock_guard<std::mutex> lock(g_devices_mutex);

  // Fast path: module already initialized for this device
  if (g_device_modules.find(device_id) != g_device_modules.end()) {{
    return CUDA_SUCCESS;
  }}

  CUresult result;
  result = cuInit(0);
  if (result != CUDA_SUCCESS) {{
    std::cerr << "Failed to initialize CUDA: " << result << "\\n";
    return result;
  }}

  // Get device handle for the requested device_id
  CUdevice device;
  result = cuDeviceGet(&device, device_id);
  if (result != CUDA_SUCCESS) {{
    std::cerr << "Failed to get CUDA device " << device_id << ": " << result << "\\n";
    return result;
  }}

  // Retain and set the primary context for this device
  // PyTorch (Runtime API) creates and activates the primary context
  // We need to explicitly acquire it via Driver API and set it as current
  CUcontext ctx;
  result = cuDevicePrimaryCtxRetain(&ctx, device);
  if (result != CUDA_SUCCESS) {{
    std::cerr << "Failed to retain primary context for device " << device_id << ": " << result << "\\n";
    return result;
  }}

  result = cuCtxSetCurrent(ctx);
  if (result != CUDA_SUCCESS) {{
    std::cerr << "Failed to set current context for device " << device_id << ": " << result << "\\n";
    return result;
  }}

  // Store the retained context for cleanup
  g_device_contexts[device_id] = ctx;

  // Read cubin file
  std::ifstream cubin_file(cubin_path.c_str(), std::ios::binary);
  if (!cubin_file) {{
    std::cerr << "Failed to open cubin file: " << cubin_path << "\\n";
    return CUDA_ERROR_FILE_NOT_FOUND;
  }}

  std::vector<char> cubin_data((std::istreambuf_iterator<char>(cubin_file)),
                                std::istreambuf_iterator<char>());
  cubin_file.close();

  if (cubin_data.empty()) {{
    std::cerr << "Empty cubin file: " << cubin_path << "\\n";
    return CUDA_ERROR_INVALID_IMAGE;
  }}

  // Load module for this specific device
  CUmodule module;
  result = cuModuleLoadData(&module, cubin_data.data());
  if (result != CUDA_SUCCESS) {{
    std::cerr << "Failed to load CUDA module on device " << device_id << ": " << result << "\\n";
    return result;
  }}

  // Store module for this device
  g_device_modules[device_id] = module;

  return CUDA_SUCCESS;
}}

// Initialize kernel functions for a specific device (called once per device)
// Thread-safe and supports multi-GPU by tracking kernels per device
static CUresult tilelang_init_kernels(int device_id) {{
  std::lock_guard<std::mutex> lock(g_devices_mutex);

  // Fast path: kernels already initialized for this device
  if (g_device_kernels.find(device_id) != g_device_kernels.end()) {{
    return CUDA_SUCCESS;
  }}

  // Get the module for this device
  auto module_it = g_device_modules.find(device_id);
  if (module_it == g_device_modules.end()) {{
    std::cerr << "Module not initialized for device " << device_id << "\\n";
    return CUDA_ERROR_NOT_INITIALIZED;
  }}
  CUmodule module = module_it->second;

  // Initialize kernel storage for this device
  std::vector<CUfunction> kernels({num_kernels});
  CUresult result;

{kernel_inits}

  // Store kernels for this device
  g_device_kernels[device_id] = kernels;

  return CUDA_SUCCESS;
}}

// TMA descriptor initialization (host-side)
{tma_init_func}

// Main kernel launcher
extern "C" CUresult launch_kernel({launch_func_sig}, uint64_t _stream, int device_id, tvm::ffi::Bytes cubin_path) {{
  CUresult result;

  std::string cubin_path_str(reinterpret_cast<const char*>(cubin_path.data()), cubin_path.size());
  result = tilelang_init_cuda_module(cubin_path_str, device_id);
  if (result != CUDA_SUCCESS) return result;

  result = tilelang_init_kernels(device_id);
  if (result != CUDA_SUCCESS) return result;

{get_ptr_code}
  CUstream stream = (CUstream)_stream;

{tma_init_in_launch}

{kernel_launches}

  return CUDA_SUCCESS;
}}

// Cleanup function
extern "C" CUresult cleanup_module() {{
  std::lock_guard<std::mutex> lock(g_devices_mutex);

  CUresult last_error = CUDA_SUCCESS;

  // Step 1: Unload modules for all devices
  for (auto& pair : g_device_modules) {{
    if (pair.second != nullptr) {{
      CUresult result = cuModuleUnload(pair.second);
      if (result != CUDA_SUCCESS) {{
        std::cerr << "Failed to unload module for device " << pair.first
                  << ": " << result << "\\n";
        last_error = result;
        // Continue cleanup even if unload fails
      }}
    }}
  }}

  // Step 2: Release primary contexts (must execute even if module unload failed)
  // This ensures the reference count is decremented for every cuDevicePrimaryCtxRetain
  for (auto& pair : g_device_contexts) {{
    int device_id = pair.first;
    CUcontext ctx = pair.second;

    if (ctx != nullptr) {{
      CUdevice device;
      CUresult result = cuDeviceGet(&device, device_id);
      if (result == CUDA_SUCCESS) {{
        result = cuDevicePrimaryCtxRelease(device);
        if (result != CUDA_SUCCESS) {{
          std::cerr << "Failed to release primary context for device "
                    << device_id << ": " << result << "\\n";
          last_error = result;
        }}
      }} else {{
        std::cerr << "Failed to get device " << device_id
                  << " for context release: " << result << "\\n";
        last_error = result;
      }}
    }}
  }}

  // Step 3: Clear all maps
  g_device_modules.clear();
  g_device_kernels.clear();
  g_device_contexts.clear();

  return last_error;
}}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_kernel, launch_kernel);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cleanup_module, cleanup_module);
"""

# =============================================================================
# PYTHON CUBIN GENERATION TEMPLATES
# =============================================================================

# TMA descriptor atom initialization template
CUBIN_TMA_ATOM_INIT_TEMPLATE = """\
    {desc_name} = tl.Gemm_SM90.get_tma_atom(__fake_tensor__, (32, 32))"""

# Kernel launch call template
CUBIN_KERNEL_LAUNCH_TEMPLATE = """\
    {function_name}({call_args}).launch(
      grid=[{grid_x}, {grid_y}, {grid_z}],
      block=[{block_x}, {block_y}, {block_z}],
      smem={smem_size},
      stream=stream,
    )"""

# Fake tensor creation template
CUBIN_FAKE_TENSOR_TEMPLATE = """\
  __fake_{arg_name}__ = make_fake_compact_tensor(_DTYPE_MAP[str({arg_name}.dtype)], {arg_name}.shape, stride_order={arg_name}.dim_order()[::-1], assumed_align=16)"""

# Complete cubin generation code template
# {lib_code} contains the @cute.kernel definitions and is embedded here
CUBIN_GEN_CODE_TEMPLATE = """\
{lib_code}

  @cute.jit
  def kernel_wrapper({wrapper_args}):
{tma_init_code}{kernel_launches}

  # Compile kernels to generate cubin
{fake_tensor_code}{fake_tma_tensor_code}  __fake_stream__ = make_fake_stream()
  # Always generate cubin under a unique staging directory to avoid concurrent
  # processes clobbering each other's intermediate artifacts.
  _staging_dir = Path(tempfile.mkdtemp(
      prefix=Path(__file__).stem + ".cubin.staging.",
      dir=_module_dir,
  ))
  try:
    _kernel_wrapper = cute.compile(
        kernel_wrapper,
        {compile_args},
        options=f"--enable-tvm-ffi --keep-cubin --dump-dir={{_staging_dir.as_posix()}}",
    )

    # CuTeDSL generates a long, mangled cubin filename that includes argument/type info,
    # e.g. "cutlass_kernel_wrapper_FakeTensor...sm_90a.cubin". We expect exactly one cubin.
    _cubin_files = sorted(_staging_dir.glob("*.cubin"), key=lambda p: p.stat().st_mtime)
    if len(_cubin_files) != 1:
      raise RuntimeError(
          f"Expected exactly one .cubin under {{_staging_dir}}, got {{len(_cubin_files)}}: {{_cubin_files}}"
      )
    os.replace(_cubin_files[0], _cubin_path)
  finally:
    shutil.rmtree(_staging_dir, ignore_errors=True)"""

# =============================================================================
# PYTHON HOST FUNCTION TEMPLATE
# =============================================================================

PYTHON_HOST_FUNC_TEMPLATE = """\
import os
from pathlib import Path

# Minimal imports for runtime (no cutlass/cute - only needed for cubin generation)
import tvm.runtime as runtime

_cpp_launcher = None
_cpp_launcher_lib = None
_cubin_generated = False

# Pre-compute paths - cubin is stored alongside the launcher .so
# Use module basename to avoid conflicts when multiple kernels run concurrently
# e.g., "/tmp/tmp8liu__ho.py" -> "/tmp/tmp8liu__ho.cubin"
#       "kernel.py" (in cache) -> "kernel.cubin"
_module_dir = Path(os.path.dirname(__file__))
_cubin_path = _module_dir / (Path(__file__).stem + ".cubin")
_cubin_path_bytes = _cubin_path.as_posix().encode('utf-8')
_cubin_needs_generation = not _cubin_path.exists()

def _generate_cubin_if_needed({cubin_gen_params}):
  \"\"\"Generate cubin file on first call.

  All CuTeDSL imports are inside this function to avoid slow
  module-level initialization when loading from cache.
  \"\"\"
  global _cubin_generated, _cubin_path

  # Lazy import CuTeDSL only when cubin generation is needed
  from cuda.bindings.driver import CUstream
  import cutlass
  import cutlass.cute as cute
  from cutlass.cute.runtime import make_fake_stream, make_fake_compact_tensor
  import tilelang.contrib.cutedsl as tl
  # We rely on CuTeDSL's keep-cubin artifact rather than custom extraction.
  import tempfile
  import shutil

  _DTYPE_MAP = {{
      "torch.float32": cutlass.Float32,
      "torch.float16": cutlass.Float16,
      "torch.bfloat16": cutlass.BFloat16,
      "torch.float8_e4m3fnuz": cutlass.Float8E4M3FN,
      "torch.float8_e4m3fn": cutlass.Float8E4M3FN,
      "torch.float8_e5m2": cutlass.Float8E5M2,
      "torch.float64": cutlass.Float64,
      "torch.int64": cutlass.Int64,
      "torch.int32": cutlass.Int32,
      "torch.uint32": cutlass.Uint32,
      "torch.bool": cutlass.Boolean,
      "torch.int8": cutlass.Int8,
      "torch.uint8": cutlass.Uint8,
      "torch.int16": cutlass.Int16,
      "torch.uint16": cutlass.Uint16,
      "torch.uchar": cutlass.Uint8,
  }}

{cubin_gen_code}

  _cubin_generated = True

def _load_cpp_launcher():
  \"\"\"Load C++ kernel launcher.\"\"\"
  global _cpp_launcher, _cpp_launcher_lib
  if _cpp_launcher is not None:
    return _cpp_launcher

  lib_path = os.path.join(os.path.dirname(__file__), "{launcher_lib_name}")
  if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Launcher not found: {{lib_path}}")

  _cpp_launcher_lib = runtime.load_module(lib_path)
  _cpp_launcher = _cpp_launcher_lib["launch_kernel"]
  return _cpp_launcher

def call({call_func_params}, stream, device_id=0):
  \"\"\"Kernel dispatch function.

  Args:
      stream: CUDA stream handle
      device_id: CUDA device ID (should be passed from caller, defaults to 0 for backward compatibility)
  \"\"\"
  global _cubin_path_bytes, _cubin_needs_generation

  if _cubin_needs_generation:
    _generate_cubin_if_needed({cubin_gen_call_args})
    _cubin_needs_generation = False

{arg_prep_code}

  launcher = _load_cpp_launcher()
  result = launcher({launcher_call_args}, stream, device_id, _cubin_path_bytes)

  if result != 0:
    raise RuntimeError(f"Kernel launch failed with CUDA error {{result}}")
"""

# =============================================================================
# WRAPPER CLASS
# =============================================================================


class TLCuTeDSLSourceWrapper(TLCUDASourceWrapper):
    """Wrapper class for TileLang CuTe DSL backend with C++ launcher.

    Generates optimized C++ launcher code that:
    - Loads cubin via CUDA Driver API
    - Passes TMA descriptors by value (host-side, no device copy)
    - Launches kernels with minimal Python overhead
    - Supports both single and multiple kernel scenarios
    """

    _TYPE_MAP: ClassVar[dict[str, str]] = {
        "float32": "cutlass.Float32",
        "float16": "cutlass.Float16",
        "bfloat16": "cutlass.BFloat16",
        "float8_e4m3": "cutlass.Float8E4M3",
        "float8_e5m2": "cutlass.Float8E5M2",
        "float64": "cutlass.Float64",
        "int64": "cutlass.Int64",
        "int32": "cutlass.Int32",
        "uint32": "cutlass.Uint32",
        "bool": "cutlass.Boolean",
        "int8": "cutlass.Int8",
        "uint8": "cutlass.Uint8",
        "int16": "cutlass.Int16",
        "uint16": "cutlass.Uint16",
        "uchar": "cutlass.Uint8",
    }

    # C++ launcher code must not depend on cutlass Python types.
    # Use plain C/C++ types for expression rendering inside generated .cpp.
    _CXX_TYPE_MAP: ClassVar[dict[str, str]] = {
        "float32": "float",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int32_t",
        "uint32": "uint32_t",
        "bool": "bool",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
    }

    _CTYPES_MAP: ClassVar[dict[str, str]] = {
        "buffer": "ctypes.c_uint64",
        "cutlass.Float32": "ctypes.c_float",
        "cutlass.Float16": "ctypes.c_uint16",
        "cutlass.Float64": "ctypes.c_double",
        "cutlass.Int64": "ctypes.c_int64",
        "cutlass.Int32": "ctypes.c_int32",
        "cutlass.Uint32": "ctypes.c_uint32",
        "cutlass.Int8": "ctypes.c_int8",
        "cutlass.Uint8": "ctypes.c_uint8",
        "cutlass.Int16": "ctypes.c_int16",
        "cutlass.Uint16": "ctypes.c_uint16",
        "int": "ctypes.c_int32",
    }

    _generated_host_func: str | None = None
    _launcher_lib_name: str | None = None

    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        target: Target,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        super().__init__(scheduled_ir_module, source, target, device_mod, host_mod, pass_configs)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def host_func(self):
        """Override parent's host_func to return generated Python code."""
        if self._generated_host_func is not None:
            return self._generated_host_func
        return super().host_func

    @host_func.setter
    def host_func(self, value):
        """Allow setting generated host function code."""
        self._generated_host_func = value

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _pythonic_expr(self, expr: tvm.tir.PrimExpr) -> str:
        """Convert TVM expression to Python string."""
        return pythonic_expr(expr, self._TYPE_MAP, floor_div_op="//")

    def _cxx_expr(self, expr: tvm.tir.PrimExpr) -> str:
        """Convert TVM expression to C++ string for generated launcher code."""
        return pythonic_expr(expr, self._CXX_TYPE_MAP)

    @staticmethod
    def _cxx_cast(ctype: str, expr_str: str) -> str:
        return f"static_cast<{ctype}>({expr_str})"

    def _collect_function_args(self) -> tuple[list[dict], list[str]]:
        """Collect all function arguments from primary function.

        Returns:
            Tuple of (function_args, buffer_args)
        """
        function_args = []
        buffer_args = []

        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append({"name": buffer.data.name, "type": "buffer"})
                buffer_args.append(buffer.data.name)
            elif isinstance(param, tvm.tir.Var):
                function_args.append({"name": param.name, "type": self._TYPE_MAP[param.dtype]})
            else:
                raise ValueError(f"Parameter {param} not in buffer map")

        existing_names = {arg["name"] for arg in function_args}
        for dyn_sym in self.get_dynamic_symbolic_set(self.prim_func):
            dyn_sym_name, dyn_sym_dtype = dyn_sym if isinstance(dyn_sym, tuple) else (dyn_sym, "int32")
            if dyn_sym_name in existing_names:
                continue
            existing_names.add(dyn_sym_name)
            function_args.append({"name": dyn_sym_name, "type": self._TYPE_MAP.get(dyn_sym_dtype, "int")})

        return function_args, buffer_args

    @staticmethod
    def _extract_func_call_args(
        declaration: str,
        function_args: list[dict],
        function_params: list,
        desc_name_map: dict[str, str] | None = None,
        desc_name_var_map: dict[str, tvm.tir.Var] | None = None,
    ) -> list[tuple[str, str]]:
        """Extract function call arguments from Python function declaration."""

        def maybe_desc(name: str | tuple[str, str], param_names: list[str], i: int):
            name_str = name if isinstance(name, str) else name[0]
            param = param_names[i]
            if not (param == name_str + "_desc" or param.startswith(name_str + "_desc_")):
                return False
            if desc_name_map is not None:
                desc_name_map[param] = name_str
            return True

        def extract_param_names_ast(decl: str) -> list[str] | None:
            """Extract parameter names using AST parsing."""
            import ast
            import warnings

            try:
                # Build a syntactically valid function by adding a body
                func_stub = decl.rstrip()
                if not func_stub.endswith(":"):
                    func_stub += ":"
                func_stub += "\n    pass"

                # Parse and locate the FunctionDef
                tree = ast.parse(func_stub)
                func_def = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_def = node
                        break

                if func_def is None:
                    return None

                # Extract parameter names, skipping 'self'
                param_names = []
                for arg in func_def.args.args:
                    if arg.arg != "self":
                        param_names.append(arg.arg)

                return param_names
            except Exception as e:
                warnings.warn(f"AST parsing failed for function declaration, falling back to split-based parsing: {e}", stacklevel=2)
                return None

        def extract_param_names_split(decl: str) -> list[str]:
            """Fallback: extract parameter names using naive split-based parsing."""
            paren_start = decl.find("(")
            paren_end = decl.rfind(")")
            if paren_start == -1 or paren_end == -1:
                return []

            params_str = decl[paren_start + 1 : paren_end].strip()
            if not params_str:
                return []

            param_parts = params_str.split(",")
            param_names = []
            for param in param_parts:
                param = param.strip()
                if not param or param == "self":
                    continue
                if ":" in param:
                    param_name = param.split(":")[0].strip()
                else:
                    param_name = param.strip()
                param_names.append(param_name)

            return param_names

        # Try AST-based extraction first, fallback to split-based
        param_names = extract_param_names_ast(declaration)
        if param_names is None:
            param_names = extract_param_names_split(declaration)

        call_args = []
        for i, param_name in enumerate(param_names):
            for arg in function_args:
                if arg["name"] == param_name:
                    call_args.append((param_name, arg["type"]))
                elif maybe_desc(arg["name"], param_names, i):
                    call_args.append((param_name, "None"))
                    if desc_name_var_map is not None and function_params is not None:
                        assert len(call_args) <= len(function_params)
                        desc_name_var_map[param_name] = function_params[len(call_args) - 1]
        return call_args

    @staticmethod
    def _filter_non_descriptor_args(
        call_args: list[tuple[str, str]], desc_names: list[str], tma_tensors: list[str]
    ) -> list[tuple[str, str]]:
        """Filter out descriptor arguments."""
        filtered = []
        for arg_name, arg_type in call_args:
            if "desc" in arg_name and arg_name in desc_names:
                continue
            if arg_name in tma_tensors:
                continue
            filtered.append((arg_name, arg_type))
        return filtered

    # =========================================================================
    # TMA Descriptor Code Generation
    # =========================================================================

    def _generate_tma_desc_init(self, desc_name: str, desc_idx: int, tensor_name: str, info: dict) -> str:
        """Generate single TMA descriptor initialization code."""
        if info.get("is_img2col", False):
            rank = info["tensor_rank"]
            return CPP_TMA_IM2COL_DESC_INIT_TEMPLATE.format(
                desc_idx=desc_idx,
                desc_name=desc_name,
                tensor_name=tensor_name,
                rank=rank,
                stride_rank=rank - 1,
                rank_minus_two=rank - 2,
                global_dim_values=", ".join(self._cxx_cast("uint64_t", self._cxx_expr(x)) for x in info["global_dim"]),
                global_stride_values=", ".join(self._cxx_cast("uint64_t", self._cxx_expr(x)) for x in info["global_stride"][1:]),
                elem_stride_values=", ".join(self._cxx_cast("uint32_t", self._cxx_expr(x)) for x in info["element_strides"]),
                lower_corner_values=", ".join(self._cxx_cast("int32_t", self._cxx_expr(x)) for x in info["lower_corner"]),
                upper_corner_values=", ".join(self._cxx_cast("int32_t", self._cxx_expr(x)) for x in info["upper_corner"]),
                # Match NVRTC wrapper naming: channelsPerPixel then pixelsPerColumn
                channels_per_pixel=info["smem_box_channel"],
                pixels_per_column=info["smem_box_pixel"],
                dtype=info["dtype"],
                interleave=info["interleave"],
                swizzle=info["swizzle"],
                l2_promotion=info["l2Promotion"],
                oob_fill=info["oobFill"],
            )

        return CPP_TMA_DESC_INIT_TEMPLATE.format(
            desc_idx=desc_idx,
            desc_name=desc_name,
            tensor_name=tensor_name,
            rank=info["tensor_rank"],
            global_dim_values=", ".join(self._cxx_cast("uint64_t", self._cxx_expr(x)) for x in info["global_dim"]),
            stride_rank=info["tensor_rank"] - 1,
            global_stride_values=", ".join(self._cxx_cast("uint64_t", self._cxx_expr(x)) for x in info["global_stride"][1:]),
            box_dim_values=", ".join(self._cxx_cast("uint32_t", self._cxx_expr(x)) for x in info["box_dim"]),
            elem_stride_values=", ".join(self._cxx_cast("uint32_t", self._cxx_expr(x)) for x in info["element_strides"]),
            dtype=info["dtype"],
            interleave=info["interleave"],
            swizzle=info["swizzle"],
            l2_promotion=info["l2Promotion"],
            oob_fill=info["oobFill"],
        )

    def _generate_tma_init_func(
        self,
        desc_names: list[str],
        tensor_args: list[str],
        tensor_arg_map: dict[str, tuple[str, int]],
        scalar_args: list[dict[str, str]],
    ) -> str:
        """Generate TMA init function code (creates descriptors in caller-provided host array).

        TMA descriptors are stored in stack-local tma_descs[] array in launch_kernel.
        cuLaunchKernel automatically handles __grid_constant__ params.
        """
        if not desc_names:
            return ""

        func_args_parts = [f"uint64_t {arg}_ptr" for arg in tensor_args]
        for arg in scalar_args:
            if arg["type"] in ["int", "cutlass.Int32"]:
                func_args_parts.append(f"int32_t {arg['name']}")
            elif arg["type"] in ["float", "cutlass.Float32"]:
                func_args_parts.append(f"float {arg['name']}")
            else:
                # Default to int32_t for scalars used in shape/stride math
                func_args_parts.append(f"int32_t {arg['name']}")
        func_args = ", ".join(func_args_parts)
        num_descs = len(desc_names)

        desc_inits = []
        for idx, desc_name in enumerate(desc_names):
            info = self.tma_desc_info[desc_name]
            tensor_name, _ = tensor_arg_map[desc_name]
            desc_inits.append(self._generate_tma_desc_init(desc_name, idx, tensor_name, info))

        return CPP_TMA_INIT_FUNC_TEMPLATE.format(
            func_args=func_args,
            num_descs=num_descs,
            desc_init_code="\n".join(desc_inits),
        )

    def _generate_tma_launch_init(
        self, desc_names: list[str], tma_tensors: list[str], scalar_args: list[dict[str, str]], num_tma_descs: int
    ) -> str:
        """Generate TMA initialization code for launch function (host memory mode).

        TMA descriptors stay on host. cuLaunchKernel copies them to param space
        when kernel uses __grid_constant__ CUtensorMap parameter.
        """
        if not desc_names:
            return ""

        # Generate tma_init call args (no device_ptr needed)
        call_args_parts = [f"{arg}_ptr" for arg in tma_tensors] + [arg["name"] for arg in scalar_args]
        tma_tensor_args = ", ".join(call_args_parts)

        return CPP_TMA_LAUNCH_INIT_TEMPLATE.format(
            num_tma_descs=num_tma_descs,
            tma_tensor_args=tma_tensor_args,
        )

    # =========================================================================
    # Kernel Code Generation
    # =========================================================================

    def _generate_kernel_init(self, kernel_idx: int, kernel_name: str, smem_size: int) -> str:
        """Generate kernel initialization code."""
        return CPP_KERNEL_INIT_TEMPLATE.format(
            kernel_idx=kernel_idx,
            kernel_name=kernel_name,
            smem_size=smem_size,
        )

    def _generate_kernel_launch(self, kernel_meta: dict, kernel_idx: int, all_desc_names: list[str]) -> str:
        """Generate single kernel launch code.

        For __grid_constant__ CUtensorMap params:
        - Pass CUtensorMap* directly (not CUtensorMap**)
        - cuLaunchKernel copies 128 bytes to kernel param space
        """
        call_args = kernel_meta["call_args"]
        desc_names = kernel_meta["desc_names"]
        function_info = kernel_meta["function_info"]

        # Build kernel args
        kernel_args = []
        for arg_name, arg_type in call_args:
            if "desc" in arg_name and arg_name in desc_names:
                # For __grid_constant__ CUtensorMap: pass host pointer directly
                # cuLaunchKernel will copy 128-byte CUtensorMap to param space
                desc_idx = all_desc_names.index(arg_name)
                kernel_args.append(f"&tma_descs[{desc_idx}]")
            elif arg_type == "buffer":
                kernel_args.append(f"&{arg_name}_ptr")
            else:
                kernel_args.append(f"&{arg_name}")

        grid = function_info["grid_info"]
        block = function_info["block_info"]
        smem_size = function_info["dynamic_smem_buf"] or 0

        return CPP_KERNEL_LAUNCH_TEMPLATE.format(
            kernel_idx=kernel_idx,
            kernel_name=kernel_meta["function_name"],
            kernel_args=", ".join(kernel_args),
            grid_x=self._cxx_expr(grid[0]),
            grid_y=self._cxx_expr(grid[1]),
            grid_z=self._cxx_expr(grid[2]),
            block_x=self._cxx_expr(block[0]),
            block_y=self._cxx_expr(block[1]),
            block_z=self._cxx_expr(block[2]),
            smem_size=smem_size,
        )

    # =========================================================================
    # C++ Launcher Generation
    # =========================================================================

    def _generate_cpp_launcher(
        self,
        kernel_metadata_list: list[dict],
        function_args: list[dict],
        all_tma_tensors: list[str],
        all_desc_names: list[str],
        tensor_arg_map: dict[str, tuple[str, int]],
    ) -> str:
        """Generate complete C++ launcher code using templates.

        TMA descriptors are stored on HOST memory in stack-local tma_descs[] array.
        cuLaunchKernel automatically copies 128-byte CUtensorMap to kernel param space
        when kernel uses __grid_constant__ parameter.
        """
        num_kernels = len(kernel_metadata_list)
        num_tma_descs = max(len(all_desc_names), 1)  # At least 1 to avoid zero-size array

        # Generate kernel inits
        kernel_inits = "\n".join(
            self._generate_kernel_init(idx, km["function_name"], km["function_info"]["dynamic_smem_buf"] or 0)
            for idx, km in enumerate(kernel_metadata_list)
        )

        # Generate TMA init function
        scalar_args = [arg for arg in function_args if arg["type"] != "buffer"]
        tma_init_func = self._generate_tma_init_func(all_desc_names, all_tma_tensors, tensor_arg_map, scalar_args)

        # Generate launch function signature and get_ptr code
        func_sig_parts = []
        get_ptr_code = ""
        for arg in function_args:
            if arg["type"] == "buffer":
                func_sig_parts.append(f"tvm::ffi::TensorView {arg['name']}")
                get_ptr_code += f"  uint64_t {arg['name']}_ptr = reinterpret_cast<uint64_t>({arg['name']}.data_ptr());\n"
            elif arg["type"] in ["int", "cutlass.Int32"]:
                func_sig_parts.append(f"int32_t {arg['name']}")
            elif arg["type"] in ["float", "cutlass.Float32"]:
                func_sig_parts.append(f"float {arg['name']}")
            else:
                func_sig_parts.append(f"int32_t {arg['name']}")

        # Generate TMA init in launch
        tma_init_in_launch = self._generate_tma_launch_init(all_desc_names, all_tma_tensors, scalar_args, num_tma_descs)

        # Generate kernel launches
        kernel_launches = "\n".join(self._generate_kernel_launch(km, idx, all_desc_names) for idx, km in enumerate(kernel_metadata_list))

        return CPP_LAUNCHER_TEMPLATE.format(
            num_kernels=num_kernels,
            num_tma_descs=num_tma_descs,
            kernel_inits=kernel_inits,
            tma_init_func=tma_init_func,
            launch_func_sig=", ".join(func_sig_parts),
            get_ptr_code=get_ptr_code,
            tma_init_in_launch=tma_init_in_launch,
            kernel_launches=kernel_launches,
        )

    # =========================================================================
    # Python Wrapper Generation
    # =========================================================================

    def _generate_cubin_gen_code(
        self,
        kernel_metadata_list: list[dict],
        buffer_args: list[str],
        all_desc_names: list[str],
        lib_code: str = "",
    ) -> str:
        """Generate cubin generation code for Python wrapper using templates.

        Args:
            lib_code: The CuTeDSL kernel definitions (@cute.kernel decorated functions).
                      This will be embedded inside _generate_cubin_if_needed to enable
                      lazy loading of cutlass/cute modules.
        """
        # Build unified wrapper parameters
        wrapper_params_union = []
        for kernel_meta in kernel_metadata_list:
            for arg_name, _ in kernel_meta["call_args"]:
                if arg_name not in wrapper_params_union:
                    wrapper_params_union.append(arg_name)

        # Build inner args for cute.compile
        inner_args = []
        fake_inner_args = []
        for arg_name in wrapper_params_union:
            if arg_name in buffer_args:
                inner_args.append(f"{arg_name}_")
                fake_inner_args.append(f"__fake_{arg_name}__")
            elif arg_name in all_desc_names:
                continue
            else:
                inner_args.append(arg_name)
                fake_inner_args.append(arg_name)
        if all_desc_names:
            inner_args.append("__fake_tensor__")
            fake_inner_args.append("__fake_tensor__")
        fake_inner_args.append("__fake_stream__")

        # Generate TMA init code
        tma_init_code = ""
        if all_desc_names:
            tma_init_lines = ["    # Create dummy TMA atoms for compilation"]
            tma_init_lines.extend(CUBIN_TMA_ATOM_INIT_TEMPLATE.format(desc_name=desc_name) for desc_name in all_desc_names)
            tma_init_code = "\n".join(tma_init_lines) + "\n"

        # Generate kernel launch calls
        kernel_launches = "\n".join(
            CUBIN_KERNEL_LAUNCH_TEMPLATE.format(
                function_name=km["function_name"],
                call_args=", ".join(arg[0] if arg[0] not in buffer_args else f"{arg[0]}_" for arg in km["call_args"]),
                grid_x=self._pythonic_expr(km["function_info"]["grid_info"][0]),
                grid_y=self._pythonic_expr(km["function_info"]["grid_info"][1]),
                grid_z=self._pythonic_expr(km["function_info"]["grid_info"][2]),
                block_x=self._pythonic_expr(km["function_info"]["block_info"][0]),
                block_y=self._pythonic_expr(km["function_info"]["block_info"][1]),
                block_z=self._pythonic_expr(km["function_info"]["block_info"][2]),
                smem_size=km["function_info"]["dynamic_smem_buf"] or 0,
            )
            for km in kernel_metadata_list
        )

        # Generate fake tensor creation code
        # IMPORTANT: Generate fake tensors based on the *union* of parameters actually
        # passed to cute.compile (wrapper_params_union).
        #
        # In multi-kernel cases, a tensor may appear both as a TMA descriptor
        # (e.g. Output_partial_desc) for one kernel and as a plain tensor argument
        # (e.g. Output_partial_) for another kernel. Skipping fake tensor creation
        # just because a matching "{arg}_desc" exists is a correctness bug and
        # results in undefined names like "__fake_Output_partial__".
        fake_tensor_code = "\n".join(
            CUBIN_FAKE_TENSOR_TEMPLATE.format(arg_name=arg_name) for arg_name in wrapper_params_union if arg_name in buffer_args
        )
        if fake_tensor_code:
            fake_tensor_code += "\n"

        # Generate fake TMA tensor code
        fake_tma_tensor_code = ""
        if all_desc_names:
            fake_tma_tensor_code = (
                "  __fake_tensor__ = make_fake_compact_tensor(cutlass.Int32, (32, 32), stride_order=(1, 0), assumed_align=16)\n"
            )

        # Indent lib_code to be inside the function
        indented_lib_code = "\n".join("  " + line if line.strip() else line for line in lib_code.split("\n")) if lib_code else ""

        return CUBIN_GEN_CODE_TEMPLATE.format(
            lib_code=indented_lib_code,
            wrapper_args=", ".join(inner_args + ["stream: CUstream"]),
            tma_init_code=tma_init_code,
            kernel_launches=kernel_launches,
            fake_tensor_code=fake_tensor_code,
            fake_tma_tensor_code=fake_tma_tensor_code,
            compile_args=", ".join(fake_inner_args),
            primary_name=kernel_metadata_list[0]["function_name"],
        )

    def _generate_python_wrapper(
        self,
        function_args: list[dict],
        cubin_gen_code: str,
        cubin_gen_params: str,
    ) -> str:
        """Generate Python wrapper code."""
        # Build function parameters
        call_func_params = ", ".join(arg["name"] for arg in function_args)
        launcher_call_args = ", ".join(arg["name"] for arg in function_args)

        return PYTHON_HOST_FUNC_TEMPLATE.format(
            cubin_gen_params=cubin_gen_params,
            cubin_gen_code=cubin_gen_code,
            launcher_lib_name=self._launcher_lib_name,
            call_func_params=call_func_params,
            cubin_gen_call_args=cubin_gen_params,
            arg_prep_code="",
            launcher_call_args=launcher_call_args,
        )

    # =========================================================================
    # TMA Descriptor Processing
    # =========================================================================

    def _process_tma_descriptors(self, desc_names: list[str]) -> tuple[list[str], dict[str, tuple[str, int]]]:
        """Process TMA descriptors and return tensor args and mapping.

        Returns:
            Tuple of (tensor_args, tensor_arg_map)
        """
        if not hasattr(self, "tma_desc_info") or not desc_names:
            return [], {}

        tensor_args = []
        tensor_arg_map = {}

        for desc_name in desc_names:
            info = self.tma_desc_info[desc_name]
            # Extract the base buffer variable name (must be a Var, not arbitrary expression)
            global_addr = info["globalAddress"]
            if not isinstance(global_addr, tvm.tir.Var):
                raise ValueError(f"TMA globalAddress must be a buffer Var, got {type(global_addr)}: {global_addr}")
            tensor_name = global_addr.name

            if tensor_name not in tensor_args:
                tensor_args.append(tensor_name)
                tensor_arg_map[desc_name] = (tensor_name, len(tensor_args) - 1)
            else:
                tensor_arg_map[desc_name] = (tensor_name, tensor_args.index(tensor_name))

        return tensor_args, tensor_arg_map

    def generate_tma_descriptor_args(
        self,
        desc_name_map: dict[str, str],
        desc_name_var_map: dict[str, tvm.tir.Var],
        tma_desc_code_map: dict[str, str],
    ) -> list[str]:
        """Generate TMA descriptor information for C++ code generation.

        Returns:
            List of descriptor variable names in the order they were processed.
        """
        if self.tma_descriptor_args is None:
            return []

        if not hasattr(self, "tma_desc_info"):
            self.tma_desc_info = {}

        parsed_params = parse_tma_descriptor_args(self.tma_descriptor_args, desc_name_map, desc_name_var_map, self._pythonic_expr)

        desc_names_ordered = []

        for params in parsed_params:
            handle_name = params.handle_name

            if handle_name in tma_desc_code_map:
                continue

            desc_var = desc_name_var_map[handle_name]
            args = self.tma_descriptor_args[desc_var]
            _, dtype, tensor_rank, globalAddress, *remaining_args = args[1:]
            tensor_rank = int(tensor_rank)

            global_dim = remaining_args[:tensor_rank]
            global_stride = remaining_args[tensor_rank : 2 * tensor_rank]

            if not params.is_img2col:
                box_dim = remaining_args[2 * tensor_rank : 3 * tensor_rank]
                element_strides = remaining_args[3 * tensor_rank : 4 * tensor_rank]

                self.tma_desc_info[handle_name] = {
                    "desc_var": desc_var,
                    "is_img2col": False,
                    "dtype": params.dtype,
                    "tensor_rank": params.tensor_rank,
                    "globalAddress": params.global_address,
                    "global_dim": global_dim,
                    "global_stride": global_stride,
                    "box_dim": box_dim,
                    "element_strides": element_strides,
                    "interleave": params.interleave,
                    "swizzle": params.swizzle,
                    "l2Promotion": params.l2_promotion,
                    "oobFill": params.oob_fill,
                }
            else:
                element_strides = remaining_args[2 * tensor_rank : 3 * tensor_rank]

                self.tma_desc_info[handle_name] = {
                    "desc_var": desc_var,
                    "is_img2col": True,
                    "dtype": params.dtype,
                    "tensor_rank": params.tensor_rank,
                    "globalAddress": params.global_address,
                    "global_dim": global_dim,
                    "global_stride": global_stride,
                    "element_strides": element_strides,
                    "lower_corner": params.lower_corner,
                    "upper_corner": params.upper_corner,
                    "smem_box_channel": params.smem_box_channel,
                    "smem_box_pixel": params.smem_box_pixel,
                    "interleave": params.interleave,
                    "swizzle": params.swizzle,
                    "l2Promotion": params.l2_promotion,
                    "oobFill": params.oob_fill,
                }

            tma_desc_code_map[handle_name] = ""
            desc_names_ordered.append(handle_name)

        return desc_names_ordered

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    def create_dispatch_func(self, code, function_informations):
        """Create dispatch function - always use C++ launcher."""
        return self.create_dispatch_func_cpp_launcher(code, function_informations)

    def create_dispatch_func_cpp_launcher(self, code, function_informations):
        """Create dispatch function using C++ launcher."""
        function_args, buffer_args = self._collect_function_args()

        # Process each kernel and collect metadata
        kernel_metadata = []
        all_desc_names_union = []
        all_tma_tensors_union = []

        for function_name, function_info in function_informations.items():
            declaration = extract_python_func_declaration(code, function_name)
            desc_name_map: dict[str, str] = {}
            desc_name_var_map: dict[str, tvm.tir.Var] = {}
            call_args = self._extract_func_call_args(
                declaration,
                function_args,
                function_info["function_params"],
                desc_name_map,
                desc_name_var_map,
            )

            tma_desc_code_map = {}
            desc_names = self.generate_tma_descriptor_args(desc_name_map, desc_name_var_map, tma_desc_code_map)

            tma_tensor_args, _ = self._process_tma_descriptors(desc_names)

            kernel_metadata.append(
                {
                    "function_name": function_name,
                    "function_info": function_info,
                    "call_args": call_args,
                    "desc_names": desc_names,
                    "tma_tensor_args": tma_tensor_args,
                    "desc_name_map": desc_name_map,
                }
            )

            for desc in desc_names:
                if desc not in all_desc_names_union:
                    all_desc_names_union.append(desc)
            for t in tma_tensor_args:
                if t not in all_tma_tensors_union:
                    all_tma_tensors_union.append(t)

        # Process all TMA descriptors
        all_tma_tensors, tensor_arg_map = self._process_tma_descriptors(all_desc_names_union)

        # Generate C++ launcher
        launcher_cpp_code = self._generate_cpp_launcher(
            kernel_metadata, function_args, all_tma_tensors, all_desc_names_union, tensor_arg_map
        )

        self.launcher_cpp_code = launcher_cpp_code
        # Use a deterministic name so that:
        # 1) the generated kernel.py can always locate the launcher in the same directory
        # 2) KernelCache can store it under a stable filename
        self._launcher_lib_name = "launcher_lib.so"
        self.launcher_lib_name = self._launcher_lib_name

        # Generate cubin generation code (includes lib_code with @cute.kernel definitions)
        cubin_gen_code = self._generate_cubin_gen_code(
            kernel_metadata, buffer_args, all_desc_names_union, lib_code=getattr(self, "lib_code", "")
        )

        # Generate Python wrapper
        buffer_names = [arg["name"] for arg in function_args if arg["type"] == "buffer"]
        # Cubin generation may reference scalar args (e.g., dynamic symbols like m/n/k)
        # inside `kernel_wrapper` and `cute.compile(...)`. They must be visible in
        # `_generate_cubin_if_needed(...)` scope, so include them in its signature.
        scalar_names = [arg["name"] for arg in function_args if arg["type"] != "buffer"]
        cubin_gen_params = ", ".join(buffer_names + scalar_names)

        python_wrapper = self._generate_python_wrapper(function_args, cubin_gen_code, cubin_gen_params)

        return python_wrapper

    def get_launcher_cpp_code(self) -> str:
        """Get the generated C++ launcher code."""
        return getattr(self, "launcher_cpp_code", "")

    def update_lib_code(self, code: str):
        """Update the library code with the given code string."""
        self.lib_code = code

        function_informations = {}
        for function_name in self.function_names:
            if (function_name not in self.block_info) or (function_name not in self.grid_info):
                continue

            assert function_name in self.device_mod, f"Function {function_name} not found in device module"
            device_func = self.device_mod[function_name]
            kernel_params_cnt = len(device_func.params)
            function_params: list[str] = None

            def visitor(node, fn=function_name, param_cnt=kernel_params_cnt):
                nonlocal function_params
                if isinstance(node, tvm.tir.Call):
                    if not (hasattr(node, "op") and node.op == tvm.ir.Op.get("tir.tvm_call_packed")):
                        return
                    args = node.args
                    if not args or args[0] != fn:
                        return
                    if len(args) < 1 + param_cnt:
                        raise AssertionError("tvm_call_packed should have at least 1 argument and match device function parameters")
                    function_params = args[1 : 1 + param_cnt]

            post_order_visit(self.host_func.body, visitor)
            assert function_params is not None, "function_params should not be None"

            function_informations[function_name] = {
                "function_name": function_name,
                "block_info": self.block_info[function_name],
                "grid_info": self.grid_info[function_name],
                "dynamic_smem_buf": self.dynamic_smem_buf[function_name],
                "function_params": function_params,
            }

        self.host_func = self.create_dispatch_func(code, function_informations)
        return self.lib_code
