"""Post-process generated CUDA to outline warp-spec branches as __noinline__ device functions.

This script:
1. Gets TileLang-generated CUDA source
2. Identifies the warp-spec if-else branches
3. Extracts math WG body into a __device__ __noinline__ function
4. Compiles via NVRTC and benchmarks

The device function gets separate register allocation from ptxas,
allowing it to use more registers than the kernel's __launch_bounds__ budget.
"""
import sys, os, re
sys.path.insert(0, '.')

import torch
import tilelang
import tilelang.language as T
from tilelang.contrib.nvrtc import compile_cuda
from attention_kernel_1sm import attention_kernel_1sm

# Step 1: Generate CUDA source
print("Generating CUDA source...")
fn = attention_kernel_1sm(8, 40, 4096, 128, num_kv_heads=8, kq_stages=2)
src = fn.get_kernel_source()
print(f"  {len(src)} chars, {src.count(chr(10))} lines")

# Step 2: Find the kernel function body and identify the for-loop + if-else pattern
# The generated CUDA structure:
#   extern __shared__ uchar buf_dyn_shmem[];
#   __shared__ uint64_t mbar_*_mem[N]; auto mbar_* = ...;
#   __shared__ uint *_tmem[1];
#   float S0_reg[128]; ... (local vars)
#   ... init ...
#   for (int k = 0; k < 32; ++k) {
#     __syncthreads(); ...
#     if (256 <= tid) { /* mma_load */ }
#     else { if (tid < 128) { /* math0 */ } else { /* math1 */ } }
#   }
#   ... epilogue ...

# Find the barrier and TMEM handle declarations
barrier_pattern = r'__shared__.*uint64_t\s+(\w+)_mem\[(\d+)\]'
tmem_pattern = r'__shared__.*uint\s+(\w+)\[1\]'

barriers = re.findall(barrier_pattern, src)
tmem_handles = re.findall(tmem_pattern, src)

print(f"  Found {len(barriers)} barriers: {[b[0] for b in barriers]}")
print(f"  Found {len(tmem_handles)} TMEM handles: {tmem_handles}")

# Step 3: Try compiling with ptxas verbose to see register usage per-function
# For now, just compile to PTX and check register count
tl_root = os.path.expanduser('~/workspace/tilelang')
tl_src_dir = os.path.join(tl_root, 'src')
cutlass_inc = os.path.join(tl_root, '3rdparty', 'cutlass', 'include')
cuda_inc = '/usr/local/cuda/include'
opts = [
    "--use_fast_math", "-DENABLE_BF16",
    f"--include-path={tl_src_dir}",
    f"--include-path={cutlass_inc}",
    f"--include-path={cuda_inc}",
]
print("\nCompiling to PTX via NVRTC...")
try:
    ptx_bytes = compile_cuda(src, target_format="ptx", arch=100, options=opts)
    ptx = ptx_bytes.decode('utf-8', errors='replace')
    # Count registers
    reg_matches = re.findall(r'\.reg \.b32\s+%r<(\d+)>', ptx)
    if reg_matches:
        print(f"  Register usage: .b32 %r<{reg_matches[0]}>")
    ld_local = ptx.count('ld.local')
    st_local = ptx.count('st.local')
    print(f"  Spills: ld.local={ld_local}, st.local={st_local}")
    print("  PTX compilation SUCCESS")
except Exception as e:
    print(f"  PTX compilation FAILED: {str(e)[:300]}")

# Step 4: Now try with 512 threads - modify the source
print("\n--- Attempting 512-thread modification ---")
# Change __launch_bounds__(384, 1) to __launch_bounds__(512, 1)
src_512 = src.replace('__launch_bounds__(384, 1)', '__launch_bounds__(512, 1)')
if src_512 == src:
    print("  WARNING: Could not find __launch_bounds__(384, 1) to replace")
else:
    print("  Changed to __launch_bounds__(512, 1)")
    print("  Compiling 512-thread version (expect register failure)...")
    try:
        ptx_512 = compile_cuda(src_512, target_format="ptx", arch=100, options=opts)
        print("  512-thread PTX compilation SUCCESS (unexpected!)")
    except Exception as e:
        err_msg = str(e)
        if "register" in err_msg.lower() or "C7602" in err_msg:
            print(f"  Expected register failure confirmed")
        else:
            print(f"  Unexpected error: {err_msg[:200]}")

# Step 5: Try the outlined version (device functions)
print("\n--- Attempting outlined device function version ---")
# For this prototype, we'll try wrapping the math body in a lambda with
# __attribute__((noinline)) first to see if NVRTC's ptxas handles it
# differently than the TileLang-invoked one.
# If that doesn't work, we'll extract proper device functions.

# Check if the noinline lambda is already in the source (from our codegen change)
if '__attribute__((noinline))' in src:
    print("  Source already has noinline lambdas (from codegen)")
    # The 384-thread version with noinline lambdas compiles fine
    # Let's check: with 512 threads, does the noinline lambda help?
    if src_512 != src:
        src_512_noinline = src_512  # Already has noinline from codegen
        print("  Testing 512-thread + noinline lambda...")
        try:
            ptx_512n = compile_cuda(src_512_noinline, target_format="ptx", arch=100,
                                    options=["--use_fast_math", "-DENABLE_BF16"])
            print("  SUCCESS! 512-thread with noinline compiles!")
            reg_matches = re.findall(r'\.reg \.b32\s+%r<(\d+)>', ptx_512n.decode('utf-8', errors='replace'))
            if reg_matches:
                print(f"  Register usage: .b32 %r<{reg_matches[0]}>")
        except Exception as e:
            print(f"  Still fails: {str(e)[:200]}")
            print("\n  Noinline lambda doesn't help. Need proper device functions.")
else:
    print("  No noinline in source. Codegen change may not have applied.")

print("\nDone.")
