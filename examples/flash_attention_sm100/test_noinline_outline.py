#!/usr/bin/env python3
"""Test per-warp __noinline__ device function outlining.

Takes the TileLang-generated CUDA, restructures the math WG bodies into
separate __device__ __noinline__ functions, compiles with nvcc, and benchmarks.
"""
import sys, os, re, subprocess, tempfile, ctypes
sys.path.insert(0, '.')

import torch
from tilelang.profiler import do_bench
from attention_kernel_1sm import attention_kernel_1sm

# Step 1: Get generated CUDA source
print("Generating CUDA source...")
fn = attention_kernel_1sm(8, 40, 4096, 128, num_kv_heads=8, kq_stages=2)
src = fn.get_kernel_source()
print(f"  Generated: {len(src)} chars, {src.count(chr(10))} lines")

# Step 2: Find the include paths TileLang uses
tl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Should be ~/workspace/tilelang
if not os.path.exists(os.path.join(tl_root, 'src', 'tl_templates')):
    tl_root = os.path.expanduser('~/workspace/tilelang')
tl_templates = os.path.join(tl_root, 'src', 'tl_templates')
tl_3rdparty = os.path.join(tl_root, '3rdparty')
cutlass_inc = os.path.join(tl_3rdparty, 'cutlass', 'include')

print(f"  TL templates: {tl_templates}")
print(f"  CUTLASS: {cutlass_inc}")

# Step 3: Compile the UNMODIFIED source first to verify the compile pipeline works.
# Must be CUDA 13+ for sm_100a tcgen05.* + 3-operand max.ftz.f32; /usr/local/cuda
# on this box points at 12.8 which can't assemble the generated PTX.
cuda_path = '/usr/local/cuda-13.1'
nvcc = f'{cuda_path}/bin/nvcc'

with tempfile.NamedTemporaryFile(suffix='.cu', mode='w', delete=False) as f:
    f.write(src)
    cu_file = f.name

so_file = cu_file.replace('.cu', '.so')
tl_src = os.path.join(tl_root, 'src')  # parent of tl_templates/
cmd = [
    nvcc, cu_file, '-o', so_file,
    '--shared', '-Xcompiler', '-fPIC',
    # `-arch=sm_100a` alone lowers to compute_100 PTX, which ptxas then rejects
    # for tcgen05.* / 3-op max.ftz.f32. Use the explicit gencode form so the
    # virtual arch is sm_100a (where those instructions are legal).
    '-gencode', 'arch=compute_100a,code=sm_100a',
    '-std=c++17',
    f'-I{tl_src}',
    f'-I{cutlass_inc}',
    f'-I{cuda_path}/include',
    '-DENABLE_BF16',
    '-O3',
    '--use_fast_math',
    '-w',  # suppress warnings
    '-Xptxas', '-v',  # verbose to see register usage
]

print(f"\nCompiling unmodified kernel...")
print(f"  {' '.join(cmd[:5])}...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
if result.returncode != 0:
    print(f"  FAILED: {result.stderr[:500]}")
    # Try to fix includes
    print(f"\nFull stderr:\n{result.stderr[:2000]}")
else:
    # Extract register info from ptxas verbose output
    for line in result.stderr.split('\n'):
        if 'register' in line.lower() or 'spill' in line.lower() or 'smem' in line.lower():
            print(f"  ptxas: {line.strip()}")
    print(f"  SUCCESS: {so_file}")

# Cleanup
os.unlink(cu_file)
if os.path.exists(so_file):
    os.unlink(so_file)
