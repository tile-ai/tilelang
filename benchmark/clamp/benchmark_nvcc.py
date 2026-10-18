import ctypes
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import argparse
import subprocess
import shutil

import torch

parser = argparse.ArgumentParser(description="Compare the current FP32 clamp against the frozen old CUDA helper.")
parser.add_argument("--output", type=Path, default=Path("clamp-nvcc-results"))
args = parser.parse_args()
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parent
repo = source.parents[1]
cuda_math_sha256 = hashlib.sha256((repo / "src/tl_templates/cuda/math.h").read_bytes()).hexdigest()
major, minor = torch.cuda.get_device_capability()
nvcc = shutil.which("nvcc")
if nvcc is None:
    parser.error("nvcc must be on PATH")
commands = []
for mode in ["default", "fast"]:
    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-arch=sm_{major}{minor}",
        "-I",
        str(repo / "src/tl_templates"),
        "-I",
        str(repo / "3rdparty/cutlass/include"),
        str(source / "clamp.cu"),
        "-o",
        str(root / f"probe-normalized-{mode}.so"),
    ]
    if mode == "fast":
        command.append("--use_fast_math")
    commands.append(command)
    with (root / f"build-{mode}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
torch.manual_seed(0)
torch.cuda.init()
stream = torch.cuda.current_stream().cuda_stream
ptr = ctypes.c_void_p
i32 = ctypes.c_int
f32 = ctypes.c_float
u64 = ctypes.c_uint64


def library(name):
    lib = ctypes.CDLL(str(root / name))
    lib.values.argtypes = [i32, ptr, ptr, ptr, ptr, i32, i32, u64]
    lib.chains.argtypes = [i32, i32, ptr, ptr, i32, i32, f32, f32, f32, f32, u64]
    lib.fused.argtypes = [i32, ptr, ptr, ptr, ptr, ptr, i32, f32, u64]
    return lib


def check(result):
    assert result == 0, result


bits = [
    0,
    0x80000000,
    1,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00800000,
    0x80800000,
    0x3F000000,
    0xBF000000,
    0x3F800000,
    0xBF800000,
    0x40000000,
    0xC0000000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC01234,
    0x7F800001,
    0xFF800001,
]
triples = torch.tensor(list(itertools.product(bits, repeat=3)), dtype=torch.int64).to(torch.int32).view(torch.float32)
x, lo, hi = [column.contiguous().cuda() for column in triples.unbind(1)]
records = []
for mode in ["default", "fast"]:
    lib = library(f"probe-normalized-{mode}.so")
    for constants in [0, 1]:
        outputs = [torch.empty_like(x) for _ in range(2)]
        for native, out in enumerate(outputs):
            check(lib.values(native, x.data_ptr(), lo.data_ptr(), hi.data_ptr(), out.data_ptr(), x.numel(), constants, stream))
        torch.cuda.synchronize()
        a, b = (t.cpu() for t in outputs)
        mismatch = (torch.isnan(a) != torch.isnan(b)) | (~torch.isnan(a) & (a.view(torch.int32) != b.view(torch.int32)))
        indices = mismatch.nonzero().flatten().tolist()
        record = dict(mode=mode, constants=bool(constants), cases=x.numel(), mismatches=len(indices))
        record["examples"] = [
            dict(
                inputs=[hex(int(v) & 0xFFFFFFFF) for v in triples[j].view(torch.int32)],
                baseline=hex(int(a.view(torch.int32)[j]) & 0xFFFFFFFF),
                candidate=hex(int(b.view(torch.int32)[j]) & 0xFFFFFFFF),
            )
            for j in indices[:4]
        ]
        records.append(record)
        assert not indices, record


def measure_pair(call):
    samples = [[], []]
    for native in [0, 1]:
        for _ in range(5):
            call(native)
    torch.cuda.synchronize()
    for repetition in range(15):
        for native in [0, 1] if repetition % 2 == 0 else [1, 0]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                call(native)
            end.record()
            end.synchronize()
            samples[native].append(start.elapsed_time(end) / 5)
    medians = [statistics.median(s) for s in samples]
    return dict(baseline_ms=medians[0], candidate_ms=medians[1], speedup=medians[0] / medians[1], samples=samples)


benchmarks = []
for mode in ["default", "fast"]:
    lib = library(f"probe-normalized-{mode}.so")
    blocks = torch.cuda.get_device_properties(0).multi_processor_count * 4
    for count in [1, 8]:
        a = torch.rand(blocks * 256 * count, device="cuda") * 2 - 1
        outputs = [torch.empty_like(a) for _ in range(2)]

        def call(native, lib=lib, count=count, a=a, outputs=outputs, blocks=blocks):
            check(lib.chains(native, count, a.data_ptr(), outputs[native].data_ptr(), blocks, 512, -1.0, 1.0, 1.00001, 0.00001, stream))

        result = measure_pair(call)
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        benchmarks.append(dict(mode=mode, kind=f"chains-{count}", blocks=blocks, iterations=512, **result))
    for n in [1 << 20, 1 << 24]:
        a, b = [torch.randn(n, device="cuda") for _ in range(2)]
        lo, hi = torch.full_like(a, -1.0), torch.full_like(a, 1.0)
        outputs = [torch.empty_like(a) for _ in range(2)]

        def call(native, lib=lib, a=a, b=b, lo=lo, hi=hi, outputs=outputs, n=n):
            check(lib.fused(native, a.data_ptr(), b.data_ptr(), lo.data_ptr(), hi.data_ptr(), outputs[native].data_ptr(), n, 0.75, stream))

        result = measure_pair(call)
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        benchmarks.append(dict(mode=mode, kind="fused-affine", elements=n, **result))
result = dict(
    device=torch.cuda.get_device_name(),
    torch=torch.__version__,
    cuda_math_sha256=cuda_math_sha256,
    nvcc=subprocess.check_output([nvcc, "--version"], text=True),
    commands=commands,
    semantics=records,
    benchmarks=benchmarks,
)
(root / "result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
