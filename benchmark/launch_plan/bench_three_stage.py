"""Same-process three-stage invariant arithmetic benchmark."""

import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch
import tilelang
import tilelang.language as T


def decode_layout(size, gather, constraints="none", parameter_dtype="int32"):
    storage = T.dynamic("storage")
    dtype = "float32" if gather else "int32"

    @T.prim_func
    def main(
        A: T.Tensor((storage,), "float32"),
        B: T.Tensor((size,), dtype),
        h: T.dtype(parameter_dtype),
        w: T.dtype(parameter_dtype),
        c: T.dtype(parameter_dtype),
        pitch: T.dtype(parameter_dtype),
    ):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
            if constraints == "nonzero":
                T.assume(h != 0)
                T.assume(w != 0)
                T.assume(c != 0)
            if constraints == "positive" or constraints == "bounded" or constraints == "swizzle":
                T.assume(h > 0)
                T.assume(w > 0)
                T.assume(c > 0)
            if constraints == "bounded" or constraints == "swizzle":
                T.assume(h <= 1024)
                T.assume(w <= 1024)
                T.assume(c <= 1024)
            if constraints == "swizzle":
                T.assume(c >= 32)
            height = T.min(T.max(h, 1), 1024) if constraints == "clamped" else h
            width = T.min(T.max(w, 1), 1024) if constraints == "clamped" else w
            channels = T.min(T.max(c, 32), 1024) if constraints == "clamped" else c
            for lane in T.Parallel(128):
                i = bx * 128 + lane
                index = T.int64(i) if parameter_dtype == "uint32" else i
                channel = index % channels
                column = (index // channels) % width
                row = (index // (width * channels)) % height
                batch = index // (height * width * channels)
                swizzled = (channel ^ ((column % 8) * 4)) % channels
                offset = (batch * height + row) * pitch + column * channels + swizzled
                if gather:
                    B[i] = A[offset]
                else:
                    B[i] = offset

    return main


def banked_layout(size):
    @T.prim_func
    def main(B: T.Tensor((size,), "uint32"), columns: T.int32, banks: T.uint32, rotation: T.uint32):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
            for lane in T.Parallel(128):
                i = bx * 128 + lane
                row = i // columns
                column = i % columns
                bank = (T.cast(column, "uint32") + T.cast(row, "uint32") * rotation) % banks
                B[i] = T.cast(row, "uint32") * banks + bank

    return main


def aligned_gather(size):
    storage = T.dynamic("storage")

    @T.prim_func
    def main(
        A: T.Tensor((storage,), "float32"),
        Offsets: T.Tensor((size,), "int32"),
        B: T.Tensor((size,), "float32"),
        element_bytes: T.int32,
        width: T.int32,
        pitch: T.int32,
    ):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
            for lane in T.Parallel(128):
                i = bx * 128 + lane
                byte_offset = T.bind(Offsets[i])
                if byte_offset % element_bytes == 0:
                    element = byte_offset // element_bytes
                    row = element // width
                    column = element % width
                    B[i] = A[row * pitch + column]
                else:
                    B[i] = -1.0

    return main


def wide_layout(size, remainder_only):
    @T.prim_func
    def main(B: T.Tensor((size,), "int64"), divisor: T.int64, pitch: T.int64):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
            for lane in T.Parallel(128):
                i = bx * 128 + lane
                offset = T.int64(i) * T.int64(1000003) + T.int64(2**48)
                if remainder_only:
                    B[i] = offset % divisor
                else:
                    B[i] = offset // divisor * pitch + offset % divisor

    return main


def workload(case, size):
    if case in ("gather_nonzero", "gather_positive", "gather_bounded", "gather_unsigned", "gather_swizzle", "gather_clamped"):
        _, inputs, out, expected = workload("gather", size)
        constraint = case.removeprefix("gather_")
        func = decode_layout(
            size, True, "nonzero" if constraint == "unsigned" else constraint, "uint32" if constraint == "unsigned" else "int32"
        )
        return func, inputs, out, expected
    i = torch.arange(size, dtype=torch.int64, device="cuda")
    if case in ("decode", "gather"):
        h, w, c = 13, 29, 37
        pitch = w * c + 19
        batch = i // (h * w * c)
        row, column, channel = (i // (w * c)) % h, (i // c) % w, i % c
        offsets = (batch * h + row) * pitch + column * c + ((channel ^ ((column % 8) * 4)) % c)
        storage = ((size + h * w * c - 1) // (h * w * c)) * h * pitch
        a = (torch.arange(storage, device="cuda") % 1024).float()
        expected = a[offsets] if case == "gather" else offsets.to(torch.int32)
        b = torch.empty_like(expected)
        return decode_layout(size, case == "gather"), (a, b, h, w, c, pitch), b, expected
    if case == "banked":
        columns, banks, rotation = 37, 113, 0x9E3779B1
        row, column = i // columns, i % columns
        bank = ((column + row * rotation) & 0xFFFFFFFF) % banks
        expected = (row * banks + bank).to(torch.uint32)
        b = torch.empty_like(expected)
        return banked_layout(size), (b, columns, banks, rotation), b, expected
    if case == "aligned_gather":
        element_bytes, width, pitch = 12, 37, 53
        elements = (i * 17) % size
        offsets = elements * element_bytes
        offsets[::17] += 1
        storage = ((size + width - 1) // width) * pitch
        a = (torch.arange(storage, device="cuda") % 1024).float()
        addresses = (elements // width) * pitch + elements % width
        expected = torch.where(offsets % element_bytes == 0, a[addresses], -1.0)
        b = torch.empty_like(expected)
        return aligned_gather(size), (a, offsets.int(), b, element_bytes, width, pitch), b, expected
    if case in ("wide_layout", "wide_remainder"):
        divisor, pitch = 2**32 + 15, 67
        offsets = i * 1000003 + 2**48
        expected = offsets % divisor
        if case == "wide_layout":
            expected = offsets // divisor * pitch + expected
        b = torch.empty_like(expected)
        return wide_layout(size, case == "wide_remainder"), (b, divisor, pitch), b, expected
    raise ValueError(case)


def capture(call, output, expected, count):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(count):
            call()
    # An empty/wrong-stream graph must fail, rather than yield a false speedup.
    output.fill_(123456)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.to(torch.float64 if output.is_floating_point() else torch.int64),
        expected.to(torch.float64 if expected.is_floating_point() else torch.int64),
        rtol=0,
        atol=0,
    )
    return graph


def measure(graph, call, count):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        graph.replay()
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    gpu_us = start.elapsed_time(end) * 1000 / (count * 10)
    torch.cuda.synchronize()
    begin = time.perf_counter_ns()
    for _ in range(count):
        call()
    torch.cuda.synchronize()
    wall_us = (time.perf_counter_ns() - begin) / (count * 1000)
    return gpu_us, wall_us


def stress_kernel(size, depth):
    @T.prim_func
    def main(A: T.Tensor((size,), "int32"), B: T.Tensor((size,), "int32"), d: T.int32, e: T.int32):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
            for lane in T.Parallel(128):
                idx = bx * 128 + lane
                v = T.alloc_local((1,), "int32")
                v[0] = A[idx]
                for _step in T.unroll(depth):
                    v[0] = (v[0] // d) + (v[0] % e)
                B[idx] = v[0]

    return main


def narrow_divisor_kernel(size, divisor_dtype, remainder):
    @T.prim_func
    def main(B: T.Tensor((size,), "int64"), d: T.dtype(divisor_dtype)):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
            for lane in T.Parallel(128):
                x = (T.int64(bx) * 128 + T.int64(lane)) * 1000003 + T.int64(2**48)
                if remainder:
                    B[bx * 128 + lane] = x % d
                else:
                    B[bx * 128 + lane] = x // d * 67 + x % d

    return main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 1048576])
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path(".cache/launch_plan/three_stage"))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    lower_invariant = tilelang.transform.LowerInvariantArithmetic
    results = []
    cases = [
        "decode",
        "gather",
        "aligned_gather",
        "banked",
        "wide_layout",
        "wide_remainder",
        "wide_i32_layout",
        "wide_i32_remainder",
        "stress6",
        "gather_nonzero",
        "gather_positive",
        "gather_bounded",
        "gather_unsigned",
        "gather_swizzle",
        "gather_clamped",
    ]
    for size in args.sizes:
        for case in cases:
            if case == "stress6":
                torch.manual_seed(3261)
                a = torch.randint(-1000000, 1000000, (size,), device="cuda", dtype=torch.int32)
                out = torch.empty_like(a)
                expected = a.clone()
                for _ in range(6):
                    expected = expected // 7 + expected % 13
                func = stress_kernel(size, 6)
                inputs = (a, out, 7, 13)
            elif case.startswith("wide_i32"):
                remainder = case.endswith("remainder")
                x = torch.arange(size, device="cuda", dtype=torch.int64) * 1000003 + 2**48
                out = torch.empty_like(x)
                expected = x % 37 if remainder else x // 37 * 67 + x % 37
                func = narrow_divisor_kernel(size, "int32", remainder)
                inputs = (out, 37)
            else:
                func, inputs, out, expected = workload(case, size)
            variants = []
            hosts = {}
            for mode in ["off", "lower", "late"]:

                def lower_stage(stage="prepare", mode=mode):
                    if stage == "materialize" and mode != "late":
                        return lambda mod: mod
                    return lower_invariant(stage=stage)

                tilelang.transform.LowerInvariantArithmetic = lower_stage
                begin = time.perf_counter()
                try:
                    k = tilelang.compile(
                        func,
                        target="cuda",
                        target_host="c",
                        execution_backend="tvm_ffi",
                        pass_configs={"tl.enable_invariant_arithmetic": mode != "off"},
                    )
                finally:
                    tilelang.transform.LowerInvariantArithmetic = lower_invariant
                compile_s = time.perf_counter() - begin
                stem = f"{case}_{size}_{mode}"
                source = k.get_kernel_source()
                hosts[mode] = k.get_host_source()
                (root / (stem + ".cu")).write_text(source)
                (root / (stem + ".host.txt")).write_text(hosts[mode])

                def call(k=k, inputs=inputs):
                    k(*inputs)

                call()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                graph = capture(call, out, expected, args.count)
                variants.append((mode, k, call, graph, compile_s, len(source.encode())))
            assert hosts["lower"] == hosts["late"]
            # Warm every variant before collecting the interleaved samples.
            for _, _, _, graph, _, _ in variants:
                for _ in range(10):
                    graph.replay()
            torch.cuda.synchronize()
            samples = {mode: [] for mode, *_ in variants}
            rng = random.Random(3261)
            for _ in range(args.rounds):
                order = list(variants)
                rng.shuffle(order)
                for mode, _, call, graph, _, _ in order:
                    samples[mode].append(measure(graph, call, args.count))
            for mode, _, _, _, compile_s, source_bytes in variants:
                rows = samples[mode]
                row = dict(
                    case=case,
                    size=size,
                    mode=mode,
                    gpu_us=statistics.median(x[0] for x in rows),
                    wall_us=statistics.median(x[1] for x in rows),
                    samples=rows,
                    compile_s=compile_s,
                    cuda_bytes=source_bytes,
                )
                results.append(row)
                print({k: v for k, v in row.items() if k != "samples"}, flush=True)
            (root / "results.json").write_text(
                json.dumps(
                    dict(device=torch.cuda.get_device_name(), rounds=args.rounds, count=args.count, sizes=args.sizes, results=results),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
