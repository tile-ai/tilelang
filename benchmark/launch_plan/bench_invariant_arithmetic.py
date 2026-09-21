"""End-to-end layout workloads for LowerInvariantArithmetic."""

import argparse
import json
import statistics
import time
from pathlib import Path

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


def make_call(kernel, inputs):
    return lambda: kernel(*inputs)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "decode",
            "gather",
            "banked",
            "aligned_gather",
            "wide_layout",
            "wide_remainder",
            "gather_nonzero",
            "gather_positive",
            "gather_bounded",
            "gather_unsigned",
        ],
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 1048576])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path(".cache/launch_plan/invariant_layout_bench.json"))
    args = parser.parse_args()
    results = []
    source_dir = args.output.with_suffix("")
    source_dir.mkdir(parents=True, exist_ok=True)
    backend = "tvm_ffi"
    for case in args.cases:
        for size in args.sizes:
            func, inputs, output, expected = workload(case, size)
            variants = []
            for enabled in (False, True):
                kernel = tilelang.compile(
                    func,
                    target="cuda",
                    target_host="c",
                    execution_backend=backend,
                    pass_configs={"tl.enable_invariant_arithmetic": enabled},
                )
                call = make_call(kernel, inputs)
                call()
                torch.testing.assert_close(
                    output.to(torch.float64 if output.is_floating_point() else torch.int64),
                    expected.to(torch.float64 if expected.is_floating_point() else torch.int64),
                    rtol=0,
                    atol=0,
                )
                stem = f"{backend}_{case}_{size}_{int(enabled)}"
                source = kernel.get_kernel_source()
                if enabled:
                    assert "fastdiv_multiplier" in source or "barrett_reciprocal" in source
                    if case == "aligned_gather":
                        assert "exact_inverse" in source
                (source_dir / f"{stem}.cu").write_text(source)
                host = kernel.get_host_source()
                if host:
                    (source_dir / f"{stem}.host.txt").write_text(host)
                graph = capture(call, output, expected, args.count)
                variants.append((kernel, call, graph))
            samples = {False: [], True: []}
            for round_id in range(args.rounds):
                for enabled in (False, True) if round_id % 2 == 0 else (True, False):
                    _, call, graph = variants[int(enabled)]
                    samples[enabled].append(measure(graph, call, args.count))
            rows = []
            for enabled in (False, True):
                row = dict(
                    backend=backend,
                    case=case,
                    size=size,
                    enabled=enabled,
                    gpu_us=statistics.median(s[0] for s in samples[enabled]),
                    amortized_wall_us=statistics.median(s[1] for s in samples[enabled]),
                    samples=samples[enabled],
                )
                rows.append(row)
                results.append(row)
            print(
                f"{backend:7} {case:14} {size:8}: GPU {rows[0]['gpu_us']:.3f} -> {rows[1]['gpu_us']:.3f} us, "
                f"wall {rows[0]['amortized_wall_us']:.3f} -> {rows[1]['amortized_wall_us']:.3f} us",
                flush=True,
            )
            args.output.write_text(
                json.dumps(dict(device=torch.cuda.get_device_name(), rounds=args.rounds, count=args.count, results=results), indent=2)
            )


if __name__ == "__main__":
    main()
