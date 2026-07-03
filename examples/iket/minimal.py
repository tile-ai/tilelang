"""Minimal IKET example using TileLang frontend APIs on the CUDA backend.

Run with:

  conda run -n tl python -m iket.cli.main --output-dir /tmp/tl_iket_cuda \
    --clobber profile --postprocess all -- \
    conda run -n tl python examples/iket/minimal.py

This uses TileLang's regular ``target="cuda"`` backend. The kernel-level IKET
events are written with the ``T.iket`` frontend namespace, not CuTeDSL.
"""

import json
import os
import argparse
from pathlib import Path

import torch

import tilelang
import tilelang.language as T


N = 1024
THREADS = 128


def elementwise_add_with_iket(n: int, threads: int = THREADS, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype),
        B: T.Tensor((n,), dtype),
        C: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx, T.iket.range("kernel_total"):
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < n:
                    T.iket.mark("before_store")
                    C[idx] = A[idx] + B[idx]
                    T.iket.mark("after_store")

    return main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iket-output-dir",
        default=os.environ.get("TL_IKET_OUTPUT_DIR", "/tmp/tilelang_iket_frontend"),
        help="Directory used by this example for IKET trace/source exports.",
    )
    args = parser.parse_args()

    torch.cuda.init()
    with T.iket.session(output_dir=args.iket_output_dir):
        program = elementwise_add_with_iket(N)
        print(f"event_table={T.iket.event_table()}")
        print(f"iket_output_dir={T.iket.output_dir()}")
        tilelang.disable_cache()
        try:
            kernel = tilelang.compile(
                program,
                out_idx=-1,
                target="cuda",
                execution_backend="cython",
            )
        finally:
            tilelang.enable_cache()

    a = torch.arange(N, device="cuda", dtype=torch.float32)
    b = torch.full((N,), 2.0, device="cuda", dtype=torch.float32)
    c = kernel(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b)

    source = kernel.get_kernel_source()
    print("tilelang cuda backend run ok")
    print("frontend_api=True")
    print(f"instrumented={'__iket_meta_info' in source and 'TL_IKET_EVENT' in source}")
    print(f"source_len={len(source)}")
    print(f"first_values={c[:4].detach().cpu().tolist()}")

    source_out = Path(
        os.environ.get(
            "TL_IKET_SOURCE_OUT",
            str(T.iket.output_path("tilelang_iket_frontend_kernel.cu", directory=args.iket_output_dir)),
        )
    )
    source_out.write_text(source)
    print(f"source_out={source_out}")

    traces = T.iket.trace_files(directory=args.iket_output_dir)
    if traces:
        with traces[0].open() as f:
            data = json.load(f)
        launches = data.get("launches", [])
        markers = sum(len(launch.get("markers", [])) for launch in launches)
        ranges = sum(len(launch.get("ranges", [])) for launch in launches)
        print(f"trace_summary=launches:{len(launches)} markers:{markers} ranges:{ranges}")


if __name__ == "__main__":
    main()
