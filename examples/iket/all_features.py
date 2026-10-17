"""Comprehensive TileLang CUDA IKET payload example.

This example covers the current TileLang IKET feature set:

* iket.session(output_dir=...)
* iket.range(...)
* iket.mark(...)
* iket.payload(...) runtime scalar values
"""

import argparse
import os

import torch

import tilelang
import tilelang.language as T
from tilelang.tools.cuda import iket


N = 2048
THREADS = 128


def add_scale_with_iket(n: int, threads: int = THREADS, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype),
        B: T.Tensor((n,), dtype),
        Scale: T.Tensor((1,), dtype),
        C: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx, iket.range("block_total"):
            iket.mark("block_enter", payload=iket.payload(bx, dtype="int32"))
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < n:
                    iket.mark("load_inputs")
                    value = (A[idx] + B[idx]) * Scale[0]
                    iket.mark("store_index", payload=iket.payload(idx, dtype="int32"))
                    C[idx] = value
                    iket.mark("store_done")
            iket.mark("block_exit", payload=iket.payload(bx, dtype="int32"))

    return main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iket-output-dir",
        default=os.environ.get("TL_IKET_OUTPUT_DIR", "/tmp/tilelang_iket_all_features"),
    )
    parser.add_argument(
        "--iket-runtime-payloads",
        action="store_true",
        help="Emit non-NoPayload IKET metadata and capture runtime payload values.",
    )
    args = parser.parse_args()

    torch.cuda.init()
    with iket.session(output_dir=args.iket_output_dir, runtime_payloads=args.iket_runtime_payloads):
        program = add_scale_with_iket(N)
        print(f"event_table={iket.event_table()}")
        print(f"iket_output_dir={iket.output_dir()}")
        print(f"runtime_payloads_enabled={iket.runtime_payloads_enabled()}")
        kernel = tilelang.compile(
            program,
            out_idx=-1,
            target="cuda",
            execution_backend="cython",
        )

    a = torch.arange(N, device="cuda", dtype=torch.float32)
    b = torch.full((N,), 3.0, device="cuda", dtype=torch.float32)
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    c = kernel(a, b, scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, (a + b) * scale[0])

    source = kernel.get_kernel_source()
    source_out = iket.output_path("tilelang_iket_all_features_kernel.cu", directory=args.iket_output_dir)
    source_out.write_text(source)

    print("tilelang cuda backend payload run ok")
    print(f"instrumented={'__iket_meta_info' in source and 'TL_IKET_EVENT' in source}")
    print("payload_api_present=True")
    print(f"runtime_payload_metadata={args.iket_runtime_payloads}")
    print(f"source_out={source_out}")
    print(f"first_values={c[:4].detach().cpu().tolist()}")


if __name__ == "__main__":
    main()
