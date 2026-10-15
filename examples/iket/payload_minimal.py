"""Minimal TileLang CUDA IKET runtime payload example."""

import argparse
import os

import torch

import tilelang
import tilelang.language as T
from tilelang.tools.cuda import iket


N = 128
THREADS = 128


def payload_mark_kernel(n: int, threads: int = THREADS, dtype=T.float32):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < n:
                    iket.mark("store_index", payload=iket.payload(idx, dtype="int32"))
                    C[idx] = A[idx] + 1.0

    return main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iket-output-dir",
        default=os.environ.get("TL_IKET_OUTPUT_DIR", "/tmp/tilelang_iket_payload_minimal"),
    )
    parser.add_argument("--iket-runtime-payloads", action="store_true")
    args = parser.parse_args()

    torch.cuda.init()
    with iket.session(output_dir=args.iket_output_dir, runtime_payloads=args.iket_runtime_payloads):
        program = payload_mark_kernel(N)
        print(f"event_table={iket.event_table()}")
        print(f"runtime_payloads_enabled={iket.runtime_payloads_enabled()}")
        kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")

    a = torch.arange(N, device="cuda", dtype=torch.float32)
    c = kernel(a)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + 1.0)

    source = kernel.get_kernel_source()
    source_out = iket.output_path("tilelang_iket_payload_minimal_kernel.cu", directory=args.iket_output_dir)
    source_out.write_text(source)
    print("tilelang cuda backend minimal payload run ok")
    print(f"source_out={source_out}")
    print(f"first_values={c[:4].detach().cpu().tolist()}")


if __name__ == "__main__":
    main()
