#!/usr/bin/env python3
"""Validate bundled TileLang/PTO references without creating bytecode files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
REFERENCES = SKILL_ROOT / "references"


def validate_static() -> None:
    failures: list[str] = []
    old_name = "tilelang-pto-" + "performance-best-practices"
    stale_reference = "ref" + "erence/"

    skill_text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    if "name: tilelang-performance-best-practices" not in skill_text:
        failures.append("SKILL.md name does not match the integration name")

    for path in SKILL_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".json", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8")
        try:
            if path.suffix == ".py":
                compile(text, str(path), "exec")
        except SyntaxError as error:
            failures.append(f"{path.relative_to(SKILL_ROOT)}: {error}")
        if old_name in text:
            failures.append(f"{path.relative_to(SKILL_ROOT)}: stale skill name")
        if stale_reference in text:
            failures.append(f"{path.relative_to(SKILL_ROOT)}: stale singular references path")

    sys.path.insert(0, str(REFERENCES))
    from broadcast.code.broadcast_add_tiling import select_tiling as select_broadcast
    from reduce.templates.euclidean_norm_tiling import select_tiling as select_norm

    if select_broadcast(3, 129, 2, 64).tile_cols != 256:
        failures.append("broadcast cols=129 must use a 256-element padded footprint")
    if select_broadcast(2, 257, 2, 64).tile_cols != 512:
        failures.append("broadcast cols=257 must use a 512-element padded footprint")
    if select_norm(65, 65, 64).tile_cols != 128:
        failures.append("euclidean norm cols=65 must use a 128-element padded footprint")
    if select_norm(2, 257, 64).tile_cols != 512:
        failures.append("euclidean norm cols=257 must use next-power-of-two padding")
    if select_norm(2, 4097, 64).strategy != "split_r_design_only":
        failures.append("R > 4096 must not be classified as executable full-load")

    if failures:
        raise SystemExit("static validation failed:\n- " + "\n- ".join(failures))
    print("PASS static skill and tiling validation")


def validate_npu(num_cores: int) -> None:
    if os.getenv("TILELANG_DEFAULT_TARGET") != "pto":
        raise SystemExit("set TILELANG_DEFAULT_TARGET=pto before running --npu")

    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("Ascend NPU is not available")

    sys.path.insert(0, str(REFERENCES))
    from broadcast.code.broadcast_add_kernel import build as build_broadcast
    from reduce.templates.dav310.kernel_utils import euclidean_norm, softmax_full_load
    from rope.code.rope_vf_common import build as build_rope
    from rope.code.rope_vf_common import reference as rope_reference
    from scan.templates.dav310.scan_base import build as build_scan

    def check(name, actual, expected, *, rtol=1e-3, atol=1e-3):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol)
        print(f"PASS {name}")

    for rows, cols in ((3, 129), (2, 257)):
        x = torch.randn((rows, cols), device="npu", dtype=torch.float16)
        bias = torch.randn((rows, 1), device="npu", dtype=torch.float16)
        check(f"broadcast_{rows}x{cols}", build_broadcast(rows, cols, "float16", num_cores=num_cores)(x, bias), x + bias, rtol=0, atol=0)

    for rows, cols in ((65, 65), (2, 129), (2, 257)):
        x = torch.randn((rows, cols), device="npu", dtype=torch.float16)
        expected = torch.sqrt(torch.sum(x.float() * x.float(), dim=1))
        check(f"euclidean_norm_{rows}x{cols}", euclidean_norm(rows, cols, "float16", num_cores=num_cores)(x), expected, rtol=1e-4, atol=1e-4)

    x = torch.randn((2, 257), device="npu", dtype=torch.float16)
    check("softmax_2x257", softmax_full_load(2, 257, "float16", num_cores=num_cores)(x), torch.softmax(x, dim=-1), rtol=2e-3, atol=2e-3)

    x = torch.randn((2, 257), device="npu", dtype=torch.bfloat16)
    check("scan_2x257", build_scan(2, 257, num_cores=num_cores)(x), x.float().cumsum(dim=-1), rtol=2e-3, atol=2e-3)

    x = torch.randn((2, 2, 128), device="npu", dtype=torch.bfloat16)
    cos = torch.randn((2, 64), device="npu", dtype=torch.float32)
    sin = torch.randn((2, 64), device="npu", dtype=torch.float32)
    check("rope_2x2x128", build_rope(2, 2, 128, num_cores=num_cores)(x, cos, sin), rope_reference(x, cos, sin), rtol=2e-2, atol=2e-2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", action="store_true", help="also compile and run representative PTO kernels on NPU")
    parser.add_argument("--num-cores", type=int, help="available AIV cores confirmed by hardware discovery; required with --npu")
    args = parser.parse_args()
    if args.npu and (args.num_cores is None or args.num_cores <= 0):
        parser.error("--npu requires a positive --num-cores from target hardware discovery")
    validate_static()
    if args.npu:
        validate_npu(args.num_cores)


if __name__ == "__main__":
    main()
