import re
import subprocess
import sys
import traceback
from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_row_major
from tilelang.utils.target import determine_target

from testing.python.sunmmio.common.codegen_validation import assert_source_contains, find_npuir_opt
from testing.python.sunmmio.common.compile_pipeline import compile_test, target


tilelang.env.disable_cache()

LOG_ROOT = Path(__file__).resolve().parent / "logs"
STRICT_OPT_ARGS = ("--verify-each", "--suvm-to-llvm-pipeline")
MODULE_NAME = "testing.python.sunmmio.codegen.test_softplus_1d_dynamic_opt_validate"
NPU_IR_BF16_RANGE_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "NPU-IR's SUVM-to-LLVM pipeline cannot currently legalize the suvm.tile.addi introduced when splitting a 512-element i16 range"
    ),
)


@target("Sunmmio")
def softplus_1d_dynamic(block_N=256, in_dtype=T.float32, out_dtype=T.float32):
    """Original form: bind the RSRAM load to a scalar before using it."""
    N = T.dynamic("n")

    row_major_layout = make_row_major((N,))
    placement = T.placement.mesh_as_line(0)

    @T.prim_func
    def elem_softplus(
        A: T.MeshTensor((N,), placement, dtype=in_dtype, layout=row_major_layout),  # type: ignore
        B: T.MeshTensor((N,), placement, dtype=out_dtype, layout=row_major_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_N,), in_dtype)
            B_shared = T.alloc_shared((block_N,), out_dtype)

            for bx in T.serial(T.ceildiv(A.get_local_extent()[0], block_N)):
                T.copy(A[bx * block_N : (bx + 1) * block_N], A_shared)
                for i in T.Tiles([T.min(block_N, A.get_local_extent()[0] - bx * block_N)]):
                    value = A_shared[i]
                    B_shared[i] = T.max(value, 0) + T.log(1 + T.exp(-T.abs(value)))
                T.copy(B_shared, B[bx * block_N : (bx + 1) * block_N])

    return elem_softplus


@target("Sunmmio")
def softplus_1d_dynamic_inline(block_N=256, in_dtype=T.float32, out_dtype=T.float32):
    """Inline form: repeat the same RSRAM load expression at each use site."""
    N = T.dynamic("n")

    row_major_layout = make_row_major((N,))
    placement = T.placement.mesh_as_line(0)

    @T.prim_func
    def elem_softplus_inline(
        A: T.MeshTensor((N,), placement, dtype=in_dtype, layout=row_major_layout),  # type: ignore
        B: T.MeshTensor((N,), placement, dtype=out_dtype, layout=row_major_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_N,), in_dtype)
            B_shared = T.alloc_shared((block_N,), out_dtype)

            for bx in T.serial(T.ceildiv(A.get_local_extent()[0], block_N)):
                T.copy(A[bx * block_N : (bx + 1) * block_N], A_shared)
                for i in T.Tiles([T.min(block_N, A.get_local_extent()[0] - bx * block_N)]):
                    B_shared[i] = T.max(A_shared[i], 0) + T.log(1 + T.exp(-T.abs(A_shared[i])))
                T.copy(B_shared, B[bx * block_N : (bx + 1) * block_N])

    return elem_softplus_inline


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _prepare_log_dir(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    for path in log_dir.iterdir():
        if path.is_file():
            path.unlink()


def _write_exception(log_dir: Path, stage: str) -> None:
    _write_text(log_dir / f"{stage}_error.log", traceback.format_exc())


def _record_stage(log_dir: Path, stage: str) -> None:
    _write_text(log_dir / "stage_status.txt", f"current stage: {stage}\n")


def _assert_tile_domain_vars_are_bound(device_mod) -> None:
    for func in device_mod.functions.values():
        if not isinstance(func, tvm.tir.PrimFunc):
            continue

        defined_vars = list(func.params)
        tile_domains = []

        def collect(node, defined_vars=defined_vars, tile_domains=tile_domains) -> None:
            if isinstance(node, tvm.tir.For):
                defined_vars.append(node.loop_var)
                domain = node.annotations.get("tile.domain")
                if domain is not None:
                    tile_domains.extend(domain)
            elif isinstance(node, tvm.tir.AttrStmt) and isinstance(node.node, tvm.tir.IterVar):
                defined_vars.append(node.node.var)
            elif isinstance(node, tvm.tir.LetStmt):
                defined_vars.append(node.var)

        tvm.tir.stmt_functor.post_order_visit(func.body, collect)
        assert tile_domains, "expected at least one tile.domain annotation"

        for domain in tile_domains:
            unbound = [var for var in tvm.tir.analysis.undefined_vars(domain) if not any(var.same_as(defined) for defined in defined_vars)]
            assert not unbound, f"tile.domain contains unbound vars: {[var.name for var in unbound]}"


def _run_codegen_case(kernel, case_name: str, mask_index_dtype: str, mask_intrinsic: str) -> str:
    """Run every compilation stage and keep diagnostics even when one fails."""
    log_dir = LOG_ROOT / case_name
    log_dir.mkdir(parents=True, exist_ok=True)
    _record_stage(log_dir, "frontend TIR")
    _write_text(log_dir / "frontend_ast.txt", kernel.script(show_meta=True))

    try:
        _record_stage(log_dir, "TileLang lowering")
        _, device_mod = compile_test(
            kernel,
            out_idx=[1],
            target="Sunmmio",
            log_pass_output=True,
            show_meta=True,
            log_dir=str(log_dir),
        )
        _write_text(log_dir / "final_ast.txt", device_mod.script(show_meta=True))
        _assert_tile_domain_vars_are_bound(device_mod)
        _record_stage(log_dir, "SunMMIO SUVM codegen")
    except Exception:
        _write_exception(log_dir, "tilelang_lowering")
        raise

    try:
        sunmmio_target = determine_target("Sunmmio", return_object=True)
        builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
        mlir_src = builder(device_mod, sunmmio_target, "suvm").inspect_source()
        assert_source_contains(
            mlir_src,
            (
                "suvm.copy_async",
                "suvm.tile.abs",
                "suvm.tile.exp",
                "suvm.tile.ln",
                "suvm.tile.maxf",
                "suvm.tile.range",
                "suvm.tile.cmpi",
                "suvm.tile.store",
            ),
        )
        assert re.search(
            rf"suvm\.tile\.cmpi\s+ult, .* : !suvm\.tile<[^>]*x{mask_index_dtype}>",
            mlir_src,
        )
        assert not re.search(
            rf"suvm\.tile\.cmpi\s+slt, .* : !suvm\.tile<[^>]*x{mask_index_dtype}>",
            mlir_src,
        )
        mlir_path = log_dir / "softplus_suvm.mlir"
        _write_text(mlir_path, mlir_src)
        _record_stage(log_dir, "npuir-opt SUVM-to-LLVM pipeline")
    except Exception:
        _write_exception(log_dir, "sunmmio_codegen")
        raise

    npuir_opt = find_npuir_opt()
    command = [str(npuir_opt), str(mlir_path), *STRICT_OPT_ARGS]
    _write_text(log_dir / "npuir_opt_command.txt", " ".join(command) + "\n")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    _write_text(log_dir / "npuir_opt_stdout.mlir", result.stdout)
    _write_text(log_dir / "npuir_opt_stderr.log", result.stderr)
    _write_text(
        log_dir / "stage_status.txt",
        f"completed stage: npuir-opt SUVM-to-LLVM pipeline\nreturn code: {result.returncode}\n",
    )

    assert result.returncode == 0, (
        "npuir-opt --suvm-to-llvm-pipeline failed\n"
        f"command: {' '.join(command)}\n"
        f"logs: {log_dir}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert f"@llvm.riscv.sunmmio.vmsltu.{mask_intrinsic}" in result.stdout
    assert f"@llvm.riscv.sunmmio.vmslt.{mask_intrinsic}" not in result.stdout
    assert "@llvm.riscv.sunmmio.vfmax." in result.stdout
    assert "@llvm.riscv.sunmmio.vfpwln.low.exp." in result.stdout
    assert "@llvm.riscv.sunmmio.vfpwln.low.ln." in result.stdout
    return mlir_src


def _run_named_case(case_name: str) -> None:
    case_configs = {
        "bound_value": (softplus_1d_dynamic, 256, T.float32, "i32", "nxv2i32"),
        "inline_load": (softplus_1d_dynamic_inline, 256, T.float32, "i32", "nxv2i32"),
        "bound_value_bf16": (softplus_1d_dynamic, 512, T.bfloat16, "i16", "nxv4i16"),
        "inline_load_bf16": (softplus_1d_dynamic_inline, 512, T.bfloat16, "i16", "nxv4i16"),
    }
    log_dir = LOG_ROOT / case_name
    _prepare_log_dir(log_dir)
    try:
        factory, block_n, dtype, mask_index_dtype, mask_intrinsic = case_configs[case_name]
        kernel = factory(block_N=block_n, in_dtype=dtype, out_dtype=dtype)
        _run_codegen_case(kernel, case_name, mask_index_dtype, mask_intrinsic)
    except Exception:
        _write_exception(log_dir, "worker")
        raise


@pytest.mark.parametrize(
    "case_name",
    (
        "bound_value",
        "inline_load",
        pytest.param("bound_value_bf16", marks=NPU_IR_BF16_RANGE_XFAIL),
        pytest.param("inline_load_bf16", marks=NPU_IR_BF16_RANGE_XFAIL),
    ),
)
def test_softplus_1d_dynamic_codegen_with_suvm_to_llvm_pipeline(case_name):
    log_dir = LOG_ROOT / case_name
    _prepare_log_dir(log_dir)
    command = [sys.executable, "-m", MODULE_NAME, "--run-codegen-case", case_name]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    _write_text(log_dir / "worker_command.txt", " ".join(command) + "\n")
    _write_text(log_dir / "worker_stdout.log", result.stdout)
    _write_text(log_dir / "worker_stderr.log", result.stderr)
    _write_text(log_dir / "worker_return_code.txt", f"{result.returncode}\n")

    assert result.returncode == 0, (
        f"softplus codegen worker failed with return code {result.returncode}\n"
        f"logs: {log_dir}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--run-codegen-case":
        _run_named_case(sys.argv[2])
    else:
        tilelang.testing.main()
