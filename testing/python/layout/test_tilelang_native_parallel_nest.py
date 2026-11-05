"""Native loop bodies must preserve multidimensional copy vector planning."""

from itertools import product
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
from tilelang.analysis._visitor import tilelang_metadata


def _roundtrip(shape):
    @T.prim_func
    def main(A: T.Tensor(shape, "float16"), B: T.Tensor(shape, "float16")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment(shape, "float16")
            T.copy(A, fragment)
            T.copy(fragment, B)

    return main


def _infer(shape, arch):
    value = _roundtrip(shape)
    original = ir.save_json(value)
    target = Target({"kind": "cuda", "arch": arch})
    module = ir.IRModule({"main": value.with_attr("target", target)})
    module = tilelang.transform.MaterializeKernelLaunch()(module)
    with target:
        module = tilelang.transform.LayoutInference()(module)
    assert tirx.analysis.check_well_formed(module)
    assert ir.save_json(value) == original
    found = []

    def collect(node):
        metadata = tilelang_metadata(node)
        if metadata is not None:
            for tensor, layout in metadata.get("layout_map", {}).items():
                if tensor.name == "fragment":
                    found.append(layout)

    tvm_ffi.structural_walk(module["main"].body, ((ir.Call, ir.RegionStmt), collect), order="post")
    assert len(found) == 1
    assert int(found[0].replicate_size) == 1
    return found[0]


@pytest.mark.parametrize("shape", [(16384,), (128, 128), (8, 16, 128), (1, 128, 128)])
@pytest.mark.parametrize("arch, width", [("sm_80", 8), ("sm_100a", 16)])
def test_multidimensional_fragment_retains_contiguous_vector_runs(shape, arch, width):
    layout = _infer(shape, arch)
    analyzer = sym.Analyzer()

    def owner(index):
        return int(analyzer.simplify(layout.map_forward_thread(list(index))[0]))

    # The same bytes and launch geometry have the same vector run regardless
    # of whether their logical shape is flat, 2D or 3D. Adjacent vector runs
    # belong to adjacent threads; a scalar plan violates this oracle.
    zeros = [0] * (len(shape) - 1)
    assert [owner([*zeros, i]) for i in range(2 * width)] == [0] * width + [1] * width

    occupied = set()
    for index in product(*(range(extent) for extent in shape)):
        slot = tuple(int(analyzer.simplify(x)) for x in layout.map_forward_index(list(index)))
        key = (owner(index), *slot)
        assert key not in occupied, (shape, index, key)
        occupied.add(key)
    assert len(occupied) == 16384


@pytest.mark.parametrize("shape", [(768,), (3, 256), (1, 3, 256)])
def test_padding_choice_counts_the_complete_nest(shape):
    layout = _infer(shape, "sm_100a")
    analyzer = sym.Analyzer()
    zeros = [0] * (len(shape) - 1)
    owners = [int(analyzer.simplify(layout.map_forward_thread([*zeros, i])[0])) for i in range(4)]
    # 768 / (128 threads * 2 elements) is integral; larger powers of two
    # require padding. Looking only at the outer extent incorrectly picks 1.
    assert owners == [0, 0, 1, 1]


@pytest.mark.parametrize("variant", ["fp16_128x128_t128", "fp32_64x256_t256"])
def test_original_multidimensional_copy_executes_on_cuda(variant, tmp_path):
    import torch

    assert torch.cuda.is_available(), "This test requires the assigned CUDA device"
    source = Path(__file__).resolve().parents[3] / "maint/layout_inference/cases/elementwise_copy.py"
    before = source.read_bytes()
    spec = importlib.util.spec_from_file_location("native_parallel_original_copy", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = module.VARIANTS[variant]()
    major, minor = torch.cuda.get_device_capability()
    target = Target({"kind": "cuda", "arch": f"sm_{major}{minor}" + ("a" if major >= 9 else "")})
    shape = tuple(int(dim) for dim in function.params[0].ty.shape)
    dtype = torch.float16 if variant.startswith("fp16") else torch.float32
    count = shape[0] * shape[1]
    values = ((torch.arange(count, dtype=torch.int32, device="cuda") % 127) - 63).to(dtype).reshape(shape)
    kernel = tilelang.compile(function, out_idx=[1], target=target, execution_backend="tvm_ffi")
    output = kernel(values)
    torch.testing.assert_close(output, values, rtol=0, atol=0)
    assert source.read_bytes() == before
    (tmp_path / "kernel.cu").write_text(kernel.get_kernel_source())
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(before).hexdigest(),
                "variant": variant,
                "target": str(target),
                "shape": shape,
                "dtype": str(dtype),
                "oracle": "exact input copy",
                "elements": count,
            },
            indent=2,
        )
        + "\n"
    )


def _fragment_slice_in_scope(scope, *, write=False, offset=0, outside=False):
    """Native loops make the lexical fact under test independent of unrolling."""
    batch = ir.Var("batch", "int32")
    shape = (3, 32, 4)
    values = ir.Var("values", tirx.TensorType(shape, "float32"))
    fragment = ir.Var("fragment", tirx.TensorType(shape, "float32", scope="local.fragment"))
    layout = tilelang.Fragment(
        shape,
        forward_thread_fn=lambda batch, row, col: row,
        forward_index_fn=lambda batch, row, col: batch * 4 + col,
    )
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"layout_map": {fragment: layout}})
    annotation = ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))

    def access():
        # Separate loop definitions in sibling statements, even with the same
        # spelling, must not acquire each other's range facts by name.
        row, col = (ir.Var(name, "int32") for name in ("row", "col"))
        logical = [batch + offset, row, col]
        src, dst = (values, fragment) if write else (fragment, values)
        body = ir.TensorStore(dst, logical, ir.TensorLoad(src, logical))
        for var, extent in ((col, 4), (row, 32)):
            body = ir.For(var, ir.const(0), ir.const(extent), ir.const(1), ir.ForKind.PARALLEL, [body])
        return body

    params = [values]
    if scope == "serial":
        body = [ir.For(batch, ir.const(0), ir.const(3), ir.const(1), ir.ForKind.DEFAULT, [access()])]
    else:
        params.append(batch)
        guarded = ir.If((batch >= 0) & (batch < 3), [access()], None)
        body = [guarded, access()] if outside else [guarded]
    thread = ir.Var("thread", "int32")
    function = (
        tirx.Function(
            params,
            [
                ir.RegionStmt(
                    "tirx.launch_thread",
                    [ir.StringImm("threadIdx.x"), ir.const(32)],
                    [thread],
                    {},
                    [
                        ir.Bind(fragment, ir.Call("tirx.alloc_tensor", [], ty_args=[fragment.ty])),
                        annotation,
                        *body,
                    ],
                )
            ],
            ir.NoneType(),
        )
        .with_attr("target", Target({"kind": "cuda", "arch": "sm_80"}))
        .with_attr("global_symbol", "main")
    )
    assert tirx.analysis.check_well_formed(function)
    return function


@pytest.mark.parametrize("scope", ["serial", "predicate"])
@pytest.mark.parametrize("write", [False, True])
def test_fragment_slice_layout_uses_enclosing_scope(scope, write):
    function = _fragment_slice_in_scope(scope, write=write)
    original = ir.save_json(function)
    with function.attrs["target"]:
        inferred = tilelang.transform.LayoutInference()(ir.IRModule.from_expr(function))
        lowered = tilelang.transform.LowerTileOp()(inferred)
    assert tirx.analysis.check_well_formed(lowered)
    assert ir.save_json(function) == original
    layouts = []

    def collect(node):
        metadata = tilelang_metadata(node)
        if metadata is not None:
            layouts.extend(layout for tensor, layout in metadata.get("layout_map", {}).items() if tensor.name == "fragment")

    tvm_ffi.structural_walk(inferred, ((ir.Call, ir.RegionStmt), collect))
    assert len(layouts) == 1
    proof = sym.Analyzer()
    # The range premise enables this slice; it must not reassign its cells.
    for batch, row, col in product(range(3), range(32), range(4)):
        assert int(proof.simplify(layouts[0].map_forward_thread([batch, row, col])[0])) == row
        assert [int(proof.simplify(value)) for value in layouts[0].map_forward_index([batch, row, col])] == [batch * 4 + col]


@pytest.mark.parametrize("case", ["out-of-bounds", "outside-predicate"])
def test_fragment_slice_scope_does_not_invent_range_facts(case):
    function = _fragment_slice_in_scope(
        "serial" if case == "out-of-bounds" else "predicate",
        offset=1 if case == "out-of-bounds" else 0,
        outside=case == "outside-predicate",
    )
    original = ir.save_json(function)
    with function.attrs["target"], pytest.raises((ValueError, RuntimeError), match="Layout infer conflict"):
        tilelang.transform.LayoutInference()(ir.IRModule.from_expr(function))
    assert ir.save_json(function) == original
