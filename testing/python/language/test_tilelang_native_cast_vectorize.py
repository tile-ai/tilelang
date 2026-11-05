"""Random bits stay scalar when native vectorization widens a cast."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
import tilelang.testing


def _nodes(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="pre")
    return found


@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize(
    "mode",
    ["constant", "shared_load", "group2_load", "varying_load", "rng", "bound_rng", "bound_constant", "bound_load"],
)
def test_native_cast_vectorization_preserves_rbits_scope_and_effects(lanes, mode):
    lane = ir.Var("lane", "int32")
    source = ir.Var("A", tirx.TensorType([4], "float32"))
    output = ir.Var("B", tirx.TensorType([4], "float16"))
    random = ir.Var("R", tirx.TensorType([4], "uint32"))
    value = ir.TensorLoad(source, [lane])
    bound = []
    bits = ir.const(0x12345678, "uint32")
    if mode == "shared_load":
        bits = ir.TensorLoad(random, [lane // lanes])
    elif mode == "group2_load":
        bits = ir.TensorLoad(random, [lane // 2])
    elif mode in ("varying_load", "bound_load"):
        bits = ir.TensorLoad(random, [lane])
    elif mode in ("rng", "bound_rng"):
        bits = T.rng_rand()
    if mode.startswith("bound_"):
        variable = ir.Var("random_bits", bits.ty)
        bound.append(ir.Bind(variable, bits))
        bits = variable
    loc = ir.SourceLoc(ir.SourceName("cast_vectorize.py"), 17, 4, 17, 32)
    call = ir.Call("tl.cast", [value, bits], attrs={"round": "rs", "sat": True}, ty_args=[ir.PrimType("float16")], loc=loc)
    loop = ir.For(
        lane,
        ir.const(0),
        ir.const(lanes),
        ir.const(1),
        ir.ForKind.VECTORIZED,
        [*bound, ir.TensorStore(output, [lane], call)],
    )
    function = tirx.Function([source, output, random], [loop], ir.NoneType()).with_attrs(
        {"target": Target({"kind": "cuda", "arch": "sm_100a"}), "tirx.noalias": True}
    )
    assert tirx.analysis.check_well_formed(function)
    original = ir.save_json(function)
    module = tilelang.transform.VectorizeLoop()(ir.IRModule({"main": function}))
    assert ir.save_json(function) == original
    result = module["main"]
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    casts = [call for call in _nodes(result, ir.Call) if call.op.name == "tl.cast"]
    assert len(casts) == 1
    rewritten = casts[0]
    rewritten.validate()
    assert rewritten.args[1].ty == ir.PrimType("uint32")
    assert rewritten.loc.same_as(loc)
    assert rewritten.attrs["round"] == "rs" and rewritten.attrs["sat"] is True
    vectorized = mode in ("constant", "shared_load", "bound_constant") or (mode == "group2_load" and lanes == 2)
    expected_lanes = lanes if vectorized else 1
    assert rewritten.ty == ir.PrimType("float16" + (f"x{expected_lanes}" if expected_lanes > 1 else ""))
    assert len(_nodes(result, ir.For)) == (0 if vectorized else 1)
    if mode in ("rng", "bound_rng"):
        rng_calls = [call for call in _nodes(result, ir.Call) if call.op.name == "tl.rng_rand"]
        assert len(rng_calls) == 1
        assert _nodes(_nodes(result, ir.For)[0].body, ir.Call)


@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_native_cast_explicit_vectorized_loop_executes():
    import torch

    @T.prim_func
    def kernel(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float16")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            for i in T.vectorized(4):
                B[tx * 4 + i] = T.cast(A[tx * 4 + i], "float16", round="rs", rbits=T.uint32(0x12345678))

    source = torch.tensor([-8.0, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 8.0] * 16, dtype=torch.float32, device="cuda")
    compiled = tilelang.compile(kernel, out_idx=[1], target=Target({"kind": "cuda", "arch": "sm_100a"}))
    actual = compiled(source)
    torch.testing.assert_close(actual, source.to(torch.float16), rtol=0, atol=0)
    assert "__tl_cvt_f32x2_to_f16x2_rs_sat" in compiled.get_kernel_source()


@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_native_vectorizer_preserves_gpu_random_stream_and_casts():
    import torch

    def make_kernel(vectorized):
        @T.prim_func
        def kernel(
            A: T.Tensor((128,), "float32"),
            B: T.Tensor((128,), "float16"),
            R: T.Tensor((128,), "uint32"),
            S: T.Tensor((128,), "uint32"),
        ):
            with T.Kernel(1, threads=32):
                tx = T.get_thread_binding()
                T.rng_init(42, 0, tx)
                for i in T.vectorized(4) if vectorized else T.serial(4):
                    bits = T.rng_rand()
                    R[tx * 4 + i] = bits
                    B[tx * 4 + i] = T.cast(A[tx * 4 + i], "float16", round="rs", rbits=bits)
                    S[tx * 4 + i] = T.rng_rand()

        return kernel.with_attr("target", Target({"kind": "cuda", "arch": "sm_100a"}))

    scalar = make_kernel(False)
    vector = make_kernel(True)
    vector = tilelang.transform.VectorizeLoop()(ir.IRModule({"main": vector}))["main"]
    assert tirx.analysis.check_well_formed(vector)
    source = torch.tensor(
        [-8.0, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 8.0] * 16,
        dtype=torch.float32,
        device="cuda",
    )
    outputs = []
    for function in (scalar, vector):
        kernel = tilelang.compile(function, out_idx=[1, 2, 3], target=function.attrs["target"])
        outputs.append(kernel(source))
    expected, actual = outputs
    torch.testing.assert_close(actual[0], source.to(torch.float16), rtol=0, atol=0)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left.to(torch.float64), right.to(torch.float64), rtol=0, atol=0)
