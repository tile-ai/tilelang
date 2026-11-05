"""SS access stays scalar even through native tensor and pointer aliases."""

import pytest
from tvm_tirx import IRModule

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.analysis import VFChecker


_SS_VF_MESSAGE = "SS supports scalar uint32/uint64 accesses outside VF blocks only"
_ALIAS_MODES = ("direct", "view", "reinterpret", "pointer", "retagged_pointer", "nested_pointer")


def _ss_access(alias_mode, vf_kind, write):
    dtype = "uint64" if alias_mode == "reinterpret" else "uint32"
    shape = (1,) if dtype == "uint64" else (2,)

    @T.prim_func
    def main(out: T.Tensor(shape, dtype)):
        with T.Kernel(1):
            owner = T.alloc_ss((2,), "uint32")
            sink = T.alloc_shared(shape, dtype)
            owner[0] = T.uint32(0)
            owner[1] = T.uint32(1)
            sink[0] = T.Cast(dtype, 0)
            if alias_mode == "direct":
                alias = owner
            elif alias_mode in ("view", "reinterpret"):
                alias = T.view(owner, shape=shape, dtype=dtype)
            elif alias_mode == "pointer":
                alias = T.decl_buffer(shape, dtype, data=owner.data, scope="shared.ss")
            elif alias_mode == "retagged_pointer":
                alias = T.decl_buffer(shape, dtype, data=owner.data, scope="shared.dyn")
            else:
                first = T.view(owner, shape=shape, dtype=dtype)
                pointer = first.data
                alias = T.decl_buffer(shape, dtype, data=pointer, scope="shared.dyn")
            if vf_kind == "simt":
                with T.SimtVF(threads=1):
                    if write:
                        alias[0] = T.Cast(dtype, 7)
                    else:
                        sink[0] = alias[0]
            elif vf_kind == "simd":
                with T.SimdVF():
                    if write:
                        alias[0] = T.Cast(dtype, 7)
                    else:
                        sink[0] = alias[0]
            else:
                if write:
                    alias[0] = T.Cast(dtype, 7)
                else:
                    sink[0] = alias[0]
            out[0] = sink[0]

    return main


@pytest.mark.parametrize("alias_mode", _ALIAS_MODES)
@pytest.mark.parametrize("vf_kind", ["simt", "simd"])
@pytest.mark.parametrize("write", [False, True], ids=["read", "write"])
def test_ss_access_inside_vf_is_rejected_through_aliases(alias_mode, vf_kind, write):
    func = _ss_access(alias_mode, vf_kind, write)
    with pytest.raises(RuntimeError, match=_SS_VF_MESSAGE):
        VFChecker()(IRModule({"main": func}))


@pytest.mark.parametrize("alias_mode", _ALIAS_MODES)
@pytest.mark.parametrize("write", [False, True], ids=["read", "write"])
def test_ss_alias_access_outside_vf_is_unchanged(alias_mode, write):
    func = _ss_access(alias_mode, "none", write)
    actual = VFChecker()(IRModule({"main": func}))
    assert actual["main"].same_as(func)


@pytest.mark.parametrize("vf_kind", ["simt", "simd"])
def test_vf_shared_access_is_unchanged(vf_kind):
    @T.prim_func
    def main(inp: T.Tensor((2,), "float32"), out: T.Tensor((2,), "float32")):
        with T.Kernel(1):
            source = T.alloc_shared((2,), "float32")
            destination = T.alloc_shared((2,), "float32")
            T.copy(inp, source)
            if vf_kind == "simt":
                with T.SimtVF(threads=2):
                    tx = T.get_thread_binding()
                    destination[tx] = source[tx]
            else:
                with T.SimdVF():
                    for i in T.Parallel(2):
                        destination[i] = source[i]
            T.copy(destination, out)

    actual = VFChecker()(IRModule({"main": main}))
    assert actual["main"].same_as(main)


def test_ss_scope_does_not_leak_into_following_vf():
    @T.prim_func
    def main(out: T.Tensor((2,), "uint32")):
        with T.Kernel(1):
            words = T.alloc_ss((2,), "uint32")
            words[0] = T.uint32(1)
            out[0] = words[0]
            with T.SimtVF(threads=1):
                out[1] = T.uint32(2)

    actual = VFChecker()(IRModule({"main": main}))
    assert actual["main"].same_as(main)


def test_simd_global_memory_rejection_is_preserved():
    @T.prim_func
    def main(out: T.Tensor((2,), "uint32")):
        with T.Kernel(1), T.SimdVF():
            out[0] = T.uint32(1)

    with pytest.raises(ValueError, match="cannot access global memory"):
        VFChecker()(IRModule({"main": main}))


def test_simt_shared_memory_still_lowers():
    @T.prim_func
    def main(inp: T.Tensor((128,), "float32"), out: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            values = T.alloc_shared((128,), "float32")
            T.copy(inp, values)
            with T.SimtVF(threads=128):
                tx = T.get_thread_binding()
                out[tx] = values[tx]

    source = tilelang.lower(main, target="ascend", enable_device_compile=False).kernel_source
    assert "__simt_vf__" in source
    assert "__ubuf__" in source
