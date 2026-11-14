"""Manual alias inference uses completion edges on the accessing hardware pipe."""

import pytest
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes


def _call(name, *args):
    return tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl." + name), list(args)))


def _load(gm, ub):
    return _call("ascend_copy_gm_to_ubuf", ub.access_ptr("w"), gm.access_ptr("r"), 0, 1, 256, 0, 0, 0, 0, 0, 0)


def _store(ub, gm):
    return _call("ascend_copy_ubuf_to_gm", gm.access_ptr("w"), ub.access_ptr("r"), 0, 1, 256, 0, 0, 0)


def _flags(pipe):
    return [_call("ascend_set_flag", pipe, 0), _call("ascend_wait_flag", pipe, 0)]


def _module(buffers, body, gm):
    return tvm.IRModule({"main": tirx.PrimFunc([gm.data], tirx.SeqStmt([*[tirx.AllocBuffer(buffer) for buffer in buffers], *body]))})


def _may_alias(after, a, b):
    (contract,) = [node.node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.buffer_alias_map"]
    return b.data in contract[a.data]


@pytest.mark.parametrize("release", ["none", "acquire-only", "MTE3_MTE2", "PIPE_ALL"])
def test_dma_read_requires_a_completion_edge_before_reuse(release):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.dyn")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.dyn")
    between = []
    if release == "acquire-only":
        between = [_call("ascend_wait_flag", "MTE3_MTE2", 0)]
    elif release == "MTE3_MTE2":
        between = _flags(release)
    elif release == "PIPE_ALL":
        between = [_call("ascend_pipe_barrier", "PIPE_ALL")]
    before = _module(
        [a, b], [_load(gm, a), *_flags("MTE2_MTE3"), _store(a, gm), *between, _load(gm, b), *_flags("MTE2_MTE3"), _store(b, gm)], gm
    )
    after = transform.InferBufferAliases()(before)
    assert _may_alias(after, a, b) == (release in ("MTE3_MTE2", "PIPE_ALL"))


@pytest.mark.parametrize("release_pipe", ["S_MTE2", "V_MTE2"])
def test_pad_value_operand_is_scalar_even_inside_vector_block(release_pipe):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.dyn")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.dyn")
    body = tirx.SeqStmt(
        [_load(gm, a), *_flags("MTE2_S"), tirx.Evaluate(T.ascend_set_copy_pad_value(a[0])), *_flags(release_pipe), _load(gm, b)]
    )
    before = _module([a, b], [tirx.SBlock([], [], [], "VECTOR", body)], gm)
    after = transform.InferBufferAliases()(before)
    assert _may_alias(after, a, b) == (release_pipe == "S_MTE2")


@pytest.mark.parametrize("release_pipe", ["PIPE_MTE1", "PIPE_FIX"])
def test_cross_core_handshake_only_releases_its_local_pipe(release_pipe):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.l1")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.l1")
    l0a = tirx.decl_buffer((64,), "float32", name="l0a", scope="shared.l0a")
    # The Cube reader is MTE1. A FIX release cannot complete that read, even
    # if Vector returns a token that Cube acquires on MTE2 before filling b.
    cube_read = tirx.SBlock(
        [],
        [],
        [],
        "CUBE",
        tirx.SeqStmt(
            [
                _call("ascend_copy_gm_to_cbuf", a.access_ptr("w"), gm.access_ptr("r"), 0, 16, 0, 4, 16, 0, 0, 0, 4, "float32"),
                *_flags("MTE2_MTE1"),
                _call("ascend_load_cbuf_to_ca", l0a.access_ptr("w"), a.access_ptr("r")),
                _call("ascend_cross_core_set_flag", 4, release_pipe, 0),
            ]
        ),
    )
    vector = tirx.SBlock(
        [],
        [],
        [],
        "VECTOR",
        tirx.SeqStmt(
            [
                _call("ascend_cross_core_wait_flag", 4, "PIPE_MTE3", 0),
                _call("ascend_cross_core_set_flag", 4, "PIPE_MTE3", 1),
            ]
        ),
    )
    cube_write = tirx.SBlock(
        [],
        [],
        [],
        "CUBE",
        tirx.SeqStmt(
            [
                _call("ascend_cross_core_wait_flag", 4, "PIPE_MTE2", 1),
                _call("ascend_copy_gm_to_cbuf", b.access_ptr("w"), gm.access_ptr("r"), 0, 16, 0, 4, 16, 0, 0, 0, 4, "float32"),
            ]
        ),
    )
    before = _module([a, b, l0a], [cube_read, vector, cube_write], gm)
    after = transform.InferBufferAliases()(before)
    assert _may_alias(after, a, b) == (release_pipe == "PIPE_MTE1")
