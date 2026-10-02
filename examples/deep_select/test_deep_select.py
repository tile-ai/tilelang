import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

from example_deep_select import deep_select, prepare_deep_select, _warp_exclusive_sum
from deep_select_config import select_config


def check_result(x, k, result):
    values, indices = result
    assert indices.shape == (x.shape[0], k)
    assert ((indices >= 0) & (indices < x.shape[1])).all()
    ordered = indices.sort().values
    assert (ordered[:, 1:] != ordered[:, :-1]).all()
    gathered = x.gather(1, indices.long())
    if values is not None:
        bits = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
        assert torch.equal(values.view(bits), gathered.view(bits))
    expected = torch.topk(x, k, sorted=True).values
    torch.testing.assert_close(gathered.sort(descending=True).values, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strategy", ["stream", "hierarchical"])
@pytest.mark.parametrize(
    "shape,k",
    [((0, 123), 7), ((3, 1), 1), ((3, 33), 7), ((2, 511), 511), ((3, 4093), 512), ((1, 8192), 4096), ((2, 16387), 512), ((1, 65539), 1024)],
)
@tilelang.testing.requires_cuda
def test_shapes(dtype, strategy, shape, k):
    torch.manual_seed(17)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    check_result(x, k, deep_select(x, k, strategy=strategy))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strategy,n", [("stream", 12289), ("stream", 32768), ("hierarchical", 12289)])
@pytest.mark.parametrize("pattern", ["constant", "ascending", "descending", "ties", "infinities", "zeros", "subnormals"])
@tilelang.testing.requires_cuda
def test_distributions(dtype, strategy, n, pattern):
    # Cover full TMA tiles as well as masked LDG with a one-element tail. In
    # particular, subnormals must survive *filtering*, not just initial loading.
    k = 257
    x = torch.arange(n, device="cuda", dtype=torch.float32).expand(3, n).clone()
    if pattern == "constant":
        x.fill_(-3)
    elif pattern == "descending":
        x = -x
    elif pattern == "ties":
        x = (x % 7) - 3
    elif pattern == "infinities":
        x[0].fill_(-float("inf"))
        x[1].fill_(float("inf"))
        x[2, ::3] = float("inf")
        x[2, 1::3] = -float("inf")
    elif pattern == "zeros":
        x.zero_()
        x[:, ::2] = -0.0
    x = x.to(dtype)
    if pattern == "subnormals":
        bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
        raw = torch.arange(n, device="cuda", dtype=torch.int32) % (127 if dtype == torch.bfloat16 else 8191) + 1
        x = raw.to(bits).view(dtype).expand(3, n).clone()
        x[1] = -x[1]
        x[2, ::2] = -x[2, ::2]
    check_result(x, k, deep_select(x, k, strategy=strategy, splits=1))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("return_values", [False, True])
@tilelang.testing.requires_cuda
def test_long_row_and_reuse(index_dtype, return_values):
    x = torch.randn(1, 1048576, device="cuda", dtype=torch.bfloat16)
    run = prepare_deep_select(x, 512, index_dtype=index_dtype, return_values=return_values)
    assert run.configuration["launches"] >= 3
    check_result(x, 512, run())
    x.normal_()
    result = run()
    assert result[1].dtype == index_dtype
    assert (result[0] is None) == (not return_values)
    check_result(x, 512, result)


@pytest.mark.parametrize(
    "dtype,k,overrides,stages,block",
    [
        ("bfloat16", 512, {}, 6, 2048),
        ("float32", 512, {}, 3, 4096),
        ("bfloat16", 1024, {}, 4, 2048),
        ("float32", 4096, {}, 2, 2048),
        ("bfloat16", 512, {"use_tma": False}, 0, 4096),
        ("bfloat16", 512, {"n": 131073}, 0, 4096),
        ("bfloat16", 512, {"block_size": 4096}, 3, 4096),
        ("bfloat16", 512, {"vector_loads": False}, 0, 4096),
        ("bfloat16", 512, {"shared_memory_limit": 24656}, 0, 2048),
    ],
)
def test_tma_dispatch(dtype, k, overrides, stages, block):
    args = dict(batch=4096, n=131072, k=k, dtype=dtype, sm_count=170, shared_memory_limit=99 * 1024, use_tma=True)
    config = select_config(**(args | overrides))
    assert (config["tma_stages"], config["block_size"]) == (stages, block)


@pytest.mark.parametrize(
    "batch,n,overrides,expected",
    [
        (64, 8192, {}, ("hierarchical", 1, 512, None, 0)),
        (64, 32768, {}, ("hierarchical", 2, 512, None, 0)),
        (64, 65536, {}, ("hierarchical", 4, 512, None, 0)),
        (64, 131072, {}, ("stream", 4, 256, 2048, 0)),
        (128, 131072, {}, ("stream", 1, 512, 4096, 16384)),
        (170, 131072, {}, ("stream", 1, 512, 4096, 16384)),
        (170, 133120, {}, ("stream", 1, 512, 2048, 16384)),
        (171, 131072, {}, ("stream", 1, 256, 2048, 0)),
        (64, 32768, {"splits": 1}, ("stream", 1, 512, 4096, 16384)),
        (170, 131072, {"use_tma": False}, ("stream", 1, 256, 4096, 0)),
    ],
)
def test_small_dispatch(batch, n, overrides, expected):
    args = dict(batch=batch, n=n, k=512, dtype="bfloat16", sm_count=170, shared_memory_limit=99 * 1024, use_tma=True)
    config = select_config(**(args | overrides))
    assert tuple(config[key] for key in ("strategy", "splits", "threads", "block_size", "init_size")) == expected


@pytest.mark.parametrize("batch", [6, 256, 4096])
def test_short_k1024_dispatch(batch):
    args = dict(batch=batch, n=16384, k=1024, dtype="bfloat16", sm_count=170, shared_memory_limit=99 * 1024)
    config = select_config(**args)
    assert (config["strategy"], config["splits"], config["threads"]) == ("hierarchical", 1, 512)
    assert select_config(**args, splits=2)["splits"] == 2
    assert select_config(**args, threads=256)["splits"] == 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("return_values", [False, True])
@tilelang.testing.requires_cuda
def test_copy_all_and_replay(dtype, index_dtype, return_values):
    for n in (33, 1024, 4096):
        # Contiguous storage views need not have a vector-aligned base pointer.
        x = torch.randn(3 * n + 1, device="cuda", dtype=dtype)[1:].reshape(3, n)
        run = prepare_deep_select(x, n, index_dtype=index_dtype, return_values=return_values)
        assert run.configuration["strategy"] == "copy"
        assert run.configuration["launches"] == 1
        values, indices = run()
        assert (values is None) == (not return_values)
        if values is not None:
            assert values.data_ptr() != x.data_ptr()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.normal_()
        x[:, ::2] = -0.0
        indices.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert indices.dtype == index_dtype
        assert torch.equal(indices, torch.arange(n, device=x.device).expand(3, n))
        check_result(x, n, (values, indices))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("blocks,k", [(1, 512), (5, 512), (6, 512), (7, 512), (13, 512), (16, 1024), (16, 4096)])
@tilelang.testing.requires_cuda
def test_tma_ring_and_replay(dtype, blocks, k):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA requires SM90+")
    # Cover prologue/drain, both parity transitions, non-power-of-two traversal
    # and independent row/shard offsets. Ordinary calls enable TMA by default.
    x = torch.randn((3, 2 * blocks * 2048), device="cuda", dtype=dtype)
    run = prepare_deep_select(x, k, strategy="stream", splits=2, seed=17)
    assert run.configuration["tma_stages"] > 0
    if dtype == torch.bfloat16:
        assert run.configuration["init_size"] == min(blocks * 2048, 16384)
    result = run()
    check_result(x, k, result)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for constant in (False, True):
        x.fill_(-3) if constant else x.normal_()
        result[1].fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        check_result(x, k, result)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@tilelang.testing.requires_cuda
def test_tma_default_and_unaligned_fallback(dtype):
    # Contiguous does not imply a 16-byte aligned base (e.g. a storage view).
    storage = torch.randn(32 * 131072 + 1, device="cuda", dtype=dtype)
    aligned = storage[:-1].view(32, 131072)
    run = prepare_deep_select(aligned, 512)
    assert bool(run.configuration["tma_stages"]) == (torch.cuda.get_device_capability()[0] >= 9)
    check_result(aligned, 512, run())
    for x, use_tma in [(storage[1:].view(32, 131072), True), (aligned, False)]:
        run = prepare_deep_select(x, 512, use_tma=use_tma)
        assert run.configuration["tma_stages"] == 0
        check_result(x, 512, run())


@pytest.mark.parametrize("seed", [0, 17, 12345])
@tilelang.testing.requires_cuda
def test_stream_multiblock_permutation(seed):
    x = torch.arange(49157, device="cuda", dtype=torch.float32).expand(2, -1).contiguous()
    check_result(x, 513, deep_select(x, 513, strategy="stream", splits=2, seed=seed))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hit_offset", [0, 7])
@tilelang.testing.requires_cuda
def test_stream_final_compaction_after_empty_rounds(dtype, hit_offset):
    # With 13 blocks and seed zero the stride is 8. Step 8 visits block 12,
    # after the fused initialization window (or the ordinary first compact).
    # Its reservation must survive four empty rounds until final compaction.
    block, k = 4096, 512
    x = torch.full((1, 13 * block), -1.0, device="cuda", dtype=dtype)
    x[0, :k] = 10.0
    # Exercise both ends of an eight-element hit mask, including seven skipped
    # positions, before empty rounds force the final compact.
    x[0, 13 * block - 8 + hit_offset] = 11.0
    check_result(x, k, deep_select(x, k, strategy="stream", splits=1, block_size=block))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strategy", ["stream", "hierarchical"])
@pytest.mark.parametrize("k", [32, 33])
@tilelang.testing.requires_cuda
def test_full_selection_with_infinities(dtype, strategy, k):
    # An FP32 whole-bucket early exit must not decode its lower edge to NaN.
    x = torch.full((3, 33), -float("inf"), device="cuda", dtype=dtype)
    x[1].fill_(float("inf"))
    x[2, ::2] = float("inf")
    check_result(x, k, deep_select(x, k, strategy=strategy, splits=1))


@tilelang.jit
def _warp_prefix_kernel(max_count):
    @T.prim_func
    def kernel(X: T.Tensor((256,), "int32"), Y: T.Tensor((256,), "int32")):
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            Y[tx] = _warp_exclusive_sum(X[tx], max_count)

    return kernel


@pytest.mark.parametrize("max_count", [1, 3, 16, 32, 64])
@tilelang.testing.requires_cuda
def test_warp_prefix_boundaries(max_count):
    # Cover zero, inclusive power-of-two bounds, mixed counts and warp isolation.
    x = torch.arange(256, device="cuda", dtype=torch.int32) % (max_count + 1)
    x[:32] = 0
    x[32:64] = max_count
    y = torch.empty_like(x)
    _warp_prefix_kernel(max_count)(x, y)
    counts = x.reshape(8, 32)
    assert torch.equal(y.reshape(8, 32), counts.cumsum(-1) - counts)
