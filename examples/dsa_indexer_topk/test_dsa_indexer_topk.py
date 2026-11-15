import pytest
import torch
import tilelang.testing

from example_dsa_indexer_topk import causal_key_ranges, indexer_topk, reference_scores


def make_inputs(sq, sk, heads=64, dim=128):
    # Small integers make every score exact in fp32 whatever the summation
    # order, and produce many equal scores.
    torch.manual_seed(0)
    q = torch.randint(-2, 3, (sq, heads, dim), device="cuda")
    weights = torch.randint(-2, 4, (sq, heads), device="cuda")
    k = torch.randint(-2, 3, (sk, dim), device="cuda")
    return q.to(torch.bfloat16), weights.to(torch.bfloat16), k.to(torch.bfloat16)


def masked_scores(q, weights, k, ranges):
    cols = torch.arange(k.shape[0], device="cuda")
    scores = reference_scores(q, weights, k)
    return scores.masked_fill((cols < ranges[:, :1]) | (cols >= ranges[:, 1:]), -float("inf"))


def expected_topk(scores, topk):
    """Best ids per row with ties going to the smaller id, -1 where a row runs out of keys."""
    order = scores.sort(dim=1, descending=True, stable=True).indices[:, :topk]
    order = order.masked_fill(scores.gather(1, order) == -float("inf"), -1)
    return torch.nn.functional.pad(order, (0, topk - order.shape[1]), value=-1)


def check_exact(q, weights, k, ranges, topk):
    indices, lengths = indexer_topk(q, weights, k, ranges, topk)
    # The length depends on the range alone.
    size = ranges[:, 1].clamp(max=k.shape[0]) - ranges[:, 0].clamp(min=0)
    assert torch.equal(lengths, size.clamp(0, topk))
    for start in range(0, q.shape[0], 512):
        rows = slice(start, start + 512)
        want = expected_topk(masked_scores(q[rows], weights[rows], k, ranges[rows]), topk)
        got = indices[rows].long()
        # Valid ids first, then -1.
        assert torch.equal(got >= 0, torch.arange(topk, device="cuda") < lengths[rows, None])
        assert torch.equal(got.sort(dim=1).values, want.sort(dim=1).values)


@pytest.mark.parametrize(
    "sq,sk,topk,heads,dim",
    [
        (2048, 6000, 2048, 64, 128),  # every row needs several selections
        (777, 777, 2048, 64, 128),  # rows shorter than topk, odd query count
        (1500, 1500, 512, 64, 128),  # largest topk that uses 64-key tiles
        (1200, 1200, 128, 32, 64),  # smallest topk, other head count and width
        (1100, 3000, 1024, 32, 128),  # topk 1024, 32 heads of width 128
        (900, 900, 256, 64, 64),  # topk 256, 64 heads of width 64
        (1, 50, 128, 32, 64),  # one query, fewer keys than a tile
    ],
)
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_exact_with_ties(sq, sk, topk, heads, dim):
    q, weights, k = make_inputs(sq, sk, heads, dim)
    ranges = causal_key_ranges(torch.arange(sk - sq, sk, device="cuda"))
    check_exact(q, weights, k, ranges, topk)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_all_scores_equal():
    # Zero weights tie every key, so only the ids decide. The ranges start at
    # 100, not at a multiple of 256, so that the last id byte has to split a bucket.
    q, weights, k = make_inputs(1000, 5000)
    weights.zero_()
    ranges = causal_key_ranges(torch.arange(4000, 5000, device="cuda"))
    ranges[:, 0] = 100
    check_exact(q, weights, k, ranges, 2048)


@pytest.mark.parametrize("pattern", ["negative", "beyond_float16"])
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_score_ranges(pattern):
    q, weights, k = make_inputs(1000, 5000)
    if pattern == "negative":
        # Negative floats sort in the reverse order of their bit patterns.
        weights = -(weights.abs() + 1)
    else:
        # The k-th score is past the largest float16, where the coarse pass
        # saturates; 384 * 512 * 64 still fits fp32 exactly.
        weights = (weights.abs() + 1).clamp(max=3) * 128
    ranges = causal_key_ranges(torch.arange(4000, 5000, device="cuda"))
    check_exact(q, weights, k, ranges, 2048)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_key_ranges():
    q, weights, k = make_inputs(1000, 7000)
    # Packed documents, including one of five tokens and one of a single token.
    cu_seqlens = torch.tensor([0, 5, 1300, 1301, 5000, 7000], device="cuda")
    positions = torch.arange(0, 7000, 7, device="cuda")
    check_exact(q, weights, k, causal_key_ranges(positions, cu_seqlens), 2048)

    # Unrelated neighbors, empty ranges and ranges ending at the last key.
    lo = torch.randint(0, 5000, (1000,), device="cuda")
    hi = (lo + torch.randint(0, 5000, (1000,), device="cuda")).clamp_max(7000)
    hi[::7] = lo[::7]
    # A block whose two queries have no keys at all.
    lo[15] = hi[15] = lo[14]
    check_exact(q, weights, k, torch.stack([lo, hi], dim=1).int(), 2048)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_random_scores():
    # fp32 rounding depends on the summation order, so a key may be swapped for
    # another only if both are within rounding distance of the k-th score.
    sq, sk, topk = 1024, 6000, 2048
    torch.manual_seed(0)
    q = torch.randn((sq, 64, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((sq, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((sk, 128), device="cuda", dtype=torch.bfloat16)
    ranges = causal_key_ranges(torch.arange(sk - sq, sk, device="cuda"))
    indices, lengths = indexer_topk(q, weights, k, ranges, topk)
    assert (lengths == topk).all()
    for start in range(0, sq, 256):
        rows = slice(start, start + 256)
        scores = masked_scores(q[rows], weights[rows], k, ranges[rows])
        got = indices[rows].long()
        assert (got.sort(dim=1).values.diff(dim=1) > 0).all()
        kth = scores.topk(topk).values[:, -1:]
        # About a hundred fp32 rounding steps of the largest score.
        tol = 1e-5 * scores[scores.isfinite()].abs().max()
        assert (scores.gather(1, got) >= kth - tol).all()
        left_out = scores.scatter(1, got, -float("inf"))
        assert (left_out <= kth + tol).all()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_empty_inputs():
    q = torch.zeros((3, 64, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.zeros((3, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((5, 128), device="cuda", dtype=torch.bfloat16)
    ranges = torch.zeros((3, 2), device="cuda", dtype=torch.int32)
    indices, lengths = indexer_topk(q, weights, k[:0], ranges, 128)
    assert indices.shape == (3, 128) and (indices == -1).all() and (lengths == 0).all()
    indices, lengths = indexer_topk(q[:0], weights[:0], k, ranges[:0], 128)
    assert indices.shape == (0, 128) and lengths.shape == (0,)


def test_causal_key_ranges():
    positions = torch.tensor([0, 4, 5, 9])
    assert causal_key_ranges(positions).tolist() == [[0, 1], [0, 5], [0, 6], [0, 10]]
    cu_seqlens = torch.tensor([0, 5, 10])
    assert causal_key_ranges(positions, cu_seqlens).tolist() == [[0, 1], [0, 5], [5, 6], [5, 10]]


if __name__ == "__main__":
    tilelang.testing.main()
