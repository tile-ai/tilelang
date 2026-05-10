import torch
import tilelang
import tilelang.testing

from tilelang.utils.sparse import compress, randn_semi_sparse


def torch_compress(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference 2:4 sparse compressor in pure PyTorch with natural row-major metadata.

    Each 4-bit chunk of the metadata integer encodes the two nonzero positions
    within one group of 4 consecutive elements:
        bits [1:0] = index of first  nonzero (0-3)
        bits [3:2] = index of second nonzero (0-3)

    This layout matches tilelang.utils.sparse.compress() for all supported dtypes.

    For 8-bit types that do not support gather natively (e.g. float8), the dense
    tensor is temporarily viewed as int8 before indexing and restored afterward.
    """
    if dense.dim() != 2:
        raise RuntimeError(f"Expected 2D tensor, got {dense.dim()}D")
    m, k = dense.shape

    is_32bit = dense.dtype == torch.float32
    ksparse = 2 if is_32bit else 4
    # int8 uses int32 metadata to match CUTLASS convention; all others use int16
    meta_dtype = torch.int32 if dense.dtype == torch.int8 else torch.int16
    quadbits = meta_dtype.itemsize * 8 // 4  # 4-bit groups that fit in one meta element

    # 8-bit non-integer types (float8 variants) may not support gather; view as int8
    gather_dtype = torch.int8 if (dense.element_size() == 1 and dense.dtype != torch.int8) else None
    work = dense.view(gather_dtype) if gather_dtype is not None else dense

    groups = work.view(-1, k // ksparse, ksparse)
    nz = groups != 0
    if not is_32bit:
        m0, m1, _m2, m3 = nz.unbind(-1)
    else:
        m0, _m2 = m1, m3 = nz.unbind(-1)

    meta_ncols = k // (ksparse * quadbits)

    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    idxs0 = expr1.to(torch.int64) | (expr2.to(torch.int64) << 1)
    idxs1 = (expr0 | expr2 | m3).to(torch.int64) | ((expr1 | ~m1).to(torch.int64) << 1)

    if not is_32bit:
        sp0 = groups.gather(-1, idxs0.unsqueeze(-1))
        sp1 = groups.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sp0, sp1), dim=-1).view(m, k // 2)
    else:
        sparse = groups.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)

    if gather_dtype is not None:
        sparse = sparse.view(dense.dtype)

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view(-1, meta_ncols, quadbits).to(meta_dtype)
    # Pack 4-bit chunks into each meta element (little-endian)
    meta = meta_n[:, :, 0]
    for i in range(1, quadbits):
        meta = meta | (meta_n[:, :, i] << (4 * i))

    return sparse, meta


def _test_compress(M, K, dtype):
    A = randn_semi_sparse(M, K, dtype=dtype, device="cuda")
    A_sparse, E = compress(A)
    assert A_sparse.shape == (M, K // 2)


def _test_compress_matches_reference(M, K, dtype):
    """Verify compress() and torch_compress() produce the same output for 2:4 types."""
    A = randn_semi_sparse(M, K, dtype=dtype, device="cuda")
    sp_tl, meta_tl = compress(A)
    sp_ref, meta_ref = torch_compress(A)
    torch.testing.assert_close(sp_tl, sp_ref)
    torch.testing.assert_close(meta_tl, meta_ref)


@tilelang.testing.requires_cuda
def test_compress():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        _test_compress(1024, 1024, dtype)
    for name in ("float8_e4m3fn", "float8_e5m2"):
        dt = getattr(torch, name, None)
        if dt is not None:
            _test_compress(1024, 1024, dt)


@tilelang.testing.requires_cuda
def test_compress_matches_reference():
    """compress() should agree with the pure-PyTorch reference for float16/bfloat16."""
    for dtype in (torch.float16, torch.bfloat16):
        _test_compress_matches_reference(1024, 1024, dtype)


if __name__ == "__main__":
    test_compress()
    test_compress_matches_reference()
    print("All tests passed.")
