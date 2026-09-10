"""NPU tests for the SIMD op batch added on zyx/simd-instrs.

Follows the heterogeneous-pair convention of vaddc (#361): carry-style ops
return ``(carry, result)`` with carry as a ``boolx256`` predicate.
"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.language import simd as S

N = 64


# --- vusqz ---


@tilelang.jit
def vusqz_kernel():
    @T.prim_func
    def main(mask_in: T.Tensor((64,), "int32"), out: T.Tensor((64,), "int32")):
        with T.Kernel(1):
            m_ub = T.alloc_shared((64,), "int32")
            o_ub = T.alloc_shared((64,), "int32")
            T.copy(mask_in, m_ub)
            with T.SimdVF():
                full = S.pset(32)
                m = S.vld(m_ub[0])
                pred = S.vcmps(m, T.int32(1), full, "eq")
                S.vsts(o_ub[0], S.vusqz(pred, "int32"), full)
            T.copy(o_ub, out)

    return main


def test_vusqz_prefix_count():
    torch.manual_seed(0)
    kernel = vusqz_kernel()
    for _ in range(5):
        m = torch.randint(0, 2, (64,), dtype=torch.int32, device="npu")
        out = torch.empty(64, dtype=torch.int32, device="npu")
        kernel(m, out)
        torch.npu.synchronize()
        ref = (m.cpu().cumsum(0) - m.cpu()).to(torch.int32)
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


# --- vmull ---


@tilelang.jit
def vmull_kernel():
    @T.prim_func
    def main(a: T.Tensor((N,), "int32"), b: T.Tensor((N,), "int32"), lo: T.Tensor((N,), "int32"), hi: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "int32")
            b_ub = T.alloc_shared((N,), "int32")
            lo_ub = T.alloc_shared((N,), "int32")
            hi_ub = T.alloc_shared((N,), "int32")
            T.copy(a, a_ub)
            T.copy(b, b_ub)
            with T.SimdVF():
                full = S.pset(32)
                lo_v, hi_v = S.vmull(S.vld(a_ub[0]), S.vld(b_ub[0]), full)
                S.vsts(lo_ub[0], lo_v, full)
                S.vsts(hi_ub[0], hi_v, full)
            T.copy(lo_ub, lo)
            T.copy(hi_ub, hi)

    return main


def test_vmull():
    torch.manual_seed(0)
    a = torch.randint(-1000, 1000, (N,), dtype=torch.int32, device="npu")
    b = torch.randint(-1000, 1000, (N,), dtype=torch.int32, device="npu")
    lo = torch.empty(N, dtype=torch.int32, device="npu")
    hi = torch.empty(N, dtype=torch.int32, device="npu")
    vmull_kernel()(a, b, lo, hi)
    torch.npu.synchronize()
    prod64 = a.cpu().to(torch.int64) * b.cpu().to(torch.int64)
    torch.testing.assert_close(lo.cpu(), (prod64 & 0xFFFFFFFF).to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(hi.cpu(), (prod64 >> 32).to(torch.int32), rtol=0, atol=0)


# --- vunpack ---


@tilelang.jit
def vunpack_kernel():
    @T.prim_func
    def main(x: T.Tensor((128,), "int16"), lo: T.Tensor((N,), "int32"), hi: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((128,), "int16")
            lo_ub = T.alloc_shared((N,), "int32")
            hi_ub = T.alloc_shared((N,), "int32")
            T.copy(x, x_ub)
            with T.SimdVF():
                m32 = S.pset(32)
                v = S.vld(x_ub[0])
                S.vsts(lo_ub[0], S.vunpack(v, "LOWER"), m32)
                S.vsts(hi_ub[0], S.vunpack(v, "HIGHER"), m32)
            T.copy(lo_ub, lo)
            T.copy(hi_ub, hi)

    return main


def test_vunpack():
    torch.manual_seed(0)
    x = torch.randint(-30000, 30000, (128,), dtype=torch.int16, device="npu")
    lo = torch.empty(N, dtype=torch.int32, device="npu")
    hi = torch.empty(N, dtype=torch.int32, device="npu")
    vunpack_kernel()(x, lo, hi)
    torch.npu.synchronize()
    xc = x.cpu()
    torch.testing.assert_close(lo.cpu(), xc[:N].to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(hi.cpu(), xc[N:].to(torch.int32), rtol=0, atol=0)


# --- vaddc / vsubc ---


@tilelang.jit
def carry_kernel(add: int):
    @T.prim_func
    def main(a: T.Tensor((N,), "int32"), b: T.Tensor((N,), "int32"), s: T.Tensor((N,), "int32"), c: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "int32")
            b_ub = T.alloc_shared((N,), "int32")
            s_ub = T.alloc_shared((N,), "int32")
            c_ub = T.alloc_shared((N,), "int32")
            T.copy(a, a_ub)
            T.copy(b, b_ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.vdup(T.int32(1), T.int32, full)
                zero = S.vdup(T.int32(0), T.int32, full)
                if add:
                    carry, res = S.vaddc(S.vld(a_ub[0]), S.vld(b_ub[0]), full)
                else:
                    carry, res = S.vsubc(S.vld(a_ub[0]), S.vld(b_ub[0]), full)
                S.vsts(s_ub[0], res, full)
                S.vsts(c_ub[0], S.vsel(one, zero, carry), full)
            T.copy(s_ub, s)
            T.copy(c_ub, c)

    return main


def _u64_limb(t):
    return [int(v) for v in (t.cpu().to(torch.int64) & 0xFFFFFFFF)]


def test_vaddc():
    torch.manual_seed(0)
    a = torch.randint(0, 2**32 - 1, (N,), dtype=torch.int64).to(torch.int32).to("npu")
    b = torch.randint(0, 2**32 - 1, (N,), dtype=torch.int64).to(torch.int32).to("npu")
    s = torch.empty(N, dtype=torch.int32, device="npu")
    c = torch.empty(N, dtype=torch.int32, device="npu")
    carry_kernel(1)(a, b, s, c)
    torch.npu.synchronize()
    sum64 = [u + v for u, v in zip(_u64_limb(a), _u64_limb(b))]
    torch.testing.assert_close(s.cpu(), torch.tensor([v & 0xFFFFFFFF for v in sum64], dtype=torch.int64).to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(c.cpu(), torch.tensor([v >> 32 for v in sum64], dtype=torch.int64).to(torch.int32), rtol=0, atol=0)


def test_vsubc_carry():
    """carry = 1 where the subtraction did NOT borrow (add-style carry)."""
    torch.manual_seed(0)
    a = torch.randint(0, 2**32 - 1, (N,), dtype=torch.int64).to(torch.int32).to("npu")
    b = torch.randint(0, 2**32 - 1, (N,), dtype=torch.int64).to(torch.int32).to("npu")
    s = torch.empty(N, dtype=torch.int32, device="npu")
    c = torch.empty(N, dtype=torch.int32, device="npu")
    carry_kernel(0)(a, b, s, c)
    torch.npu.synchronize()
    diff64 = [u - v for u, v in zip(_u64_limb(a), _u64_limb(b))]
    torch.testing.assert_close(s.cpu(), torch.tensor([v & 0xFFFFFFFF for v in diff64], dtype=torch.int64).to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(
        c.cpu(), torch.tensor([1 if v >= 0 else 0 for v in diff64], dtype=torch.int64).to(torch.int32), rtol=0, atol=0
    )


# --- vaddcs chain (64-bit add) ---


@tilelang.jit
def carry_chain_kernel():
    @T.prim_func
    def main(
        a0: T.Tensor((N,), "int32"),
        a1: T.Tensor((N,), "int32"),
        b0: T.Tensor((N,), "int32"),
        b1: T.Tensor((N,), "int32"),
        s0: T.Tensor((N,), "int32"),
        s1: T.Tensor((N,), "int32"),
        co: T.Tensor((N,), "int32"),
    ):
        with T.Kernel(1):
            a0_ub = T.alloc_shared((N,), "int32")
            a1_ub = T.alloc_shared((N,), "int32")
            b0_ub = T.alloc_shared((N,), "int32")
            b1_ub = T.alloc_shared((N,), "int32")
            s0_ub = T.alloc_shared((N,), "int32")
            s1_ub = T.alloc_shared((N,), "int32")
            c_ub = T.alloc_shared((N,), "int32")
            T.copy(a0, a0_ub)
            T.copy(a1, a1_ub)
            T.copy(b0, b0_ub)
            T.copy(b1, b1_ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.vdup(T.int32(1), T.int32, full)
                zero = S.vdup(T.int32(0), T.int32, full)
                carry0, res0 = S.vaddc(S.vld(a0_ub[0]), S.vld(b0_ub[0]), full)
                carry1, res1 = S.vaddcs(S.vld(a1_ub[0]), S.vld(b1_ub[0]), carry0, full)
                S.vsts(s0_ub[0], res0, full)
                S.vsts(s1_ub[0], res1, full)
                S.vsts(c_ub[0], S.vsel(one, zero, carry1), full)
            T.copy(s0_ub, s0)
            T.copy(s1_ub, s1)
            T.copy(c_ub, co)

    return main


def test_vaddcs_chain():
    torch.manual_seed(0)
    a0 = torch.randint(0, 2**32 - 1, (N,), dtype=torch.int64).to(torch.int32).to("npu")
    a1 = torch.randint(0, 2**32 - 1, (N,), dtype=torch.int64).to(torch.int32).to("npu")
    b0 = torch.randint(0, 2**32 - 1, (N,), dtype=torch.int64).to(torch.int32).to("npu")
    b1 = torch.randint(0, 2**32 - 1, (N,), dtype=torch.int64).to(torch.int32).to("npu")
    s0 = torch.empty(N, dtype=torch.int32, device="npu")
    s1 = torch.empty(N, dtype=torch.int32, device="npu")
    c = torch.empty(N, dtype=torch.int32, device="npu")
    carry_chain_kernel()(a0, a1, b0, b1, s0, s1, c)
    torch.npu.synchronize()
    Ssum = [
        (a0u | (a1u << 32)) + (b0u | (b1u << 32)) for a0u, a1u, b0u, b1u in zip(_u64_limb(a0), _u64_limb(a1), _u64_limb(b0), _u64_limb(b1))
    ]
    torch.testing.assert_close(s0.cpu(), torch.tensor([v & 0xFFFFFFFF for v in Ssum], dtype=torch.int64).to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(
        s1.cpu(), torch.tensor([(v >> 32) & 0xFFFFFFFF for v in Ssum], dtype=torch.int64).to(torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        c.cpu(), torch.tensor([(v >> 64) & 0xFFFFFFFF for v in Ssum], dtype=torch.int64).to(torch.int32), rtol=0, atol=0
    )


# --- vlrelu / vprelu ---


@tilelang.jit
def relu_kernel():
    @T.prim_func
    def main(x: T.Tensor((N,), "float32"), o1: T.Tensor((N,), "float32"), o2: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), "float32")
            o1_ub = T.alloc_shared((N,), "float32")
            o2_ub = T.alloc_shared((N,), "float32")
            T.copy(x, x_ub)
            with T.SimdVF():
                full = S.pset(32)
                v = S.vld(x_ub[0])
                S.vsts(o1_ub[0], S.vlrelu(v, T.float32(0.1), full), full)
                slope = S.vdup(T.float32(0.2), T.float32, full)
                S.vsts(o2_ub[0], S.vprelu(v, slope, full), full)
            T.copy(o1_ub, o1)
            T.copy(o2_ub, o2)

    return main


def test_vlrelu_vprelu():
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32, device="npu") * 2
    o1 = torch.empty(N, dtype=torch.float32, device="npu")
    o2 = torch.empty(N, dtype=torch.float32, device="npu")
    relu_kernel()(x, o1, o2)
    torch.npu.synchronize()
    xc = x.cpu()
    torch.testing.assert_close(o1.cpu(), torch.where(xc > 0, xc, xc * 0.1), rtol=0, atol=1e-6)
    torch.testing.assert_close(o2.cpu(), torch.where(xc > 0, xc, xc * 0.2), rtol=0, atol=1e-6)


# --- update_mask ---


@tilelang.jit
def umask_kernel():
    @T.prim_func
    def main(cnt: T.Tensor((1,), "int32"), out: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            cnt_ub = T.alloc_shared((1,), "int32")
            out_ub = T.alloc_shared((N,), "int32")
            T.copy(cnt, cnt_ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.vdup(T.int32(1), T.int32, full)
                zero = S.vdup(T.int32(0), T.int32, full)
                m = S.update_mask(cnt_ub[0], 32)
                S.vsts(out_ub[0], S.vsel(one, zero, m), full)
            T.copy(out_ub, out)

    return main


def test_update_mask():
    for c in [0, 1, 13, 63, 64, 100]:
        cnt = torch.tensor([c], dtype=torch.int32, device="npu")
        out = torch.empty(N, dtype=torch.int32, device="npu")
        umask_kernel()(cnt, out)
        torch.npu.synchronize()
        ref = torch.zeros(N, dtype=torch.int32)
        ref[: min(c, N)] = 1
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


# --- ppack / punpack / pintlv / pdintlv round trips ---


@tilelang.jit
def pred_rt_kernel():
    @T.prim_func
    def main(
        dummy: T.Tensor((64,), "int32"),
        o0: T.Tensor((64,), "int32"),
        o1: T.Tensor((64,), "int32"),
        o2: T.Tensor((64,), "int32"),
        o3: T.Tensor((64,), "int32"),
    ):
        with T.Kernel(1):
            d_ub = T.alloc_shared((64,), "int32")
            u0 = T.alloc_shared((64,), "int32")
            u1 = T.alloc_shared((64,), "int32")
            u2 = T.alloc_shared((64,), "int32")
            u3 = T.alloc_shared((64,), "int32")
            T.copy(dummy, d_ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.vdup(T.int32(1), T.int32, full)
                zero = S.vdup(T.int32(0), T.int32, full)
                v = S.vld(d_ub[0])
                pa = S.vcmps(v, T.int32(32), full, "lt")
                pb = S.vcmps(v, T.int32(48), full, "ge")
                S.vsts(u0[0], S.vsel(one, zero, S.punpack(S.ppack(pa, "LOWER"), "LOWER")), full)
                ip0, ip1 = S.pintlv(pa, pb, 32)
                rp0, rp1 = S.pdintlv(ip0, ip1, 32)
                S.vsts(u1[0], S.vsel(one, zero, rp0), full)
                S.vsts(u2[0], S.vsel(one, zero, rp1), full)
                S.vsts(u3[0], S.vsel(one, zero, S.punpack(pa, "LOWER")), full)
            T.copy(u0, o0)
            T.copy(u1, o1)
            T.copy(u2, o2)
            T.copy(u3, o3)

    return main


def test_predicate_roundtrips():
    dummy = torch.arange(64, dtype=torch.int32, device="npu")
    o0 = torch.empty(64, dtype=torch.int32, device="npu")
    o1 = torch.empty(64, dtype=torch.int32, device="npu")
    o2 = torch.empty(64, dtype=torch.int32, device="npu")
    o3 = torch.empty(64, dtype=torch.int32, device="npu")
    pred_rt_kernel()(dummy, o0, o1, o2, o3)
    torch.npu.synchronize()
    dc = dummy.cpu()
    pa = (dc < 32).to(torch.int32)
    pb = (dc >= 48).to(torch.int32)
    torch.testing.assert_close(o0.cpu(), pa, rtol=0, atol=0)
    torch.testing.assert_close(o1.cpu(), pa, rtol=0, atol=0)
    torch.testing.assert_close(o2.cpu(), pb, rtol=0, atol=0)
    spread = torch.zeros(64, dtype=torch.int32)
    spread[0:64:2] = 1
    torch.testing.assert_close(o3.cpu(), spread, rtol=0, atol=0)


if __name__ == "__main__":
    for fn in [
        test_vusqz_prefix_count,
        test_vmull,
        test_vunpack,
        test_vaddc,
        test_vsubc_carry,
        test_vaddcs_chain,
        test_vlrelu_vprelu,
        test_update_mask,
        test_predicate_roundtrips,
    ]:
        fn()
        print(f"PASS: {fn.__name__}")
