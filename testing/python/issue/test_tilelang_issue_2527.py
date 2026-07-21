import tilelang
import tilelang.language as T
import torch

# 2-D partial-region shared->global store, column offset n0=1 (4 bytes, NOT 16B aligned), int32.
M, N = 64, 64
m0, n0, mm, nn = 0, 1, 32, 32

@T.prim_func
def main(B: T.Tensor((M, N), "int32")):
    with T.Kernel(1, threads=128):
        As = T.alloc_shared((mm, nn), "int32")
        T.fill(As, 7)
        T.copy(As, B[m0:m0+mm, n0:n0+nn])          # -> 2-D bulk TMA store, crd0 = n0 = 1 (unaligned)

m = tilelang.compile(main)
src = m.get_kernel_source()
print("USES_TMA_STORE:", "tma_store" in src)       # -> True
for line in src.splitlines():
    if "tl::tma_store(" in line:
        print("  ", line.strip())                   # -> tl::tma_store(B_desc, (&(As[0])), 1, 0);

B = torch.zeros((M, N), dtype=torch.int32, device="cuda")
try:
    m(B); torch.cuda.synchronize()
    print("default OK?", bool((B.cpu()[m0:m0+mm, n0:n0+nn] == 7).all()))  # -> region == 7
except Exception as e:
    print("default CRASH:", repr(e)[:120])          # -> illegal instruction