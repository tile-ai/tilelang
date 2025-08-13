# Gated Delta Net(GDN) kernel implementation in TileLang

## Requirement

### The Tilelang version for test is 0.1.5+17fafc1b3026d910a83eb8052fdf811ba56be0b1

### We currently use triton=3.3.0 and FLA commit id=f03cb3ae for comparison

## Get started

### The common/chunk_delta_h.py implements the most critical forward kernel of GDN. It's a good start to understand the GDN logic and the tilelang optimization