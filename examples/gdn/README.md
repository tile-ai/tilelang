# Gated Delta Net(GDN) kernel implementation in TileLang

## Requirement

### The Tilelang version for test is 0.1.5+fdbf4d6cbc3c856e475244c5796fa88687d79cd4

### We currently use triton=3.3.0 and FLA commit id=f03cb3ae for comparison

## Get started

### The common/chunk_delta_h.py implements the most critical forward kernel of GDN. It's a good start to understand the GDN logic and the tilelang optimization