import tilelang.tools.bench
import tilelang
import tilelang.language as T
import torch


def bench_rms_norm():
    tilelang.tools.bench.process_func(rms_norm.main)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()
