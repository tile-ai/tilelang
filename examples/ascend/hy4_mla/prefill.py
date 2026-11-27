"""Compilation entry point for the H64 BF16 Hy4 sparse prefill (QK576, V512)."""


def compile_prefill(
    nq,
    nk,
    topk=2048,
    *,
    scale,
    sink=True,
    variable_lengths=False,
    pipeline_stages=True,
    num_cores=None,
    q_strides=None,
    kv_stride=None,
    index_stride=None,
    out_strides=None,
    explicit_outputs=False,
):
    """Compile allocating outputs by default, or accepting O/Max/LSE explicitly.

    With explicit_outputs=True call kernel(Q, KV, Indices, Sink, Lengths,
    O, Max, LSE). No output is allocated by the TileLang adapter in that mode.
    """
    if not explicit_outputs and out_strides is not None and tuple(out_strides) != (64 * 512, 512, 1):
        raise ValueError("Non-contiguous out_strides require explicit_outputs=True")
    import tilelang

    from examples.ascend.hy4_mla.kernel import hy4_sparse_attention

    if num_cores is None:
        import torch

        from tilelang.carver.arch.driver.ascend_driver import get_num_cube_cores

        num_cores = int(get_num_cube_cores(torch.npu.current_device()))

    return tilelang.compile(
        hy4_sparse_attention(
            nq,
            nk,
            topk,
            scale=scale,
            sink=sink,
            variable_lengths=variable_lengths,
            pipeline_stages=pipeline_stages,
            num_cores=num_cores,
            q_strides=q_strides,
            kv_stride=kv_stride,
            index_stride=index_stride,
            out_strides=out_strides,
        ),
        target="ascend",
        out_idx=[] if explicit_outputs else [5, 6, 7],
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )
