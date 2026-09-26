"""Native FP32 staging is retained when reducer costs tie."""

import tilelang.language as T


def make_reducer(streams=8, width=None, swizzle=False):
    elements, threads = 2048, 128

    @T.prim_func
    def main(
        inputs: T.Tensor((elements,), "float32"),
        masks: T.Tensor((streams, elements), "int8"),
        output: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1, threads=threads):
            shared = T.alloc_shared((elements,), "float32")
            mask_shared = T.alloc_shared((streams, elements), "int8")
            values = T.alloc_fragment((elements,), "float32")
            weights = T.alloc_fragment((streams, elements), "int32")
            acc = T.alloc_reducer((1,), "float32")
            result = T.alloc_fragment((1,), "float32")
            if width is not None:
                T.annotate_layout(
                    {
                        values: T.Fragment(
                            (elements,),
                            forward_thread_fn=lambda element: element // width % threads,
                            forward_index_fn=lambda element: element // (threads * width) * width + element % width,
                        )
                    }
                )
            if swizzle:
                T.annotate_layout({shared: T.Layout((elements,), lambda element: element ^ 16)})
            T.copy(inputs, shared, disable_tma=True)
            T.copy(masks, mask_shared, disable_tma=True)
            T.copy(shared, values)
            T.copy(mask_shared, weights)
            T.reducer_init(acc)
            for element in T.Parallel(elements):
                for stream in T.serial(streams):
                    T.reducer_update(acc[0], values[element] * T.cast(weights[stream, element], "float32"))
            T.finalize_reducer(acc, result)
            T.copy(result, output)

    return main


VARIANTS = {"mixed_dtype": make_reducer}


def check(variant, model, result):
    values = result["buffers"]["values"]
    assert values["replicate"] == 1
    if model == "register-count":
        assert values["forward_thread"] == "_i % 512 // 4"
